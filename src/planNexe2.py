from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ToolCallLimitMiddleware, AgentMiddleware, ModelRequest, wrap_tool_call
from langchain.agents import create_agent

from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple, Union, Optional
import functools
import os
import traceback

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langchain_anthropic import ChatAnthropic
# from langchain_openai import AzureChatOpenAI
# from langchain_deepseek import ChatDeepSeek

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel, Field, model_validator

from src.tools import *
from src.safety_guard import generate_structured_report
from src.prompt import dft_agent_prompt,hpc_agent_prompt,supervisor_prompt, judge_agent_prompt
from src import var
from src.myCANVAS import CANVAS
from src.safety_guard import verify_structured_report, debug_artifact_chain

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import time

members = ["DFT_Agent", "HPC_Agent"]
instructions = [dft_agent_prompt, hpc_agent_prompt]
OPTIONS = members


# =========================================================
# Debug helper (mirrors safety_guard._dbg, with the sleep ACTIVE so the
# live log is readable). Toggle the whole channel with _DBG_ENABLED and
# tune pacing with _DBG_SLEEP. Messages also append to his.txt when
# var.my_SAVE_DIALOGUE is on, matching the rest of this module.
# =========================================================

_DBG_ENABLED = True       # master switch for all _dbg output in this module
_DBG_SLEEP = 2            # base factor for the read-pacing sleep
_DBG_SLEEP_MAX = 6        # hard cap (seconds) so a long message can't stall too long


def _dbg(msg: str) -> None:
    """Print a debug message and sleep briefly so the live log is easy to
    follow. Sleep duration scales with message length (longer message ->
    longer pause), capped at _DBG_SLEEP_MAX seconds."""
    if not _DBG_ENABLED:
        return
    body = f"[DBG] {msg}"
    sleep_time = min(_DBG_SLEEP_MAX, max(1, int(len(body) * 0.01 * _DBG_SLEEP)))
    outStr = f"[DBG][{sleep_time}s] {msg}"
    print(outStr, flush=True)
    if getattr(var, "my_SAVE_DIALOGUE", False):
        try:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(outStr + "\n")
        except Exception:
            pass
    time.sleep(sleep_time)  # ACTIVE sleep so messages can be read live


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     next: str

# Single source of truth for the legal `required_tools` values a plan step
# may name. Referenced by the myStep schema AND the plan-editing tool
# validators so the two can never drift. The empty string "" means "no
# specific final tool required for this step".
WORKER_TOOL_NAMES: List[str] = [
    "calculate_formation_E",
    "generateSurface_and_getPossibleSite",
    "generate_myAdsorbate",
    "add_myAdsorbate",
    "init_structure_data",
    "find_pseudopotential",
    "write_QE_script_w_ASE",
    "calculate_lc",
    "generate_convergence_test",
    "find_optimal_parameter",
    "generate_eos_test",
    "read_energy_from_output",
    "get_convergence_suggestions",
    "analyze_BEEF_result",
    "extract_numeric_from_tool_output",
    "math_expression_tool",
    "submit_and_monitor_job",
    "add_resource_suggestion",
    "generate_structured_report",
    "extract_text_from_tool_output",
    "find_optimal_parameter_from_derived",
    "",
]
# Mirror onto var so tool modules can reference the same list at runtime.
var.WORKER_TOOL_NAMES = WORKER_TOOL_NAMES


class myStep(BaseModel):
    """Step in the plan."""

    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )
    required_tools: Literal[tuple(WORKER_TOOL_NAMES)] = Field(  # type: ignore[valid-type]
        description=f"The one final tool your worker agent must use to obtain the desired output of a certain step. Please read the CANVAS with key Worker_available_tools to see more details about each tools."
    )
    # what would be that final one tool must use to obtain the desired output of a certain step
    required_quantities: List[str] = Field(
        default_factory=list,
        description=(
            "ONLY meaningful when required_tools == 'generate_structured_report'. "
            "Leave empty ([]) for every non-report step. For a report step, this "
            "is the deterministic list of the EXACT named quantities the report"
            "MUST contain, and therefore verified by the judge."
        ),
    )

class Response(BaseModel):
    """End everything and response to the user."""
    kind: Literal["response"] = "response"
    response: str


class Proceed(BaseModel):
    """Execute the plan as it now stands.

    Use this once the current plan is how you want it. The framework will
    dispatch the worker for step 1 (index 0) of the plan. Use plan-editing tools
    (insert_plan_steps / modify_plan_steps / delete_plan_steps) to change/update the plan.
    If no edits were needed, just Proceed directly.
    """
    kind: Literal["proceed"] = "proceed"
    comment: str = Field(
        description="Short note on your reasoning for this turn. (e.g. why you edited the plan, or 'no change needed')."
        )


class Act(BaseModel):
    """Action to perform."""

    action: Union[Proceed, Response] = Field(
        description="""Action to perform. To change the plan, FIRST call the
        plan-editing tools (insert_plan_steps / modify_plan_steps /
        delete_plan_steps), THEN return Proceed to execute the (now-updated)
        plan. If the plan already looks right, return Proceed directly. If
        the whole objective is finished, return Response to end.""",
        discriminator="kind"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_action(cls, data):
        if not isinstance(data, dict):
            return data
        action = data.get("action")

        # Case 1: model returned action as a JSON string
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return data  # let pydantic raise the real error

        # Case 2: model wrapped variant as {"Proceed": {...}}
        if isinstance(action, dict) and len(action) == 1:
            key = next(iter(action))
            if key in {"Proceed", "Response"}:
                inner = dict(action[key])
                inner.setdefault(
                    "kind",
                    {"Proceed": "proceed", "Response": "response"}[key],
                )
                action = inner

        return {**data, "action": action}

class wokerResponse(BaseModel):
    """Response from the worker agent."""

    answer: str = Field(
        description="a short summary of the answer to the question or task."
    )
    
    summary: str = Field(
        description="""what have you done + what did you note down? i.e. I did xxx, and got xxx. I did xxx, and found xxx ..... In the end, I answered xxx/finished xxx/failed xxx/... I have noted down xxx, xxx, and xxx on CANVAS"""
    )
    
    success: bool = Field(
        description="whether the task is successfully finished. True or False."
    )

class judgeResponse(BaseModel):
    verdict: Literal["pass", "fail", "warning"] = Field(
        description="the final verdict of the judge after careful and critical evaluation based on the information given. Should be pass, fail, or warning."
    )
    reasoning: str = Field(
        description="the reasoning behind the verdict. Please be specific and detailed in your reasoning, with references that support your verdict and your reasoning."
    )
    
class ExecutedRecord(BaseModel):
    """One executed plan step bundled with the worker's response to it.
    """
    step: str = Field(description="The plan step that was executed.")
    agent: str = Field(description="The worker agent that executed it.")
    required_tools: str = Field(
        default="",
        description="The required_tools value the step carried.",
    )
    worker_summary: str = Field(
        default="",
        description="The worker's own summary of what it did for this step.",
    )
    verdict_line: str = Field(
        default="",
        description="Verifier verdict line for report steps; empty for "
                    "non-report steps (nothing to verify).",
    )
    # ---- Compression bookkeeping (point 5) ----
    verdict: str = Field(
        default="",
        description="Structured verifier verdict for report records: one of "
                    "'pass' | 'warning' | 'fail'. Empty for non-report records. "
                    "Used to find report boundaries and decide auto-compression.",
    )
    judge_key: str = Field(
        default="",
        description="For non-pass report records: the JUDGE:: canvas key "
                    "holding the full verifier feedback (for the compression "
                    "advisory). Empty otherwise.",
    )
    is_compressed: bool = Field(
        default=False,
        description="True if this record is a compressed summary standing in "
                    "for a span of earlier records (whose full text is on "
                    "canvas under an ARCHIVE:: key named in worker_summary).",
    )


class PlanExecute(TypedDict):
    inputs: str
    plan: List[myStep]
    executed_steps: List[ExecutedRecord]
    response: str
    canvas: dict
    artifacts: dict
    next: str


# =====================================================================
# Plan-editing tools (supervisor-only)
#
# These mutate a turn-scoped working plan held on `var.WORKING_PLAN`,
# which supervisor_chain_node seeds from state["plan"] at the start of
# each supervisor turn and flushes back to state["plan"] when the
# supervisor returns Proceed. The plan is therefore canonically stored
# in state["plan"] BETWEEN turns; var.WORKING_PLAN is only the scratch
# the tools edit DURING a turn. (A plain @tool cannot reach the outer
# PlanExecute state directly, hence the bridge — same pattern as
# var.reportName.)
#
# Indexing is 1-based everywhere the supervisor sees or addresses a
# step, matching the numbered "Current plan" display: at=1 is the next
# step to execute. For insert, at may be len(plan)+1 (append).
# =====================================================================

def _render_plan_for_tool(plan: List[myStep]) -> str:
    if not plan:
        return "(the plan is currently empty)"
    lines = []
    for i, s in enumerate(plan):
        lines.append(
            f"{i+1}. [{s.agent}] {s.step} "
            f"(required_tools={s.required_tools!r}, "
            f"required_quantities={list(s.required_quantities)})"
        )
    return "\n".join(lines)


def _validate_steps_for_tool(steps: List[myStep]) -> Optional[str]:
    """Return an error string if any step is invalid, else None."""
    legal = list(getattr(var, "WORKER_TOOL_NAMES", WORKER_TOOL_NAMES))
    for j, s in enumerate(steps):
        pos = f"step #{j+1} in your provided list"
        if s.agent not in members:
            return (f"{pos} names agent {s.agent!r}, which is not a valid "
                    f"worker. Valid agents: {members}.")
        if s.required_tools not in legal:
            return (f"{pos} names required_tools {s.required_tools!r}, which "
                    f"is not a recognized tool. Valid tools: {legal}.")
        if s.required_tools == "generate_structured_report" and not list(s.required_quantities):
            return (f"{pos} is a report step "
                    f"(required_tools='generate_structured_report') but has "
                    f"an empty required_quantities list. A report step MUST "
                    f"name the exact quantities the report has to certify.")
    return None


def _coerce_steps(steps) -> List[myStep]:
    out = []
    for s in steps:
        if isinstance(s, myStep):
            out.append(s)
        else:
            out.append(myStep.model_validate(s))
    return out


@tool
def insert_plan_steps(
    at: Annotated[
        int,
        "1-based position to insert BEFORE. at=1 inserts at the very front "
        "(these become the next steps executed); at=len(plan)+1 appends to "
        "the end. Use this to build the initial plan (insert at=1 into the "
        "empty plan), to inject discussion/fix steps at the front, or to "
        "append follow-up work.",
    ],
    steps: Annotated[
        List[myStep],
        "The step(s) to insert, in execution order.",
    ],
):
    """Insert one or more steps into the current plan at a 1-based position.
    Returns the full updated plan. Does not change which step runs next
    unless you insert at=1. Use this to build the initial plan (insert at=1 into the
    empty plan), to inject discussion/fix steps at the front, or to append follow-up work.
    """
    plan = list(getattr(var, "WORKING_PLAN", []))
    steps = _coerce_steps(steps)
    if not steps:
        return "INSERT_FAILED: you provided no steps to insert."
    if at < 1 or at > len(plan) + 1:
        return (f"INSERT_FAILED: at={at} is out of range. Valid range is "
                f"1..{len(plan)+1} (at={len(plan)+1} appends). Current plan "
                f"has {len(plan)} step(s).")
    err = _validate_steps_for_tool(steps)
    if err:
        return f"INSERT_FAILED: {err} No change to the plan was made."
    new_plan = plan[: at - 1] + steps + plan[at - 1:]
    var.WORKING_PLAN = new_plan
    _dbg(f"[PT1 plan-edit] insert_plan_steps: at={at}, inserted {len(steps)} "
         f"step(s); plan now has {len(new_plan)} step(s). "
         f"next-to-execute=[{new_plan[0].agent}] {new_plan[0].step!r}")
    return (f"Inserted {len(steps)} step(s) at position {at}. Updated plan:\n"
            f"{_render_plan_for_tool(new_plan)}")


@tool
def modify_plan_steps(
    at: Annotated[
        int,
        "1-based position of the first step to replace. at=1 is the next "
        "step to execute.",
    ],
    count: Annotated[
        int,
        "How many existing steps (starting at `at`) to replace. Use count=1 "
        "to modify a single step.",
    ],
    steps: Annotated[
        List[myStep],
        "The replacement step(s), in execution order. The number of "
        "replacements need NOT equal `count` (you may expand or shrink the "
        "plan).",
    ],
):
    """Replace `count` existing steps starting at 1-based `at` with the given
    step(s). Returns the full updated plan. Use this to modify just the next
    step (at=1, count=1) or more without rewriting the rest of the plan."""
    plan = list(getattr(var, "WORKING_PLAN", []))
    steps = _coerce_steps(steps)
    if not steps:
        return ("MODIFY_FAILED: you provided no replacement steps. To remove "
                "steps, use delete_plan_steps instead.")
    if at < 1 or at > len(plan):
        return (f"MODIFY_FAILED: at={at} is out of range. Valid range is "
                f"1..{len(plan)} (the plan has {len(plan)} step(s)).")
    if count < 1 or at - 1 + count > len(plan):
        return (f"MODIFY_FAILED: count={count} starting at {at} runs past the "
                f"end of the plan (which has {len(plan)} step(s)). You can "
                f"replace at most {len(plan) - (at - 1)} step(s) from "
                f"position {at}.")
    err = _validate_steps_for_tool(steps)
    if err:
        return f"MODIFY_FAILED: {err} No change to the plan was made."
    new_plan = plan[: at - 1] + steps + plan[at - 1 + count:]
    var.WORKING_PLAN = new_plan
    _dbg(f"[PT1 plan-edit] modify_plan_steps: at={at}, replaced {count} step(s) "
         f"with {len(steps)}; plan now has {len(new_plan)} step(s). "
         f"next-to-execute=[{new_plan[0].agent}] {new_plan[0].step!r}"
         if new_plan else
         f"[PT1 plan-edit] modify_plan_steps: at={at}, replaced {count} step(s); plan now EMPTY")
    return (f"Replaced {count} step(s) at position {at} with {len(steps)} "
            f"step(s). Updated plan:\n{_render_plan_for_tool(new_plan)}")


@tool
def delete_plan_steps(
    at: Annotated[
        int,
        "1-based position of the first step to delete. at=1 is the next "
        "step to execute.",
    ],
    count: Annotated[
        int,
        "How many steps (starting at `at`) to delete.",
    ] = 1,
):
    """Delete `count` steps starting at 1-based `at`. Returns the full
    updated plan."""
    plan = list(getattr(var, "WORKING_PLAN", []))
    if at < 1 or at > len(plan):
        return (f"DELETE_FAILED: at={at} is out of range. Valid range is "
                f"1..{len(plan)} (the plan has {len(plan)} step(s)).")
    if count < 1 or at - 1 + count > len(plan):
        return (f"DELETE_FAILED: count={count} starting at {at} runs past the "
                f"end of the plan (which has {len(plan)} step(s)). You can "
                f"delete at most {len(plan) - (at - 1)} step(s) from "
                f"position {at}.")
    removed = plan[at - 1: at - 1 + count]
    new_plan = plan[: at - 1] + plan[at - 1 + count:]
    var.WORKING_PLAN = new_plan
    _dbg(f"[PT1 plan-edit] delete_plan_steps: at={at}, deleted {len(removed)} "
         f"step(s); plan now has {len(new_plan)} step(s).")
    return (f"Deleted {len(removed)} step(s) at position {at}. Updated "
            f"plan:\n{_render_plan_for_tool(new_plan)}")


# ---- Shared history rendering (supervisor AND worker see the same view) ----

def _render_executed_section(executed_steps: List[ExecutedRecord]) -> str:
    """Render the two history sections shared by supervisor and worker:
    the numbered executed-steps list and the per-step worker responses
    (with the verifier line only for report steps, per point 4)."""
    if not executed_steps:
        steps_block = "(nothing executed yet)"
        resp_block = "(no worker responses yet)"
    else:
        steps_lines = [
            f"{i+1}. [{r.agent}] {r.step}"
            for i, r in enumerate(executed_steps)
        ]
        steps_block = "\n".join(steps_lines)
        resp_lines = []
        for i, r in enumerate(executed_steps):
            line = f"{i+1}. [{r.step}] -> worker: {r.worker_summary}"
            if r.verdict_line:
                line += f"; verifier: {r.verdict_line}"
            resp_lines.append(line)
        resp_block = "\n".join(resp_lines)
    return (
        f"## Executed plan steps (most recent last):\n{steps_block}\n\n"
        f"## Worker agents' responses on executed plan items:\n{resp_block}"
    )


# =====================================================================
# Episode compression at verified-report boundaries (point 5)
#
# Boundaries are ALWAYS report records; a "span" is the run of ordinary
# records strictly between two consecutive reports (or between start and
# the first report). Closing-report-governs: the span is judged by the
# report that CLOSES it (the later fencepost).
#   - closing report = clean pass     -> auto-compress the span
#   - closing report = warning / fail -> do NOT auto-compress; offer it as
#       an opt-in "compression opportunity" to the supervisor
# Every compression (auto or supervisor-chosen) is non-destructive: the
# full span text is archived to ARCHIVE::steps_<a>-<b> first, then the span
# is replaced in executed_steps by ONE summarized record. Nothing is lost.
# =====================================================================

_COMPRESSED_AGENT_SENTINEL = "(compressed)"


def _is_report_record(r) -> bool:
    return bool(getattr(r, "verdict", "")) or getattr(r, "required_tools", "") == "generate_structured_report"


def _render_record_for_archive(idx_label, r) -> str:
    line = f"{idx_label}. [{r.agent}] {r.step}\n    worker: {r.worker_summary}"
    if r.verdict_line:
        line += f"\n    verifier: {r.verdict_line}"
    return line


def _summarize_episode_llm(records, goal: str) -> str:
    """One LLM summary of a report-bounded span. Reuses the Sonnet/judge
    plumbing. Retries 3x; exits the run on persistent (server-side) failure,
    consistent with the other LLM calls in this module."""
    config = var.OTHER_GLOBAL_VARIABLES
    body = "\n".join(
        _render_record_for_archive(i + 1, r) for i, r in enumerate(records)
    )
    sys_prompt = (
        "You compress a span of executed plan steps from a scientific-computing "
        "agent run into a single tight summary (3-5 sentences). This span sits "
        "between two verified reports and its full text is archived separately, "
        "so you may drop narration — but you MUST preserve, verbatim where "
        "possible: (1) any result_id / reference id that downstream steps may "
        "still cite, (2) filenames, (3) the concrete decided/converged values "
        "and their units, and (4) any unresolved warnings or known problems. "
        "Do not invent anything. Output only the summary prose."
    )
    user_prompt = (
        f"Overall objective: {goal}\n\n"
        f"Span of executed steps to compress:\n{body}\n\n"
        f"Write the 3-5 sentence summary now."
    )
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key=config["ANTHROPIC_API_KEY"],
        temperature=0.0,
    )
    last_err = None
    for attempt in range(3):
        try:
            resp = llm.invoke([("system", sys_prompt), ("user", user_prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            text = text.strip()
            if text:
                return text
            last_err = "empty summary"
        except Exception as e:  # noqa: BLE001
            last_err = e
            _dbg(f"[PT5 compression] summarizer attempt {attempt+1}/3 FAILED: "
                 f"{type(e).__name__}: {e}")
            print(f"[episode-compression summarizer] attempt {attempt+1} failed: {e}")
    _dbg(f"[PT5 compression] summarizer FAILED after 3 attempts ({last_err}) "
         f"-> exit(0).")
    print(f"[episode-compression summarizer] failed after 3 attempts: {last_err}")
    exit(0)


def _compress_span(executed_steps, start_idx, end_idx, goal):
    """Archive executed_steps[start_idx:end_idx] (the records strictly
    between two reports) to canvas and replace them with one compressed
    record. Returns (new_executed_steps, (a,b) | None). No-op (returns the
    input unchanged, None) if the span is empty or already-compressed-only."""
    span = executed_steps[start_idx:end_idx]
    if not span:
        return executed_steps, None
    # Don't re-compress: if every record in the span is already a compressed
    # stand-in, there's nothing to do.
    if all(getattr(r, "is_compressed", False) for r in span):
        return executed_steps, None

    # Human-friendly 1-based step range for labeling.
    a = start_idx + 1
    b = end_idx  # inclusive 1-based end
    archive_body = "\n\n".join(
        _render_record_for_archive(a + j, r) for j, r in enumerate(span)
    )
    archive_key, _ = CANVAS.framework_write_versioned(
        f"ARCHIVE::steps_{a}-{b}",
        f"Full record of executed steps {a}-{b} (compressed in-line):\n\n{archive_body}",
    )
    _dbg(f"[PT5 compression] archived steps {a}-{b} ({len(span)} record(s), "
         f"{len(archive_body)} chars) to canvas '{archive_key}'. "
         f"Requesting LLM summary ...")
    summary = _summarize_episode_llm(span, goal)
    _dbg(f"[PT5 compression] summary ready ({len(summary)} chars): "
         f"{summary[:160]!r}")
    compressed = ExecutedRecord(
        step=f"Steps {a}-{b} (compressed)",
        agent=_COMPRESSED_AGENT_SENTINEL,
        required_tools="",
        worker_summary=(
            f"{summary}\n[Full detail of steps {a}-{b} at canvas key "
            f"'{archive_key}'.]"
        ),
        verdict_line="",
        is_compressed=True,
    )
    new_steps = executed_steps[:start_idx] + [compressed] + executed_steps[end_idx:]
    return new_steps, (a, b)


def _report_boundary_indices(executed_steps):
    """Indices of report records, in order."""
    return [i for i, r in enumerate(executed_steps) if _is_report_record(r)]


def _auto_compress_on_pass(executed_steps, goal):
    """Called right after a report record is appended. If that just-closed
    report is a clean pass, auto-compress the span it closes (the ordinary
    records between it and the previous report/start). Returns the new
    executed_steps list (unchanged if no auto-compression applies)."""
    if not executed_steps:
        return executed_steps
    closing = executed_steps[-1]
    if not _is_report_record(closing) or getattr(closing, "verdict", "") != "pass":
        return executed_steps
    # Find the previous report (the opening fencepost); span is between it
    # and the closing report (exclusive of both).
    end_idx = len(executed_steps) - 1  # the closing report's index
    prev_report_idx = -1
    for i in range(end_idx - 1, -1, -1):
        if _is_report_record(executed_steps[i]):
            prev_report_idx = i
            break
    start_idx = prev_report_idx + 1
    new_steps, _rng = _compress_span(executed_steps, start_idx, end_idx, goal)
    return new_steps


def _compression_opportunities(executed_steps):
    """List spans eligible for OPT-IN supervisor compression: spans whose
    closing report is a warning or fail (clean-pass spans are already
    auto-compressed and never appear here). Each opportunity is
    (start_step_1based, end_step_1based, closing_verdict, judge_key)."""
    opps = []
    report_idxs = _report_boundary_indices(executed_steps)
    prev = -1
    for ridx in report_idxs:
        closing = executed_steps[ridx]
        verdict = getattr(closing, "verdict", "")
        span = executed_steps[prev + 1: ridx]
        has_uncompressed = any(not getattr(r, "is_compressed", False) for r in span)
        if span and has_uncompressed and verdict in ("warning", "fail"):
            opps.append((
                prev + 2,            # 1-based first step of span
                ridx,                # 1-based last step of span
                verdict,
                getattr(closing, "judge_key", ""),
            ))
        prev = ridx
    return opps


def _render_compression_advisory(executed_steps) -> str:
    """Advisory section for the supervisor message: lists opt-in compression
    opportunities. Empty string if none."""
    opps = _compression_opportunities(executed_steps)
    if not opps:
        return ""
    lines = [
        "## Compression opportunities (optional):",
        "The following report-bounded step spans were NOT auto-compressed "
        "because their closing report did not cleanly pass. You MAY compress "
        "any of them with compress_history(start_step, end_step) to shrink the "
        "history — this is non-destructive (full detail is archived to canvas "
        "and remains readable). Read the linked verifier feedback first if you "
        "want to judge whether the span is still relevant.",
    ]
    for (a, b, verdict, judge_key) in opps:
        jk = f" (verifier feedback: '{judge_key}')" if judge_key else ""
        lines.append(
            f"- steps {a}-{b}: closing report = {verdict.upper()}{jk}. "
            f"Compress with compress_history({a}, {b})."
        )
    return "\n".join(lines)


@tool
def compress_history(
    start_step: Annotated[
        int,
        "1-based number of the FIRST executed step in the span to compress "
        "(as shown in the 'Compression opportunities' advisory).",
    ],
    end_step: Annotated[
        int,
        "1-based number of the LAST executed step in the span to compress. "
        "This must be the step just before the span's closing report.",
    ],
):
    """Compress a report-bounded span of executed steps that was offered in
    the 'Compression opportunities' advisory (spans whose closing report was a
    WARNING or FAIL, and so were not auto-compressed). This is NON-DESTRUCTIVE:
    the full detail is archived to a canvas 'ARCHIVE::' key (readable later)
    and the span is replaced in the running history by one summary. Use it to
    shrink a long history once you've decided a span's verbose detail is no
    longer needed inline. You may compress a span even if its failure was not
    resolved — nothing is lost, the detail stays on canvas."""
    steps = list(getattr(var, "WORKING_EXECUTED_STEPS", []))
    n = len(steps)
    if start_step < 1 or end_step < 1 or start_step > end_step or end_step > n:
        return (f"COMPRESS_FAILED: invalid span {start_step}-{end_step}. "
                f"There are {n} executed steps; start_step<=end_step and both "
                f"must be within 1..{n}.")
    start_idx = start_step - 1
    end_idx = end_step  # slice end is exclusive -> covers up to end_step
    # The span must be report-bounded: the record immediately AFTER end_step
    # must be a report (its closing fencepost), and the span itself must not
    # contain an interior report.
    if end_idx < n and not _is_report_record(steps[end_idx]):
        return (f"COMPRESS_FAILED: step {end_step+1} (just after your span) is "
                f"not a report. A compressible span must end immediately "
                f"before its closing report. Use the exact ranges listed in "
                f"the Compression opportunities advisory.")
    if any(_is_report_record(r) for r in steps[start_idx:end_idx]):
        return (f"COMPRESS_FAILED: span {start_step}-{end_step} contains a "
                f"report record. Spans are bounded by reports and cannot "
                f"contain one. Use the exact ranges from the advisory.")
    goal = getattr(var, "OVERALL_GOAL", "") or ""
    new_steps, rng = _compress_span(steps, start_idx, end_idx, goal)
    if rng is None:
        return (f"COMPRESS_FAILED: span {start_step}-{end_step} is empty or "
                f"already compressed; nothing to do.")
    var.WORKING_EXECUTED_STEPS = new_steps
    a, b = rng
    _dbg(f"[PT5 compression] supervisor compress_history({start_step}, "
         f"{end_step}) applied: span {a}-{b} collapsed to one summary "
         f"(archived to canvas). WORKING_EXECUTED_STEPS now has "
         f"{len(new_steps)} record(s).")
    return (f"Compressed executed steps {a}-{b} into a single summary "
            f"(full detail archived to canvas). The running history is now "
            f"shorter.")


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return await handler(request)


# Tools a worker is allowed to use AFTER it has generated a structured
# report in the current round. The report's numerical claims are not yet
# verified at this point, so the worker must NOT be able to feed them into
# any further computation/extraction/registration/submission. It may only
# do canvas bookkeeping (write summaries, version-edit its notes) and then
# return to the supervisor.
_POST_REPORT_ALLOWED_TOOLS = {
    "inspect_my_canvas",
    "read_my_canvas",
    "write_my_canvas",
    "version_controlled_edit_canvas_doc",
    "write_progress_note",
}


class RestrictToolsAfterReportMiddleware(AgentMiddleware):
    """Once a structured report has been generated in the current worker
    round, restrict the worker to canvas bookkeeping tools only.

    The trigger is driven off `var.reportName`, which generate_structured_report
    sets (as its last action) on success and which the surrounding control
    flow clears when appropriate:
      - set on a successful report  -> restriction engages for the rest of
        that attempt;
      - cleared by worker_agent_node's required-quantities retry (which
        deletes the bad report)  -> restriction lifts so the retry can
        regenerate the report with the full toolset;
      - cleared after judging each round, plus a defensive reset at the top
        of worker_agent_node  -> each fresh round starts unrestricted.

    Because the compute/extraction/registration/submission tools are simply
    absent from the request after a report, the worker cannot propagate an
    unverified quantity downstream; it can only summarize and return.
    """

    def _maybe_restrict(self, request):
        if getattr(var, "reportName", ""):
            before = len(request.tools)
            kept = []
            for t in request.tools:
                name = getattr(t, "name", None)
                if name is None and isinstance(t, dict):
                    name = t.get("name")
                if name in _POST_REPORT_ALLOWED_TOOLS:
                    kept.append(t)
            request.tools = kept
            _dbg(f"[PT3 post-report restriction] report '{var.reportName}' "
                 f"present -> worker tools narrowed {before} -> {len(kept)} "
                 f"(canvas-bookkeeping only). Worker can no longer compute/"
                 f"extract/register/submit until it returns.")

    def wrap_model_call(self, request, handler):
        self._maybe_restrict(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._maybe_restrict(request)
        return await handler(request)

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Only handle errors that occur during tool execution due to invalid inputs
        # that pass schema validation but fail at runtime (e.g., invalid SQL syntax).
        # Do NOT handle:
        # - Network failures (use tool retry middleware instead)
        # - Incorrect tool implementation errors (should bubble up)
        # - Schema mismatch errors (already auto-handled by the framework)
        #
        # Return a custom error message to the model
        outStr = f"Tool error: Please check your input and try again. ({str(e)}), traceback: {traceback.format_exc()}"
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"Error during tool execution: {str(e)}\n")
                f.write(traceback.format_exc())
                f.write("\n")
        return ToolMessage(
            content=outStr,
            tool_call_id=request.tool_call["id"]
        )
        
_act_failures = {"count": 0}

def on_act_parse_error(exc: Exception) -> str:
    _act_failures["count"] += 1
    print(f"[Act parse #{_act_failures['count']}] {type(exc).__name__}: {exc}")
    # whatever string you return is what the model sees as the tool error
    if _act_failures["count"] > 3:
        exit()
    
    return (
        "Your previous tool call did not match the Act schema. "
        "Return exactly one of {Proceed, Response} as `action`, "
        "with the inner fields directly — do NOT wrap in {'Proceed': {...}}. "
        "To change the plan, call the plan-editing tools first, then return "
        "Proceed."
    )


def print_stream(s):
    if "messages" not in s:
        print("#################")
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write("#################\n")
        print(s)
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(repr(s))
                f.write("\n")
    else:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
            if var.my_SAVE_DIALOGUE:
                with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                    f.write(repr(message))
                    f.write("\n")
        else:
            if hasattr(message, 'usage_metadata'):
                var.TOKEN_USAGE.append(message.usage_metadata)
                print(f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}")
                if var.my_SAVE_DIALOGUE:
                    with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                        f.write(f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}\n")
            message.pretty_print()
            if var.my_SAVE_DIALOGUE:
                with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                    f.write(message.pretty_repr())
                    f.write("\n")
    print()
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write("\n")
   
_REPORT_QUANTITY_GUIDANCE = """
Intermediate reports containing one numerical claim must be generated immediately after that numerical claim was made. Only report tool generated result. Never report input settings, and etc. Anything other than tool generated numerical result should be noted down under the "conclusion" section, instead of a stand alone numerical claim.
The judge will then evaluate the validity of the result.
**What counts as a numerical claim worth reporting.** A `required_quantities`
entry must name a quantity that was COMPUTED by a tool and registered as
that tool's output. Do NOT schedule a report quantity for:

  - **Input decisions** (convergence thresholds, sweep ranges, layer
    counts to test, kspacing grids, etc.) — these live in the consuming
    tool's `args` with provenance there but they are NOT tool outputs.
    Put them in the report's prose narrative, not as numerical claims.

  - **Agent estimates / expert judgments** (ad-hoc uncertainties, "good
    enough" assessments, values laundered through trivial math like
    `x0 + 0` or `x0 / 100`). NEVER schedule these. If a value you want
    reported cannot come from a real tool computation, fix the upstream
    step first (e.g. re-run the convergence test with an adjusted
    threshold and a stated reason) so a tool produces the value, then
    schedule the report.

Concrete examples of UNACCEPTABLE `required_quantities` (the judge will
flag these as fabrication):
  * `slab_thickness_convergence_threshold_eV` — threshold is an input
    decision.
  * `estimated_slab_thickness_uncertainty_eV` — agent-derived estimate.
  * `chosen_n_layers_after_nonconvergence` — picking a value yourself
    after `find_optimal_parameter` said "no value within threshold" is
    an agent decision; recall the tool with an adjusted threshold so
    the value comes from the tool.

Acceptable examples are tool outputs: optimal parameters from
`find_optimal_parameter[_from_derived]`, energies from
`read_energy_from_output`, adsorption / formation energies from
`calculate_formation_E`, lattice constants from `calculate_lc`,
differences computed by `math_expression_tool` between two registered
source values.

**Responding to verifier failures on a required quantity.** If a prior
report's verifier output flags a failure on one of the scheduled
`required_quantities`, the correct response is to insert plan steps
that OBTAIN the value properly (re-run the upstream tool with adjusted
inputs and a stated reason, or run the missing upstream computation),
then re-schedule the same report with the SAME `required_quantities`.
You may NOT respond by removing the failing quantity from
`required_quantities` to make the next report pass — that is a silent
regression of the user's request. The single legitimate exception is
that the quantity was scheduled in error to begin with (it was an
input decision or an agent estimate, not a real tool output); in that
case, drop it ONLY with an explicit acknowledgment in your reasoning
that the original scheduling violated the rule above. If multiple
genuine attempts to obtain a value have all failed, declare it
unachievable LOUDLY in the final report's narrative with the reason —
do not silently shorten the list.

If the worker ask you to remove the failing quantity from
`required_quantities` in order to make the next report pass, or you found the worker
went ahead and generated a report themselves with failing quantity removed. NEVER EVER agree
to do that unless you find that the quantity was scheduled in error to begin with.
If the worker ask you to removing the failing quantity just to pass the verifier,
that is a clear sign that the worker is acting MALICIOUS. Judge the request CRITICALLY.
"""

_PLAN_EDIT_GUIDANCE = """
To change the plan, use the plan-editing tools, then return Proceed:
  - insert_plan_steps(at, steps): add step(s). at=1 puts them at the FRONT
    (executed next); at=len(plan)+1 appends. Build the initial plan by
    inserting at=1 into the empty plan. To discuss with a worker, insert a
    step at=1 assigned to that worker.
  - modify_plan_steps(at, count, steps): replace count step(s) starting at
    `at`. Use at=1, count=1 to change ONLY the next step without touching
    the rest of the plan.
  - delete_plan_steps(at, count): remove step(s).
All indices are 1-based and match the numbered "Current plan" list below
(step 1 = the next step to execute). After editing (or if no edit is
needed), return Proceed to run step 1. Return Response only when the whole
objective is complete.
"""

_JUDGE_FEEDBACK_GUIDANCE = """
Reading verifier feedback: when a report step in the executed-steps history
shows a non-PASS verifier verdict (FAIL or WARNING), its full feedback was
saved to the canvas under a 'JUDGE::<report name>' key (named in that
history line). You MUST read that canvas key with read_my_canvas before
editing the plan in response, so your fix is grounded in the verifier's
specific issues and remediation options — do not react to the one-line
verdict alone.
"""


def supervisor_chain_node(state, agent, name):
    # CANVAS.snap()
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, supervisor is waiting. cwd: {var.my_WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
    
    print(f"supervisor is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"supervisor is processing!!!!!\n")

    # no longer print state since it may contain too much information from canvas
    # print(state)
    # if var.my_SAVE_DIALOGUE:
    #     with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
    #         f.write(str(state))
    #         f.write("\n")
            
    plan = state["plan"]
    executed_steps = state.get("executed_steps", []) or []

    # Seed the turn-scoped working plan the plan-editing tools mutate.
    # Reseeded every turn so a stale buffer can never leak between turns.
    var.WORKING_PLAN = list(plan)
    # Seed the turn-scoped working executed_steps the compress_history tool
    # mutates (point 5). Same bridge pattern as WORKING_PLAN.
    var.WORKING_EXECUTED_STEPS = list(executed_steps)
    var.OVERALL_GOAL = state.get("inputs", "")

    history_block = _render_executed_section(executed_steps)
    plan_block = _render_plan_for_tool(plan)
    compression_advisory = _render_compression_advisory(executed_steps)

    _dbg(f"[SUPERVISOR] turn start: plan has {len(plan)} step(s), "
         f"{len(executed_steps)} executed record(s). "
         f"Seeded WORKING_PLAN & WORKING_EXECUTED_STEPS.")
    _n_compressed = sum(1 for r in executed_steps if getattr(r, 'is_compressed', False))
    if _n_compressed:
        _dbg(f"[PT5 compression] history currently holds {_n_compressed} "
             f"compressed record(s).")
    if compression_advisory:
        _dbg(f"[PT5 compression] advisory PRESENTED to supervisor: "
             f"{compression_advisory.count('compress_history(')} opt-in "
             f"opportunity(ies).")

    # TODO: notify the supervisor if the worker returned successed=false
    supervisorMessage = ""
    if len(plan) == 0:
        supervisorMessage = f"""
The overall goal is: {state['inputs']}.

{history_block}

## Current plan:
(the plan is currently empty — nothing has been scheduled yet)

There is no plan yet. Please inspect and extract related information from
CANVAS, then build the initial plan by calling insert_plan_steps(at=1,
steps=[...]) and then return Proceed.

Important: you are a coordinator, not a domain expert. Before adding any
execution steps to the plan (input file generation, job submission,
calculations), consult with the DFT agent: what are the major calculations
this objective requires? What known caveats apply to this system class?
{_PLAN_EDIT_GUIDANCE}
{_REPORT_QUANTITY_GUIDANCE}
"""
    else:
        supervisorMessage = f"""
The overall goal is: {state['inputs']}.

{history_block}

## Current plan (step 1 below will be executed next unless you edit it):
{plan_block}

Please inspect and extract related information from CANVAS, then update the
plan with the plan-editing tools if needed, and return Proceed.
{_PLAN_EDIT_GUIDANCE}
{_JUDGE_FEEDBACK_GUIDANCE}
{_REPORT_QUANTITY_GUIDANCE}
"""
        if compression_advisory:
            supervisorMessage += "\n" + compression_advisory + "\n"

    if len(plan) >= 1 and plan[0].required_tools == "generate_structured_report":
        supervisorMessage += f"""
The next step (step 1) is a report step. Carefully think about whether you
need to update its list of required quantities:
{plan[0].required_quantities}
If you do, use modify_plan_steps(at=1, count=1, steps=[...]) with the
corrected required_quantities.
"""
    for agent_response in agent.stream(
        {"messages": [("user", supervisorMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    ):
        # set agent_response to be the value of the first key of the dictionary
        agent_response = next(iter(agent_response.values()))
        print_stream(agent_response)

    # CANVAS.snap_save()
    agent_response = agent_response['structured_response']

    # The plan-editing tools (insert/modify/delete) have already validated
    # every step they accepted and mutated var.WORKING_PLAN in place, so by
    # the time the supervisor returns Proceed the working plan is the
    # authoritative, validated result of this turn. We flush it back to
    # state["plan"].
    new_plan = list(getattr(var, "WORKING_PLAN", plan))

    if isinstance(agent_response.action, Response):
        _dbg(f"[SUPERVISOR] action=Response -> FINISH. "
             f"response={agent_response.action.response[:120]!r}")
        return {"response": agent_response.action.response, "next": "FINISH", "canvas": CANVAS.canvas}

    # ---- Proceed ----
    _dbg(f"[SUPERVISOR] action=Proceed. Working plan now has "
         f"{len(new_plan)} step(s). comment={getattr(agent_response.action, 'comment', '')[:80]!r}")
    if len(new_plan) == 0:
        # Nothing to execute. The supervisor must either schedule steps or
        # end. Re-prompt once; if it still proceeds on an empty plan, fail
        # loudly (consistent with the existing exit-on-bad-supervisor).
        _dbg("[SUPERVISOR] WARNING: Proceed on an EMPTY plan — re-prompting once.")
        print("Supervisor returned Proceed on an empty plan.")
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write("Supervisor returned Proceed on an empty plan.\n")
        retry_msg = (
            supervisorMessage
            + "\n\nWARNING: you returned Proceed but the plan is empty, so "
            "there is nothing to execute. Either call insert_plan_steps to "
            "schedule at least one step and then Proceed, or return Response "
            "if the whole objective is already complete."
        )
        for agent_response in agent.stream(
            {"messages": [("user", retry_msg)]}, {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
        ):
            agent_response = next(iter(agent_response.values()))
            print_stream(agent_response)
        agent_response = agent_response['structured_response']
        new_plan = list(getattr(var, "WORKING_PLAN", []))
        if isinstance(agent_response.action, Response):
            _dbg("[SUPERVISOR] after empty-plan retry: Response -> FINISH.")
            return {"response": agent_response.action.response, "next": "FINISH", "canvas": CANVAS.canvas}
        if len(new_plan) == 0:
            _dbg("[SUPERVISOR] still empty after retry -> exit(0).")
            print("Supervisor failed: Proceed on empty plan after retry.")
            exit(0)

    plan_str = "\n".join(
        f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}, required_quantities: {step.required_quantities}"
        for i, step in enumerate(new_plan)
    )
    print(plan_str)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(plan_str)
            f.write("\n")

    # Index 0 is ALWAYS the next step to execute (the framework maintains
    # this by popping plan[0] into executed_steps after each worker round).
    # Flush any compress_history edits the supervisor made this turn.
    _flushed_es = list(getattr(var, "WORKING_EXECUTED_STEPS", executed_steps))
    if len(_flushed_es) != len(executed_steps):
        _dbg(f"[PT5 compression] supervisor compress_history changed history: "
             f"{len(executed_steps)} -> {len(_flushed_es)} record(s) (flushed to state).")
    _dbg(f"[SUPERVISOR] routing to next worker=[{new_plan[0].agent}] for "
         f"step={new_plan[0].step!r}")
    return {
        "plan": new_plan,
        "next": new_plan[0].agent,
        "executed_steps": _flushed_es,
        "canvas": CANVAS.canvas,
    }
    
class judge():
    def __init__(self):
        config = var.OTHER_GLOBAL_VARIABLES
        self.llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0).with_structured_output(judgeResponse, include_raw=True)
    
    def invoke(self, input):
        with open(f"{var.my_WORKING_DIRECTORY}/judge_status.txt", "r") as f:
            status = f.read()
        while status == "stop":
            print(f"Calculation pause, judge is waiting. cwd: {var.my_WORKING_DIRECTORY}")
            # wait for 5 second
            time.sleep(5)
            with open(f"{var.my_WORKING_DIRECTORY}/judge_status.txt", "r") as f:
                status = f.read()
        
        print(f"Judge Agent is processing!!!!!")
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"Judge Agent is processing!!!!!\n")
        
        # print(input)    
        agent_response_raw = self.llm.invoke(input)
        # print(agent_response_raw)
        agent_response = agent_response_raw['raw'].content[0]['input']
        # print(agent_response)
        # 'input_tokens': 2102, 'output_tokens': 393, '
        token_usage = agent_response_raw['raw'].usage_metadata
        # print(token_usage)
        
        outStr = f"Judge's verdict: {agent_response['verdict']}\nJudge's reasoning: {agent_response['reasoning']}\nJudge's token usage: input_tokens: {token_usage['input_tokens']}, output_tokens: {token_usage['output_tokens']}"
        print(outStr)
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(outStr)
                f.write("\n")
                
        # exit()
                
        return {"verdict": agent_response['verdict'], "reasoning": agent_response['reasoning']}



def worker_agent_node(state, agent, name):
    # CANVAS.snap()
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, {name} Agent is waiting. cwd: {var.my_WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
    
    print(f"Agent {name} is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"Agent {name} is processing!!!!!\n")
        
    plan = state["plan"]
    # Defensive reset: ensure no stale report state leaks into this round.
    # var.reportName is what RestrictToolsAfterReportMiddleware keys off of;
    # generate_structured_report sets it on success, and the control flow
    # below clears it after judging / on a bad-report retry. Clearing it
    # here guarantees a fresh round always starts with the full toolset.
    var.reportName = ""
    # Reset the per-round progress-note backstop overrides (Q1): an override
    # confirmed in one round must not grant a free pass in the next.
    var.NOTE_BACKSTOP_OVERRIDES = {}
    _dbg(f"[WORKER {name}] round start: reset var.reportName (PT3 full toolset) "
         f"and var.NOTE_BACKSTOP_OVERRIDES (PT2a). plan has {len(plan)} step(s).")
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan))
    # print(plan_str)
    # if var.my_SAVE_DIALOGUE:
    #     with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
    #         f.write(plan_str)
    #         f.write("\n")
    task = plan[0]
    executed_steps = state.get("executed_steps", []) or []
    history_block = _render_executed_section(executed_steps)
    task_formatted = f"""
The overall goal is: {state["inputs"]}

{history_block}

Now, you are tasked with: {task}. Please only do this task! Do not do anything else! Please note down important information on CANVAS together with their reference id before you end.
"""

    # For report steps, the supervisor has specified the exact set of
    # quantities the report MUST contain. Surface that list to the worker
    # so it knows which named claims are mandatory.
    is_report_step = (task.required_tools == "generate_structured_report")
    required_quantities = list(getattr(task, "required_quantities", []) or [])
    if is_report_step and required_quantities:
        _rq_list = ", ".join(f"'{q}'" for q in required_quantities)
        task_formatted += f"""

IMPORTANT — REQUIRED REPORT QUANTITIES:
This is a report step. Your structured report MUST include a numerical claim
for EACH of the following quantities, using the quantity_name EXACTLY as
written here (character-for-character): {_rq_list}.
Each of these MUST be backed by a registered result_id from a real tool
output. You may include additional supporting claims, but none of the
required quantities above may be omitted. If you do not have a registered
result for one of them — for example because you determined the value by
your own analysis or arithmetic — that is strictly prohibited: 
Report back to the supervisor which required quantity you need to obtain first
to be able to generate the report, leave a note on the CANVAS and return immediately.
Do NOT attempt to fabricate those values yourself, and
Do NOT attempt to generate the report. Do NOT omit any required quantity,
and do NOT substitute your own judgement for a tool's output.
"""
    
    print(task_formatted)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(task_formatted)
            f.write("\n")
    print(f"Agent {name} is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"Agent {name} is processing!!!!!\n")
    old_task_formatted = task_formatted
    CANVAS.rest_curr_round_result_ids()
    workerGood = False
    workerGood_patient = 2
    while not workerGood and workerGood_patient > 0:
        workerGood_patient -= 1
        for agent_response in agent.stream(
            {"messages": [("user", task_formatted)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
        ):
            # set agent_response to be the value of the first key of the dictionary
            agent_response = next(iter(agent_response.values()))
            print_stream(agent_response)
        
        # agent_response = agent.invoke(
        #     {"messages": [("user", task_formatted)]},  {"configurable": {"thread_id": "1"}}
        # )
        structured_response = agent_response['structured_response']
        if not structured_response.success:
            print(f"worker {name} didn't finish")
            workerGood = True # if the worker agent fails, we want the supervisor to know and make a new plan.
        else:
            print(f"worker {name} finished the task successfully, now checking tool use...")
            # check if the worker used all required tools
            if task.required_tools == "" or task.required_tools == "submit_and_monitor_job":
                tool_use_passed = True
                tool_use_msg = "No required tools for this step."
            else:
                tool_use_passed, tool_use_msg = CANVAS.check_required_tool_use(task.required_tools)
            print(tool_use_msg)
            if tool_use_passed:
                # LLM sanity check
                # if LLM_check_passed:
                #     workerGood = True

                # ---- Required-quantities check (report steps only) -------
                # The supervisor specified, deterministically, the exact
                # named quantities this report must certify. Verify the
                # generated report (a) contains every required quantity and
                # (b) each required quantity's claimed value matches its
                # cited result_id. The value<->ref match is also done by the
                # judge, but checking it here is cheap and catches the
                # problem one stage earlier.
                rq_passed = True
                rq_msg = ""
                if is_report_step and required_quantities:
                    if not var.reportName or var.reportName not in CANVAS.canvas:
                        rq_passed = False
                        rq_msg = (
                            "No structured report was found on the CANVAS. A "
                            "report step MUST produce a report via "
                            "generate_structured_report containing every "
                            "required quantity."
                        )
                    else:
                        _report = CANVAS.canvas[var.reportName]
                        _claims = _report.get("numerical_results", []) or []
                        # quantity_name -> claim
                        _claim_by_name = {}
                        for _c in _claims:
                            if isinstance(_c, dict):
                                _claim_by_name[_c.get("quantity_name")] = _c
                            else:
                                _claim_by_name[getattr(_c, "quantity_name", None)] = _c
                        _missing = [q for q in required_quantities
                                    if q not in _claim_by_name]
                        # value<->ref check for the required quantities that
                        # are present.
                        _mismatched = []
                        for q in required_quantities:
                            if q not in _claim_by_name:
                                continue
                            _c = _claim_by_name[q]
                            if isinstance(_c, dict):
                                _val = _c.get("value")
                                _rid = _c.get("result_id")
                            else:
                                _val = getattr(_c, "value", None)
                                _rid = getattr(_c, "result_id", None)
                            try:
                                _ok, _vmsg = CANVAS.verify_artifact(_val, _rid)
                            except Exception as _e:  # noqa: BLE001
                                _ok, _vmsg = False, f"verification raised: {_e}"
                            if not _ok:
                                _mismatched.append(
                                    f"'{q}' (claimed value {_val} vs "
                                    f"result_id '{_rid}': {_vmsg})"
                                )
                        if _missing or _mismatched:
                            rq_passed = False
                            _parts = []
                            if _missing:
                                _parts.append(
                                    "MISSING required quantities (the report "
                                    "does not contain a claim for these exact "
                                    f"quantity_name values): {_missing}."
                                )
                            if _mismatched:
                                _parts.append(
                                    "Required quantities whose claimed value "
                                    "does NOT match the cited result_id: "
                                    f"{_mismatched}."
                                )
                            rq_msg = " ".join(_parts)

                if not rq_passed:
                    # The report is not trustworthy / not complete. Delete it
                    # so the worker cannot treat a stale incomplete report as
                    # done, record why it was removed, and admonish.
                    _removed_name = var.reportName
                    if _removed_name and _removed_name in CANVAS.canvas:
                        del CANVAS.canvas[_removed_name]
                        
                    if _removed_name:
                        CANVAS.canvas[f"report_{_removed_name}_removal_cause"] = (
                            f"Report '{_removed_name}' was removed because it "
                            f"did not satisfy the required-quantities check: "
                            f"{rq_msg}"
                        )

                    var.reportName = ""
                    task_formatted = old_task_formatted
                    task_formatted += (
                        f"\n\nWARNING: Your previous report did not pass"
                        f"the required-quantities check and was REMOVED."
                        f"{rq_msg}\n"
                        f"Your report MUST include a numerical claim for "
                        f"EVERY required quantity, using the EXACT "
                        f"quantity_name, each backed by a registered "
                        f"result_id whose value matches the claim. "
                        f"If you decided any of these values yourself "
                        f"(by your own analysis, estimation, or arithmetic) "
                        f"that is STRICTLY PROHIBITED — you must obtain the "
                        f"value from the proper tool, register it, and cite "
                        f"that result_id. If you are not sure which ID to use,"
                        f"You can use the search tool to look it up."
                        f"Do not omit any required quantity."
                    )
                    workerGood = False
                else:
                    DAG_title = f"step_{len(state.get('executed_steps', []) or [])+1}_DAG"
                    CANVAS.gen_DAG(
                        filename=f"{var.my_WORKING_DIRECTORY}/{DAG_title}.html",
                        title=DAG_title,
                    )
                    workerGood = True
            else:
                # if tool use fail, report is not trustworthy
                if var.reportName:
                    del CANVAS.canvas[var.reportName]
                task_formatted = old_task_formatted
                task_formatted += f"\n\nWARNING: You didn't use the following required tools: {tool_use_msg}. Retry again!"
    
    if not workerGood:
        print(f"Worker Agent {name} failed")
        exit(0)
    
    reportReviewResult = ""
    record_verdict = ""   # structured verdict for this record (point 5)
    record_judge_key = "" # JUDGE:: key for non-pass reports (point 5)
    config = var.OTHER_GLOBAL_VARIABLES
    if var.reportName:
        print("#######################")
        print("Judging")
        print("#######################")
        _dbg(f"[PT4 judge] verifying report '{var.reportName}' ...")
        myJudge = judge()
        rawReport = verify_structured_report(var.reportName, sensitive_parameters=config["sensitive_para"], judge=myJudge)
        def _format_verification_issue(issue: dict) -> str:
            """Render one verifier issue into supervisor-facing text.

            The safety guard's verifier (see
            safety_guard.summarize_verification_for_supervisor /
            ReportVerificationIssue) now emits a richer, structured issue
            schema. The old flat fields (`level`, `location`, `verdict`,
            `message`) no longer exist; each issue now carries:
              - issue_number:        1-indexed position in the issue list
              - category:            IssueCategory.* string (the error type)
              - severity:            "fail" | "warning"
              - where:               structured location dict (claim_name /
                                     result_id / tool_name / parameter / etc.)
              - context_at_site:     the merged `Context:` line at the
                                     offending site, if extractable
              - problem:             one-sentence plain-prose statement of
                                     what is wrong
              - judge_reasoning:     the LLM judge's verdict text, when the
                                     issue originated from a judge call
              - remediation_options: the verifier's proposed, general
                                     fix-paths for this error category

            `remediation_options` is the safety guard's whole purpose here:
            it tells the supervisor, in general, how each type of error can
            be resolved. It is surfaced verbatim so the supervisor can pick
            the applicable fix-path.
            """
            num = issue.get("issue_number", "?")
            category = issue.get("category", "uncategorized")
            severity = issue.get("severity", "fail")
            lines = [
                f"\n--- Issue #{num} "
                f"[category={category}, severity={severity}] ---"
            ]

            where = issue.get("where") or {}
            if where:
                where_str = ", ".join(f"{k}={v!r}" for k, v in where.items())
                lines.append(f"Location: {where_str}")

            context_at_site = issue.get("context_at_site")
            if context_at_site:
                lines.append(f"Context at site: {context_at_site}")

            problem = issue.get("problem")
            if problem:
                lines.append(f"Problem: {problem}")

            judge_reasoning = issue.get("judge_reasoning")
            if judge_reasoning:
                lines.append(f"Judge reasoning: {judge_reasoning}")

            remediation_options = issue.get("remediation_options") or []
            if remediation_options:
                lines.append("Please follow the instruction below to fix the issue:")
                lines.append("".join(remediation_options))

            return "\n".join(lines)

        verdict = rawReport["overall_verdict"]
        summary_line = rawReport.get("summary", "")
        n_fails = rawReport.get("n_fails", 0)
        n_warnings = rawReport.get("n_warnings", 0)
        record_verdict = verdict  # 'pass' | 'warning' | 'fail' (point 5)

        if verdict == "pass":
            # Nothing to read; keep the inline marker to a single token and
            # write no canvas record (no issues to remediate).
            reportReviewResult = "PASS"
            _dbg(f"[PT4 judge] report '{var.reportName}' verdict=PASS; "
                 f"no JUDGE:: doc written (nothing to remediate). Inline "
                 f"marker kept to a single token.")
        else:
            # Build the FULL feedback document and store it on the canvas
            # under a guarded JUDGE:: key, so it never pollutes the running
            # message history. Only a short pointer goes inline (into
            # verdict_line). Re-judgements of the same report name get a new
            # version (JUDGE::<name>, JUDGE::<name>_v2, ...).
            header = (
                f"Verifier feedback for report '{var.reportName}'\n"
                f"Overall verdict: {verdict.upper()} "
                f"({n_fails} fail(s), {n_warnings} warning(s))\n"
                f"Summary: {summary_line}\n"
                f"{'='*45}"
            )
            body_parts = [header]
            for issue in rawReport["issues"]:
                body_parts.append(_format_verification_issue(issue))
                body_parts.append("-----------------------------")
            if verdict == "fail":
                body_parts.append(
                    "\nHow to respond: for each issue above, pick whichever "
                    "of the listed remediation options fits the situation, "
                    "then insert plan steps that obtain the value properly "
                    "and re-schedule the SAME report with the SAME "
                    "required_quantities. The report must be truthful and "
                    "accurate based on the data you have; it must NOT contain "
                    "fabricated information unsupported by data. Do NOT make a "
                    "report pass by dropping a failing required_quantity "
                    "unless it was scheduled in error to begin with."
                )
            feedback_doc = "\n".join(body_parts)

            judge_key, _ = CANVAS.framework_write_versioned(
                f"JUDGE::{var.reportName}", feedback_doc
            )
            record_judge_key = judge_key  # for compression advisory (point 5)
            _dbg(f"[PT4 judge] report '{var.reportName}' verdict="
                 f"{verdict.upper()} ({n_fails} fail/{n_warnings} warn). "
                 f"Full feedback -> canvas '{judge_key}' ({len(feedback_doc)} "
                 f"chars kept OUT of message history); only a 1-line pointer "
                 f"goes inline.")

            # Short inline pointer the supervisor sees in the executed-steps
            # history. It names the canvas key to read for the detail.
            verdict_word = "FAIL" if verdict == "fail" else "WARNING"
            reportReviewResult = (
                f"{verdict_word} — {n_fails} fail(s), {n_warnings} "
                f"warning(s). Full verifier feedback at canvas key "
                f"'{judge_key}'; READ it before editing the plan."
            )

        var.All_Report_Names.append(copy.deepcopy(var.reportName))
        with open(f"{var.my_WORKING_DIRECTORY}/All_Report_Names.txt", "a") as f:
            f.write(f"{var.reportName}\n")
        var.reportName = ""
        
    
    # ---- Advance the plan: pop the just-executed step (plan[0]) into
    # executed_steps, bundled with the worker's response and the verifier
    # verdict line. This is what guarantees plan[0] is ALWAYS the next step
    # to execute when the supervisor runs again. ----
    executed_steps = list(state.get("executed_steps", []) or [])
    # verdict_line: only meaningful for report steps; empty otherwise so the
    # supervisor/worker history omits the verifier clause for non-report
    # steps (point 4).
    verdict_line = reportReviewResult.strip() if (task.required_tools == "generate_structured_report" and reportReviewResult.strip()) else ""
    executed_steps.append(ExecutedRecord(
        step=task.step,
        agent=name,
        required_tools=task.required_tools,
        worker_summary=structured_response.summary,
        verdict_line=verdict_line,
        verdict=record_verdict,
        judge_key=record_judge_key,
    ))
    _dbg(f"[PT1 advance] popped step into executed_steps as record "
         f"#{len(executed_steps)} (agent={name}, "
         f"report={'yes' if record_verdict else 'no'}"
         f"{', verdict='+record_verdict if record_verdict else ''}). "
         f"plan: {len(plan)} -> {len(plan)-1} step(s).")

    # ---- Episode compression (point 5): if this round just closed a report
    # with a clean PASS, auto-compress the span of steps it closes (the
    # ordinary records between it and the previous report/start). Non-pass
    # reports are not auto-compressed; they're offered to the supervisor as
    # opt-in 'compression opportunities' instead.
    _es_before = len(executed_steps)
    executed_steps = _auto_compress_on_pass(executed_steps, state.get("inputs", ""))
    if len(executed_steps) != _es_before:
        _dbg(f"[PT5 compression] clean-PASS auto-compression fired: "
             f"executed_steps {_es_before} -> {len(executed_steps)} record(s). "
             f"Span archived to canvas (non-destructive); inline history "
             f"shrunk. Latest record: {executed_steps[-2].step!r} (compressed)."
             if len(executed_steps) >= 2 else
             f"[PT5 compression] auto-compression fired: {_es_before} -> "
             f"{len(executed_steps)}.")
    elif record_verdict in ("warning", "fail"):
        _dbg(f"[PT5 compression] report verdict={record_verdict}: span NOT "
             f"auto-compressed; will be offered to supervisor as an opt-in "
             f"compression opportunity.")

    new_plan = list(plan[1:])  # drop the executed step

    print_stream(structured_response.summary)
    # CANVAS.snap_save()
    return {
        "plan": new_plan,
        "executed_steps": executed_steps,
        "canvas": CANVAS.canvas,
        "artifacts": CANVAS.result_registry
    }
    

def recusive_agent_node(state, agent, name):
    print(f"Agent {name} is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"Agent {name} is processing!!!!!\n")
    
def whos_next(state):
    return state["next"]

        
def create_planning_graph(config: dict) -> StateGraph:
    # create a file named status.txt in the working directory
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    with open(f"{WORKING_DIRECTORY}/status.txt", "w") as f:
        f.write("run")
    with open(f"{WORKING_DIRECTORY}/judge_status.txt", "w") as f:
        f.write("run")
    
    # Define the model
    # llm = ChatAnthropic(model="claude-opus-4-7", api_key=config['ANTHROPIC_API_KEY'])
    # workerllm = ChatAnthropic(model="claude-opus-4-7", api_key=config['ANTHROPIC_API_KEY'], tool_choice="auto")
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    workerllm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0, tool_choice="auto")
    # workerllm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # workerllm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"], model_kwargs={'parallel_tool_calls': False})
    # llm = ChatDeepSeek(model_name=config["DeepSeek_MDL"], api_key=config['DeepSeek_API_KEY'], api_base=config['DeepSeek_BASE_URL'], temperature=0.0)
    
    var.my_WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # Expose the judge factory for the call-time tool guards (point 6). The
    # on-site math/extraction guards build their own judge via this factory,
    # so they share the same run-control (pause) and logging behavior as the
    # report verifier's judge, while remaining an independent LLM call.
    var.JUDGE_FACTORY = judge
    
    if not eval(config["SAVE_DIALOGUE"]):
        var.my_SAVE_DIALOGUE = False
    
    # CANVASCheckpoints = []
    # with open(f"{WORKING_DIRECTORY}/canvas_checkpoints.pickle", 'rb') as f:
    #     CANVASCheckpoints = pickle.load(f)
    
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
        supervisor_get_available_report_names,
        debug_artifact_chain,
        insert_plan_steps,
        modify_plan_steps,
        delete_plan_steps,
        compress_history,
        ]
    
    supervisor_agent = create_agent(
        model=llm,
        tools=supervisor_tools, 
        system_prompt=supervisor_prompt,
        # Structured output via ToolStrategy (tool-calling fallback)
        response_format=ToolStrategy(Act, handle_errors=on_act_parse_error),  # Or ProviderStrategy for native models
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors]
    )
    
    # supervisor_agent = create_react_agent(llm, tools=supervisor_tools,
    #                                prompt=supervisor_prompt, response_format=Act)   
    supervisor_node = functools.partial(supervisor_chain_node, agent=supervisor_agent, name="Supervisor_Agent")
        
    ### DFT Agent
    dft_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        version_controlled_edit_canvas_doc,
        write_progress_note,
        calculate_formation_E,
        generateSurface_and_getPossibleSite,
        generate_myAdsorbate,
        add_myAdsorbate,
        init_structure_data,
        find_pseudopotential,
        write_QE_script_w_ASE,
        calculate_lc,
        generate_convergence_test,
        find_optimal_parameter,
        find_optimal_parameter_from_derived,
        generate_eos_test,
        read_energy_from_output,
        get_convergence_suggestions,
        analyze_BEEF_result,
        extract_numeric_from_tool_output,
        extract_text_from_tool_output,
        math_expression_tool,
        generate_structured_report,
        list_referenceable_inputs,
        search_artifacts
        # get_ase_atoms_property,
        # inspect_ase_atoms,
        ]
    dft_agent = create_agent(
        model=workerllm,
        tools=dft_tools,
        system_prompt=dft_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[DisableParallelToolCallsMiddleware(), RestrictToolsAfterReportMiddleware(), handle_tool_errors]
    )
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent")


    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        version_controlled_edit_canvas_doc,
        write_progress_note,
        submit_and_monitor_job,
        add_resource_suggestion
        ]
    hpc_agent = create_agent(
        model=workerllm,
        tools=hpc_tools,
        system_prompt=hpc_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[DisableParallelToolCallsMiddleware(), RestrictToolsAfterReportMiddleware(), handle_tool_errors]
    )

    hpc_node = functools.partial(worker_agent_node, agent=hpc_agent, name="HPC_Agent")
    
    ### MD Agent
    # md_tools = [
    #     find_classical_potential,
    #     init_structure_data,
    #     write_LAMMPS_script
    # ]
    
    # md_agent = create_react_agent(llm, tools=md_tools,
    #                               state_modifier=md_agent_prompt)
    
    # md_node = functools.partial(worker_agent_node, agent=md_agent, name="MD_Agent")

    # save_graph_to_file(dft_agent, config['working_directory'], "dft_agent")
    


    # Create the graph
    graph = StateGraph(PlanExecute)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
    # graph.add_node("MD_Agent", md_node)
    # graph.add_node("CSS_Agent", css_node)

    graph.add_node("Supervisor", supervisor_node)
    
    for member in members:             
    # We want our workers to ALWAYS "report back" to the supervisor when done
        graph.add_edge(member, "Supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    conditional_map["Supervisor"] = "Supervisor" 
    graph.add_conditional_edges("Supervisor", whos_next, conditional_map)
    graph.add_edge(START, "Supervisor") 
    
    # return graph.compile(checkpointer=checkpointer)
    # return graph.compile()
    return graph