from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ToolCallLimitMiddleware, AgentMiddleware, ModelRequest, wrap_tool_call, before_model
from langchain.agents import create_agent, AgentState

import copy
from langgraph.types import Command
try:
    from typing import NotRequired
except ImportError:  # python < 3.11
    from typing_extensions import NotRequired

from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple, Union, Any
import functools
import pandas as pd
import os
import threading
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
from pydantic import BaseModel, Field
from pydantic import model_validator

from src.tools import *
from src.prompt import dft_agent_prompt,hpc_agent_prompt,supervisor_prompt, oer_agent_prompt, boss_prompt
from src.disposition_messages import (
    classify_wait_handback,
    format_study_status_block,
    format_supervisor_handback_directive,
)
from src.forgotten_jobs import find_forgotten_jobs
from src.history_log import write_history
from src.past_steps import (
    HARD_TOKENS,
    K_VERBATIM,
    STEP_CHAR_CAP,
    build_report_digest,
    build_summary_digest,
    count_leading_digests,
    first_verbatim_index,
    fold_digests,
    is_digest,
    plan_compaction,
    render_past_steps,
    should_request_report,
    steps_completed,
    truncate_step_text,
)
from src.supervisor_actions import (
    LEGACY_KINDS,
    action_warning,
    route_action,
)
from src import var
from src.myCANVAS import CANVAS
from src.live_visualizer import LiveVisualizer
from gnome_dreams_oer_screening.explog.explog import EXPLOG

import json, hashlib, time
from collections import defaultdict, deque
from datetime import timedelta  # was only reaching us via `from src.tools import *`
import traceback

viz = LiveVisualizer(
    canvas_obj=CANVAS,       # has a `.canvas` dict
    explog_obj=EXPLOG,       # has `.relational_frame.<name>.df`
)

members = ["OER_Agent"]

class myStep(BaseModel):
    """Step in the plan."""

    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )
    required_tools: List[Literal[
                            "inspect_my_canvas",
                            "write_my_canvas",
                            "read_my_canvas",
                            "OER_data_analasis_v2",
                            "browse_df",
                            "arXiv_search",
                            "enter_candidate_in_log",
                            "submit_dft_job",
                            "get_terminations_ranking",
                            "list_adsorption_sites",
                            "read_explog",
                            "wait_for_update",
                            "query_explog",
                            "get_disposition_info",
                            "update_disposition_info",
                            # "math_expression_tool",
                            # "extract_numeric_from_tool_output",
                            "write_report",
                            "search_artifacts",
                            ""
                ]] = Field(f"must-use tools for this step, should be a subset of the tools available to the agent. read the CANVAS with key Worker_available_tools to see more details about each tools.")
    enforce_queue_floor: bool = Field(True, description=f"Whether the worker must keep the HPC queue stocked (at least {var.QUEUE_MIN_PENDING} QUEUED jobs) before it may wait for current jobs to finish. Keep True to ensure maximum HPC utilization: the worker submits ready continuation work itself, and only returns to the supervisor to discuss expanding the study when no ready work remains. Set False only when genuinely winding the study down -- when the remaining time is too short for newly-submitted jobs to finish -- so the worker may instead wait for and finalize the in-flight results. False is honored only within the final {var.FLOOR_DISARM_WINDOW_DAYS} days of the {var.STUDY_BUDGET_DAYS}-day study budget; set earlier, it is ignored (coerced back to True) and the worker is told so.")

class myPastStep(BaseModel):
    """Step in the plan."""

    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )
    timeStamp: Any = Field(description="The time when the step is completed.")
    timeSpent: str = Field(description="The time spent on this step.")
class Plan(BaseModel):
    """Need to add/modify current plan, which is going to be followed by your worker agents in future"""
    kind: Literal["plan"] = "plan"
    steps: List[myStep] = Field(
        description=f"""
        Steps to follow in future. Each step is a tuple of (step, agent). agent can only be chosen from {members}.
        """
        # description="different steps to follow, should be in sorted order"
        # description="""different steps to follow (first element of the Tuple), and the agent in charge for each step (second element of the Tuple),
        # should be in sorted order by the order of execution"""
    )
    
class Response(BaseModel):
    """Supervisor's proposed final answer for boss review."""
    kind: Literal["response"] = "response"
    response: str

# The two ways to keep going with the CURRENT plan, named for what they do to the
# plan pointer. `Advance` was called `NoChange` until 2026-08-18: that name
# described the wrong half (the action DOES change the plan -- it removes the
# completed step) and was read as "nothing to do here". See
# src/supervisor_actions.py for the outage that came of it.
class Advance(BaseModel):
    """Finish the first step of the current plan and execute the SECOND step next, unchanged. The first step is REMOVED, so the plan must hold at least two steps."""
    kind: Literal["advance"] = "advance"
    comment: str = Field(description="any comment from the supervisor if needed, otherwise just put 'First step finished, continue to execute the second step of the original plan.'")


class Repeat(BaseModel):
    """Execute the CURRENT first step of the plan AGAIN. The plan is NOT modified and no step is removed. Use this when the current step is a standing duty that should keep cycling."""
    kind: Literal["repeat"] = "repeat"
    comment: str = Field(description="why the current step should be executed again rather than advanced past.")


class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, Advance, Repeat, Response] = Field(
        description="""Action to perform. Each one is described by what it does to the plan -- choose on that basis, not on whether the plan "needs changing".
        Plan: replace the plan with the steps you write; the first of them is executed next. Use it when steps must be added or adjusted.
        Advance: the first step of the current plan is FINISHED -- remove it and execute the second step unchanged. Requires at least two steps in the plan.
        Repeat: execute the CURRENT first step AGAIN; the plan is not modified and nothing is removed. Use this when the current step is a standing duty that should keep cycling.
        Response: you believe the overall goal is complete -- submit a proposed final answer for boss review.""",
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

        # Case 2: model wrapped variant as {"Repeat": {...}}
        wrapped = {
            "Plan": "plan",
            "Advance": "advance",
            "Repeat": "repeat",
            "Response": "response",
            "NoChange": "advance",   # pre-2026-08-18 name for Advance
        }
        if isinstance(action, dict) and len(action) == 1:
            key = next(iter(action))
            if key in wrapped:
                inner = dict(action[key])
                inner.setdefault("kind", wrapped[key])
                action = inner

        # Case 3: legacy `kind`. MUST come after case 2, which leaves an inner
        # kind untouched when the model supplied one. The supervisor's inner agent
        # appends retries to the previous attempt's conversation in a shared
        # checkpoint namespace (see the sup_good loop), so the first resume after
        # the rename can still surface kind="no_change".
        if isinstance(action, dict) and action.get("kind") in LEGACY_KINDS:
            action = {**action, "kind": LEGACY_KINDS[action["kind"]]}

        return {**data, "action": action}
    
class BossReview(BaseModel):
    """Structured response from the boss review gate."""

    decision: Literal["approve", "revise"] = Field(
        description="Review decision. Use 'approve' if the supervisor draft can be returned to the user, otherwise use 'revise'."
    )

    feedback: str = Field(
        description="Concrete review feedback. Leave this empty if the decision is 'approve'. If the decision is 'revise', explain exactly what is missing, unclear, or contradictory."
    )
    
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
    

class PlanExecute(TypedDict):
    inputs: str
    plan: List[myStep]
    # A BOUNDED ledger now: [digest steps ...] + [last K_VERBATIM steps]; see
    # src/past_steps.py. NOTE: the schema (channel set) is intentionally left
    # unchanged for resume compatibility -- the monotone step count is derived from
    # the ledger via past_steps.steps_completed(), NOT stored in a new channel.
    past_steps: List[myStep]
    draft_response: str
    boss_feedback: str
    response: str
    # VESTIGIAL -- declared so they can be EMPTIED, never written again.
    #
    # These four used to hold the real CANVAS/EXPLOG/artifact state and were
    # dropped from this schema: each is persisted by its own owner
    # (canvas.pickle, artifacts.sqlite, vasp_calcs/explog.pkl) and carrying them
    # here re-serialized ~375 MB into EVERY tool-level checkpoint to record a
    # ~12 KB delta. The globals are rehydrated from those files at startup
    # instead. Consequence: TIME TRAVEL IS GONE -- rewinding would give an old
    # conversation against current globals. Resume is unaffected.
    #
    # Simply deleting the keys was NOT enough. LangGraph copies channel_values
    # it does not recognise straight into every new checkpoint, so a run resumed
    # from a pre-change checkpoint keeps dragging the old blobs along untouched
    # -- measured at 139.8 MB, 99.9% of the 27-05 head checkpoint, rewritten on
    # every parent super-step under durability="sync". And writing them through
    # update_state while they were OUT of schema silently did nothing (verified:
    # accepted without error, value unchanged), which is the worst kind of fix.
    # Declaring them here makes them real channels again, so `invoke.py continue`
    # can write empties into them once; nothing writes them afterwards, so they
    # stay ~5 bytes for the rest of the run.
    canvas: Dict
    artifacts: Dict
    explog_candidates: pd.DataFrame
    explog_processes: pd.DataFrame
    time: float
    next: str


# ---------------------------------------------------------------------------
# Tool-call-level checkpointing of the full state
#
# The inner create_agent graphs run as *subgraphs* of the outer PlanExecute
# graph: each node function receives the parent RunnableConfig (which carries
# the SqliteSaver + thread_id + a per-task checkpoint namespace) and passes it
# into agent.stream(). LangGraph then checkpoints every super-step of the
# react loop (model call / tool call) under a namespaced checkpoint_ns like
# "OER_Agent:<task_id>", in the SAME sqlite file — while
# graph.get_state_history() on the parent thread keeps returning only the
# round-level (checkpoint_ns == "") checkpoints, so time travel still indexes
# big rounds.
#
# Because CANVAS / EXPLOG are process globals that only enter graph state when
# a node returns, a mid-round checkpoint would otherwise contain messages but
# not canvas/explog. SyncedAgentState extends the inner agents' state schema
# with the mutable parts of PlanExecute, and StateSyncMiddleware copies the
# globals into the subgraph state after EVERY tool call, so every tool-level
# checkpoint is a complete, resumable snapshot.
# ---------------------------------------------------------------------------

class SyncedAgentState(AgentState):
    # Only the small, genuinely graph-scoped values. CANVAS / EXPLOG are NOT
    # mirrored here any more -- see full_state_snapshot below.
    curr_round_result_ids: NotRequired[List[str]]
    time: NotRequired[float]


def full_state_snapshot():
    """The per-tool-call state sync: what a mid-round checkpoint must carry
    that is not already durable on disk.

    This used to deepcopy the whole CANVAS (canvas + artifact registry) and both
    EXPLOG dataframes into every tool-level checkpoint. Measured on the 27-05
    run that was ~375 MB written per tool call -- to record a delta of about
    12 KB (one new artifact, maybe an EXPLOG row).

    All of that is now persisted by its owner as it changes:
        canvas            -> canvas.pickle          (CANVAS._save_canvas)
        artifact registry -> artifacts.sqlite       (CANVAS._register_artifact)
        EXPLOG frames     -> vasp_calcs/explog.pkl  (EXPLOG._save_pickle)
    and rehydrated from there at startup, so duplicating them here bought
    nothing except I/O. What remains is genuinely graph state:

      curr_round_result_ids -- backs CANVAS.check_required_tool_use after a
          mid-round resume; it is per-ROUND, so no file owns it.
      time -- elapsed study seconds; invoke.py rebuilds var.startTime from it.

    NOTE on crash consistency: the disk sources are written at the moment of
    mutation, so on resume they are at-or-AHEAD of the checkpoint. That is the
    safe direction -- the worst case is the agent repeating an action it does
    not remember (and submit_dft_job's duplicate guard now actually sees the
    earlier submission, which it could not when EXPLOG came from the lagging
    checkpoint).
    """
    return {
        "curr_round_result_ids": list(CANVAS.curr_round_result_ids),
        "time": time.time() - var.startTime,
    }


class StateSyncMiddleware(AgentMiddleware):
    """After each tool execution, write the full state snapshot into the
    agent (subgraph) state via Command, so the checkpoint taken at the end of
    that tool super-step contains canvas/artifacts/explog/time.

    Must be FIRST in the middleware list (outermost wrapper), so it sees the
    final ToolMessage after prevent_redundant_polling / handle_tool_errors
    have run."""

    state_schema = SyncedAgentState

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        update = full_state_snapshot()
        if isinstance(result, Command):
            result.update = {**(result.update or {}), **update}
            return result
        # plain ToolMessage -> wrap into a Command that also updates state
        update["messages"] = [result]
        return Command(update=update)

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        update = full_state_snapshot()
        if isinstance(result, Command):
            result.update = {**(result.update or {}), **update}
            return result
        update["messages"] = [result]
        return Command(update=update)

class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return await handler(request)

_POLL_HISTORY = defaultdict(lambda: deque(maxlen=10))

def _stable_json(x):
    try:
        return json.dumps(x, sort_keys=True, default=str)
    except Exception:
        return repr(x)

def _fingerprint(x):
    return hashlib.sha256(_stable_json(x).encode()).hexdigest()

POLLING_TOOLS = {
    "query_explog",
    "read_explog",
    "extract_df",
    "OER_data_analasis_v2",
    "read_df",
    "arXiv_search",
    "enter_candidate_in_log",
    "submit_dft_job",
    "browse_df",
    }

@wrap_tool_call
def prevent_redundant_polling(request, handler):
    # print(request)
    tool_name = request.tool_call["name"]
    args = request.tool_call.get("args", {}) or {}

    result = handler(request)
    

    if tool_name not in POLLING_TOOLS:
        print(f"Tool {tool_name} is not monitored for redundant polling.")
        return result

    # normalize tool output
    content = getattr(result, "content", result)
    fp = _fingerprint(content)
    arg_key = _stable_json(args)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"Tool called: {tool_name}\nwith args: {args}\n\ngot content: {content}\n\nfingerprint: {fp}\nand arg_key: {arg_key}\n\n")
    
    # choose a scope key; thread / agent / task if available
    scope = (
        request.runtime.config["configurable"].get("thread_id", "default"),
        tool_name,
        arg_key,
    )

    now = time.time()
    hist = _POLL_HISTORY[scope]
    hist.append((now, fp))

    recent_same = [
        1 for t, old_fp in hist
        if now - t < 60 and old_fp == fp
    ]
    
    print(f"has repeated {len(recent_same)} times in the last 60 seconds for the same args and result")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    complainPatient = 4
    goWildPatient = 6
    
    if len(recent_same) >= complainPatient and len(recent_same) < goWildPatient:
        toolMsg = f"""
        You have called {tool_name} with the same arguments {args} and got the same result {content} for {len(recent_same)} times in the last 60 seconds.
        Please do not call {tool_name} again right now. You could:
            1) Move on to other tasks and come back to this later.
            2) Call wait_for_update and wait for next dft job to finish if you have nothing else to do
        You have a grace period of {goWildPatient - len(recent_same)} more calls with the same args and result before the entire system halts indefinitely to prevent damage. Please use this grace period wisely to avoid halting the system.
        """
        
        return ToolMessage(
            content=(toolMsg),
            tool_call_id=request.tool_call["id"],
        )
        
    if len(recent_same) >= goWildPatient:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"TOOL WITH NAME {tool_name} AND ARGS {args} HAS BEEN CALLED MORE THAN {goWildPatient} TIMES WITH THE SAME RESULT IN THE LAST 60 SECONDS. PLEASE CHECK IF THERE IS A BUG IN THE MODEL OR THE TOOL IMPLEMENTATION CAUSING THIS ISSUE.")
        print("STUDY HALTED TO PREVENT POTENTIAL DAMAGE. PLEASE FIX THE ISSUE AND RESUME.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # time.sleep(99999)
        quit()

    return result

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
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)}), the traceback is: {traceback.format_exc()}",
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
        "Return exactly one of {Plan, Advance, Repeat, Response} as `action`, "
        "with the inner fields directly — do NOT wrap in {'Repeat': {...}}."
    )

teamCapability = """
<DFT Agent>:
    - Create intial structure of the system
    - Find pseudopotential
    - Write initial script
    - generate convergence test input files
    - determine the best parameters from convergence test result
    - generate EOS calculation input files using the best parameters
    - generate production run input files
    - generate BEEF input files from finished relax calculation
    - Read output file to get energy
    - Calculate lattice constant
    - Calculate formation energy
<HPC Agent>:
    - find job list from the job list file
    - Add resource suggestion base on the DFT input file
    - Submit job to HPC and report back once all jobs are done
"""

teamRestriction = """
<DFT Agent>:
    - Cannot submit job to HPC
<HPC Agent>:
    - Cannot determine the best parameters from convergence test result
"""


def print_stream(s, DAG=None):
    # viz.on_event throttles its own writes (see var.LIVE_VIZ_* and
    # LiveVisualizer._flush); gen_DAG has no such gate, so honour the kill
    # switch here. These step_<N>_DAG.html files reached 18 MB each on the
    # 27-05 run (2.8 GB in total) and are purely for human inspection --
    # nothing in the workflow reads them back.
    viz.on_event(s, DAG=DAG)

    if DAG is not None and getattr(var, "LIVE_VIZ_ENABLED", True):
        DAG_title = f"step_{DAG}_DAG"
        CANVAS.gen_DAG(
            filename=f"{var.my_WORKING_DIRECTORY}/{DAG_title}.html",
            title=DAG_title,
        )
    
    if "messages" not in s:
        print("#################")
        write_history("#################\n")
        print(s)
        write_history(repr(s) + "\n")
    else:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
            write_history(repr(message) + "\n")
        else:
            if hasattr(message, 'usage_metadata'):
                var.TOKEN_USAGE.append(message.usage_metadata)
                print(f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}")
                write_history(f"input_tokens: {message.usage_metadata['input_tokens']}, output_tokens: {message.usage_metadata['output_tokens']}\n")
            message.pretty_print()
            write_history(message.pretty_repr() + "\n")
    print()
    write_history("\n")


def render_history(state, *, with_timing: bool = True) -> str:
    """The single rendering of past_steps, shared by the boss, supervisor and
    worker prompts (it used to be four copy-pasted f-strings that could drift).

    Numbering is ABSOLUTE, so after a compaction the supervisor still sees
    "27." rather than a list that restarts at "1." and implies the study is
    younger than it is. The absolute index is derived from the ledger's digest
    block (past_steps.first_verbatim_index), not a stored counter.
    """
    return render_past_steps(
        state["past_steps"],
        with_timing=with_timing,
        start_index=first_verbatim_index(state["past_steps"]),
    )


def boss_node(state, config, agent=None, name=None):
    # Parent RunnableConfig is injected by LangGraph (carries the checkpointer,
    # thread_id and this task's checkpoint namespace). Propagating it into
    # agent.stream() makes the inner agent a checkpointed subgraph.
    inner_cfg = {**config, "recursion_limit": 1000}
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, {name} is waiting. cwd: {var.my_WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
            
    # Time thing. may or may not need this for the boss 
    currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    timeElapsed = timedelta(seconds= time.time() - var.startTime)
    print(f"{name} is processing!!!!! Current time: {timeElapsed}.")
    write_history(f"{name} is processing!!!!! Current time: {timeElapsed}.\n")

    # can't print state anymore because it now contains canvas and explog, and printing them will cause too much output
    # print(state)

    old_tasks_string = render_history(state)

    # The boss used to be handed ONLY `Current time: <elapsed>` -- no budget, no
    # remaining time, no queue state -- and on 22-07 approved a study as complete
    # at 14.68 of 30 days with an idle cluster. Nothing here GATES its decision;
    # it is simply given the numbers that decision needs. Same two reads as the
    # handback classification below, but wrapped: boss_node runs on every review,
    # so a raise here would kill the study at its very last step, and reporting
    # "unavailable" beats defaulting to 0/0 (which reads as an idle cluster and
    # would argue for exactly the wrong conclusion).
    try:
        _pending = EXPLOG.job_handler.count_pending()
        _running = EXPLOG.relational_frame.processes.df["status"].tolist().count("running")
    except Exception:
        traceback.print_exc()
        _pending = _running = None
    study_status = format_study_status_block(
        elapsed_seconds=time.time() - var.startTime,
        pending_count=_pending,
        running_count=_running,
    )

    bossMessage = f"""
    The overall goal is:
    {state['inputs']}

    This is what has been done so far:
    {old_tasks_string}

    The supervisor's draft final answer is:
    {state['draft_response']}

    Current time: {timeElapsed}.

    {study_status}

    Please review the supervisor's draft final answer and decide whether to approve it or send it back for revision.
    """

    # Pritn the message to the boss:
    write_history(bossMessage + "\n")


    # stream_mode MUST be pinned: with the parent config propagated, the
    # inner stream otherwise inherits the parent's stream context and yields
    # "values"-mode chunks (full cumulative state) instead of {node: update},
    # breaking print_stream and the structured_response extraction below.
    for agent_response in agent.stream(
        {"messages": [("user", bossMessage)]},  inner_cfg, stream_mode="updates", durability="sync"
    ):
        # set agent_response to be the value of the first key of the dictionary
        agent_response = next(iter(agent_response.values()))
        print_stream(agent_response)

    agent_response = agent_response['structured_response']

    if agent_response.decision == "approve":
        return {
            "response": state["draft_response"],
            "boss_feedback": "",
            "next": "FINISH",
        }
    elif agent_response.decision == "revise":
        return {
            # Boss rejection stores concrete review feedback and hands control back to the supervisor.
            "boss_feedback": agent_response.feedback,
            "next": "Supervisor",
        }
    else:
        raise ValueError(f"Unexpected boss decision: {agent_response.decision}")
            
def supervisor_chain_node(state, config, agent=None, name=None):
    # Parent RunnableConfig injected by LangGraph -> inner agent runs as a
    # checkpointed subgraph (see StateSyncMiddleware notes above).
    inner_cfg = {**config, "recursion_limit": 1000}
    
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, supervisor is waiting. cwd: {var.my_WORKING_DIRECTORY}")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
    
    currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    timeElapsed = timedelta(seconds= time.time() - var.startTime)
    
    print(f"supervisor is processing!!!!! Current time: {timeElapsed}.")
    write_history(f"supervisor is processing!!!!! Current time: {timeElapsed}.\n")


    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan))
    # task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = render_history(state)

    current_boss_feedback = state["boss_feedback"].strip() # Remove leading/trailing whitespaces
    if current_boss_feedback == "":
        current_boss_feedback = "None"
    
    if state["boss_feedback"].strip() != "":
        supervisorMessage =  f"""
        Your available agents are: {members}.

        The overall goal is: {state['inputs']}. 

        Below is what's left from your previous plan:
        {plan_str}

        this is what has been done:
        {old_tasks_string}

        Your previous draft final answer has been reviewed and rejected by the boss and received the following feedback:
        {current_boss_feedback}

        Current time: {timeElapsed}.
        
        Please inspect and extract related information from CANVAS and EXPLOG, then update the plan accordingly.
        """
    elif len(plan) == 0:
        supervisorMessage =  f"""
        Current time: {timeElapsed}.
        
        The overall goal is: {state['inputs']}.

        Nothing has been done yet and there is no plan yet. 

        Please inspect and extract related information from CANVAS, then only update the plan accordingly if needed.
        """
    else:
        supervisorMessage =  f"""
        Your available agents are: {members}.

        The overall goal is: {state['inputs']}. 

        the current plan is:
        {plan_str}

        this is what has been done:
        {old_tasks_string}
        
        Current time: {timeElapsed}.

        Please inspect and extract related information from CANVAS and EXPLOG, then update the plan accordingly. 
        Choose <Advance> if the first step of the current plan is finished and the second step should be executed next without any change -- this REMOVES the first step, so it needs at least two steps in the plan.
        Choose <Repeat> if the worker should execute the CURRENT first step again -- the plan is not modified and nothing is removed. This is the right choice for a standing-duty step that should keep cycling.
        Otherwise choose <Plan> and rewrite the plan with the steps you want, and the first step of the plan you just wrote will be executed.
        Or if you think the overall goal is finished, you can choose <Response> and write a draft final answer for the boss review.
        """

    # --- Queue-floor handback -------------------------------------------------
    # wait_for_update raises var.wait_handback when it refuses on the "expand"
    # path (queue below floor with nothing ready, or idle) -- a situation only
    # the supervisor can resolve (Path A ready work is the worker's own standing
    # duty and raises no handback). Consume-and-clear it here (one-shot) and
    # inject the expand directive so the handback is ACTED on instead of reaching
    # the supervisor only as ignorable worker prose. The path is re-derived from
    # live EXPLOG (self-correcting: if the queue refilled since the worker's
    # turn, classify returns None and nothing is injected). Uses only the pure
    # gate -- never the real wait tool, whose loop sleeps regardless of patience.
    if getattr(var, "wait_handback", False):
        var.wait_handback = False
        _hb_forgotten = find_forgotten_jobs(EXPLOG, var.GO_DEV_OH_THRESHOLD)
        _hb_status = EXPLOG.relational_frame.processes.df["status"].tolist()
        _hb_path = classify_wait_handback(
            candidates_need_disposition=EXPLOG.candidates_needing_disposition(),
            pending_count=EXPLOG.job_handler.count_pending(),
            running_count=_hb_status.count("running"),
            enforce_queue_floor=getattr(var, "enforce_queue_floor", True),
            queue_min_pending=var.QUEUE_MIN_PENDING,
            remaining_seconds=var.STUDY_BUDGET_SECONDS - (time.time() - var.startTime),
            forgotten_jobs=_hb_forgotten,
        )
        if _hb_path is not None:
            supervisorMessage = (
                supervisorMessage
                + "\n\n"
                + format_supervisor_handback_directive(_hb_path, _hb_forgotten)
            )

    # --- Early floor-disarm notice ---------------------------------------------
    # worker_agent_node raises var.floor_coerce_notice when it ignored a plan
    # step's enforce_queue_floor=False (requested before the final disarm
    # window). Consume-and-clear it and tell the supervisor DIRECTLY -- the rule
    # must be learned from feedback at the moment it matters, not only from the
    # myStep field description.
    if getattr(var, "floor_coerce_notice", False):
        var.floor_coerce_notice = False
        _notice_remaining = var.STUDY_BUDGET_SECONDS - (time.time() - var.startTime)
        supervisorMessage += (
            f"\n\nNOTE -- QUEUE FLOOR: your last plan set enforce_queue_floor="
            f"False on a step, but the study still has "
            f"{_notice_remaining / 86400:.1f} days remaining, so it was IGNORED "
            f"(the floor stayed armed). enforce_queue_floor=False is honored "
            f"only within the final {var.FLOOR_DISARM_WINDOW_DAYS} days of the "
            f"{var.STUDY_BUDGET_DAYS}-day budget -- you cannot disarm the floor "
            f"at will before then, so do not plan around an early disarm; plan "
            f"submit/expansion steps instead."
        )

    # --- Context budget (SOFT watermark) --------------------------------------
    # The step history used to grow ~666 tokens/step forever and was injected into
    # every prompt. It is now bounded, and a report is the cheapest way to bound it:
    # once the worker writes one, the steps it covers collapse to a one-line pointer
    # at the report on CANVAS, at zero LLM cost.
    #
    # So when the history crosses SOFT, STEER -- ask the supervisor to plan a report
    # step. It measures its OWN state["past_steps"], so no flag has to be carried
    # over from the worker and there is no process-global to lose on resume.
    #
    # This is advisory: the worker may ignore it. The HARD watermark in
    # worker_agent_node is the backstop that makes the bound unconditional.
    if should_request_report(state["past_steps"]):
        supervisorMessage += (
            "\n\nCONTEXT BUDGET -- WRITE A REPORT NOW.\n"
            "The record of completed steps is long enough that it is crowding the context "
            "window of both you and your worker. The remedy is a report: once the worker "
            "calls write_report, every step that report covers is replaced by a short "
            "pointer to it on CANVAS, and the detail stays retrievable via read_my_canvas.\n"
            "Add a step NOW instructing the worker to write an intermediate report covering "
            "the work done SINCE THE LAST REPORT (not the whole study -- earlier work is "
            "already captured in earlier reports), and pin required_tools=['write_report'] "
            "on that step. Give the report a NEW, unique name; reports are never overwritten.\n"
            "If you ignore this, the history will be compacted mechanically instead, which "
            "produces a worse summary than one you commission deliberately."
        )

    old_supervisorMessage = supervisorMessage
    sup_good = False
    sup_good_patient = 3
    while not sup_good and sup_good_patient > 0:
        sup_good_patient -= 1
        # Log the ACTUAL prompt, inside the loop so retries (which append a
        # WARNING to old_supervisorMessage) are captured too. boss_node has
        # always written its own bossMessage; the supervisor's was the one
        # prompt in the workflow that was never recorded, which makes every
        # injected directive -- handback, floor-coercion notice, context budget
        # -- unauditable after the fact.
        write_history(supervisorMessage + "\n")
        try:
            # NOTE: with subgraph checkpointing, retry iterations of this loop
            # share one checkpoint namespace, so the retry message is APPENDED
            # to the previous attempt's conversation (add_messages reducer)
            # instead of starting a fresh one.
            for agent_response in agent.stream(
                {"messages": [("user", supervisorMessage)]},  inner_cfg, stream_mode="updates", durability="sync"
            ):
                # set agent_response to be the value of the first key of the dictionary
                agent_response = next(iter(agent_response.values()))
                # print("agent reposonse:")
                # print(agent_response)
                print_stream(agent_response)
        except Exception as e:
            # fall back to a default Act, retry with a smaller schema, escalate to a human, etc.
            print(f"Supervisor halted: structured output failures: {e}")
            exit()

    
        agent_response = agent_response['structured_response']
        sup_good = True
        if isinstance(agent_response.action, Plan):
            for step in agent_response.action.steps:
                ToolList = [
                    "inspect_my_canvas",
                    "write_my_canvas",
                    "read_my_canvas",
                    "OER_data_analasis_v2",
                    "browse_df",
                    "arXiv_search",
                    "enter_candidate_in_log",
                    "submit_dft_job",
                    "get_terminations_ranking",
                    "list_adsorption_sites",
                    "read_explog",
                    "wait_for_update",
                    "query_explog",
                    "get_disposition_info",
                    "update_disposition_info",
                    # "math_expression_tool",
                    # "extract_numeric_from_tool_output",
                    "write_report",
                    "search_artifacts",
                    ""
                ]
                wrongTools = set(step.required_tools) - set(ToolList)
                print(f"wrongTools: {wrongTools}")
                if len(wrongTools) > 0:
                    supervisorMessage = old_supervisorMessage + f"\n\nWARNING: In step '{step.step}', you required the following tools that are not in the tool list: {', '.join(wrongTools)}. Please check the CANVAS and try again!"
                    sup_good = False
                    break
        
        # Plan-length rules for Advance/Repeat live in src/supervisor_actions.py so
        # they can be tested without importing this module (and with it the GNoME
        # database). The warning names Repeat as the alternative: before it existed,
        # this refusal offered only Plan or Response -- neither of which was what
        # the supervisor wanted -- and three of them in a row ended the run.
        _warning = action_warning(agent_response.action.kind, len(state["plan"]))
        if _warning is not None:
            sup_good = False
            supervisorMessage = old_supervisorMessage + _warning
        elif isinstance(agent_response.action, Plan) and len(agent_response.action.steps) == 0:
            sup_good = False
            supervisorMessage = old_supervisorMessage + f"\n\nWARNING: you chose to rewrite the plan, but the new plan is empty. Please provide a non-empty plan with at least one step, or if you think the overall goal is finished, you can choose 'Response' and write a draft final answer for the boss review."
        else:
            sup_good = True
            
    if not sup_good:
        print("Supervisor failed")
        exit(0)
        
    # CONSUME boss_feedback on every exit path. It is written ONLY by boss_node
    # (set on revise, cleared on approve), so without this it survives every
    # supervisor round until the boss next APPROVES -- and the boss only runs
    # when a draft final answer is proposed. A rejection therefore used to be
    # re-rendered into the prompt indefinitely, still headed "Your previous draft
    # final answer has been reviewed and rejected by the boss", long after it had
    # been acted on. Same for the `invoke.py continue` operator directive, which
    # arrives through this channel and whose one-time instructions ("ingest the
    # finished results") would otherwise repeat for the rest of the run.
    # The feedback's EFFECT is carried forward by the plan it produced and by
    # past_steps; the text itself is needed once, on the round that reads it.
    if isinstance(agent_response.action, Response):
        return {
            "draft_response": agent_response.action.response,
            "boss_feedback": "",
            "next": "Boss_Agent",
            }
    # elif isinstance(output.action, Response):
    #     return {"response": "Plan is not finished! Do not use response!", "next": "Supervisor"}
    elif isinstance(agent_response.action, (Advance, Repeat)):
        # route_action decides the delta; the logging below is what makes a Repeat
        # round as auditable as any other -- it is the record that later says
        # whether the supervisor actually uses the action.
        delta = route_action(agent_response.action.kind, plan)
        headline = (
            "Advancing: first step finished, continue with the rest of the plan."
            if isinstance(agent_response.action, Advance)
            else "Repeating the current step; the plan is unchanged."
        )
        plan_str = "\n".join(
            f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}"
            for i, step in enumerate(delta["plan"])
        )
        print(headline)
        print(plan_str)
        write_history(headline + "\n" + plan_str + "\n")
        return delta
    else:
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}" for i, step in enumerate(agent_response.action.steps))
        print(plan_str)
        write_history(plan_str + "\n")
        return {
            "plan": agent_response.action.steps,
            "boss_feedback": "",
            "next": agent_response.action.steps[0].agent,
            }
    
    

def _unique_canvas_key(base: str) -> str:
    """`base`, or `base_2`, `base_3`, ... -- the first not already on CANVAS."""
    key, n = base, 2
    while key in CANVAS.canvas:
        key, n = f"{base}_{n}", n + 1
    return key


def _stash_on_canvas(base_key: str, value) -> str:
    """Store evicted raw step text ON CANVAS (was a jsonl file) under a unique key,
    and return the key.

    On CANVAS deliberately, so the agents can read the archive back via
    read_my_canvas instead of it living in an on-disk jsonl that drifts out of
    sync with the run.

    MUST persist through CANVAS._save_canvas(). This used to be a bare
    ``CANVAS.canvas[key] = value`` -- skipping the disk write was safe only
    because the canvas was a checkpointed channel and the value reached durable
    storage via the worker node's returned "canvas". The canvas is no longer
    checkpointed (see full_state_snapshot), so a bare dict assignment would keep
    the archived step text in memory only and lose it on the next restart --
    silently, since nothing reads it until an agent asks for that key.

    Assigning + saving directly rather than calling CANVAS.write() keeps the
    existing behaviour of overwriting without the "key already exists" guard
    (_unique_canvas_key has already made the key unique) and without the
    multi-MB pretty-print dump that write() also triggers.
    """
    key = _unique_canvas_key(base_key)
    CANVAS.canvas[key] = value
    CANVAS._save_canvas()
    return key


def _summarize_evicted(summarizer, objective, evicted_text):
    """The HARD-path summary agent: only reached when the agent ignored the SOFT
    nudge and never wrote a report covering this work. Returns (summary, key), where
    `key` is the CANVAS key the agent itself names for this batch of compacted work
    (empty string if it failed to, so the caller falls back to a deterministic key)."""
    prompt = f"""You are compacting the working memory of a long-running research agent.

Here is the overall objective:
{objective}

Below are completed steps that are about to be dropped from the agent's live context.
No report covers them; your summary plus the raw steps will be stored on CANVAS under
a key you choose, and the agent can read them back with read_my_canvas(key=...).

{evicted_text}

Return ONLY a JSON object with exactly two string fields:
  "key": a short, descriptive snake_case CANVAS key naming this batch of compacted
         work (e.g. "compacted_ru_ir_oxide_screening"). No spaces.
  "summary": a compact summary (at most ~250 words) preserving what is NOT recoverable
         from the experiment log -- the strategy and how it evolved, hypotheses formed
         and whether confirmed or rejected, literature findings, what was tried and
         FAILED, and any CANVAS keys or reference IDs worth revisiting. Do NOT restate
         per-candidate numbers (G(O), G(OH), overpotentials) -- those live in EXPLOG,
         retrievable with query_explog.
Output only the JSON object, nothing else."""
    response = summarizer.invoke(prompt)
    content = response.content
    if isinstance(content, list):  # Anthropic can return content blocks, not a str
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    content = str(content).strip()
    # Tolerate a fenced ```json block or leading/trailing prose around the object.
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        obj = json.loads(content[start:end])
        key = str(obj.get("key", "")).strip().replace(" ", "_")
        summary = str(obj.get("summary", "")).strip()
        if summary:
            return summary, key
    except (ValueError, TypeError):
        pass
    # Parse failed -- keep the raw text as the summary, let the caller pick the key.
    return content, ""


def _compact_ledger(ledger, *, report_written, report_name,
                    report_id, report_gist, summarizer, objective):
    """Enforce  past_steps == [digests] + [last K verbatim steps].

    Two ways in, and the difference is the whole point of the design:

      * A report was just written -> FREE. The report is already on CANVAS with a
        result id, so the evicted steps collapse to a one-line pointer at it. No
        LLM call, no latency, nothing new that can fail.
      * We are at/over HARD and no report covers this work -> pay for an LLM
        summary. This is the backstop, and it only runs on the path where the
        agent already declined to report.
    """
    plan = plan_compaction(ledger, k=K_VERBATIM, hard=HARD_TOKENS,
                           report_written=report_written)
    if not plan.should:
        return ledger

    evicted = ledger[plan.evict_lo:plan.evict_hi]
    # The first evicted step is the first verbatim step (eviction starts right after
    # the leading digest block), so its absolute index is first_verbatim_index.
    lo_abs = first_verbatim_index(ledger)
    hi_abs = lo_abs + plan.n_evicted - 1

    # The raw evicted steps, kept so the agents can read them back later.
    raw_steps = [
        {"step": lo_abs + i, "agent": s.agent, "timeStamp": str(s.timeStamp),
         "timeSpent": s.timeSpent, "text": s.step}
        for i, s in enumerate(evicted)
    ]

    if plan.reason == "report":
        # The report path already has a CANVAS key -- report_name, created by the
        # worker's write_report call. Park the raw steps beside it under a derived
        # key; recovery of the *substance* is the report itself.
        raw_steps_key = _stash_on_canvas(f"{report_name}__steps", raw_steps)
        digest_text = build_report_digest(
            report_name=report_name, report_id=report_id,
            step_lo=lo_abs, step_hi=hi_abs, gist=report_gist, raw_steps_key=raw_steps_key,
        )
        print(f"Compacted steps {lo_abs}-{hi_abs} into a pointer at report '{report_name}' "
              f"(raw steps on CANVAS: '{raw_steps_key}').")
    else:
        # HARD path: the summary agent writes the summary AND names the CANVAS key.
        summary, llm_key = _summarize_evicted(
            summarizer, objective,
            render_past_steps(evicted, with_timing=False, start_index=lo_abs),
        )
        base_key = llm_key or f"compacted_steps_{lo_abs}_{hi_abs}"
        raw_steps_key = _stash_on_canvas(base_key, {"summary": summary, "raw_steps": raw_steps})
        digest_text = build_summary_digest(
            summary=summary, step_lo=lo_abs, step_hi=hi_abs, raw_steps_key=raw_steps_key,
        )
        print(f"HARD watermark hit: mechanically summarized steps {lo_abs}-{hi_abs} "
              f"(CANVAS: '{raw_steps_key}'). A report would have done this better, and free.")

    # Not routed through print_stream: that expects a graph stream event, and its
    # `"messages" not in s` guard is a substring test on a raw string.
    print(digest_text)
    write_history(digest_text + "\n")

    # Keep the digest block itself bounded -- ~40 digests over a 500-step campaign.
    digest_texts = fold_digests([s.step for s in ledger[:plan.evict_lo]] + [digest_text])
    digests = [
        myPastStep(step=t, agent="system", timeStamp=timedelta(seconds=0), timeSpent="0:00:00")
        for t in digest_texts
    ]
    return digests + list(ledger[plan.evict_hi:])


def worker_agent_node(state, config, agent=None, name=None, summarizer=None):
    # Parent RunnableConfig injected by LangGraph -> inner agent runs as a
    # checkpointed subgraph; every tool call gets a resumable checkpoint.
    inner_cfg = {**config, "recursion_limit": 1000}
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
    write_history(f"Agent {name} is processing!!!!!\n")

    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan))
    task = plan[0]
    # Bridge the per-task queue-floor flag to the global wait_for_update tool
    # (Gate 2, Step 6). Re-derived from plan[0] every turn, so it survives resume;
    # defaults True when the supervisor omits it.
    # False is honored only inside the final FLOOR_DISARM_WINDOW_DAYS of the
    # study budget; requested earlier, it is coerced back to True and both
    # agents are told (worker note below + supervisor notice next planning turn).
    # Running every worker turn, this also re-arms False steps already sitting
    # in a resumed checkpoint -- no checkpoint surgery needed.
    _floor = getattr(task, "enforce_queue_floor", True)
    _remaining = var.STUDY_BUDGET_SECONDS - (time.time() - var.startTime)
    _floor_coerced = (not _floor) and _remaining >= var.FLOOR_DISARM_WINDOW_SECONDS
    if _floor_coerced:
        _floor = True
        var.floor_coerce_notice = True  # one-shot; supervisor told next turn
    var.enforce_queue_floor = _floor

    # This step's absolute number. Derived from the ledger (past_steps.steps_completed
    # reads the digest block's step range back out), NOT len(past_steps) -- compaction
    # shrinks the ledger, which used to make the step_<N>_DAG.html filenames march
    # backwards and overwrite each other. No state channel is added for this.
    step_no = steps_completed(state["past_steps"]) + 1

#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = render_history(state)
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Here is the overall objective:
{state["inputs"]}

Now, you are tasked with: {task}. Focus on this task. Your standing duties are always in scope regardless of the task: recording dispositions for finished results, submitting pipeline-continuation jobs (e.g. a finalized bulk -> surface relaxation, a finalized surface -> O adsorption on its sites, a competitive O site -> OH adsorption), and complying with instructions returned by tools. Do not otherwise go beyond the task -- no new candidates, no reports, no study-scope changes unless the task asks for them. Please note down important information on CANVAS together with their reference id before you end.
"""
    if _floor_coerced:
        task_formatted += (
            f"\nNOTE: this step requested enforce_queue_floor=False, but the "
            f"study still has {_remaining / 86400:.1f} days remaining -- the "
            f"disarm is honored only within the final "
            f"{var.FLOOR_DISARM_WINDOW_DAYS} days of the "
            f"{var.STUDY_BUDGET_DAYS}-day budget. The queue floor remains ARMED "
            f"for this step.\n"
        )

    old_task_formatted = task_formatted
    # On a mid-round resume, this node re-executes from the top while the
    # inner agent continues after its last completed tool call — so the
    # pre-crash result ids (restored by invoke.py into
    # var.resume_curr_round_result_ids) must survive instead of being reset,
    # or check_required_tool_use() will wrongly fail the round. One-shot.
    _resume_ids = getattr(var, "resume_curr_round_result_ids", None)
    if _resume_ids is not None:
        CANVAS.curr_round_result_ids = list(_resume_ids)
        var.resume_curr_round_result_ids = None
    else:
        CANVAS.rest_curr_round_result_ids()
    workerGood = False
    workerGood_patient = 2
    while not workerGood and workerGood_patient > 0:
        print(task_formatted)
        print(f"Agent {name} is processing!!!!!")
        write_history(task_formatted + "\n" + f"Agent {name} is processing!!!!!\n")
        workerGood_patient -= 1
        # stream_mode pinned: see note in boss_node — without it the propagated
        # parent config flips chunks to "values" mode.
        for agent_response in agent.stream(
            {"messages": [("user", task_formatted)]},  inner_cfg, stream_mode="updates", durability="sync"
        ):
            # set agent_response to be the value of the first key of the dictionary
            agent_response = next(iter(agent_response.values()))
            print_stream(agent_response, DAG=step_no)
        
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
            if task.required_tools == "":
                tool_use_passed = True
                tool_use_msg = "No required tools for this step."
            else:
                tool_use_passed, tool_use_msg = CANVAS.check_required_tool_use(task.required_tools)
            print(tool_use_msg)
            # Force true, not as necessary in OER case
            if not tool_use_passed:
                tool_use_passed = True
                tool_use_msg += "\nHowever, we will not enforce the tool use in this case, and let the supervisor decide whether the tool use is necessary or not based on the worker's execution result and notes on CANVAS."
            if tool_use_passed:
                # LLM sanity check
                # if LLM_check_passed:
                #     workerGood = True
                DAG_title = f"step_{step_no}_DAG"
                CANVAS.gen_DAG(
                    filename=f"{var.my_WORKING_DIRECTORY}/{DAG_title}.html",
                    title=DAG_title,
                )
                workerGood = True
            else:
                task_formatted = old_task_formatted
                task_formatted += f"\n\nWARNING: You didn't use the following required tools: {tool_use_msg}. Retry again!"
    
    if not workerGood:
        print(f"Worker Agent {name} failed")
        exit(0)
        
    timeElapsed_tmp = time.time() - var.startTime
    timeElapsed = timedelta(seconds=timeElapsed_tmp)

    if len(state["past_steps"]) > 0:
        prevTimeStamp = state["past_steps"][-1].timeStamp
    else:
        prevTimeStamp = timedelta(seconds=0)

    # Cap this step's text on the way in. The median step is ~1,500 chars, but the
    # worst one ever observed was 33,209 (~8,300 tokens) -- a single step able to eat
    # a third of the history budget. The full text goes to CANVAS under a
    # deterministic per-step key, so nothing is lost and the agent can read it back;
    # the in-ledger version points there. Deterministic key (no uniquify) -- a
    # re-run just overwrites it with identical content, keeping CANVAS in sync.
    _fulltext_key = f"step_{step_no}_fulltext"
    step_text, was_truncated = truncate_step_text(
        structured_response.summary, STEP_CHAR_CAP,
        archive_ref=f"read_my_canvas(key='{_fulltext_key}')",
    )
    if was_truncated:
        # _save_canvas(): the canvas is no longer a checkpointed channel, so a
        # bare dict assignment would keep this full text in memory only and lose
        # it on restart -- and it is exactly what the truncation notice tells the
        # agent to read back via read_my_canvas(key='step_<N>_fulltext').
        CANVAS.canvas[_fulltext_key] = structured_response.summary
        CANVAS._save_canvas()

    # Build a NEW list -- never mutate state["past_steps"] in place. LangGraph hands
    # us the channel's own value; mutating it aliases the checkpointed list and makes
    # time travel unsound.
    ledger = list(state["past_steps"]) + [
        myPastStep(
            step=step_text,
            agent=name,
            timeStamp=timeElapsed,
            timeSpent=str(timeElapsed - prevTimeStamp).split(".")[0],
        )
    ]

    print_stream(structured_response.summary)

    # Consume-and-clear the report flag HERE -- AFTER the agent has run, so it
    # reflects a write_report THIS turn just made (var.reportName is set by the tool
    # mid-stream), not last turn's leftover. Cleared UNCONDITIONALLY, whether or not
    # compaction actually fires below: clearing only inside the compaction branch
    # would let the flag LATCH when a compaction is skipped (e.g. under the MIN_EVICT
    # floor), so the next turn would wrongly think a fresh report had landed.
    report_written = bool(getattr(var, "reportName", ""))
    report_name = var.reportName if report_written else ""
    report_id = getattr(var, "reportId", "") if report_written else ""
    report_gist = getattr(var, "reportGist", "") if report_written else ""
    var.reportName = ""
    var.reportId = ""
    var.reportGist = ""

    # --- Compaction -----------------------------------------------------------
    # Wrapped whole: if the summariser 529s or the archive write fails, log it and
    # carry on with an UNCOMPACTED ledger. The HARD watermark will simply try again
    # next turn. A transient API error must never kill a 30-day campaign.
    try:
        ledger = _compact_ledger(
            ledger,
            report_written=report_written,
            report_name=report_name,
            report_id=report_id,
            report_gist=report_gist,
            summarizer=summarizer,
            objective=state["inputs"],
        )
    except Exception:
        print("Compaction failed; continuing with the uncompacted history.")
        print(traceback.format_exc())
        write_history("Compaction failed; continuing with the uncompacted history.\n")

    return {
        "past_steps": ledger,
        "time": timeElapsed_tmp,
    }

def whos_next(state):
    return state["next"]
    
def create_planning_graph(config: dict) -> StateGraph:
    # create a file named status.txt in the working directory
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    viz.set_working_directory(WORKING_DIRECTORY)
    with open(f"{WORKING_DIRECTORY}/status.txt", "w") as f:
        f.write("run")
    
    # Define the model
    # llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # workerllm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0, tool_choice="auto")
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    workerllm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0, tool_choice="auto")
    # Built ONCE and injected into worker_agent_node, rather than constructed inline
    # per compaction as it used to be. It now runs on the HARD-watermark path (the
    # agent ignored the report nudge), so it needs a retry and a bounded output --
    # a transient 529 here must not take down a 30-day campaign.
    summarizer_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key=config['ANTHROPIC_API_KEY'],
        temperature=0.0,
        max_tokens=1500,
    ).with_retry(stop_after_attempt=3)
    # workerllm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # workerllm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"], model_kwargs={'parallel_tool_calls': False})
    # llm = ChatDeepSeek(model_name=config["DeepSeek_MDL"], api_key=config['DeepSeek_API_KEY'], api_base=config['DeepSeek_BASE_URL'], temperature=0.0)
    
    var.my_WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    if not eval(config["SAVE_DIALOGUE"]):
        var.my_SAVE_DIALOGUE = False
    
    
    
    # System Supervisor with tool bind and with_structured_output
    
    # supervisor_chain = supervisor_prompt | llm.bind_tools(supervisor_tools).with_structured_output(Act)
    # supervisor_agent = functools.partial(supervisor_chain_node, chain=supervisor_chain, name="Supervisor")
    
    
        
    
    # def supervisor_agent(state):
    #     print("Supervisor!!!!!!!!!")
    #     supervisor_chain = (
    #         prompt
    #         | llm.with_structured_output(routeResponse)
    #     )
    #     return supervisor_chain.invoke(state)
    
    ## Memory Saver
    memory = MemorySaver()

    PAST_STEPS = []
    myCANVAS = {}
    


    # The boss used to have NO tools, which was fine while the full step history sat
    # in its prompt. Now that older steps are compacted to digests, a tool-less boss
    # would be reviewing the final answer against pointers -- gutting exactly the gate
    # that is supposed to catch claims unsupported by data. Give it the two read paths
    # the digests point at. It runs once per review, so this is cheap.
    boss_tools = [read_my_canvas, query_explog]
    boss_agent = create_agent(
        model=llm, # <-- Same as supervisor
        tools=boss_tools,
        system_prompt=boss_prompt,
        response_format=ToolStrategy(BossReview),
        middleware=[StateSyncMiddleware(), DisableParallelToolCallsMiddleware(), handle_tool_errors]
    )
    boss_agent_node = functools.partial(boss_node, agent=boss_agent, name="Boss_Agent")
    
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
        query_explog,
        # Added alongside compaction: once older steps collapse to digests, the
        # supervisor can no longer see the reference IDs they mentioned. This is how
        # it finds one again (by content) instead of guessing.
        search_artifacts,
        ]
    
    supervisor_agent = create_agent(
        model=llm,
        tools=supervisor_tools, 
        system_prompt=supervisor_prompt,
        # Structured output via ToolStrategy (tool-calling fallback)
        response_format=ToolStrategy(Act, handle_errors=on_act_parse_error),  # Or ProviderStrategy for native models
        middleware=[StateSyncMiddleware(), DisableParallelToolCallsMiddleware(), handle_tool_errors]  # Middleware to handle tool errors and cap structured output retries
    )
    
    # supervisor_agent = create_react_agent(llm, tools=supervisor_tools,
    #                                prompt=supervisor_prompt, response_format=Act)   
    supervisor_node = functools.partial(supervisor_chain_node, agent=supervisor_agent, name="Supervisor_Agent")
    
    ### DFT Agent
    dft_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
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
        generate_eos_test,
        read_energy_from_output,
        get_convergence_suggestions,
        analyze_BEEF_result,
        extract_numeric_from_tool_output,
        math_expression_tool
        # get_ase_atoms_property,
        # inspect_ase_atoms,
        ]
    # dft_agent = create_react_agent(workerllm, tools=dft_tools,
    #                                prompt="You are a DFT expert")   
    dft_agent = create_agent(
        model=workerllm,
        tools=dft_tools,
        system_prompt=dft_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[StateSyncMiddleware(), DisableParallelToolCallsMiddleware(), handle_tool_errors]
    )
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent")

    
    oer_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        write_report,
        check_time,
        OER_data_analasis_v2,
        browse_df,
        arXiv_search,
        enter_candidate_in_log,
        submit_dft_job,
        get_terminations_ranking,
        list_adsorption_sites,
        read_explog,
        wait_for_update,
        query_explog,
        get_disposition_info,
        update_disposition_info,
        # math_expression_tool,
        # extract_numeric_from_tool_output,
        search_artifacts,
        ]
    # oer_agent = create_react_agent(workerllm, tools=oer_tools,
    #                                prompt=oer_agent_prompt)
    oer_agent = create_agent(
        model=workerllm,
        tools=oer_tools,
        system_prompt=oer_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[StateSyncMiddleware(), DisableParallelToolCallsMiddleware(), prevent_redundant_polling, handle_tool_errors]
    )
    oer_node = functools.partial(worker_agent_node, agent=oer_agent, name="OER_Agent", summarizer=summarizer_llm)

    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        submit_and_monitor_job,
        add_resource_suggestion
        ]

    # hpc_agent = create_react_agent(workerllm, tools=hpc_tools,
    #                                prompt=hpc_agent_prompt)
    hpc_agent = create_agent(
        model=workerllm,
        tools=hpc_tools,
        system_prompt=hpc_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[StateSyncMiddleware(), DisableParallelToolCallsMiddleware(), handle_tool_errors]
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
    # graph.add_node("DFT_Agent", dft_node)
    # graph.add_node("HPC_Agent", hpc_node)
    graph.add_node("OER_Agent", oer_node)
    graph.add_node("Boss_Agent", boss_agent_node)
    # graph.add_node("MD_Agent", md_node)
    # graph.add_node("CSS_Agent", css_node)

    graph.add_node("Supervisor", supervisor_node)
    
    for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
        graph.add_edge(member, "Supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    # above line will crete a dict like:
    # {"DFT_Agent": "DFT_Agent_node", "HPC_Agent": "HPC_Agent", "OER_Agent": "OER_Agent", ...}
    
    conditional_map["Boss_Agent"] = "Boss_Agent"
    conditional_map["FINISH"] = END
    conditional_map["Supervisor"] = "Supervisor" 
    graph.add_conditional_edges("Supervisor", whos_next, conditional_map)

    boss_conditional_map = {
        "FINISH": END,
        "Supervisor": "Supervisor",
    }
    graph.add_conditional_edges("Boss_Agent", whos_next, boss_conditional_map)
    graph.add_edge(START, "Supervisor") 
    # checkpointer = InMemorySaver()
    # return graph.compile(checkpointer=checkpointer)
    # return graph.compile()
    return graph