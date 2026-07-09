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
from src.disposition_messages import classify_wait_handback, format_supervisor_handback_directive
from src.forgotten_jobs import find_forgotten_jobs
from src.history_log import write_history
from src import var
from src.myCANVAS import CANVAS
from src.live_visualizer import LiveVisualizer
from gnome_dreams_oer_screening.explog.explog import EXPLOG

import json, hashlib, time
from collections import defaultdict, deque
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
    enforce_queue_floor: bool = Field(True, description="Whether the worker must keep the HPC queue stocked with ready work before it may wait for current jobs to finish. Keep True to ensure maximum HPC utilization (generally desirable): if the queue falls below the floor while HPC resources are free and the worker has no more ready work to submit, the worker returns to the supervisor to discuss expanding the study, rather than being allowed to pause and wait. Set False only when genuinely winding the study down -- when the remaining time is too short for newly-submitted jobs to finish -- so the worker may instead wait for and finalize the in-flight results.")

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

# the class supervisor will choose if the plan doesn't need to be changed
class NoChange(BaseModel):
    """The first step of the current plan will be removed, and the second step to be executed next without any change."""
    kind: Literal["no_change"] = "no_change"
    comment: str = Field(description="any comment from the supervisor if needed, otherwise just put 'No change to the plan, continue to execute the second step of the original plan.'")


class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, NoChange, Response] = Field(
        description="""Action to perform. If the team need to further use tools to get the answer, and if you need to add more steps or adjust the steps, use Plan.
        If the team can continue to execute the original plan without any change, use NoChange.
        Use Response when you believe the task is complete and want to submit a proposed final answer for boss review.""",
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

        # Case 2: model wrapped variant as {"NoChange": {...}}
        if isinstance(action, dict) and len(action) == 1:
            key = next(iter(action))
            if key in {"Plan", "NoChange", "Response"}:
                inner = dict(action[key])
                inner.setdefault(
                    "kind",
                    {"Plan": "plan", "NoChange": "no_change", "Response": "response"}[key],
                )
                action = inner

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
    past_steps: List[myStep]
    draft_response: str
    boss_feedback: str
    response: str
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
    canvas: NotRequired[Dict]
    artifacts: NotRequired[Dict]
    curr_round_result_ids: NotRequired[List[str]]
    explog_candidates: NotRequired[pd.DataFrame]
    explog_processes: NotRequired[pd.DataFrame]
    time: NotRequired[float]


def full_state_snapshot():
    """Snapshot the global CANVAS / EXPLOG / elapsed time as state-channel
    values. deepcopy/copy so later in-place mutation of the globals cannot
    alias into an already-taken checkpoint."""
    return {
        "canvas": copy.deepcopy(CANVAS.canvas),
        "artifacts": copy.deepcopy(CANVAS.result_registry),
        # backs CANVAS.check_required_tool_use after a mid-round resume
        "curr_round_result_ids": list(CANVAS.curr_round_result_ids),
        "explog_candidates": EXPLOG.relational_frame.candidates.df.copy(),
        "explog_processes": EXPLOG.relational_frame.processes.df.copy(),
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
        "Return exactly one of {Plan, NoChange, Response} as `action`, "
        "with the inner fields directly — do NOT wrap in {'NoChange': {...}}."
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
    viz.on_event(s, DAG=DAG)
    
    if DAG is not None:
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

    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step} [total time elapsed since project start: {str(step.timeStamp).split('.')[0]}, time spent on step {i+1}: {step.timeSpent}]" for i, step in enumerate(state["past_steps"]))
    bossMessage = f"""
    The overall goal is:
    {state['inputs']}

    This is what has been done so far:
    {old_tasks_string}
    
    The supervisor's draft final answer is:
    {state['draft_response']}
    
    Current time: {timeElapsed}.

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
            "canvas": CANVAS.canvas,
            "artifacts": CANVAS.result_registry,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
        }
    elif agent_response.decision == "revise":
        return {
            # Boss rejection stores concrete review feedback and hands control back to the supervisor.
            "boss_feedback": agent_response.feedback,
            "next": "Supervisor",
            "canvas": CANVAS.canvas,
            "artifacts": CANVAS.result_registry,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
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
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step} [total time elapsed since project start: {str(step.timeStamp).split('.')[0]}, time spent on step {i+1}: {step.timeSpent}]" for i, step in enumerate(state["past_steps"]))

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
        Only choose <NoChange> if you want the first step of the current plan to be removed, and the second step to be executed next without any change.
        Otherwise choose <Plan> and rewrite the plan with the steps you want, and the first step of the plan you just wrote will be executed.
        Or if you think the overall goal is finished, you can choose <Response> and write a draft final answer for the boss review.
        """

    # --- Queue-floor handback -------------------------------------------------
    # wait_for_update raises var.wait_handback when it refuses on a queue-floor
    # (Path A/B) or idle handback -- a situation only the supervisor can resolve.
    # Consume-and-clear it here (one-shot) and inject a path-specific directive so
    # the handback is ACTED on instead of reaching the supervisor only as ignorable
    # worker prose. The path is re-derived from live EXPLOG (self-correcting: if the
    # queue refilled since the worker's turn, classify returns None and nothing is
    # injected). Uses only the pure gate -- never the real wait tool, whose loop
    # sleeps regardless of patience.
    if getattr(var, "wait_handback", False):
        var.wait_handback = False
        _hb_forgotten = find_forgotten_jobs(EXPLOG, var.GO_DEV_OH_THRESHOLD)
        _hb_status = EXPLOG.relational_frame.processes.df["status"].tolist()
        _hb_path = classify_wait_handback(
            candidates_need_disposition=EXPLOG.candidates_needing_disposition(),
            pending_count=EXPLOG.job_handler.count_pending(),
            has_running=("running" in _hb_status),
            enforce_queue_floor=getattr(var, "enforce_queue_floor", True),
            queue_min_pending=var.QUEUE_MIN_PENDING,
            forgotten_jobs=_hb_forgotten,
        )
        if _hb_path is not None:
            supervisorMessage = (
                supervisorMessage
                + "\n\n"
                + format_supervisor_handback_directive(_hb_path, _hb_forgotten)
            )

    old_supervisorMessage = supervisorMessage
    sup_good = False
    sup_good_patient = 3
    while not sup_good and sup_good_patient > 0:
        sup_good_patient -= 1
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
        
        if isinstance(agent_response.action, NoChange) and len(state["plan"]) <= 1:
            sup_good = False
            supervisorMessage = old_supervisorMessage + f"\n\nWARNING: there is less than 2 steps left in the current plan, and there's no 'second' step to execute. You cannot choose 'NoChange' as the action. Please first carefully review what has been done, then either 'Response' with a message, or 'Plan' more steps! If you choose 'Plan', the first step of the new plan you just wrote will be executed next."
        elif isinstance(agent_response.action, Plan) and len(agent_response.action.steps) == 0:
            sup_good = False
            supervisorMessage = old_supervisorMessage + f"\n\nWARNING: you chose to rewrite the plan, but the new plan is empty. Please provide a non-empty plan with at least one step, or if you think the overall goal is finished, you can choose 'Response' and write a draft final answer for the boss review."
        else:
            sup_good = True
            
    if not sup_good:
        print("Supervisor failed")
        exit(0)
        
    if isinstance(agent_response.action, Response):
        return {
            "draft_response": agent_response.action.response,
            "next": "Boss_Agent", 
            "canvas":CANVAS.canvas, 
            "artifacts": CANVAS.result_registry,
            "explog_candidates": EXPLOG.relational_frame.candidates.df, 
            "explog_processes": EXPLOG.relational_frame.processes.df,
            }
    # elif isinstance(output.action, Response):
    #     return {"response": "Plan is not finished! Do not use response!", "next": "Supervisor"}
    elif isinstance(agent_response.action, NoChange):
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}" for i, step in enumerate(plan[1:]))
        print("No change to the plan, continue to execute the original plan.")
        print(plan_str)
        write_history("No change to the plan, continue to execute the original plan.\n" + plan_str + "\n")
        return {
            "plan": plan[1:],
            "next": plan[1].agent,
            "canvas":CANVAS.canvas,
            "artifacts": CANVAS.result_registry,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
            }
    else:
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}" for i, step in enumerate(agent_response.action.steps))
        print(plan_str)
        write_history(plan_str + "\n")
        return {
            "plan": agent_response.action.steps, 
            "next": agent_response.action.steps[0].agent, 
            "canvas":CANVAS.canvas,
            "artifacts": CANVAS.result_registry,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
            }
    
    

def worker_agent_node(state, config, agent=None, name=None):
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
    var.enforce_queue_floor = getattr(task, "enforce_queue_floor", True)
#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step} [total time elapsed since project start: {str(step.timeStamp).split('.')[0]}, time spent on step {i+1}: {step.timeSpent}]" for i, step in enumerate(state["past_steps"]))
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Here is the overall objective:
{state["inputs"]}

Now, you are tasked with: {task}. Please only do this task! Do not do anything else! Please note down important information on CANVAS together with their reference id before you end.
"""
    
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
            print_stream(agent_response, DAG=len(state["past_steps"])+1)
        
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
                DAG_title = f"step_{len(state['past_steps'])+1}_DAG"
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
    # state["past_steps"].append((task, agent_response["messages"][-1].content))
    if len(state["past_steps"]) > 0:
        prevTimeStamp = state["past_steps"][-1].timeStamp
    else:
        prevTimeStamp = timedelta(seconds=0)
    state["past_steps"].append(
        myPastStep(
            step=structured_response.summary, 
            agent=name, 
            timeStamp=timeElapsed,
            timeSpent=str(timeElapsed-prevTimeStamp).split(".")[0]
            )
        )
    
    print_stream(structured_response.summary)
    
    if var.reportName:
        old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(state["past_steps"]))
        task_formatted = f"""
Here is the overall objective:
{state["inputs"]}

During the end of the last step, you just: 
{structured_response.summary}

and generated the following report:
{CANVAS.canvas[var.reportName]}

Now, please summarize previous steps that has been done so far:
{old_tasks_string}

Please only do this task! Do not do anything else! The summarized old steps will be noted down by the system, and you don't have to worry about that.
"""

        print(task_formatted)
        print(f"Summarize Agent is processing!!!!!")
        write_history(task_formatted + "\n" + f"Summarize Agent is processing!!!!!\n")

        config = var.OTHER_GLOBAL_VARIABLES
        workerllm = ChatAnthropic(model="claude-sonnet-4-5-20250929", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
        response = workerllm.invoke(task_formatted)
        
        timeElapsed_tmp = time.time() - var.startTime
        timeElapsed = timedelta(seconds=timeElapsed_tmp)
        
        state["past_steps"] = []
        state["past_steps"].append(
            myPastStep(
                step= f"Summary of all previous steps: {response.content}\nDetailed previous steps can be found in CANVAS with key '{var.reportName}_compressed_steps'", 
                agent=name, 
                timeStamp=timeElapsed,
                timeSpent=str(timeElapsed).split(".")[0]
                )
            )
        CANVAS.canvas[f"{var.reportName}_compressed_steps"] = old_tasks_string
        
        print_stream(f"Summary of all previous steps: {response.content}")

        var.reportName = ""
        
    return {
        "past_steps": state["past_steps"], 
        "canvas":CANVAS.canvas,
         "artifacts": CANVAS.result_registry,
        "explog_candidates": EXPLOG.relational_frame.candidates.df,
        "explog_processes": EXPLOG.relational_frame.processes.df,
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
    


    boss_tools = []
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
    oer_node = functools.partial(worker_agent_node, agent=oer_agent, name="OER_Agent")

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