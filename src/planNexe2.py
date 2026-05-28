from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ToolCallLimitMiddleware, AgentMiddleware, ModelRequest, wrap_tool_call
from langchain.agents import create_agent

from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple, Union
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

members = ["DFT_Agent", "HPC_Agent"]
instructions = [dft_agent_prompt, hpc_agent_prompt]
OPTIONS = members

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     next: str

class myStep(BaseModel):
    """Step in the plan."""

    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )
    required_tools:Literal[
                    # "inspect_my_canvas",
                    # "write_my_canvas",
                    # "read_my_canvas",
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
                    ""
                ] = Field(description=f"The one final tool your worker agent must use to obtain the desired output of a certain step. Please read the CANVAS with key Worker_available_tools to see more details about each tools.")
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
    """End everything and response to the user."""
    kind: Literal["response"] = "response"
    response: str

# the class supervisor will choose if the plan doesn't need to be changed
class NoChange(BaseModel):
    """No change to the plan, just continue to execute the original plan."""
    kind: Literal["no_change"] = "no_change"
    comment: str = Field(description="any comment from the supervisor if needed, otherwise just put 'No change to the plan, continue to execute the original plan.'")


class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, NoChange, Response] = Field(
        description="""Action to perform. If the team need to further use tools to get the answer, and if you need to add more steps or adjust the steps, use Plan.
        If the team can continue to execute the original plan without any change, use NoChange.
        If you want to end the conversation, use Response.""",
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
    
class PlanExecute(TypedDict):
    inputs: str
    plan: List[myStep]
    past_steps: List[myStep]
    response: str
    canvas: dict
    artifacts: dict
    next: str
    
    

class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
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
        "Return exactly one of {Plan, NoChange, Response} as `action`, "
        "with the inner fields directly — do NOT wrap in {'NoChange': {...}}."
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
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan[1:]))
    # task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(state["past_steps"]))
    
    
    # TODO: notify the supervisor if the worker returned successed=false
    supervisorMessage = ""
    if len(plan) == 0:
        supervisorMessage =  f"""
The overall goal is: {state['inputs']}.

Nothing has been done yet and there is no plan yet.

Please inspect and extract related information from CANVAS, then build the initial plan.

Important: you are a coordinator, not a domain expert. Before adding any
execution steps to the plan (input file generation, job submission,
calculations), consult with the DFT agent: what are the major calculations
this objective requires? What known caveats apply to this system class?

Intermediate reports containing one numerical claim must be generated immediately after that numerical claim was made (result claim or determination of some settings or etc)
so the judge evaluate the validity of the result.
"""
    else:
        supervisorMessage =  f"""
The overall goal is: {state['inputs']}. 

previous task of {plan[0].agent} was:
{plan[0].step}.

this is what has been done:
{old_tasks_string}

the current plan is:
{plan_str}

Please inspect and extract related information from CANVAS, then only update the plan accordingly if needed.
If you need to discuss with the worker agent insert extra step(s) at the beginning of the plan, and assign those step(s) to the worker agent you want to talk to.
Intermediate reports containing one numerical claim must be generated immediately after that numerical claim was made (result claim or determination of some settings or etc).
        """
    old_supervisorMessage = supervisorMessage
    sup_good = False
    sup_good_patient = 3
    while not sup_good and sup_good_patient > 0:
        sup_good_patient -= 1
        for agent_response in agent.stream(
            {"messages": [("user", supervisorMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
        ):
            # set agent_response to be the value of the first key of the dictionary
            agent_response = next(iter(agent_response.values()))
            print_stream(agent_response)

        # output = agent.invoke(
        #     {"messages": [("user", supervisorMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
        #     )
        # CANVAS.snap_save()
        agent_response = agent_response['structured_response']
        sup_good = True
        if isinstance(agent_response.action, Plan):
            for step in agent_response.action.steps:
                ToolList = [
                    "inspect_my_canvas",
                    "write_my_canvas",
                    "read_my_canvas",
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
                # step.required_tools is no longer a list of tools
                # wrongTools = set(step.required_tools) - set(ToolList)
                if step.required_tools not in ToolList:
                    print(f"wrongTools: {step.required_tools}")
                    supervisorMessage = old_supervisorMessage + f"\n\nWARNING: In step '{step.step}', you required the following tool that are not in the tool list: {step.required_tools}. Please check the CANVAS and try again!"
                    sup_good = False
                    break
                # A report step MUST carry a deterministic, non-empty list of
                # required_quantities — the exact named quantities the report
                # must certify. Catch the case where the supervisor scheduled
                # a report step but forgot to specify them.
                if step.required_tools == "generate_structured_report":
                    _rq = list(getattr(step, "required_quantities", []) or [])
                    if len(_rq) == 0:
                        print(
                            f"missing required_quantities for report step: "
                            f"{step.step}"
                        )
                        supervisorMessage = old_supervisorMessage + (
                            f"\n\nWARNING: Step '{step.step}' is a report step "
                            f"(required_tools='generate_structured_report') "
                            f"but you did not specify `required_quantities`. "
                            f"Every report step MUST have a non-empty "
                            f"`required_quantities` list naming the EXACT "
                            f"quantities the report must certify."
                            f"Add the required_quantities for this step and try again!"
                        )
                        sup_good = False
                        break
                    
        else:
            sup_good = True
    
    if not sup_good:
        print("Supervisor failed")
        exit(0)
        
        
    if isinstance(agent_response.action, Response):
        return {"response": agent_response.action.response, "next": "FINISH", "canvas":CANVAS.canvas}
    # elif isinstance(output.action, Response):
    #     return {"response": "Plan is not finished! Do not use response!", "next": "Supervisor"}
    elif isinstance(agent_response.action, NoChange):
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}, required_quantities: {step.required_quantities}" for i, step in enumerate(plan[1:]))
        print("No change to the plan, continue to execute the original plan.")
        print(plan_str)
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write("No change to the plan, continue to execute the original plan.\n")
                f.write(plan_str)
                f.write("\n")
        return {
            "plan": plan[1:],
            "next": plan[1].agent,
            "canvas":CANVAS.canvas
            }
    else:
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}, required_quantities: {step.required_quantities}" for i, step in enumerate(agent_response.action.steps))
        print(plan_str)
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(plan_str)
                f.write("\n")
        return {"plan": agent_response.action.steps, "next": agent_response.action.steps[0].agent, "canvas":CANVAS.canvas}
    
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
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan))
    # print(plan_str)
    # if var.my_SAVE_DIALOGUE:
    #     with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
    #         f.write(plan_str)
    #         f.write("\n")
    task = plan[0]
#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}." for i, step in enumerate(state["past_steps"]))
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Here is the overall objective:
{state["inputs"]}

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
                    DAG_title = f"step_{len(state['past_steps'])+1}_DAG"
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
    config = var.OTHER_GLOBAL_VARIABLES
    if var.reportName:
        print("#######################")
        print("Judging")
        print("#######################")
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
        if verdict == "pass":
            reportReviewResult = (
                "\nA report was generated, and the verifier's review on the "
                "report: PASS."
            )
        elif verdict == "warning":
            reportReviewResult = (
                f"\nA report was generated, and the verifier's review on the "
                f"report: WARNING ({n_fails} fail(s), {n_warnings} "
                f"warning(s))."
                f"\nSummary: {summary_line}"
            )
            for issue in rawReport["issues"]:
                reportReviewResult += _format_verification_issue(issue)
                reportReviewResult += "\n-----------------------------"
        else:
            reportReviewResult = (
                f"\nA report was generated, but the verifier's review on the "
                f"report: FAIL ({n_fails} fail(s), {n_warnings} warning(s))."
                f"\nSummary: {summary_line}"
            )
            for issue in rawReport["issues"]:
                reportReviewResult += _format_verification_issue(issue)
                reportReviewResult += "\n-----------------------------"
            reportReviewResult += (
                "\nPlease fix the issue(s) and try again! For each issue "
                "above, pick whichever of the listed remediation options "
                "fits the situation. Note that the report should be truthful "
                "and accurate based on the information you have, and should "
                "not contain any fabricated information that is not supported "
                "by data!"
            )
        var.All_Report_Names.append(copy.deepcopy(var.reportName))
        var.reportName = ""
        
    
    # state["past_steps"].append((task, agent_response["messages"][-1].content))
    state["past_steps"].append(myStep(step=structured_response.summary + f"\n{reportReviewResult}", agent=name, required_tools=task.required_tools))
    
    print_stream(structured_response.summary)
    # CANVAS.snap_save()
    return {
        "past_steps": state["past_steps"], 
        "canvas":CANVAS.canvas,
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
    
    if not eval(config["SAVE_DIALOGUE"]):
        var.my_SAVE_DIALOGUE = False
    
    # CANVASCheckpoints = []
    # with open(f"{WORKING_DIRECTORY}/canvas_checkpoints.pickle", 'rb') as f:
    #     CANVASCheckpoints = pickle.load(f)
    
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
        supervisor_get_available_report_names,
        debug_artifact_chain
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
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors]
    )
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent")


    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        submit_and_monitor_job,
        add_resource_suggestion
        ]
    hpc_agent = create_agent(
        model=workerllm,
        tools=hpc_tools,
        system_prompt=hpc_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors]
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