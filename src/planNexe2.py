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
from pydantic import BaseModel, Field

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
                    ""
                ] = Field(f"The one final tool your worker agent must use to obtain the desired output of a certain step. Please read the CANVAS with key Worker_available_tools to see more details about each tools.")
    # what would be that final one tool must use to obtain the desired output of a certain step

class Plan(BaseModel):
    """Need to add/modify current plan, which is going to be followed by your worker agents in future"""

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

    response: str

# the class supervisor will choose if the plan doesn't need to be changed
class NoChange(BaseModel):
    """No change to the plan, just continue to execute the original plan."""
    comment: str = Field(description="any comment from the supervisor if needed, otherwise just put 'No change to the plan, continue to execute the original plan.'")


class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, NoChange, Response] = Field(
        description="""Action to perform. If the team need to further use tools to get the answer, and if you need to add more steps or adjust the steps, use Plan.
        If the team can continue to execute the original plan without any change, use NoChange.
        If you want to end the conversation, use Response."""
        # "DO NOT use response unless absolutly necessary."
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

Please inspect and extract related information from CANVAS, then only update the plan accordingly if needed.

A mid-project report before production run and a final report at the very end are needed. Feel free to ask worker agent to generate other intermediate report if you think it's necessary to let judge evaluate what has been done.
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
                    "",
                ]
                # step.required_tools is no longer a list of tools
                # wrongTools = set(step.required_tools) - set(ToolList)
                if step.required_tools not in ToolList:
                    print(f"wrongTools: {step.required_tools}")
                    supervisorMessage = old_supervisorMessage + f"\n\nWARNING: In step '{step.step}', you required the following tool that are not in the tool list: {step.required_tools}. Please check the CANVAS and try again!"
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
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}" for i, step in enumerate(plan[1:]))
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
        plan_str = "\n".join(f"{i+1}. {step.step}, agent={step.agent}, required_tools: {step.required_tools}" for i, step in enumerate(agent_response.action.steps))
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
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
        while status == "stop":
            print(f"Calculation pause, judge is waiting. cwd: {var.my_WORKING_DIRECTORY}")
            # wait for 5 second
            time.sleep(5)
            with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
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
            tool_use_passed, tool_use_msg = CANVAS.check_required_tool_use(task.required_tools)
            print(tool_use_msg)
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
        if rawReport["overall_verdict"] == "pass":
            reportReviewResult = f"\nA reprot was generated, and the judge's review on the report: PASS."
        elif rawReport["overall_verdict"] == "warning":
            reportReviewResult = f"\nA reprot was generated, and the judge's review on the report: WARNING."
            for issue in rawReport["issues"]:
                reportReviewResult += f"\nIssue level: {issue['level']}, location: {issue['location']}, verdict: {issue['verdict']}, message: {issue['message']}."
        else:
            reportReviewResult = f"\nA reprot was generated, but the judge's review on the report: FAIL."
            for issue in rawReport["issues"]:
                reportReviewResult += f"\nIssue level: {issue['level']}, location: {issue['location']}, verdict: {issue['verdict']}, message: {issue['message']}."
            reportReviewResult += "\nPlease fix the issue and try again! Note that the report should be truthful and accurate based on the information you have, and should not contain any fabricated information that is not supported by data!"
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
        response_format=ToolStrategy(Act),  # Or ProviderStrategy for native models
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
        generate_eos_test,
        read_energy_from_output,
        get_convergence_suggestions,
        analyze_BEEF_result,
        extract_numeric_from_tool_output,
        extract_text_from_tool_output,
        math_expression_tool,
        generate_structured_report,
        list_referenceable_inputs
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


