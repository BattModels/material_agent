from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ToolCallLimitMiddleware, AgentMiddleware, ModelRequest, wrap_tool_call
from langchain.agents import create_agent

from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple, Union
import functools
import pandas as pd
import os
import threading

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
from src.prompt import dft_agent_prompt,hpc_agent_prompt,supervisor_prompt, oer_agent_prompt, boss_prompt
from src import var
from src.myCANVAS import CANVAS
from gnome_dreams_oer_screening.explog.explog import EXPLOG

import traceback

members = ["OER_Agent"]

class myStep(BaseModel):
    """Step in the plan."""

    step: str = Field(description="Step to perform.")
    agent: str = Field(
        description=f"Agent to perform the step. Should be one of {members}."
    )

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[myStep] = Field(
        description=f"""
        Steps to follow in future. Each step is a tuple of (step, agent). agent can only be chosen from {members}.
        """
        # description="different steps to follow, should be in sorted order"
        # description="""different steps to follow (first element of the Tuple), and the agent in charge for each step (second element of the Tuple),
        # should be in sorted order by the order of execution"""
    )
    
# NOTE:
class Response(BaseModel):
    """Supervisor's proposed final answer for boss review."""

    response: str



class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, Response] = Field(
        description="Use Plan if more work is needed. Use Response when you believe the task is complete and want to submit a proposed final answer for boss review."
    )


# NOTE <<<-----
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

class PlanExecute(TypedDict):
    inputs: str
    plan: List[myStep]
    past_steps: List[myStep]
    draft_response: str # NOTE: separate the supervisor's proposed final answer from the boss-approved final answer.
    boss_feedback: str # NOTE:  store boss review feedback when the draft answer is rejected.
    response: str
    canvas: Dict
    explog_candidates: pd.DataFrame
    explog_processes: pd.DataFrame
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
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)}), the traceback is: {traceback.format_exc()}",
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



### ------------------------------------------------------------------- NOTE
def boss_node(state, agent, name):
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
    print(f"{name} is processing!!!!! Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"{name} is processing!!!!! Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.\n")
    # can't print state anymore because it now contains canvas and explog, and printing them will cause too much output
    # print(state)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(str(state))
            f.write("\n")
            
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(state["past_steps"]))
    prior_boss_feedback = state["boss_feedback"].strip()
    if prior_boss_feedback == "":
        prior_boss_feedback = "None"

    bossMessage = f"""
    Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.

    The overall goal is:
    {state['inputs']}

    The supervisor's draft final answer is:
    {state['draft_response']}

    This is what has been done so far:
    {old_tasks_string}

    Previous boss feedback:
    {prior_boss_feedback}

    Please review the supervisor's draft final answer and decide whether to approve it or send it back for revision.
    """

    for agent_response in agent.stream(
        {"messages": [("user", bossMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
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
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
        }
    elif agent_response.decision == "revise":
        return {
            # Boss rejection stores concrete review feedback and hands control back to the supervisor.
            "boss_feedback": agent_response.feedback,
            "next": "Supervisor",
            "canvas": CANVAS.canvas,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
        }
    else:
        # NOTE NOTE NOTE: fail loudly if the boss returns an unexpected decision value, so invalid control flow is visible during development.
        raise ValueError(f"Unexpected boss decision: {agent_response.decision}")
### ------------------------------------------------------------------- 
            
def supervisor_chain_node(state, agent, name):
    
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
    
    print(f"supervisor is processing!!!!! Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"supervisor is processing!!!!! Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.\n")

    # can't print state anymore because it now contains canvas and explog, and printing them will cause too much output
    # print(state)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(str(state))
            f.write("\n")
            
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(plan))
    # task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(state["past_steps"]))

    current_boss_feedback = state["boss_feedback"].strip() # Remove leading/trailing whitespaces
    if current_boss_feedback == "":
        current_boss_feedback = "None"
    
    # NOTE: include boss review feedback in the supervisor's next turn context.
    supervisorMessage =  f"""
    Current time: {currentTime}, time elapsed since the start of the project: {timeElapsed}.

    Your available agents are: {members}.

    The overall goal is: {state['inputs']}. 

    the current plan is:
    {plan_str}

    this is what has been done:
    {old_tasks_string}

    Previous boss feedback:
    {current_boss_feedback}

    Please update the plan accordingly.
    """
        
    for agent_response in agent.stream(
        {"messages": [("user", supervisorMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    ):
        # set agent_response to be the value of the first key of the dictionary
        agent_response = next(iter(agent_response.values()))
        print_stream(agent_response)

    # output = agent.invoke(
    #     {"messages": [("user", supervisorMessage)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    #     )
    
    agent_response = agent_response['structured_response']
    if isinstance(agent_response.action, Response):
        return {
            # NOTE: supervisor completion - create a draft for boss review, not terminate immediately.
            "draft_response": agent_response.action.response,
            # NOTE empty string for boss feedback - when submitting a fresh draft for review.
            "boss_feedback": "",
            "next": "Boss_Agent", 
            "canvas":CANVAS.canvas, 
            "explog_candidates": EXPLOG.relational_frame.candidates.df, 
            "explog_processes": EXPLOG.relational_frame.processes.df,
            }
    # elif isinstance(output.action, Response):
    #     return {"response": "Plan is not finished! Do not use response!", "next": "Supervisor"}
    else:
        plan_str = "\n".join(f"{i+1}. {step.step}" for i, step in enumerate(agent_response.action.steps))
        print(plan_str)
        if var.my_SAVE_DIALOGUE:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(plan_str)
                f.write("\n")
        return {
            "plan": agent_response.action.steps,
            "next": agent_response.action.steps[0].agent,
            "canvas":CANVAS.canvas,
            "explog_candidates": EXPLOG.relational_frame.candidates.df,
            "explog_processes": EXPLOG.relational_frame.processes.df,
            }
    
def worker_agent_node(state, agent, name):
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
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(state["past_steps"]))
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Here is the overall objective:
{state["inputs"]}

Now, you are tasked with: {task}. Please only do this task! Do not do anything else! Please note down important information on CANVAS before you end.
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
    
    
    # state["past_steps"].append((task, agent_response["messages"][-1].content))
    state["past_steps"].append(myStep(step=structured_response.summary, agent=name))
    
    print_stream(structured_response.summary)
    
    return {
        "past_steps": state["past_steps"], 
        "canvas":CANVAS.canvas,
        "explog_candidates": EXPLOG.relational_frame.candidates.df,
        "explog_processes": EXPLOG.relational_frame.processes.df,
    }
    
def whos_next(state):
    # NOTE: Route nodes purely from the shared state's `next` field.
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
        # NOTE: boss agent sends BossReview structured output.
        response_format=ToolStrategy(BossReview),
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors] # NOTE what is this?
    )
    boss_agent_node = functools.partial(boss_node, agent=boss_agent, name="Boss_Agent")
    
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
        inspect_explog,
        query_explog,
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
        get_kspacing_ecutwfc,
        generate_eos_test,
        read_energy_from_output,
        get_convergence_suggestions,
        analyze_BEEF_result
        ]
    # dft_agent = create_react_agent(workerllm, tools=dft_tools,
    #                                prompt="You are a DFT expert")   
    dft_agent = create_agent(
        model=workerllm,
        tools=dft_tools,
        system_prompt=dft_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors]
    )
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent")

    
    oer_tools = [
        inspect_explog,
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        OER_data_analasis_v2,
        read_df,
        arXiv_search,
        enter_candidate_in_log,
        submit_dft_job,
        get_terminations_ranking,
        list_adsorption_sites,
        read_explog,
        get_top_k_candidates,
        extract_df,
        wait_for_update,
        query_explog,
        ]
    # oer_agent = create_react_agent(workerllm, tools=oer_tools,
    #                                prompt=oer_agent_prompt)
    oer_agent = create_agent(
        model=workerllm,
        tools=oer_tools,
        system_prompt=oer_agent_prompt,
        response_format=ToolStrategy(wokerResponse),
        middleware=[DisableParallelToolCallsMiddleware(), handle_tool_errors]
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
    
    # NOTE: BOSS added to conditioanl map:
    conditional_map["Boss_Agent"] = "Boss_Agent"
    conditional_map["FINISH"] = END
    conditional_map["Supervisor"] = "Supervisor" 
    graph.add_conditional_edges("Supervisor", whos_next, conditional_map)

    # NOTE: - boss can either end the workflow or hand control back to the supervisor.
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
