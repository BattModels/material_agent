import operator
from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple, Union
import functools
import os

from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
# from langchain_deepseek import ChatDeepSeek

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,create_react_agent
from pydantic import BaseModel, Field

from src.agent import create_agent
from src.tools import *
from src.prompt import dft_agent_prompt,hpc_agent_prompt,supervisor_prompt
from src import var

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

class PlanExecute(TypedDict):
    input: str
    plan: List[myStep]
    past_steps: List[myStep]
    response: str
    next: str

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
    

class Response(BaseModel):
    """End everything and response to the user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Plan, Response] = Field(
        description="Action to perform. If you need to further use tools to get the answer, use Plan."
        "If you want to end the conversation, use Response."
        # "DO NOT use response unless absolutly necessary."
    )

teamCapability = """
<DFT Agent>:
    - Create intial structure of the system
    - Find pseudopotential
    - Write initial script
    - generate convergence test input files
    - determine the best parameters from convergence test result
    - generate EOS calculation input files using the best parameters
    - Read output file to get energy
    - Calculate lattice constant
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
            message.pretty_print()
            if var.my_SAVE_DIALOGUE:
                with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                    f.write(message.pretty_repr())
                    f.write("\n")
    print()
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write("\n")

# def agent_node(state, agent, name):
#     print(f"Agent {name} is processing!!!!!")
#     for s in agent.stream(state, {"recursion_limit": 1000}):
#         print_stream(s)
#     return {"messages": [HumanMessage(content=s["messages"][-1].content, name=name)]}

def reflector_node(state, agent, name):
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, supervisor is waiting")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
    
    print(f"supervisor is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"supervisor is processing!!!!!\n")

    print(state)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(str(state))
            f.write("\n")
            
    # if first time, save orginal objective
    # empty the plan
    # Get current response
    # compare to the orginal objective
    # review what has been done
    # review what's on CANVAS
    # formulate a new input to be this is the goal, you've done this, but ....., please try .....
    # pass back to supervisor
    
    if var.reflector_first_visit:
        var.reflector_first_visit = False
        # save the original objective
        var.original_objective = state["input"]
        
    # empty the plan
    state["plan"] = []
    
    current_response = state["response"]
    
    task_formatted = f"""
Here is the original objective: {var.original_objective}
Here is the current response: {current_response}
Please 
"""
    
    for agent_response in agent.stream(
        {"messages": [("user", task_formatted)]},  {"configurable": {"thread_id": "1"}, "recursion_limit": 1000}
    ):
        # set agent_response to be the value of the first key of the dictionary
        agent_response = next(iter(agent_response.values()))
        print_stream(agent_response)
    
    
    

def supervisor_chain_node(state, chain, name):
    
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, supervisor is waiting")
        # wait for 5 second
        time.sleep(5)
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
    
    print(f"supervisor is processing!!!!!")
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(f"supervisor is processing!!!!!\n")

    print(state)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(str(state))
            f.write("\n")

    output = chain.invoke(state)

    if isinstance(output.action, Response):
        return {"response": output.action.response, "next": "FINISH"}
    # elif isinstance(output.action, Response):
    #     return {"response": "Plan is not finished! Do not use response!", "next": "Supervisor"}
    else:
        return {"plan": output.action.steps, "next": output.action.steps[0].agent}
    
    

def worker_agent_node(state, agent, name, past_steps_list):
    # read "status.txt" in the working directory
    with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
        status = f.read()
    while status == "stop":
        print(f"Calculation pause, {name} Agent is waiting")
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
    print(plan_str)
    if var.my_SAVE_DIALOGUE:
        with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
            f.write(plan_str)
            f.write("\n")
    task = plan[0]
#     task_formatted = f"""For the following plan:
# {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    old_tasks_string = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(past_steps_list))
    task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Now, you are tasked with: {task}. Please only do this task! Do not do anything else!
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
    
    # past_steps_list.append((task, agent_response["messages"][-1].content))
    past_steps_list.append(myStep(step=agent_response["messages"][-1].content, agent=name))
    
    print_stream(agent_response)
    
    return {
        "past_steps": past_steps_list,
    }
    
def recusive_agent_node(state, agent, name, past_steps_list):
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
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    workerllm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # workerllm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # workerllm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"], model_kwargs={'parallel_tool_calls': False})
    # llm = ChatDeepSeek(model_name=config["DeepSeek_MDL"], api_key=config['DeepSeek_API_KEY'], api_base=config['DeepSeek_BASE_URL'], temperature=0.0)
    
    var.my_WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    if not eval(config["SAVE_DIALOGUE"]):
        var.my_SAVE_DIALOGUE = False
    
    supervisor_prompt = ChatPromptTemplate.from_template(
        f"""
<Role>
    You are a scientist supervisor tasked with managing a conversation for scientific computing between the following workers: {members}. You don't have to use all the members, nor all the capabilities of the members.
<Objective>
    Given the following user request, decide which the member to act next, and do what
<Instructions>:
    1.  If the plan is empty, For the given objective, come up with a simple, high level plan based on the capability of the team listed here: {teamCapability} and the restrictions listed here: {teamRestriction} 
        You don't have to use all the members, nor all the capabilities of the members.
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
        If you are given a list of systems, process them one by one: generate plan for one system first, finish that system, then generate plan for the next system.

        If the plan is not empty, update the plan:
        Your objective was this:
        {{input}}

        Your original plan was this:
        {{plan}}

        Your pass steps are:
        {{past_steps}}

        Update your plan accordingly, and fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. 
        choose plan if there are still steps to be done, or response if everything is done.
    2.  Given the conversation above, suggest who should act next. next could only be selected from: {OPTIONS}.
    3.  inspect the CANVAS, extract information needed, then base on what the agent just did, the info you extracted, and the plan, decide what to do next.
    4.  If your end result is different from your expectation, please reflect on what you have done by inspect and read through the canvas, scientificly, try to understand why the result is different, list out possible reasons, adjust your plan accordingly, and try to eliminate possible causes one by one.
        Do not stop until the end result is within user specified margin of error, or you have tried everything you can think of. Only if user did not specify a margin of error, you can judge by yourself.
<Requirements>:
    0.  Do not generate convergence test for all systems and all configurations.
    1.  Please only generate one batch of convergence test for the most complicated system using the most complicated configuration. 
    2.  when trying to explore the configuration design space, try a couple of configurations first, and then refine the search based on the results.
    3.  Do not touch the canvas unless you are reflecting.
        """
    )
    
    
    # System Supervisor with tool bind and with_structured_output
    
    supervisor_tools = [
        inspect_my_canvas,
        read_my_canvas,
        ]
    supervisor_chain = supervisor_prompt | llm.bind_tools(supervisor_tools).with_structured_output(Act)
    supervisor_agent = functools.partial(supervisor_chain_node, chain=supervisor_chain, name="Supervisor")
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
        ]
    dft_agent = create_react_agent(workerllm, tools=dft_tools,
                                   state_modifier=dft_agent_prompt)   
    dft_node = functools.partial(worker_agent_node, agent=dft_agent, name="DFT_Agent", past_steps_list=PAST_STEPS)


    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [
        inspect_my_canvas,
        write_my_canvas,
        read_my_canvas,
        submit_and_monitor_job,
        add_resource_suggestion
        ]

    hpc_agent = create_react_agent(workerllm, tools=hpc_tools,
                                   state_modifier=hpc_agent_prompt)

    hpc_node = functools.partial(worker_agent_node, agent=hpc_agent, name="HPC_Agent", past_steps_list=PAST_STEPS)
    
    ### MD Agent
    # md_tools = [
    #     find_classical_potential,
    #     init_structure_data,
    #     write_LAMMPS_script
    # ]
    
    # md_agent = create_react_agent(llm, tools=md_tools,
    #                               state_modifier=md_agent_prompt)
    
    # md_node = functools.partial(worker_agent_node, agent=md_agent, name="MD_Agent", past_steps_list=PAST_STEPS)

    # save_graph_to_file(dft_agent, config['working_directory'], "dft_agent")
    


    # Create the graph
    graph = StateGraph(PlanExecute)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
    # graph.add_node("MD_Agent", md_node)
    # graph.add_node("CSS_Agent", css_node)

    graph.add_node("Supervisor", supervisor_agent)
    
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
    # return graph.compile(checkpointer=memory)
    return graph.compile()


