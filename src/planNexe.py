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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,create_react_agent
from pydantic import BaseModel, Field

from src.agent import create_agent
from src.tools import find_pseudopotential, submit_single_job,write_script,calculate_lc,generate_convergence_test,generate_eos_test,\
submit_and_monitor_job,find_job_list,read_energy_from_output,add_resource_suggestion
from src.prompt import dft_agent_prompt,hpc_agent_prompt

############# utility print function ################
def print_stream(s):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()
        
        
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    next: str

members = ["DFT_Agent", "HPC_Agent", "Planner", "Replanner"]
instructions = [dft_agent_prompt, hpc_agent_prompt]

options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

class Response(BaseModel):
    """Response to user."""

    response: str
    
class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
"""
)


system_prompt = (f'''
    <Role>
        You are a supervisor tasked with managing a conversation for scientific computing between the following workers: {members}.
    <Objective>
        Given the following user request, respond with the member to act next. When finished,respond with FINISH.
    <Member>
        Here are the ability of each member. 
        <DFT Agent>:
            - Find pseudopotential
            - Write initial script
            - Calculate lattice constants.
        <HPC Agent>:
            - Submit jobs and read output.
    '''
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt+"Given the conversation above, who should act next?\
            Or should we FINISH? Select one of: {options}." ),
        MessagesPlaceholder(variable_name="input"),
    ]
).partial(options=str(options), members=", ".join(members))




def agent_node(state, agent, name):
    for s in agent.stream(state, {"recursion_limit": 1000}):
        print_stream(s)
    return {"messages": [HumanMessage(content=s["messages"][-1].content, name=name)]}

def execute_step(state: PlanExecute, agent):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    # agent_response = agent_executor.invoke(
    #     {"messages": [("user", task_formatted)]}
    # )
    # return {
    #     "past_steps": [(task, agent_response["messages"][-1].content)],
    # }
    for s in agent.stream(
        {
            "messages": [
                HumanMessage(content=task_formatted)
            ]
        },{"recursion_limit": 1000}):
        print_stream(s)
        
    return {"pass_steps": [(task, s["messages"][-1].content)]}

def create_graph(config: dict) -> StateGraph:
    # Define the model
    if 'claude' in config['LANGSIM_MODEL']:
        llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)

    def supervisor_agent(state):
        supervisor_chain = (
            prompt
            | llm.with_structured_output(routeResponse)
        )
        return supervisor_chain.invoke(state)
    
    ### DFT Agent
    dft_tools = [find_pseudopotential,write_script,calculate_lc,generate_convergence_test,generate_eos_test,read_energy_from_output]
    dft_agent = create_react_agent(llm, tools=dft_tools,
                                    state_modifier=dft_agent_prompt)   
    dft_node = functools.partial(execute_step, agent=dft_agent)


    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [submit_and_monitor_job,submit_single_job,find_job_list,add_resource_suggestion]

    hpc_agent = create_react_agent(llm, tools=hpc_tools,
                                    state_modifier=hpc_agent_prompt)

    hpc_node = functools.partial(execute_step, agent=hpc_agent)
    
    


    planner = planner_prompt | llm.with_structured_output(Plan)
    
    replanner = replanner_prompt | llm.with_structured_output(Act)
    
    
        
    def plan_step(state: PlanExecute):
        plan = planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}


    def replan_step(state: PlanExecute):
        output = replanner.invoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}


    def should_end(state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"
    
     # Create the graph
    graph = StateGraph(PlanExecute)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
    graph.add_node("Planner", plan_step)
    graph.add_node("Replanner", replan_step)
    # graph.add_node("CSS_Agent", css_node)
    graph.add_node("Supervisor", supervisor_agent)
    
    for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
        graph.add_edge(member, "Supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    graph.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
    graph.add_edge(START, "Supervisor") 
    return graph.compile()