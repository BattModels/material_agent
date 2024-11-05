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
from src.graph import create_graph


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
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

teamCapability = """
<DFT Agent>:
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

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""For the given objective, come up with a simple, high level plan based on the capability of the team listed here: {teamCapability} and the restrictions listed here: {teamRestriction} \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
            f"""For the given objective, come up with a simple, high level plan based on the capability of the team listed here: {teamCapability} \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{{input}}

Your original plan was this:
{{plan}}

You have currently done the follow steps:
{{past_steps}}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

def create_planning_graph(config):
    simTeam = create_graph(config)
    
    llm = None
    if 'claude' in config['LANGSIM_MODEL']:
        llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    
    planner = planner_prompt | llm.with_structured_output(Plan)
    replanner = replanner_prompt | llm.with_structured_output(Act)
    
    past_steps_list = []
    
    def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        print(plan_str)
        task = plan[0]
    #     task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        old_tasks_string = "\n".join(f"{i+1}. {step[0]}: {step[1]}" for i, step in enumerate(past_steps_list))
        task_formatted = f"""
Here are what has been done so far:
{old_tasks_string}

Now, you are tasked with executing step {1}, {task}.
"""
        
        config = {"configurable": {"thread_id": "1"}}
        
        agent_response = simTeam.invoke(
            {"messages": [("user", task_formatted)]},  {"configurable": {"thread_id": "1"}}
        )
        
        past_steps_list.append((task, agent_response["messages"][-1].content))
        
        return {
            "past_steps": [past_steps_list[-1]],
        }


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
    
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )
    
    return workflow.compile()



