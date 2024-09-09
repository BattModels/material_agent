import operator
from typing import Annotated, Sequence, TypedDict,Literal
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
from langgraph.prebuilt import ToolNode,create_react_agent
from pydantic import BaseModel

from src.agent import create_agent
from src.tools import *
from src.prompt import *
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Either agent can decide to end
# This is a simple router that used in the first version
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

members = ["DFT_Agent", "HPC_Agent",'Chem_Agent']
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt+"Given the conversation above, who should act next?\
            Or should we FINISH? Select one of: {options}. We don't need HPC_Agent at the moment" ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(options=str(options), members=", ".join(members))

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

def create_graph(config: dict) -> StateGraph:
    if 'claude' in config['LANGSIM_MODEL']:
        llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    def supervisor_agent(state):
        supervisor_chain = (
            prompt
            | llm.with_structured_output(routeResponse)
        )
        return supervisor_chain.invoke(state)
   
    dft_agent = create_react_agent(llm, tools=[get_kpoints, dummy_structure, find_pseudopotential,write_script,get_bulk_modulus,get_lattice_constant],
                                   state_modifier=dftwriter_prompt)   
    dft_node = functools.partial(agent_node, agent=dft_agent, name="DFT_Agent")

    hpc_agent = create_react_agent(llm, tools=[])
    hpc_node = functools.partial(agent_node, agent=hpc_agent, name="HPC_Agent")

    # hpc_agent2 = create_react_agent(llm, tools=[])
    # hpc_node2 = functools.partial(agent_node, agent=hpc_agent2, name="Chem_Agent")


    save_graph_to_file(dft_agent, config['working_directory'], "dft_agent")
    # save_graph_to_file(dft_agent, config['working_directory'], "dft_agent")


    # Create the graph
    graph = StateGraph(AgentState)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
    # graph.add_node("Chem_Agent", hpc_node2)

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


