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
from langgraph.prebuilt import ToolNode

from src.agent import create_agent
from src.tools import get_kpoints, dummy_structure, write_script
from src.prompt import *
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Either agent can decide to end
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

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

def create_graph(config: dict) -> StateGraph:
    if 'claude' in config['LANGSIM_MODEL']:
        llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'])

    # Define the agents
    input_agent = create_agent(llm, [get_kpoints, dummy_structure,write_script], prompt_content=dftwriter_prompt,system_message="")
    # Define the node to run the agents
    input_node = functools.partial(agent_node, agent=input_agent, name="input_generator")

    calculate_agent = create_agent(llm, [], prompt_content=calculater_prompt,system_message="")
    calculate_node = functools.partial(agent_node, agent=calculate_agent, name="calculator")
    # Define the node to run tools
    tools = [get_kpoints, dummy_structure, write_script]
    tool_node = ToolNode(tools)

    # Create the graph
    graph = StateGraph(AgentState)
    graph.add_node("input_generator", input_node)
    graph.add_node("calculator", calculate_node)
    graph.add_node("call_tool", tool_node)
    graph.add_conditional_edges("input_generator",
                               router,
                               {'continue': "input_generator",'call_tool': "call_tool", '__end__': END})
    graph.add_conditional_edges("calculator",
                               router,
                               {'continue': "calculator",'call_tool': "call_tool", '__end__': END})
    graph.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "input_generator": "input_generator",
        'calculator': "calculator",
    },
)
    graph.add_edge(START, "input_generator") 
    return graph.compile()


