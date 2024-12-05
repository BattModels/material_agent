import operator
from typing import Annotated, Sequence, TypedDict,Literal, List, Dict, Tuple
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
from pydantic import BaseModel

from src.agent import create_agent
from src.tools import find_pseudopotential, submit_single_job,write_script,calculate_lc,generate_convergence_test,generate_eos_test,\
submit_and_monitor_job,find_job_list,read_energy_from_output,add_resource_suggestion, get_kspacing_ecutwfc
from src.prompt import dft_agent_prompt,hpc_agent_prompt
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

members = ["DFT_Agent", "HPC_Agent"]
instructions = [dft_agent_prompt, hpc_agent_prompt]

membersInstruction = ""
for member, instruction in zip(members, instructions):
    membersInstruction += f"{member}'s instruction is: {instruction}\n"

# system_prompt = (f'''
#     <Role>
#         You are a supervisor tasked with managing a conversation for scientific computing between the following workers: {members}.
#     <Objective>
#         Given the following user request, respond with the member to act next. When you have the result terminate immediately. When finished,respond with FINISH.
#     <Member>
#         Here are the ability of each member. 
#         <DFT Agent>:
#             - Find pseudopotential
#             - Write initial script
#             - generate convergence test script
#             - determine kspacing and ecutwfc from convergence test result
#             - generate EOS script
#             - read energy from output
#             - Calculate lattice constants.
#         <HPC Agent>:
#             - Suggest resources for each job.
#             - Submit jobs.
#     '''
# )

system_prompt = (f'''
You will be told to use which agent and what to do. Follow the instruction strictly.  i.e. if asked to find pseudopotential, once the agent found the potential, terminate immediately.
Once you have the result from any agent that achives the task given, respond with FINISH immediately. DO NOT do anything else.
Once you see 'Final Answer' in the response, respond with FINISH immediately. DO NOT do anything else.
    '''
)

# system_prompt = (f'''
#     You are a supervisor tasked with managing a conversation between the
#     following workers:  {members}, based on {membersInstruction}. Given the following user request,
#     respond with the worker to act next. 
#     When finished,respond with FINISH.
#     '''
# )
# DFT_Agent is responsible for generating scripts and computing lattice constants. HPC_Agent is responsible for submitting jobs and reading output.

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt+"Given the conversation above, who should act next?\
            Or should we FINISH? Select one of: {options}." ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(options=str(options), members=", ".join(members))

def print_stream(s):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        print("##########should've pretty printed##############")
        print(message)
        # message.pretty_print()

def agent_node(state, agent, name):
    print(f"Agent {name} is processing!!!!!")
    for s in agent.stream(state, {"recursion_limit": 1000}):
        print_stream(s)
    return {"messages": [HumanMessage(content=s["messages"][-1].content, name=name)]}

def create_graph(config: dict) -> StateGraph:
    # Define the model
    if 'claude' in config['LANGSIM_MODEL']:
        llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # System Supervisor
    def supervisor_agent(state):
        print("Supervisor!!!!!!!!!")
        supervisor_chain = (
            prompt
            | llm.with_structured_output(routeResponse)
        )
        return supervisor_chain.invoke(state)
    ## Memory Saver
    memory = MemorySaver()


    ### DFT Agent
    dft_tools = [
        find_pseudopotential,
        write_script,
        calculate_lc,
        generate_convergence_test,
        get_kspacing_ecutwfc,
        generate_eos_test,
        read_energy_from_output
        ]
    dft_agent = create_react_agent(llm, tools=dft_tools,
                                   state_modifier=dft_agent_prompt)   
    dft_node = functools.partial(agent_node, agent=dft_agent, name="DFT_Agent")


    ### HPC Agent
    # hpc_tools = [read_script, submit_and_monitor_job, read_energy_from_output]
    hpc_tools = [
        submit_and_monitor_job,
        find_job_list,
        add_resource_suggestion
        ]

    hpc_agent = create_react_agent(llm, tools=hpc_tools,
                                   state_modifier=hpc_agent_prompt)

    hpc_node = functools.partial(agent_node, agent=hpc_agent, name="HPC_Agent")

    # save_graph_to_file(dft_agent, config['working_directory'], "dft_agent")
    


    # Create the graph
    graph = StateGraph(AgentState)
    graph.add_node("DFT_Agent", dft_node)
    graph.add_node("HPC_Agent", hpc_node)
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
    return graph.compile(checkpointer=memory)


