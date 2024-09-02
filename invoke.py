import os
import yaml
import getpass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent
from src.prompt import *
from src.graph import create_graph


from src.utils import load_config, save_graph_to_file
from src.tools import *


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

if __name__ == "__main__":

    
    config = load_config(os.path.join('./config', "default.yaml"))
    WORKING_DIRECTORY = config['working_directory']
    pseudo_dir = config['pseudo_dir']
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)

    _set_if_undefined("ANTHROPIC_API_KEY")
    _set_if_undefined("LANGSIM_API_KEY")
    _set_if_undefined("LANGSIM_PROVIDER")
    _set_if_undefined("LANGSIM_MODEL")

    graph = create_graph(config)
    # llm = ChatAnthropic(model=config["LANGSIM_MODEL"], api_key=config['ANTHROPIC_API_KEY'])

    # graph = create_react_agent(llm, tools=[get_kpoints, dummy_structure, find_pseudopotential,write_script,get_bulk_modulus],
    #                                state_modifier=dftwriter_prompt)

    save_graph_to_file(graph, WORKING_DIRECTORY)
    # exit()
    # events = graph.stream(
    #     {
    #         "messages": [
    #             HumanMessage(
    #                 content=f"Generate a quantum espresso input for a crystal structure with 100% Cu atoms and save into a file. The working directory is {WORKING_DIRECTORY}.\
    #                     The pseduopotential directory is {config['pseudo_dir']}. The k point distance is normal."
    #             ) 
    #         ],
    #     },
    #     # Maximum number of steps to take in the graph
    #     {"recursion_limit": 150},
    # )
    # for s in events:
    #     print(s)
    #     print("----")


#     for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content=f"Generate a quantum espresso input for a crystal structure with 100% Cu atoms and calculate its bulk modules. The working directory is {WORKING_DIRECTORY}.\
#                         The pseduopotential directory is {pseudo_dir}.")
#         ]
#     }
# ):
#         if "__end__" not in s:
#             print(s)
#             print("----")
    for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=f"Generate a quantum espresso input for a crystal structure with 100% Cu atoms and calculate its bulk modules. The working directory is {WORKING_DIRECTORY}.\
                        The pseduopotential directory is {pseudo_dir}.")
        ]
    }
):
        if "__end__" not in s:
            print(s)
            print("----")