import os
import yaml
import getpass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from src.graph import create_graph


from src.utils import load_config, save_graph_to_file


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

if __name__ == "__main__":

    
    config = load_config(os.path.join('./config', "default.yaml"))
    WORKING_DIRECTORY = config['working_directory']
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)

    _set_if_undefined("ANTHROPIC_API_KEY")
    _set_if_undefined("LANGSIM_API_KEY")
    _set_if_undefined("LANGSIM_PROVIDER")
    _set_if_undefined("LANGSIM_MODEL")

    graph = create_graph(config)
    

    save_graph_to_file(graph, WORKING_DIRECTORY)
    exit()
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=f"Generate a quantum espresso input for a crystal structure with 50% Cu atoms and save into a file. The working directory is {WORKING_DIRECTORY}.",
                ) 
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")