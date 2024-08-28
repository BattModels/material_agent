import os
import yaml
import getpass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from src.graph import create_graph


def load_config(path: str):
    ## Load the configuration file
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ## Set up environment variables
    os.environ['ANTHROPIC_API_KEY'] = config['ANTHROPIC_API_KEY']   
    os.environ["LANGSIM_PROVIDER"] = config['LANGSIM_PROVIDER']
    os.environ["LANGSIM_API_KEY"] = config['LANGSIM_API_KEY']
    os.environ["LANGSIM_MODEL"] = config['LANGSIM_MODEL']
    return config


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

if __name__ == "__main__":

    
    config = load_config(os.path.join('./config', "default.yaml"))
    _set_if_undefined("ANTHROPIC_API_KEY")
    _set_if_undefined("LANGSIM_API_KEY")
    _set_if_undefined("LANGSIM_PROVIDER")
    _set_if_undefined("LANGSIM_MODEL")

    graph = create_graph(config)
    

    try:
        im = graph.get_graph(xray=True).draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(im)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Generate a quantum espresso input for a crystal structure with 50% Cu atoms and save into a file.",
                ) 
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")