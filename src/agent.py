import os,yaml

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful compututation material scientist that produces high-quality quantum espresso input files for density functional theory calculations, but don't know current events. \
                    For each query vailidate that it contains chemical elements from the periodic table and otherwise cancel.\
                    Always generate conventional cell with ibrav=0 and do not use celldm and angstrom at the same time.\
                    Please include CONTROL, SYSTEM, ELECTRONS, ATOMIC_SPECIES, K_POINTS, ATOMIC_POSITIONS, and CELL. \
                    Use the right smearing based on the material.\
                    If not specified, use normal for k point spacing.\
                    If the system involves hubbard U correction, specify starting magnetization in SYSTEM card and hubbard U parameters in HUBBARD card, and use the pre-defined hubbard correction tool.\
                    The conv_thr should scale with number of atoms in the system.\
                    Do not use the get crystal structure tool unless you are explicitly told to do so.\
                    Do not ask for the user's permission to proceed.\
                    Please make sure that the input is the most optimal. \
                    The input dictionary should be readable by ase.Espresso.\
                    ",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)