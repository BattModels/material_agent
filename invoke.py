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


from src.utils import load_config, save_graph_to_file,_set_if_undefined
from src.tools import *



if __name__ == "__main__":



    # userMessage = "Generate a quantum espresso input for a crystal structure with 50% Cu atoms and Au atoms and calculate its bulk modules. Try until reaches 5 trials."
    userMessage = "You are going to do cenvergence test for Li BCC structure. Compute the the total energy for different kpoints based on kspacing 0.1,0.2 ,0.3 and low, normal, high ecutwfc. \
        Use the highest ecutwfc and kpoints convergence test. Use the highest kpoints for ecutwfc convergence test. Report the results when finished."

    # userMessage = "Generate a quantum espresso input for a crystal structure with 50% Cu atoms and Au atoms and calculate its bulk modules. Try until reaches 5 trials."
    # userMessage = "Generate a quantum espresso input for a crystal structure with 50% Cu atoms and Au atoms, and run the calculation with slurm."
    # userMessage = 'Generate 27 quantum espresso input file for {namelist}. \
    #                         calculate their lattice constant and report it for each time, if failed ,just report the error message and continue the next.'


    
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

    save_graph_to_file(graph, WORKING_DIRECTORY, "super_graph")

    namelist = ['Li (bcc)', 'Na (bcc)', 'K (bcc)', 'Rb (bcc)', 'Ca (fcc)', 'Sr (fcc)', 'Ba (bcc)', 'V (bcc)', 'Nb (bcc)', 'Ta (bcc)', 'Mo (bcc)', 'W (bcc)', 'Fe (bcc)', 'Rh (fcc)', 'Ir (fcc)', 'Ni (fcc)', 'Pd (fcc)', 'Pt (fcc)', 'Cu (fcc)', 'Ag (fcc)', 'Au (fcc)', 'Al (fcc)', 'Pb (fcc)', 'C (dia)', 'Si (dia)', 'Ge (dia)', 'Sn (dia)']
    filelist = ['Li_bcc.in', 'Na_bcc.in', 'K_bcc.in', 'Si_dia.in', 'Pd_fcc.in', 'Ge_dia.in', 'Au_fcc.in', 'C_dia.in', 'Cu_fcc.in', 'Fe_bcc.in', 'Ca_fcc.in', 'Pb_fcc.in', 'W_bcc.in', 'Mo_bcc.in', 'Pt_fcc.in', 'Ag_fcc.in', 'Rh_fcc.in', 'Sr_fcc.in', 'Nb_bcc.in', 'Al_fcc.in', 'Rb_bcc.in', 'Ta_bcc.in', 'Ir_fcc.in', 'Sn_dia.in', 'Ba_bcc.in', 'V_bcc.in', 'Ni_fcc.in']
    for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=f"{userMessage} \
                     The working directory is {WORKING_DIRECTORY}.\
                        The pseduopotential directory is {pseudo_dir}.")
        ]
    }
,{"recursion_limit": 1000}):
        if "__end__" not in s:
            print(s)
            print("----")
