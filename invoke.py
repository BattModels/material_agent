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
# from src.prompt import hpc_agent_prompt,dft_agent_prompt
from src.graph import create_graph


from src.utils import load_config, save_graph_to_file,check_config
# from src.tools import 



if __name__ == "__main__":

    namelist = ['Li (bcc)', 'Na (bcc)', 'K (bcc)', 'Rb (bcc)', 'Ca (fcc)', 'Sr (fcc)', 'Ba (bcc)', 'V (bcc)', 'Nb (bcc)', 'Ta (bcc)', 'Mo (bcc)', 'W (bcc)',\
             'Fe (bcc)', 'Rh (fcc)', 'Ir (fcc)', 'Ni (fcc)'\
            , 'Pd (fcc)', 'Pt (fcc)', 'Cu (fcc)', 'Ag (fcc)', 'Au (fcc)', 'Al (fcc)', 'Pb (fcc)', 'C (dia)', 'Si (dia)', 'Ge (dia)', 'Sn (dia)']
    filelist = ['Li_bcc.in', 'Na_bcc.in', 'K_bcc.in', 'Si_dia.in', 'Pd_fcc.in', 'Ge_dia.in', 'Au_fcc.in', 'C_dia.in', 'Cu_fcc.in', 'Fe_bcc.in', 'Ca_fcc.in', 'Pb_fcc.in', 'W_bcc.in', 'Mo_bcc.in', 'Pt_fcc.in', 'Ag_fcc.in', 'Rh_fcc.in', 'Sr_fcc.in', 'Nb_bcc.in', 'Al_fcc.in', 'Rb_bcc.in', 'Ta_bcc.in', 'Ir_fcc.in', 'Sn_dia.in', 'Ba_bcc.in', 'V_bcc.in', 'Ni_fcc.in']

    ## Convergence Test
    userMessage_1 = '''
    You are going to do cenvergence test for Li BCC structure. Compute the the total energy for different kpoints based on kspacing 0.1,0.2 ,0.3 and 40,60,80 ecutwfc. Run the calculation through slurm and report the result.
    '''

    userMessage_2 = '''
    Based on previous result, calculate the lattice constant for Li BCC structure.

    1. choose appropriate kpoints and ecutwfc.
    2. Generate input script with different scale factor 
    3. Submit the job through slurm
    4. Calculate the lattice constant
    '''
   

    userMessage_3 = '''
    Calculate the lattice constant for Li BCC structure, from previous result, use 0.1 kspacing and 100 ecutwfc.
    First generate 5 input script with different scale factor in order to calculate EOS, then run the calculation through slurm. When the calculation is done, calculate the lattice constant
    '''

    userMessage_4 = '''
    You are going to do cenvergence test for Li BCC structure. 
    1. Compute the the total energy for different kpoints based on kspacing 0.1,0.2 ,0.3 and 40,60,80,100,120 ecutwfc. Run the calculation through slurm and report the result.
    2. After the first batch calculation, choose appropriate kpoints and ecutwfc. Then generate input script for EOS and submit the job. When the calculation is done, calculate the lattice constant
    '''
    
    config = load_config(os.path.join('./config', "default.yaml"))
    check_config(config)
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    os.makedirs(WORKING_DIRECTORY, exist_ok=False)

    graph = create_graph(config)
    llm_config = {"thread_id": "1", 'recursion_limit': 1000}
    

    # save_graph_to_file(graph, WORKING_DIRECTORY, "super_graph")

    
    for s in graph.stream(
    {
        "messages": [
            HumanMessage(content=f"{userMessage_4}")
        ]
    },llm_config):
        if "__end__" not in s:
            print(s)
            print("----")

    # for s in graph.stream(
    # {
    #     "messages": [
    #         HumanMessage(content=f"{userMessage_2}")
    #     ]
    # },llm_config):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")
