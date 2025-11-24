import os
import yaml
import getpass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
# from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent
# from src.prompt import hpc_agent_prompt,dft_agent_prompt
# from src.graph import create_graph
from src.planNexe2 import create_planning_graph as create_graph
# from src.planNexeHighPlan import create_planning_graph as create_graph

import time
from src.utils import load_config, save_graph_to_file,initialize_database
from src.myCANVAS import CANVAS
from src import var


if __name__ == "__main__":

    namelist = ['Li (bcc)', 'Na (bcc)', 'K (bcc)', 'Rb (bcc)', 'Ca (fcc)', 'Sr (fcc)', 'Ba (bcc)', 'V (bcc)', 'Nb (bcc)', 'Ta (bcc)', 'Mo (bcc)', 'W (bcc)',\
             'Fe (bcc)', 'Rh (fcc)', 'Ir (fcc)', 'Ni (fcc)'\
            , 'Pd (fcc)', 'Pt (fcc)', 'Cu (fcc)', 'Ag (fcc)', 'Au (fcc)', 'Al (fcc)', 'Pb (fcc)', 'C (dia)', 'Si (dia)', 'Ge (dia)', 'Sn (dia)']
    filelist = ['Li_bcc.in', 'Na_bcc.in', 'K_bcc.in', 'Si_dia.in', 'Pd_fcc.in', 'Ge_dia.in', 'Au_fcc.in', 'C_dia.in', 'Cu_fcc.in', 'Fe_bcc.in', 'Ca_fcc.in', 'Pb_fcc.in', 'W_bcc.in', 'Mo_bcc.in', 'Pt_fcc.in', 'Ag_fcc.in', 'Rh_fcc.in', 'Sr_fcc.in', 'Nb_bcc.in', 'Al_fcc.in', 'Rb_bcc.in', 'Ta_bcc.in', 'Ir_fcc.in', 'Sn_dia.in', 'Ba_bcc.in', 'V_bcc.in', 'Ni_fcc.in']
    
    structure = 'Li (bcc)'
    # structure = 'Li (bcc)'

    ## Convergence Test
    userMessage_1 = f'''
    You are going to do cenvergence test for {structure} structure, use initial lattice constant 3.451. Compute the the total energy for different kpoints based on kspacing 0.1,0.15,0.2,0.25 ,0.3 and 40,60,80,100,120 ecutwfc. Run the calculation through slurm and report the result.
    '''

    userMessage_2 = f'''
    Based on previous result, Choose appropriate kpoints and ecutwfc, generate input script with different scale factor and Submit the job through slurm
    '''
   

    userMessage_3 = f'''
    Now we have all the result, please calculate the lattice constant of {structure} structure and report the result.
    '''

    userMessage_4 = '''
    You are going to calculate the lattic constant for FCC Ca through DFT, the experiment value is 5.556. 
    1. Compute the the total energy for different kpoints based on kspacing 0.1,0.2 ,0.3 and 40,60,80,100,120 ecutwfc. Run the calculation through slurm and report the result.
    2. After the first batch calculation, choose appropriate kpoints and ecutwfc. Then generate input script for EOS and submit the job.
    3. When the calculation is done, calculate the lattice constant
    '''
    
    userMessage_5 = '''
    through DFT, please calculate the lattic constant for for following system listed in the following format: Lattice_structure Species (experimental_value)
    Li (bcc)	3.451
    Na (bcc)	4.209
    K (bcc)	5.212
    Rb (bcc)	5.577
    Ca (fcc)	5.556
    Sr (fcc)	6.04
    Ba (bcc)	5.002
    V (bcc)	3.024
    Nb (bcc)	3.294
    Ta (bcc)	3.299
    Mo (bcc)	3.141
    W (bcc)	3.16
    Fe (bcc)	2.853
    Rh (fcc)	3.793
    Ir (fcc)	3.831
    Ni (fcc)	3.508
    Pd (fcc)	3.876
    Pt (fcc)	3.913
    Cu (fcc)	3.596
    Ag (fcc)	4.062
    Au (fcc)	4.062
    Al (fcc)	4.019
    Pb (fcc)	4.912
    C (dia)	3.544
    Si (dia)	5.415
    Ge (dia)	5.639
    Sn (dia)	6.474
    '''
    
    userMessage_6 = "You are going to calculate the lattice constant for BCC Li through DFT, the experiment value is 3.451, use this to create the initial structure."
    userMessage_7 = "You are going to generat a Pt surface structure with 2x2x4 supercell, then do a convergence test, use maximum ecutwfc = 160. Get the optimal kspacing and ecutwfc."
    userMessage_8 = """Please generate intial structures required to calculate CO adsorbtion on Pt(111) surface with 1/4 coverage (2x2x4 supercell), and calculate the adsorbtion energy."""
    userMessage_9 = """
    Please find out the most perfered adsorbtion site and adsorbate orientation (up or down) for CO adsorbtion on Pt(111) surface with 1/4 coverage (2x2x4 supercell).
    """
    userMessage_10 = """please find the adsorption energy difference between the most favorable configurations (different adsorbate orientations 0, 90, 180) at fcc site and
    most favorable configuration (different adsorbate orientations 0, 90, 180) at ontop site for CO on Pt(111) surface with p(2x2) adsorbate overlayer (1/4 coverage). 
    Please use PBE pseudopotential and PBE exchange correlation function.
    Literatures suggest that ontop site is 0.108 eV less stable than fcc site when using PBE xc. 
    If your result is not within 10 percent of the literature, please find out possible reasons and resolve it."""
    
    userMessage_11 = "I am trying to study adsorption of CO on Pt111 surface at fcc site. Job CO_Pt111_fcc_upright_k_0.3_ecutwfc_60.pwi did not converge, please figure out why and resolve the convergence issue."
    
    userMessage_12 = """please find the adsorption energy difference between the most favorable configurations (different adsorbate orientations 0, 90, 180) at fcc site and most favorable configuration (different adsorbate orientations 0, 90, 180) at ontop site for CO on Pt(111) surface with p(2x2) adsorbate overlayer (1/4 coverage), and analyze the uncertainty.
    Please use PBE pseudopotential and Bayesian Error Estimation Functional (BEEF) exchange correlation function.
    Literatures suggest that ontop site is 0.18 eV less stable than fcc site when using PBE xc.
    If your result is not within 10 percent of the literature, please find out possible reasons and resolve it."""
    
    testMessage = '''
    please generate a single input script for Li BCC structure with kspacing 0.1 and ecutwfc 40
    '''
    
    config = load_config(os.path.join('./config', "default.yaml"))
    # check_config(config)
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # print N number of '#', where n = len("##  Working directory: " + WORKING_DIRECTORY + " ##")
    print("#" * (len("##  Working directory: " + WORKING_DIRECTORY + " ##")))
    print("##  Working directory: " + WORKING_DIRECTORY + " ##")
    print("#" * (len("##  Working directory: " + WORKING_DIRECTORY + " ##")))
    
    assert WORKING_DIRECTORY is not None, "Please set the WORKING_DIRECTORY var"
    
    CANVAS.set_working_directory(WORKING_DIRECTORY)
    # CANVAS.canvas["finished_job_list"] = ["CO_Pt111_fcc_upright_k_0.3_ecutwfc_60.pwi"]
    
    # set environment variable
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # check if working directory exists, if so delete it
    if os.path.exists(WORKING_DIRECTORY):
        os.system(f"rm -rf {WORKING_DIRECTORY}")
    
    os.makedirs(WORKING_DIRECTORY, exist_ok=False)
    
    # check if resource_suggestions.db exist in the working directory
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if os.path.exists(db_file):
        os.remove(db_file)
    initialize_database(db_file)

    graph = create_graph(config)
    llm_config = {"thread_id": "1", 'recursion_limit': 1000}
    
    # print(graph)
    

    # save_graph_to_file(graph, WORKING_DIRECTORY, "super_graph")
    # exit()

    
    # for s in graph.stream(
    # {
    #     "messages": [
    #         HumanMessage(content=f"{userMessage_4}")
    #     ]
    # },llm_config):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")

    # for s in graph.stream(
    # {
    #     "messages": [
    #         HumanMessage(content=f"{userMessage_2}")
    #     ]
    # },llm_config):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")

    # for s in graph.stream(
    # {
    #     "input": f"{userMessage_6}",
    #     "plan": [],
    #     "past_steps": []
    # },llm_config):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")
            
    # for s in graph.stream(
    # {
    #     "input": [
    #         HumanMessage(content=f"{userMessage_5}")
    #     ]
    # },llm_config):
    #     if "__end__" not in s:
    #         print(s)
    #         print("----")

    print("Start, check the log file for details")
    log_filename = f"./log/agent_stream_{int(time.time())}.log"  # Add timestamp to filename
    with open(log_filename, "a") as log_file:
        log_file.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        if eval(config["SAVE_DIALOGUE"]):
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        for s in graph.stream(
            {
                "input": f"{userMessage_6}",
                "plan": [],
                "past_steps": []
            }, llm_config):
            
            if "__end__" not in s:
                print(s)
                print("----")
                if eval(config["SAVE_DIALOGUE"]):
                    with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                        f.write(repr(s) + "\n")
                        f.write("----\n")
                    
                # Print to console
                log_file.write(f"{s}\n")
                log_file.write("----\n")
                log_file.flush()
        log_file.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        if eval(config["SAVE_DIALOGUE"]):
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    print("End, check the log file for details")
