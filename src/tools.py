

from math import e
from src.utils import *
from ase import Atoms, Atom
from langchain.agents import tool
import os 
from typing import Annotated,Dict, Literal,Optional
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic
import ast
import re
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.eos import calculate_eos,EquationOfState
from ase.units import kJ
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.io.trajectory import Trajectory
import subprocess
import time
from pysqa import QueueAdapter
import json
import pandas as pd
import sqlite3
from filecmp import cmp
### DFT tools

@tool
def get_kpoints(atom_dict: AtomsDict, kspacing: float) -> list:
    """Returns the kpoints of a given ase atoms object and specific kspacing."""
    atoms = Atoms(**atom_dict.dict())
    cell = atoms.cell
    ## Check input kspacing is valid
    if kspacing <= 0:
        return "Invalid kspacing, should be greater than 0"
    if kspacing > 0.5:
        return "Too Coarse kspacing, should be less than 0.5"
    ## Calculate the kpoints
    kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in cell
        ]
    return kpoints

@tool
def get_kspacing_ecutwfc(threshold: float = 1.0) -> str:
    '''Read the convergen test result and determine the kspacing and ecutwfc used in the production
    Input:  threshold: float , the threshold mev/atom to determine the convergence
    output: str, the kspacing and ecutwfc used in the production
    '''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    job_list = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    with open(job_list,"r") as file:
        job_dict = json.load(file)
        job_list = job_dict['job_list']
    ### Find the kpoints and ecutwfc from the output file
    kspacing = []
    ecutwfc = []
    energy_list = []
    Natom = None
    for job in job_list:
        ## Read the output file
        print(f'reading {job}')
        kspacing.append(job_dict[job]['k'])
        ecutwfc.append(job_dict[job]['ecutwfc'])
        
        atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
        energy = atom.get_potential_energy()
        energy_list.append(energy)
        Natom = atom.get_number_of_atoms()
        
    convergence_df = pd.DataFrame({'job':job_list,'kspacing':kspacing, 'ecutwfc':ecutwfc, 'energy':energy_list})
    ## Save the convergence test result if file exist then append to it
    if os.path.exists(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv')):
        convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'), mode='a', header=False)
    else:
        convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'))
    
    ## Determine the kpoints and ecutwfc based on the threshold
    k_chosen, ecutwfc_chosen = select_k_ecut(convergence_df, threshold, Natom)

    return f"Please use kspacing {k_chosen} and ecutwfc {ecutwfc_chosen} for the production calculation"

@tool
def dummy_structure(concentration: float,
                    scale_factor: float) -> AtomsDict:
    """Returns a crystal structure with a given concentration of Cu atoms and the rest Au atoms, and a scale factor for the cell size."""  
    atoms = FaceCenteredCubic("Cu", latticeconstant=3.58)
    atoms *= (1,1,2)
    # Calculate the number of Cu atoms to replace
    num_atoms_to_replace = int((1.0-concentration) * len(atoms))
    # Randomly select indices to replace
    indices_to_replace = np.random.choice(len(atoms), num_atoms_to_replace, replace=False)
    atoms.numbers[indices_to_replace] = 79
    # scaleFactor = (1.0 - concentration) * (6.5 - 3.58) / 3.58 + 1
    # scaleFactor = 1.0
    atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)

    return atoms.todict()

@tool
def write_script(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "Name of the file to be saved."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the quantum espresso input file to the specified file path"""
    ## Error when '/' in the content, manually delete
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")

    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    path = os.path.join(WORKING_DIRECTORY, f'{file_name}')

    ## If content ends with '/' then remove it
    if content.endswith('/'):
        content = content[:-1]
    
    with open(path,"w",encoding="ascii") as file:
        file.write(content)
    
    os.environ['INITIAL_FILE'] = file_name
    return f"Initial file is created named {file_name}"

@tool
def calculate_lc() -> str:
    """Read the output file and calculate the lattice constant"""
    
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    job_list = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    with open(job_list,"r") as file:
        print('reading job list')
        job_json = json.load(file)

    volume_list = []
    energy_list = []
    for job in job_json['job_list']:
        print(f'reading {job}')
        atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
        volume_list.append(atom.get_volume())
        energy_list.append(atom.get_potential_energy())
        print(f'{job} volume is {atom.get_volume()}, energy is {atom.get_potential_energy()}')
    eos = EquationOfState(volume_list, energy_list)
    v0, e0, B = eos.fit()
    lc = (v0)**(1/3)

    # Check if the json file exists
    json_file = os.path.join(WORKING_DIRECTORY, '../lattice_constant.json')
    if not os.path.exists(json_file):
        with open(json_file, "w") as file:
            json.dump({}, file)

    # Load the existing dictionary from the json file
    with open(json_file, "r") as file:
        try:
            lc_dict = json.load(file)
        except:
            lc_dict = {}

    # Update the dictionary with the new lattice constant
    lc_dict[str(atom.symbols)] = lc

    # Save the updated dictionary back to the json file
    with open(json_file, "w") as file:
        json.dump(lc_dict, file)

    return f'The lattice constant is {lc}'


@tool
def find_pseudopotential(element: str) -> str:
    """Return the pseudopotential file path for given element symbol."""
    pseudo_dir = os.environ.get("PSEUDO_DIR")
    for roots, dirs, files in os.walk(f'{pseudo_dir}'):
        for file in files:
            if element == file.split('.')[0].split('_')[0].capitalize():
                return f'The pseudopotential file for {element} is {file} under {pseudo_dir}'
    return f"Could not find pseudopotential for {element}"


@tool
def get_bulk_modulus(
    working_directory: str,
    pseudo_dir: str,
    input_file: str,
) -> float:
    '''Calculate the bulk modulus of the given quantum espresso input file, pseudopotential directory and working directory'''
    atoms = read(os.path.join(working_directory,input_file))
    with open(os.path.join(working_directory,input_file),'r') as file:
        content = file.read()
    input_data = parse_qe_input_string(content)
    pseudopotentials = filter_potential(input_data)

    profile = EspressoProfile(command='mpiexec -n 8 pw.x', pseudo_dir=pseudo_dir)

    atoms.calc = Espresso(
    profile=profile,
    pseudopotentials=pseudopotentials,
    input_data=input_data
)

    # run variable cell relax first to make sure we have optimum scaling factor
    # ecf = ExpCellFilter(atoms)
    # dyn = FIRE(ecf)
    # traj = Trajectory(os.path.join(working_directory,'relax.traj'), 'w', atoms)
    # dyn.attach(traj)
    # dyn.run(fmax=1.5)

    # now we calculate eos
    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    bulk_modulus = B / kJ * 1.0e24

    return bulk_modulus


@tool
def get_lattice_constant(
    working_directory: str,
    pseudo_dir: str,
    input_file: str,
) -> float:
    '''Calculate the lattice constant of the given quantum espresso input file, pseudopotential directory and working directory'''
    atoms = read(os.path.join(working_directory,input_file))
    with open(os.path.join(working_directory,input_file),'r') as file:
        content = file.read()
    input_data = parse_qe_input_string(content)
    pseudopotentials = filter_potential(input_data)

    profile = EspressoProfile(command='mpiexec -n 2 pw.x', pseudo_dir=pseudo_dir)

    atoms.calc = Espresso(
    profile=profile,
    pseudopotentials=pseudopotentials,
    input_data=input_data
)

    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    lc = (v)**(1/3)
    print(f'{input_file} lattice constant is {lc}')
    with open(os.path.join(working_directory,input_file.split('.')[0]+'.out'),'w') as file:
        file.write(f'\n# {input_file} Lattice constant is {lc}')
    return lc

@tool
def generate_convergence_test(input_file_name:str,kspacing:list[float], ecutwfc:list[int]):
    '''
    Generate the convergence test input scripts for quantum espresso calculation and save the job list. 

    Input:  input_file_name: str, the name of the input file
            kspacing: list[float], the list of kspacing to be tested
            ecutwfc: list[int], the list of ecutwfc to be tested
    '''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
    # Read the atom object from the input script
    try:
        atom = read(input_file)
    except:
        INITIAL_FILE = os.environ.get("INITIAL_FILE")
        return f"Invalid input file, do you want to use {INITIAL_FILE} as the input file?"
    
    cell = atom.cell
    ecutwfc_max = max(ecutwfc)
    kspacing_min = min(kspacing)
    job_list_dict = {}
    job_list = []
    # Generate the input script for highest ecutwfc different kspacing
    for k in kspacing:
        kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / k).astype(int)) // 2 + 1) for ii in cell
        ]
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                ## Change the prefix of the output file
                # if 'outdir' in line:
                #     lines[i] = f"    outdir = './out_k_{k}_ecutwfc_{ecutwfc_max}'\n"

                ## Find the ecutwfc line
                if 'ecutwfc' in line:
                    lines[i] = f'    ecutwfc = {ecutwfc_max},\n'
                if 'ecutrho' in line:
                    lines[i] = f"    ecutrho = {ecutwfc_max*8},\n"
                
                ## Find the kpoints line
                if 'K_POINTS' in line:
                    lines[i+1] = ' '.join(map(str,kpoints)) +' 0 0 0' +'\n'

            ## Write the new input script
            new_file_name = f'{os.path.splitext(input_file_name)[0]}_k_{k}_ecutwfc_{ecutwfc_max}.in'
            print(new_file_name)
            job_list_dict[new_file_name] = {'k':k, 'ecutwfc':ecutwfc_max}
            new_input_file = os.path.join(WORKING_DIRECTORY, new_file_name)
            job_list.append(new_file_name)
            with open(new_input_file, 'w') as f:
                f.writelines(lines)
    # Generate the input script for highest kspacing different ecutwfc
    for e in ecutwfc:
        kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing_min).astype(int)) // 2 + 1) for ii in cell
        ]
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if 'outdir' in line:
                #     lines[i] = f"    outdir = './out_k_{kspacing_min}_ecutwfc_{e}',\n"
                ## Find the ecutwfc line
                if 'ecutwfc' in line:
                    lines[i] = f'    ecutwfc = {e},\n'
                if 'ecutrho' in line:
                    lines[i] = f"    ecutrho = {e*8},\n"
                
                ## Find the kpoints line
                if 'K_POINTS' in line:
                    lines[i+1] = ' '.join(map(str,kpoints)) +' 0 0 0' +'\n'

            ## Write the new input script
            new_file_name = f'{os.path.splitext(input_file_name)[0]}_k_{kspacing_min}_ecutwfc_{e}.in'
            job_list_dict[new_file_name] = {'k':kspacing_min, 'ecutwfc':e}
            new_input_file = os.path.join(WORKING_DIRECTORY, new_file_name)
            job_list.append(new_file_name)
            with open(new_input_file, 'w') as f:
                f.writelines(lines)
    ## Remove duplicate files
    job_list = list(set(job_list))
    ## Save the job list as json file
    job_list_dict['job_list'] = job_list
    with open(os.path.join(WORKING_DIRECTORY, 'job_list.json'), 'w') as f:
        json.dump(job_list_dict, f)
    
    return f"Job list is saved scucessfully, continue to submit the jobs"


@tool
def generate_eos_test(input_file_name:str,kspacing:float, ecutwfc:int):
    '''
    Generate the equation of state test input scripts for quantum espresso calculation and save the job list.
    
    Input:  input_file_name: str, the name of the input file
            kspacing: float, the kspacing to be tested
            ecutwfc: int, the ecutwfc to be tested
    '''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
    prefix = input_file_name.split('.')[0]
    # Read the atom object from the input script
    try:
        atom = read(input_file)
    except:
        INITIAL_FILE = os.environ.get("INITIAL_FILE")
        return f"Invalid input file, try to use {INITIAL_FILE} as the input file?"
    job_list = []
    
    cell = atom.cell
    ## Calculate the kpoints
    kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in cell
        ]
    
    for scale in np.linspace(0.95, 1.05, 5):
        # Read the input script
        with open(input_file, 'r') as f:
            lines = f.readlines()
        # Update the scale
        for i, line in enumerate(lines):
            # if 'outdir' in line:
            #     lines[i] = f"    outdir = './out_{scale}'\n"

            if 'ecutwfc' in line:
                lines[i] = f"    ecutwfc = {ecutwfc},\n"
            if 'ecutrho' in line:
                lines[i] = f"    ecutrho = {ecutwfc*8},\n"
            if 'CELL_PARAMETERS' in line:
                lines[i+1] = f"{cell[0][0]*scale} {cell[0][1]*scale} {cell[0][2]*scale}\n"
                lines[i+2] = f"{cell[1][0]*scale} {cell[1][1]*scale} {cell[1][2]*scale}\n"
                lines[i+3] = f"{cell[2][0]*scale} {cell[2][1]*scale} {cell[2][2]*scale}\n"
                
            if 'K_POINTS' in line:
                lines[i+1] = f"{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n"
    
        ## New input file name
        new_file_name = f"{prefix}_{scale}.in"
        job_list.append(new_file_name)
        new_file = os.path.join(WORKING_DIRECTORY, new_file_name)
        with open(new_file, 'w') as f:
            f.writelines(lines)
    ## Remove duplicate files
    job_list = list(set(job_list))
    ## Save the job list as json file
    job_list_dict = {'job_list':job_list}
    with open(os.path.join(WORKING_DIRECTORY, 'job_list.json'), 'w') as f:
        json.dump(job_list_dict, f)
    
    return f"Job list is saved scucessfully, continue to submit the jobs"

# @tool
# def generate_eos_test(input_file_name:str,kspacing:float, ecutwfc:int):
#     '''
#     Generate the equation of state test input scripts for quantum espresso calculation and save the job list.
    
#     Input:  input_file_name: str, the name of the input file
#             kspacing: float, the kspacing to be tested
#             ecutwfc: int, the ecutwfc to be tested
#     '''
#     WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
#     input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
#     # Read the atom object from the input script
#     try:
#         atom = read(input_file)
#     except:
#         return "Invalid input file, please check the file name"
#     job_list = []
    
#     cell = atom.cell
#     ## Calculate the kpoints
#     kpoints = [
#             2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in cell
#         ]
    
#     for scale in np.linspace(0.9, 1.1, 5):
#         # Read the input script
#         with open(input_file, 'r') as f:
#             lines = f.readlines()
#         # Update the scale
#         for i, line in enumerate(lines):
#             if 'outdir' in line:
#                 lines[i] = f"outdir = '    ./out_{scale}'\n"

#             if 'ecutwfc' in line:
#                 lines[i] = f"    ecutwfc = {ecutwfc},\n"
#             if 'CELL_PARAMETERS' in line:
#                 lines[i+1] = f"{cell[0][0]*scale} {cell[0][1]*scale} {cell[0][2]*scale}\n"
#                 lines[i+2] = f"{cell[1][0]*scale} {cell[1][1]*scale} {cell[1][2]*scale}\n"
#                 lines[i+3] = f"{cell[2][0]*scale} {cell[2][1]*scale} {cell[2][2]*scale}\n"
                
#             if 'K_POINTS' in line:
#                 lines[i+1] = f"{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n"
    
#         ## New input file name
#         new_file_name = f"Li_bcc_{scale}.in"
#         job_list.append(new_file_name)
#         new_file = os.path.join(WORKING_DIRECTORY, new_file_name)
#         with open(new_file, 'w') as f:
#             f.writelines(lines)
#     ## Remove duplicate files
#     job_list = list(set(job_list))
#     ## Save the job list as json file
#     job_list_dict = {'job_list':job_list}
#     with open(os.path.join(WORKING_DIRECTORY, 'job_list.json'), 'w') as f:
#         json.dump(job_list_dict, f)
    
#     return f"Job list is saved scucessfully, continue to submit the jobs"

@tool
def save_job_list(
    script_list: Annotated[list[str], "List of scripts to be calculated."]
) -> Annotated[str, "Path of the saved json file."]:
    """Save the list of quantum espresso input files to the specified json file path"""
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    path = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    job = {'job_list':script_list}
    with open(path,"w") as file:
        json.dump(job, file)
    return f"Job list saved to {path}"

### HPC tools
@tool
def find_job_list() -> str:
    """Find the job list from the specified json file path"""
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    path = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    with open(path,"r") as file:
        job_json = json.load(file)
    job_list = job_json['job_list']
    return f'The files need to be submitted are {job_list}. Please continue to submit the job.'


@tool
def read_script(
    WORKING_DIRECTORY: Annotated[str, "The working directory."],
    input_file: Annotated[str, "The input file to be read."]
) -> Annotated[str, "read content"]:
    """read the quantum espresso input file from the specified file path"""
    ## Error when '/' in the content, manually delete
    path = os.path.join(WORKING_DIRECTORY, input_file)
    with open(path,"r") as file:
        content = file.read()
    return content

@tool
def add_resource_suggestion(
    qeInputFileName: str,
    partition: str,
    nnodes: int,
    ntasks: int,
    runtime: Annotated[str, "Time limit for the job, in minutes"],
) -> Annotated[str, "source suggestion saved location"]:
    """
    After agent generate resource suggestions based on the QE input file, add it to the json file "resource_suggestions.json" in the WORKING_DIRECTORY.
    For example: {"input1.pwi": {"nnodes": 2, "ntasks": 4, "runtime": 60}, "input2.pwi": {"nnodes": 1, "ntasks": 2, "runtime": 30}}
    """
    if not isinstance(partition, str) or not isinstance(nnodes, int) or not isinstance(ntasks, int) or not isinstance(runtime, str):
        return "Invalid input, please check the input format"
    # craete the json file if it does not exist, otherwise load it
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")

    new_resource_dict = {qeInputFileName: {"partition": partition, "nnodes": 1, "ntasks": 4, "runtime": 30}}

    
    # check if resource_suggestions.db exist in the working directory
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if not os.path.exists(db_file):
        initialize_database(db_file)
    
    add_to_database(new_resource_dict, db_file)
    
    return f"Resource suggestion for {qeInputFileName} saved scucessfully"

@tool
def submit_and_monitor_job() -> str:
    '''Submit jobs in the job list to supercomputer, return the location of the output file once the job is done'''
    
    # check if resource_suggestions.json exist
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if not os.path.exists(resource_suggestions):
        return "Resource suggestion file not found, please use the add_resource_suggestion tool to add the resource suggestion"
    
    print("checking pysqa prerequisites...")
    # check if slurm.sh and queue.yaml exist in the working directory
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, "slurm.sh")) or not os.path.exists(os.path.join(WORKING_DIRECTORY, "queue.yaml")):
        print("Creating pysqa prerequisites...")
        create_pysqa_prerequisites(WORKING_DIRECTORY)
    
    qa = QueueAdapter(directory=WORKING_DIRECTORY)
        
    # load jobs frm job_list.json
    job_list_dir = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    with open(job_list_dir,"r") as file:
        job_list = json.load(file)['job_list']
    
    # load reousrce suggestions
    # resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.json')
    # with open(resource_suggestions, "r") as file:
    #     resource_dict = json.load(file)
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query all rows from the resources table
    cursor.execute('SELECT * FROM resources')
    rows = cursor.fetchall()

    # Reconstruct the original dictionary
    resource_dict = {}
    for row in rows:
        filename, partition, nnodes, ntasks, runtime = row
        resource_dict[filename] = {
            'partition': partition,
            'nnodes': nnodes,
            'ntasks': ntasks,
            'runtime': runtime
        }
        
    conn.close()
    print(f"loaded resource suggestions: {json.dumps(resource_dict, indent=4)}")
    
    ## Check resource key is valid
    for job in job_list:
        if job not in resource_dict.keys():
            return f"Resource suggestion for {job} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    
    print(f"loaded {len(job_list)} jobs from job_list.json, and {len(resource_dict)} resource suggestions from resource_suggestions.json")
    
    queueIDList = []
    while True:
        for inputFile in job_list:    
            
            ## Check if the input file exists
            if not os.path.exists(os.path.join(WORKING_DIRECTORY, inputFile)):
                return f"Input file {inputFile} does not exist, please use the find job list tool to submit the file in the job list"
            print("Generating batch script...")

            ## Check if the output file exists 
            outputFile = f"{inputFile}.pwo"
            if os.path.exists(os.path.join(WORKING_DIRECTORY, outputFile)):
                ## Supervisor sometimes ask to submit the job again, so we need to check if the output file exists
                try:
                    tmp = read(os.path.join(WORKING_DIRECTORY, outputFile))
                    _ = tmp.get_potential_energy()
                    print(f"Output file {inputFile}.pwo already exists, the calculation is done")
                    continue
                except:
                    print("output file exists but the calculation is not done, will resubmit the job")
                    
            
            
            job_id = qa.submit_job(
            working_directory=WORKING_DIRECTORY,
            cores=resource_dict[inputFile]['ntasks'],
            memory_max=2000,
            queue="slurm",
            job_name="agent_job",
            cores_max=resource_dict[inputFile]['ntasks'],
            nodes_max=resource_dict[inputFile]['nnodes'],
            partition=resource_dict[inputFile]['partition'],
            run_time_max=resource_dict[inputFile]['runtime'],
            command =f"""
export OMP_NUM_THREADS=1

spack load quantum-espresso@7.2

echo "Job started on `hostname` at `date`"

mpirun pw.x < {inputFile} > {inputFile}.pwo

echo " "
echo "Job Ended at `date`"
        """
            )
            
            if job_id is None:
                return "Job submission failed"

            queueIDList.append(job_id)
            ## Sleep for 1.5 second to avoid the job submission too fast
            time.sleep(5)
            
            #  Change the bash script name to avoid the job submission too fast
            os.rename(os.path.join(WORKING_DIRECTORY, "run_queue.sh"), os.path.join(WORKING_DIRECTORY, f"slurm_{inputFile}.sh"))
            time.sleep(5)
        
        prevCount = len(queueIDList)
        while True:
            count = 0
            print("waiting for", end=" ")
            for queueID in queueIDList:
                if qa.get_status_of_job(process_id=queueID):
                    count += 1
                    print(queueID, end=" ")
            print("to finish", end="\r")
            
            if count < prevCount:
                print()
                prevCount = count
            if count == 0:
                break
            time.sleep(1)
            
        print(f"All job in job_list has finished")
        print("waiting for files...")
        time.sleep(10)
            
        print("Checking jobs")
        
        checked = set()
        unchecked = set(job_list)
        while checked != unchecked:
            for inputFile in job_list:
                outputFile = f"{inputFile}.pwo"
                print(f"Checking job {inputFile}")
                checked.add(inputFile)
                try:
                    atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
                    print(atoms.get_potential_energy())
                    # delete inputFile from job_list
                    job_list.remove(inputFile)
                    print(f"Job list: {job_list}")
                    print()
                except:
                    # if outputFile exsit remove outputFile
                    try:
                        os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
                        print(f"{outputFile} removed")
                    except:
                        print("output file does not exist")
                    print(f"Job {inputFile} failed, will resubmit the job")
        
        
        # for idx, inputFile in enumerate(job_list):
        #     outputFile = f"{inputFile}.pwo"
        #     print(f"Checking job {inputFile}")
        #     try:
        #         atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
        #         print(atoms.get_potential_energy())
        #         # delete inputFile from job_list
        #         job_list.remove(inputFile)
        #         print(f"Job list: {job_list}")
        #         print()
        #     except:
        #         # remove outputFile
        #         os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
        #         print(f"Job {inputFile} failed, will resubmit the job")
                
        if len(job_list) == 0:
            # load jobs frm job_list.json
            job_list_dir = os.path.join(WORKING_DIRECTORY, f'job_list.json')
            with open(job_list_dir,"r") as file:
                job_list = json.load(file)['job_list']
            
            # read all energies into a dict
            energies = {}
            for inputFile in job_list:
                outputFile = f"{inputFile}.pwo"
                atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
                energies[inputFile] = atoms.get_potential_energy()
            
            job_list = []
            
            # check two or more key has the same value, if so, add the key back to the job_list
            for key, value in energies.items():
                if list(energies.values()).count(value) > 1:
                    print(f"!!!!!!!Job {key} has the same energy as other jobs, may resubmit the job!!!!!!!!")
                    job_list.append(key)
            
            print()
            # check whether job in job_list has the same inputFile content, if so, remove the job from job_list
            tobeRemoved = np.zeros(len(job_list))
            for jobIdx in range(len(job_list)):
                for jobIdx2 in range(jobIdx+1, len(job_list)):
                    if cmp(os.path.join(WORKING_DIRECTORY, job_list[jobIdx]), os.path.join(WORKING_DIRECTORY, job_list[jobIdx2]), shallow=False):
                        print(f"!!!!!!!Job {job_list[jobIdx]} has the same content as {job_list[jobIdx2]}, will remove the job!!!!!!!!")
                        tobeRemoved[jobIdx] = 1
                        tobeRemoved[jobIdx2] = 1
            
            job_list = [job_list[i] for i in range(len(job_list)) if tobeRemoved[i] == 0]
            
            print("##########")
            print(f"Final jobs to be resubmitted: {job_list}")
            print("##########")
            
            # remove outputFile for jobs in job_list
            for inputFile in job_list:
                outputFile = f"{inputFile}.pwo"
                print(f"Removing {outputFile}")
                os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
        
            if len(job_list) == 0:
                break
            

    return f"All job in job_list has finished, please check the output file in the {WORKING_DIRECTORY}"

@tool
def submit_single_job(
    inputFile: str
) -> str:
    '''Submit a single job to supercomputer, return the location of the output file once the job is done'''
    print("checking pysqa prerequisites...")
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    # check if slurm.sh and queue.yaml exist in the working directory
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, "slurm.sh")) or not os.path.exists(os.path.join(WORKING_DIRECTORY, "queue.yaml")):
        print("Creating pysqa prerequisites...")
        create_pysqa_prerequisites(WORKING_DIRECTORY)
    
    qa = QueueAdapter(directory=WORKING_DIRECTORY)
        
    
    # load reousrce suggestions
    resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.json')
    with open(resource_suggestions, "r") as file:
        resource_dict = json.load(file)
    
    ## Check resource key is valid
    
    if inputFile not in resource_dict.keys():
        return f"Resource suggestion for {inputFile} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    

    
    queueIDList = []


    ## Check if the input file exists
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, inputFile)):
        return f"Input file {inputFile} does not exist, please use the find job list tool to submit the file in the job list"
    print("Generating batch script...")

    ## Check if the output file exists 
    if os.path.exists(os.path.join(WORKING_DIRECTORY, f"{inputFile}.pwo")):
        ## Supervisor sometimes ask to submit the job again, so we need to check if the output file exists
        return f"Output file {inputFile}.pwo already exists, the calculation is done"
        
        
    job_id = qa.submit_job(
        working_directory=WORKING_DIRECTORY,
        cores=resource_dict[inputFile]['ntasks'],
        memory_max=2000,
        queue="slurm",
        job_name="agent_job",
        cores_max=resource_dict[inputFile]['ntasks'],
        nodes_max=resource_dict[inputFile]['nnodes'],
        partition=resource_dict[inputFile]['partition'],
        run_time_max=resource_dict[inputFile]['runtime'],
        command =f"""
export OMP_NUM_THREADS=1

spack load quantum-espresso@7.2

echo "Job started on `hostname` at `date`"

mpirun pw.x < {inputFile} > {inputFile}.pwo

echo " "
echo "Job Ended at `date`"
    """
        )
        
    if job_id is None:
        return "Job submission failed"

    queueIDList.append(job_id)
    
    
    prevCount = len(queueIDList)
    while True:
        count = 0
        print("waiting for", end=" ")
        for queueID in queueIDList:
            if qa.get_status_of_job(process_id=queueID):
                count += 1
                print(queueID, end=" ")
        print("to finish", end="\r")
        
        if count < prevCount:
            print()
            prevCount = count
        if count == 0:
            break
        time.sleep(1)
        
    print(f"Job has finished")

    return f"Job has finished, please check the output file"   

@tool
def read_energy_from_output(
) -> str:
    '''Read the total energy from the output file in job list and return it in a string'''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    # load job_list.jason
    job_list_dir = os.path.join(WORKING_DIRECTORY, f'job_list.json')
    with open(job_list_dir,"r") as file:
        job_list = json.load(file)['job_list']
        
    result = ""
    for job in job_list:
        
        output_file = job + '.pwo'
        # print(f"Reading output file {output_file}")
        file_path = os.path.join(WORKING_DIRECTORY, output_file)
        # print(file_path)
        # Check if the output file exists
        if not os.path.exists(file_path):
            return f"Output file {output_file} does not exist, please check the job list"
        try:
            atoms = read(file_path)
        except:
            return f"Invalid output file {output_file} or calculation failed, please submit the {job} again."
        result += f"Energy read from {job} is {atoms.get_potential_energy()}. "
        # print(result)
        time.sleep(1)
    print(result)
    # check input file in job list
    # file_path = os.path.join(WORKING_DIRECTORY, input_file)
    # atoms = read(file_path)
    # return f"Energy read from job {input_file} is {atoms.get_potential_energy()}"
        
    return result


@tool
def read_single_output(
    input_file: str
) -> str:
    '''Read the total energy from the file in job list and return it in a string'''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    # load job_list.jason
    output_file = input_file + '.pwo'
    file_path = os.path.join(WORKING_DIRECTORY, output_file)
    # print(file_path)
    # Check if the output file exists
    if not os.path.exists(file_path):
        return f"Output file {output_file} does not exist, please check the job list"
    try:
        atoms = read(file_path)
    except:
        return f"Invalid output file {output_file} or calculation failed, please submit the {input_file} again."
    return f"Energy read from job {input_file} is {atoms.get_potential_energy()}"