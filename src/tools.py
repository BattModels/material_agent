
from fileinput import filename
from matplotlib.pyplot import sca
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
from ase.eos import calculate_eos
from ase.units import kJ
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.io.trajectory import Trajectory
import subprocess
import time

@tool
def get_kpoints(atom_dict: AtomsDict, k_point_distance: str) -> str:
    """Returns the kpoints of a given ase atoms object and user specified k_point_distance (k_point_distance could be fine, normal, coarse or very fine, default is normal)"""
    atoms = Atoms(**atom_dict.dict())
    cell = atoms.cell
    if 'fine' in k_point_distance and 'very' not in k_point_distance:
        kspacing = 0.2
    elif 'very' in k_point_distance and 'fine' in k_point_distance:
        kspacing = 0.1
    elif 'coarse' in k_point_distance:
        return [1,1,1]
    else:
        kspacing = 0.3
    kpoints = [
        (np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) for ii in cell
    ]
    return kpoints

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
    WORKING_DIRECTORY: Annotated[str, "The working directory."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the quantum espresso input file to the specified file path"""
    ## Error when '/' in the content, manually delete
    path = os.path.join(WORKING_DIRECTORY, f'{file_name}')
    with open(path,"w") as file:
        file.write(content)
    return f"Document saved to {path}"

@tool
def read_script(
    WORKING_DIRECTORY: Annotated[str, "The working directory."]
) -> Annotated[str, "read content"]:
    """read the quantum espresso input file from the specified file path"""
    ## Error when '/' in the content, manually delete
    path = os.path.join(WORKING_DIRECTORY, 'input.in')
    with open(path,"r") as file:
        content = file.read()
    return content

@tool
def find_pseudopotential(element: str,
                         pseudo_dir: str) -> str:
    """Return the pseudopotential file path for given element symbol and directory."""
    
    for roots, dirs, files in os.walk(f'{pseudo_dir}'):
        for file in files:
            if element == file.split('.')[0].split('_')[0].capitalize():
                return file
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
def generate_batch_script(
    partition: str,
    nnodes: int,
    ntasks: int,
    time: str,
    inputFile: str,
) -> str:
    '''Generate a slurm sbatch submission script for quantum espresso with given parameters'''
    print("Generating batch script...")
    batchScript = f"""#!/bin/bash
#SBATCH -J agentJob # Job name
#SBATCH -n {ntasks} # Number of total cores
#SBATCH -N {nnodes} # Number of nodes
#SBATCH --time={time}
#SBATCH -p {partition}
#SBATCH --mem-per-cpu=2000M # Memory pool for all cores in MB
#SBATCH -e out/err.err #change the name of the err file 
#SBATCH -o out/out.out # File to which STDOUT will be written %j is the job #

export OMP_NUM_THREADS=1

spack load /qn6ee2y

echo "Job started on `hostname` at `date`"

mpirun pw.x < out/{inputFile} > out/{inputFile}.pwo

echo " "
echo "Job Ended at `date`"
    """
    with open("out/run.sh", "w") as file:
        file.write(batchScript)
    
    return "Batch script saved as run.sh"


# Function to check if a job is still running
def is_job_running(job_id):
    # Run a command and capture its output
    result = subprocess.run(f"squeue --job {job_id}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Get the standard output
    output = result.stdout
    return job_id in output  # If job_id is found in squeue output, job is running

# Function to wait for all jobs to finish
def wait_for_jobs(job_id):
    while is_job_running(job_id):
        print(f"Waiting for job {job_id} to finish...")
        time.sleep(30)
    print(f"Job {job_id} finished")


@tool
def submit_and_monitor_job(
    WORKING_DIRECTORY: Annotated[str, "The working directory."],
    slurmScript: Annotated[str, "The slurm script to be submitted."]
) -> str:
    '''submit the quantum espresso job to HPC, monitor the progress, and return the result once the job is done'''
    
    path = os.path.join(WORKING_DIRECTORY, slurmScript)
    
    print(f"submitting {path}")
    # Run a command and capture its output
    result = subprocess.run("sbatch out/run.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # print(f"submission result:")
    # Get the standard output
    output = result.stdout
    
    print(output)
    
    # # find the first job's ID
    # jobID = re.search(r'^\s*(\d+)', output, re.MULTILINE)
    # jobID = jobID.group(1)
    jobID = output.split()[-1]
    
    # print(f"waiting for job {jobID} to finish")
    wait_for_jobs(jobID)
    
    return "Job submitted successfully"




