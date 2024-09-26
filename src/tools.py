
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
from pysqa import QueueAdapter

@tool
def get_kpoints(atom_dict: AtomsDict, kspacing: float) -> str:
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
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    path = os.path.join(WORKING_DIRECTORY, f'{file_name}')
    with open(path,"w") as file:
        file.write(content)
    return f"Document saved to {path}"

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
   


def create_pysqa_prerequisites(WORKING_DIRECTORY: str):
    '''Create the pysqa prerequisites in the working directory'''
    with open(os.path.join(WORKING_DIRECTORY, "slurm.sh"), "w") as file:
        file.write(r"""#!/bin/bash
#SBATCH -J {{job_name}} # Job name
#SBATCH -n {{cores_max}} # Number of total cores
#SBATCH -N {{nodes_max}} # Number of nodes
#SBATCH --time={{run_time_max | int}}
#SBATCH -p {{partition}}
#SBATCH --mem-per-cpu={{memory_max}}M # Memory pool for all cores in MB
#SBATCH -e sqa.err #change the name of the err file 
#SBATCH -o sqa.out # File to which STDOUT will be written %j is the job #

{{command}}

                   """)
        
    with open(os.path.join(WORKING_DIRECTORY, "queue.yaml"), "w") as file:
        file.write(r"""queue_type: SLURM
queue_primary: slurm
queues:
  slurm: {
    job_name: testPysqa,
    cores_max: 4, 
    cores_min: 1, 
    nodes_max: 1,
    memory_max: 2000,
    partition: venkvis-cpu,
    script: slurm.sh
    }
                   """)


@tool
def generate_submit_and_monitor_job(
    WORKING_DIRECTORY: str,
    partition: str,
    nnodes: int,
    natom: int,
    runtime: Annotated[str, "Time limit for the job, in minutes"],
    inputFile: str,
) -> str:
    '''Generate a slurm sbatch submission script for quantum espresso with given parameters, submit the quantum espresso job to HPC, monitor the progress, and return the location of the output file once the job is done'''
    print("checking pysqa prerequisites...")
    # check if slurm.sh and queue.yaml exist in the working directory
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, "slurm.sh")) or not os.path.exists(os.path.join(WORKING_DIRECTORY, "queue.yaml")):
        print("Creating pysqa prerequisites...")
        create_pysqa_prerequisites(WORKING_DIRECTORY)
    print("Generating batch script...")
    qa = QueueAdapter(directory=WORKING_DIRECTORY)
    
    job_id = qa.submit_job(
    working_directory=WORKING_DIRECTORY,
    cores=8,
    memory_max=2000,
    queue="slurm",
    job_name="hh",
    cores_max=natom,
    nodes_max=nnodes,
    partition=partition,
    run_time_max=runtime,
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
    
    while qa.get_status_of_job(process_id=job_id):
        print(f"Waiting for job {job_id} to finish...")
        time.sleep(30)
    print(f"Job {job_id} finished")

    outputFile = f"{inputFile}.pwo"
    return f"Batch script saved as run_queue.sh, Job finished successfully, the output file is avaible at {os.path.join(WORKING_DIRECTORY, outputFile)}"


@tool
def read_energy_from_output(
    WORKING_DIRECTORY: Annotated[str, "The working directory."],
    output_file: Annotated[str, "The output file to be read."]
) -> str:
    '''Read the total energy from the quantum espresso job output file and return it in a string'''
    atoms = read(os.path.join(WORKING_DIRECTORY, output_file))
    return f"the energy is {atoms.get_potential_energy()} eV"


