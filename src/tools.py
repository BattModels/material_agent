
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
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory

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
def dummy_structure(concentration: float) -> AtomsDict:
    """Returns a crystal structure with a given concentration of Cu atoms and the rest Au atoms"""  
    atoms = FaceCenteredCubic("Cu", latticeconstant=3.58)
    atoms *= (1,1,2)
    # Calculate the number of Cu atoms to replace
    num_atoms_to_replace = int((1.0-concentration) * len(atoms))
    # Randomly select indices to replace
    indices_to_replace = np.random.choice(len(atoms), num_atoms_to_replace, replace=False)
    atoms.numbers[indices_to_replace] = 79
    scaleFactor = (1.0 - concentration) * (6.5 - 3.58) / 3.58 + 1
    # scaleFactor = 1.0
    atoms.set_cell(atoms.cell * scaleFactor, scale_atoms=True)

    return atoms.todict()

@tool
def write_script(
    content: Annotated[str, "Text content to be written into the document."],
    WORKING_DIRECTORY: Annotated[str, "The working directory."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the quantum espresso input file to the specified file path"""
    ## Error when '/' in the content, manually delete
    path = os.path.join(WORKING_DIRECTORY, 'input.in')
    with open(path,"w") as file:
        file.write(content)
    return f"Document saved to {path}"

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
    ecf = ExpCellFilter(atoms)
    dyn = BFGS(ecf)
    traj = Trajectory('relax.traj', 'w', atoms)
    dyn.attach(traj)
    dyn.run(fmax=0.05)

    # now we calculate eos
    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    bulk_modulus = B / kJ * 1.0e24

    return bulk_modulus
   





