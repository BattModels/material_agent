
from src.utils import AtomsDict,element_list
from ase import Atoms, Atom
from langchain.agents import tool
import os 
from typing import Annotated,Dict, Literal,Optional
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic
import ast
import re

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
    """Returns a crystal structure with a given concentration of Cu atoms"""
    atoms = FaceCenteredCubic("Cu", latticeconstant=3.58)
    atoms *= (1,1,2)
    # Calculate the number of Cu atoms to replace
    num_atoms_to_replace = int((1.0-concentration) * len(atoms))
    # Randomly select indices to replace
    indices_to_replace = np.random.choice(len(atoms), num_atoms_to_replace, replace=False)
    atoms.numbers[indices_to_replace] = 79
    scaleFactor = concentration * (6.5 - 3.58) / 3.58 + 1
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





