from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
from math import e
from src.utils import *
from src.postProcessUtils import post_0_1_process
from src.myCANVAS import CANVAS
from ase import Atoms, Atom
from ase.data import atomic_numbers
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
import os 
from typing import Annotated, Dict, Literal, Optional, Sequence, Tuple, Any
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic
import ast
import re
import io
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.eos import calculate_eos,EquationOfState
from ase.units import kJ
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE
from ase.io.trajectory import Trajectory
from ase.io.lammpsdata import write_lammps_data
from ase.build import bulk, surface, add_adsorbate
import ase.build
from ase import Atoms
import subprocess
import time
from pysqa import QueueAdapter
import json
import pandas as pd
import sqlite3
from filecmp import cmp
import contextlib
from autocat.surface import generate_surface_structures
from autocat.adsorption import get_adsorption_sites, get_adsorbate_height_estimate
from src import var
from pymatgen.io.ase import AseAtomsAdaptor

##################################################################################################
##                                        Common tools                                          ##
##################################################################################################
@tool
def inspect_my_canvas():
    """Inspect the working canvas to get available keys"""
    # get all keys in myCANVAS and return them as a list [key1, key2, ...]
    return CANVAS.inspect()

@tool
def read_my_canvas(key: Annotated[str, "key"]):
    """Read a value from the working canvas"""
    # read a value from myCANVAS given a key
    return CANVAS.read(key)

@tool
def write_my_canvas(key: Annotated[str, "key"],
                    value: Annotated[Any, "value"],
                    overwrite: Annotated[bool, "True to overwrite if key already exist. only set to True if you are certain you want to overwrite the existing value"] = False):
    """Write a value to the working canvas. If the key already exists, it will not overwrite unless specified."""
    # write a value to myCANVAS given a key and a value
    return CANVAS.write(key, value, overwrite)

##################################################################################################
##                                          DFT tools                                           ##
##################################################################################################

# @tool
# def get_my_WORKING_DIRECTORY() -> str:
#     """Get the working directory."""
#     return var.my_WORKING_DIRECTORY

def get_kpoints(atoms, kspacing: float) -> list:
    """Returns the kpoints of a given ase atoms object and specific kspacing."""
    cell = atoms.cell
    # ## Check input kspacing is valid
    # if kspacing <= 0:
    #     return "Invalid kspacing, should be greater than 0"
    # if kspacing > 0.5:
    #     return "Too Coarse kspacing, should be less than 0.5"
    ## Calculate the kpoints
    kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in cell
        ]
    
    ## Check if kpoints is even
    for i in range(len(kpoints)):
        if kpoints[i] % 2 == 0:
            if kpoints[i] > 1:
                kpoints[i] -= 1
            else:
                kpoints[i] += 1
    # time.sleep(60)
    return kpoints

@tool
def get_files_in_dir(dir_path: Annotated[str, "Directory path"],
                     file_extension: Annotated[str, "File extension to filter by. If you want all files and folders, use ''"] = ''
                     ) -> list:
    """Returns a list of files in a given directory with a specific file extension."""
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    files = ""
    # list all files in the directory
    for file in os.listdir(os.path.join(WORKING_DIRECTORY, dir_path)):
        # check if the file has the specified extension
        if file.endswith(file_extension):
            files += file + "\n"
    # time.sleep(60)
    return files

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
    # time.sleep(60)
    return atoms.todict()


@tool
def init_structure_data(
    element: Annotated[str, "Element symbol"],
    lattice: Annotated[str, "Lattice type. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
    a: Annotated[float, "Lattice constant"],
    b: Annotated[float, "Lattice constant. If only a and b is given, b will be interpreted as c instead."] = None,
    c: Annotated[float, "Lattice constant"] = None,
) -> Annotated[str, "Path of the saved initial structure data file."]:
    """Create single element bulk initial structure based on composite, crystal lattice, lattice info, save to the working dir, and return filename."""
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    atoms = bulk(element, lattice, a=a, b=b, c=c, cubic=True)
    # atoms *= (2, 2, 2)

    # atoms.set_cell(atoms.cell * 0.95, scale_atoms=True)

    # write_lammps_data(os.path.join(WORKING_DIRECTORY, f'{element}.data'), atoms, masses=True)
    
    # return f"Initial structure data is created named {element}.data"
    
    # save the atoms into working dir
    saveDir = os.path.join(WORKING_DIRECTORY, f"{element}-{lattice}.xyz")
    write(saveDir, atoms)
    # time.sleep(60)
    return f"Created atoms saved in {saveDir}"

@tool
def generate_HEA(elements: Annotated[Dict[str, float], "elements and their compositions in the HEA, e.g. {'Fe': 10, 'Co': 5, 'Ni': 2}"],
                 lattice: Annotated[str, "Lattice type. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
                 a: Annotated[float, "Lattice constant a"],
                 b: Annotated[float, "Lattice constant b. If only a and b is given, b will be interpreted as c instead."] = None,
                 c: Annotated[float, "Lattice constant c"] = None,
                 supercell: Annotated[Tuple[int, int, int], "Supercell size in the form of (x, y, z). "] = (4,4,4),
                 ) -> Annotated[str, "Path of the saved HEA structure data file."]:
    """Generate a high entropy alloy (HEA) structure based on the given elements and their compositions, crystal lattice, lattice info, save to the working dir, and return filename."""
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    parentElements = list(elements.keys())[0]
    
    atoms = bulk(parentElements, lattice, a=a, b=b, c=c, cubic=True)
    atoms *= supercell
    
    print(list(elements.values()))
    
    assert len(atoms) > np.lcm.reduce([int(x) for x in list(elements.values())]), "The total number of atoms in the supercell must be greater than the least common multiple of the element compositions."
    
    pool = list(np.arange(len(atoms)))
    for element, composition in elements.items():
        num_atoms_to_replace = int((composition / sum(elements.values())) * len(atoms))
        if len(pool) > num_atoms_to_replace:
            indices_to_replace = np.random.choice(pool, num_atoms_to_replace, replace=False)
            atoms.numbers[indices_to_replace] = atomic_numbers[element]
            pool = [x for x in pool if x not in indices_to_replace]
        else:
            atoms.numbers[pool] = atomic_numbers[element]
        
    saveDir = os.path.join(WORKING_DIRECTORY, "".join(f"{elem}{count}" for elem, count in elements.items()) + ".xyz")
    write(saveDir, atoms)
    # time.sleep(60)
    return f"Created atoms saved in {saveDir}"

@tool
def generateSurface_and_getPossibleSite(species: Annotated[str, "Element symbol"],
                                        crystal_structures: Annotated[str, "Crystal structure. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
                                        a_dict: Annotated[Dict[str, float], "Dictionary of lattice parameters for the crystal structure: Dict[species, lattice_parameter_a]. i.e. {'Pt': 4.0}"],
                                        facets: Annotated[str, "Facet of the surface. Must be one of 100, 110, 111, 210, 211, 310, 311, 320, 321, 410, 411, 420, 421, 510, 511, 520, 521, 530, 531, 540, 541, 610, 611, 620, 621, 630, 631, 640, 641, 650, 651, 660, 661"],
                                        supercell_dim: Annotated[List[int], "typically [int, int, 6]. Supercell dimension, how many times do you want to repeat the primitive cell in each direction: [int, int, int]"],
                                        n_fixed_layers: Annotated[int, "typically 3. Number of fixed layers in the slab"] = 3
                                        ):
    """Generate a surface structure and get the available adsorption sites."""
    a_dict = {'Pt': 3.92}
    supercell_dim[-1] = 6
    surface_dict = generate_surface_structures(
        species_list=[species],
        crystal_structures={species: crystal_structures},
        a_dict=a_dict,
        facets={species: [facets]},
        supercell_dim=supercell_dim,
        n_fixed_layers=n_fixed_layers,
        dirs_exist_ok=True,
        write_to_disk=True,
        write_location=var.my_WORKING_DIRECTORY,
    )
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
    mySurface = surface_dict[species][f'{crystal_structures}{facets}']["structure"]
    # mySites = get_adsorption_sites(mySurface, symm_reduce=0)
    # mySites = get_adsorption_sites(mySurface)
    mySites = mySurface.info['adsorbate_info']['sites']
    
    func = eval(f"ase.build.{crystal_structures}{facets}")
    tmpAtom = func(species, size=(1,1,1), a = a_dict[species])
    for site in mySites.keys():
        mySites[site] = np.sum(tmpAtom.cell*[mySites[site][0], mySites[site][1], 0], axis=0)[:2]
    
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        print(mySites)
    
    mySites_str = output_capture.getvalue()
    
    CANVAS.write('Possible_CO_site_on_Pt_surface', mySites)
    
    absPath = surface_dict[species][f'{crystal_structures}{facets}']['traj_file_path']
    # trim the absPath, remove the part before out, including out
    relaPath = absPath.split(f'{DirOfInterests}/')[-1]
    # time.sleep(60)
    return f"the surface generated is saved at {relaPath}, available adsorbate sites are: {mySites_str}"

@tool
def generate_myAdsorbate(symbols: Annotated[str, "Element symbols of the adsorbate (Do not use any delimiters)"],
                         positions: Annotated[List[List[float]], "Positions of the atoms in the adsorbate, e.g. [[x1, y1, z1], [x2, y2, z2], ...], following the same order as the symbols."],
                         AdsorbateFileName: Annotated[str, "Name (not a path) of the adsorbate file to be saved in traj format"]
                         ):
    """Generate an adsorbate structure and save it."""
    assert AdsorbateFileName.endswith('.traj'), "AdsorbateFileName should end with .traj"
    assert not '/' in AdsorbateFileName, "AdsorbateFileName should not contain '/'"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    os.makedirs(os.path.join(WORKING_DIRECTORY, "adsorbates"), exist_ok=True)
    tmpAtoms = Atoms(symbols=symbols, positions=positions)
    tmpAtoms.center(vacuum=10.0)
    write(os.path.join(WORKING_DIRECTORY, "adsorbates", f"{AdsorbateFileName}"), tmpAtoms)
    # time.sleep(60)
    return f"Adsorbate saved under working directory at adsorbates/{AdsorbateFileName}"

@tool
def add_myAdsorbate(mySurfacePath: Annotated[str, "Path to the surface structure"],
                    adsorbatePath: Annotated[str, "Path to the adsorbate structure"],
                    mySites: Annotated[List[List[float]], "List of adsorption sites you want to put adsorbates on, e.g. [[x1, y1], [x2, y2], ...]"],
                    rotations: Annotated[List[Tuple[float, str]], "List of rotations for the ith adsorbates, e.g. [[90.0, 'x'], [180.0, 'y'], ...]"],
                    surfaceWithAdsorbateFileName: Annotated[str, "Name (not a path) of the surface adsorbated with adsorbate to be saved in traj format"]
                    ):
    """
    Add adsorbate to the surface structure and save it.
    The third argument must be a list in the form of [[x1, y1], [x2, y2], ...], where x and y are the coordinates of the adsorption sites.
    The forth argument must be a list of tuple in the form of [[float(angle), str(axis)], ...], where the first element is the rotation angle and the second element is the axis of rotation.
    """
# @tool
# def add_myAdsorbate(mySurfacePath: Annotated[str, "Path to the surface structure"],
#                     adsorbatePath: Annotated[str, "Path to the adsorbate structure"],
#                     mySites: Annotated[List[List[float]], "List of adsorption sites you want to put adsorbates on, e.g. [[x1, y1], [x2, y2], ...]"],
#                     rotations: Annotated[List[List[str]], "List of rotations for the ith adsorbates, e.g. [['90.0', 'x'], ['180.0', 'y'], ...]"],
#                     surfaceWithAdsorbateFileName: Annotated[str, "Name (not a path) of the surface adsorbated with adsorbate to be saved in traj format"]
#                     ):
#     """
#     Add adsorbate to the surface structure and save it.
#     The third argument must be in the form of [[x1, y1], [x2, y2], ...], where x and y are the coordinates of the adsorption sites.
#     The forth argument must be in the form of [[str(angle), str(axis)], ...], where the first element is the rotation angle and the second element is the axis of rotation.
#     """
    assert surfaceWithAdsorbateFileName.endswith('.traj'), "surfaceWithAdsorbateFileName should end with .traj"
    assert not '/' in surfaceWithAdsorbateFileName, "surfaceWithAdsorbateFileName should not contain '/'"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
    try:
        if not mySurfacePath.startswith(DirOfInterests) and not mySurfacePath.startswith(f'./{DirOfInterests}') and not mySurfacePath.startswith('/nfs'):
            mySurfacePath = os.path.join(WORKING_DIRECTORY, mySurfacePath)
        mySurface = read(mySurfacePath)
    except:
        # time.sleep(60)
        return f"Invalid input atoms directory: {mySurfacePath}. make sure to supply either absolute path, or relative path starting with './{DirOfInterests}'. Please check the path in canvas and try again."

    
    try:
        if not adsorbatePath.startswith(DirOfInterests) and not adsorbatePath.startswith(f'./{DirOfInterests}') and not adsorbatePath.startswith('/nfs'):
            adsorbatePath = os.path.join(WORKING_DIRECTORY, adsorbatePath)
        myAdsorbate = read(adsorbatePath)
    except:
        # time.sleep(60)
        return f"Invalid input atoms directory: {adsorbatePath}. make sure to supply either absolute path, or relative path starting with './{DirOfInterests}'. Please check the path in canvas and try again."
    
    # Load the adsorbate structure
    myAdsorbate = read(adsorbatePath)
    
    for oneSites, oneRotation in zip(mySites, rotations):
        print(oneSites, oneRotation)
        _myAdsorbate = myAdsorbate.copy()
        _myAdsorbate.rotate(float(oneRotation[0]), oneRotation[1], center="COP")
        
        # get the index of the atom with the lowest z-coordinate
        lowestAtomIndex = _myAdsorbate.positions[:,2].argmin()
        
        myHeight = get_adsorbate_height_estimate(mySurface, _myAdsorbate, (oneSites[0], oneSites[1]), anchor_atom_index=lowestAtomIndex)
        add_adsorbate(mySurface, _myAdsorbate, height=myHeight, position=(oneSites[0], oneSites[1]), mol_index=lowestAtomIndex)
    
    # get the parent path of mySurfacePath
    parentPath = os.path.dirname(mySurfacePath)
    
    absPath = os.path.join(parentPath, surfaceWithAdsorbateFileName)
    # save the new structure
    write(absPath, mySurface)
    
    relaPath = absPath.split(f'{DirOfInterests}/')[-1]
    # time.sleep(60)
    return f"Surface with adsorbate saved at {relaPath}"


@tool
def get_ground_state(
    unrelaxedStructuresPath: Annotated[List[str], "Path to all the unrelaxed structures you want to find the ground state"],
    initRelax: Annotated[bool, "Wether to perform initial relaxation before Monte Carlo simulated annealing"] = True,
):
    """
    Generate python jobs for geeting the ground state structure through Monte Carlo simulated annealing.
    """
    
    for i in len(unrelaxedStructuresPath):
        for j in range(3):
    
            pythonFileScript = f"""import sys
sys.path.append('/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent')

from src.utils import *
from mattersim.forcefield import MatterSimCalculator
import copy

SAVING_DIR = var.my_WORKING_DIRECTORY

mattersimModel = 'MatterSim-v1.0.0-1M.pth'
myCalc = MatterSimCalculator(load_path=mattersimModel, device='cuda')

def gen_ground_state(nonGroundAtoms, myCalc, trialID, initRelax=True):
    currAtoms = copy.deepcopy(nonGroundAtoms)
    currAtoms.calc = myCalc

    if initRelax:
        eos = calculate_eos(currAtoms, eps=0.1)
        try:
            v, _, _ = eos.fit()
        except:
            print("EOS fit failed")
            write(os.path.join(SAVING_DIR, str(trialID), "eos-failed.xyz"), currAtoms)
            raise ValueError("EOS fit failed")
        currAtoms.set_cell(currAtoms.get_cell() * (v / currAtoms.get_volume())**(1/3), scale_atoms=True)
    
    annealer = CanonicalSwapAnnealer(currAtoms, myCalc)

    best_atoms, log = annealer.run(
        AnnealSettings(target_accept_uphill=0.8, alpha=0.92, min_temperature=2.0, sweeps_per_T=20, max_T_steps=150, seed=None),
        RelaxSettings(do_relax=True, fmax=0.03, steps=800),
        verbose=True,
    )

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].plot(log['T'])
    ax[0, 0].set_title("Temperature")
    ax[0, 1].plot(log['E'])
    ax[0, 1].set_title('Energy')
    ax[1, 0].plot(log['accept_rate'])
    ax[1, 0].set_title('Acceptance Ratio')
    ax[1, 1].plot(log['best_E_progress'])        
    ax[1, 1].set_title('Best Energy Progress')
    plt.tight_layout()
    

    fileIdx = 0
    
    # while os.path.isfile(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_log_{fileIdx}.png")):
    #     fileIdx += 1
    plt.savefig(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_log_{fileIdx}.png"))
    plt.close(fig)

    fileIdx = 0
    # while os.path.isfile(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_traj_{fileIdx}.xyz")):
    #     fileIdx += 1
    write(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_traj_{fileIdx}.xyz"), log['MCtraj'])

    fileIdx = 0
    # while os.path.isfile(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_out_{fileIdx}.xyz")):
    #     fileIdx += 1  
    write(os.path.join(SAVING_DIR, str(trialID), f"monteCarlo_out_{fileIdx}.xyz"), best_atoms)
    
    return best_atoms
    
myAtom = read({unrelaxedStructuresPath[i]})
gen_ground_state(myAtom, myCalc, {j}, initRelax={initRelax})
            """
            
            with open(os.path.join(var.my_WORKING_DIRECTORY, f'gen_ground_state_{i}_{j}.py'), 'w') as f:
                f.write(pythonFileScript)
            
    return f"Python scripts for generating ground state structures are created in {var.my_WORKING_DIRECTORY}, please run them in the cluster."
    
@tool
def write_script(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "Name of the file to be saved."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the quantum espresso input script to the specified file path"""
    ## Error when '/' in the content, manually delete
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY

    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    path = os.path.join(WORKING_DIRECTORY, f'{file_name}')

    ## If content ends with '/' then remove it
    if content.endswith('/'):
        content = content[:-1]
    
    with open(path,"w",encoding="ascii") as file:
        file.write(content)
    
    os.environ['INITIAL_FILE'] = file_name
    # time.sleep(60)
    return f"Initial file is created named {file_name}"


@tool
def write_QE_script_w_ASE(
    listofElements: Annotated[List[str], "List of distinct element symbols in the unit cell"],
    ppfiles: Annotated[List[str], "List of pseudopotential files in the order of the elements"],
    filename: Annotated[str, "Name of the Quantum Espresso input file, end with .pwi"],
    inputAtomsDir: Annotated[str, "Directory of the input Atoms object (i.e. traj or xyz), or the name of the job that contains the relaxed structure (i.e. xxxx.pwi)."],
    ensembleCalculation: Annotated[bool, "Whether this calculation is ensemble calculation"],
    calculation: Annotated[str, "Type of calculation to perform, e.g. 'scf', 'relax', or 'ensemble'. Set to 'ensemble', when running ensemble calculation"],
    restart_mode: Annotated[Literal['from_scratch', 'restart'], "Restart mode"],
    prefix: Annotated[str, "Prefix for the output files"],
    disk_io: Annotated[Literal['none', 'minimal', 'nowf', 'low', 'medium', 'high'], "Disk I/O level"],
    ibrav: Annotated[int, "Bravais-lattice index. Optional only if space_group is set."],
    nat: Annotated[int, "Number of atoms in the unit cell"],
    ntyp: Annotated[int, "Number of atom types in the unit cell"],
    ecutwfc: Annotated[float, "kinetic energy cutoff (Ry) for wavefunctions, typically between 30-100 Ry"],
    ecutrho: Annotated[float, "Kinetic energy cutoff (Ry) for charge density and potential. typically ecutwfc*4"],
    occupations: Annotated[Literal['smearing', 'tetrahedra', 'tetrahedra_lin', 'tetrahedra_opt', 'fixed', 'from_input'], "Occupation type"],
    smearing: Annotated[Literal['gaussian', 'methfessel-paxton', 'marzari-vanderbilt', 'fermi-dirac'], "Smearing type, please start with methfessel-paxton first"],
    degauss: Annotated[float, "value of the gaussian spreading (Ry) for brillouin-zone integration in metals."],
    conv_thr: Annotated[float, "Convergence threshold for self-consistent loop"],
    electron_maxstep: Annotated[int, "Maximum number of SCF iterations"],
    kspacing: Annotated[float, "K-point spacing (in Angstrom^-1)"],
    input_dft: Annotated[Literal['LDA', 'PBE', 'BEEF-vdW'], "DFT functional. You'll be told which functional to use"],
    ready_to_run_job: Annotated[bool, "True if the job is intended to be run directly without further modification, False if this file is intended to be used to generate other files"] = False,
    additional_input: Annotated[Dict[str, Any], "Additional input parameters to be added to the input script. Should be in the format of a flat dict, {'input_parameter_1': parameter_1, 'input_parameter_2': parameter_2, ...}, parameter_x remain in their native type, str, float, bool, etc. Do not use unless you know what you are doing."] = {},
):
    """Write a Quantum Espresso input script using ASE. Bool value have no quote around them. For smearing start with methfessel-paxton. For ecutwfc choose between 30-100 Ry. When asked to run ensemble calculation, set calculation to 'ensemble'"""

    assert isinstance(additional_input, dict), "additional_input must be a dictionary"
    
    if ensembleCalculation:
        assert calculation == 'ensemble', "When running ensemble calculation, please set calculation to 'ensemble'"
    
    if calculation == 'ensemble':
        assert inputAtomsDir.endswith('.pwi'), "inputAtomsDir must be a .pwi file with relaxed structure when running ensemble calculation with BEEF-vdW functional"
        assert input_dft == 'BEEF-vdW', "input_dft must be 'BEEF-vdW' when running ensemble calculation"
    
    disk_io = 'none'
    
    
    
    # assemble the pseudopotentials dict from the list of elements and pseudopotentials
    pseudopotentials = {}
    for element, pseudo in zip(listofElements, ppfiles):
        if not os.path.exists(os.path.join("/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/all_lda_pbe_UPF", pseudo)):
            # time.sleep(60)
            return f"Invalid pseudopotential file: {pseudo}. Make sure to supply the correct pseudopotential file name."
        pseudopotentials[element] = pseudo
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
    tmpinputAtomsDir = inputAtomsDir
    try:
        if not inputAtomsDir.startswith(DirOfInterests) and not inputAtomsDir.startswith(f'./{DirOfInterests}') and not inputAtomsDir.startswith('/nfs'):
            inputAtomsDir = os.path.join(WORKING_DIRECTORY, inputAtomsDir)
            
        if inputAtomsDir.endswith('.pwi'):
            inputAtomsDir += '.pwo'
        atoms = read(inputAtomsDir)
    except:
        # check if file exists
        if os.path.exists(inputAtomsDir):
            raise ValueError(f"Job {tmpinputAtomsDir} failed or did not converge. Please only use converged jobs.")
        else:
            raise ValueError(f"Invalid input atoms directory: {tmpinputAtomsDir}. make sure to supply either absolute path, or relative path starting with './{DirOfInterests}'. Please check the path in canvas and try again.")
    
    filenameWDir = os.path.join(WORKING_DIRECTORY, filename)
    
    
    kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in atoms.cell
        ]
        
    ## Check if kpoints is even
    for i in range(len(kpoints)):
        if kpoints[i] % 2 == 0:
            if kpoints[i] > 1:
                kpoints[i] -= 1
            else:
                kpoints[i] += 1

    # Write the input script
    write(filenameWDir,
          atoms,
          input_data={
                'calculation': calculation,
                'restart_mode': restart_mode,
                'prefix': prefix,
                'pseudo_dir': "/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/all_lda_pbe_UPF",
                'outdir': './out',
                'disk_io': disk_io,
                'ibrav': ibrav,
                'nat': nat,
                'ntyp': ntyp,
                'ecutwfc': ecutwfc,
                'ecutrho': ecutrho,
                'occupations': occupations,
                'smearing': smearing,
                'degauss': degauss,
                'conv_thr': conv_thr,
                'electron_maxstep': electron_maxstep,
                'input_dft': input_dft,
                **additional_input
          },
          format='espresso-in',
          pseudopotentials=pseudopotentials,
          kpts=tuple(kpoints)
          )
    
    
    if not ready_to_run_job:
        destiJobList = 'scratch_job_list'
    else:
        destiJobList = 'ready_to_run_job_list'
    
    job_list = [filename]
    old_job_list = CANVAS.canvas.get(destiJobList, []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write(destiJobList,job_list, overwrite=True)
    
    # time.sleep(60)
    return f"Quantum Espresso input script is written to {filename}"

@tool
def write_LAMMPS_script(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "Name of the file to be saved."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the LAMMPS input script to the specified file path"""
    ## Error when '/' in the content, manually delete
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    path = os.path.join(WORKING_DIRECTORY, f'{file_name}')
    
    job_list_dict = {}
    job_list = []

    ## If content ends with '/' then remove it
    if content.endswith('/'):
        content = content[:-1]
    
    with open(path,"w",encoding="ascii") as file:
        file.write(content)
    
    os.environ['INITIAL_FILE'] = file_name
    
    job_list.append(file_name)
    
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
        
    # time.sleep(60)
    return f"Initial file is created named {file_name}"

@tool
def find_classical_potential(element: str) -> str:
    """Return classical potential file path for given element symbol."""
    # time.sleep(60)
    return f'The classcial potential file for {element} is located at /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/EAM/Li_v2.eam.fs'

@tool
def find_pseudopotential(element: str) -> str:
    """Return the pseudopotential file path for given element symbol."""
    spList = []
    pseudo_dir = var.OTHER_GLOBAL_VARIABLES["PSEUDO_DIR"]
    if pseudo_dir is None:
        print("find_pseudopotential tool faulty! please terminate the calculation!")
        while(1):
            time.sleep(60)
    for roots, dirs, files in os.walk(f'{pseudo_dir}'):
        for file in files:
            # if element == file.split('.')[0].split('_')[0].capitalize():
            if element == file.split('_')[0].capitalize():
                spList.append(file)
    
    if len(spList) > 0:
        ans = f'The pseudopotential file for {element} is:\n'
        for sp in spList:
            ans += f'{sp}\n'
        ans += f'under {pseudo_dir}'
        # time.sleep(60)
        return ans
    else:
        # time.sleep(60)
        return f"Could not find pseudopotential for {element}"

@tool
def generate_convergence_test(input_file_name: Annotated[str, "Name of the template quantum espresso input file"],
                              kspacing:Annotated[list[float], "List of kspacing to be tested. Typically between 0.1-0.4"],
                              ecutwfc:Annotated[list[int], "List of ecutwfc to be tested. Typically between 40-100"],
                              ):
    '''
    Generate the convergence test input scripts for quantum espresso calculation using another quantum espresso input file as a template and save the job list. 
    '''
    # kspacing = [0.6, 0.8, 1.0]
    # ecutwfc = [10, 20, 30]
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
    # Read the atom object from the input script
    try:
        atom = read(input_file)
    except:
        # time.sleep(60)
        return f"Invalid input file, please inspect CANVAS and select the correct template file."
    
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
        
        ## Check if kpoints is even
        for i in range(len(kpoints)):
            if kpoints[i] % 2 == 0:
                if kpoints[i] > 1:
                    kpoints[i] -= 1
                else:
                    kpoints[i] += 1
                
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
                    lines[i] = f"    ecutrho = {ecutwfc_max*4},\n"
                
                ## Find the kpoints line
                if 'K_POINTS' in line:
                    lines[i+1] = ' '.join(map(str,kpoints)) +' 0 0 0' +'\n'

            ## Write the new input script
            tmpName = os.path.splitext(input_file_name)[0].split('_k_')[0]
            new_file_name = f'{tmpName}_k_{k}_ecutwfc_{ecutwfc_max}.pwi'
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
        
        ## Check if kpoints is even
        for i in range(len(kpoints)):
            if kpoints[i] % 2 == 0:
                if kpoints[i] > 1:
                    kpoints[i] -= 1
                else:
                    kpoints[i] += 1
                
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if 'outdir' in line:
                #     lines[i] = f"    outdir = './out_k_{kspacing_min}_ecutwfc_{e}',\n"
                ## Find the ecutwfc line
                if 'ecutwfc' in line:
                    lines[i] = f'    ecutwfc = {e},\n'
                if 'ecutrho' in line:
                    lines[i] = f"    ecutrho = {e*4},\n"
                
                ## Find the kpoints line
                if 'K_POINTS' in line:
                    lines[i+1] = ' '.join(map(str,kpoints)) +' 0 0 0' +'\n'

            ## Write the new input script
            new_file_name = f'{os.path.splitext(input_file_name)[0]}_k_{kspacing_min}_ecutwfc_{e}.pwi'
            job_list_dict[new_file_name] = {'k':kspacing_min, 'ecutwfc':e}
            new_input_file = os.path.join(WORKING_DIRECTORY, new_file_name)
            job_list.append(new_file_name)
            with open(new_input_file, 'w') as f:
                f.writelines(lines)
    ## Remove duplicate files
    job_list = list(set(job_list))
    ## Save the job list
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
    CANVAS.write('jobs_K_and_ecut',job_list_dict)
    # time.sleep(60)
    return f"Job list is saved scucessfully. Please tell the supervisor in your response that convergence job has generated sucessfully, please continue to submit the jobs"

@tool
def generate_eos_test(input_file_name:str,kspacing:float, ecutwfc:int, stepSize:float=0.025):
    '''
    Generate the equation of state test input scripts for quantum espresso calculation and save the job list.
    
    Input:  input_file_name: str, the name of the input file
            kspacing: float, the kspacing to be tested
            ecutwfc: int, the ecutwfc to be tested
            stepSize: float, the step size for the scale factor, default is 0.025, which will scale the cell size from 0.95 to 1.05
    '''
    assert stepSize > 0.01 and stepSize < 0.1, "stepSize should be between 0.01 and 0.1"
    
    # CANVAS.write('job_list', [], overwrite=True)
    CANVAS.canvas['jobs_K_and_ecut'] = {}
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
    prefix = input_file_name.split('.')[0]
    # Read the atom object from the input script
    try:
        atom = read(input_file)
    except:
        # time.sleep(60)
        return f"Invalid input file, try inspect the shared CANVAS and use the inital pwi file as the input file"
    
    job_list = []
    
    cell = atom.cell
    ## Calculate the kpoints
    kpoints = [
            2 * ((np.ceil(2 * np.pi / np.linalg.norm(ii) / kspacing).astype(int)) // 2 + 1) for ii in cell
        ]
    
    ## Check if kpoints is even
    for i in range(len(kpoints)):
        if kpoints[i] % 2 == 0:
            if kpoints[i] > 1:
                kpoints[i] -= 1
            else:
                kpoints[i] += 1
            
    for scale in np.linspace(1-stepSize*2, 1+stepSize*2, 5):
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
                lines[i] = f"    ecutrho = {ecutwfc*4},\n"
            if 'CELL_PARAMETERS' in line:
                lines[i+1] = f"{cell[0][0]*scale} {cell[0][1]*scale} {cell[0][2]*scale}\n"
                lines[i+2] = f"{cell[1][0]*scale} {cell[1][1]*scale} {cell[1][2]*scale}\n"
                lines[i+3] = f"{cell[2][0]*scale} {cell[2][1]*scale} {cell[2][2]*scale}\n"
                
            if 'K_POINTS' in line:
                lines[i+1] = f"{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n"
    
        ## New input file name
        new_file_name = f"{prefix}_{scale}.pwi"
        job_list.append(new_file_name)
        new_file = os.path.join(WORKING_DIRECTORY, new_file_name)
        with open(new_file, 'w') as f:
            f.writelines(lines)
    ## Remove duplicate files
    job_list = list(set(job_list))
    print(job_list)
    ## Save the job list as json file
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
    
    # time.sleep(60)
    return f"Job list is saved scucessfully, continue to submit the jobs. Files of interest are {job_list}"

###################################### DFT POST-PROCESSING TOOLS ######################################

@tool
def get_convergence_suggestions(
    filename: Annotated[str, "Name of the Quantum Espresso input file that did not converge, end with .pwi"],
    question: Annotated[str, "Question about this job, e.g. 'Why this job did not converge?' or 'how to improve the accuracy of this job?'"],
):
    "Get suggestions on how to resolve issues for a certain job, i.e. converge or not accurate enough."
    outFile = filename + ".pwo"
    errFile = filename + ".err"
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # WORKING_DIRECTORY = "/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/out"
    
    # config = load_config(os.path.join('./config', "default.yaml"))
    config = var.OTHER_GLOBAL_VARIABLES
    # llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    workerllm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # workerllm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # workerllm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # llm = ChatDeepSeek(model_name=config["DeepSeek_MDL"], api_key=config['DeepSeek_API_KEY'], api_base=config['DeepSeek_BASE_URL'], temperature=0.0)
    
    finalSuggestion = ""
    for myfile in [
                   filename, 
                   outFile, 
                #    errFile
                   ]:
        if os.path.exists(os.path.join(WORKING_DIRECTORY, myfile)):
            finalSuggestion += f"Suggestion based on {myfile}:\n"
            print(f"Suggestion based on {myfile}:\n")
            
            with open(os.path.join(WORKING_DIRECTORY, myfile),"r") as file:
                content = file.read()
            
            task_formatted = f"{content}\n I have a question about the DFT calculation related to the file above: {question}. Please think about what could be the reason, and give me suggestions to address it. Never give suggestion to lower the accuracy of the calculation, such as loosen the convergence threshold."
            
            # for agent_response in dft_reader_agent.stream({"messages": [("user", task_formatted)]}, {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}):
            #     agent_response = next(iter(agent_response.values()))
            #     print_stream(agent_response)
            
            system_msg = """
You are a DFT expert who's good at giving concise suggestions on how to resolve issues in DFT calculations. Do not modify nosym and pesudopotentials. Never make any adjustment to make the calculation less accurate.
Please use the format: parameterX: suggestionX, reasonX; parameterY: suggestionY, reasonY; ...
"""
            
            invokingMsg = [
                ("system", system_msg),
                ("user", task_formatted)
            ]
            agent_response = workerllm.invoke(invokingMsg)
            
            finalSuggestion += agent_response.content + "\n\n"
            print(agent_response.content + "\n\n")
            
    if finalSuggestion == "":
        # time.sleep(60)
        return f"Job {filename} has no related files, please check the job list and make sure the job is finished."
        
    finalSuggestion += "Please check the suggestions above and come up with a plan to fix the issue. Never take suggestions that will lower the accuracy of the calculation."
    # time.sleep(60)
    return finalSuggestion
        

@tool
def calculate_formation_E(slabFilePath: Annotated[str, "the slab calculation file name, ending in pwi"],
                          adsorbateFilePath: Annotated[str, "the adsorbate calculation file name, ending in pwi"],
                          systemFilePath: Annotated[str, "the slab with adsorbate calculation file name, ending in pwi"],
                          ):
    """using the energies of the slab, adsorbate, and slab with adsorbate, calculate the formation energy of the adsorbate on the slab. """
    working_directory = var.my_WORKING_DIRECTORY
    slabFilePath = os.path.join(working_directory, slabFilePath + '.pwo')
    adsorbateFilePath = os.path.join(working_directory, adsorbateFilePath + '.pwo')
    systemFilePath = os.path.join(working_directory, systemFilePath + '.pwo')
    
    # Load the energies
    slab = read(slabFilePath)
    adsorbate = read(adsorbateFilePath)
    system = read(systemFilePath)
    
    slabEnergy = slab.get_potential_energy()/len(slab)
    adsorbateEnergy = read(adsorbateFilePath).get_potential_energy()
    systemEnergy = read(systemFilePath).get_potential_energy()
    
    # assume slab only have one species
    slabSpecies = slab.numbers[0]
    NslabInSystem = system.numbers.tolist().count(slabSpecies)
    # NadsorbateInSystem = (len(system) - NslabInSystem)/len(adsorbate)
    
    formationEnergy = systemEnergy - slabEnergy * NslabInSystem - adsorbateEnergy
    
    # time.sleep(60)
    return f"The formation energy of the adsorbate on the slab is {formationEnergy} eV"

@tool
def calculate_lc(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, which will be used to calculate the lattice constant"]
    ) -> str:
    """Read the output file and calculate the lattice constant"""
    
    assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
    for i in jobFileIdx:
        assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    job_list = CANVAS.canvas.get('finished_job_list', []).copy()
    job_list = np.array(job_list, dtype=str)[jobFileIdx]
    print(f"actual job list: {job_list}")

    volume_list = []
    energy_list = []
    for job in job_list:
        print(f'reading {job}')
        try:
            atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
        except:
            # time.sleep(60)
            return f"Job {job} is not finished or failed."
        volume_list.append(atom.get_volume())
        energy_list.append(atom.get_potential_energy())
        print(f'{job} volume is {atom.get_volume()}, energy is {atom.get_potential_energy()}')
    
    # plot the volume vs energy
    plt.plot(volume_list, energy_list, 'o-')
    plt.xlabel('Volume (A^3)')
    plt.ylabel('Energy (eV)')
    plt.title('Volume vs Energy')
    plt.savefig(os.path.join(WORKING_DIRECTORY, 'volume_vs_energy.png'))
    plt.close()
    
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

    # time.sleep(60)
    return f'The lattice constant is {lc}'

@tool
def get_convex_hull(dir: Annotated[str, "Directory containing the finished jobs to generate the convex hull"]):
    """
    Generate the convex hull using the data in the given directory.
    """
    post_0_1_process(dir)
    return f'Convex hull is generated in {dir}/convex_hull'

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

    # time.sleep(60)
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
    # time.sleep(60)
    return lc

@tool
def get_kspacing_ecutwfc(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, which will be used to determine the kspacing and ecutwfc"],
                         threshold: Annotated[float, "the threshold mev/atom to determine the convergence"] = 1.0) -> str:
    '''Read the convergen test result and determine the kspacing and ecutwfc used in the production
    Input:
        jobFileIdx: list, the indexs of files in the finished job list, which will be used to determine the kspacing and ecutwfc
        threshold: float , the threshold mev/atom to determine the convergence
    output: str, the kspacing and ecutwfc used in the production
    '''
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
    for i in jobFileIdx:
        assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
    job_dict = CANVAS.canvas.get('jobs_K_and_ecut', {})
    job_list = CANVAS.canvas.get('finished_job_list', []).copy()
    job_list = np.array(job_list, dtype=str)[jobFileIdx]
    print(f"actual job list: {job_list}")
    assert len(job_list) > 0, "job list 0"
    
    print(f"successfully read {len(job_list)} jobs, and {len(job_dict)} job_dict")

    ### Find the kpoints and ecutwfc from the output file
    kspacing = []
    ecutwfc = []
    energy_list = []
    goodJob = []
    Natom = None
    for job in job_list:
        ## Read the output file
        print(f'reading {job}')
        try:
            atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
        except:
            print(f"Job {job} is not finished or failed.")
            continue
        energy = atom.get_potential_energy()
        energy_list.append(energy)
        Natom = atom.get_number_of_atoms()
        
        kspacing.append(job_dict[job]['k'])
        ecutwfc.append(job_dict[job]['ecutwfc'])
        goodJob.append(job)
    
    print(f"successfully read {len(kspacing)} kspacing and {len(ecutwfc)} ecutwfc")
    
    if len(set(kspacing)) == 1 and len(set(ecutwfc)) > 1:
        # time.sleep(60)
        return f"Only one kspacing is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
    if len(set(ecutwfc)) == 1 and len(set(kspacing)) > 1:
        # time.sleep(60)
        return f"Only one ecutwfc is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
    if len(set(kspacing)) == 1 and len(set(ecutwfc)) == 1:
        # time.sleep(60)
        return f"Only one job is good, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing and ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
        
    convergence_df = pd.DataFrame({'job':goodJob,'kspacing':kspacing, 'ecutwfc':ecutwfc, 'energy':energy_list})
    ## Save the convergence test result if file exist then append to it
    if os.path.exists(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv')):
        convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'), mode='a', header=False)
    else:
        convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'))
    
    ## Determine the kpoints and ecutwfc based on the threshold
    k_chosen, ecutwfc_chosen,finnerEcut,df_kspacing, df_ecutwfc,finnerKspacing = select_k_ecut(convergence_df, threshold, Natom)
    
    print(f"Chosen kspacing: {k_chosen}, Chosen ecutwfc: {ecutwfc_chosen}")
    
    ## Save the chosen kspacing and ecutwfc
    if os.path.exists(os.path.join(WORKING_DIRECTORY, 'df_k.csv')):
        df_kspacing.to_csv(os.path.join(WORKING_DIRECTORY, 'df_k.csv'), mode='a', header=False)
    else:
        df_kspacing.to_csv(os.path.join(WORKING_DIRECTORY, 'df_k.csv'))
    
    if os.path.exists(os.path.join(WORKING_DIRECTORY, 'df_e.csv')):
        df_ecutwfc.to_csv(os.path.join(WORKING_DIRECTORY, 'df_e.csv'), mode='a', header=False)
    else:
        df_ecutwfc.to_csv(os.path.join(WORKING_DIRECTORY, 'df_e.csv'))  
        
    print("saved the chosen kspacing and ecutwfc")
    
    
    if finnerEcut and ecutwfc_chosen < 120 and finnerKspacing and k_chosen > 0.1:
        ans = "Only the calculation with the finest settings is finished. Please regenerate the convergence test with finner ecutwfc and finner kspacing. Do not infer converged settings yourself!"
        # ans += f"\nHowever, the calculation is not converged, please consider redo the convergence test and using a finner ecutwfc and finner kspacing"
    elif finnerEcut and ecutwfc_chosen < 120:
        ans = "Only calculations with the finest ecutwfc is finished. Please regenerate the convergence test with finner ecutwfc. Do not infer converged settings yourself!"
    elif finnerKspacing and k_chosen > 0.1:
        ans = "Only the calculation with the finest kspacing is finished. Please regenerate the convergence test with finner kspacing. Do not infer converged settings yourself!"
    else:
        ans = f"Please use kspacing {k_chosen} and ecutwfc {ecutwfc_chosen} for the production calculation"
    # time.sleep(60)
    return ans

@tool
def analyze_BEEF_result(
    slabFilePath: Annotated[str, "the slab calculation file"],
    adsorbateFilePath: Annotated[str, "the adsorbate calculation file"],
    ontopFilePath: Annotated[str, "the slab with ontop adsorbate calculation file"],
    fccFilePath: Annotated[str, "the slab with fcc adsorbate calculation file"],
) -> str:
    '''Read the BEEF output, calculate the abrosption energy and analyze the BEEF result'''
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
    PathList = [slabFilePath, adsorbateFilePath, ontopFilePath, fccFilePath]
    
    for i in range(len(PathList)):
        tmp = PathList[i]
        try:
            if not PathList[i].startswith(DirOfInterests) and not PathList[i].startswith(f'./{DirOfInterests}') and not PathList[i].startswith('/nfs'):
                PathList[i] = os.path.join(WORKING_DIRECTORY, PathList[i]) + '.pwo'
            _ = read(PathList[i])
        except:
            if os.path.exists(PathList[i]):
                return f"{tmp} did not finish successfully."
            return f"Invalid input atoms directory: {tmp}. make sure to supply either absolute path, or relative path starting with './{DirOfInterests}'. Please check the path in canvas and try again."

    
    ## Read energy
    slab_e = read_BEEF_output(PathList[0])
    if slab_e == "WrongCalc":
        return f"Please run slab ensemble calculation using BEEF-vdW with relaxed slab structure! Do not proceed any further!"
    adsorbate_e = read_BEEF_output(PathList[1])
    if adsorbate_e == "WrongCalc":
        return f"Please run adsorbate ensemble calculation using BEEF-vdW with relaxed adsorbate structure! Do not proceed any further!"
    ontop_e = read_BEEF_output(PathList[2])
    if ontop_e == "WrongCalc":
        return f"Please run ontop ensemble calculation using BEEF-vdW with relaxed slab and adsorbate structure! Do not proceed any further!"
    fcc_e = read_BEEF_output(PathList[3])
    if fcc_e == "WrongCalc":
        return f"Please run fcc ensemble calculation using BEEF-vdW with relaxed slab and adsorbate structure! Do not proceed any further!"
    
    ## Plot
    try:
        energy_dict = {}
        energy_dict['clean'] = slab_e
        energy_dict['CO'] = adsorbate_e
        energy_dict['ontop_down'] = ontop_e
        energy_dict['fcc_down'] = fcc_e
        
        fontsize=15
        plot_settings = {
            # "font.family": "times new roman",
            "axes.labelsize": fontsize,
            "axes.labelweight": "bold",
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.size": fontsize,
            "axes.linewidth": 2.0,
            "lines.dashed_pattern": [5, 2.5],
            "lines.markersize": 10,
            "lines.linewidth": 2,
            "lines.markeredgewidth": 1,
            # "lines.markeredgecolor": "k",
            "legend.fontsize": fontsize,
            "legend.frameon": False,
            'figure.figsize': [6, 6],
        }

        # Update rcParams with settings from JSON file
        rcParams.update(plot_settings)
        
        df = pd.DataFrame(energy_dict)
        df.to_csv('energies.csv', index=False)
        ads_fcc = df['fcc_down'] - df['clean'] - df['CO']
        ads_ontop = df['ontop_down'] - df['clean'] - df['CO']
        # plot energy distribution
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0,0,1,1])
        ax.hist(ads_fcc, bins=50, color='blue', alpha=0.7, label='FCC')
        ax.axvline(ads_fcc.mean(), color='k', linestyle=':', linewidth=2, label='FCC mean')
        ax.hist(ads_ontop, bins=50, color='red', alpha=0.7, label='Ontop')
        ax.axvline(ads_ontop.mean(), color='k', linestyle='-.', linewidth=2, label='Ontop mean')
        plt.legend()
        plt.xlabel('Adsorption Energy (eV)')
        plt.ylabel('Frequency')
        plt.savefig('energy_distribution.png',dpi=300,bbox_inches='tight')
    except:
        print("Failed to plot the energy distribution.")

    ## Formation
    ontop_formation = ontop_e - slab_e - adsorbate_e
    fcc_formation = fcc_e - slab_e - adsorbate_e

    print(f"ontop formation energy: {ontop_formation.mean()} eV")
    print(f"fcc formation energy: {fcc_formation.mean()} eV")
    ## Formation Energy Difference
    formation_energy_diff = ontop_formation - fcc_formation

    ## Distribution of Formation energy differernce 
    if formation_energy_diff.all() > 0:
        result = f"fcc is more stable than ontop by average {formation_energy_diff.mean()} eV"
    elif formation_energy_diff.all() < 0:
        result = f"ontop is more stable than fcc by average {abs(formation_energy_diff.mean())} eV"
    else:
        result = f" {sum(formation_energy_diff>0)} xc functionals prefer fcc, {sum(formation_energy_diff<0)} xc functionals prefer ontop"
    return result

##################################################################################################
##                                          HPC tools                                           ##
##################################################################################################

@tool
def find_job_list() -> str:
    """Return the list of job files to be submitted."""

    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    
    # time.sleep(60)
    return f'The files need to be submitted are {job_list}. Please continue to submit the job.'

@tool
def read_file(
    input_file: Annotated[str, "The file to be read."]
) -> Annotated[str, "read content"]:
    """read file content from the specified file path"""
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    ## Error when '/' in the content, manually delete
    path = os.path.join(WORKING_DIRECTORY, input_file)
    with open(path,"r") as file:
        content = file.read()
    # time.sleep(60)
    return content

@tool
def add_resource_suggestion(
    qeInputFileName: str,
    partition: str,
    nnodes: int,
    ntasks: int,
    runtime: Annotated[str, "Time limit for the job, in minutes"],
    submissionScript: Annotated[str, "submission script based on the types of jobs. Do not include any #SBATCH stuff. output filename must be <full input filename with extension>.<output_file_type>"],
    outputFilename: Annotated[str, "the output filename of the job"],
) -> Annotated[str, "source suggestion saved location"]:
    """
    After agent generate resource suggestions and submission script based on the DFT input file, add it to the json file "resource_suggestions.json" in the WORKING_DIRECTORY.
    output filename must be <full input filename with extension>.<output_file_type>, 
    For example: {"input1.pwi": {"nnodes": 2, "ntasks": 4, "runtime": 60, "submissionScript": "
spack load quantum-espresso@7.2\n \
\n \
echo "Job started on `hostname` at `date`"\n \
\n \
mpirun pw.x -i input1.pwi > input1.pwi.pwo\n \
\n \
echo " "\n \
echo "Job Ended at `date`"
    ", "outputFilename": "input1.pwi.pwo"}, "gpawScript.py": {"nnodes": 1, "ntasks": 1, "runtime": 30, "submissionScript": "
echo "Job started on `hostname` at `date`"\n \
\n \
export GPAW_SETUP_PATH=/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/gpaw-setups-24.11.0\n \
spack load py-gpaw\n \
\n \
python gpawScript.py\n \
echo " "\n \
echo "Job Ended at `date`"\n \
    ", "outputFilename": ""}}
    """
    if not isinstance(partition, str) or not isinstance(nnodes, int) or not isinstance(ntasks, int) or not isinstance(runtime, str):
        # time.sleep(60)
        return "Invalid input, please check the input format"
    # craete the json file if it does not exist, otherwise load it
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY

    new_resource_dict = {qeInputFileName: {"partition": "venkvis-cpu", "nnodes": 1, "ntasks": 48, "runtime": 2800, "submissionScript": submissionScript, "outputFilename": outputFilename}}
    
    # check if resource_suggestions.db exist in the working directory
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if not os.path.exists(db_file):
        initialize_database(db_file)
    
    add_to_database(new_resource_dict, db_file)
    
    # time.sleep(60)
    return f"Resource suggestion for {qeInputFileName} saved scucessfully"


@tool
def submit_and_monitor_job(
    jobType: Annotated[str, "The type of job to be submitted, e.g. DFT, LAMMPS"]
    ) -> str:
    '''
    Submit jobs in the job list to supercomputer, return the location of the output file once the job is done. Do not call this tool until you added the resource suggestion.
    '''
    
    # check if resource_suggestions.json exist
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if not os.path.exists(resource_suggestions):
        # time.sleep(60)
        return "Resource suggestion file not found, please use the add_resource_suggestion tool to add the resource suggestion"
        
    # job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = []
    
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
        filename, partition, nnodes, ntasks, runtime, submissionScript, outputFilename = row
        job_list.append(filename)
        resource_dict[filename] = {
            'partition': partition,
            'nnodes': nnodes,
            'ntasks': ntasks,
            'runtime': runtime,
            'submissionScript': submissionScript,
            'outputFilename': outputFilename
        }
    
    conn.close()
    print(f"loaded resource suggestions: {json.dumps(resource_dict, indent=4)}")
    
    CANVAS.canvas['ready_to_run_job_list'] = job_list.copy()
    wasJobList = deepcopy(job_list)
    
    ## Check resource key is valid
    for job in job_list:
        if job not in resource_dict.keys():
            # time.sleep(60)
            return f"Resource suggestion for {job} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    
    if len(job_list) == 0:
        # time.sleep(60)
        return f"Resource suggestion not found, please use the add_resource_suggestion tool to add the resource suggestion."
    
    print(f"loaded {len(job_list)} jobs from job_list.json, and {len(resource_dict)} resource suggestions from resource_suggestions.json")
    
    print("checking pysqa prerequisites...")
    # check if slurm.sh and queue.yaml exist in the working directory
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, "slurm.sh")) or not os.path.exists(os.path.join(WORKING_DIRECTORY, "queue.yaml")):
        print("Creating pysqa prerequisites...")
        create_pysqa_prerequisites(WORKING_DIRECTORY)
    
    qa = QueueAdapter(directory=WORKING_DIRECTORY)
    
    queueIDList = []
    notConvergedList = []
    while True:
        for inputFile in job_list:    
            
            ## Check if the input file exists
            if not os.path.exists(os.path.join(WORKING_DIRECTORY, inputFile)):
                # time.sleep(60)
                return f"Input file {inputFile} does not exist, please use the find job list tool to submit the file in the job list"
            print("Generating batch script...")

            ## Check if the output file exists 
            outputFile = resource_dict[inputFile]['outputFilename']
            if os.path.exists(os.path.join(WORKING_DIRECTORY, outputFile)):
                ## Supervisor sometimes ask to submit the job again, so we need to check if the output file exists
                try:
                    # temporay disable the read function to avoid the calculation
                    # tmp = read(os.path.join(WORKING_DIRECTORY, outputFile))
                    # _ = tmp.get_potential_energy()
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
            command=resource_dict[inputFile]['submissionScript'],
            errNoutName=inputFile
            )
            
            if job_id is None:
                # time.sleep(60)
                return "Job submission failed"

            queueIDList.append(job_id)
            ## Sleep for 1.5 second to avoid the job submission too fast
            time.sleep(1)
            
            #  Change the bash script name to avoid the job submission too fast
            os.rename(os.path.join(WORKING_DIRECTORY, "run_queue.sh"), os.path.join(WORKING_DIRECTORY, f"slurm_{inputFile}.sh"))
            time.sleep(1)
        
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
        break
        
        # if jobType == "DFT":
        #     print("Checking jobs")
            
        #     checked = set()
        #     unchecked = set(job_list)
        #     while checked != unchecked:
        #         for inputFile in job_list:
        #             outputFile = resource_dict[inputFile]['outputFilename']
        #             print(f"Checking job {inputFile}")
        #             checked.add(inputFile)
        #             try:
        #                 atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
        #                 print(atoms.get_potential_energy())
        #                 # delete inputFile from job_list
        #                 job_list.remove(inputFile)
        #                 print(f"Job list: {job_list}")
        #                 print()
        #             except:
        #                 # see if the job did not converge
        #                 # read the output file as text
        #                 with open(os.path.join(WORKING_DIRECTORY, outputFile), 'r') as f:
        #                     lines = f.readlines()
        #                 # check if the output file contains "convergence NOT achieved"
        #                 notConverge = False
        #                 for line in lines:
        #                     if "convergence NOT achieved" in line:
        #                         notConverge = True
        #                         notConvergedList.append(inputFile)
        #                         break
                            
        #                 if notConverge:
        #                     # remove inputFile from job_list
        #                     job_list.remove(inputFile)
        #                 else:
        #                     # if outputFile exsit remove outputFile
        #                     try:
        #                         # temporay disable remove to avoid the calculation
        #                         # os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
        #                         print(f"{outputFile} removed")
        #                     except:
        #                         print("output file does not exist")
        #                     print(f"Job {inputFile} failed, will resubmit the job")
            
            
        #     # for idx, inputFile in enumerate(job_list):
        #     #     outputFile = resource_dict[inputFile]['outputFilename']
        #     #     print(f"Checking job {inputFile}")
        #     #     try:
        #     #         atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
        #     #         print(atoms.get_potential_energy())
        #     #         # delete inputFile from job_list
        #     #         job_list.remove(inputFile)
        #     #         print(f"Job list: {job_list}")
        #     #         print()
        #     #     except:
        #     #         # remove outputFile
        #     #         os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
        #     #         print(f"Job {inputFile} failed, will resubmit the job")
        #     if len(job_list) == 0:
        #         # load jobs frm job_list.json
        #         job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
                
        #         # read all energies into a dict
        #         energies = {}
        #         for inputFile in job_list:
        #             if inputFile in notConvergedList:
        #                 continue
        #             outputFile = resource_dict[inputFile]['outputFilename']
        #             atoms = read(os.path.join(WORKING_DIRECTORY, outputFile))
        #             energies[inputFile] = atoms.get_potential_energy()
                
        #         job_list = []
                
        #         # check two or more key has the same value, if so, add the key back to the job_list
        #         for key, value in energies.items():
        #             if list(energies.values()).count(value) > 1:
        #                 print(f"!!!!!!!Job {key} has the same energy as other jobs, may resubmit the job!!!!!!!!")
        #                 job_list.append(key)
                
        #         print()
        #         # check whether job in job_list has the same inputFile content, if so, remove the job from job_list
        #         tobeRemoved = np.zeros(len(job_list))
        #         for jobIdx in range(len(job_list)):
        #             for jobIdx2 in range(jobIdx+1, len(job_list)):
        #                 if cmp(os.path.join(WORKING_DIRECTORY, job_list[jobIdx]), os.path.join(WORKING_DIRECTORY, job_list[jobIdx2]), shallow=False):
        #                     print(f"!!!!!!!Job {job_list[jobIdx]} has the same content as {job_list[jobIdx2]}, will remove the job!!!!!!!!")
        #                     tobeRemoved[jobIdx] = 1
        #                     tobeRemoved[jobIdx2] = 1
                
        #         job_list = [job_list[i] for i in range(len(job_list)) if tobeRemoved[i] == 0]
                
        #         print("##########")
        #         print(f"Final jobs to be resubmitted: {job_list}")
        #         print("##########")
        #         # remove outputFile for jobs in job_list
        #         for inputFile in job_list:
        #             outputFile = resource_dict[inputFile]['outputFilename']
        #             print(f"Removing {outputFile}")
        #             os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
            
        #         if len(job_list) == 0:
        #             break
    
    # reset resource_suggestions.db and job lists
    finishedJobs = CANVAS.canvas.get('finished_job_list', [])
    finishedJobs += wasJobList
    CANVAS.canvas['finished_job_list'] = finishedJobs
    CANVAS.write('ready_to_run_job_list', [], overwrite=True)
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    os.remove(db_file)
    time.sleep(1)
    initialize_database(db_file)
    time.sleep(1)
    
    notConvergedListString = ""
    
    numberOfSucc = 0
    for job in job_list:
        try:
            # temporay disable the read function to avoid the calculation
            tmp = read(os.path.join(WORKING_DIRECTORY, job + '.pwo'))
            _ = tmp.get_potential_energy()
            print(f"Job {job} has finished")
            numberOfSucc += 1
        except:
            notConvergedListString += job + ", "
    
    if notConvergedListString != "":
        notConvergedListString = "However, the following jobs did not converge: " + notConvergedListString
    
    # if all job failed
    if numberOfSucc == 0:
        # time.sleep(60)
        return f"All jobs failed. Please figure out why they failed, then regenerate the job. Tell the supervisor in your response that new runs, with problems resolved, need to be regenerated and calculated."
    
    # time.sleep(60)
    return f"All job in job_list has finished. {notConvergedListString}please check the output file in the {WORKING_DIRECTORY}"

@tool
def submit_single_job(
    inputFile: str
) -> str:
    '''Submit a single job to supercomputer, return the location of the output file once the job is done'''
    print("checking pysqa prerequisites...")
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
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
        # time.sleep(60)
        return f"Resource suggestion for {inputFile} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    

    
    queueIDList = []


    ## Check if the input file exists
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, inputFile)):
        # time.sleep(60)
        return f"Input file {inputFile} does not exist, please use the find job list tool to submit the file in the job list"
    print("Generating batch script...")

    ## Check if the output file exists 
    if os.path.exists(os.path.join(WORKING_DIRECTORY, f"{inputFile}.pwo")):
        ## Supervisor sometimes ask to submit the job again, so we need to check if the output file exists
        # time.sleep(60)
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

mpirun pw.x -i {inputFile} > {inputFile}.pwo

echo " "
echo "Job Ended at `date`"
    """
        )
        
    if job_id is None:
        # time.sleep(60)
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

    # time.sleep(60)
    return f"Job has finished, please check the output file"   

@tool
def read_energy_from_output(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, energies of which will be read and printed"]
) -> str:
    '''Read the total energy from the output file in job list and return it in a string'''
    
    assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
    for i in jobFileIdx:
        assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # load job_list.jason
    job_list = CANVAS.canvas.get('finished_job_list', []).copy()
    job_list = np.array(job_list, dtype=str)[jobFileIdx]
    print(f"actual job list: {job_list}")
    
    result = ""
    for job in job_list:
        
        output_file = job + '.pwo'
        # print(f"Reading output file {output_file}")
        file_path = os.path.join(WORKING_DIRECTORY, output_file)
        # print(file_path)
        # Check if the output file exists
        if not os.path.exists(file_path):
            # time.sleep(60)
            return f"Output file {output_file} does not exist, please check the job list"
        try:
            atoms = read(file_path)
        except:
            try:
                # read in as text
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                # check if the output file contains "convergence NOT achieved"
                notConverge = False
                for line in lines:
                    if "convergence NOT achieved" in line:
                        notConverge = True
                        result += f"Job {job} did not converge\n"
                        break
                if not notConverge:
                    # time.sleep(60)
                    return f"Invalid output file {output_file} or calculation failed, please submit the {job} again."
            except:
                # time.sleep(60)
                return f"Invalid output file {output_file} or calculation failed, please submit the {job} again."
        result += f"Energy read from {job} is {atoms.get_potential_energy()} eV.\n"
        # print(result)
        time.sleep(1)
    print(result)
    # check input file in job list
    # file_path = os.path.join(WORKING_DIRECTORY, input_file)
    # atoms = read(file_path)
    # return f"Energy read from job {input_file} is {atoms.get_potential_energy()}"
        
    # time.sleep(60)
    return result


@tool
def read_single_output(
    input_file: str
) -> str:
    '''Read the total energy from the file in job list and return it in a string'''
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # load job_list.jason
    output_file = input_file + '.pwo'
    file_path = os.path.join(WORKING_DIRECTORY, output_file)
    # print(file_path)
    # Check if the output file exists
    if not os.path.exists(file_path):
        # time.sleep(60)
        return f"Output file {output_file} does not exist, please check the job list"
    try:
        atoms = read(file_path)
    except:
        # time.sleep(60)
        return f"Invalid output file {output_file} or calculation failed, please submit the {input_file} again."
    # time.sleep(60)
    return f"Energy read from job {input_file} is {atoms.get_potential_energy()}"