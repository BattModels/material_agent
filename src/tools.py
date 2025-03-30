
from matplotlib import pyplot as plt
from math import e
from src.utils import *
from src.myCANVAS import CANVAS
from ase import Atoms, Atom
from langchain.agents import tool
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

##################################################################################################
##                                        Common tools                                          ##
##################################################################################################
@tool
def inspect_my_canvas():
    """Inspect the working canvas to get available keys"""
    # get all keys in myCANVAS and return them as a list [key1, key2, ...]
    return CANVAS.inspect()

@tool
def read_my_canvas(key: str):
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
def init_structure_data(
    element: Annotated[str, "Element symbol"],
    lattice: Annotated[str, "Lattice type. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
    a: Annotated[float, "Lattice constant"],
    b: Annotated[float, "Lattice constant. If only a and b is given, b will be interpreted as c instead."] = None,
    c: Annotated[float, "Lattice constant"] = None,
) -> Annotated[str, "Path of the saved initial structure data file."]:
    """Create single element bulk initial structure based on composite, crystal lattice, lattice info, save to the working dir, and return filename."""
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    atoms = bulk(element, lattice, a=a, b=b, c=c, cubic=True)
    # atoms *= (2, 2, 2)

    # atoms.set_cell(atoms.cell * 0.95, scale_atoms=True)

    # write_lammps_data(os.path.join(WORKING_DIRECTORY, f'{element}.data'), atoms, masses=True)
    
    # return f"Initial structure data is created named {element}.data"
    
    # save the atoms into working dir
    saveDir = os.path.join(WORKING_DIRECTORY, f"{element}-{lattice}.xyz")
    write(saveDir, atoms)
    
    return f"Created atoms saved in {saveDir}"

@tool
def generateSurface_and_getPossibleSite(species: Annotated[str, "Element symbol"],
                                        crystal_structures: Annotated[str, "Crystal structure. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
                                        facets: Annotated[str, "Facet of the surface. Must be one of 100, 110, 111, 210, 211, 310, 311, 320, 321, 410, 411, 420, 421, 510, 511, 520, 521, 530, 531, 540, 541, 610, 611, 620, 621, 630, 631, 640, 641, 650, 651, 660, 661"],
                                        supercell_dim: Annotated[List[int], "typically [int, int, 4]. Supercell dimension, how many times do you want to repeat the primitive cell in each direction: [int, int, int]"],
                                        n_fixed_layers: Annotated[int, "typically 2. Number of fixed layers in the slab"] = 2
                                        ):
    """Generate a surface structure and get the available adsorption sites."""
    surface_dict = generate_surface_structures(
        species_list=[species],
        crystal_structures={species: crystal_structures},
        facets={species: [facets]},
        supercell_dim=supercell_dim,
        n_fixed_layers=n_fixed_layers,
        dirs_exist_ok=True,
        write_to_disk=True,
        write_location=os.environ.get("WORKING_DIR"),
    )
    
    mySurface = surface_dict[species][f'{crystal_structures}{facets}']["structure"]
    # mySites = get_adsorption_sites(mySurface, symm_reduce=0)
    # mySites = get_adsorption_sites(mySurface)
    mySites = mySurface.info['adsorbate_info']['sites']
    
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        print(mySites)
    
    mySites_str = output_capture.getvalue()
    
    CANVAS.write('Possible_CO_site_on_Pt_surface', mySites)
    
    return f"the surface generated is saved at {surface_dict[species][f'{crystal_structures}{facets}']['traj_file_path']}, available adsorbate sites are: {mySites_str}"

@tool
def generate_myAdsorbate(symbols: Annotated[str, "Element symbols of the adsorbate (Do not use any delimiters)"],
                         positions: Annotated[List[List[float]], "Positions of the atoms in the adsorbate, e.g. [[x1, y1, z1], [x2, y2, z2], ...], following the same order as the symbols."],
                         AdsorbateFileName: Annotated[str, "Name of the adsorbate file to be saved in traj format"]
                         ):
    """Generate an adsorbate structure and save it."""
    assert AdsorbateFileName.endswith('.traj'), "AdsorbateFileName should end with .traj"
    
    os.makedirs("adsorbates", exist_ok=True)
    tmpAtoms = Atoms(symbols=symbols, positions=positions)
    tmpAtoms.center(vacuum=10.0)
    write(os.path.join("adsorbates", f"{AdsorbateFileName}"), tmpAtoms)
    
    return f"Adsorbate saved at adsorbates/{AdsorbateFileName}"

@tool
def add_myAdsorbate(mySurfacePath: Annotated[str, "Path to the surface structure"],
                    adsorbatePath: Annotated[str, "Path to the adsorbate structure"],
                    mySites: Annotated[List[List[float]], "List of adsorption sites you want to put adsorbates on, e.g. [[x1, y1], [x2, y2], ...]"],
                    rotations: Annotated[List[Tuple[float, str]], "List of rotations for the ith adsorbates, e.g. [[90.0, 'x'], [180.0, 'y'], ...]"],
                    surfaceWithAdsorbateFileName: Annotated[str, "File name of the surface adsorbated with adsorbate to be saved in traj format"]
                    ):
    """
    Add adsorbate to the surface structure and save it.
    The third argument must be in the form of [[x1, y1], [x2, y2], ...], where x and y are the coordinates of the adsorption sites.
    The forth argument must be in the form of [[float(angle), str(axis)], ...], where the first element is the rotation angle and the second element is the axis of rotation.
    """
# @tool
# def add_myAdsorbate(mySurfacePath: Annotated[str, "Path to the surface structure"],
#                     adsorbatePath: Annotated[str, "Path to the adsorbate structure"],
#                     mySites: Annotated[List[List[float]], "List of adsorption sites you want to put adsorbates on, e.g. [[x1, y1], [x2, y2], ...]"],
#                     rotations: Annotated[List[List[str]], "List of rotations for the ith adsorbates, e.g. [['90.0', 'x'], ['180.0', 'y'], ...]"]
#                     ):
#     """
#     Add adsorbate to the surface structure and save it.
#     The third argument must be in the form of [[x1, y1], [x2, y2], ...], where x and y are the coordinates of the adsorption sites.
#     The forth argument must be in the form of [[str(angle), str(axis)], ...], where the first element is the rotation angle and the second element is the axis of rotation.
#     """
    assert surfaceWithAdsorbateFileName.endswith('.traj'), "surfaceWithAdsorbateFileName should end with .traj"
    # Load the surface structure
    mySurface = read(mySurfacePath)
    
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
    
    # save the new structure
    write(os.path.join(parentPath, surfaceWithAdsorbateFileName), mySurface)
    
    return f"Surface with adsorbate saved at {parentPath}/{surfaceWithAdsorbateFileName}"

@tool
def write_script(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "Name of the file to be saved."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the quantum espresso input script to the specified file path"""
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
def write_QE_script_w_ASE(
    listofElements: Annotated[List[str], "List of distinct element symbols in the unit cell"],
    ppfiles: Annotated[List[str], "List of pseudopotential files in the order of the elements"],
    pseudo_dir: Annotated[str, "Path to the directory containing the pseudopotentials"],
    filename: Annotated[str, "Name of the Quantum Espresso input file, end with .pwi"],
    inputAtomsDir: Annotated[str, "Directory of the input Atoms object"],
    calculation: Annotated[str, "Type of calculation to perform, e.g. 'scf' or 'relax'"],
    restart_mode: Annotated[Literal['from_scratch', 'restart'], "Restart mode"],
    prefix: Annotated[str, "Prefix for the output files"],
    outdir: Annotated[str, "Path to the directory where the output files are written"],
    disk_io: Annotated[Literal['none', 'minimal', 'nowf', 'low', 'medium', 'high'], "Disk I/O level"],
    ibrav: Annotated[int, "Bravais-lattice index. Optional only if space_group is set."],
    nat: Annotated[int, "Number of atoms in the unit cell"],
    ntyp: Annotated[int, "Number of atom types in the unit cell"],
    ecutwfc: Annotated[float, "kinetic energy cutoff (Ry) for wavefunctions"],
    ecutrho: Annotated[float, "Kinetic energy cutoff (Ry) for charge density and potential"],
    occupations: Annotated[Literal['smearing', 'tetrahedra', 'tetrahedra_lin', 'tetrahedra_opt', 'fixed', 'from_input'], "Occupation type"],
    smearing: Annotated[Literal['gaussian', 'methfessel-paxton', 'marzari-vanderbilt', 'fermi-dirac'], "Smearing type"],
    degauss: Annotated[float, "value of the gaussian spreading (Ry) for brillouin-zone integration in metals."],
    conv_thr: Annotated[float, "Convergence threshold for self-consistent loop"],
    kpts: Annotated[List[int], "Number of k-points along each reciprocal lattice vector. it would be a list of int: [int, int, int]"],
    ready_to_run_job: Annotated[bool, "True if the job is intended to be run directly without further modification, False if this file is intended to be used to generate other files"] = False
):
    """Write a Quantum Espresso input script using ASE."""

    # assemble the pseudopotentials dict from the list of elements and pseudopotentials
    pseudopotentials = {}
    for element, pseudo in zip(listofElements, ppfiles):
        pseudopotentials[element] = pseudo
    
    # Create a dummy Atoms object
    atoms = read(inputAtomsDir)
    
    filenameWDir = os.path.join(os.environ.get("WORKING_DIR"), filename)

    # Write the input script
    write(filenameWDir,
          atoms,
          input_data={
              'control': {
                  'calculation': calculation,
                  'restart_mode': restart_mode,
                  'prefix': prefix,
                  'pseudo_dir': "/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/all_lda_pbe_UPF",
                  'outdir': './out',
                  'disk_io': disk_io,
              },
              'system': {
                  'ibrav': ibrav,
                  'nat': nat,
                  'ntyp': ntyp,
                  'ecutwfc': ecutwfc,
                  'ecutrho': ecutrho,
                  'occupations': occupations,
                  'smearing': smearing,
                  'degauss': degauss,
              },
              'electrons': {
                  'conv_thr': conv_thr,
              }
          },
          format='espresso-in',
          pseudopotentials=pseudopotentials,
          kpts=tuple(kpts)
          )
    
    
    if not ready_to_run_job:
        destiJobList = 'scratch_job_list'
    else:
        destiJobList = 'ready_to_run_job_list'
    
    job_list = [filename]
    old_job_list = CANVAS.canvas.get(destiJobList, []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write(destiJobList,job_list, overwrite=True)
    
    return f"Quantum Espresso input script is written to {filename}"

@tool
def write_LAMMPS_script(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "Name of the file to be saved."],
) -> Annotated[str, "Path of the saved document file."]:
    """Save the LAMMPS input script to the specified file path"""
    ## Error when '/' in the content, manually delete
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    
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
        
    return f"Initial file is created named {file_name}"

@tool
def find_classical_potential(element: str) -> str:
    """Return classical potential file path for given element symbol."""
    return f'The classcial potential file for {element} is located at /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/EAM/Li_v2.eam.fs'


@tool
def find_pseudopotential(element: str) -> str:
    """Return the pseudopotential file path for given element symbol."""
    spList = []
    pseudo_dir = os.environ.get("PSEUDO_DIR")
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
        return ans
    else:
        return f"Could not find pseudopotential for {element}"

@tool
def generate_convergence_test(input_file_name:str,kspacing:list[float], ecutwfc:list[int]):
    '''
    Generate the convergence test input scripts for quantum espresso calculation and save the job list. 

    Input:  input_file_name: str, the name of the input file
            kspacing: list[float], the list of kspacing to be tested
            ecutwfc: list[int], the list of ecutwfc to be tested
    '''
    # kspacing = [0.6, 0.8, 1.0]
    # ecutwfc = [10, 20, 30]
    
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
            new_file_name = f'{os.path.splitext(input_file_name)[0]}_k_{k}_ecutwfc_{ecutwfc_max}.pwi'
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
    return f"Job list is saved scucessfully, continue to submit the jobs"


@tool
def generate_eos_test(input_file_name:str,kspacing:float, ecutwfc:int):
    '''
    Generate the equation of state test input scripts for quantum espresso calculation and save the job list.
    
    Input:  input_file_name: str, the name of the input file
            kspacing: float, the kspacing to be tested
            ecutwfc: int, the ecutwfc to be tested
    '''
    # CANVAS.write('job_list', [], overwrite=True)
    CANVAS.canvas['jobs_K_and_ecut'] = {}
    
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    input_file = os.path.join(WORKING_DIRECTORY, input_file_name)
    prefix = input_file_name.split('.')[0]
    # Read the atom object from the input script
    try:
        atom = read(input_file)
    except:
        INITIAL_FILE = os.environ.get("INITIAL_FILE")
        return f"Invalid input file, try inspect the shared CANVAS and use the inital pwi file as the input file"
    
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
    print(job_list)
    ## Save the job list as json file
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
    
    return f"Job list is saved scucessfully, continue to submit the jobs"

###################################### DFT POST-PROCESSING TOOLS ######################################

@tool
def calculate_formation_E(slabFilePath: Annotated[str, "the slab calculation output file path"],
                          adsorbateFilePath: Annotated[str, "the adsorbate calculation output file path"],
                          systemFilePath: Annotated[str, "the slab with adsorbate calculation output file path"]
                          ):
    """using the energies of the slab, adsorbate, and slab with adsorbate, calculate the formation energy of the adsorbate on the slab. """
    working_directory = os.environ.get("WORKING_DIR")
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
    
    return f"The formation energy of the adsorbate on the slab is {formationEnergy} eV"

@tool
def calculate_lc(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, which will be used to calculate the lattice constant"]
    ) -> str:
    """Read the output file and calculate the lattice constant"""
    
    assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
    for i in jobFileIdx:
        assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    job_list = CANVAS.canvas.get('finished_job_list', []).copy()
    job_list = np.array(job_list, dtype=str)[jobFileIdx]
    print(f"actual job list: {job_list}")

    volume_list = []
    energy_list = []
    for job in job_list:
        print(f'reading {job}')
        atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
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

    return f'The lattice constant is {lc}'

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
def get_kspacing_ecutwfc(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, which will be used to determine the kspacing and ecutwfc"],
                         threshold: Annotated[float, "the threshold mev/atom to determine the convergence"] = 1.0) -> str:
    '''Read the convergen test result and determine the kspacing and ecutwfc used in the production
    Input:
        jobFileIdx: list, the indexs of files in the finished job list, which will be used to determine the kspacing and ecutwfc
        threshold: float , the threshold mev/atom to determine the convergence
    output: str, the kspacing and ecutwfc used in the production
    '''
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    
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
    
    print(f"successfully read {len(kspacing)} kspacing and {len(ecutwfc)} ecutwfc")
        
    convergence_df = pd.DataFrame({'job':job_list,'kspacing':kspacing, 'ecutwfc':ecutwfc, 'energy':energy_list})
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
    
    ans = f"Please use kspacing {k_chosen} and ecutwfc {ecutwfc_chosen} for the production calculation"
    
    if finnerEcut and finnerKspacing:
        ans += f"\nHowever, the calculation is not converged, please consider redo the convergence test and using a finner ecutwfc and finner kspacing"
    elif finnerEcut:
        ans += f"\nHowever, the calculation is not converged, please consider redo the convergence test and using a finner ecutwfc"
    elif finnerKspacing:
        ans += f"\nHowever, the calculation is not converged, please consider redo the convergence test and using a finner kspacing"
    
    return ans

##################################################################################################
##                                          HPC tools                                           ##
##################################################################################################

@tool
def find_job_list() -> str:
    """Return the list of job files to be submitted."""

    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
    job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    
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

# @tool
# def add_resource_suggestion(
#     qeInputFileName: str,
#     partition: str,
#     nnodes: int,
#     ntasks: int,
#     runtime: Annotated[str, "Time limit for the job, in minutes"],
# ) -> Annotated[str, "source suggestion saved location"]:
#     """
#     After agent generate resource suggestions based on the QE input file, add it to the json file "resource_suggestions.json" in the WORKING_DIRECTORY.
#     For example: {"input1.pwi": {"nnodes": 2, "ntasks": 4, "runtime": 60}, "input2.pwi": {"nnodes": 1, "ntasks": 2, "runtime": 30}}
#     """
#     if not isinstance(partition, str) or not isinstance(nnodes, int) or not isinstance(ntasks, int) or not isinstance(runtime, str):
#         return "Invalid input, please check the input format"
#     # craete the json file if it does not exist, otherwise load it
#     WORKING_DIRECTORY = os.environ.get("WORKING_DIR")

#     new_resource_dict = {qeInputFileName: {"partition": partition, "nnodes": 1, "ntasks": 4, "runtime": 30}}

    
#     # check if resource_suggestions.db exist in the working directory
#     db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
#     if not os.path.exists(db_file):
#         initialize_database(db_file)
    
#     add_to_database(new_resource_dict, db_file)
    
#     return f"Resource suggestion for {qeInputFileName} saved scucessfully"

@tool
def add_resource_suggestion(
    qeInputFileName: str,
    partition: str,
    nnodes: int,
    ntasks: int,
    runtime: Annotated[str, "Time limit for the job, in minutes"],
    submissionScript: Annotated[str, "submission script based on the types of jobs. output filename must be <full input filename with extension>.<output_file_type>"],
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
        return "Invalid input, please check the input format"
    # craete the json file if it does not exist, otherwise load it
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")

    new_resource_dict = {qeInputFileName: {"partition": "venkvis-cpu", "nnodes": 1, "ntasks": 96, "runtime": 1440, "submissionScript": submissionScript, "outputFilename": outputFilename}}
    
    # check if resource_suggestions.db exist in the working directory
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    if not os.path.exists(db_file):
        initialize_database(db_file)
    
    add_to_database(new_resource_dict, db_file)
    
    return f"Resource suggestion for {qeInputFileName} saved scucessfully"


@tool
def submit_and_monitor_job(
    jobType: Annotated[str, "The type of job to be submitted, e.g. DFT, LAMMPS"]
    ) -> str:
    '''
    Submit jobs in the job list to supercomputer, return the location of the output file once the job is done
    '''
    
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
        
    job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    
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
        
        if jobType == "DFT":
            print("Checking jobs")
            
            checked = set()
            unchecked = set(job_list)
            while checked != unchecked:
                for inputFile in job_list:
                    outputFile = resource_dict[inputFile]['outputFilename']
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
                            # temporay disable remove to avoid the calculation
                            # os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
                            print(f"{outputFile} removed")
                        except:
                            print("output file does not exist")
                        print(f"Job {inputFile} failed, will resubmit the job")
            
            
            # for idx, inputFile in enumerate(job_list):
            #     outputFile = resource_dict[inputFile]['outputFilename']
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
                job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
                
                # read all energies into a dict
                energies = {}
                for inputFile in job_list:
                    outputFile = resource_dict[inputFile]['outputFilename']
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
                    outputFile = resource_dict[inputFile]['outputFilename']
                    print(f"Removing {outputFile}")
                    os.remove(os.path.join(WORKING_DIRECTORY, outputFile))
            
                if len(job_list) == 0:
                    break
    
    # reset resource_suggestions.db and job lists
    finishedJobs = CANVAS.canvas.get('finished_job_list', [])
    readyToRunJobs = CANVAS.canvas.get('ready_to_run_job_list', [])
    finishedJobs += readyToRunJobs
    CANVAS.write('finished_job_list', finishedJobs, overwrite=True)
    CANVAS.write('ready_to_run_job_list', [], overwrite=True)
    db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    os.remove(db_file)
    time.sleep(1)
    initialize_database(db_file)
    time.sleep(1)
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

mpirun pw.x -i {inputFile} > {inputFile}.pwo

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
def read_energy_from_output(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, energies of which will be read and printed"]
) -> str:
    '''Read the total energy from the output file in job list and return it in a string'''
    
    assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
    for i in jobFileIdx:
        assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
    WORKING_DIRECTORY = os.environ.get("WORKING_DIR")
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
            return f"Output file {output_file} does not exist, please check the job list"
        try:
            atoms = read(file_path)
        except:
            return f"Invalid output file {output_file} or calculation failed, please submit the {job} again."
        result += f"Energy read from {job} is {atoms.get_potential_energy()}.\n"
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