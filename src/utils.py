import sqlite3
import os,yaml
import pandas as pd
from xml.dom.minidom import Element
from typing import Callable, List, Literal
from pydantic import BaseModel
# from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import getpass
import pandas as pd
from src import var
from ase.io import read
import numpy as np
from ase.io import read,write
from ase.io.lammpsdata import write_lammps_data
import ase

def load_config(path: str):
    ## Load the configuration file
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ## Set up environment variables
    for key, value in config.items():
        var.OTHER_GLOBAL_VARIABLES[key] = value
    var.my_WORKING_DIRECTORY = config["WORKING_DIR"]
    return config
# def check_config(config: dict):
#     for key, value in config.items():
#         _set_if_undefined(key)
#     return 'Loaded config successfully'
class AtomsDict(BaseModel):
    numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]


# def _set_if_undefined(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"Please provide your {var}")

def save_graph_to_file(graph, path: str, name: str):
    try:
        im = graph.get_graph(xray=True).draw_mermaid_png()
        # print(graph.get_graph().draw_mermaid())
        with open(os.path.join(path, f"{name}.png"), "wb") as f:
            f.write(im)
        # print(f"Graph saved to {os.path.join(path, f'{name}.png')}")
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    return


def parse_qe_input_string(input_string):
    sections = ['control', 'system', 'electrons', 'ions', 'cell']
    input_data = {section: {} for section in sections}
    input_data['atomic_species'] = {}
    input_data['hubbard'] = {}
    
    lines = input_string.strip().split('\n')
    current_section = None
    atomic_species_section = False
    hubbard_section = False

    for line in lines:
        line = line.strip()

        if line.startswith('&') and line[1:].lower() in sections:
            current_section = line[1:].lower()
            atomic_species_section = False
            hubbard_section = False
            continue
        elif line == '/':
            current_section = None
            continue
        elif line.lower() == 'atomic_species':
            atomic_species_section = True
            hubbard_section = False
            continue
        elif line.lower() == 'hubbard (ortho-atomic)':
            hubbard_section = True
            atomic_species_section = False
            continue

        if current_section:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip("'")
                
                # Convert to appropriate type
                if value.lower() in ['.true.', '.false.']:
                    value = value.lower() == '.true.'
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                
                input_data[current_section][key] = value
        elif atomic_species_section:
            parts = line.split()
            if len(parts) == 3:
                input_data['atomic_species'][parts[0]] = {
                    'mass': float(parts[1]),
                    'pseudopotential': parts[2]
                }
        elif hubbard_section:
            parts = line.split()
            if len(parts) == 3:
                input_data['hubbard'][parts[1]] = float(parts[2])
    
    return input_data

element_list = ['Se', 'W', 'Rb', 'Cl', 'Bk', 'Ge', 'Mg', 'Pt', 'Tl', 'Ir', 'Pm', 'Fr', 'Er', 'Sb', 'Zn', 'Be', 'Rn', 'K', 'Dy', 'Es', 'Ar', 'Br', 'Hg'
                       , 'Pa', 'Nd', 'Li', 'Am', 'Te', 'Np', 'He', 'Os', 'In', 'Cu', 'Lr', 'Ga', 'Cs', 'Hf-sp', 'Si', 'Zr', 'Ac', 'U', 'At', 'Y', 'Po', 'Al'
                       , 'Fm', 'F', 'Nb', 'B', 'Cd', 'P', 'Ag', 'Ne', 'Au', 'No', 'Sc', 'Eu', 'Pd', 'Ni', 'Bi', 'Ce', 'Ho', 'Ru', 'Gd', 'I', 'As', 'Na', 'Th'
                       , 'Ca', 'Tc', 'Lu', 'Ta', 'Re', 'Cm', 'Md', 'Sn', 'Kr', 'Yb', 'La', 'Ra', 'Cr', 'Co', 'N', 'Pr', 'Rh', 'C', 'Cf', 'Tm', 'V', 'Sm', 'Pb', 
                       'H', 'O', 'Mo', 'Tb', 'Pu', 'Xe', 'Ti', 'Fe', 'S', 'Mn', 'Sr', 'Ba']


def filter_potential(input_data: dict) -> dict:
    pseudopotentials = {}
    for k,v in input_data['atomic_species'].items():
        if k in element_list:
            pseudopotentials[k] = v['pseudopotential']
    return pseudopotentials




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
#SBATCH -e {{errNoutName}}.err #change the name of the err file 
#SBATCH -o {{errNoutName}}.out # File to which STDOUT will be written %j is the job #

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

def select_k_ecut(convergence_data: pd.DataFrame, error_threshold: float, natom: int):
    """
    Select the k-point and ecut based on the provided error threshold from DFT convergence test results.

    Parameters:
    convergence_data (pd.DataFrame): A DataFrame containing the following columns:
                                     'k_point' (int/float), 'ecut' (int/float), 'total_energy' (float)
    error_threshold (float): The acceptable energy difference (absolute error threshold) in eV.

    Returns: 
    (int/float, int/float): The selected k-point and ecut values.
    """
    finnerEcut = False
    finnerKspacing = False
    
    # sorted_data = convergence_data.sort_values(by=['ecutwfc', 'kspacing'],ascending=[False,True])
    min_kspacing = convergence_data['kspacing'].min()
    max_ecutwfc = convergence_data['ecutwfc'].max()
    df_kspacing = convergence_data.loc[convergence_data['kspacing'] == min_kspacing].sort_values(by='ecutwfc',ascending=True)
    ## convert the energy to meV/atom
    df_kspacing['error'] = (df_kspacing['energy']-df_kspacing.iloc[-1]['energy']).abs()/natom*1000
    df_kspacing['Acceptable'] = df_kspacing['error'] < error_threshold  
    ecutwfc_chosen = df_kspacing[df_kspacing['Acceptable'] == True].iloc[0]['ecutwfc']
    print(df_kspacing)
    print(f'Chosen ecutwfc: {ecutwfc_chosen}')
    if ecutwfc_chosen == max_ecutwfc:
        finnerEcut = True


    df_ecutwfc = convergence_data.loc[convergence_data['ecutwfc'] == max_ecutwfc].sort_values(by='kspacing',ascending=False)
    ## convert the energy to meV/atom
    df_ecutwfc['error'] = (df_ecutwfc['energy']-df_ecutwfc.iloc[-1]['energy']).abs()/natom*1000
    df_ecutwfc['Acceptable'] = df_ecutwfc['error'] < error_threshold
    k_chosen = df_ecutwfc[df_ecutwfc['Acceptable'] == True].iloc[0]['kspacing']

    if k_chosen == min_kspacing:
        finnerKspacing = True
        
    print(df_ecutwfc)
    print(f'Chosen kspacing: {k_chosen}')


    return k_chosen, ecutwfc_chosen, finnerEcut, df_kspacing, df_ecutwfc, finnerKspacing


def initialize_database(db_file):
    # Connect to the SQLite database (create it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resources (
        filename TEXT PRIMARY KEY,
        partition TEXT,
        nnodes INTEGER,
        ntasks INTEGER,
        runtime INTEGER,
        submissionScript TEXT,
        outputFilename TEXT
    )
    ''')
    
    # Commit and close the connection for initialization
    conn.commit()
    conn.close()

def add_to_database(resource_dict, db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Insert or update each item in the resource_dict
    for filename, resources in resource_dict.items():
        cursor.execute('''
        INSERT OR REPLACE INTO resources (filename, partition, nnodes, ntasks, runtime, submissionScript, outputFilename)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename,
              resources['partition'],
              resources['nnodes'],
              resources['ntasks'],
              resources['runtime'],
              resources['submissionScript'],
              resources['outputFilename']))
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    
def read_BEEF_output(file_path: str):
    """
    Read the BEEF output file and extract relevant information.

    Args:
        file_path (str): Path to the BEEF output file.

    Returns:
        dict: Dictionary containing extracted information.
        or error info
    """
    with open(file_path,'r') as f:
        lines = f.readlines()
    atoms = read(file_path)
    reference = atoms.get_potential_energy()
    start_index = 0
    end_index = 0
    for i,line in enumerate(lines):
        if 'BEEFens 2000 ensemble energies' in line:
            start_index = i + 1
        if 'BEEF-vdW xc energy contributions' in line:
            end_index = i - 2

    if start_index == 0 or end_index == 0:
        return "WrongCalc"
    
    energies = []
    for i in range(start_index, end_index + 1):
        line = lines[i].split()
        energies.append(float(line[0])+reference)
    
    energies = np.array(energies)

    return energies


def generate_single_input_file(
    atoms_file_name,
    timestep,
    run_nsteps,
    temperature,
    elastic_tensor_component,
    strain,
    potential_dir,
    potential_file,
    input_file_name,
    dump_file_name,
    log_file_name,
    lammps_input_data = 'structure_input.data',
):
    # note down the cwd
    cwd = os.getcwd()
    
    # change the cwd to var.my_WORKING_DIRECTORY
    os.chdir(var.my_WORKING_DIRECTORY)
    
    atoms = read(atoms_file_name)
    # repeat number of atoms if len(atoms) is too small to ensure statistical significance of pressure
    if len(atoms) < 100:
        repeat_factor = np.floor((100 / len(atoms))**(1.0/3.0)).astype(int) + 1
        atoms = atoms.repeat((repeat_factor, repeat_factor, repeat_factor))
    atoms.positions = atoms.positions.round(4)
    atoms.cell = atoms.cell.round(4)
    lx = atoms.cell[0, 0]
    ly = atoms.cell[1, 1]
    lz = atoms.cell[2, 2]
    write_lammps_data(lammps_input_data, atoms, force_skew=True)
    if not dump_file_name:
        dump_file_name = f"dump_{strain:.3f}.lammpstrj"
    if not log_file_name:
        log_file_name = f"output_{strain:.3f}.log"
    input_content = ""
    input_content += f"log {log_file_name}\n"    
    input_content += f"""variable        nsteps          equal {run_nsteps:d}
variable        thermo_freq     equal 1000
variable        dump_freq       equal 10000
variable        restart_freq    equal 5000000
variable        nevery          equal 10000
variable        nrepeat         equal 10
variable        pres            equal 1.0000
variable        Tproduction     equal {temperature:.2f}
 
variable    strain      equal {strain:.3f}
 
# grab the old bounds
# variable    xlo_old     equal xlo
# variable    xhi_old     equal xhi
# variable    ylo_old     equal ylo
# variable    yhi_old     equal yhi
# variable    zlo_old     equal zlo
# variable    zhi_old     equal zhi
"""
    input_content += """variable        nfreq           equal ${nevery}*${nrepeat}
"""
    input_content += f"""variable        strain          equal {strain:.4f}
"""
    input_content += f"""
# define units
boundary p p p
atom_style atomic
kim init {potential_file} metal unit_conversion_mode
# initial structure
read_data "{lammps_input_data}"
"""
    elements = set([atom.symbol for atom in atoms])
    atomic_numbers = [ase.data.atomic_numbers[element] for element in elements]
    atomic_masses = [ase.data.atomic_masses[atomic_number] for atomic_number in atomic_numbers]
    kim_interaction_elements = " ".join(elements)
    input_content += f"""kim interactions {kim_interaction_elements}"""
    input_content += """
neighbor    0.3 bin
neigh_modify    delay 10
"""
    for i, mass in enumerate(atomic_masses):
        input_content += f"mass {i+1} {mass:.4f}\n"
    input_content += f"""\n"""
    input_content += f"timestep {timestep:.4f}\n"
 
    if elastic_tensor_component == 'C11C12':
        input_content += f"""change_box all x scale {1.0 + strain:.3f} y scale {1.0 + strain:.3f} z scale 1.0 remap units box\n"""
    elif elastic_tensor_component == 'C44':
        input_content += f"""change_box all xy delta {lx*strain/2.0:.3f} remap units box\n
        """
    input_content += """
# initial velocities
velocity all create 100.0 2845 rot yes dist gaussian
reset_timestep 0
fix nvt all nvt temp ${Tproduction} ${Tproduction} $(100.0*dt)
fix avp all ave/time 1 10 10 c_thermo_press mode vector ave running start 2000 file f_avp.out overwrite
compute msd all msd
thermo 10
thermo_style custom step time temp pe ke etotal fmax fnorm lx press pxx pyy pzz pxy pxz pyz c_msd[4] f_avp[1] f_avp[2] f_avp[3] f_avp[4] f_avp[5] f_avp[6]
"""
    input_content += f"""
dump 1 all custom 100 {dump_file_name} id type mass x y z fx fy fz vx vy vz
"""
    input_content += "restart ${restart_freq} nvt.restart\n"
    input_content += "run ${nsteps}\n"
    input_content += """
# Calculate stress
variable pxx1 equal f_avp[1]
variable pyy1 equal f_avp[2]
variable pzz1 equal f_avp[3]
variable pyz1 equal f_avp[4]
variable pxz1 equal f_avp[5]
variable pxy1 equal f_avp[6]
print "pxx1 = ${pxx1}"
print "pyy1 = ${pyy1}"
print "pzz1 = ${pzz1}"
print "pxy1 = ${pxy1}"
print "pxz1 = ${pxz1}"
print "pyz1 = ${pyz1}"
 
 
    """
    if not input_file_name:
        input_file_name = f"input_{strain:.3f}.in"
    with open(input_file_name, 'w') as f:
        f.write(input_content)
    print(f"Input file '{input_file_name}' generated successfully.")
    
    # change cwd to the original cwd
    os.chdir(cwd)
    
    print(f"lammps input data file '{lammps_input_data}' generated successfully.")
    return input_file_name