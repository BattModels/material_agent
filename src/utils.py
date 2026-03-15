import sqlite3
import os,yaml
import pandas as pd
from xml.dom.minidom import Element
from typing import Callable, List, Literal, Any, Optional
from pydantic import BaseModel, Field
# from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import getpass
import pandas as pd
from src import var
from ase.io import read
import numpy as np

def load_config(path: str):
    ## Load the configuration file
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ## Set up environment variables
    for key, value in config.items():
        var.OTHER_GLOBAL_VARIABLES[key] = value
    var.my_WORKING_DIRECTORY = config["WORKING_DIR"]
    try:
        import torch
        print("Setting GPU_AVAILABLE == True")
        var.GPU_AVAILABLE = torch.cuda.is_available()
        print(f"GPU_AVAILABLE: {var.GPU_AVAILABLE}")
    except ImportError:
        var.GPU_AVAILABLE = False
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
    print(f"there are {len(df_kspacing)} data points for kspacing {min_kspacing}")
    if len(df_kspacing) <= 1:
        finnerEcut = True


    df_ecutwfc = convergence_data.loc[convergence_data['ecutwfc'] == max_ecutwfc].sort_values(by='kspacing',ascending=False)
    ## convert the energy to meV/atom
    df_ecutwfc['error'] = (df_ecutwfc['energy']-df_ecutwfc.iloc[-1]['energy']).abs()/natom*1000
    df_ecutwfc['Acceptable'] = df_ecutwfc['error'] < error_threshold
    k_chosen = df_ecutwfc[df_ecutwfc['Acceptable'] == True].iloc[0]['kspacing']

    if len(df_ecutwfc) <= 1:
        finnerKspacing = True

    print(df_ecutwfc)
    print(f'Chosen kspacing: {k_chosen}')
    print(f"there are {len(df_ecutwfc)} data points for ecutwfc {max_ecutwfc}")


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

# Keep operators explicit and finite (avoid eval / df.query injection).
Op = Literal[
    "eq", "ne", "gt", "ge", "lt", "le",
    "in", "not_in",
    "contains", "not_contains",
    "regex", "not_regex",
    "isnull", "notnull",
    "contains_any", "not_contains_any",
    "contains_all", "not_contains_all",
]


class Filter(BaseModel):
    column: str = Field(description="Column name to filter on.")
    op: Op = Field(description="Filter operation.")
    value: Optional[Any] = Field(
        default=None,
        description="Value for the operation.\n"
            "Expected types by op:\n"
            "- isnull, notnull: no value needed (omit or set null).\n"
            "- eq, ne: scalar matching the column dtype (e.g., str, int, float, bool).\n"
            "- gt, ge, lt, le: numeric scalar (int/float) or other comparable scalar matching the column dtype.\n"
            "- in, not_in: list-like (e.g., list[str], list[int], list[float]); pass a list even for one item.\n"
            "- contains, not_contains: str substring/pattern (uses pandas string contains).\n"
            "- regex, not_regex: str regex pattern.\n"
            "- contains_any, not_contains_any, contains_all, not_contains_all: list[str] "
            "(for list-like columns such as Elements: List[str]).",
    )
    
class SortSpec(BaseModel):
    column: str = Field(description="Column name to sort by.")
    ascending: bool = Field(default=True, description="Sort ascending if True else descending.")


def _apply_filter(frame: pd.DataFrame, f) -> pd.DataFrame:
    allowed_cols = set(frame.columns)
    if f.column not in allowed_cols:
        raise ValueError(f"Unknown column: {f.column}. Allowed: {sorted(allowed_cols)}")

    s = frame[f.column]

    if f.op in ("isnull", "notnull"):
        mask = s.isna() if f.op == "isnull" else s.notna()
        return frame[mask]

    if f.value is None:
        raise ValueError(f"Filter op '{f.op}' requires a value.")

    val = f.value

    # List-like column support (your 'Elements' is List[str])
    if f.op in ("contains_any", "not_contains_any", "contains_all", "not_contains_all"):
        if not isinstance(val, list):
            raise ValueError(f"'{f.op}' requires a list value.")
        target = set(val)

        def ok(x):
            if not isinstance(x, list):
                return False
            xs = set(x)
            if f.op in ("contains_any", "not_contains_any"):
                return len(xs & target) > 0
            else:  # contains_all / not_contains_all
                return target.issubset(xs)

        mask = s.apply(ok)
        if f.op.startswith("not_"):
            mask = ~mask
        return frame[mask]

    # String ops
    if f.op == "contains":
        return frame[s.astype(str).str.contains(str(val), na=False)]

    if f.op == "not_contains":
        return frame[~s.astype(str).str.contains(str(val), na=False)]

    if f.op == "regex":
        pattern = re.compile(str(val))
        return frame[s.astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

    if f.op == "not_regex":
        pattern = re.compile(str(val))
        return frame[~s.astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

    # Membership
    if f.op == "in":
        if not isinstance(val, list):
            raise ValueError("'in' requires a list value.")
        return frame[s.isin(val)]
    if f.op == "not_in":
        if not isinstance(val, list):
            raise ValueError("'not_in' requires a list value.")
        return frame[~s.isin(val)]

    # Numeric comparisons (works for floats/ints; NaNs drop out naturally)
    if f.op == "eq":
        return frame[s == val]
    if f.op == "ne":
        return frame[s != val]
    if f.op == "gt":
        return frame[s > val]
    if f.op == "ge":
        return frame[s >= val]
    if f.op == "lt":
        return frame[s < val]
    if f.op == "le":
        return frame[s <= val]

    raise ValueError(f"Unsupported op: {f.op}")

def df_query(
    frame: pd.DataFrame,
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    select: Optional[List[str]] = None,
) -> str:
    """Filter/sort/select rows from the materials dataframe and return a paginated result."""

    allowed_cols = set(frame.columns)
    
    # Apply filters (AND semantics)
    for f in filters:
        frame = _apply_filter(frame, f)

    # Apply sort
    if sort:
        for s in sort:
            if s.column not in allowed_cols:
                raise ValueError(f"Unknown sort column: {s.column}")
        frame = frame.sort_values(
            by=[s.column for s in sort],
            ascending=[s.ascending for s in sort],
            kind="mergesort",  # stable
        )

    # Select columns
    if select is not None:
        unknown = [c for c in select if c not in allowed_cols]
        if unknown:
            raise ValueError(f"Unknown columns in select: {unknown}")
        frame = frame[select]

    return frame