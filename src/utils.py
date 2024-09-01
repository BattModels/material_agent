import os,yaml
from xml.dom.minidom import Element
from typing import Callable, List, Literal
from langchain_core.pydantic_v1 import BaseModel


def load_config(path: str):
    ## Load the configuration file
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ## Set up environment variables
    os.environ['ANTHROPIC_API_KEY'] = config['ANTHROPIC_API_KEY']   
    os.environ["LANGSIM_PROVIDER"] = config['LANGSIM_PROVIDER']
    os.environ["LANGSIM_API_KEY"] = config['LANGSIM_API_KEY']
    os.environ["LANGSIM_MODEL"] = config['LANGSIM_MODEL']
    return config

class AtomsDict(BaseModel):
    numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]


def save_graph_to_file(graph, path: str):
    try:
        im = graph.get_graph(xray=True).draw_mermaid_png()
        with open(os.path.join(path, "graph.png"), "wb") as f:
            f.write(im)
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

element_list = Literal['Se', 'W', 'Rb', 'Cl', 'Bk', 'Ge', 'Mg', 'Pt', 'Tl', 'Ir', 'Pm', 'Fr', 'Er', 'Sb', 'Zn', 'Be', 'Rn', 'K', 'Dy', 'Es', 'Ar', 'Br', 'Hg'
                       , 'Pa', 'Nd', 'Li', 'Am', 'Te', 'Np', 'He', 'Os', 'In', 'Cu', 'Lr', 'Ga', 'Cs', 'Hf-sp', 'Si', 'Zr', 'Ac', 'U', 'At', 'Y', 'Po', 'Al'
                       , 'Fm', 'F', 'Nb', 'B', 'Cd', 'P', 'Ag', 'Ne', 'Au', 'No', 'Sc', 'Eu', 'Pd', 'Ni', 'Bi', 'Ce', 'Ho', 'Ru', 'Gd', 'I', 'As', 'Na', 'Th'
                       , 'Ca', 'Tc', 'Lu', 'Ta', 'Re', 'Cm', 'Md', 'Sn', 'Kr', 'Yb', 'La', 'Ra', 'Cr', 'Co', 'N', 'Pr', 'Rh', 'C', 'Cf', 'Tm', 'V', 'Sm', 'Pb', 
                       'H', 'O', 'Mo', 'Tb', 'Pu', 'Xe', 'Ti', 'Fe', 'S', 'Mn', 'Sr', 'Ba']