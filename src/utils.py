import sqlite3
import os,yaml
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

# mc_sa_alloy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from copy import deepcopy

# ASE imports (only used for optional relaxation)
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter  # if you ever want cell relax
from ase.io import write
# from ase.filters import StrainFilter     # alternative cell filter

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


kB = 8.617333262145e-5  # eV/K

@dataclass
class AnnealSettings:
    target_accept_uphill: float = 0.8   # aim ~0.7–0.9 for robust exploration
    alpha: float = 0.90                 # geometric cooling: T_{i+1} = alpha*T_i
    min_temperature: float = 1.0        # K; stop when below this
    sweeps_per_T: int = 20              # 1 sweep = N attempted swaps
    max_T_steps: int = 200              # safety cap
    seed: Optional[int] = None

@dataclass
class RelaxSettings:
    do_relax: bool = True
    fmax: float = 0.02   # eV/Å
    steps: int = 500

class CanonicalSwapAnnealer:
    """
    Canonical (NVT, fixed composition) Metropolis Monte Carlo using species-swap moves
    with simulated annealing to target low-energy orderings.

    - Composition is conserved by swapping chemical symbols of two atoms with different species.
    - Energy differences are computed with the provided ASE calculator.
    - Optional 0 K local relaxation at the end with FIRE.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure. Composition is fixed by its current symbols.
    calculator : ase.Calculator
        Should provide get_potential_energy/forces/stress (placeholder 'myCalc' in your setup).
    allowed_pairs : Optional[List[Tuple[str,str]]]
        If provided, restrict swaps to these unordered species pairs (e.g., [('Li','Mg'), ('Li','Na')]).
        If None, any pair of different species in the structure can be swapped.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        allowed_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.atoms0 = atoms
        self.calc = calculator
        self.allowed_pairs = None
        if allowed_pairs is not None:
            # Normalize to frozensets for fast membership
            self.allowed_pairs = {frozenset(p) for p in allowed_pairs}

    # ---------- utility ----------
    @staticmethod
    def _random_two_of_diff_species(symbols: np.ndarray, rng: np.random.Generator,
                                    allowed_pairs: Optional[set]) -> Tuple[int, int]:
        # map species to indices
        species_to_idx: Dict[str, List[int]] = {}
        for i, s in enumerate(symbols):
            species_to_idx.setdefault(s, []).append(i)
        species = list(species_to_idx.keys())
        if len(species) < 2:
            raise ValueError("Need at least two species to perform swap moves.")
        # choose two distinct species (respect allowed_pairs if given)
        tries = 0
        while True:
            s1, s2 = rng.choice(species, size=2, replace=False)
            if allowed_pairs is None or frozenset((s1, s2)) in allowed_pairs:
                i = rng.choice(species_to_idx[s1])
                j = rng.choice(species_to_idx[s2])
                return int(i), int(j)
            tries += 1
            if tries > 1000:
                raise RuntimeError("Could not sample an allowed pair for swapping.")

    @staticmethod
    def _energy(atoms: Atoms) -> float:
        return atoms.get_potential_energy()

    @staticmethod
    def _swap_symbols_inplace(atoms: Atoms, i: int, j: int):
        syms = atoms.get_chemical_symbols()
        si, sj = syms[i], syms[j]
        syms[i], syms[j] = sj, si
        atoms.set_chemical_symbols(syms)

    # ---------- temperature heuristics ----------
    def _estimate_initial_temperature(
        self, atoms: Atoms, settings: AnnealSettings, n_trials: int = 200
    ) -> float:
        """
        Probe random swaps to estimate a T0 that yields a target uphill-acceptance probability.
        Uses mean positive ΔE: T0 ≈ - ⟨ΔE | ΔE>0⟩ / ln(p_target).
        """
        rng = np.random.default_rng(settings.seed)
        work = atoms.copy()
        work.calc = self.calc
        base_E = self._energy(work)
        dEs_pos = []

        # collect independent ΔE samples via propose-revert
        for _ in range(n_trials):
            i, j = self._random_two_of_diff_species(
                np.array(work.get_chemical_symbols()), rng, self.allowed_pairs
            )
            self._swap_symbols_inplace(work, i, j)
            E_new = self._energy(work)
            dE = E_new - base_E
            # revert
            self._swap_symbols_inplace(work, i, j)
            if dE > 0:
                dEs_pos.append(dE)

        if not dEs_pos:
            # no uphill moves seen - pick a modest T0
            return 500.0

        mean_pos = float(np.mean(dEs_pos))
        p = np.clip(settings.target_accept_uphill, 1e-6, 0.999999)
        T0 = -mean_pos / (kB * np.log(p))
        # guardrails
        T0 = float(np.clip(T0, 50.0, 5000.0))
        return T0

    # ---------- main driver ----------
    def run(
        self,
        anneal: AnnealSettings = AnnealSettings(),
        relax: RelaxSettings = RelaxSettings(),
        verbose: bool = True,
    ) -> Tuple[Atoms, Dict[str, np.ndarray]]:
        """
        Returns
        -------
        best_atoms : Atoms
            Lowest-energy structure found (after optional final relaxation).
        log : dict
            'T' (K), 'E' (eV), 'accept_rate' per temperature step, and 'best_E_progress'.
        """
        rng = np.random.default_rng(anneal.seed)

        state = deepcopy(self.atoms0)
        state.calc = self.calc
        symbols = np.array(state.get_chemical_symbols())

        E = self._energy(state)
        best_atoms = state.copy()
        best_E = E

        T0 = self._estimate_initial_temperature(state, anneal)
        if verbose:
            print(f"[SA] Initial temperature guess T0 = {T0:.1f} K")

        T_vals, E_vals, acc_vals, best_progress, MCtraj = [], [], [], [], []

        T = T0
        n = len(state)
        frozen_count = 0

        for tstep in range(anneal.max_T_steps):
            accepts = 0
            attempts = 0

            sweeps = max(1, int(anneal.sweeps_per_T))
            total_moves = sweeps * n

            if verbose:
                print(f"############################ starting T step {tstep+1} at {T:8.2f} K ############################")
            for currStep in range(total_moves):
                attempts += 1
                i, j = self._random_two_of_diff_species(symbols, rng, self.allowed_pairs)
                # swap, evaluate, Metropolis, maybe keep
                self._swap_symbols_inplace(state, i, j)
                E_new = self._energy(state)
                dE = E_new - E

                if dE <= 0.0:
                    accept = True
                else:
                    p_acc = np.exp(-dE / (kB * T))
                    accept = rng.random() < p_acc

                if accept:
                    accepts += 1
                    E = E_new
                    symbols[i], symbols[j] = symbols[j], symbols[i]  # keep symbols array in sync
                    if E < best_E:
                        best_E = E
                        best_atoms = state.copy()
                else:
                    # revert swap
                    self._swap_symbols_inplace(state, i, j)
                    
                if currStep % n == 0:
                    if verbose:
                        print(f"[SA]({currStep}/{total_moves}) T={T:8.2f} K | E={E: .6f} eV | accept so far={accepts/max(1,attempts)*100:5.1f}% | best={best_E: .6f}", end='\r')
                #     MCtraj.append(state.copy())
            MCtraj.append(state.copy())

            acc_rate = accepts / max(1, attempts)
            T_vals.append(T)
            E_vals.append(E)
            acc_vals.append(acc_rate)
            best_progress.append(best_E)

            if verbose:
                print(f"############################ result at {T:8.2f} K ############################")
                print(f"[SA] T={T:8.2f} K | E={E: .6f} eV | accept={acc_rate*100:5.1f}% | best={best_E: .6f}")
                print("\n")

            # early freeze detection: if essentially frozen for a few T steps, stop
            if acc_rate < 0.01:
                frozen_count += 1
            else:
                frozen_count = 0
            if frozen_count >= 3:
                if verbose:
                    print("[SA] System frozen for 3 consecutive temperatures. Stopping.")
                break

            # cool
            T = T * anneal.alpha
            if T < anneal.min_temperature:
                if verbose:
                    print(f"[SA] Reached min temperature {anneal.min_temperature} K. Stopping.")
                break

        # optional 0 K local relaxation
        if relax.do_relax:
            if verbose:
                print("[RELAX] Running 0 K local minimization (FIRE).")
            best_atoms_relax = best_atoms.copy()
            best_atoms_relax.calc = self.calc
            dyn = FIRE(best_atoms_relax, logfile=None)
            dyn.run(fmax=relax.fmax, steps=relax.steps)
            E_relaxed = self._energy(best_atoms_relax)
            if E_relaxed <= best_E:
                best_atoms = best_atoms_relax
                best_E = E_relaxed
                if verbose:
                    print(f"[RELAX] Relaxed energy {best_E: .6f} eV adopted.")
            else:
                if verbose:
                    print(f"[RELAX] Relaxation increased energy; keeping pre-relax structure.")

        log = {}
        log['T'] = np.array(T_vals)
        log['E'] = np.array(E_vals)
        log['accept_rate'] = np.array(acc_vals)
        log['best_E_progress'] = np.array(best_progress)
        log['best_E'] = best_E
        log['MCtraj'] = MCtraj
        
        return best_atoms, log

# ---- add below your existing code in mc_sa_alloy.py ----
from collections import OrderedDict

@dataclass
class CoupledRelaxSettings:
    mode: str = "propose"          # "propose" (recommended), "accept", or "interval"
    interval: int = 10             # if mode == "interval": relax after every N accepted moves
    fmax: float = 0.03             # eV/Å for per-move relax (slightly looser than final polish)
    steps: int = 200               # keep cheap per-move relax
    use_cell_relax: bool = False   # if True, relax cell as well as positions
    constant_volume: bool = True   # respected only if use_cell_relax is True
    scalar_pressure: float = 0.0   # GPa, only if use_cell_relax
    cache_size: int = 256          # LRU cache on symbol patterns to skip repeated relaxations

class CoupledRelaxSwapAnnealer:
    """
    Canonical swap Monte Carlo with local relaxation coupled to moves.

    mode="propose": basin-hopping style. For each proposed swap, locally relax the candidate,
    compute energy of the relaxed candidate, then do Metropolis with those relaxed energies.

    mode="accept": do Metropolis using unrelaxed energies as a cheap filter, then relax only if accepted
    (faster but not strictly detailed-balanced on the minimized surface).

    mode="interval": accept using unrelaxed energies, and perform a local relaxation after every
    'interval' accepted moves. Useful when relaxations are very expensive.

    Notes:
    - The per-move relax uses FIRE by default. You can switch to BFGS if that is better for your calculator.
    - If use_cell_relax is True, UnitCellFilter wraps atoms for joint cell+position relax.
      Consider constant_volume=True for alloys on a fixed lattice.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        allowed_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.atoms0 = atoms
        self.calc = calculator
        self.allowed_pairs = None
        if allowed_pairs is not None:
            self.allowed_pairs = {frozenset(p) for p in allowed_pairs}
        # simple LRU cache: key is tuple(symbols); value is (energy, positions, cell)
        self._cache: "OrderedDict[Tuple[str,...], Tuple[float, np.ndarray, np.ndarray]]" = OrderedDict()

    # ---------- internals ----------
    @staticmethod
    def _symbols_key(atoms: Atoms) -> Tuple[str, ...]:
        return tuple(atoms.get_chemical_symbols())

    def _cache_get(self, key):
        if key in self._cache:
            e, pos, cell = self._cache[key]
            # refresh LRU order
            self._cache.move_to_end(key)
            return e, pos.copy(), cell.copy()
        return None

    def _cache_set(self, key, e, pos, cell, cache_size: int):
        self._cache[key] = (float(e), pos.copy(), cell.copy())
        self._cache.move_to_end(key)
        while len(self._cache) > cache_size:
            self._cache.popitem(last=False)

    def _relax_local(
        self,
        atoms: Atoms,
        relax: CoupledRelaxSettings,
    ) -> float:
        # Optionally wrap with UnitCellFilter
        if relax.use_cell_relax:
            ucf = UnitCellFilter(
                atoms,
                constant_volume=relax.constant_volume,
                scalar_pressure=relax.scalar_pressure,
            )
            dyn = FIRE(ucf, logfile=None)
            dyn.run(fmax=relax.fmax, steps=relax.steps)
        else:
            dyn = FIRE(atoms, logfile=None)
            dyn.run(fmax=relax.fmax, steps=relax.steps)
        return atoms.get_potential_energy()

    @staticmethod
    def _swap_symbols_inplace(atoms: Atoms, i: int, j: int):
        syms = atoms.get_chemical_symbols()
        si, sj = syms[i], syms[j]
        syms[i], syms[j] = sj, si
        atoms.set_chemical_symbols(syms)

    @staticmethod
    def _choose_swap_indices(symbols: np.ndarray, rng: np.random.Generator,
                             allowed_pairs: Optional[set]) -> Tuple[int, int]:
        species_to_idx: Dict[str, List[int]] = {}
        for idx, s in enumerate(symbols):
            species_to_idx.setdefault(s, []).append(idx)
        species = list(species_to_idx.keys())
        if len(species) < 2:
            raise ValueError("Need at least two species to perform swap moves.")
        tries = 0
        while True:
            s1, s2 = rng.choice(species, size=2, replace=False)
            if allowed_pairs is None or frozenset((s1, s2)) in allowed_pairs:
                i = int(rng.choice(species_to_idx[s1]))
                j = int(rng.choice(species_to_idx[s2]))
                return i, j
            tries += 1
            if tries > 1000:
                raise RuntimeError("Could not sample an allowed pair for swapping.")

    def run(
        self,
        anneal: AnnealSettings = AnnealSettings(),
        coupled: CoupledRelaxSettings = CoupledRelaxSettings(),
        final_polish: RelaxSettings = RelaxSettings(do_relax=True, fmax=0.02, steps=600),
        verbose: bool = True,
    ):
        rng = np.random.default_rng(anneal.seed)

        # start from a relaxed baseline so E is always a minimized energy in propose mode
        state = deepcopy(self.atoms0)
        state.calc = self.calc

        # quick baseline relax and cache it
        key0 = self._symbols_key(state)
        cached = self._cache_get(key0)
        if cached is None:
            E = self._relax_local(state, coupled)
            self._cache_set(key0, E, state.get_positions(), state.get_cell().array, coupled.cache_size)
        else:
            E, pos, cell = cached
            state.set_positions(pos)
            state.set_cell(cell, scale_atoms=True)

        symbols = np.array(state.get_chemical_symbols(), dtype=object)

        best_atoms = state.copy()
        best_E = E

        # temperature schedule
        T = CanonicalSwapAnnealer(atoms=state, calculator=self.calc, allowed_pairs=None)._estimate_initial_temperature(
            state, anneal
        )
        if verbose:
            print(f"[CR-SA] Initial temperature guess T0 = {T:.1f} K")

        T_vals, E_vals, acc_vals, best_progress, MCtraj = [], [], [], [], []
        n = len(state)
        frozen_count = 0
        accepted_since_last_relax = 0

        for tstep in range(anneal.max_T_steps):
            accepts = 0
            attempts = 0
            total_moves = max(1, int(anneal.sweeps_per_T)) * n

            if verbose:
                print(f"############ coupled relax T step {tstep+1} at {T:8.2f} K ############")

            for move in range(total_moves):
                attempts += 1
                i, j = self._choose_swap_indices(symbols, rng, self.allowed_pairs)

                if coupled.mode == "propose":
                    # propose on a copy, relax, then decide
                    cand = state.copy()
                    self._swap_symbols_inplace(cand, i, j)
                    cand.calc = self.calc
                    key = self._symbols_key(cand)

                    cache_hit = self._cache_get(key)
                    if cache_hit is None:
                        E_new = self._relax_local(cand, coupled)
                        self._cache_set(key, E_new, cand.get_positions(), cand.get_cell().array, coupled.cache_size)
                    else:
                        E_new, pos_new, cell_new = cache_hit
                        cand.set_positions(pos_new)
                        cand.set_cell(cell_new, scale_atoms=True)

                    dE = E_new - E
                    accept = (dE <= 0.0) or (rng.random() < np.exp(-dE / (kB * T)))

                    if accept:
                        accepts += 1
                        accepted_since_last_relax += 1
                        state = cand
                        E = E_new
                        symbols[i], symbols[j] = symbols[j], symbols[i]
                        if E < best_E:
                            best_E = E
                            best_atoms = state.copy()

                elif coupled.mode in ("accept", "interval"):
                    # cheap screening using unrelaxed energy
                    # swap in place
                    self._swap_symbols_inplace(state, i, j)
                    E_unrel = state.get_potential_energy()
                    dE = E_unrel - E
                    accept = (dE <= 0.0) or (rng.random() < np.exp(-dE / (kB * T)))

                    if accept:
                        accepts += 1
                        accepted_since_last_relax += 1
                        symbols[i], symbols[j] = symbols[j], symbols[i]
                        if coupled.mode == "accept":
                            # relax now
                            E = self._relax_local(state, coupled)
                            # update cache
                            key = self._symbols_key(state)
                            self._cache_set(key, E, state.get_positions(), state.get_cell().array, coupled.cache_size)
                        elif coupled.mode == "interval" and (accepted_since_last_relax % max(1, coupled.interval) == 0):
                            E = self._relax_local(state, coupled)
                            key = self._symbols_key(state)
                            self._cache_set(key, E, state.get_positions(), state.get_cell().array, coupled.cache_size)
                        else:
                            # keep unrelaxed energy as approximate running energy
                            E = E_unrel
                        if E < best_E:
                            best_E = E
                            best_atoms = state.copy()
                    else:
                        # reject: revert
                        self._swap_symbols_inplace(state, i, j)

                if move % n == 0:
                    if verbose:
                        print(f"[CR-SA]({move}/{total_moves}) T={T:8.2f} K | E={E: .6f} eV | "
                              f"accept so far={accepts/max(1,attempts)*100:5.1f}% | best={best_E: .6f}",
                              end="\r")
                    MCtraj.append(state.copy())

            acc_rate = accepts / max(1, attempts)
            T_vals.append(T); E_vals.append(E); acc_vals.append(acc_rate); best_progress.append(best_E)

            if verbose:
                print(f"\n[CR-SA] T={T:8.2f} K | E={E: .6f} eV | accept={acc_rate*100:5.1f}% | best={best_E: .6f}\n")

            if acc_rate < 0.01:
                frozen_count += 1
            else:
                frozen_count = 0
            if frozen_count >= 3:
                if verbose:
                    print("[CR-SA] System appears frozen for 3 consecutive temperatures. Stopping.")
                break

            T *= anneal.alpha
            if T < anneal.min_temperature:
                if verbose:
                    print(f"[CR-SA] Reached min temperature {anneal.min_temperature} K. Stopping.")
                break

        # final polish at 0 K
        if final_polish.do_relax:
            if verbose:
                print("[CR-SA] Final 0 K polish.")
            best_atoms_polish = best_atoms.copy()
            best_atoms_polish.calc = self.calc
            dyn = FIRE(best_atoms_polish, logfile=None)
            dyn.run(fmax=final_polish.fmax, steps=final_polish.steps)
            E_polish = best_atoms_polish.get_potential_energy()
            if E_polish <= best_E:
                best_atoms = best_atoms_polish
                best_E = E_polish

        log = dict(
            T=np.array(T_vals),
            E=np.array(E_vals),
            accept_rate=np.array(acc_vals),
            best_E_progress=np.array(best_progress),
            best_E=best_E,
            MCtraj=MCtraj,
        )
        return best_atoms, log
