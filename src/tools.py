import sys
sys.path.append('/home/energy/matnis/projects/dreams_colab/material_agent/src')
from copy import deepcopy
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
from math import e
from networkx import predecessor
import pandas as pd
from src.utils import *
from src.myCANVAS import CANVAS
from ase import Atoms, Atom
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
# from langchain_openai import AzureChatOpenAI
import os 
from typing import Annotated, Dict, Literal, Optional, Sequence, Tuple, Any, Union, Iterable
import numpy as np
from ase.lattice.cubic import FaceCenteredCubic
import ast
import re
import io
from ase.io import read, write
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic, Diamond
from ase.io import read
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
from datetime import timedelta
from pysqa import QueueAdapter
import json
import pandas as pd
import sqlite3
from filecmp import cmp
import contextlib
from autocat.surface import generate_surface_structures
from autocat.adsorption import get_adsorption_sites, get_adsorbate_height_estimate
from pymatgen.io.ase import AseAtomsAdaptor
from src import var
import pickle
from ursa.agents import ArxivAgent

from GNoME_aqueous_stability.src.gnome_aqueous_stability.data_utils import Data_Handler
from GNoME_aqueous_stability.src.gnome_aqueous_stability.analysis_utils import (
    plot_periodic_table_with_values, get_col_dict_for_atoms, 
    Stable_Entries, Stability_Criteria, get_simplified_df, 
    atoms_from_db
)
from gnome_dreams_oer_screening.oer.oer_study import OER_catalyst_study
from gnome_dreams_oer_screening.vasp.pre_defined_vasp_sets import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set 
)
from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    run_vasp_via_custodian, read_vasp_results, clean_up_vasp_directory
)
from gnome_dreams_oer_screening.explog.explog import EXPLOG

try:
    import torch
    if torch.cuda.is_available():
        try:
            from mace.calculators import MACECalculator, mace_mp 
            print("MACE imported successfully")
        except:
            mace_mp = None
    else:
        mace_mp = None
except ImportError:
    mace_mp = None 
    
##################################################################################################
##                                         OER tools                                            ##
##################################################################################################
import asyncio  # If needed for defining async_func

async def _arXiv_search(arxiv_search_query, context):  # Your async operation
    config = var.OTHER_GLOBAL_VARIABLES
    ursaWorkspace = Path(os.path.join(var.my_WORKING_DIRECTORY, "ursa_workspace"))
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    agent = ArxivAgent(llm=llm, process_images=False, max_results=1, workspace=ursaWorkspace)
    result = await agent.ainvoke(
        arxiv_search_query=arxiv_search_query, 
        context=context
    )
    os.makedirs(ursaWorkspace/"arxiv_papers_used", exist_ok=True)
    # move all files under ursaWorkspace / "arxiv_papers" into ursaWorkspace/"arxiv_papers_used"
    for file in os.listdir(ursaWorkspace/"arxiv_papers"):
        os.rename(ursaWorkspace/"arxiv_papers"/file, ursaWorkspace/"arxiv_papers_used"/file)
    
    return result["final_summary"]


@tool
def arXiv_search(
    arxiv_search_query: Annotated[str, "key word search query for arXiv."],
    context: Annotated[str, "The context or question to focus the search on."]
    ) -> str:
    """Perform an arXiv search for papers with a given arxiv_search_query and context and provide a summary"""

    result = asyncio.run(_arXiv_search(arxiv_search_query, context))

    return result


@tool
def wait_for_update():
    """Only call wait tool after checking the EXPLOG and you decided there is nothing you want to do currently"""
    
    statusList = EXPLOG.relational_frame.processes.df["status"].tolist()
    if 'pending' not in statusList and 'running' not in statusList:
        return "No pending or running jobs found in the EXPLOG. Please check the EXPLOG and see if there is anything you can do to move the study forward, instead of waiting for updates."

    waitStartTime = time.time()
    
    while True:



        # --- This enabels the tempoary stop of the agent ---
        with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
            status = f.read()
        while status == "stop":
            # print(f"Calculation pause, Agent is waiting. cwd: {var.my_WORKING_DIRECTORY}")
            # # wait for 5 second
            time.sleep(5)
            with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
                status = f.read()


        time.sleep(2)
        tmpUpdate = EXPLOG.update_log()
        # Sort through the updates, remove non-failed/completed jobs (ignore going from pending to running)
        
        print('Init dict:', tmpUpdate)

        for_deletion = []
        for key, value in tmpUpdate.items():

            if value not in ["completed", "failed"]:
                for_deletion.append(key)

        for key in for_deletion:
            tmpUpdate.pop(key)

        print('After deletion:', tmpUpdate)

        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timeElapsed = timedelta(seconds= time.time() - var.startTime)
        print(timeElapsed, tmpUpdate)
        print('-----------------------\n')


        # time.sleep(10)
        # tmpUpdate = EXPLOG.update_log()
        # # Sort through the updates, remove non-failed/completed jobs (ignore going from pending to running)
        # for key, value in tmpUpdate.items():
        #     if value not in ["completed", "failed"]:
        #         del tmpUpdate[key]

        # currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # timeElapsed = timedelta(seconds= time.time() - var.startTime)
        # print(timeElapsed)
        if len(tmpUpdate) > 0:
            timeWaited = timedelta(seconds= time.time() - waitStartTime)

            outText = f"Current time is {currentTime}, time waited: {timeWaited}, time elapsed since the start of the study: {timeElapsed}.\n Here are the updates while you are waiting: "
            for key, value in tmpUpdate.items():
                outText += f"\nprocess_id {key} status is now {value}."
            return outText
        elif time.time() - var.startTime > 24*60*60:
            return f"Current time is {currentTime}, you have been waiting for more than 24 hours with no update in the EXPLOG, time elapsed since the start of the study: {timeElapsed}. You may want to check the EXPLOG and see if there is anything you can do to move the study forward."
                       

@tool
def inspect_explog(only_get_updates: Annotated[bool, "Whether to only get updates since last inspection."] = False) -> str:
    """Inspect the experiment log to get a summary of the candidates and processes."""
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    
    all_candidates_id = EXPLOG.relational_frame.candidates.df["candidate_id"].tolist()
    
    finishish_mask = EXPLOG.relational_frame.candidates.df["idealOverPotential"].notna()
    finishish_candidate_ids = EXPLOG.relational_frame.candidates.df.loc[finishish_mask, "candidate_id"].tolist()
    
    unfinished_candidate_ids = [can for can in all_candidates_id if can not in finishish_candidate_ids]
          
    finalAnswer = f"""You'v started {len(all_candidates_id)} candidates in total,
You've finished study at least one oxygen adsorption on {len(finishish_candidate_ids)} systems,
The following systems is still in progress:
    """
    
    pdf = EXPLOG.relational_frame.processes.df

    for cant_id in unfinished_candidate_ids:

        cand_status = None

        sub_pdf = pdf[pdf['candidate_id'] == cant_id]

        jobs = sub_pdf['job_type'].tolist()

        if 'OH_adsorption' in jobs:
            cand_status = f'candidate {cant_id} has OH adsorption job which is: '
            sub_pdf = sub_pdf[sub_pdf.job_type == 'OH_adsorption']
        elif 'O_adsorption' in jobs:
            cand_status = f'candidate {cant_id} has O adsorption job which is: '
            sub_pdf = sub_pdf[sub_pdf.job_type == 'O_adsorption']
        elif 'surface_relaxation' in jobs:
            cand_status = f'candidate {cant_id} has surface relaxation job which is: '
            sub_pdf = sub_pdf[sub_pdf.job_type == 'surface_relaxation']
        elif 'bulk_relaxation' in jobs:
            cand_status = f'candidate {cant_id} has bulk relaxation job which is: '
            sub_pdf = sub_pdf[sub_pdf.job_type == 'bulk_relaxation']
        else:
            cand_status = f'no job submitted for {cant_id}\n'
            sub_pdf = None
        
        if sub_pdf is not None:
            if 'completed' in sub_pdf.status.tolist():
                cand_status += 'completed'
            elif 'failed' in sub_pdf.status.tolist():
                cand_status += 'failed'
            elif 'running' in sub_pdf.status.tolist():
                cand_status += 'running'
            elif 'pending' in sub_pdf.status.tolist():
                cand_status += 'pending'
            # elif 'submitted' in sub_pdf.status.tolist():
            #     cand_status += 'submitted'
            elif 'un-submitted' in sub_pdf.status.tolist():
                cand_status += 'un-submitted'
            else:
                raise ValueError(f'unknown status for {cant_id}, status list: {sub_pdf.status.tolist()}')
            
        finalAnswer += cand_status + "\n"

    return finalAnswer

@tool
def query_explog(
    tableName: Annotated[str, "Name of the table in the relational frame to query, either 'candidates' or 'processes'."],
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
) -> str:
    """Query the experiment log relational frame with a given filter and sort criteria.
    candidates table contains: 
        candidate_id (str, MaterialID of the candidate),
        reason_or_hypothesis (str, for selecting the candidate),
        notes (str, any notes you've added for the candidate),
        OHDone (boolean, whether the OH adsorption calculation has been done for the candidate),
        idealOverPotential (Float64, the ideal overpotential calculated based on currently available data)
    processes table contains:
        process_id (str, unique id for each process),
        candidate_id (str, MaterialID of the candidate this process belongs to),
        job_type (str, type of the DFT calculation, either bulk_relaxation, surface_relaxation, O_adsorption, or OH_adsorption),
        slurmID (str, the slurm ID of the job, int in str format. NaN for un-submitted jobs),
        stutas (str, current status of the job, either un-submitted, submitted, pending, running, completed, or failed),
        termination_index (Int64, termination index for surface relaxation and adsorption calculations, NaN for bulk relaxation),
        site_index (Int64, adsorption site index for adsorption calculations, NaN for bulk and surface relaxation),
        processNote (str, any note you've left for this process)
    """

    # print dtype of both df
    print("candidates df dtype:\n", EXPLOG.relational_frame.candidates.df.dtypes)
    print("processes df dtype:\n", EXPLOG.relational_frame.processes.df.dtypes)
    
    if tableName == 'candidates':
        df = EXPLOG.relational_frame.candidates.df.copy()
        # drop the "study_obj" column since it contains complex objects that are not easy to display in a dataframe format
        df = df.drop(columns=["study_obj"])
    elif tableName == 'processes':
        df = EXPLOG.relational_frame.processes.df.copy()
        df = df.drop(columns=["VASP_dir"]) # drop the "VASP_dir" column since it contains file directory strings that are not easy to display in a dataframe format
    else:
        return "tableName must be either 'candidates' or 'processes'"
    
    filteredDF = df_query(df, filters, sort)
    
    print(filteredDF)
    return filteredDF.to_string(index=True)
    
    

@tool
def read_explog(
    candidate_id: Annotated[str, "MaterialId of the candidate to read the experiment log for."],
    ) -> str:
    """Read the experiment log for a specific candidate and return all information about the candidate together with all related jobs info"""
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    cadidate_row_df = EXPLOG.relational_frame.candidates[candidate_id].df
    cadidate_row_df = cadidate_row_df.copy().drop(columns=["study_obj"])
    related_process_df = EXPLOG.relational_frame.candidates[candidate_id].processes.df
    related_process_df = related_process_df.copy().drop(columns=["VASP_dir"])    
    
    answer = f"Candidate information:\n{cadidate_row_df.to_string(index=False)}\n\nRelated processes information:\n{related_process_df.to_string(index=False)}"
    return answer

@tool
def get_top_k_candidates(
    k: Annotated[int, "Number of top candidates to retrieve based on ideal overpotential."],
    ) -> str:
    """Get the top k candidates with the lowest ideal overpotential from the experiment log."""
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    candidates_df = EXPLOG.relational_frame.candidates.df.copy()
    candidates_df = candidates_df[candidates_df['idealOverPotential'].notna()]
    if len(candidates_df) == 0:
        return "No candidates has ideal overpotential information."
    candidates_df["idealOverPotential"] = candidates_df["idealOverPotential"].apply(lambda x: float(x))
    N_finished = len(candidates_df)
    top_k_candidates = candidates_df.nsmallest(k, 'idealOverPotential')
    top_k_candidates = top_k_candidates.copy().drop(columns=["study_obj"])
    answer = f"Top {k} out of {N_finished} candidates with the lowest ideal overpotential:\n{top_k_candidates.to_string(index=False)}\n\nYou may run more calculations on those candidates at different terminations and sites, or you can also run more calculations on other candidates to expand the pool and find more promising candidates."
    return answer

# @tool
# def get_explog_updates()

@tool
def enter_candidate_in_log(
    reason_or_hypothesis: Annotated[str, "Detailed Reason or hypothesis for selecting this candidate."],
    df_name: Annotated[str, "Name of the dataframe in canvas to read."],
    MaterialId: Annotated[str, "MaterialId of the candidate in the dataframe."],
    note: Annotated[str | None, "Any notes you want to add."] = None,
    ) -> str:
    """
    Initialize a catalyst candidate in the experiment-log, such that it
    can be studied further.
    """

    afdb = CANVAS.canvas.get('afdb', None)
    if afdb is None:
        afdb = atoms_from_db(None)
        CANVAS.canvas['afdb'] = afdb
    
    df = CANVAS.read(df_name)
    atoms = afdb.get_atoms_material_id(MaterialId, df)
    
    CANVAS.write(f"{MaterialId}_OER_catalyst_study_atoms", atoms, 
                 overwrite=True)

    catalyst_study = OER_catalyst_study(
        init_atoms = atoms, 
        H2O_gas_free_energy = -14.217, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
        H2_gas_free_energy = -6.77, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
                                        )

    EXPLOG.add_candidate(candidate_id=MaterialId,
                         reason_or_hypothesis=reason_or_hypothesis,
                         notes=note,
                         study_obj=catalyst_study)
    
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)

    message = f"Material {MaterialId} added to the experiment log with \
    reason: {reason_or_hypothesis} and note: {note}. Candidate can now \
    be studied further applying DFT"

    return message

@tool
def submit_dft_job(
    MaterialId: Annotated[str, "MaterialId of the candidate to submit DFT job for."],
    calculation_type: Annotated[Literal['bulk_relaxation', 'surface_relaxation', 'OH_adsorption', 'O_adsorption'], "Type of DFT calculation to submit. The OH_adsorption calculation will submit three jobs with slightly different initial OH adsorbate positions, increasing the likelihood of finding the global minimum."],
    note: Annotated[str, "Short note you want to leave for the calculation"],
    termination_index: Annotated[int, "termination index. Only needed for surface and adsorption calculations"] = None,
    ad_site_index: Annotated[int, "index of the site you want to adsorb O or OH onto. Only neeeded for adsorption calculations"] = None,
    partition: Annotated[Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'], "Partition to submit the job to"] = "auto",
):
    """Submit different types of DFT jobs to the cluster for a cadidate"""

    # --- Sanity checks for input arguments ----------------------------
    if ad_site_index is not None and termination_index is None:
        raise ValueError("termination_index must be provided for" \
        " adsorption calculations")
    
    # MORE CHECKS NEEDED...!!!

    # ------------------------------------------------------------------
    
    # --- Initializing surface- and adsorption studies if not 
    # already initialized ----------------------------------------------
    if termination_index is not None:
        study = EXPLOG.relational_frame.candidates[MaterialId].study_obj

        surface_study_dict = study.get_surface_studies()
        if termination_index not in surface_study_dict.keys():
            study.initialize_oer_surface_study(termination_index)
        surface_study = study.get_surface_studies()[termination_index]

        if ad_site_index is not None:
            ad_site_studies_dict = surface_study.get_adsorption_site_studies_dict()
            if ad_site_index not in ad_site_studies_dict.keys():
                surface_study.initialize_adsorption_site_study(ad_site_index)
            ad_site_study = surface_study.get_adsorption_site_studies_dict()[ad_site_index]
    # ------------------------------------------------------------------

    # a list of ids will be provided for OH_calculations and not for all other:
    id_list = EXPLOG.add_process(MaterialId, calculation_type, termination_index, ad_site_index, note)
    if not isinstance(id_list, list):
        id_list = [id_list]
    else:
        id_list = id_list

    for id in id_list:
        EXPLOG.submit_process(id, partition)
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    
    return f"Submitted {calculation_type} for candidate {MaterialId}"


@tool
def get_terminations_ranking(
    candidate_id: Annotated[str, "MaterialId of the candidate to get termination rankings for."],
    max_miller: Annotated[int, "Maximum miller index to consider for surface generation."] = 1,
):
    """
    Get a ranking of suface terminations for a given candidate. The ranking is based on the reduced
    coordination of the surface atoms, with respect to the coorsponding bulk coordination. The smaller
    the diffrence the higher the Normalized ranking, equaling a higher likelihood of being the most 
    stable termination. Uniqe surfaces will be created up to the maximum miller index specified.
    This function must be run before any surface relaxation or adsorption calculations are performed, 
    since this fucntion creates all initial surfaces and the corresponding terminations. Any number
    of terminations may be studied after the ranking is preformed. One the ranking has been 
    performed once, it will not be performed again, and the same ranking will be used for any 
    subsequent calls of this function. Since this function can be called repeatedly, there is no need to write the result
     to the canvas.
    """
    
    # Args to leve out (for now):
    method = 'all'      # What coordination to consider
    stoichiometry_tolerance = 0.2 # Allowed stoichiometry diviation from bulk
    all_species_present = True # Only surfaces with all bulk species present
    symmetrize = True # Whether to symmetrize the surfaces
    select_closest_O_stoichiometry = True # Select surfaces with closest O stoichiometry to bulk
    min_slab_thickness = 9 # Minimum slab thickness in Å
    max_slab_thickness = 20 # Maximum slab thickness in Å
    min_atoms = 20 # Minimum number of atoms in the slab
    max_atoms = 120 # Maximum number of atoms in the slab
    max_layers = 6 # Maximum number of layers, considered when building the slabs

    study = EXPLOG.relational_frame.candidates[candidate_id].study_obj

    out_string = ''

    ranking = study.get_termination_rankings() # None on first call
    if ranking is None:
        out_string += f"This is the first termination ranking for candidate {candidate_id}:"\
        # No arguments needs to be provided (all have defaults as shown above):
        study.predict_most_likely_surfaces(
        max_miller = max_miller,
        method = method,
        stoichiometry_tolerance = stoichiometry_tolerance,
        all_species_present = all_species_present,
        symmetrize = symmetrize,
        select_closest_O_stoichiometry = select_closest_O_stoichiometry,
        min_slab_thickness = min_slab_thickness,
        max_slab_thickness = max_slab_thickness,
        min_atoms = min_atoms,
        max_atoms = max_atoms,
        max_layers = max_layers
        )
        ranking = study.get_termination_rankings()
    else:
        out_string += f"Termination ranking for candidate {candidate_id} has already been determined, "\
             "hence the same ranking is provided as before:"
        
    ranking.sort_values('Normalized score', ascending=False)
    
    out_string += ranking.to_string(index=True)

    if True:
        out_string += '\n\n Original reason or hypothesis for selecting this candidate:\n'
        out_string += EXPLOG.relational_frame.candidates[candidate_id].reason_or_hypothesis

    return out_string

@tool
def list_adsorption_sites(
    candidate_id: Annotated[str, "MaterialId of the candidate to list adsorption sites for."],
    termination_index: Annotated[int, "termination index, of the surface to list adsorbtion sites for."],
    # only_reduced_coord_O_sites = True, DISABELD FOR NOW...
): 
    """
    Gives an preliminary list of adsorbtion sites if the termination has not been relaxed yet, or a list of final 
    adsorption sites if the termination has been relaxed. Sites may be 'on-top' or 'lattice O' the latter beeing an 
    espoused surface oxygen atom that is part of the lattice and may act as an adsorption site. For 'on-top' sites, the 
    'element of the adsorption site' is listed, meaning the element which the adsobate is placed on top of. 
    Additionally, a list of the closest neighboring elements of the adsorption site is given, with the  distance to 
    these neighbors given as (neighbor element, distance [Å]). For 'lattice O' sites, the reduced coordination of the 
    lattice O is given, which is a measure of how many neighboring atoms the lattice O has compared to a fully 
    coordinated lattice O in the bulk structure (e.g., a reduced coordination of 1 means that the lattice O atom has a
     decreased coordination of 1.). Since this function can be called repeatedly, there is no need to write the result
     to the canvas.
    """



    only_reduced_coord_O_sites = True # <<<--- FIXED FOR NOW...

    out_string = ''
    df = None

    study = EXPLOG.relational_frame.candidates[candidate_id].study_obj

    if study.get_termination_rankings() is None:
        raise ValueError(f"Termination rankings have not been determined yet for candidate {candidate_id}. "\
                          "Please determine the termination rankings first.")

    surface_study_dict = study.get_surface_studies()
    if termination_index in surface_study_dict.keys():
        surface_study = surface_study_dict[termination_index]
        if surface_study.get_relaxed_surface() != None:
            if surface_study.get_adsorption_sites_df() is None:
                    surface_study.determine_adsorption_sites(
                        only_reduced_coord_O_sites = only_reduced_coord_O_sites)
                    
            out_string += 'The requested termination has been relaxed, hence the provided adsorption sites are final,'\
                 ' though adorbtion energies may change after relaxation of the adsorption structures.\n\n' 
            df = surface_study.get_adsorption_sites_df()

    if df is None:
        out_string += f'The requested termination has not been relaxed yet, '\
        'hence, the provided sites are only preliminary and may change after '\
        'relaxation.\n\n'

        df = study.get_init_adsorption_sites_df(termination_index, 
                    only_reduced_coord_O_sites=only_reduced_coord_O_sites)

    # Removing unnecessary columns
    df = df.drop(columns=['position', 'atom_index'])
    
    # Reshaping format:
    df['ad site neighboring elements'] = df['ad site neighboring elements'].apply(lambda x: [(x[i][0], np.round(x[i][1],1)) for i in 
        range(len(x))])

    # Renaming columns for easier interpretation:
    df.rename(columns={"ad site neighboring elements": 
        "closest neighboring elemetns of adsorption site (element, distance [Å])"}, inplace=True)
    df.rename(columns={"ad site element": "element of the adsorption site"}, inplace=True)
    df.rename(columns={"reduced coordination": 
        "reduced coordination of lattice O"}, inplace=True)
    df.rename(columns={"site type": 
    "type of adsorption site"}, inplace=True)

    out_string += df.to_string(index=True)

    if True:
        out_string += '\n\n Original reason or hypothesis for selecting this candidate:\n'
        out_string += EXPLOG.relational_frame.candidates[candidate_id].reason_or_hypothesis

    return out_string


@tool
def OER_data_analasis_v2(
    pHs: Annotated[Union[List[float], float], "The pH in which the materials should be stable, may either be a float (specifying a single pH) or a pH range specified as two floats in a list i.e. [min, max]"],
    Us: Annotated[Union[List[float], float], "Eletrochemical potantial in which the materials should be stable, may either be a float (specifying a single potential) or a potential range specified as two floats in a list i.e. [min, max]"],
    decomposition_threshold: Annotated[float, "Decomposition energy threshold for stability criteria. (pourbaix stability)"],
    solid_filter: Annotated[bool, "Whether to apply solid filter: "],
    gga_only: Annotated[bool, "Whether to use only GGA calculations (True), or include r2SCAN data via the MP-mixing scheme (False)."],
    dir_of_data: Annotated[Optional[str], "Path to data directory. If None, use default data directory."] = None,
    # elements_to_exclude: Annotated[List[str], "List of element symbols to exclude from the analysis."] = [],
    # elements_whic_must_be_included: Annotated[List[str], "List of element symbols that must be included in the analysis."] = [],
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    ) -> None:
    """Perform data analysis on stable entries for OER based on specified criteria and filters, sort, and save the resulting dataframe on CANVAS. If you want to exclude or include certain elements, just apply filter to the Elements column"""

    dh = Data_Handler(
    # Whether to apply solid filter:
        solid_filter = solid_filter, 
    # Whether to use only GGA calculations (True), or include r2SCAN data via the MP-mixing scheme (False):
        gga_only = gga_only,
    # Path to data directory. None if not specified in config yaml:
        path_to_data_directory = var.OTHER_GLOBAL_VARIABLES.get('path_to_data_directory', None)
        
        )

    dh.remove_entries_without_elements(['Ir'], True)
    
    # if len(elements_to_exclude) > 0:
    #     dh.remove_entries_with_elements(elements_to_exclude)
    # if len(elements_whic_must_be_included) > 0:
    #     dh.remove_entries_without_elements(elements_whic_must_be_included, True)
    
    SCS = [Stability_Criteria(pHs=pHs, Us=Us, decomposition_threshold=decomposition_threshold),
       # Stability_Criteria(pHs=0, Us=[1.2, 1.6], decomposition_threshold=0.05),
       # Stability_Criteria(pHs=[2, 5], Us=[0., 2], decomposition_threshold=1),
       ]
    
    se = Stable_Entries(dh, SCS)
    df = se.get_stable_df()
    
    df = df_query(df, filters, sort)
    df = get_simplified_df(df) # <<<<---- new change 2026-02-18
    if len(df) == 0:
        return "No stable entries found based on the specified criteria and filters."
    
    # save df
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    save_path = os.path.join(WORKING_DIRECTORY, 'stable_entries.csv')
    df.to_csv(save_path, index=False)
    
    # write to canvas
    CANVAS.write('OER_stable_entries_df', df, overwrite=True)
    
    # if dataframe is too long
    if len(df) > 20:
        return f"Stable entries data analysis completed. Results saved to {save_path}. The dataframe has {len(df)} entries, too long to display here. Please check the dataframe in canvas with key 'OER_stable_entries_df' using read dataframe tool."
    else:
        return f"Stable entries data analysis completed. Results saved to {save_path}. Below shows the dataframe with row index: \n{df.to_string(index=True)}. the same dataframe is also saved in canvas with key 'OER_stable_entries_df' and can be accessed using read dataframe tool."


@tool
def extract_df(
    df_name: Annotated[str, "Name of the dataframe in canvas to extract."],
    filters: List[Filter] = [],
    sort: List[SortSpec] = []
    ):
    """read the dataframe with a given filter and sort. This is useful to exam the filtered dataframe without altering its data"""
    df = CANVAS.read(df_name)
    df = df_query(df, filters, sort)
    if len(df) > 50:
        return f"Too many entries pass the filter. Please apply more filters to narrow down the results or check with material_IDs to find the specific entries you want to look at. showing the first 50 entries:\n {df.head(50).to_string(index=True)}"
    return df.to_string(index=True)

    
@tool
def read_df(
    df_name: Annotated[str, "Name of the dataframe in canvas to read."],
    startIdx: Annotated[int, "Starting index of the dataframe to read."] = 0,
    endIdx: Annotated[int, "Ending index of the dataframe to read."] = 10,
    ) -> str:
    """Read a portion of a dataframe (from row i to row j) from canvas and return it as a string with row index."""
    if endIdx - startIdx > 50:
        return "Read no more than 50 rows at a time."
    df = CANVAS.read(df_name)
    print(df)
    return df.iloc[startIdx:endIdx].to_string(index=True)

# def get_facets(
#     df_name: Annotated[str, "Name of the dataframe in canvas to read."],
#     MaterialId: Annotated[str, "MaterialId of the dataframe to get the atoms object from."],
#     max_miller: Annotated[int, "Maximum miller index to consider for surface generation."] = 1,
#     ) -> str:
#     """Determine which facet to study: from the dataframe saved in CANVAS, generate facets for a catalyst study of a system with a certain MaterialId, and save the results to canvas. The result is different facets with corresponding score of likelihood"""
#     afdb = CANVAS.canvas.get('afdb', None)
#     if afdb is None:
#         afdb = atoms_from_db(None)
#         CANVAS.canvas['afdb'] = afdb
#     df = CANVAS.read(df_name)
#     atoms = afdb.get_atoms_material_id(MaterialId, df)
#     # atoms_list = afdb.get_atoms_objects_from_df(df.iloc[dfIdx])
#     # atoms = atoms_list[0]
#     CANVAS.write(f"{MaterialId}_OER_catalyst_study_atoms", atoms, overwrite=True)
    
#     # get facets for the catalyst study

#     catalyst_study = OER_catalyst_study(
#         bulk = atoms, 
#         H2O_gas_free_energy = -14.217, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
#         H2_gas_free_energy = -6.77, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
#                                         )

#     catalyst_study.identify_distinct_surfaces(max_miller = max_miller)
#     catalyst_study.predict_most_likely_surfaces(method = 'coordination', stoichiometry='stoichiometric') # An method should be chosen
#     catalyst_study_df = catalyst_study.get_df_with_surface_rankings().sort_values(by='Normalized Score', ascending=False)
#     CANVAS.write(f"{MaterialId}_OER_catalyst_study", catalyst_study, overwrite=True)
#     CANVAS.write(f"{MaterialId}_OER_catalyst_study_surface_ranking_df", catalyst_study_df, overwrite=True)
#     return f"Facets for the catalyst study have been generated and saved in canvas with key '{MaterialId}_OER_catalyst_study_surface_ranking_df'. Below shows the dataframe with row index: \n{catalyst_study_df.to_string(index=True)} \nHigher Normalized Score means more likely surface."

# @tool
# def get_terminations(
#     MaterialId: Annotated[str, "MaterialId of the dataframe to get the atoms object from."],
#     facets: Annotated[Tuple[int, int, int], "Miller indices of the facet to get terminations for."],
# )-> str:
#     """Determin which termination to study: get available terminations for a specific facet in the catalyst study, and save the results to canvas."""
#     catalyst_study = CANVAS.read(f"{MaterialId}_OER_catalyst_study")
#     catalyst_study.initialize_OER_surface_studies(facets)
#     surface_study_dict = catalyst_study.get_surface_studies() # dict with miller as key and list of surface studies as value
#     CANVAS.write(f"{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_OER_catalyst_study_surface_study_dict", surface_study_dict, overwrite=True)
#     return f"available terminations for material {MaterialId} facet {facets} are: {repr(surface_study_dict[facets])}"
        
# def study_termination(
#     MaterialId: Annotated[str, "MaterialId of the dataframe to get the atoms object from."],
#     facets: Annotated[Tuple[int, int, int], "Miller indices of the facet to study."],
#     termination_index: Annotated[int, "Index of the termination to study."],
#     calculationType: Annotated[Literal['MLIP', 'VASP'], "Type of calculation to perform for relaxation and adsorption site determination. use MLIP or VASP" ] = 'MLIP',
# ):
#     """Once you've determined which termination of which facet, Study a specific termination of a facet for OER catalysis, including relaxation and adsorption site determination, and save the results to canvas."""
#     # relax a given termination and determine adsorption sites
#     surface_study_dict = CANVAS.read(f"{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_OER_catalyst_study_surface_study_dict")
#     sur_study = surface_study_dict[facets][termination_index] # Get the frist termination of the (1,1,0) surfaces
#     atoms = sur_study.get_non_relaxed_surface()
#     slurm_template = Path("slurm_templates/test_template.sh").read_text()
#     if calculationType == 'VASP':
#         # prepare VASP calculation for relaxation
#         structure = AseAtomsAdaptor.get_structure(atoms)
#         vasp_set = RPBE_relax_bulk_set(structure = structure)
#         # write VASP input files
#         calculation_dir = os.path.join(var.my_WORKING_DIRECTORY, f"{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_termination{termination_index}_bulk_relaxation")
#         os.makedirs(calculation_dir, exist_ok=True)
#         vasp_set.write_input(output_dir = calculation_dir, potcar_spec=True)
        
#         # --- Setup of VASP directories ---
#         # vasp_calc_main_dir = Path("VASP_calculations/test_run_" + run_id)
#         # calc_paths = [vasp_calc_main_dir / f"calc_{i}" for i in range(5)]

#         # for path in calc_paths:
#         #     vasp_set = TestRelaxSet(test_structure)
#         #     vasp_set.write_input(output_dir = path) 

#         # --- Cache location (for executorlib) ---
#         # cache_dir = (
#         #     Path("/home/scratch3/")
#         #     / 'matnis'
#         #     / "executorlib"
#         #     / "first_VASP_test"
#         #     / run_id
#         # )
#         exlib_run_vasp(calc_path.resolve())
#         # --- Submit job ---
#         print("--- Submitting job to Slurm cluster... ---", flush=True)
#         with SlurmClusterExecutor(cache_directory=os.path.join(calculation_dir, "cache")) as exe:
#             futures = []

#             for i, calc_path in enumerate([calculation_dir]):
#                 calc_path = Path(calc_path)
#                 f = exe.submit(
#                     exlib_run_vasp,
#                     calc_path.resolve(),
#                     resource_dict={
#                         "submission_template": slurm_template,
#                     },
#                 )
#                 futures.append(f)
#                 print(f"Submitted job {i}, future: {f}", flush=True)
#             print("--- All jobs submitted ---", flush=True)

#             results = [f.result() for f in futures]
#             print("Results:", results)

#         for result in results:
#             print("VASP run result:", result['E'])
        
#         # out = {"atoms_result": atoms_result, "E": E, "error": None}
#         if results[0].get("atoms_result", None) is None:
#             return f"VASP calculation failed for relaxation of {MaterialId} {facets} termination {termination_index}: {results[0].get('error', 'Unknown error')}"
        
#         sur_study.set_relaxed_base_surface(results[0].get("atoms_result"), energy = results[0].get("E", None))
        
#         # fake
#         # relaxed_atoms = atoms.copy() # Fake relaxed structure for testing purposes
#         # sur_study.set_relaxed_base_surface(relaxed_atoms, energy = -100.0) # should this energy be the DFT energy of the relaxed surface? include free energy corrections?
        
#     elif calculationType == 'MLIP':
#         if not var.GPU_AVAILABLE:
#             relaxed_atoms = atoms.copy() # Fake relaxed structure for testing purposes
#             sur_study.set_relaxed_base_surface(relaxed_atoms, energy = -100.0) # should this energy be the DFT energy of the relaxed surface? include free energy corrections?
#         else:
#             # relax the structure
#             try:
#                 relaxed_atoms = atoms.copy()
#                 relaxed_atoms.calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
#                 opt = FIRE(relaxed_atoms, logfile=None)
#                 converged = opt.run(fmax=0.02, steps=2000)
#                 if not converged:
#                     print(f"FIRE MAXSTEP REACHED!!!")
#                 eos = calculate_eos(relaxed_atoms, eps=0.15)
#                 try:
#                     v, e, _ = eos.fit()
#                 except:
#                     print("EOS fit failed")
#                     write(f"eos-failed.xyz", relaxed_atoms)
#                     raise ValueError("EOS fit failed")
#                 relaxed_atoms.set_cell(relaxed_atoms.get_cell() * (v / relaxed_atoms.get_volume())**(1/3), scale_atoms=True)
#                 sur_study.set_relaxed_base_surface(relaxed_atoms, energy = relaxed_atoms.get_potential_energy()) # should this energy be the DFT energy of the relaxed surface? include free energy corrections?
#             except Exception as e:
#                 print(f"Relaxation with MACE failed: {e}")
#                 relaxed_atoms = atoms.copy() # Fake relaxed structure for testing purposes
#                 sur_study.set_relaxed_base_surface(relaxed_atoms, energy = -100.0) # should this energy be the DFT energy of the relaxed surface? include free energy corrections?
#     else:
#         return "calculationType must be either 'MLIP' or 'VASP'"
    
#     sur_study.determine_adsorption_sites(use_relaxed_surface = True)
#     sites_df = sur_study.get_adsorption_sites_df()
#     site_studies_dict = sur_study.get_adsorption_site_studies_dict()

#     for index, row in sites_df.iterrows():

#         # Skip initialized sites:
#         if index in site_studies_dict:
#             continue

#         # Initialize other sites:
#         sur_study.initialize_adsorption_site_study(index)

#     list_of_atoms_to_relax = []
#     for site_index, site_study in site_studies_dict.items():

#         adsorbed_energies_dict = site_study.get_adsorption_site_energies()
#         if adsorbed_energies_dict['O_adsorbed_energy'] is not None:
#             continue

#         list_of_atoms_to_relax.append(
#             [site_index, site_study.get_initial_surface_with_O()]
#             )
        
#     for site_index, atoms in list_of_atoms_to_relax:
#         # Here you would normally relax the structure and get the energy
#         # For testing purposes we just set a fake energy
#         # fake_energy = -104 - 2*np.random.rand()
#         if calculationType == 'VASP':
#             # prepare VASP calculation for relaxation
#             structure = AseAtomsAdaptor.get_structure(atoms)
#             vasp_set = RPBE_relax_surface_set(structure = structure)
#             # write VASP input files
#             calculation_dir = os.path.join(var.my_WORKING_DIRECTORY, f"{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_termination{termination_index}_site{site_index}_surface_relaxation")
#             os.makedirs(calculation_dir, exist_ok=True)
#             vasp_set.write_input(output_dir = calculation_dir, potcar_spec=True)
                    
#             # --- Setup of VASP directories ---
#             # vasp_calc_main_dir = Path("VASP_calculations/test_run_" + run_id)
#             # calc_paths = [vasp_calc_main_dir / f"calc_{i}" for i in range(5)]

#             # for path in calc_paths:
#             #     vasp_set = TestRelaxSet(test_structure)
#             #     vasp_set.write_input(output_dir = path) 

#             # --- Cache location (for executorlib) ---
#             # cache_dir = (
#             #     Path("/home/scratch3/")
#             #     / 'matnis'
#             #     / "executorlib"
#             #     / "first_VASP_test"
#             #     / run_id
#             # )

#             # --- Submit job ---
#             print("--- Submitting job to Slurm cluster... ---", flush=True)
#             with SlurmClusterExecutor(cache_directory=os.path.join(calculation_dir, "cache")) as exe:
#                 futures = []

#                 for i, calc_path in enumerate([calculation_dir]):
#                     calc_path = Path(calc_path)
#                     f = exe.submit(
#                         exlib_run_vasp,
#                         calc_path.resolve(),
#                         resource_dict={
#                             "submission_template": slurm_template,
#                         },
#                     )
#                     futures.append(f)
#                     print(f"Submitted job {i}, future: {f}", flush=True)
#                 print("--- All jobs submitted ---", flush=True)

#                 results = [f.result() for f in futures]
#                 print("Results:", results)

#             for result in results:
#                 print("VASP run result:", result['E'])
            
#             # out = {"atoms_result": atoms_result, "E": E, "error": None}
#             if results[0].get("atoms_result", None) is None:
#                 return f"VASP calculation failed for relaxation of {MaterialId} {facets} termination {termination_index}: {results[0].get('error', 'Unknown error')}"

#             site_studies_dict[site_index].set_relaxed_surface_with_O(results[0].get("atoms_result"), energy = results[0].get("E", None))
            
#         elif calculationType == 'MLIP':
#             if not var.GPU_AVAILABLE:
#                 relaxed_atoms = atoms.copy() # Fake relaxed structure for testing purposes
#                 fake_energy = -104 - 2*np.random.rand()
#                 site_studies_dict[site_index].set_relaxed_surface_with_O(relaxed_atoms, energy = fake_energy)
#             else:
#                 # relax the structure
#                 try:
#                     relaxed_atoms = atoms.copy()
#                     relaxed_atoms.calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
#                     opt = FIRE(relaxed_atoms, logfile=None)
#                     converged = opt.run(fmax=0.02, steps=2000)
#                     if not converged:
#                         print(f"FIRE MAXSTEP REACHED!!!")
#                     eos = calculate_eos(relaxed_atoms, eps=0.15)
#                     try:
#                         v, e, _ = eos.fit()
#                     except:
#                         print("EOS fit failed")
#                         write(f"eos-failed.xyz", relaxed_atoms)
#                         raise ValueError("EOS fit failed")
#                     relaxed_atoms.set_cell(relaxed_atoms.get_cell() * (v / relaxed_atoms.get_volume())**(1/3), scale_atoms=True)
#                     site_studies_dict[site_index].set_relaxed_surface_with_O(relaxed_atoms, energy = relaxed_atoms.get_potential_energy())
#                 except Exception as e:
#                     print(f"Relaxation with MACE failed: {e}")
#                     relaxed_atoms = atoms.copy() # Fake relaxed structure for testing purposes
#                     site_studies_dict[site_index].set_relaxed_surface_with_O(relaxed_atoms, energy = fake_energy)
#         else:
#             return "calculationType must be either 'MLIP' or 'VASP'"
            
        
#     sur_study.update_adsorption_energies()
#     sur_study_resultdf = sur_study.get_adsorption_sites_df()
#     CANVAS.write(f"{calculationType}_{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_termination{termination_index}_OER_catalyst_study_surface_study", sur_study, overwrite=True)
#     CANVAS.write(f"{calculationType}_{MaterialId}_{facets[0]}{facets[1]}{facets[2]}_termination{termination_index}_OER_catalyst_study_surface_study_resultdf", sur_study_resultdf, overwrite=True)
#     return f"Termination study completed. Below shows the study result dataframe with row index: \n{sur_study_resultdf.to_string(index=True)}"
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

# @tool
# def inspect_my_explog():
#     pass

# @tool
# def read_my_explog():
#     pass



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
    """Write a Quantum Espresso input script using ASE. Bool value have no quote around them. For smearing start with methfessel-paxton. For ecutwfc choose between 30-100 Ry. When asked to run ensemble calculation, set calculation to 'ensemble'. When generating template for convergence test, use scf calculation and set ready_to_run_job to False."""

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
    job_list_dict = CANVAS.canvas.get('jobs_K_and_ecut', {})
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
    
    convergence_df = pd.DataFrame({'job':goodJob,'kspacing':kspacing, 'ecutwfc':ecutwfc, 'energy':energy_list})
    
    min_kspacing = convergence_df['kspacing'].min()
    max_ecutwfc = convergence_df['ecutwfc'].max()
    df_kspacing = convergence_df.loc[convergence_df['kspacing'] == min_kspacing].sort_values(by='ecutwfc',ascending=True)
    df_ecutwfc = convergence_df.loc[convergence_df['ecutwfc'] == max_ecutwfc].sort_values(by='kspacing',ascending=False)

    print(f"successfully read {len(df_kspacing)} kspacing and {len(df_ecutwfc)} ecutwfc")
    
    if len(df_kspacing) == 1 and len(df_ecutwfc) > 1:
        # time.sleep(60)
        return f"Only one kspacing is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
    if len(df_ecutwfc) == 1 and len(df_kspacing) > 1:
        # time.sleep(60)
        return f"Only one ecutwfc is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
    if len(df_kspacing) == 1 and len(df_ecutwfc) == 1:
        # time.sleep(60)
        return f"Only one job of either kspacing or ecutwfc is good, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing and ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
        
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
    span: Annotated[str, "Time limit for the job, in minutes"],
    submissionScript: Annotated[str, "submission script based on the types of jobs. Do not include any #SBATCH stuff. output filename must be <full input filename with extension>.<output_file_type>"],
    outputFilename: Annotated[str, "the output filename of the job"],
) -> Annotated[str, "source suggestion saved location"]:
    """
    After agent generate resource suggestions and submission script based on the DFT input file, add it to "resource_suggestion".
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
    if not isinstance(partition, str) or not isinstance(nnodes, int) or not isinstance(ntasks, int) or not isinstance(span, str):
        # time.sleep(60)
        return "Invalid input, please check the input format"

    assert qeInputFileName in CANVAS.canvas.get('ready_to_run_job_list', []), f"{qeInputFileName} is not in the ready_to_run_job_list, please check the job list and make sure the file name is correct"

    # craete the json file if it does not exist, otherwise load it
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY

    # new_resource_dict = {qeInputFileName: {"partition": "venkvis-cpu", "nnodes": 1, "ntasks": 48, "runtime": 2800, "submissionScript": submissionScript, "outputFilename": outputFilename}}
    new_resource_dict = {qeInputFileName: {"partition": "venkvis-cpu", "nnodes": 1, "ntasks": 4, "runtime": 30, "submissionScript": submissionScript, "outputFilename": outputFilename}}

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
    # resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    # if not os.path.exists(resource_suggestions):
    #     # time.sleep(60)
    if not var.my_RESOURCE_DIRECTORY:
        return "Resource suggestion not found, please use the add_resource_suggestion tool to add the resource suggestion"
    
    # job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = []
    
    # load reousrce suggestions
    # resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.json')
    # with open(resource_suggestions, "r") as file:
    #     resource_dict = json.load(file)
    # db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    # conn = sqlite3.connect(db_file)
    # cursor = conn.cursor()

    # # Query all rows from the resources table
    # cursor.execute('SELECT * FROM resources')
    # rows = cursor.fetchall()

    # # Reconstruct the original dictionary
    resource_dict = deepcopy(var.my_RESOURCE_DIRECTORY)
    # for row in rows:
    #     filename, partition, nnodes, ntasks, runtime, submissionScript, outputFilename = row
    #     job_list.append(filename)
    #     resource_dict[filename] = {
    #         'partition': partition,
    #         'nnodes': nnodes,
    #         'ntasks': ntasks,
    #         'runtime': runtime,
    #         'submissionScript': submissionScript,
    #         'outputFilename': outputFilename
    #     }
    
    for key in resource_dict.keys():
        job_list.append(key)
    
    # conn.close()
    print(f"loaded resource suggestions: {json.dumps(resource_dict, indent=4)}")
    
    CANVAS.canvas['ready_to_run_job_list'] = job_list.copy()
    wasJobList = deepcopy(job_list)
    
    # ## Check resource key is valid
    # for job in job_list:
    #     if job not in resource_dict.keys():
    #         # time.sleep(60)
    #         return f"Resource suggestion for {job} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    
    if len(job_list) == 0:
        # time.sleep(60)
        return f"Resource suggestion not found, please use the add_resource_suggestion tool to add the resource suggestion."
    
    print(f"loaded {len(job_list)} jobs from job_list.json, and {len(resource_dict)} resource suggestions from resource_suggestions")
    
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
    # db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
    # os.remove(db_file)
    time.sleep(1)
    # initialize_database(db_file)
    var.my_RESOURCE_DIRECTORY = {}
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

# @tool
# def submit_single_job(
#     inputFile: str
# ) -> str:
#     '''Submit a single job to supercomputer, return the location of the output file once the job is done'''
#     print("checking pysqa prerequisites...")
#     WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
#     # check if slurm.sh and queue.yaml exist in the working directory
#     if not os.path.exists(os.path.join(WORKING_DIRECTORY, "slurm.sh")) or not os.path.exists(os.path.join(WORKING_DIRECTORY, "queue.yaml")):
#         print("Creating pysqa prerequisites...")
#         create_pysqa_prerequisites(WORKING_DIRECTORY)
    
#     qa = QueueAdapter(directory=WORKING_DIRECTORY)
        
    
#     # load reousrce suggestions
#     resource_suggestions = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.json')
#     with open(resource_suggestions, "r") as file:
#         resource_dict = json.load(file)
    
#     ## Check resource key is valid
    
#     if inputFile not in resource_dict.keys():
#         # time.sleep(60)
#         return f"Resource suggestion for {inputFile} is not found, please use the add_resource_suggestion tool to add the resource suggestion"
    

    
#     queueIDList = []


#     ## Check if the input file exists
#     if not os.path.exists(os.path.join(WORKING_DIRECTORY, inputFile)):
#         # time.sleep(60)
#         return f"Input file {inputFile} does not exist, please use the find job list tool to submit the file in the job list"
#     print("Generating batch script...")

#     ## Check if the output file exists 
#     if os.path.exists(os.path.join(WORKING_DIRECTORY, f"{inputFile}.pwo")):
#         ## Supervisor sometimes ask to submit the job again, so we need to check if the output file exists
#         # time.sleep(60)
#         return f"Output file {inputFile}.pwo already exists, the calculation is done"
        
        
#     job_id = qa.submit_job(
#         working_directory=WORKING_DIRECTORY,
#         cores=resource_dict[inputFile]['ntasks'],
#         memory_max=2000,
#         queue="slurm",
#         job_name="agent_job",
#         cores_max=resource_dict[inputFile]['ntasks'],
#         nodes_max=resource_dict[inputFile]['nnodes'],
#         partition=resource_dict[inputFile]['partition'],
#         run_time_max=resource_dict[inputFile]['runtime'],
#         command =f"""
# export OMP_NUM_THREADS=1

# spack load quantum-espresso@7.2

# echo "Job started on `hostname` at `date`"

# mpirun pw.x -i {inputFile} > {inputFile}.pwo

# echo " "
# echo "Job Ended at `date`"
#     """
#         )
        
#     if job_id is None:
#         # time.sleep(60)
#         return "Job submission failed"

#     queueIDList.append(job_id)
    
    
#     prevCount = len(queueIDList)
#     while True:
#         count = 0
#         print("waiting for", end=" ")
#         for queueID in queueIDList:
#             if qa.get_status_of_job(process_id=queueID):
#                 count += 1
#                 print(queueID, end=" ")
#         print("to finish", end="\r")
        
#         if count < prevCount:
#             print()
#             prevCount = count
#         if count == 0:
#             break
#         time.sleep(1)
        
#     print(f"Job has finished")

#     # time.sleep(60)
#     return f"Job has finished, please check the output file"   

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
