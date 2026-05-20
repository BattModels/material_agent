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
from src.myCANVAS import CANVAS, ListedArtifact
from ase import Atoms, Atom
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
# from langchain_openai import AzureChatOpenAI
import math
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
from pymatgen.analysis.magnetism.analyzer import DEFAULT_MAGMOMS
from src import var
import pickle
from ursa.agents import ArxivAgent

from aq_gnome import Data_Handler, Stable_Entries, Stability_Criteria, get_simplified_df, atoms_from_db
from gnome_dreams_oer_screening.oer.oer_study import OER_catalyst_study
from gnome_dreams_oer_screening.explog.explog import EXPLOG
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    count_magnetic_sites_from_formula
)

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
    agent = ArxivAgent(llm=llm, process_images=False, max_results=3, workspace=ursaWorkspace)
    result = await agent.ainvoke(
        arxiv_search_query=arxiv_search_query, 
        context=context
    )
    os.makedirs(ursaWorkspace/"arxiv_papers_used", exist_ok=True)
    # move all files under ursaWorkspace / "arxiv_papers" into ursaWorkspace/"arxiv_papers_used"
    for file in os.listdir(ursaWorkspace/"arxiv_papers"):
        os.rename(ursaWorkspace/"arxiv_papers"/file, ursaWorkspace/"arxiv_papers_used"/file)
    
    return result["final_summary"]

# FOR DEMO 
# async def _arXiv_search(arxiv_search_query, context):  # Your async operation
#     config = var.OTHER_GLOBAL_VARIABLES
#     llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
#     result = await llm.ainvoke(
#         [("user", f"Please answer the question: {arxiv_search_query} with context: {context}.")]
#     )
    
#     return result.content


@tool
def arXiv_search(
    arxiv_search_query: Annotated[str, "Keyword search query used to retrieve papers from arXiv."],
    context: Annotated[str, "The specific question or focus that the summary should address."]
    ) -> str:
    """
    Perform an arXiv search for papers with a given arxiv_search_query and context and provide a summary.
    Only 5 papers will be considered in the search. If you want to consider more papers, you will need 
    to refine your search arguments.
    """
    # Only 5 papers will be considered in the search. If you want to consider more papers, you will need to refine
    # your search arguments.

    result = asyncio.run(_arXiv_search(arxiv_search_query, context))

    id = CANVAS.register_tool_output(
        tool_name="arXiv_search",
        args={
            "arxiv_search_query": arxiv_search_query,
            "context": context,
        },
        value=result,
        description="Summary of arXiv search results for the query: {arxiv_search_query} with context: {context}",
        parent_result_ids=[],
        metadata={
            "arxiv_search_query": arxiv_search_query,
            "context": context,
        }
    )
    
    return f"{result}\nLiterature_result_ID={id}. Please extract the numerical values if you need to use numerical values from the result to make decisions or conclusions."

@tool
def check_time():
    """
    Check the total time elapsed since the project start. If the time elapsed is getting close to the given time constrain (if there's any) in the overall goal you may want to report back early instead of waiting for the completion of all the calculations.
    """
    timeElapsed_tmp = time.time() - var.startTime
    timeElapsed = timedelta(seconds=timeElapsed_tmp)
    return f"total time elapsed since project start: {str(timeElapsed).split('.')[0]} "

@tool
def wait_for_update(
    patience: Annotated[int, "Timeout in minutes: if no jobs complete or fail within this time, the tool returns regardless. Defaults to 24 hours."] = 1440,
) -> str:
    """
    Pause execution and periodically check the HPC job queue until at least one job completes
    or fails, or until the timeout (`patience` minutes) is reached.

    Only call this tool after checking the EXPLOG and confirming there is nothing productive
    to do while waiting. The tool will refuse to wait if no jobs are currently pending or running.

    Returns a message listing which process IDs changed status (completed or failed) during
    the wait, along with the time waited and total time elapsed. If the timeout is reached
    with no updates, returns a timeout message prompting further action.
    """
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
            time.sleep(60)
            with open(f"{var.my_WORKING_DIRECTORY}/status.txt", "r") as f:
                status = f.read()


        time.sleep(15) # TODO - consider before production run
        tmpUpdate = EXPLOG.update_log()
        # Sort through the updates, remove non-failed/completed jobs (ignore going from pending to running)
        
        print('Init dict:', tmpUpdate)

        for_deletion = []
        for key, value in tmpUpdate.items():

            if value not in ["completed", "failed"] or 'unrecoverable' in value:
                for_deletion.append(key)

        for key in for_deletion:
            tmpUpdate.pop(key)

        print('After deletion:', tmpUpdate)

        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timeElapsed = timedelta(seconds= time.time() - var.startTime)
        print(timeElapsed, tmpUpdate)
        print('-----------------------\n')


        if len(tmpUpdate) > 0:
            hWaited = int((time.time() - waitStartTime)/3600)
            mWaited = int(((time.time() - waitStartTime)%3600)/60)

            outText = f"Total time elapsed since project start {timeElapsed}, time waited: {hWaited}hours and {mWaited} minutes.\n Here are the updates while you are waiting: "
            for key, value in tmpUpdate.items():
                outText += f"\nprocess_id {key} status is now {value}."
            
            id = CANVAS.register_tool_output(
                tool_name="wait_for_update",
                args={
                    "patience": patience,
                },
                value=outText,
                description=f"Updates on job statuses after waiting for updates with patience {patience} minutes.",
                parent_result_ids=[],
                metadata={
                    "patience": patience,
                }
            )    
            
            return f"{outText}\nMessage_ID={id}. Please refer to this ID for the updates while waiting."
        elif time.time() - waitStartTime > patience*60:
            
            id = CANVAS.register_tool_output(
                tool_name="wait_for_update",
                args={
                    "patience": patience,
                },
                value=f"Timeout reached after waiting for {patience} minutes with no updates in job statuses.",
                description=f"Message indicating timeout after waiting for updates with patience {patience} minutes.",
                parent_result_ids=[],
                metadata={
                    "patience": patience,
                }
            )
            
            return f"Total time elapsed since project start {timeElapsed}, you have been waiting for {patience} minutes with no update in the EXPLOG. You may want to check the EXPLOG and see if there is anything you can do to move the study forward.\nMessage_ID={id}. Please refer to this ID for the timeout message."
        
        
                       
@tool
def inspect_explog():
    """
    Inspect the experiment log to get a high level summary of the candidates study progress.
    More specifically _/_ bulk job finished, _/_ surface job finished, _/_ O job finished, and _/_ OH job finished for each candidate.
    """
    
    # outString = ""
    # For each candidate
    #     outString += this candidate has x/y bulk job finished, x/y surface job finished, x/y O job finished, and x/y OH job finished.\n 
    #     Below is what's still in progress: 
    #     extract process of each candidate
    #         if bulk not finished
    #             outString += bulk status\n
    #         else 
    #             For each surface (termination_idx)
    #                 if surface not finished
    #                     outString += surface status\n
    #                 else
    #                     for each O_relax (at site_idx)
    #                         if O_relax not finish
    #                             outString += O_relax status\n
    #                         else
    #                             if delta_G(O) is small and no OH_relax at site_idx on termi_idx:
    #                                 outString += this site is good but no OH_relax\n
    #                             elif delta_G(O) is small 
    #                                 gather status of all OH_relax at site_idx on termi_idx
    #                                 if any status contain "failed":
    #                                     outString += OH_relax failed\n
    #                                 elif any status conttain "pending":
    #                                     outString += OH_relax pending\n
    #                                 elif any status contain "running":
    #                                     outString += OH_relax running\n
    #                                 else
    #                                     do nothing
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    outString = ""
    all_candidates_id = EXPLOG.relational_frame.candidates.df["candidate_id"].tolist()
    for cant_id in all_candidates_id:
        sub_pdf = EXPLOG.relational_frame.processes.df[EXPLOG.relational_frame.processes.df['candidate_id'] == cant_id]
        # number of bulk job finished
        N_finished_bulk = len(sub_pdf[(sub_pdf['job_type'] == 'bulk_relaxation') & (sub_pdf['status'] == 'completed')])
        N_finished_surface = len(sub_pdf[(sub_pdf['job_type'] == 'surface_relaxation') & (sub_pdf['status'] == 'completed')])
        N_finished_O = len(sub_pdf[(sub_pdf['job_type'] == 'O_adsorption') & (sub_pdf['status'] == 'completed')])
        N_finished_OH = len(sub_pdf[(sub_pdf['job_type'] == 'OH_adsorption') & (sub_pdf['status'] == 'completed')])
        
        N_bulk_tot = len(sub_pdf[sub_pdf['job_type'] == 'bulk_relaxation'])
        N_surf_tot = len(sub_pdf[sub_pdf['job_type'] == 'surface_relaxation'])
        N_O_tot = len(sub_pdf[sub_pdf['job_type'] == 'O_adsorption'])
        N_OH_tot = len(sub_pdf[sub_pdf['job_type'] == 'OH_adsorption'])
        
        outString += f'candidate {cant_id} has {N_finished_bulk}/{N_bulk_tot} bulk relaxation job finished, {N_finished_surface}/{N_surf_tot} surface relaxation job finished, {N_finished_O}/{N_O_tot} O adsorption job finished, and {N_finished_OH}/{N_OH_tot} OH adsorption job finished.\n'
        
        #-------------------------------------------------
        # Maybe we don't need to display so much info
        #-------------------------------------------------
        # outString += "Below is what's still in progress: \n"
        # # extrct job with type "bulk_relaxation"
        # bulk_relaxation_job = sub_pdf[sub_pdf['job_type'] == 'bulk_relaxation']
        # if len(bulk_relaxation_job) == 0:
        #     continue
        # elif bulk_relaxation_job.status.tolist()[0] != 'completed':
        #     outString += f'    candidate {cant_id} has bulk relaxation job which is {bulk_relaxation_job.status.tolist()[0]}\n'
        # else:
        #     # extract jobs with type "surface_relaxation"
        #     surface_relaxation_jobs = sub_pdf[sub_pdf['job_type'] == 'surface_relaxation']
        #     for _, row in surface_relaxation_jobs.iterrows():
        #         termi_idx = row['termination_index']
        #         if row['status'] != 'completed':
        #             outString += f'        candidate {cant_id} has surface relaxation job for termination {termi_idx} which is {row["status"]}\n'
        #         else:
        #             O_relaxation_jobs = sub_pdf[(sub_pdf['job_type'] == 'O_adsorption') & (sub_pdf['termination_index'] == termi_idx)]
        #             for _, row in O_relaxation_jobs.iterrows():
        #                 site_idx = row['site_index']
        #                 if row['status'] != 'completed':
        #                     outString += f'            candidate {cant_id} has O adsorption job for termination {termi_idx} and site {site_idx} which is {row["status"]}\n'
        #                 else:
        #                     # check if delta_G_O is small
        #                     # if small, check OH jobs at the same termination and site
        #                     # if not small, do nothing
        #                     tmp_delta_GO = EXPLOG.relational_frame.candidates[cant_id].study_obj.get_adsorption_site_study(term_idx=termi_idx, site_idx=site_idx).delta_G_O
        #                     # extract OH adsorption jobs at the same termination and site
        #                     OH_relaxation_jobs = sub_pdf[(sub_pdf['job_type'] == 'OH_adsorption') & (sub_pdf['termination_index'] == termi_idx) & (sub_pdf['site_index'] == site_idx)]
        #                     if tmp_delta_GO < 0.1 and len(OH_relaxation_jobs) == 0:
        #                         outString += f'                candidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, but no OH adsorption job yet\n'
        #                     elif tmp_delta_GO < 0.1:
        #                         tmp_status_list = OH_relaxation_jobs['status'].tolist()
        #                         if 'failed' in tmp_status_list:
        #                             outString += f'                andidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, but OH adsorption job failed\n'
        #                         elif 'pending' in tmp_status_list:
        #                             outString += f'                candidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, but OH adsorption job pending\n'
        #                         elif 'running' in tmp_status_list:
        #                             outString += f'                candidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, and OH adsorption job is running\n'
        #                         elif 'completed' in tmp_status_list:
        #                             outString += f'                candidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, and OH adsorption job is completed\n'
        #                         else:
        #                             # unrecoverable
        #                             outString += f'                candidate {cant_id} has a promising O adsorption at termination {termi_idx} and site {site_idx} with delta_G_O {tmp_delta_GO}, but OH adsorption job status is unrecoverable\n'
    
    id = CANVAS.register_tool_output(
        tool_name="inspect_explog",
        args={},
        value=outString,
        description=f"High level summary of the candidates study progress based on the latest EXPLOG update.",
        parent_result_ids=[],
        metadata={}
    )
    
    return f"{outString}\nMessage_ID: {id}. Please refer to this ID for the summary of the candidates study progress."
            

@tool
def old_inspect_explog(only_get_updates: Annotated[bool, "Whether to only get updates since last inspection."] = False) -> str:
    """Inspect the experiment log to get a high level and short summary of the candidates and processes."""
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    
    all_candidates_id = EXPLOG.relational_frame.candidates.df["candidate_id"].tolist()
    
    finishish_mask = EXPLOG.relational_frame.candidates.df["idealOverPotential"].notna()
    finishish_candidate_ids = EXPLOG.relational_frame.candidates.df.loc[finishish_mask, "candidate_id"].tolist()
    
    unfinished_candidate_ids = [can for can in all_candidates_id if can not in finishish_candidate_ids]
          
    finalAnswer = f"""You'v started {len(all_candidates_id)} candidates in total,
    You've finished study at least one OH adsorption on {len(finishish_candidate_ids)} systems,
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
            elif 'unrecoverable' in sub_pdf.status.tolist():
                cand_status += 'unrecoverable'
            # elif 'un-submitted' in sub_pdf.status.tolist():
            #     cand_status += 'un-submitted'
            else:
                raise ValueError(f'unknown status for {cant_id}, status list: {sub_pdf.status.tolist()}')
            
        finalAnswer += cand_status + "\n"

    return finalAnswer

@tool
def query_explog(
    table_name: Annotated[str, "Table to query: 'candidates' (one row per candidate, with best available OER metrics) or 'processes' (one row per DFT job, with per-site adsorption energies and overpotentials)."],
    reason: Annotated[str, "reason behind the query. Why are you using such filters and sort? What are you looking for?"],
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
) -> str:
    """Query the experiment log (EXPLOG) with optional filter and sort criteria. Automatically
    fetches the latest job updates before returning. Returns the filtered table as a string with row index.

    candidates table contains:
        candidate_id (str, MaterialID of the candidate),
        reason_or_hypothesis (str, for selecting the candidate),
        notes (str, any notes you've added for the candidate),
        G(O) deviation (Float64, deviation of best available G(O) from ideal 2.46 eV),
        Overpotential_from_scaling (Float64, best available overpotential from OH-OOH scaling relation),
        idealOverPotential (Float64, best available ideal overpotential across all studied sites)
    processes table contains:
        process_id (str, unique id for each process),
        candidate_id (str, MaterialID of the candidate this process belongs to),
        job_type (str, type of the DFT calculation, either bulk_relaxation, surface_relaxation, O_adsorption, or OH_adsorption),
        slurmID (str, the slurm ID of the job, int in str format. NaN for un-submitted jobs),
        status (str, current status of the job, either un-submitted, submitted, pending, running, completed, or failed),
        termination_index (Int64, termination index for surface relaxation and adsorption calculations, NaN for bulk relaxation),
        site_index (Int64, adsorption site index for adsorption calculations, NaN for bulk and surface relaxation),
        processNote (str, any note you've left for this process),
        G(O) (Float64, adsorption free energy of O* at this site in eV, NaN until O_adsorption job completes),
        G(O) deviation (Float64, absolute deviation of G(O) from ideal 2.46 eV),
        G(OH) (Float64, adsorption free energy of OH* at this site in eV, NaN until OH_adsorption job completes),
        G(OH) deviation (Float64, absolute deviation of G(OH) from ideal 1.23 eV),
        G(OOH) from scaling relation (Float64, G(OOH) = G(OH) + 3.2 eV),
        G(OOH, scaling) deviation (Float64, absolute deviation of G(OOH) from ideal 3.69 eV),
        ideal overpotential (Float64, best-case overpotential assuming optimal G(OOH), NaN until both G(O) and G(OH) are available),
        overpotential from OH-OOH scaling relation (Float64, overpotential using G(OOH) = G(OH) + 3.2 eV)
    """

    _ = EXPLOG.update_log()  # fetch latest job updates before querying

    # print dtype of both df
    print("candidates df dtype:\n", EXPLOG.relational_frame.candidates.df.dtypes)
    print("processes df dtype:\n", EXPLOG.relational_frame.processes.df.dtypes)

    if table_name == 'candidates':
        df = EXPLOG.relational_frame.candidates.df.copy()
        # drop the "study_obj" column since it contains complex objects that are not easy to display in a dataframe format
        df = df.drop(columns=["study_obj"])
    elif table_name == 'processes':
        df = EXPLOG.relational_frame.processes.df.copy()
        df = df.drop(columns=["VASP_dir"]) # drop the "VASP_dir" column since it contains file directory strings that are not easy to display in a dataframe format
    else:
        return "table_name must be either 'candidates' or 'processes'"

    filteredDF = df_query(df, filters, sort)
    
    result = filteredDF.to_string(index=True)
    
    id = CANVAS.register_tool_output(
        tool_name="query_explog",
        args={
            "table_name": table_name,
            "reason": reason,
        },
        value=result,
        description=f"Result of querying the EXPLOG {table_name} table with reason: {reason}, filters: {filters}, and sort: {sort}.",
        reasons={'reason': reason},
        parent_result_ids=[],
        metadata={
            "table_name": table_name,
            "reason": reason,
        }
    )

    print(result)
    return f"{result}\nQuery_result_ID={id}. Please refer to this ID if you want to use the query result for further analysis or decision making."
    
    

@tool
def read_explog(
    candidate_id: Annotated[str, "MaterialId of the candidate to read the experiment log for."],
    reasons: Annotated[str, "Why are you interested in this candidate? What do you want to find out from the experiment log?"],
    ) -> str:
    """
    Get a summary of the experiment log for a specific candidate, including all related job information and details
    with respect to calculated adsorption energies. Automatically fetches the latest updates from the job handler
    before returning: no manual update needed.

    Details such as the site type, on-top element, closest neighboring elements, reduced coordination, G(O), G(OH),
    G(OOH) from the OH-OOH scaling relation, ideal overpotential, and overpotential from the scaling relation are
    provided given the necessary calculations have finished.
    """
    _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    cadidate_row_df = EXPLOG.relational_frame.candidates[candidate_id].df
    cadidate_row_df = cadidate_row_df.copy().drop(columns=["study_obj"])
    related_process_df = EXPLOG.relational_frame.candidates[candidate_id].processes.df
    related_process_df = related_process_df.copy().drop(columns=["VASP_dir"])    
    
    answer = f"Candidate information:\n{cadidate_row_df.to_string(index=False)}\n\nRelated processes information:\n{related_process_df.to_string(index=False)}\n"
    # for each row in related_process_df, if the job_type is either O_adsorption or OH_adsorption, add the corresponding site information to the answer by calling the _list_adsorption_sites function
    
    # Init set of "seen" pairs of (termination_index, site_index).
    # Is used to avoid repeating the same site information.
    seen_pairs = set()
    
    rows = []  # collect each 1-row dataframe here
    for index, row in related_process_df.iterrows():
        if row['job_type'] in ['O_adsorption', 'OH_adsorption']:

            # The the indecies identifying the adsorption site:
            termination_index = row['termination_index']
            site_index = row['site_index']

            # If either is Nan - continue:
            if pd.isna(termination_index) or pd.isna(site_index):
                continue

            # Define the pair of indecies, and continue if it has already been considered before:
            index_pair = (int(termination_index), int(site_index)) 
            if index_pair in seen_pairs:
                continue
            seen_pairs.add(index_pair) # Add pair to the set of "seen_pairs"

            site_info_df = _list_adsorption_sites(candidate_id, termination_index, only_reduced_coord_O_sites=True)
            # extract the row where the "Site index" column is equal to site_index
            site_info_row = site_info_df[site_info_df['Site index'] == site_index]
            
            new_site_info_row = site_info_row.copy()
            rows.append(new_site_info_row)
    
    if len(rows) > 0:                    
        final_site_info = pd.concat(rows, ignore_index=True)
        answer += f"\nThe adsorption site information is:\n{final_site_info.to_string}\n"
        
    id = CANVAS.register_tool_output(
        tool_name="read_explog",
        args={
            "candidate_id": candidate_id,
        },
        value=answer,
        description=f"Summary of the experiment log for candidate {candidate_id} with reason: {reasons}.",
        reasons={'reasons': reasons},
        parent_result_ids=[],
        metadata={
            "candidate_id": candidate_id,
            "reasons": reasons,
        }
    )

    return f"{answer}\nQuery_result_ID={id}. Please refer to this ID if you want to use this query result for further analysis or decision making."

# @tool
# def get_top_k_candidates(
#     k: Annotated[int, "Number of top candidates to retrieve based on ideal overpotential."],
#     ) -> str:
#     """Get the top k candidates with the lowest ideal overpotential from the experiment log."""
#     _ = EXPLOG.update_log() # get the latest updates from the job handler and update the relational frame accordingly
#     # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
#     # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
#     #     pickle.dump(EXPLOG, f)
#     candidates_df = EXPLOG.relational_frame.candidates.df.copy()
#     candidates_df = candidates_df[candidates_df['idealOverPotential'].notna()]
#     if len(candidates_df) == 0:
#         return "No candidates has ideal overpotential information."
#     candidates_df["idealOverPotential"] = candidates_df["idealOverPotential"].apply(lambda x: float(x))
#     N_finished = len(candidates_df)
#     top_k_candidates = candidates_df.nsmallest(k, 'idealOverPotential')
#     top_k_candidates = top_k_candidates.copy().drop(columns=["study_obj"])
#     answer = f"Top {k} out of {N_finished} candidates with the lowest ideal overpotential:\n{top_k_candidates.to_string(index=False)}\n\nYou may run more calculations on those candidates at different terminations and sites, or you can also run more calculations on other candidates to expand the pool and find more promising candidates."
    
#     id = CANVAS.register_tool_output(
#         tool_name="get_top_k_candidates",
#         args={
#             "k": k,
#         },
#         value=answer,
#         description=f"Top {k} candidates with the lowest ideal overpotential from the EXPLOG.",
#         parent_result_ids=[],
#         metadata={
#             "k": k,
#         }
#     )
    
#     return f"{answer}/nQuery_result_ID={id}. Please refer to this ID if you want to use the query result for further analysis or decision making."

# @tool
# def get_explog_updates()

@tool
def enter_candidate_in_log(
    reason_or_hypothesis: Annotated[str, "Detailed Reason and hypothesis for selecting this candidate. To be used later for analysis and summarization."],
    df_name: Annotated[str, "Key of the dataframe in CANVAS containing the candidate entry."],
    df_name_ref: Annotated[str, "Reference ID for the dataframe name, used for traceability."],
    MaterialId: Annotated[str, "MaterialId of the candidate in the dataframe."],
    MaterialId_ref: Annotated[str, "Reference ID of the result where you find the MaterialId of interests."],
    note: Annotated[str | None, "Any notes you want to add."] = None,
    ) -> str:
    """
    Initialize a catalyst candidate from an AQ-GNoME dataframe in the experiment log (EXPLOG),
    enabling it to be studied further with DFT.

    Reads the candidate's structure from the dataframe stored in CANVAS under
    `df_name`, initialises an OER catalyst study object.
    """
    for ref in [df_name_ref, MaterialId_ref]:
        if ref:
            art = CANVAS.get_artifact(ref)
            if art is None:
                return f"Error: Reference ID {ref} not found in CANVAS."

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
        H2O_gas_free_energy = -14.183498, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
        H2_gas_free_energy = -7.027336, # <--- should be the DFT energy + free energy corrections, at the relevant level of theory
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
    
    id = CANVAS.register_tool_output(
        tool_name="enter_candidate_in_log",
        args={
            "reason_or_hypothesis": reason_or_hypothesis,
            "df_name": df_name,
            "MaterialId": MaterialId,
            "note": note,
        },
        value=message,
        description=f"Entry of candidate {MaterialId} into the experiment log with reason: {reason_or_hypothesis} and note: {note}.",
        reasons={'reason_or_hypothesis': reason_or_hypothesis},
        parent_result_ids=[df_name_ref, MaterialId_ref],
        metadata={
            "reason_or_hypothesis": reason_or_hypothesis,
            "df_name": df_name,
            "df_name_ref": df_name_ref,
            "MaterialId": MaterialId,
            "MaterialId_ref": MaterialId_ref,
            "note": note,
        }
    )

    return f"{message}\nMessage_ID={id}. Refer to this ID if you need to refer back to this message later"

@tool
def submit_dft_job(
    MaterialId: Annotated[str, "MaterialId of the candidate for which to submit a DFT job."],
    MaterialId_ref: Annotated[str, "Reference ID of the result where you find the MaterialId of interests."],
    calculation_type: Annotated[Literal['bulk_relaxation', 'surface_relaxation', 'OH_adsorption', 'O_adsorption'], "Type of DFT calculation to submit. O_adsorption yields G(O); OH_adsorption yields G(OH) — both are required to compute overpotentials. OH_adsorption submits three jobs with slightly different initial adsorbate positions to increase the likelihood of finding the global minimum."],
    note: Annotated[str, "Short note for the calculation; state the reason for submitting the job, including why the selected termination and adsorption site are relevant."],
    termination_index: Annotated[int | None, "Termination index. Only required for surface and adsorption calculations."] = None,
    termination_index_ref: Annotated[str, "Reference ID of the result or output message where you determind to submit a dft job for this termination index."] = "",
    ad_site_index: Annotated[int | None, "Index of the site onto which O or OH is adsorbed. Only required for adsorption calculations."] = None,
    ad_site_index_ref: Annotated[str, "Reference ID of the result or output message where you determind to submit a dft job for this adsorption site index."] = "",
    partition: Annotated[Literal['xeon56', 'xeon40el8', 'xeon24el8', 'auto'], "HPC partition to submit the job to. Use 'auto' to let the system select the partition automatically."] = "auto",
) -> str:
    """
    Submit a DFT job for a candidate to the HPC cluster.

    Prerequisites:
        - The candidate must first be registered in EXPLOG via enter_candidate_in_log.
        - For surface_relaxation, O_adsorption, and OH_adsorption: call
          get_terminations_ranking and list_adsorption_sites first to identify
          a suitable termination index and adsorption site index.

    For bilk_relaxation jobs, if magnetic elements are present in a bulk structure, 
    several bulk jobs will be submitted to explore the most likely magnetic configurations.
    OH_adsorption submits three jobs with slightly varied initial adsorbate
    positions to increase the likelihood of finding the global minimum.
    Only the global minimum will be reported and used.
    To obtain the OH adsorption energy, the O adsorption calculation must be completed first.

    Returns a confirmation message including the submitted process ID(s).
    """
    
    for ref in [MaterialId_ref, termination_index_ref, ad_site_index_ref]:
        if ref:
            art = CANVAS.get_artifact(ref)
            if art is None:
                return f"Error: Reference ID {ref} not found in CANVAS."

    # Does candidate exist in EXPLOG:
    try:
        study = EXPLOG.relational_frame.candidates[MaterialId].study_obj
    except Exception:
        raise ValueError(f"Unknown candidate MaterialId: {MaterialId}")

    # Ensure surface study exists when needed - initialize it if needed:
    if calculation_type in ["surface_relaxation", "O_adsorption", "OH_adsorption"]:
        if termination_index is None:
            raise ValueError(
                f"{calculation_type} requires termination_index."
            )
        if not isinstance(termination_index, int) or termination_index < 0:
            raise ValueError("termination_index must be a non-negative int.")

        surface_studies = study.get_surface_studies() or {}
        if termination_index not in surface_studies:
            study.initialize_oer_surface_study(termination_index)

        # Re-fetch after potential initialization
        surface_studies = study.get_surface_studies() or {}
        if termination_index not in surface_studies:
            raise RuntimeError(
                "Failed to initialize/get surface study for "
                f"termination_index={termination_index}."
            )
        surface_study = surface_studies[termination_index]

    # Ensure adsorption-site study exists when needed - initialize it if needed:
    if calculation_type in ["O_adsorption", "OH_adsorption"]:
        if ad_site_index is None:
            raise ValueError(f"{calculation_type} requires ad_site_index.")
        if not isinstance(ad_site_index, int) or ad_site_index < 0:
            raise ValueError("ad_site_index must be a non-negative int.")

        ad_site_studies = surface_study.get_adsorption_site_studies_dict() or {}
        if ad_site_index not in ad_site_studies:
            surface_study.initialize_adsorption_site_study(ad_site_index)

        # Re-fetch after potential initialization
        ad_site_studies = surface_study.get_adsorption_site_studies_dict() or {}
        if ad_site_index not in ad_site_studies:
            raise RuntimeError(
                "Failed to initialize/get adsorption-site study for "
                f"ad_site_index={ad_site_index} on "
                f"termination_index={termination_index}."
            )

    # EXPLOG check --- before mutation of EXPLOG
    validation_result = EXPLOG.can_add_process(
        MaterialId,
        calculation_type,
        termination_index,
        ad_site_index,
    )
    if not validation_result.ok:
        validation_result.raise_for_error()
        assert False, "Unreachable: raise_for_error() should have raised"

    # a list of ids will be provided for OH_calculations and not for all other:
    id_list = EXPLOG.add_process(MaterialId, calculation_type, termination_index, ad_site_index, note)
    if not isinstance(id_list, list):
        id_list = [id_list]

    for process_id in id_list:
        EXPLOG.submit_process(process_id, partition)
    # save EXPLOG into a pickle file under WORKING_DIRECTORY for record and future reference
    # with open(os.path.join(var.my_WORKING_DIRECTORY, "EXPLOG.pkl"), "wb") as f:
    #     pickle.dump(EXPLOG, f)
    
    outStr = f"Submitted {calculation_type} for candidate {MaterialId}. Process ID(s): {id_list}."
    
    id = CANVAS.register_tool_output(
        tool_name="submit_dft_job",
        args={
            "MaterialId": MaterialId,
            "calculation_type": calculation_type,
            "termination_index": termination_index,
            "ad_site_index": ad_site_index,
            "note": note,
        },
        value=id_list,
        listed_value=True,
        description=f"Submission of {calculation_type} for candidate {MaterialId} with termination index {termination_index} and adsorption site index {ad_site_index}. Note: {note}",
        reasons={'note': note},
        parent_result_ids=[id for id in [MaterialId_ref, termination_index_ref, ad_site_index_ref] if id],
        metadata={
            "MaterialId": MaterialId,
            "calculation_type": calculation_type,
            "termination_index": termination_index,
            "ad_site_index": ad_site_index,
            "note": note,
        }
    )
    
    return f"{outStr}\nReference ID for the Process ID(s) is {id}. Please refer to this reference ID if the corresponding process id is needed."


@tool
def get_terminations_ranking(
    candidate_id: Annotated[str, "MaterialId of the candidate for which to get termination rankings."],
    candidate_id_ref: Annotated[str, "Reference ID of the result where you find the MaterialId of interests."],
    reasons: Annotated[str, "Why are you interested to know the termination ranking for this candidate? What do you want to find out from the termination ranking?"],
    #max_miller: Annotated[int, "Maximum Miller index to consider for surface generation."] = 1,
) -> str:
    """
    Ranks surface terminations for a given candidate using a coordination-based
    heuristic: terminations where surface atoms retain more of their bulk
    coordination are ranked higher (in terms of normalized score) as a proxy for stability. This is not based
    on calculated surface energies — it is a heuristic guide only.
    The score is the reduced coordination per surface area (Å⁻²) — a less negative
    score indicates less disruption of bulk coordination, corresponding to a higher
    normalized score.

    This tool must be called before any surface relaxation or adsorption calculations.
    Once run, the ranking is fixed and will not be recalculated on subsequent calls.

    Output includes: Miller indices, termination index, normalized score, and
    exposed surface atom types. It is recommended to also call
    'list_adsorption_sites' to inspect the available adsorption sites before
    committing to a termination.
    """
    for ref in [candidate_id_ref]:
        if ref:
            art = CANVAS.get_artifact(ref)
            if art is None:
                return f"Error: Reference ID {ref} not found in CANVAS."

    # Old part of docksrting:
        # This function must be run before any surface relaxation or adsorption
        # calculations, as it generates all surface structures up to the specified
        # maximum Miller index. Once run, the ranking is fixed and will not be
        # recalculated on subsequent calls.

    # max_miller now fixed to 1
    max_miller = 1
    
    # Arguments left fixed for now:
    method = 'all'      # What coordination to consider
    stoichiometry_tolerance = 0.2 # Allowed stoichiometry deviation from bulk
    all_species_present = True # Only surfaces with all bulk species present
    symmetrize = True # Whether to symmetrize the surfaces
    select_closest_O_stoichiometry = True # Select surfaces with closest O stoichiometry to bulk
    min_slab_thickness = 9 # Minimum slab thickness in Å
    max_slab_thickness = 20 # Maximum slab thickness in Å
    min_atoms = 20 # Minimum number of atoms in the slab
    max_atoms = 120 # Maximum number of atoms in the slab
    max_layers = 6 # Maximum number of layers considered when building slabs

    if max_miller < 1:
        raise ValueError("max_miller must be at least 1.")

    # Ensure candidate exists in EXPLOG:
    try:
        study = EXPLOG.relational_frame.candidates[candidate_id].study_obj
    except Exception:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")

    out_string = ''

    ranking = study.get_termination_rankings() # None on first call
    if ranking is None:
        out_string += f"This is the first termination ranking for candidate {candidate_id}:"
        
        # Will fail if the relaxed bulk has not been set (will raise an RuntimeError)
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
        out_string += f"Termination ranking for candidate {candidate_id} has already been " \
            "determined, so the same ranking is returned as before:"
        
    # Sort terminations by normalized score in descending order:
    ranking = ranking.sort_values('Normalized score', ascending=False)

    has_valid_surfaces = len(study.get_terminations_dict()) > 0
    if not has_valid_surfaces:
        new_status = "unrecoverable: no valid surface can be determined"

        pdf = EXPLOG.relational_frame.processes.df
        mask = (
            (pdf["candidate_id"] == candidate_id)
            & (pdf["job_type"] == "bulk_relaxation")
        )

        for idx in pdf[mask].index:
            process_id = int(pdf.at[idx, "process_id"])
            EXPLOG.relational_frame.processes.set_value(
                process_id, "status", new_status
            )
    
    out_string += ranking.to_string(index=True)

    # Append original reason or hypothesis if it exists:
    candidate_reason = EXPLOG.relational_frame.candidates[candidate_id].reason_or_hypothesis
    if candidate_reason is not None and candidate_reason != "":
        out_string += "\n\nOriginal reason or hypothesis for selecting this candidate:\n"
        out_string += candidate_reason

    if not has_valid_surfaces:
        out_string += "\n\nNo valid surface terminations could be determined for this candidate. "\
                      "No further surface or adsorption calculations can be performed for this " \
                      "candidate, and the candidate is marked as unrecoverable."
                      
    id = CANVAS.register_tool_output(
        tool_name="get_terminations_ranking",
        args={
            "candidate_id": candidate_id,
            "reasons": reasons,
        },
        value=out_string,
        description=f"Termination ranking for candidate {candidate_id} with reason: {reasons}.",
        reasons={'reasons': reasons},
        parent_result_ids=[candidate_id_ref],
        metadata={
            "candidate_id": candidate_id,
            "reasons": reasons,
        }
    )

    return f"{out_string}\nMessage_ID: {id}. Please refer to this ID if you want to refer back to this message later or use the termination ranking for further analysis or decision making."


def _list_adsorption_sites(
        
    candidate_id, 
    termination_index, 
    only_reduced_coord_O_sites = True
):

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
    
    return df

@tool
def list_adsorption_sites(
    candidate_id: Annotated[str, "MaterialId of the candidate to list adsorption sites for."],
    candidate_id_ref: Annotated[str, "Reference ID of the result where you find the MaterialId of interests."],
    termination_index: Annotated[int, "Termination index of the surface to list adsorption sites for."],
    termination_index_ref: Annotated[str, "Reference ID of the result where you determine the termination index for which to list adsorption sites."],
    reasons: Annotated[str, "Why are you interested to know the adsorption sites for this candidate at this termination? What do you want to find out from the adsorption sites information?"],
    # only_reduced_coord_O_sites = True, DISABLED FOR NOW...
):
    """
    This tool requires get_terminations_ranking to have been called first for the candidate.

    Gives a preliminary list of adsorption sites if the termination has not been relaxed yet, or a list of final
    adsorption sites if the termination has been relaxed. Sites may be 'on-top' or 'lattice O', the latter being an
    exposed surface oxygen atom that is part of the lattice and may act as an adsorption site. For 'on-top' sites, the
    'element of the adsorption site' is listed, meaning the element which the adsorbate is placed on top of.
    Additionally, a list of the closest neighboring elements of the adsorption site is given, with the distance to
    these neighbors given as (neighbor element, distance [Å]). For 'lattice O' sites, the reduced coordination of the
    lattice O is given, which is a measure of how many neighboring atoms the lattice O has compared to a fully
    coordinated lattice O in the bulk structure (e.g., a reduced coordination of 1 means that the lattice O atom has a
    decreased coordination of 1.). Since this function can be called repeatedly, there is no need to write the result
    to the canvas.
    """
    for ref in [candidate_id_ref, termination_index_ref]:
        if ref:
            art = CANVAS.get_artifact(ref)
            if art is None:
                return f"Error: Reference ID {ref} not found in CANVAS."
    
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
        
    id = CANVAS.register_tool_output(
        tool_name="list_adsorption_sites",
        args={
            "candidate_id": candidate_id,
            "termination_index": termination_index,
        },
        value=out_string,
        description=f"List of adsorption sites for candidate {candidate_id} on termination index {termination_index}.",
        reasons={'reasons':reasons},
        parent_result_ids=[candidate_id_ref, termination_index_ref],
        metadata={
            "candidate_id": candidate_id,
            "termination_index": termination_index,
        }
    )

    return f"{out_string}\nMessage_ID: {id}. Please refer to this ID if you want to refer back to this message later or use the adsorption sites information for further analysis or decision making."


def _count_magnetic_sites_vectorized(compositions: pd.Series) -> pd.Series:
    """
    Vectorized alternative to applying count_magnetic_sites_from_formula row-by-row.

    How it works:
      1. str.extractall runs a single C-level regex pass over the entire Series,
         pulling out every (element_symbol, count) pair from every formula string
         at once — e.g. "Fe2O3" yields [("Fe","2"), ("O","3")].
         The result is a multi-index DataFrame: outer index = original row index,
         inner index = match number within that row.

      2. We cast the count column to int and keep only rows whose element symbol
         appears in _MAGNETIC_ELEMENTS (a frozenset built once from pymatgen's
         DEFAULT_MAGMOMS keys).

      3. groupby(level=0) groups by the original row index and sums the remaining
         counts, giving one total per formula.

      4. reindex restores any rows that had zero magnetic atoms (they were dropped
         by the filter in step 2) and fills them with 0.

    This avoids constructing a pymatgen Composition object per row, which is the
    bottleneck in the .apply() approach on a database of hundreds of thousands of entries.

    Assumes formulas always carry explicit integer counts per element
    (e.g. "Cs1S6Zr3"), which holds for the GNoME dataset.
    """
    _magnetic_elements = frozenset(DEFAULT_MAGMOMS.keys())

    extracted = compositions.str.extractall(r'([A-Z][a-z]?)(\d+)')
    extracted.columns = ['element', 'count']
    extracted['count'] = extracted['count'].astype(int)
    extracted = extracted[extracted['element'].isin(_magnetic_elements)]
    result = extracted.groupby(level=0)['count'].sum()
    return result.reindex(compositions.index, fill_value=0)


class _StabilityCache:
    """
    Initialised once at module load. Loads Data_Handler, applies all standard
    screening filters, and stores the filtered df for the acidic-OER workflow.
    """
    SOLID_FILTER: bool = True
    GGA_ONLY: bool = False
    PHS: float = 0              # pH point (single value = exact grid point, not a range)
    US: list[float] = [1.2, 2.0]  # U vs SHE window (V)

    ELEMENTS_TO_EXCLUDE: list[str] = [
        'P', 'B', 'S', 'C', 'F',
        'Tc', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
        'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Pm', 'Ac', 'Th', 'Pa',
        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
        'Lr', 'Po', 'At', 'Rn',
    ]
    ELEMENTS_TO_INCLUDE: list[str] = ['O']
    MAX_MAGNETIC_SITES: int = 10

    def __init__(self):
        print("  [1/4] Loading Data_Handler (CSVs + H5PY databases)...", flush=True)
        _t = time.time()
        self.dh = Data_Handler(solid_filter=self.SOLID_FILTER,
                               gga_only=self.GGA_ONLY,
                               path_to_data_directory=None)
        print(f"  [1/4] Done in {time.time() - _t:.1f}s.", flush=True)

        print("  [2/4] Applying element filters...", flush=True)
        _t = time.time()
        self.dh.remove_entries_with_elements(self.ELEMENTS_TO_EXCLUDE)
        self.dh.remove_entries_without_elements(self.ELEMENTS_TO_INCLUDE, True)
        print(f"  [2/4] Done in {time.time() - _t:.1f}s.", flush=True)

        print("  [3/4] Applying dimensionality filter (keep 3D only)...", flush=True)
        _t = time.time()
        wdf = self.dh._working_df
        wdf = wdf[wdf['Dimensionality Cheon'] == '3D']
        self.dh._working_df = wdf
        print(f"  [3/4] Done in {time.time() - _t:.1f}s. Entries remaining: {len(wdf)}", flush=True)

        print("  [4/4] Snapshotting filtered dataframe...", flush=True)
        _t = time.time()
        self._df = self.dh.get_df()
        print(f"  [4/4] Done in {time.time() - _t:.1f}s.", flush=True)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

print("Loading GNoME database into _STABILITY_CACHE (may take several minutes)...", flush=True)
_t0 = time.time()
_STABILITY_CACHE = _StabilityCache()
print(f"GNoME database loaded in {time.time() - _t0:.1f}s.", flush=True)


@tool
def OER_data_analasis_v2(
    # pHs: Annotated[Union[List[float], float], "The pH in which the materials should be stable, may either be a float (specifying a single pH) or a pH range specified as two floats in a list i.e. [min, max]"],
    # Us: Annotated[Union[List[float], float], "Electrochemical potential in which the materials should be stable, may either be a float (specifying a single potential) or a potential range specified as two floats in a list i.e. [min, max]"],
    decomposition_threshold: Annotated[float, "Decomposition energy threshold for stability criteria, in units of eV/atom (pourbaix stability)"],
    save_name: Annotated[str, "Key under which the resulting dataframe is saved in CANVAS. Use a descriptive name to distinguish between runs with different criteria."],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'decomposition_threshold', 'save_name', 'filters', 'sort'"],
    overwrite: Annotated[bool, "If True, overwrite an existing dataframe stored under the same key in CANVAS. If False (default), the tool will abort if a dataframe with that key already exists."] = False,
    # dir_of_data: Annotated[Optional[str], "Path to data directory. If None, use default data directory."] = None,
    # elements_to_exclude: Annotated[List[str], "List of element symbols to exclude from the analysis."] = [], 
    # elements_whic_must_be_included: Annotated[List[str], "List of element symbols that must be included in the analysis."] = [],
    # ref_pHs: Annotated[Union[List[str], str], "List or a single value of reference_ID where you determined the value of pH(s) from"] = "",
    # ref_Us: Annotated[Union[List[str], str], "List or a single value of reference_ID where you determined the value of potential(s) from"] = "",
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    ) -> str:
    """
    Get a pandas dataframe of material entries from the AQ-GNoME database that fulfill
    given Pourbaix stability criteria under specified electrochemical conditions
    (pH and potential U vs. SHE), and save the resulting filtered dataframe to
    CANVAS under the key specified by `save_name`.

    The output dataframe contains material entries that are stable under the
    specified conditions, with columns including: MaterialId, Composition,
    Reduced Formula, Elements, Bandgap, HHI indices (availability/cost proxy),
    and Disorder Probability.

    The following filters are hardcoded and always applied:
        - Elements P, B, S, C, F are excluded.
        - All radioactive elements are excluded.
        - Only O-containing materials are included.
        - Candidates already present in the experiment log (EXPLOG) are excluded.

    The optional `filters` and `sort` parameters can be used to further refine
    or order the results based on the output dataframe columns.
    """
    
    # verify refs
    # for refs in [ref_pHs, ref_Us]:
    #     refs = refs if isinstance(refs, list) else [refs]
    #     for ref in refs:
    #         if ref:
    #             art = CANVAS.get_artifact(ref)
    #             if art is None:
    #                 return f"Error: Reference ID {ref} not found in CANVAS."
    
    # Data handler with all standard filters pre-applied:
    dh = _STABILITY_CACHE.dh  

    # Stablity criteria, with decomposition threshold set by the agent:
    SCS = [Stability_Criteria(pHs = _STABILITY_CACHE.PHS,
                              Us = _STABILITY_CACHE.US,
                              decomposition_threshold = decomposition_threshold)]
    # Define the sorter object, and get entries that satisfy the stability criteria:
    se = Stable_Entries(dh, SCS)
    df = se.get_stable_df() # df with stable entries

    # Applied here (not at cache load time) because it runs row-by-row in Python
    # and is fast on the small stable subset but slow on the full database.
    df = df[df['Composition'].apply(count_magnetic_sites_from_formula) <= _STABILITY_CACHE.MAX_MAGNETIC_SITES]

    # Exclude candidates already being studied in the experiment log
    explog_ids = set(EXPLOG.relational_frame.candidates.df["candidate_id"].tolist())
    df = df[~df['MaterialId'].isin(explog_ids)]
    
    df = df_query(df, filters, sort)
    df = get_simplified_df(df)
    if len(df) == 0:
        return "No stable entries found based on the specified criteria and filters."
    
    # Save df:
    # WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # save_path = os.path.join(WORKING_DIRECTORY, 'stable_entries.csv')
    # df.to_csv(save_path, index=False)

    # Write to canvas:
    canvas_result = CANVAS.write(save_name, df, overwrite=overwrite)
    if "already exists" in canvas_result:
        return f"Aborted: {canvas_result} Use overwrite=True to overwrite."
    
    # tmp_ref_pHs = ref_pHs if isinstance(ref_pHs, list) else [ref_pHs]
    # tmp_ref_Us = ref_Us if isinstance(ref_Us, list) else [ref_Us]
    
    # parent_result_ids = [ref for ref in [*tmp_ref_pHs, *tmp_ref_Us] if ref] # only include non-empty refs
    # print("from tool OER_data_analasis_v2, parent_result_ids determined to be:")
    # print(parent_result_ids)
    
    df_id = CANVAS.register_tool_output(
        tool_name="OER_data_analasis_v2",
        args={
            # "pHs": pHs,
            # "Us": Us,
            "decomposition_threshold": decomposition_threshold,
            "save_name": save_name,
            "overwrite": overwrite,
            "filters":filters, 
            "sort":sort
        },
        value=save_name,
        description=f"key name of the saved dataframe",
        reasons=reasons,
        # parent_result_ids=parent_result_ids,
        metadata={},
    )
    
    outStr = ""
    if len(df) > 20:
        outStr += f"Stable entries data analysis completed, yielding {len(df)} entries. The dataframe has {len(df)} entries, too long to display here. Please check the dataframe in canvas with key '{save_name}' using browse_df tool. dataframe_ID={df_id}. Only use this ID as the reference ID when asked, still load the dataframe from canvas using the key name '{save_name}'."
    else:
        outStr += f"Stable entries data analysis completed, yielding {len(df)} entries. Below shows the dataframe with row index: \n{df.to_string(index=True)}. The same dataframe is also saved in canvas with key '{save_name}' and can be accessed using browse_df tool. dataframe_ID={df_id}. Only use this ID as the reference ID when asked, still load the dataframe from canvas using the key name '{save_name}'."
    
    result_id = CANVAS.register_tool_output(
        tool_name="OER_data_analasis_v2",
        args={
            # "pHs": pHs,
            # "Us": Us,
            "decomposition_threshold": decomposition_threshold,
            "save_name": save_name,
            "overwrite": overwrite,
        },
        value=outStr,
        description=f"Out string of the tool",
        reasons=reasons,
        # parent_result_ids=parent_result_ids,
        metadata={},
    )
    
    outStr += f"\nIf you want to reference any other information of the tool result, please refer to the result ID {result_id} if you need to use them to make decisions or conclusions."
    
    return outStr




@tool
def get_candidate_data(
    material_ids: Annotated[List[str], "List of MaterialIds to retrieve from the database."],
) -> str:
    """
    For a given list of MaterialIds, retrieve their full database rows from the
    AQ-GNoME dataset, including composition, bandgap, HHI availability/cost indices,
    disorder probability, and other properties.

    An additional column is appended with the worst-case Pourbaix decomposition
    energy (eV/atom) across the standard acidic-OER window stored in
    _STABILITY_CACHE (pH 0, U [1.2, 2.0] V vs SHE). The column is named
    e.g. 'max_dG_U[1.2,2.0]_pH0'. Lower values mean greater thermodynamic
    stability; negative values are fully stable in the window.

    Standard filters are pre-applied (solid_filter=True, gga_only=False, O-containing,
    no P/B/S/C/F/radioactives, 3D only, ≤10 magnetic sites). Results are sorted
    by the stability column ascending.
    """
    df = _STABILITY_CACHE.df  # returns a copy via @property
    df = df[df['MaterialId'].isin(material_ids)]

    not_found = [mid for mid in material_ids if mid not in df['MaterialId'].values]
    # TODO: reconsider handling before production — currently raises hard error
    if not_found:
        raise ValueError(f"MaterialIds not found in database: {not_found}")

    dh = _STABILITY_CACHE.dh
    # decomposition_threshold=10**5: dummy large value — only window indices are used, never evaluate()
    sc = Stability_Criteria(pHs=_STABILITY_CACHE.PHS, Us=_STABILITY_CACHE.US,
                            decomposition_threshold=10**5)

    max_decomp_values = []
    for _, row in df.iterrows():
        mixed_pbx_id = row['mixed_pbx_save_id']
        if mixed_pbx_id != 'Not computed':
            decom_G = dh.mixed_results.read_id(mixed_pbx_id)
        else:
            decom_G = dh.gga_results.read_id(row['gga_only_pbx_save_id'])
        max_decomp_values.append(sc.max_dG_in_region(decom_G))

    df = df.copy()
    df[sc.col_name] = max_decomp_values
    df = df.sort_values(sc.col_name)
    df = get_simplified_df(df)

    return df.to_string(index=True)


# @tool
# def extract_df(
#     df_name: Annotated[str, "Name of the dataframe in canvas to extract."],
#     filters: List[Filter] = [],
#     sort: List[SortSpec] = []
#     ):
#     """read the dataframe with a given filter and sort. This is useful to exam the filtered dataframe without altering its data"""
#     df = CANVAS.read(df_name)
#     df = df_query(df, filters, sort)
#     if len(df) > 50:
#         return f"Too many entries pass the filter. Please apply more filters to narrow down the results or check with material_IDs to find the specific entries you want to look at. showing the first 50 entries:\n {df.head(50).to_string(index=True)}"
#     return df.to_string(index=True)

    
# @tool
# def read_df(
#     df_name: Annotated[str, "Name of the dataframe in canvas to read."],
#     startIdx: Annotated[int, "Starting index of the dataframe to read."] = 0,
#     endIdx: Annotated[int, "Ending index of the dataframe to read."] = 10,
#     ) -> str:
#     """Read a portion of a dataframe (from row i to row j) from canvas and return it as a string with row index."""
#     if endIdx - startIdx > 50:
#         return "Read no more than 50 rows at a time."
#     df = CANVAS.read(df_name)
#     print(df)
#     return df.iloc[startIdx:endIdx].to_string(index=True)


@tool
def browse_df(
    df_name: Annotated[str, "Name of the dataframe in canvas to extract."],
    df_name_ref: Annotated[str, "Reference ID for the dataframe name"],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'df_name', 'startIdx', 'endIdx', 'filters', 'sort'"],
    startIdx: Annotated[int, "Row index to start reading from (inclusive). Use 0 to start from the beginning."] = 0,
    endIdx: Annotated[int, "Row index to stop reading at (exclusive). Maximum window is 50 rows (endIdx - startIdx <= 50)."] = 50,
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    ) -> str:
    """
    Read and inspect a dataframe stored in CANVAS by key, with optional filtering,
    sorting, and pagination.

    Filters and sort are applied first to the full dataframe, then the specified
    row window [startIdx, endIdx) is returned. This allows systematic exploration
    of large dataframes — for example, narrowing down candidates by column
    values without modifying the stored dataframe.

    Use the `filters` parameter to select rows by column value (e.g. filter by
    element composition, HHI index, or bandgap), and `sort` to order results before
    reading. Pagination via startIdx/endIdx can then be used to step through results
    that exceed the 50-row display limit.

    Returns the selected rows as a string with row index. The row index refers to
    the position in the filtered/sorted dataframe, not the original stored dataframe.
    """
    
    if df_name_ref:
        art = CANVAS.get_artifact(df_name_ref)
        if art is None:
            return f"Error: Reference ID {df_name_ref} not found in CANVAS."
        
    arg_ph = art.args["pHs"]
    arg_us = art.args["Us"]
    arg_decomposition_threshold = art.args["decomposition_threshold"]
    arg_filter = art.args.get("filters", [])
    arg_sort = art.args.get("sort", [])
    
    header = f"Browsing dataframe '{df_name}', created with pHs={arg_ph}, Us={arg_us}, decomposition_threshold={arg_decomposition_threshold}, filters={arg_filter}, sort={arg_sort}.\n"

    # Ensure that the provided start/end indices to not exceed 50 (as speficied in the docstring/annotations):
    if endIdx - startIdx > 50:
        return "Read no more than 50 rows at a time. Adjust startIdx and endIdx."
    
    # Get df, apply filters, sort and read length:
    df = CANVAS.read(df_name)
    df = df_query(df, filters, sort)
    total = len(df)

    # "cut-out" the requested portion of the dataframe, and convert to string for display:
    result = df.iloc[startIdx:endIdx].to_string(index=True)

    # Add a footer to indicate the range of rows being shown and the total number of rows after filtering/sorting:
    footer = f"\nShowing rows {startIdx}–{min(endIdx, total)-1} of {total} total."
    if total > endIdx:
        footer += f" Call again with startIdx={endIdx} to see more rows."
        
    outStr = header + result + footer
    
    id = CANVAS.register_tool_output(
        tool_name="browse_df",
        args={
            "df_name": df_name,
            "startIdx": startIdx,
            "endIdx": endIdx,
            "filters": filters,
            "sort": sort,
        },
        value=outStr,
        description=f"Output string of the requested portion of the dataframe after applying filters and sort, with a footer indicating the range of rows shown and total rows.",
        reasons=reasons,
        parent_result_ids=[df_name_ref],
        metadata={},
    )

    return f"{outStr}\nMessage_ID: {id}. Please refer to this ID if you want to refer back to this message later or use the displayed information for further analysis or decision making."

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

_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "mean": lambda *x: sum(x) / len(x),
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.UAdd, ast.USub, ast.Tuple, ast.List,
)


def _safe_eval(expr: str, variables: dict[str, float]) -> float:
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {ast.dump(node.func)}")

        if isinstance(node, ast.Name):
            if node.id not in variables and node.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Unknown variable: {node.id}")

    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {**_ALLOWED_FUNCS, **variables}))

def _merge_context(context: str, reasons: Any) -> Dict[str, str]:
    """Prepend the per-call `context` onto each per-parameter rationale.

    Accepts `reasons` in either of two shapes:
      * `Dict[str, str]` — multi-key rationale, one entry per tool parameter.
      * `str`            — single rationale used by tools with one logical input.

    In both cases, the return value is a dict so that every artifact registered
    via `register_tool_output(reasons=...)` has a uniform shape on disk. For
    str-shape inputs, the merged value is stored under the key `"reasons"`,
    matching the pre-existing convention used by tools like
    `math_expression_tool` and `extract_numeric_from_tool_output`.

    Empty / whitespace-only `context` is rejected at the tool boundary so that
    a missing study description fails loudly instead of silently degrading the
    verifier's signal.
    """
    if not context or not context.strip():
        raise ValueError(
            "context is required and must be non-empty. Describe in one "
            "sentence which study or exploration this tool call is part of "
            "(e.g. 'convergence test for ecutwfc', 'production run for "
            "adsorption energy', 'sensitivity sweep over n_fixed_layers', "
            "'one-off check')."
        )
    prefix = f"Context: {context.strip()}"
    if isinstance(reasons, dict):
        return {k: f"{prefix}\n\nRationale: {v}" for k, v in reasons.items()}
    return {"reasons": f"{prefix}\n\nRationale: {reasons}"}

@tool
def math_expression_tool(
    values_w_ref: Annotated[
        List[Tuple[float, str]],
        "List of (value, ref_result_id) pairs. They will be mapped in order to "
        "x0, x1, x2, ... When you obtain a value from a tool, you will be "
        "given the ref_result_id; place the (value, ref_result_id) pair here. "
        "Each ref_result_id accepts an 8-char id to reference the output, or "
        "`<8-char-id>.<param_name>` to reference an input parameter of a "
        "past tool call (see `list_referenceable_inputs`).",
    ],
    expression: Annotated[
        str,
        "Math expression using x0, x1, x2, ... Example: '(x0 - x1) / x2' or 'sqrt(x0**2 + x1**2)'",
    ],
    context: Annotated[
        str,
        "1-2 sentence describing which study or exploration this tool call "
        "is part of (e.g. 'convergence test for ecutwfc', 'production run "
        "for adsorption energy', 'sensitivity sweep over n_fixed_layers', "
        "'one-off check'), and the reason why you call this tool."
    ],
    reasons: Annotated[
        str,
        "Per-parameter rationale. Write 2-3 sentences covering: "
        "(a) THE ROLE this calculation plays in the study described in "
        "`context` (e.g. 'computing an intermediate value used downstream', "
        "'final reported quantity', 'sanity check'); "
        "(b) WHY THIS SPECIFIC EXPRESSION AND VALUES: how the inputs were "
        "chosen, what evidence supports them, and the expected meaning of "
        "the output. Since this tool has only one logical parameter (the "
        "expression+values pair), provide one combined rationale rather "
        "than per-key entries.",
    ],
) -> str:
    """
    Evaluate a math expression on arbitrary input floats.
    Inputs are mapped by order to x0, x1, x2, ...
    reference_id in the output ties with: computed math result, float
    """
    if not values_w_ref:
        return "No values were provided."

    merged_reasons = _merge_context(context, reasons)

    # Verify each value against its ref.
    for v, ref in values_w_ref:
        art = CANVAS.get_artifact(ref)
        if art is None:
            return f"Error: Reference ID {ref} not found in CANVAS."

    bare_values = [float(v) for v, _ in values_w_ref]
    bare_refs = [ref for _, ref in values_w_ref]

    variables = {f"x{i}": v for i, v in enumerate(bare_values)}
    mapping = {f"x{i}": ref for i, ref in enumerate(bare_refs)}

    try:
        result = _safe_eval(expression, variables)
        result_id = CANVAS.register_tool_output(
            tool_name="math_expression_tool",
            args={
                "values": bare_values,
                "expression": expression,
            },
            value=result,
            description=f"Result of evaluating expression '{expression}' with mapping {mapping}",
            reasons=merged_reasons,
            parent_result_ids=list(set(bare_refs)),
            metadata={"mapping": mapping},
        )
        return f"Result: {result}. Result_ID={result_id}."
    except Exception as e:
        return f"Failed to evaluate expression: {e}"

@tool
def inspect_my_canvas():
    """Inspect the working canvas to get available keys"""
    # get all keys in myCANVAS and return them as a list [key1, key2, ...]
    # print(CANVAS)
    _ = CANVAS.register_tool_output(
        tool_name="inspect_my_canvas",
        args={},
        value="",
        description=f"Inspecting the working canvas to get available keys.",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.inspect()

@tool
def read_my_canvas(key: Annotated[str, "key"]):
    """Read a value from the working canvas"""
    # read a value from myCANVAS given a key
    _ = CANVAS.register_tool_output(
        tool_name="read_my_canvas",
        args={
            "key": key,
        },
        value="",
        description=f"Reading value from canvas with key '{key}'",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.read(key)

@tool
def write_my_canvas(key: Annotated[str, "key"],
                    value: Annotated[Any, "value"],
                    overwrite: Annotated[bool, "True to overwrite if key already exist. only set to True if you are certain you want to overwrite the existing value"] = False):
    """Write a value to the working canvas. If the key already exists, it will not overwrite unless specified."""
    # write a value to myCANVAS given a key and a value
    
    _ = CANVAS.register_tool_output(
        tool_name="write_my_canvas",
        args={
            "key": key,
            "value": value,
            "overwrite": overwrite,
        },
        value=value,
        description=f"Writing value to canvas with key '{key}'",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.write(key, value, overwrite)

@tool
def write_report(
    report: Annotated[str, "Intermediate/final report content in markdown format."],
    report_name: Annotated[str, "Name of the report."],
    ):
    """Note down your report on CANVAS and let the supervisor know you've generated a report"""
    outStr = CANVAS.write(report_name, report, False)
    if outStr == f"Key '{report_name}' already exists. Please choose a different key. If you want to overwrite the value, set the 'overwrite' flag to True.":
        outStr = f"Report '{report_name}' already exists. Please choose a different name for the report. You should never overwirte a report."
    else:
        var.reportName = report_name
        
    id = CANVAS.register_tool_output(
        tool_name="write_report",
        args={
            "report": report,
            "report_name": report_name,
        },
        value=report,
        description=f"Writing report to canvas with key '{report_name}'",
        parent_result_ids=[],
        metadata={},
    )
    
    return outStr + f"\nReport_ID: {id}. Please refer to this ID if you want to reference this report later or use the information in the report for further analysis or decision making."

# @tool
# def write_my_canvas(key: Annotated[str, "key"],
#                     value: Annotated[Any, "value"],
#                     entry_type: Annotated[Literal["note", "numerical_result"], "entry type. 'note' is for general note or text, not allowed to be use to generate final report. 'numerical_result' is verifiable with tools output and will be verified. Must provide source_result_id if entry_type is 'numerical_result'."],
#                     overwrite: Annotated[bool, "True to overwrite if key already exist. only set to True if you are certain you want to overwrite the existing value"] = False,
#                     source_result_id: Annotated[Optional[str], "the result_id of the tool output that this numerical canvas entry is based on."] = None
#                     ):
#     """Write a value to the working canvas. If the key already exists, it will not overwrite unless specified."""
#     # write a value to myCANVAS given a key and a value
#     if entry_type == "numerical_result":
#         assert source_result_id is not None, "source_result_id must be provided for numerical_result entry type."
#     return CANVAS.write(
#         key=key,
#         value=value,
#         entry_type=entry_type,
#         overwrite=overwrite,
#         source_result_id=source_result_id
#         )
    
# @tool
# def register_parameter_choice_by_LLM_agent(
#     value: Annotated[float, "The value of the parameter chosen by the LLM agent. It should be a numeric value that can be used for calculations."],
#     reason: Annotated[str, "A brief reason of why a certain parameter was chosen to be this value. You must clarify the unit of the value."]
# ):
#     """Register the choice of a specific parameter by the LLM agent, along with the reason for the choice with unit clarified. """

    
#     result_id = CANVAS.register_tool_output(
#         tool_name="register_parameter_choice_by_LLM_agent",
#         value=value,
#         numerical_result=True,
#         parent_result_ids=[],
#         metadata={
#             "reason": reason,
#         },
#     )
    
#     return f"value: {value} is registered with result_id: {result_id}."


@tool
def extract_numeric_from_tool_output(
    source_tool_call_id: Annotated[str, "The ID of the text result from a prior tool call that you want to extract the numeric value from."],
    value: Annotated[float, "The numeric value you want to verify and extract from the tool output."],
    evidence_snippet: Annotated[str, "An substring from the tool output that contains the number. To avoid ambiguate when the same number may appear multiple times." ],
    description: Annotated[str, "A brief description of what this number represents. You must clarify the unit of the number in this description, e.g. 'the adsorption energy in eV', 'the length of the cell in Angstrom', etc."],
):
    """
    Verify that a numeric value was explicitly present in the raw output of a prior tool call,
    then register it as a trusted numeric artifact and return a result_id.

    Args:
        source_tool_call_id: The tool_call_id of the previously executed text-returning tool.
        value: The numeric value the agent wants to extract.
        evidence_snippet: An substring from the tool output that contains the number. 
                          To avoid ambiguate when the same number may appear multiple times.
        description: A brief description of what this number represents. You must clarify the unit of the number in this description. 
                     e.g. 'the adsorption energy in eV', 'the length of the cell in Angstrom', etc.

    Returns:
        str: A message indicating the result of the extraction and verification process.
    """
    abs_tol = 1e-8,
    record = CANVAS.get_artifact(source_tool_call_id)
    if record is None:
        return (
            f"EXTRACTION_FAILED: source_tool_call_id='{source_tool_call_id}' "
            "was not found in the tool output registry. Please check the canvas and try again, or regenerate the source result with corresponding tool"
        )
        
    raw_text = ""
    if isinstance(record, ListedArtifact):
        for arti in record.value:
            raw_text += repr(arti.value) + "\n"
    else:
        raw_text = record.value
    
    # For demo only
    if str(value).strip() not in raw_text:
        if str(int(value)).strip() not in raw_text:
            return (
                f"EXTRACTION_FAILED: neither {str(value).strip()!r} nor {str(int(value)).strip()} was not found in the recorded tool output: {raw_text!r} "
                f"tool_call_id='{source_tool_call_id}'"
            )
    
    result_id = CANVAS.register_tool_output(
        tool_name="extract_numeric_from_tool_output",
        args={
            "source_tool_call_id": source_tool_call_id,
            "value": value,
            "evidence_snippet": evidence_snippet,
            "description": description,
        },
        value=value,
        description=description,
        parent_result_ids=[source_tool_call_id],
        metadata={}
    )

    return f"value: {value} is now registered with id {result_id} and can be used for further calculations."
    

    # Search space
    snippet_spans = util_find_all_substring_spans(raw_text, evidence_snippet)
    if not snippet_spans:
        return (
            "EXTRACTION_FAILED: evidence_snippet was not found in the recorded tool output. "
            f"tool_call_id='{source_tool_call_id}'"
        )

    candidate_matches = []
    for s0, s1 in snippet_spans:
        candidate_matches.extend(
            util_numeric_matches_in_region(
                text=raw_text,
                region_start=s0,
                region_end=s1,
                target_value=float(value),
                abs_tol=abs_tol,
            )
        )


    if not candidate_matches:
        return (
            "EXTRACTION_FAILED: The claimed numeric value was not found in the recorded tool output "
            f"for tool_call_id='{source_tool_call_id}'."
        )

    # Disambiguation
    # if occurrence_index is not None:
    #     if occurrence_index < 0 or occurrence_index >= len(candidate_matches):
    #         return (
    #             "EXTRACTION_FAILED: occurrence_index is out of range. "
    #             f"Found {len(candidate_matches)} matching occurrence(s), got occurrence_index={occurrence_index}."
    #         )
    #     chosen = candidate_matches[occurrence_index]
    # else:
    if len(candidate_matches) > 1:
        preview = [
            {
                "token": m["token"],
                "char_span": [m["start"], m["end"]],
            }
            for m in candidate_matches[:10]
        ]
        return (
            "EXTRACTION_FAILED: Multiple matching numeric occurrences were found. "
            "Please refine your evidence_snippet disambiguate."
            f"Candidates: {preview}"
        )
    chosen = candidate_matches[0]

    matched_text = raw_text[chosen["start"]:chosen["end"]]

    result_id = CANVAS.register_tool_output(
        tool_name="extract_numeric_from_tool_output",
        args={
            "source_tool_call_id": source_tool_call_id,
            "value": value,
            "evidence_snippet": evidence_snippet,
            "description": description,
        },
        value=value,
        description=description,
        parent_result_ids=[source_tool_call_id],
        metadata={
            "matched_text": matched_text,
            "matched_span": [chosen["start"], chosen["end"]],
        },
    )

    return f"value: {value} is now registered with id {result_id} and can be used for further calculations."

    
@tool
def extract_numeric_from_tool_output_NOTDEMO(
    source_tool_call_id: Annotated[str, "The ID of the text result from a prior tool call that you want to extract the numeric value from."],
    value: Annotated[float, "The numeric value you want to verify and extract from the tool output."],
    evidence_snippet: Annotated[str, "An substring from the tool output that contains the number. To avoid ambiguate when the same number may appear multiple times." ],
    description: Annotated[str, "A brief description of what this number represents. You must clarify the unit of the number in this description, e.g. 'the adsorption energy in eV', 'the length of the cell in Angstrom', etc."],
):
    """
    Verify that a numeric value was explicitly present in the raw output of a prior tool call,
    then register it as a trusted numeric artifact and return a result_id.

    Args:
        source_tool_call_id: The tool_call_id of the previously executed text-returning tool.
        value: The numeric value the agent wants to extract.
        evidence_snippet: An substring from the tool output that contains the number. 
                          To avoid ambiguate when the same number may appear multiple times.
        description: A brief description of what this number represents. You must clarify the unit of the number in this description. 
                     e.g. 'the adsorption energy in eV', 'the length of the cell in Angstrom', etc.

    Returns:
        str: A message indicating the result of the extraction and verification process.
    """
    abs_tol = 1e-8,
    record = CANVAS.get_artifact(source_tool_call_id)
    if record is None:
        return (
            f"EXTRACTION_FAILED: source_tool_call_id='{source_tool_call_id}' "
            "was not found in the tool output registry. Please check the canvas and try again, or regenerate the source result with corresponding tool"
        )
                
    raw_text = ""
    if isinstance(record, ListedArtifact):
        for arti in record.value:
            raw_text += repr(arti.value) + "\n"
    else:
        raw_text = record.value

    # Search space
    snippet_spans = util_find_all_substring_spans(raw_text, evidence_snippet)
    if not snippet_spans:
        return (
            "EXTRACTION_FAILED: evidence_snippet was not found in the recorded tool output. "
            f"tool_call_id='{source_tool_call_id}'"
        )

    candidate_matches = []
    for s0, s1 in snippet_spans:
        candidate_matches.extend(
            util_numeric_matches_in_region(
                text=raw_text,
                region_start=s0,
                region_end=s1,
                target_value=float(value),
                abs_tol=abs_tol,
            )
        )


    if not candidate_matches:
        return (
            "EXTRACTION_FAILED: The claimed numeric value was not found in the recorded tool output "
            f"for tool_call_id='{source_tool_call_id}'."
        )

    # Disambiguation
    # if occurrence_index is not None:
    #     if occurrence_index < 0 or occurrence_index >= len(candidate_matches):
    #         return (
    #             "EXTRACTION_FAILED: occurrence_index is out of range. "
    #             f"Found {len(candidate_matches)} matching occurrence(s), got occurrence_index={occurrence_index}."
    #         )
    #     chosen = candidate_matches[occurrence_index]
    # else:
    if len(candidate_matches) > 1:
        preview = [
            {
                "token": m["token"],
                "char_span": [m["start"], m["end"]],
            }
            for m in candidate_matches[:10]
        ]
        return (
            "EXTRACTION_FAILED: Multiple matching numeric occurrences were found. "
            "Please refine your evidence_snippet disambiguate."
            f"Candidates: {preview}"
        )
    chosen = candidate_matches[0]

    matched_text = raw_text[chosen["start"]:chosen["end"]]

    result_id = CANVAS.register_tool_output(
        tool_name="extract_numeric_from_tool_output",
        args={
            "source_tool_call_id": source_tool_call_id,
            "value": value,
            "evidence_snippet": evidence_snippet,
            "description": description,
        },
        value=value,
        description=description,
        parent_result_ids=[source_tool_call_id],
        metadata={
            "matched_text": matched_text,
            "matched_span": [chosen["start"], chosen["end"]],
        },
    )

    return f"value: {value} is now registered with id {result_id} and can be used for further calculations."
##################################################################################################
##                                          DFT tools                                           ##
##################################################################################################

# @tool
# def get_my_WORKING_DIRECTORY() -> str:
#     """Get the working directory."""
#     return var.my_WORKING_DIRECTORY


def _to_serializable(x: Any) -> Any:
    """Convert common ASE / NumPy objects into JSON-serializable Python types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def _safe_call(fn, default_name: str) -> Any:
    """Call a getter and return a readable unavailable message on failure."""
    try:
        return fn()
    except Exception as e:
        return f"<unavailable: {default_name}: {e}>"


@tool
def inspect_ase_atoms(
    atomsFilename: Annotated[str, "Path to the ASE Atoms object file (e.g. .traj, .xyz) or the name of the job that contains the Atoms object (e.g. xxxx.pwi, xxx.pwi.pwo)."]
    ) -> str:
    """
    Broad inspection tool for an ASE Atoms object.

    Returns a wide, agent-friendly summary of the structure, including:
    - composition
    - geometry
    - cell / PBC
    - arrays / info / constraints
    - calculator-backed results when available

    Parameters
    ----------
    atoms
        ASE Atoms object.

    Returns
    -------
    dict
        JSON-serializable dictionary containing extracted information.
    """
    try:
        from ase import Atoms
    except Exception as e:
        return {"ok": False, "error": f"Failed to import ASE: {e}"}
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    try:
        atoms = read(os.path.join(WORKING_DIRECTORY, atomsFilename))
    except Exception as e:
        return f"Failed to read Atoms object from {atomsFilename}: {e}"

    if not isinstance(atoms, Atoms):
        return {
            "ok": False,
            "error": f"Expected ase.Atoms, got {type(atoms).__name__}",
        }

    result: Dict[str, Any] = {
        "ok": True,
        "natoms": len(atoms),
        "formula": _safe_call(atoms.get_chemical_formula, "formula"),
        "chemical_symbols": _safe_call(atoms.get_chemical_symbols, "chemical_symbols"),
        "atomic_numbers": _to_serializable(
            _safe_call(atoms.get_atomic_numbers, "atomic_numbers")
        ),
        "positions": _to_serializable(_safe_call(atoms.get_positions, "positions")),
        "scaled_positions": _to_serializable(
            _safe_call(atoms.get_scaled_positions, "scaled_positions")
        ),
        "cell": _to_serializable(atoms.cell.array),
        "cell_lengths_and_angles": _to_serializable(
            _safe_call(atoms.cell.cellpar, "cell_lengths_and_angles")
        ),
        "pbc": _to_serializable(atoms.get_pbc()),
        "volume": _to_serializable(_safe_call(atoms.get_volume, "volume")),
        "masses": _to_serializable(_safe_call(atoms.get_masses, "masses")),
        "center_of_mass": _to_serializable(
            _safe_call(atoms.get_center_of_mass, "center_of_mass")
        ),
        "momenta": _to_serializable(_safe_call(atoms.get_momenta, "momenta")),
        "velocities": _to_serializable(_safe_call(atoms.get_velocities, "velocities")),
        "tags": _to_serializable(_safe_call(atoms.get_tags, "tags")),
        "initial_charges": _to_serializable(
            _safe_call(atoms.get_initial_charges, "initial_charges")
        ),
        "initial_magnetic_moments": _to_serializable(
            _safe_call(atoms.get_initial_magnetic_moments, "initial_magnetic_moments")
        ),
        "info": _to_serializable(dict(atoms.info)),
        "arrays": {k: _to_serializable(v) for k, v in atoms.arrays.items()},
        "constraints": [repr(c) for c in atoms.constraints],
        "has_calculator": atoms.calc is not None,
        "calculator": type(atoms.calc).__name__ if atoms.calc is not None else None,
    }

    if atoms.calc is not None:
        result["calculator_results"] = {
            "potential_energy": _to_serializable(
                _safe_call(atoms.get_potential_energy, "potential_energy")
            ),
            "forces": _to_serializable(_safe_call(atoms.get_forces, "forces")),
            "stress": _to_serializable(_safe_call(atoms.get_stress, "stress")),
            "charges": _to_serializable(_safe_call(atoms.get_charges, "charges")),
            "magnetic_moments": _to_serializable(
                _safe_call(atoms.get_magnetic_moments, "magnetic_moments")
            ),
        }
        
    result = json.dumps(result, indent=2)
    id = CANVAS.register_tool_output(
        tool_name="inspect_ase_atoms",
        args={
            "atomsFilename": atomsFilename,
        },
        value=result,
        description=f"Inspection result of ASE Atoms object from {atomsFilename}",
        parent_result_ids=[],
        metadata={},
    )

    return f"{result}\n\nThe above result is registered as an entire string with id={id}. Please extract and register specific information you need."


@tool
def get_ase_atoms_property(
    atomsFilename: Annotated[str, "Path to the ASE Atoms object file (e.g. .traj, .xyz) or the name of the job that contains the Atoms object (e.g. xxxx.pwi, xxx.pwi.pwo)."],
    property_name: str
    ) -> str:
    """
    Extract one specific property from an ASE Atoms object.

    Supported property_name values include:
    - formula
    - natoms
    - chemical_symbols
    - atomic_numbers
    - positions
    - scaled_positions
    - cell
    - cell_lengths_and_angles
    - pbc
    - volume
    - masses
    - center_of_mass
    - momenta
    - velocities
    - tags
    - initial_charges
    - initial_magnetic_moments
    - info
    - arrays
    - constraints
    - has_calculator
    - calculator
    - potential_energy
    - forces
    - stress
    - charges
    - magnetic_moments

    Parameters
    ----------
    atoms
        ASE Atoms object.
    property_name : str
        Name of the property to extract.

    Returns
    -------
    dict
        JSON-serializable dictionary with the requested property.
    """
    try:
        from ase import Atoms
    except Exception as e:
        return {"ok": False, "error": f"Failed to import ASE: {e}"}
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    try:
        atoms = read(os.path.join(WORKING_DIRECTORY, atomsFilename))
    except Exception as e:
        return f"Failed to read Atoms object from {atomsFilename}: {e}"

    if not isinstance(atoms, Atoms):
        return {
            "ok": False,
            "error": f"Expected ase.Atoms, got {type(atoms).__name__}",
        }

    key = property_name.strip().lower()

    property_map = {
        "formula": lambda: atoms.get_chemical_formula(),
        "natoms": lambda: len(atoms),
        "chemical_symbols": atoms.get_chemical_symbols,
        "atomic_numbers": atoms.get_atomic_numbers,
        "positions": atoms.get_positions,
        "scaled_positions": atoms.get_scaled_positions,
        "cell": lambda: atoms.cell.array,
        "cell_lengths_and_angles": atoms.cell.cellpar,
        "pbc": atoms.get_pbc,
        "volume": atoms.get_volume,
        "masses": atoms.get_masses,
        "center_of_mass": atoms.get_center_of_mass,
        "momenta": atoms.get_momenta,
        "velocities": atoms.get_velocities,
        "tags": atoms.get_tags,
        "initial_charges": atoms.get_initial_charges,
        "initial_magnetic_moments": atoms.get_initial_magnetic_moments,
        "info": lambda: dict(atoms.info),
        "arrays": lambda: {k: _to_serializable(v) for k, v in atoms.arrays.items()},
        "constraints": lambda: [repr(c) for c in atoms.constraints],
        "has_calculator": lambda: atoms.calc is not None,
        "calculator": lambda: type(atoms.calc).__name__ if atoms.calc is not None else None,
        "potential_energy": atoms.get_potential_energy,
        "forces": atoms.get_forces,
        "stress": atoms.get_stress,
        "charges": atoms.get_charges,
        "magnetic_moments": atoms.get_magnetic_moments,
    }

    if key not in property_map:
        return {
            "ok": False,
            "error": f"Unsupported property_name: {property_name}",
            "supported_properties": sorted(property_map.keys()),
        }

    value = _safe_call(property_map[key], key)
    
    result = json.dumps({
        "ok": True,
        "property_name": key,
        "value": _to_serializable(value),
    }, indent=2)
    
    id = CANVAS.register_tool_output(
        tool_name="get_ase_atoms_property",
        args={
            "atomsFilename": atomsFilename,
            "property_name": property_name,
        },
        value=value,
        description=f"{key} property extracted from ASE Atoms object from {atomsFilename}",
        parent_result_ids=[],
        metadata={},
    )
    
    return f"Extracted result {result}.\nThe above result is registered as an entire string with id={id}. Please extract and register specific information you need."


@tool
def init_structure_data(
    element: Annotated[str, "Element symbol"],
    lattice: Annotated[str, "Lattice type. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
    a: Annotated[float, "Lattice constant"],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'element', 'lattice', 'a', 'b', 'c'."],
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
    result_id = CANVAS.register_tool_output(
        tool_name="init_structure_data",
        args={
            "element": element,
            "lattice": lattice,
            "a": a,
            "b": b,
            "c": c,
        },
        value=f"{element}-{lattice}.xyz",
        description="Path of the saved initial structure data file.",
        reasons=reasons,
        parent_result_ids=[],
        metadata={},
        # include modification check, to ensure validity of the actualy file content
    )
    
    # time.sleep(60)
    return f"Created atoms saved in the working directory with name '{element}-{lattice}.xyz' Directory info registered with ID={result_id}"

@tool
def generateSurface_and_getPossibleSite(species: Annotated[str, "Element symbol"],
                                        crystal_structures: Annotated[str, "Crystal structure. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite."],
                                        # a_dict: Annotated[Dict[str, float], "Dictionary of lattice parameters for the crystal structure: Dict[species, lattice_parameter_a]. i.e. {'Pt': 4.0}"],
                                        facets: Annotated[str, "Facet of the surface. Must be one of 100, 110, 111, 210, 211, 310, 311, 320, 321, 410, 411, 420, 421, 510, 511, 520, 521, 530, 531, 540, 541, 610, 611, 620, 621, 630, 631, 640, 641, 650, 651, 660, 661"],
                                        supercell_dim_xy: Annotated[List[int], "Supercell dimension, how many times do you want to repeat the primitive cell in XY direction: [int, int]"],
                                        supercell_dim_z:Annotated[int, "typically 6. Supercell dimension, how many times do you want to repeat the primitive cell in Z direction."],
                                        n_fixed_layers: Annotated[int, "typically 3. Number of fixed layers in the slab"],
                                        vacuum: Annotated[float, "typically 10.0. Vacuum size in Angstrom"],
                                        surfaceFilename: Annotated[str, "Name (not a path) of the surface file to be saved in traj format"],
                                        reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'supercell_dim_xy', 'supercell_dim_z', 'n_fixed_layers', 'vacuum'."],
                                        supercell_dim_z_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of supercell_dim_z. If not provided, the result will not be registered and you can't use the result in the final report"] = "",
                                        n_fixed_layers_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of n_fixed_layers. If not provided, the result will not be registered and you can't use the result in the final report"] = "",
                                        vacuum_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of vacuum size. If not provided, the result will not be registered and you can't use the result in the final report"] = "",
                                        ):
    """Generate a surface structure and get the available adsorption sites. 
    You can try out different supercell_dim_z, n_fixed_layers and vacuum size to see the effect.
    However, only when you specify the source_result_id reference for these parameters, the result will be registered in the canvas and you can use them in the production run.
    Otherwise, the tool will still execute and return the generated surface structure and available adsorption sites, but they will not be registered and you can't use them in the production run"""
    
    # verfiy all *_ref:
    for value, ref in zip(
        [supercell_dim_z, n_fixed_layers, vacuum],
        [supercell_dim_z_ref, n_fixed_layers_ref, vacuum_ref]
    ):
        if ref != "":
            ok, msg = CANVAS.verify_artifact(value, ref)
            if not ok:
                return msg
    
    a_dict = {'Pt': 3.92}
    supercell_dim = [supercell_dim_xy[0], supercell_dim_xy[1], supercell_dim_z]
    surface_dict = generate_surface_structures(
        species_list=[species],
        crystal_structures={species: crystal_structures},
        a_dict=a_dict,
        facets={species: [facets]},
        supercell_dim=supercell_dim,
        vacuum=vacuum,
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
        mySites[site] = (np.asarray(mySites[site]) @ tmpAtom.cell.array[:2])[:2]
    
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        print(mySites)
        
    mySites_copy = copy.deepcopy(mySites)
    
    mySites_str = output_capture.getvalue()
    
    if supercell_dim_z_ref != "" and n_fixed_layers_ref != "" and vacuum_ref != "":
        parent_result_ids = [supercell_dim_z_ref, n_fixed_layers_ref, vacuum_ref]
    else:
        parent_result_ids = []
    
    ids = {}
    for k, v in mySites.items():
        ids[k] = CANVAS.register_tool_output(
            tool_name="generateSurface_and_getPossibleSite",
            args={
                "species": species,
                "crystal_structures": crystal_structures,
                "facets": facets,
                "supercell_dim_xy": supercell_dim_xy,
                "supercell_dim_z": supercell_dim_z,
                "n_fixed_layers": n_fixed_layers,
                "vacuum": vacuum,
                "surfaceFilename": surfaceFilename,
            },
            value=v,
            description=f"Adsorption {k} site",
            reasons=reasons,
            parent_result_ids=parent_result_ids,
            metadata={}
        )
        mySites[k] = [v, f"ID={ids[k]}"]
    
    CANVAS.write('Possible_CO_site_on_Pt_surface', mySites)
    
    absPath = surface_dict[species][f'{crystal_structures}{facets}']['traj_file_path']
    # trim the absPath, remove the part before out, including out
    relaPath = absPath.split(f'{DirOfInterests}/')[-1]
    # time.sleep(60)
    
    os.makedirs(os.path.join(WORKING_DIRECTORY, "surface"), exist_ok=True)
    write(os.path.join(WORKING_DIRECTORY, "surface", surfaceFilename), mySurface)
    path_id = CANVAS.register_tool_output(
        tool_name="generateSurface_and_getPossibleSite",
        args={
            "species": species,
            "crystal_structures": crystal_structures,
            "facets": facets,
            "supercell_dim_xy": supercell_dim_xy,
            "supercell_dim_z": supercell_dim_z,
            "n_fixed_layers": n_fixed_layers,
            "vacuum": vacuum,
            "surfaceFilename": surfaceFilename,
        },
        value=f"surface{surfaceFilename}",
        description="Path of the saved surface structure file in traj format.",
        reasons=reasons,
        parent_result_ids=parent_result_ids,
        metadata={}
    )
    if supercell_dim_z_ref != "" and n_fixed_layers_ref != "" and vacuum_ref != "":
        return f"the surface generated is saved at surface/{surfaceFilename}, Path_ID={path_id}\navailable adsorbate sites are: {repr(mySites)}"
    
    return f"the surface generated is saved at surface/{surfaceFilename}\navailable adsorbate sites are: {repr(mySites_copy)}"

@tool
def generate_myAdsorbate(symbols: Annotated[str, "Element symbols of the adsorbate (Do not use any delimiters)"],
                         positions: Annotated[List[List[float]], "Positions of the atoms in the adsorbate, e.g. [[x1, y1, z1], [x2, y2, z2], ...], following the same order as the symbols."],
                         AdsorbateFileName: Annotated[str, "Name (not a path) of the adsorbate file to be saved in traj format"],
                         vaccum: Annotated[float, "Vacuum size in Angstrom around the adsorbate structure. Typically 10.0 Angstrom should be sufficient"],
                         reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'symbols', 'positions', 'vaccum'."],
                         ):
    """Generate an adsorbate structure and save it."""
    assert AdsorbateFileName.endswith('.traj'), "AdsorbateFileName should end with .traj"
    assert not '/' in AdsorbateFileName, "AdsorbateFileName should not contain '/'"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    os.makedirs(os.path.join(WORKING_DIRECTORY, "adsorbates"), exist_ok=True)
    tmpAtoms = Atoms(symbols=symbols, positions=positions)
    tmpAtoms.center(vacuum=vaccum)
    write(os.path.join(WORKING_DIRECTORY, "adsorbates", f"{AdsorbateFileName}"), tmpAtoms)
    id = CANVAS.register_tool_output(
        tool_name="generate_myAdsorbate",
        args={
            "symbols": symbols,
            "positions": positions,
            "AdsorbateFileName": AdsorbateFileName,
            "vaccum": vaccum,
        },
        value=f"adsorbates/{AdsorbateFileName}",
        description="Path of the saved adsorbate structure file in traj format.",
        reasons=reasons,
        parent_result_ids=[],
        metadata={}
    )
    return f"Adsorbate saved under working directory at adsorbates/{AdsorbateFileName}. Path_ID={id}"

@tool
def add_myAdsorbate(mySurfacePath: Annotated[str, "Path to the surface structure"],
                    adsorbatePath: Annotated[str, "Path to the adsorbate structure"],
                    mySites: Annotated[List[List[float]], "List of adsorption sites you want to put adsorbates on, e.g. [[x1, y1], [x2, y2], ...]"],
                    rotations: Annotated[List[Tuple[float, str]], "List of rotations for the ith adsorbates, e.g. [[90.0, 'x'], [180.0, 'y'], ...]"],
                    surfaceWithAdsorbateFileName: Annotated[str, "Name (not a path) of the surface adsorbated with adsorbate to be saved in traj format"],
                    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'mySurfacePath', 'adsorbatePath', 'mySites', 'rotations'."],
                    mySites_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of adsorption sites. If not provided, the result will not be registered and you can't use the result in final report"] = "",
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
    
    
    
    for value, ref in zip(
        [mySites],
        [mySites_ref]
    ):
        if ref != "":
            ok, msg = CANVAS.verify_artifact(value, ref)
            if not ok:
                return msg
    
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
    
    if mySites_ref != "":
        parent_result_ids = [mySites_ref]
    else:
        parent_result_ids = []
    
    outStr = f"Surface with adsorbate saved at {relaPath}."
    id = CANVAS.register_tool_output(
        tool_name="add_myAdsorbate",
        args={
            "mySurfacePath": mySurfacePath,
            "adsorbatePath": adsorbatePath,
            "mySites": mySites,
            "rotations": rotations,
            "surfaceWithAdsorbateFileName": surfaceWithAdsorbateFileName,
        },
        value=relaPath,
        description="Path of the saved surface with adsorbate structure file in traj format.",
        reasons=reasons,
        parent_result_ids=parent_result_ids,
        metadata={}
    )
    if mySites_ref != "":
        outStr += f" Path_ID={id}"
    
    return outStr

# register regardless. only return ID when refs are provided

@tool
def write_QE_script_w_ASE(
    listofElements: Annotated[List[str], "List of distinct element symbols in the unit cell"],
    ppfiles_w_ref: Annotated[List[Tuple[str, str]], "List of pseudopotential files in the order of the elements together with the reference source_result_id for each pp file. e.g. [('Pt.pbe-n-rrkjus_psl.1.0.0.UPF', 'ref_id_1'), ('C.pbe-n-rrkjus_psl.1.0.0.UPF', 'ref_id_2')]"],
    filename: Annotated[str, "Name of the Quantum Espresso input file, end with .pwi"],
    inputAtomsDir_w_ref: Annotated[Tuple[str, str], "Directory of the input Atoms object (i.e. traj or xyz), or the name of the job that contains the relaxed structure (i.e. xxxx.pwi), together with the reference source_result_id of the structure."],
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
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'calculation', 'restart_mode', 'prefix', 'disk_io', 'ibrav', 'nat', 'ntyp', 'ecutwfc', 'ecutrho', 'occupations', 'smearing', 'degauss', 'conv_thr', 'electron_maxstep', 'kspacing', 'input_dft', 'ready_to_run_job', 'additional_input'."],
    ready_to_run_job: Annotated[bool, "True if the job is intended to be run directly without further modification, False if this file is intended to be used to generate other files"] = False,
    additional_input: Annotated[Dict[str, Any], "Additional input parameters to be added to the input script. Should be in the format of a flat dict, {'input_parameter_1': parameter_1, 'input_parameter_2': parameter_2, ...}, parameter_x remain in their native type, str, float, bool, etc. Do not use unless you know what you are doing."] = {},
    ecutwfc_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of ecutwfc. If not provided, the result will not be registered and you can't use the result in final report"] = "",
    kspacing_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of kspacing. If not provided, the result will not be registered and you can't use the result in final report"] = "",
):
    """Write a Quantum Espresso input script using ASE. Bool value have no quote around them. For smearing start with methfessel-paxton. For ecutwfc choose between 30-100 Ry. When asked to run ensemble calculation, set calculation to 'ensemble'. When generating template for convergence test, use scf calculation and set ready_to_run_job to False."""

    assert isinstance(additional_input, dict), "additional_input must be a dictionary"
    
    if ensembleCalculation:
        assert calculation == 'ensemble', "When running ensemble calculation, please set calculation to 'ensemble'"
    
    inputAtomsDir, inputAtomsDir_ref = inputAtomsDir_w_ref
    
    if calculation == 'ensemble':
        assert inputAtomsDir.endswith('.pwi'), "inputAtomsDir must be a .pwi file with relaxed structure when running ensemble calculation with BEEF-vdW functional"
        assert input_dft == 'BEEF-vdW', "input_dft must be 'BEEF-vdW' when running ensemble calculation"
    
    disk_io = 'none'
    
    # verify refs
    for value, ref in zip(
        [ecutwfc, kspacing],
        [ecutwfc_ref, kspacing_ref]
    ):
        if ref != "":
            ok, msg = CANVAS.verify_artifact(value, ref)
            if not ok:
                return msg
    
    for pseudo, ref in ppfiles_w_ref:
        ok, msg = CANVAS.verify_artifact(pseudo, ref)
        if not ok:
            return msg
        
    ok, msg = CANVAS.verify_artifact(inputAtomsDir, inputAtomsDir_ref)
    if not ok:
        return msg
    
    # assemble the pseudopotentials dict from the list of elements and pseudopotentials
    pseudopotentials = {}
    ppfiles = []
    ppfilesID = []
    for pseudo, ref in ppfiles_w_ref:
        ppfiles.append(pseudo)
        ppfilesID.append(ref)
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
    CANVAS.write(destiJobList, job_list, overwrite=True)
    
    outStr = f"Quantum Espresso input script is written to {filename}"

    if ecutwfc_ref != "" and kspacing_ref != "":
        parent_result_ids = [inputAtomsDir_ref, ecutwfc_ref, kspacing_ref, *ppfilesID]
    else:   
        parent_result_ids = [inputAtomsDir_ref, *ppfilesID]
    id = CANVAS.register_tool_output(
        tool_name="write_QE_script_w_ASE",
        args={
            "listofElements": listofElements,
            "ppfiles": ppfiles,
            "filename": filename,
            "inputAtomsDir": tmpinputAtomsDir,
            "ensembleCalculation": ensembleCalculation,
            "calculation": calculation,
            "restart_mode": restart_mode,
            "prefix": prefix,
            "disk_io": disk_io,
            "ibrav": ibrav,
            "nat": nat,
            "ntyp": ntyp,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": occupations,
            "smearing": smearing,
            "degauss": degauss,
            "conv_thr": conv_thr,
            "electron_maxstep": electron_maxstep,
            "input_dft": input_dft,
        },
        value=filename,
        description="Path of the saved Quantum Espresso input script.",
        reasons=reasons,
        parent_result_ids=parent_result_ids,
        metadata={}
    )
    outStr += f" Filename_ID={id}"
    
    # if ecutwfc_ref == "" or kspacing_ref == "":
    #     pass
        
    # time.sleep(60)
    return outStr

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
            
            id = CANVAS.register_tool_output(
                tool_name="find_pseudopotential",
                args={
                    "element": element,
                },
                value=sp,
                description=f"Pseudopotential file for {element}",
                parent_result_ids=[],
                metadata={}
            )
            
            ans += f'{sp}  ID={id}\n'
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
                              input_file_name_ref: Annotated[str, "source_result_id identifing which tool output to reference for this choice of input_file_name."],
                              reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'input_file_name', 'kspacing', 'ecutwfc'."],
                              ):
    '''
    Generate the convergence test input scripts for quantum espresso calculation using another quantum espresso input file as a template and save the job list. 
    '''
    # kspacing = [0.6, 0.8, 1.0]
    # ecutwfc = [10, 20, 30]
    
    ok, msg = CANVAS.verify_artifact(input_file_name, input_file_name_ref)
    if not ok:
        return msg

    
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
    job_list_to_register = copy.deepcopy(job_list)
    ## Save the job list
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
    CANVAS.write('jobs_K_and_ecut',job_list_dict)
    
    id = CANVAS.register_tool_output(
        tool_name="generate_convergence_test",
        args={
            "input_file_name": input_file_name,
            "kspacing": kspacing,
            "ecutwfc": ecutwfc,
        },
        value=job_list_to_register,
        listed_value=True,
        description="A dict containing the generated convergence test job list with their corresponding kspacing and ecutwfc values.",
        reasons=reasons,
        parent_result_ids=[input_file_name_ref],
        metadata={}
    )
    
    return f"Job list is saved scucessfully. ID={id}. Please tell the supervisor in your response that convergence job has generated sucessfully, please continue to submit the jobs"

@tool
def generate_eos_test(
    input_file_name: Annotated[str, "Name of the template quantum espresso input file"],
    kspacing: Annotated[float, "K-point spacing (in Angstrom^-1) for the equation of state test."],
    ecutwfc: Annotated[int, "Kinetic energy cutoff (Ry) for wavefunctions for the equation of state test."],
    stepSize: Annotated[float, "Step size for scaling the cell size. The cell will be scaled from (1-2*stepSize) to (1+2*stepSize). Typically 0.025 should be good."],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'input_file_name', 'kspacing', 'ecutwfc', 'stepSize'."],
    input_file_name_ref: Annotated[str, "Source_result_id identifing which tool output to reference for this choice of template quantum espresso input file."],
    kspacing_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of kspacing. If not provided, the result will not be registered and you can't use the result in final report"] = "",
    ecutwfc_ref: Annotated[str, "Optional source_result_id identifing which tool output to reference for this choice of ecutwfc. If not provided, the result will not be registered and you can't use the result in final report"] = "",
    ):
    '''
    Generate the equation of state test input scripts for quantum espresso calculation and save the job list.
    '''
    assert stepSize > 0.01 and stepSize < 0.1, "stepSize should be between 0.01 and 0.1"
    
    for value, ref in zip(
        [kspacing, ecutwfc],
        [kspacing_ref, ecutwfc_ref]
    ):
        if ref != "":
            ok, msg = CANVAS.verify_artifact(value, ref)
            if not ok:
                return msg
            
    ok, msg = CANVAS.verify_artifact(input_file_name, input_file_name_ref)
    if not ok:
        return msg
    
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
    job_list_to_register = copy.deepcopy(job_list)
    
    ## Save the job list as json file
    old_job_list = CANVAS.canvas.get('ready_to_run_job_list', []).copy()
    job_list = list(set(old_job_list + job_list))
    CANVAS.write('ready_to_run_job_list',job_list, overwrite=True)
    
    outStr = f"Job list is saved scucessfully, continue to submit the jobs. Files of interest are {job_list}."
    
    if kspacing_ref != "" and ecutwfc_ref != "":
        parent_result_ids = [kspacing_ref, ecutwfc_ref, input_file_name_ref]
    else:
        parent_result_ids = [input_file_name_ref]
        
    id = CANVAS.register_tool_output(
        tool_name="generate_eos_test",
        args={
            "input_file_name": input_file_name,
            "kspacing": kspacing,
            "ecutwfc": ecutwfc,
            "stepSize": stepSize,
        },
        value=job_list_to_register,
        listed_value=True,
        description="The generated EOS test job list.",
        reasons=reasons,
        parent_result_ids=parent_result_ids,
        metadata={}
    )
    if kspacing_ref != "" and ecutwfc_ref != "":
        outStr += f" Filename_ID={id}"
    
    # time.sleep(60)
    return outStr

###################################### DFT POST-PROCESSING TOOLS ######################################

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    def _slice_tokens(text: str, start_tok: int, end_tok: int) -> str:
        ids = _ENC.encode(text)
        return _ENC.decode(ids[start_tok:end_tok])
except Exception:
    _ENC = None
    # Heuristic: ~4 chars per token
    _CHARS_PER_TOKEN = 4
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN)
    def _slice_tokens(text: str, start_tok: int, end_tok: int) -> str:
        start_ch = start_tok * _CHARS_PER_TOKEN
        end_ch = end_tok * _CHARS_PER_TOKEN
        return text[start_ch:end_ch]

@tool
def get_convergence_suggestions(
    filename: Annotated[str, "Name of the Quantum Espresso input file that did not converge, end with .pwi"],
    question: Annotated[str, "Question about this job, e.g. 'Why this job did not converge?' or 'how to improve the accuracy of this job?'"],
    start_block: Annotated[int, "The block index to start with when the content is too long. Each block contains around 150k tokens. Set to 0 for the first block." ] = 0
):
    "Get suggestions on how to resolve issues for a certain job, i.e. converge or not accurate enough. If the output file is too long, you can call this tool multiple times for the same file with the same question but different start_block index to get suggestions based on different part of the output."
    outFile = filename + ".pwo"
    errFile = filename + ".err"
    block_size_tokens = 150000
    # WORKING_DIRECTORY = "/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/out"
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    # config = load_config(os.path.join('./config', "default.yaml"))
    config = var.OTHER_GLOBAL_VARIABLES
    # llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    workerllm = ChatAnthropic(model="claude-haiku-4-5", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # workerllm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    # llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # workerllm = AzureChatOpenAI(model="gpt-4o", api_version="2024-08-01-preview", api_key=config["OpenAI_API_KEY"], azure_endpoint = config["OpenAI_BASE_URL"])
    # llm = ChatDeepSeek(model_name=config["DeepSeek_MDL"], api_key=config['DeepSeek_API_KEY'], api_base=config['DeepSeek_BASE_URL'], temperature=0.0)
    
    fileTrimed = False
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
            total_tokens = _count_tokens(task_formatted)

            # Small enough: return all
            if total_tokens <= block_size_tokens:
                system_msg = """
    You are a DFT expert who's good at giving concise suggestions on how to resolve issues in DFT calculations. Do not modify nosym and pesudopotentials. Never make any adjustment to make the calculation less accurate.
    Please use the format: parameterX: suggestionX, reasonX; parameterY: suggestionY, reasonY; ...
    You must include target values for the parameters you suggest to change, e.g. if you suggest to increase the ecutwfc, you should give a specific value for the new ecutwfc, not just say "increase ecutwfc".
    """
                
                invokingMsg = [
                    ("system", system_msg),
                    ("user", task_formatted)
                ]
                agent_response = workerllm.invoke(invokingMsg)
                
                finalSuggestion += agent_response.content + "\n\n"
                print(agent_response + "\n\n")
                if var.my_SAVE_DIALOGUE:
                    with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                        f.write(repr(agent_response))
            else:
                fileTrimed = True
                # Large: compute block boundaries
                block_size = block_size_tokens
                total_blocks = (total_tokens + block_size - 1) // block_size
                sb = max(0, int(start_block or 0))
                if sb >= total_blocks:
                    sb = total_blocks - 1  # clamp to last block

                start_tok = sb * block_size
                end_tok = min((sb + 1) * block_size, total_tokens)
                chunk = _slice_tokens(content, start_tok, end_tok)
                task_formatted = f"{chunk}\n I have a question about the DFT calculation related to the file above: {question}. Please think about what could be the reason, and give me suggestions to address it. Never give suggestion to lower the accuracy of the calculation, such as loosen the convergence threshold."
                system_msg = """
    You are a DFT expert who's good at giving concise suggestions on how to resolve issues in DFT calculations based on part of the output files. Do not modify nosym and pesudopotentials. Never make any adjustment to make the calculation less accurate.
    Please use the format: parameterX: suggestionX, reasonX; parameterY: suggestionY, reasonY; ...
    """
                
                invokingMsg = [
                    ("system", system_msg),
                    ("user", task_formatted)
                ]
                agent_response = workerllm.invoke(invokingMsg)
                
                finalSuggestion += agent_response.content + "\n\n"
                print(agent_response + "\n\n")
                if var.my_SAVE_DIALOGUE:
                    with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                        f.write(repr(agent_response))

            
    if finalSuggestion == "":
        # time.sleep(60)
        return f"Job {filename} has no related files, please check the job list and make sure the job is finished."
    
    finalSuggestion += "Please check the suggestions above and come up with a plan to fix the issue. Never take suggestions that will lower the accuracy of the calculation.\n"
    if fileTrimed:
        finalSuggestion += f"Note: The suggestions are based on {start_block}th part of the output file, if you want to get more comprehensive suggestions, please call this tool multiple times with different start_block index to cover different part of the output file.\n"
    # time.sleep(60)
    
    id = CANVAS.register_tool_output(
        tool_name="get_convergence_suggestions",
        args={
            "filename": filename,
            "question": question,
            "start_block": start_block,
        },
        value=finalSuggestion,
        description=f"Suggestions for the question: {question} based on the output of {filename}",
        parent_result_ids=[],
        metadata={}
    )
    
    return f"{finalSuggestion}\nSuggestion_ID={id}. Please extract the numerical values if you need to use them for adjusting the input parameters for the next calculation."

@tool
def find_optimal_parameter(
    sweeping_parameter: Annotated[str, "Name of the sweeping parameter, e.g. 'ecutwfc', 'kspacing', and etc."],
    Filename_n_parameters_w_ref: Annotated[List[Tuple[str, float, str]], "List of (filename, parameter_value, filename_ref_id) pairs. filename is the name of the output file corresponding to the parameter value, filename_ref_id is the source_result_id of the file that you want to reference for this file."],
    reference_file: Annotated[str, "Among the list of files, the reference_file filename corresponding to the most expensive / most accurate reference calculation."],
    threshold: Annotated[float, "Maximum allowed absolute energy difference from the reference energy."],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'threshold'."],
) -> Dict[str, Any]:
    """
    Find the most optimal parameter value for production run.
    """
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    Filename_n_parameters = []
    Filename_n_parameters_ref = []
    for filename, param_value, filename_ref_id in Filename_n_parameters_w_ref:
        ok, msg = CANVAS.verify_artifact(filename, filename_ref_id)
        if not ok:
            raise ValueError(f"Verification failed for file {filename} with reference ID {filename_ref_id}: {msg}")
        Filename_n_parameters.append((filename, param_value))
        Filename_n_parameters_ref.append(filename_ref_id)
    
    # add .pwo to all file names if haven't been added
    for i in range(len(Filename_n_parameters)):
        filename, param_value = Filename_n_parameters[i]
        if not filename.endswith('.pwo'):
            Filename_n_parameters[i] = (filename + '.pwo', param_value)
    if not reference_file.endswith('.pwo'):
        reference_file += '.pwo'
    
    file_to_param = dict(Filename_n_parameters)

    if reference_file not in file_to_param:
        raise ValueError(
            f"reference_file '{reference_file}' is not present in Filename_n_parameters."
        )

    # os.path.join(WORKING_DIRECTORY, myfile)
    reference_param = file_to_param[reference_file]
    reference_energy = read(os.path.join(WORKING_DIRECTORY, reference_file)).get_potential_energy()

    acceptable = []

    for filename, param_value in Filename_n_parameters:
        energy = read(os.path.join(WORKING_DIRECTORY, filename)).get_potential_energy()
        if abs(energy - reference_energy) <= threshold:
            acceptable.append((filename, param_value))

    if len(acceptable) == 1:
        return "Only the reference file is within threshold. No acceptable cheaper setting found. Please consider increasing the calculation settings to increase the accuracy of the calculation."

    chosen = max(acceptable, key=lambda x: abs(x[1] - reference_param))
    
    id = CANVAS.register_tool_output(
        tool_name="find_optimal_parameter",
        args={
            "sweeping_parameter": sweeping_parameter,
            "Filename_n_parameters": Filename_n_parameters,
            "reference_file": reference_file,
            "threshold": threshold,
        },
        value=chosen[1],
        description=f"The most optimal parameter value for production run based on the reference file {reference_file} and the threshold {threshold}. The chosen parameter value is {chosen[1]} with file name {chosen[0]}",
        reasons=reasons,
        parent_result_ids=[*set(Filename_n_parameters_ref)],
        metadata={}
    )
        
    
    return f"Please choose {sweeping_parameter}={chosen[1]}. result_ID={id}."

@tool
def calculate_formation_E(slabFilePath: Annotated[str, "the slab calculation file name, ending in pwi"],
                          adsorbateFilePath: Annotated[str, "the adsorbate calculation file name, ending in pwi"],
                          systemFilePath: Annotated[str, "the slab with adsorbate calculation file name, ending in pwi"],
                          reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'slabFilePath', 'adsorbateFilePath', 'systemFilePath'."],
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
    
    id = CANVAS.register_tool_output(
        tool_name="calculate_formation_E",
        args={
            "slabFilePath": slabFilePath,
            "adsorbateFilePath": adsorbateFilePath,
            "systemFilePath": systemFilePath,
        },
        value=formationEnergy,
        description=f"The formation energy of the adsorbate on the slab calculated using {slabFilePath}, {adsorbateFilePath}, and {systemFilePath}",
        reasons=reasons,
        parent_result_ids=[],
        metadata={}
    )
    
    # time.sleep(60)
    return f"The formation energy of the adsorbate on the slab is {formationEnergy} eV. Energy_ID={id}."

@tool
def calculate_lc(
    jobFileIdx_w_ref: Annotated[List[Tuple[int, str]], "indexs of files in the finished job list of files of interest, which will be used to calculate the lattice constant, together with the reference_id for each filename."],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'jobFileIdx' (why do you choose those job)."],
    ) -> str:
    """Read the output file and calculate the lattice constant"""
    
    
    assert isinstance(jobFileIdx_w_ref, list), "jobFileIdx_w_ref should be a list"
    for i, ref in jobFileIdx_w_ref:
        assert isinstance(i, int), "jobFileIdx_w_ref should be a list of (index of files of interest, reference_id) pairs"
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    job_list = CANVAS.canvas.get('finished_job_list', []).copy()
    
    jobFileIdx = []
    jobFileIdx_ref = []
    for idx, ref in jobFileIdx_w_ref:
        ok, msg = CANVAS.verify_artifact(job_list[idx] ,ref)
        if not ok:
            return f"Verification failed for job index {idx} with reference ID {ref}: {msg}"
        jobFileIdx.append(idx)
        jobFileIdx_ref.append(ref)
    
    jobFileIdx_ref = set(jobFileIdx_ref)
    
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
        
    id = CANVAS.register_tool_output(
        tool_name="calculate_lc",
        args={
            "jobFileIdx": jobFileIdx,
        },
        value=lc,
        description=f"The lattice constant calculated using the job list with index {jobFileIdx}",
        reasons=reasons,
        parent_result_ids=[*jobFileIdx_ref],
        metadata={}
    )

    # time.sleep(60)
    return f'The lattice constant is {lc}. LC_ID={id}'



# @tool
# def get_kspacing_ecutwfc(jobFileIdx: Annotated[List[int], "indexs of files in the finished job list of files of interest, which will be used to determine the kspacing and ecutwfc"],
#                          threshold: Annotated[float, "the threshold mev/atom to determine the convergence"] = 1.0) -> str:
#     '''Read the convergen test result and determine the kspacing and ecutwfc used in the production
#     Input:
#         jobFileIdx: list, the indexs of files in the finished job list, which will be used to determine the kspacing and ecutwfc
#         threshold: float , the threshold mev/atom to determine the convergence
#     output: str, the kspacing and ecutwfc used in the production
#     '''
#     WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
#     assert isinstance(jobFileIdx, list), "jobFileIdx should be a list"
#     for i in jobFileIdx:
#         assert isinstance(i, int), "jobFileIdx should be a list of index of files of interest"
    
#     job_dict = CANVAS.canvas.get('jobs_K_and_ecut', {})
#     job_list = CANVAS.canvas.get('finished_job_list', []).copy()
#     job_list = np.array(job_list, dtype=str)[jobFileIdx]
#     print(f"actual job list: {job_list}")
#     assert len(job_list) > 0, "job list 0"
    
#     print(f"successfully read {len(job_list)} jobs, and {len(job_dict)} job_dict")

#     ### Find the kpoints and ecutwfc from the output file
#     kspacing = []
#     ecutwfc = []
#     energy_list = []
#     goodJob = []
#     Natom = None
#     for job in job_list:
#         ## Read the output file
#         print(f'reading {job}')
#         try:
#             atom = read(os.path.join(WORKING_DIRECTORY, job+'.pwo'))
#         except:
#             print(f"Job {job} is not finished or failed.")
#             continue
#         energy = atom.get_potential_energy()
#         energy_list.append(energy)
#         Natom = atom.get_number_of_atoms()
        
#         kspacing.append(job_dict[job]['k'])
#         ecutwfc.append(job_dict[job]['ecutwfc'])
#         goodJob.append(job)
    
#     convergence_df = pd.DataFrame({'job':goodJob,'kspacing':kspacing, 'ecutwfc':ecutwfc, 'energy':energy_list})
    
#     min_kspacing = convergence_df['kspacing'].min()
#     max_ecutwfc = convergence_df['ecutwfc'].max()
#     df_kspacing = convergence_df.loc[convergence_df['kspacing'] == min_kspacing].sort_values(by='ecutwfc',ascending=True)
#     df_ecutwfc = convergence_df.loc[convergence_df['ecutwfc'] == max_ecutwfc].sort_values(by='kspacing',ascending=False)
    
#     print(f"successfully read {len(df_kspacing)} kspacing and {len(df_ecutwfc)} ecutwfc")
    
#     if len(df_kspacing) == 1 and len(df_ecutwfc) > 1:
#         # time.sleep(60)
#         return f"Only one kspacing is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
#     if len(df_ecutwfc) == 1 and len(df_kspacing) > 1:
#         # time.sleep(60)
#         return f"Only one ecutwfc is found, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
#     if len(df_kspacing) == 1 and len(df_ecutwfc) == 1:
#         # time.sleep(60)
#         return f"Only one job for either kspacing or ecutwfc is good, the rest of the jobs seems unfinished or not converged. DO NOT infer optimal parameters from converged jobs. Please regenerate the convergence test with finer kspacing and ecutwfc. Also, adjust some other settings may help (regenerating template script is then needed). Remember, you NEED TO REDO the convergence test (tell the supervisor in your response that new convergence test need to be done and you've already generated the script)."
        
#     ## Save the convergence test result if file exist then append to it
#     if os.path.exists(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv')):
#         convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'), mode='a', header=False)
#     else:
#         convergence_df.to_csv(os.path.join(WORKING_DIRECTORY, 'convergence_test.csv'))
    
#     ## Determine the kpoints and ecutwfc based on the threshold
#     k_chosen, ecutwfc_chosen,finnerEcut,df_kspacing, df_ecutwfc,finnerKspacing = select_k_ecut(convergence_df, threshold, Natom)
    
#     print(f"Chosen kspacing: {k_chosen}, Chosen ecutwfc: {ecutwfc_chosen}")
    
#     ## Save the chosen kspacing and ecutwfc
#     if os.path.exists(os.path.join(WORKING_DIRECTORY, 'df_k.csv')):
#         df_kspacing.to_csv(os.path.join(WORKING_DIRECTORY, 'df_k.csv'), mode='a', header=False)
#     else:
#         df_kspacing.to_csv(os.path.join(WORKING_DIRECTORY, 'df_k.csv'))
    
#     if os.path.exists(os.path.join(WORKING_DIRECTORY, 'df_e.csv')):
#         df_ecutwfc.to_csv(os.path.join(WORKING_DIRECTORY, 'df_e.csv'), mode='a', header=False)
#     else:
#         df_ecutwfc.to_csv(os.path.join(WORKING_DIRECTORY, 'df_e.csv'))  
        
#     print("saved the chosen kspacing and ecutwfc")
    
    
#     if finnerEcut and ecutwfc_chosen < 120 and finnerKspacing and k_chosen > 0.1:
#         ans = "Only the calculation with the finest settings is finished. Please regenerate the convergence test with finner ecutwfc and finner kspacing. Do not infer converged settings yourself!"
#         # ans += f"\nHowever, the calculation is not converged, please consider redo the convergence test and using a finner ecutwfc and finner kspacing"
#     elif finnerEcut and ecutwfc_chosen < 120:
#         ans = "Only calculations with the finest ecutwfc is finished. Please regenerate the convergence test with finner ecutwfc. Do not infer converged settings yourself!"
#     elif finnerKspacing and k_chosen > 0.1:
#         ans = "Only the calculation with the finest kspacing is finished. Please regenerate the convergence test with finner kspacing. Do not infer converged settings yourself!"
#     else:
#         ans = f"Please use kspacing {k_chosen} and ecutwfc {ecutwfc_chosen} for the production calculation"
#     # time.sleep(60)
#     return ans

# @tool
# def analyze_BEEF_result(
#     slabFilePath: Annotated[str, "the slab calculation file"],
#     adsorbateFilePath: Annotated[str, "the adsorbate calculation file"],
#     ontopFilePath: Annotated[str, "the slab with ontop adsorbate calculation file"],
#     fccFilePath: Annotated[str, "the slab with fcc adsorbate calculation file"],
# ) -> str:
#     '''Read the BEEF output, calculate the abrosption energy and analyze the BEEF result'''
    
#     WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
#     DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
#     PathList = [slabFilePath, adsorbateFilePath, ontopFilePath, fccFilePath]
    
#     for i in range(len(PathList)):
#         tmp = PathList[i]
#         try:
#             if not PathList[i].startswith(DirOfInterests) and not PathList[i].startswith(f'./{DirOfInterests}') and not PathList[i].startswith('/nfs'):
#                 PathList[i] = os.path.join(WORKING_DIRECTORY, PathList[i]) + '.pwo'
#             _ = read(PathList[i])
#         except:
#             if os.path.exists(PathList[i]):
#                 return f"{tmp} did not finish successfully."
#             return f"Invalid input atoms directory: {tmp}. make sure to supply either absolute path, or relative path starting with './{DirOfInterests}'. Please check the path in canvas and try again."

    
#     ## Read energy
#     slab_e = read_BEEF_output(PathList[0])
#     if slab_e == "WrongCalc":
#         return f"Please run slab ensemble calculation using BEEF-vdW with relaxed slab structure! Do not proceed any further!"
#     adsorbate_e = read_BEEF_output(PathList[1])
#     if adsorbate_e == "WrongCalc":
#         return f"Please run adsorbate ensemble calculation using BEEF-vdW with relaxed adsorbate structure! Do not proceed any further!"
#     ontop_e = read_BEEF_output(PathList[2])
#     if ontop_e == "WrongCalc":
#         return f"Please run ontop ensemble calculation using BEEF-vdW with relaxed slab and adsorbate structure! Do not proceed any further!"
#     fcc_e = read_BEEF_output(PathList[3])
#     if fcc_e == "WrongCalc":
#         return f"Please run fcc ensemble calculation using BEEF-vdW with relaxed slab and adsorbate structure! Do not proceed any further!"
    
#     ## Plot
#     try:
#         energy_dict = {}
#         energy_dict['clean'] = slab_e
#         energy_dict['CO'] = adsorbate_e
#         energy_dict['ontop_down'] = ontop_e
#         energy_dict['fcc_down'] = fcc_e
        
#         fontsize=15
#         plot_settings = {
#             # "font.family": "times new roman",
#             "axes.labelsize": fontsize,
#             "axes.labelweight": "bold",
#             "xtick.labelsize": fontsize,
#             "ytick.labelsize": fontsize,
#             "xtick.major.size": 7,
#             "ytick.major.size": 7,
#             "xtick.major.width": 2.0,
#             "ytick.major.width": 2.0,
#             "xtick.direction": "in",
#             "ytick.direction": "in",
#             "font.size": fontsize,
#             "axes.linewidth": 2.0,
#             "lines.dashed_pattern": [5, 2.5],
#             "lines.markersize": 10,
#             "lines.linewidth": 2,
#             "lines.markeredgewidth": 1,
#             # "lines.markeredgecolor": "k",
#             "legend.fontsize": fontsize,
#             "legend.frameon": False,
#             'figure.figsize': [6, 6],
#         }

#         # Update rcParams with settings from JSON file
#         rcParams.update(plot_settings)
        
#         df = pd.DataFrame(energy_dict)
#         df.to_csv('energies.csv', index=False)
#         ads_fcc = df['fcc_down'] - df['clean'] - df['CO']
#         ads_ontop = df['ontop_down'] - df['clean'] - df['CO']
#         # plot energy distribution
#         fig = plt.figure(figsize=(6, 5))
#         ax = fig.add_axes([0,0,1,1])
#         ax.hist(ads_fcc, bins=50, color='blue', alpha=0.7, label='FCC')
#         ax.axvline(ads_fcc.mean(), color='k', linestyle=':', linewidth=2, label='FCC mean')
#         ax.hist(ads_ontop, bins=50, color='red', alpha=0.7, label='Ontop')
#         ax.axvline(ads_ontop.mean(), color='k', linestyle='-.', linewidth=2, label='Ontop mean')
#         plt.legend()
#         plt.xlabel('Adsorption Energy (eV)')
#         plt.ylabel('Frequency')
#         plt.savefig('energy_distribution.png',dpi=300,bbox_inches='tight')
#     except:
#         print("Failed to plot the energy distribution.")

#     ## Formation
#     ontop_formation = ontop_e - slab_e - adsorbate_e
#     fcc_formation = fcc_e - slab_e - adsorbate_e

#     print(f"ontop formation energy: {ontop_formation.mean()} eV")
#     print(f"fcc formation energy: {fcc_formation.mean()} eV")
#     ## Formation Energy Difference
#     formation_energy_diff = ontop_formation - fcc_formation

#     ## Distribution of Formation energy differernce 
#     if formation_energy_diff.all() > 0:
#         result = f"fcc is more stable than ontop by average {formation_energy_diff.mean()} eV"
#     elif formation_energy_diff.all() < 0:
#         result = f"ontop is more stable than fcc by average {abs(formation_energy_diff.mean())} eV"
#     else:
#         result = f" {sum(formation_energy_diff>0)} xc functionals prefer fcc, {sum(formation_energy_diff<0)} xc functionals prefer ontop"
#     return result


@tool
def analyze_BEEF_result(
    slabFilePath: Annotated[str, "the slab calculation file"],
    adsorbateFilePath: Annotated[str, "the adsorbate calculation file"],
    systemFilePath: Annotated[str, "the slab with ontop adsorbate calculation file"],
    reasons: Annotated[Dict[str, str], "reason behind each parameter choice. For each parameter explain why do you make such choice? proof? what potential effect choosing such parameter has on the output? any hypothesis are you testing (it's okay to say no)? how did you obtained the value? The keys should be: 'slabFilePath', 'adsorbateFilePath', 'systemFilePath'. (why do you choose those three files)"],
) -> str:
    '''Read, extract, and analyze BEEF calculation results for slab, adsorbate, and surface with adsorbate. Return the mean and standard deviation of the adsorption energy.'''
    
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    
    DirOfInterests = WORKING_DIRECTORY.split('/')[-1]
    
    PathList = [slabFilePath, adsorbateFilePath, systemFilePath]
    
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
    system_e = read_BEEF_output(PathList[2])
    if system_e == "WrongCalc":
        return f"Please run surface with adsorbate ensemble calculation using BEEF-vdW with relaxed slab and adsorbate structure! Do not proceed any further!"
    
    ## Plot
    try:
        energy_dict = {}
        energy_dict['clean'] = slab_e
        energy_dict['CO'] = adsorbate_e
        energy_dict['system'] = system_e
        
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
        ads_E = df['system'] - df['clean'] - df['CO']
        # plot energy distribution
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_axes([0,0,1,1])
        ax.hist(ads_E, bins=50, color='blue', alpha=0.7, label='E distribution')
        ax.axvline(ads_E.mean(), color='k', linestyle=':', linewidth=2, label='E mean')
        # ax.hist(ads_ontop, bins=50, color='red', alpha=0.7, label='Ontop')
        # ax.axvline(ads_ontop.mean(), color='k', linestyle='-.', linewidth=2, label='Ontop mean')
        # plt.legend()
        plt.xlabel('Adsorption Energy (eV)')
        plt.ylabel('Frequency')
        plt.savefig('energy_distribution.png',dpi=300,bbox_inches='tight')
    except:
        print("Failed to plot the energy distribution.")

    ## Formation
    E_formation = system_e - slab_e - adsorbate_e
    
    E_formation_mean = E_formation.mean()
    E_formation_std = E_formation.std()
    
    mean_id = CANVAS.register_tool_output(
        tool_name="analyze_BEEF_result",
        args={
            "slabFilePath": slabFilePath,
            "adsorbateFilePath": adsorbateFilePath,
            "systemFilePath": systemFilePath,
        },
        value=E_formation_mean,
        description=f"The mean adsorption energy calculated using {slabFilePath}, {adsorbateFilePath}, and {systemFilePath}",
        reasons=reasons,
        parent_result_ids=[],
        metadata={}
    )
    
    std_id = CANVAS.register_tool_output(
        tool_name="analyze_BEEF_result",
        args={
            "slabFilePath": slabFilePath,
            "adsorbateFilePath": adsorbateFilePath,
            "systemFilePath": systemFilePath,
        },
        value=E_formation_std,
        description=f"The standard deviation of adsorption energy calculated using {slabFilePath}, {adsorbateFilePath}, and {systemFilePath}",
        reasons=reasons,
        parent_result_ids=[],
        metadata={}
    )

    return f"The mean adsorption energy is {E_formation_mean} eV, and the standard deviation is {E_formation_std} eV. Mean_ID={mean_id}, Std_ID={std_id}."

##################################################################################################
##                                          HPC tools                                           ##
##################################################################################################


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
    
    id = CANVAS.register_tool_output(
        tool_name="add_resource_suggestion",
        args={
            "qeInputFileName": qeInputFileName,
            "partition": partition,
            "nnodes": nnodes,
            "ntasks": ntasks,
            "span": span,
            "submissionScript": submissionScript,
            "outputFilename": outputFilename,
        },
        value=new_resource_dict,
        description=f"Resource suggestion for {qeInputFileName} with partition {partition}, nnodes {nnodes}, ntasks {ntasks}, runtime {span}, submission script {submissionScript}, and output filename {outputFilename}",
        reasons={},
        parent_result_ids=[],
        metadata={}
    )
    
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
    
    id = CANVAS.register_tool_output(
        tool_name="submit_and_monitor_job",
        args={
            "jobType": jobType,
        },
        value="places_holder",
        description=f"HPC places_holder",
        reasons={},
        parent_result_ids=[],
        metadata={}
    )
    
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
