import os
import yaml
import sys
with open(os.path.join('./config', "default.yaml"), 'r') as f:
    config = yaml.safe_load(f)
    pathList = config.get("PATH_LIST", [])
    sys.path.extend(p for p in pathList if p not in sys.path)
    
import getpass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
import pickle
# from langchain_anthropic import ChatAnthropic

# from src.prompt import hpc_agent_prompt,dft_agent_prompt
# from src.graph import create_graph
from src.planNexe2 import create_planning_graph
# from src.planNexeHighPlan import create_planning_graph as create_graph
import sys
import sys
import time
from src.utils import load_config, save_graph_to_file,initialize_database
from src.myCANVAS import CANVAS
from src import var

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from pathlib import Path

from gnome_dreams_oer_screening.explog.explog import EXPLOG

from src.disposition_reconcile import reconcile_dispositions
# _STABILITY_CACHE is already loaded by this point (src.planNexe2 -> src.tools
# above); this import is free, just a reference to the same module-level
# singleton -- not a second load of the AQ-GNoME database.
from src.tools import _STABILITY_CACHE
from src.aq_gnome_candidate_sync import sync_reduced_formula


# =====================================================================
# Checkpoint retention
# =====================================================================
THREAD_ID = "1"
MAX_CHECKPOINT_DB_BYTES = 20 * 1024**3   # prune once sqlite (+wal) exceeds 20 GB
KEEP_ROUNDS = 50                         # newest big-round checkpoints to keep


def _checkpoint_db_size(db_path):
    """Live size of the checkpoint db = main file + WAL + shm."""
    return sum(
        os.path.getsize(db_path + suffix)
        for suffix in ("", "-wal", "-shm")
        if os.path.exists(db_path + suffix)
    )


def prune_old_rounds(checkpointer, db_path, thread_id=THREAD_ID,
                     max_bytes=MAX_CHECKPOINT_DB_BYTES, keep_rounds=KEEP_ROUNDS):
    """If the checkpoint db exceeds `max_bytes`, keep only the newest
    `keep_rounds` big-round checkpoints and drop everything older.

    How it works:
    - Round-level checkpoints are the rows with checkpoint_ns = '' (the parent
      PlanExecute graph). The tool-level checkpoints written by the inner
      agents live in namespaced rows (e.g. "OER_Agent:<task_id>").
    - LangGraph checkpoint_ids are UUIDv6 -> lexicographically time-ordered,
      so one cutoff id cleanly separates old from new ACROSS all namespaces:
      deleting `checkpoint_id < cutoff` with NO namespace filter removes old
      rounds and their inner tool-level checkpoints in one sweep, while
      leaving the kept rounds' inner history (and the latest, resumable
      checkpoints) untouched.
    - The oldest surviving checkpoint ends up with a dangling
      parent_checkpoint_id; LangGraph tolerates this - get_state_history just
      ends there. Time travel to pruned rounds is gone permanently.

    Safe to call between super-steps (which is where the driver loop calls it).
    """
    if _checkpoint_db_size(db_path) < max_bytes:
        return

    conn = checkpointer.conn
    deleted = 0
    cutoff = None
    with checkpointer.lock:
        row = conn.execute(
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = '' "
            "ORDER BY checkpoint_id DESC LIMIT 1 OFFSET ?",
            (thread_id, keep_rounds - 1),
        ).fetchone()
        if row is None:          # fewer than keep_rounds rounds exist yet
            return
        cutoff = row[0]
        cur = conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id < ?",
            (thread_id, cutoff),
        )
        deleted += cur.rowcount
        cur = conn.execute(
            "DELETE FROM writes WHERE thread_id = ? AND checkpoint_id < ?",
            (thread_id, cutoff),
        )
        deleted += cur.rowcount
        conn.commit()

    if deleted:
        # DELETE alone never shrinks the file; VACUUM rewrites it. Needs up to
        # ~2x the file size in temp space and locks the db while running.
        try:
            with checkpointer.lock:
                conn.execute("VACUUM")
        except sqlite3.OperationalError as e:
            print(f"[prune] VACUUM skipped ({e}); space will be reused, file not shrunk")
        print(f"[prune] removed {deleted} checkpoint/write rows older than round cutoff {cutoff}; "
              f"db now {_checkpoint_db_size(db_path) / 1024**2:.1f} MB")


# --- Ensure the new GNoME_DREAMS_OER_screening package meets the minimum version requirement ------
# from importlib.metadata import version
# from packaging.version import Version

# if Version(version("GNoME_DREAMS_OER_screening")) < Version("0.5.4"):
#     raise RuntimeError("GNoME_DREAMS_OER_screening>=0.5.4 is required")
# --- End of version check -------------------------------------------------------------------------

if __name__ == "__main__":

    namelist = ['Li (bcc)', 'Na (bcc)', 'K (bcc)', 'Rb (bcc)', 'Ca (fcc)', 'Sr (fcc)', 'Ba (bcc)', 'V (bcc)', 'Nb (bcc)', 'Ta (bcc)', 'Mo (bcc)', 'W (bcc)',\
             'Fe (bcc)', 'Rh (fcc)', 'Ir (fcc)', 'Ni (fcc)'\
            , 'Pd (fcc)', 'Pt (fcc)', 'Cu (fcc)', 'Ag (fcc)', 'Au (fcc)', 'Al (fcc)', 'Pb (fcc)', 'C (dia)', 'Si (dia)', 'Ge (dia)', 'Sn (dia)']
    filelist = ['Li_bcc.in', 'Na_bcc.in', 'K_bcc.in', 'Si_dia.in', 'Pd_fcc.in', 'Ge_dia.in', 'Au_fcc.in', 'C_dia.in', 'Cu_fcc.in', 'Fe_bcc.in', 'Ca_fcc.in', 'Pb_fcc.in', 'W_bcc.in', 'Mo_bcc.in', 'Pt_fcc.in', 'Ag_fcc.in', 'Rh_fcc.in', 'Sr_fcc.in', 'Nb_bcc.in', 'Al_fcc.in', 'Rb_bcc.in', 'Ta_bcc.in', 'Ir_fcc.in', 'Sn_dia.in', 'Ba_bcc.in', 'V_bcc.in', 'Ni_fcc.in']
    
    structure = 'Li (bcc)'
    # structure = 'Li (bcc)'

    ## Convergence Test
    userMessage_1 = f'''
    You are going to do cenvergence test for {structure} structure, use initial lattice constant 3.451. Compute the the total energy for different kpoints based on kspacing 0.1,0.15,0.2,0.25 ,0.3 and 40,60,80,100,120 ecutwfc. Run the calculation through slurm and report the result.
    '''

    userMessage_2 = f'''
    Based on previous result, Choose appropriate kpoints and ecutwfc, generate input script with different scale factor and Submit the job through slurm
    '''
   

    userMessage_3 = f'''
    Now we have all the result, please calculate the lattice constant of {structure} structure and report the result.
    '''

    userMessage_4 = '''
    You are going to calculate the lattic constant for FCC Ca through DFT, the experiment value is 5.556. 
    1. Compute the the total energy for different kpoints based on kspacing 0.1,0.2 ,0.3 and 40,60,80,100,120 ecutwfc. Run the calculation through slurm and report the result.
    2. After the first batch calculation, choose appropriate kpoints and ecutwfc. Then generate input script for EOS and submit the job.
    3. When the calculation is done, calculate the lattice constant
    '''
    
    userMessage_5 = '''
    through DFT, please calculate the lattic constant for for following system listed in the following format: Lattice_structure Species (experimental_value)
    Li (bcc)	3.451
    Na (bcc)	4.209
    K (bcc)	5.212
    Rb (bcc)	5.577
    Ca (fcc)	5.556
    Sr (fcc)	6.04
    Ba (bcc)	5.002
    V (bcc)	3.024
    Nb (bcc)	3.294
    Ta (bcc)	3.299
    Mo (bcc)	3.141
    W (bcc)	3.16
    Fe (bcc)	2.853
    Rh (fcc)	3.793
    Ir (fcc)	3.831
    Ni (fcc)	3.508
    Pd (fcc)	3.876
    Pt (fcc)	3.913
    Cu (fcc)	3.596
    Ag (fcc)	4.062
    Au (fcc)	4.062
    Al (fcc)	4.019
    Pb (fcc)	4.912
    C (dia)	3.544
    Si (dia)	5.415
    Ge (dia)	5.639
    Sn (dia)	6.474
    '''
    
    userMessage_6 = "You are going to calculate the lattice constant for BCC Li through DFT."
    userMessage_7 = "You are going to generat a Pt surface structure with 2x2x4 supercell, then do a convergence test, use maximum ecutwfc = 160. Get the optimal kspacing and ecutwfc."
    userMessage_8 = """Please generate intial structures required to calculate CO adsorbtion on Pt(111) surface with 1/4 coverage (2x2x4 supercell), and calculate the adsorbtion energy."""
    userMessage_9 = """
    Please find out the most perfered adsorbtion site and adsorbate orientation (up or down) for CO adsorbtion on Pt(111) surface with 1/4 coverage (2x2x4 supercell).
    """
    userMessage_10 = """please find the adsorption energy difference between the most favorable configurations (different adsorbate orientations 0, 90, 180) at fcc site and
    most favorable configuration (different adsorbate orientations 0, 90, 180) at ontop site for CO on Pt(111) surface with p(2x2) adsorbate overlayer (1/4 coverage). 
    Please use PBE pseudopotential and PBE exchange correlation function.
    Literatures suggest that ontop site is 0.108 eV less stable than fcc site when using PBE xc. 
    If your result is not within 10 percent of the literature, please find out possible reasons and resolve it."""

    userMessage_11 = "I am trying to study adsorption of CO on Pt111 surface at fcc site. Job CO_Pt111_fcc_upright_k_0.3_ecutwfc_60.pwi did not converge, please figure out why and resolve the convergence issue."

    userMessage_12 = """please find the adsorption energy difference between the most favorable configurations (different adsorbate orientations 0, 90, 180) at fcc site and most favorable configuration (different adsorbate orientations 0, 90, 180) at ontop site for CO on Pt(111) surface with p(2x2) adsorbate overlayer (1/4 coverage), and analyze the uncertainty.
    Please use PBE pseudopotential and Bayesian Error Estimation Functional (BEEF) exchange correlation function.
    Literatures suggest that ontop site is 0.18 eV less stable than fcc site when using PBE xc.
    If your result is not within 10 percent of the literature, please find out possible reasons and resolve it."""

    userMessage_13 = """please conduct a initial screening on the default dataset on potential candidates as a catalyst for OER reaction. Please only consider O only and skip the study of OH and OOH for now. Please save the cadidates dataframe into a csv file."""

    userMessage_14 = """please conduct a OER screening study to find out the best system to use as catalyst for OER reaction. You must evaluate more less than 3 different systems. Please only consider O only and skip the study of OH and OOH for now. Available systems can be found in the default dataset. you must include Ir, with Nsite < 20. Please use VASP as the calculator. 
    """

    testMessage = """
    please conduct a acidic OER screening study to find out the best system to use as catalyst for OER reaction with limited computational budget.
    You must decide a screening strategy for selecting candidates materials and performing relavent DFT calculation to evaluate the OER activity.
    Available systems can be found in the default dataset. you must use Nsite < 20 and decomposition_energy < 0.5. Please use VASP as the calculator. 
    """

    testMessage2 = """
    Please conduct an acidic OER screening study to identify the best catalytic system for the oxygen evolution reaction 
    (OER) under a limited computational budget.
    You must design a screening strategy for selecting candidate materials and performing relevant DFT calculations to
    evaluate OER activity, with the available tools. We suggest an iterative approach: compute adsorption energies for 
    an initial set of candidates, then use these results to guide the selection of candidates for subsequent rounds. 
    This process can be repeated over several iterations until the most promising catalyst is identified.
    Available systems can be found in the default dataset. The backend DFT calculator employs a PBE+U level of theory,
    with U parameters taken from the Materials Project computational framework. Furthermore, the overall reaction
    2H₂O → 2H₂ + O₂ is assumed to have an energy cost of 4.92 eV. This implies ideal adsorption energies of: 
    G(OH) = 1.23 eV, G(O) = 2.46 eV, G(OOH) = 3.69 eV. An ideal overpotential will be calculated once G(O) and G(OH) 
    have been obtained via DFT, which assumes that the missing G(OOH) value minimizes the overpotential. Additionally,
     an overpotential based on the scaling relation G(OOH) = G(OH) + 3.2 will also be calculated.
    Zero-point energy (ZPE), entropy, and heat capacity (Cp) contributions are assumed to be constant and are taken from
    the literature beforhand, and applied in the backend.
    """
    
    testMessage3 = "please ask your worker agent to check the explog repetitivly for 6 times."

    revised_message = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution 
    reaction (OER) in the Google DeepMind GNoME database under a limited computational budget. Consider both catalytic 
    activity and cost/availability of candidates.

    Design a screening strategy to select candidate materials and perform relevant DFT calculations 
    and analysis using the available tools.

    You must:
    Perform multiple rounds of screening.
    Learn from each round and apply insights to new candidates, surfaces/terminations, or active sites.

    In your conclusion, state:
    What you learned.
    What worked and what did not.
    Which hypotheses were confirmed or rejected, and why.
    Provide a comparison of the best candidates with available literature.

    Leverage available data from the provided tools, including:
    AQ-GNoME: aqueous stability data and HHI metrics (supply availability).
    The ideal overpotential, which assumes the ideal adsorption energy of OOH given the calculated energies of O and 
    OH, and the overpotential from scaling relations calculated assuming G(OOH)=G(OH)+3.2.

    You have a maximum of 4 hours to complete the entire study.
    """
    
    revised_message_v3 = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution
    reaction (OER) in the Google DeepMind GNoME database. 
    You have a maximum of 24 hours to complete the entire study and make your final report.
    """
    
    # removed info below (all other info has been integrated into the system prompt):
    # Consider catalytic activity, cost/availability, and stability under operating conditions when evaluating candidates. -> repetitive in the system prompt
    # Choose an appropriate Pourbaix decomposition threshold for the filtering. -> it need to choose all filtering parameters appropriately, not just pourbaix decomposition threshold.
    # To make best use of your computational resources, you should consider your filtering criteria and relevant literature when making a candidate selection. -> I don't really see how this benifits/guide the agent. It already select candidates based on literature result
    # Use literature searches to inform your candidate selection and hypothesis formation. -> repetitive to requirement point 2
    # Prioritise O adsorption calculations broadly across many candidates. Use the resulting G(O) values to identify the most promising candidates and sites before submitting OH adsorption calculations. -> I added point 13. maybe we don't need this line anymore.

    # TODO - ajust the time limit
    revised_message_v4 = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution
    reaction (OER) in the Google DeepMind GNoME database. Please do a iterative multi-round screening, learning from each round and applying insights to new candidates, 
    surfaces/terminations, or active sites if and when possible.Please use literature searches to inform your per-round candidate selection and hypothesis formation, and note them down clearly.
    Prioritise O adsorption calculations broadly across many candidates. Use the resulting G(O) values to identify the most promising candidates and sites before proceeding with OH adsorption calculations.
    You have a maximum of 7 hours to complete the entire study and make your final report.
    
    The AQ-GNoME database is available, which enables filtering based on aqueous stability across pH and
    electrochemical potential (V vs. SHE). You need to make use of literature when choosing filtering criteria and making candidate selections.
    You should also consider catalytic activity, cost/availability, and stability under operating conditions when evaluating candidates.
    Toxicity of the constituent elements should also be considered where possible, though note that no toxicity data is
    available in the dataset, hence this assessment will be limited to qualitative reasoning based on literature.
    Where appropriate, revisit the AQ-GNoME database during the study using refined selection criteria based on emerging insights, to explore new candidates.
    
    Final report:
    At the end of the study, produce an extensive report structured as a mini scientific paper. Every conclusion and
    claim must be directly supported by concrete results from the study — cite specific candidates, sites, terminations,
    G(O), G(OH), and overpotential values explicitly. Be critical of your conclusions and assumptions: acknowledge
    limitations, uncertainties, and cases where the data is inconclusive. Do n8ot make claims that are not backed by
    data. The report should include:
    - A summary of the screening strategy and how it evolved.
    - The best candidates identified, with their G(O), G(OH), ideal overpotential, and scaling-relation overpotential.
    - A comparison of the best candidates with available literature.
    - What was learned, what worked, and what did not.
    - Which hypotheses were confirmed or rejected, with explicit reference to the supporting data.
    - Any trends worth noting across the dataset, even if these trends do not lead to competitive candidates.
    - Any recommendations for future studies or next steps based on the findings and limitations of the current study.
    """

    revised_message_v5 = """Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution reaction (OER) in the Google DeepMind GNoME database. Please do an iterative multi-round screening, learning from each round and applying insights to new candidates, surfaces/terminations, or active sites when sensible. Please use literature searches to inform your per-round candidate selection and hypothesis formation and note them down clearly. 
Prioritize O adsorption calculations broadly across many candidates Prioritize O adsorption calculations across many candidates, focusing on hypothesis-relevant unique sites instead of exhaustively evaluating all sites for each candidate (It may be relevant to consider many adsorption sites for a few candidates). Use the resulting G(O) values to identify the most promising sites before proceeding with OH adsorption calculations (possibly delaying OH calculations to later rounds). When evaluating overpotentials and ranking candidates, you will need to consider both the overpotential calculated assuming an idea OOH binding and the one calculated via the scaling relation.
The AQ-GNoME database is available, which enables filtering based on aqueous stability across pH and electrochemical potential (V vs. SHE). You need to make use of literature when choosing filtering criteria and making candidate selections. You should also consider catalytic activity, cost/availability, and stability under operating conditions when selecting and evaluating candidates. Toxicity of the constituent elements should also be considered, though note that no toxicity data is available in the dataset, hence this assessment will be limited to qualitative reasoning based on literature. Where appropriate, revisit the AQ-GNoME database during the study using refined selection criteria based on emerging insights, to explore new candidates.
To leverage the available HPC resources best, aim to have relevant DFT jobs pending/queued most of the time, and do not wait for all jobs to finish within one round before submitting new jobs. If all jobs are running, aim to submit more relevant jobs. If many jobs are pending (more than 50), hold off on submitting more to allow for flexibility later when you want to prioritize specific jobs. Consider that, when you are nearing the end of the study, you may not have time to wait for all jobs to finish. We recommend starting by submitting about 50 diverse candidates and then adjusting the number of jobs based on how many jobs are pending vs running. Note that surface and adsorption jobs will take much longer than bulk jobs.
Reporting:
At the end of EACH ROUND as well as the end of the study, produce an extensive report structured as a mini scientific paper. Every conclusion and claim must be directly supported by concrete results from the study — cite specific candidates, sites, terminations, G(O), G(OH), and overpotential values explicitly. Be critical of your conclusions and assumptions: acknowledge limitations, uncertainties, and cases where the data is inconclusive. Do not make claims that are not backed by data. The report should include:
    - A summary of the screening strategy and how it evolved.
    - The best candidates identified, with their G(O), G(OH), ideal overpotential, and scaling-relation overpotential.
    - A comparison of the best candidates with available literature.
    - What was learned, what worked, and what did not.
    - Which hypotheses were confirmed or rejected, with explicit reference to the supporting data.
    - Any trends worth noting across the dataset, even if these trends do not lead to competitive candidates.
    - Any recommendations for future studies or next steps based on the findings and limitations of the current study.
You have a maximum of 1 hours to complete the entire study and make your final report.
"""


    revised_message_temp = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution
    reaction (OER) in the Google DeepMind GNoME database. Please do a iterative multi-round screening, of at least 10 rounds, learning from each round and applying insights to new candidates, 
    surfaces/terminations, or active sites if and when possible.Please use literature searches to inform your per-round candidate selection and hypothesis formation, and note them down clearly.
    Prioritise O adsorption calculations broadly across many candidates. Use the resulting G(O) values to identify the most promising candidates and sites before proceeding with OH adsorption calculations.
    You have a maximum of 1 hour and 30 minutes to complete the entire study and make your final report.
    
    The AQ-GNoME database is available, which enables filtering based on aqueous stability across pH and
    electrochemical potential (V vs. SHE). You need to make use of literature when choosing filtering criteria and making candidate selections.
    You should also consider catalytic activity, cost/availability, and stability under operating conditions when evaluating candidates.
    Toxicity of the constituent elements should also be considered where possible, though note that no toxicity data is
    available in the dataset, hence this assessment will be limited to qualitative reasoning based on literature.
    
    Final report:
    At the end of the study, produce an extensive report structured as a mini scientific paper. Every conclusion and
    claim must be directly supported by concrete results from the study — cite specific candidates, sites, terminations,
    G(O), G(OH), and overpotential values explicitly. Be critical of your conclusions and assumptions: acknowledge
    limitations, uncertainties, and cases where the data is inconclusive. Do not make claims that are not backed by
    data. The report should include:
    - A summary of the screening strategy and how it evolved.
    - The best candidates identified, with their G(O), G(OH), ideal overpotential, and scaling-relation overpotential.
    - A comparison of the best candidates with available literature.
    - What was learned, what worked, and what did not.
    - Which hypotheses were confirmed or rejected, with explicit reference to the supporting data.
    - Any trends worth noting across the dataset, even if these trends do not lead to competitive candidates.
    - Any recommendations for future studies or next steps based on the findings and limitations of the current study.
    """

    revised_message_v2 = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate for the oxygen evolution
    reaction (OER) in the Google DeepMind GNoME database. 
    You have a maximum of XXXXXX hours/days to complete the entire study and make your final report.
    Adjust your plan as needed to make full use of the available time. Specifically, if the study is progressing
    faster than expected, extend the scope of the study; or if time is running short, narrow the focus to ensure the
    study is completed with overpotential computed for key candidates and a final report is produced.
    Consider catalytic activity, cost/availability, and stability under operating conditions when evaluating candidates.
    Toxicity of the constituent elements should also be considered where possible, though note that no toxicity data is
    available in the dataset, hence this assessment will be limited to qualitative reasoning based on literature.

    Dataset filtering:
    The AQ-GNoME database is available, which enables filtering based on aqueous stability across pH and
    electrochemical potential (V vs. SHE). As this is an acidic study, filter for materials that are stable under
    acidic conditions. Choose an appropriate Pourbaix decomposition threshold for the filtering.
    To make best use of your computational resources, you should consider your filtering criteria and
    relevant literature when making a candidate selection.

    Screening strategy:
    Use literature searches to inform your candidate selection and hypothesis formation. Record your hypotheses
    and major reasoning behind each high-level screening strategy on the CANVAS and detaild resoning with respect to 
    candidate, terminatino and site selection in the EXPLOG as you go. This will support your final report.
    Design a multi-round screening strategy, learning from each round and applying insights to new candidates,
    surfaces/terminations, or active sites if and when possible.
    Prioritise O adsorption calculations broadly across many candidates. Use the resulting G(O) values to identify the
    most promising candidates and sites before submitting OH adsorption calculations. Avoid submitting OH adsorption
    jobs for sites where G(O) is far from the ideal value of 2.46 eV, as such sites will not yield competitive
    overpotentials regardless of G(OH).

    HPC usage:
    Try to keep the HPC queue occupied with meaningful jobs, to make sure you make the most of the available
    resources. Do not wait for all jobs in one phase to complete before submitting new ones. Instead, submit new
    calculations opportunistically if you have no queued jobs. Periodically check the EXPLOG for completed results and
    use these to guide the next submissions, rather than sticking strictly to one phase at a time.

    Final report:
    At the end of the study, produce an extensive report structured as a mini scientific paper. Every conclusion and
    claim must be directly supported by concrete results from the study — cite specific candidates, sites, terminations,
    G(O), G(OH), and overpotential values explicitly. Be critical of your conclusions and assumptions: acknowledge
    limitations, uncertainties, and cases where the data is inconclusive. Do not make claims that are not backed by
    data. The report should include:
    - A summary of the screening strategy and how it evolved.
    - The best candidates identified, with their G(O), G(OH), ideal overpotential, and scaling-relation overpotential.
    - A comparison of the best candidates with available literature.
    - What was learned, what worked, and what did not.
    - Which hypotheses were confirmed or rejected, with explicit reference to the supporting data.
    - Any trends worth noting across the dataset, even if these trends do not lead to competitive candidates.
    - Any recommendations for future studies or next steps based on the findings and limitations of the current study.
    """

    minimal_test_message = """
    Please conduct a minimal OER screening study for testing purposes, considering only a limited number of candidates.
    Select 5 candiates which only contains atoms which are genneraly non-magnetic. Also, select 5 candidates which 
    contains elements which are genneraly magnetic, such as Gd, Eu, Fe, Mn or Co. You need to select a candidate with 
    at least 2 diffrent magnetic elements in the same candidate structure. Try to keep the computational cost low.
    Filter the database, enter candidates into the experiment log, run bulk relaxation, surface relaxation,
    and one to two O and OH adsorption calculation per candidate. If you experience many tool failures, end and report
    what issues you encountered. Try to make use of each tool, once for testing purposes. Report your findings.
    """
    #    You have a maximum of 25 minutes to complete your entire run and report your findings.
    
    minimal_test_message_2 = """
    Conduct 1 literature search about best OER catalyst. Then ask your worker to search the that result ID with the
    corresponding tool. Pretend that you forgot the id, ask your worker agent to search for that id using some other information that's associated with that ID.
    Verify that the search contains the correct ID. Then generate a report about the literature search result. After the literature search, if the next step in the plan
    is to ask your worker to search the the result ID, do not touch the plan, keep it as is.
    """
    
    minimal_test_message_3 = """
    do not do any literature search, just filter through the database and, in 3 round, enter 15 candidates into the experiment log, 5 per round.
    """

    revised_message_v6 = """
    Please conduct an acidic OER screening study to identify the best catalytic candidate
    for the oxygen evolution reaction (OER) in the Google DeepMind GNoME database.
    Please do an iterative multi-round screening, learning from each round and applying
    insights to new candidates, surfaces/terminations, or active sites when sensible.
    Please use literature searches to inform your per-round candidate selection and
    hypothesis formation and note them down clearly.

    Prioritize O adsorption calculations broadly across many candidates Prioritize O
    adsorption calculations across many candidates, focusing on hypothesis-relevant
    unique sites instead of exhaustively evaluating all sites for each candidate (It may be
    relevant to consider many adsorption sites for a few candidates). Use the resulting
    G(O) values to identify the most promising candidates and sites before proceeding
    with OH adsorption calculations (possibly delaying OH calculations to later rounds).
    When evaluating overpotentials and ranking candidates, you will need to consider
    both the overpotential calculated assuming an idea OOH binding and the one
    calculated via the scaling relation.

    The AQ-GNoME database is available for stability-based filtering. The Pourbaix
    stability screening is fixed at pH = 0 and U = 1.2-2.0 V vs. SHE (acidic OER
    operating conditions). Many additional filters are available (e.g. decomposition
    threshold, bandgap, HHI, disorder probability) and literature should inform the
    selection of these criteria and candidate choices. You
    should also consider catalytic activity, cost/availability, and stability under operating
    conditions when selecting and evaluating candidates. Toxicity of the constituent
    elements should also be considered where possible, though note that no toxicity data
    is available in the dataset, hence this assessment will be limited to qualitative
    reasoning based on literature. Where appropriate, revisit the AQ-GNoME database
    during the study using refined selection criteria based on emerging insights, to
    explore new candidates.

    To leverage the available HPC resources best, aim to have relevant DFT jobs
    pending/queued most of the time, and do not wait for all jobs to finish within one
    round before submitting new jobs. If all jobs are running, aim to submit more
    relevant jobs. If many jobs are pending (more than 50), hold off on submitting more to
    allow for flexibility later when you want to prioritize specific jobs. Consider that,
    when you are nearing the end of the study, you may not have time to wait for all jobs
    to finish. We recommend starting by submitting about 40 diverse candidates and then
    adjusting the number of jobs based on how many jobs are pending vs running. Note
    that surface and adsorption jobs will take much longer than bulk jobs.

    Report:
    At the end of EACH ROUND as well as the end of the study, produce an extensive report structured as a mini scientific
    paper. Every conclusion and claim must be directly supported by concrete results
    from the study — cite specific candidates, sites, terminations, G(O), G(OH), and
    overpotential values explicitly. Be critical of your conclusions and assumptions:
    acknowledge limitations, uncertainties, and cases where the data is inconclusive. Do
    not make claims that are not backed by data. The report should include:

    - A summary of the screening strategy and how it evolved.
    - The best candidates identified, with their G(O), G(OH), ideal overpotential, and
    scaling-relation overpotential.
    - A comparison of the best candidates with available literature.
    - What was learned, what worked, and what did not.
    - Which hypotheses were confirmed or rejected, with explicit reference to the
    supporting data.
    - Any trends worth noting across the dataset, even if these trends do not lead to
    competitive candidates.
    - Any recommendations for future studies or next steps based on the findings and
    limitations of the current study.

    You have a maximum of 30 days to complete the entire study and make your final report.
    Try to make use of the available time.
    """
    # TODO TODO TODO - adjust the time limit
    
    # revised_message = "wait for 1 minutes. repeat 3 times"
    config = load_config(os.path.join('./config', "default.yaml"))
    # check_config(config)

    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    # print N number of '#', where n = len("##  Working directory: " + WORKING_DIRECTORY + " ##")
    print("#" * (len("##  Working directory: " + WORKING_DIRECTORY + " ##")))
    print("##  Working directory: " + WORKING_DIRECTORY + " ##")
    print("#" * (len("##  Working directory: " + WORKING_DIRECTORY + " ##")))

    assert WORKING_DIRECTORY is not None, "Please set the WORKING_DIRECTORY var"

    CANVAS.set_working_directory(WORKING_DIRECTORY)
    # CANVAS.canvas["finished_job_list"] = ["CO_Pt111_fcc_upright_k_0.3_ecutwfc_60.pwi"]

    # set environment variable
    os.environ["OMP_NUM_THREADS"] = "1"

    timeTravelToXFrameBefore = 0
    overwrite = False
    if len(sys.argv) > 1 and sys.argv[1] == "ow":
        print()
        print("####################")
        print("# Force over write #")
        print("####################")
        print()
        overwrite = True
    # no folder created. starting fresh
    elif not os.path.exists(WORKING_DIRECTORY):
        print()
        print("################################################")
        print("# WORKING_DIRECTORY doesn't exist. Start fresh #")
        print("################################################")
        print()
        overwrite = True
    # if only created folder but nothing in it. start fresh
    elif os.path.exists(WORKING_DIRECTORY) and sum(1 for p in Path(WORKING_DIRECTORY).iterdir()) == 0:
        print()
        print("##############################################################")
        print("# WORKING_DIRECTORY exist, but contains nothing. Start fresh #")
        print("##############################################################")
        print()
        overwrite = True
    elif len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            timeTravelToXFrameBefore = int(sys.argv[1])
            print()
            print("############################################################")
            print(f"# Time traveling {timeTravelToXFrameBefore} frames back")
            print("############################################################")
            print()
        else:
            assert False, "Invalid argument. Use 'ow' for overwrite or a number for time travel or leave blank to resume."
    else:
        print()
        print("############")
        print("# Resuming #")
        print("############")
        print()
        overwrite = False

    if overwrite:
        # check if working directory exists, if so delete it
        if os.path.exists(WORKING_DIRECTORY):
            os.system(f"rm -rf {WORKING_DIRECTORY}")
        
        os.makedirs(WORKING_DIRECTORY, exist_ok=False)
    
        # check if resource_suggestions.db exist in the working directory
        db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
        if os.path.exists(db_file):
            os.remove(db_file)
        initialize_database(db_file)
    else:
        db_file = os.path.join(WORKING_DIRECTORY, 'resource_suggestions.db')
        if not os.path.exists(db_file):
            initialize_database(db_file)

    EXPLOG_MODE = "production"
    EXPLOG.init(Path(WORKING_DIRECTORY)/"vasp_calcs",
                EXPLOG_MODE,
                reject_if_failed_exists = True,
                require_relaxed_o_for_oh = True,
                legacy_disposition_exempt_ids = var.LEGACY_DISPOSITION_EXEMPT_IDS,
                disposition_decisions = var.DISPOSITION_DECISIONS,
                )
    # SAFETY: a positive var.QUEUE_MIN_PENDING arms the queue-floor gate, which
    # pushes the worker to keep the HPC queue stocked with REAL DFT jobs. In any
    # test mode (anything other than "production") that is dangerous, so warn
    # loudly if a non-zero floor was left on. Set var.QUEUE_MIN_PENDING = 0 for tests.
    if EXPLOG_MODE != "production" and var.QUEUE_MIN_PENDING > 0:
        print("\n" + "!" * 80, flush=True)
        print(
            f"!! WARNING: QUEUE_MIN_PENDING = {var.QUEUE_MIN_PENDING} (> 0) while "
            f"EXPLOG mode is '{EXPLOG_MODE}' (a TEST mode).",
            flush=True,
        )
        print(
            "!! The queue-floor gate will push the agent to keep REAL DFT jobs "
            "queued. Set var.QUEUE_MIN_PENDING = 0 for test modes.",
            flush=True,
        )
        print("!" * 80 + "\n", flush=True)
    # print(EXPLOG.relational_frame.candidates.df.dtypes)
    # print(EXPLOG.relational_frame.processes.df.dtypes)
    # new_candidate_row = {"candidate_id": "test_candidate",
    #                          "reason_or_hypothesis": "test_reason",
    # }
    # EXPLOG.relational_frame.candidates.add_row(new_candidate_row, 
    #                                              allow_update=False)
    # print(EXPLOG.relational_frame.candidates.df)
    # print(EXPLOG.relational_frame.candidates.df.dtypes)
    # assert False    
    CANVAS.set_working_directory(WORKING_DIRECTORY)
    Worker_available_tools = {}
    with open(os.path.join('./config', "oer_available_tools.yaml"), "r") as f:
        Worker_available_tools = yaml.safe_load(f)
    CANVAS.write("Worker_available_tools", Worker_available_tools)


    print("Building agent graph...", flush=True)
    rawGraph = create_planning_graph(config)
    # graph = create_graph(config)
    llm_config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 2000}

    print("Start, check the log file for details", flush=True)
    log_filename = f"./log/agent_stream_{int(time.time())}.log"  # Add timestamp to filename
    with open(log_filename, "a") as log_file:
        log_file.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        var.startTime = time.time()
        if eval(config["SAVE_DIALOGUE"]):
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
              
        try:
            # Allow our own classes (artifacts, plan steps, tool arg models)
            # to round-trip through checkpoint deserialization without the
            # "unregistered type ... will be blocked" path in newer langgraph.
            serde = JsonPlusSerializer(
                pickle_fallback=True,
                allowed_msgpack_modules=[
                    ("src.myCANVAS",),
                    ("src.planNexe2",),
                    ("src.tools",),
                ],
            )
        except TypeError:
            # older langgraph-checkpoint without allowed_msgpack_modules
            serde = JsonPlusSerializer(pickle_fallback=True)
        db_path = f"{WORKING_DIRECTORY}/checkpoints.sqlite"
        checkpointer = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False), serde=serde)
        # WAL: concurrent-friendly journaling; the live db size is then
        # checkpoints.sqlite + checkpoints.sqlite-wal (handled in _checkpoint_db_size)
        # checkpointer.conn.execute("PRAGMA journal_mode=WAL")
        # with SqliteSaver.from_conn_string(f"{WORKING_DIRECTORY}/checkpoints.sqlite") as checkpointer:
        graph = rawGraph.compile(checkpointer=checkpointer)

        if overwrite:
            inputs = {
                "inputs": f"{minimal_test_message_3}",
                "plan": [],
                "past_steps": [],
                "draft_response": "",
                "boss_feedback": "",
                "response": "",
                "canvas": CANVAS.canvas,
                "artifacts": CANVAS.result_registry,
                "explog_candidates": EXPLOG.relational_frame.candidates.df,
                "explog_processes": EXPLOG.relational_frame.processes.df,
                "time": 0
            }
            llm_config = llm_config
        else:
            history = list(graph.get_state_history(llm_config))
            # print(history)
            snap = history[timeTravelToXFrameBefore]
            # print("\n\n\n")
            # print(snap)
            # print("\n\n\n")

            
            # NOTE: FAIL when trying to resume a checkpoint that predates the boss-era state format.
            required_boss_state_keys = {"draft_response", "boss_feedback", "response"}
            missing_boss_state_keys = required_boss_state_keys - set(snap.values.keys())
            if missing_boss_state_keys:
                raise ValueError(
                    "Checkpoint is missing boss-era state keys "
                    f"{sorted(missing_boss_state_keys)}. "
                    "This checkpoint likely predates the boss implementation. Start fresh with 'ow'."
                )
            

            inputs = None
            var.startTime = time.time() - snap.values["time"]
            print(f"Time since the start of the session: {time.time() - var.startTime} seconds")
            # snap.config only carries configurable{thread_id, checkpoint_ns,
            # checkpoint_id}; merging keeps our recursion_limit (otherwise it
            # silently falls back to the default of 25 and long runs die).
            llm_config = {**snap.config, "recursion_limit": 2000}
            CANVAS.canvas = snap.values["canvas"]
            CANVAS.result_registry = snap.values.get("artifacts", {})
            EXPLOG.relational_frame.candidates.df = snap.values["explog_candidates"]
            EXPLOG.relational_frame.processes.df = snap.values["explog_processes"]

            # ----------------------------------------------------------------
            # Tool-level resume: on a plain resume (frame 0), the interrupted
            # round may have fresher state in the inner agent's namespaced
            # checkpoints (StateSyncMiddleware writes canvas/explog/time into
            # the subgraph state after every tool call). Overlay it so the
            # globals match the point the subgraph will continue from.
            #
            # NOTE: after a hard crash (no interrupt), get_state(subgraphs=True)
            # does NOT hydrate task.state, and get_state on a namespace fails
            # for subgraphs invoked inside node functions — so we read the
            # newest namespaced checkpoint directly from the checkpointer.
            # checkpoint_ids are UUIDv6 (time-ordered strings): only overlay if
            # the namespaced checkpoint is NEWER than the parent round
            # checkpoint (a crash between rounds leaves older inner
            # checkpoints whose data the round checkpoint already contains).
            # Old-format checkpoints (pre tool-level era) have no namespaced
            # rows at all -> the round-level restore above stands.
            # When time traveling (frame > 0) we deliberately skip this and
            # fork from the round boundary.
            # ----------------------------------------------------------------
            if timeTravelToXFrameBefore == 0:
                parent_ckpt_id = snap.config["configurable"].get("checkpoint_id", "")
                row = checkpointer.conn.execute(
                    "SELECT checkpoint_ns, checkpoint_id FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_ns != '' "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (THREAD_ID,),
                ).fetchone()
                deepest = None
                if row is not None and row[1] > parent_ckpt_id:
                    sub_tuple = checkpointer.get_tuple(
                        {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": row[0]}}
                    )
                    if sub_tuple is not None:
                        vals = sub_tuple.checkpoint.get("channel_values", {}) or {}
                        if vals.get("canvas"):
                            deepest = vals
                if deepest is not None:
                    print()
                    print("##################################################################")
                    print("# Found mid-round (tool-level) checkpoint in the interrupted    #")
                    print(f"# step (ns: {row[0]}).")
                    print("# Restoring the fresher state from it; the inner agent will     #")
                    print("# resume after its last completed tool call.                    #")
                    print("##################################################################")
                    print()
                    CANVAS.canvas = deepest["canvas"]
                    CANVAS.result_registry = deepest.get("artifacts", CANVAS.result_registry)
                    if deepest.get("explog_candidates") is not None:
                        EXPLOG.relational_frame.candidates.df = deepest["explog_candidates"]
                    if deepest.get("explog_processes") is not None:
                        EXPLOG.relational_frame.processes.df = deepest["explog_processes"]
                    if deepest.get("time"):
                        var.startTime = time.time() - deepest["time"]
                    # one-shot handoff: worker_agent_node consumes this instead
                    # of resetting curr-round ids, so check_required_tool_use
                    # still sees the tools that ran before the crash
                    var.resume_curr_round_result_ids = deepest.get("curr_round_result_ids")

            # The canvas restore above reverts Worker_available_tools to the
            # checkpointed (possibly outdated) tool list; re-write it from the
            # yaml so resumed agents see the current tool set.
            CANVAS.write("Worker_available_tools", Worker_available_tools, overwrite=True)

            # Part 3: reconcile legacy dispositions against the new vocabulary +
            # terminal-tag gate (idempotent; runs every resume). Deletes trailing
            # invalid terminal decisions (re-flagging that work for a fresh
            # disposition) and renames orphaned 'Investigating' -> the neutral
            # active default. Mutating the loaded frame here persists forward via
            # the normal checkpointing of the next node.

            print("----------------------------------------------------------------")
            print("RECONCILING LEGACY DISPOSITIONS (idempotent, runs every resume)", flush=True)
            _recon = reconcile_dispositions(EXPLOG)
            if _recon:
                print(f"[reconcile] adjusted {len(_recon)} candidate disposition(s):")
                for _cid, _info in _recon.items():
                    print(f"  {_cid}: {_info}")
            print("----------------------------------------------------------------")

            # Part 4: self-heal + populate reduced_formula for every candidate
            # (idempotent, runs every resume). Explog.add_candidate already
            # self-heals the column's mere existence on its own -- this
            # additionally populates real AQ-GNoME values for candidates
            # already in the table immediately, rather than waiting for the
            # agent's first query_explog/read_explog call. Deliberately does
            # NOT call EXPLOG.update_log() (which also polls the live SLURM
            # queue in production mode -- a side effect this step has no
            # reason to trigger).
            EXPLOG.job_handler._ensure_reduced_formula_column()
            sync_reduced_formula(EXPLOG.relational_frame.candidates.df,
                                 _STABILITY_CACHE.candidate_lookup)

            CANVAS.print()
            print(CANVAS.result_registry)
            print(EXPLOG.relational_frame.flatten_explode())
            print()
            print("########################################################################################################")
            print("Resuming with the above snapshot. If you want to start fresh, please run with 'ow' argument. If you want to time travel x frame back, run 'python invoke.py x'.")
            print("########################################################################################################")
            
        # assert False
        # durability="sync": the checkpoint (parent round OR inner tool-level)
        # is committed to sqlite BEFORE execution continues, so a hard kill at
        # any point is resumable from the last completed step.
        for s in graph.stream(
            inputs, 
            llm_config,
            durability="sync",
            ):
            if "__end__" not in s:
                # print(s)
                print("----")
                if eval(config["SAVE_DIALOGUE"]):
                    with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                        # f.write(repr(s) + "\n")
                        f.write("----\n")
                
                # time.sleep(5)
                # Print to console
                log_file.write(f"{s}\n")
                log_file.write("----\n")
                log_file.flush()

            # Each yielded update == one completed super-step (one big round)
            # of the parent graph, i.e. a safe boundary to prune at.
            prune_old_rounds(checkpointer, db_path, THREAD_ID, max_bytes=30 * 1024**3, keep_rounds=3)
        log_file.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        if eval(config["SAVE_DIALOGUE"]):
            with open(f"{WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    print("End, check the log file for details")
