"""One-off script: inject a new `inputs` message into a forked run's latest
LangGraph checkpoint. Not part of the app.

Reuses invoke.py's own setup by `import invoke` -- this is safe because
invoke.py's SqliteSaver/graph.stream logic lives inside `if __name__ ==
"__main__":`, so importing it only runs module-level defs/imports (see
tests/test_query_explog_tool.py, which relies on the same property).

Run with (same env invoke.py itself needs):
    ml Python/3.11.3-GCCcore-12.3.0
    source venv2/bin/activate
    python _fork_inject.py
"""
import os
import sqlite3

import invoke  # noqa: F401  (side effect: makes src.*, gnome_dreams_oer_screening.* importable & loaded)

from src.utils import load_config
from src import var
from src.myCANVAS import CANVAS
from src.planNexe2 import create_planning_graph
from gnome_dreams_oer_screening.explog.explog import EXPLOG
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

FORK_DIR = "/home/energy/matnis/projects/dreams_colab/v2/material_agent/production_run_27-05-2026_fork_16_07_2026"
THREAD_ID = "1"

# EDIT THIS before running: the real message to inject into the fork's state.
NEW_MESSAGE = """
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
    Make use of the available time and be ambitious in your investigations of the candidate
    space. If you have plenty of time left, expand your study to more candidates and plan
    accordingly.
"""

assert os.path.isdir(FORK_DIR), f"fork dir missing: {FORK_DIR}"
assert FORK_DIR.endswith("_fork_16_07_2026"), "safety check: refusing to run against a non-fork path"

os.environ["OMP_NUM_THREADS"] = "1"

config = load_config(os.path.join("./config", "default.yaml"))
config["WORKING_DIR"] = FORK_DIR
var.my_WORKING_DIRECTORY = FORK_DIR
CANVAS.set_working_directory(FORK_DIR)

EXPLOG.init(
    __import__("pathlib").Path(FORK_DIR) / "vasp_calcs",
    "production",
    reject_if_failed_exists=True,
    require_relaxed_o_for_oh=True,
    legacy_disposition_exempt_ids=var.LEGACY_DISPOSITION_EXEMPT_IDS,
    disposition_decisions=var.DISPOSITION_DECISIONS,
)

import yaml
with open(os.path.join("./config", "oer_available_tools.yaml"), "r") as f:
    Worker_available_tools = yaml.safe_load(f)
CANVAS.write("Worker_available_tools", Worker_available_tools)

try:
    serde = JsonPlusSerializer(
        pickle_fallback=True,
        allowed_msgpack_modules=[
            ("src.myCANVAS",),
            ("src.planNexe2",),
            ("src.tools",),
        ],
    )
except TypeError:
    serde = JsonPlusSerializer(pickle_fallback=True)

db_path = f"{FORK_DIR}/checkpoints.sqlite"
checkpointer = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False), serde=serde)

rawGraph = create_planning_graph(config)
graph = rawGraph.compile(checkpointer=checkpointer)

llm_config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 2000}

before = graph.get_state(llm_config)
print("BEFORE inputs (first 200 chars):", repr(before.values.get("inputs", ""))[:200])

graph.update_state(llm_config, {"inputs": NEW_MESSAGE})

after = graph.get_state(llm_config)
print("AFTER inputs:", repr(after.values.get("inputs", "")))
print("Injected successfully.")
