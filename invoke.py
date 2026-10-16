"""Driver for the directional-coupler screening study.

    python invoke.py            resume the run in config WORKING_DIR (or start fresh if empty)
    python invoke.py ow         delete WORKING_DIR and start fresh
    python invoke.py <N>        time-travel: resume from N round-checkpoints back
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

import yaml
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src import var
from src.fem_explog import EXPLOG
from src.history_log import write_history
from src.myCANVAS import CANVAS
from src.planNexe2 import create_planning_graph
from src.utils import load_config


# =====================================================================
# Checkpoint retention
# =====================================================================
THREAD_ID = "1"
MAX_CHECKPOINT_DB_BYTES = 20 * 1024**3   # prune once sqlite (+wal) exceeds this
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

    Round-level checkpoints are the rows with checkpoint_ns = '' (the parent
    PlanExecute graph); the inner agents' tool-level checkpoints live in
    namespaced rows. LangGraph checkpoint_ids are UUIDv6 (time-ordered), so one
    cutoff id separates old from new across all namespaces. Safe to call between
    super-steps.
    """
    if _checkpoint_db_size(db_path) < max_bytes:
        return

    conn = checkpointer.conn
    deleted = 0
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
        for table in ("checkpoints", "writes"):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_id < ?",
                (thread_id, cutoff),
            )
            deleted += cur.rowcount
        conn.commit()

    if deleted:
        try:
            with checkpointer.lock:
                conn.execute("VACUUM")
        except sqlite3.OperationalError as e:
            print(f"[prune] VACUUM skipped ({e}); space will be reused, file not shrunk")
        print(f"[prune] removed {deleted} checkpoint/write rows older than round cutoff {cutoff}; "
              f"db now {_checkpoint_db_size(db_path) / 1024**2:.1f} MB")


# =====================================================================
# Objective
# =====================================================================
# target = f"""at 1.55 um, find designs whose coupling length is as close as possible to {var.TARGET_COUPLING_LENGTH_UM:g} um
# (minimise coupling_length_deviation_um) while keeping the excess loss below 0.22 dB (maximum power transfer
# above 0.95), and report how far each swept parameter moves the coupling length.
# """

target = "find designs that minimize both the coupling length deviation and the excess loss"

OBJECTIVE = f"""
Please conduct a screening study of two-waveguide directional couplers using finite-element mode solving
and coupled-mode theory, to understand how the design parameters control the coupling behaviour and to
identify the best designs.

Design space to explore (one parameter at a time, everything else fixed, unless you have a reason not to):
  - gap between the cores: 0.1 to 1.0 um (how separation affects coupling)
  - waveguide 1 width: 0.40 to 0.55 um (how geometry affects its mode)
  - waveguide 2 width: 0.40 to 0.55 um (effect of mismatch between neighbours)
  - wavelength: 1.3 to 1.6 um (how coupling changes with wavelength)
  - core / cladding materials from the material library (how index contrast affects confinement and coupling)

Target: {target}

Please do an iterative multi-round screening: each round tests one hypothesis with a small deliberate set
of designs, learning from each round and applying insights to the next. Use literature searches to inform
your hypotheses and to compare your results with published couplers, and note them down clearly.

Report:
At the end of EACH ROUND as well as the end of the study, produce an extensive report structured as a mini
scientific paper. Every conclusion and claim must be directly supported by concrete results from the study --
cite specific designs (candidate ids and parameters), supermode indices, coupling lengths and power-transfer
values explicitly. Be critical of your conclusions and assumptions: acknowledge limitations, uncertainties,
and cases where the data is inconclusive. Do not make claims that are not backed by data. The report should include:
  - A summary of the screening strategy and how it evolved.
  - The best designs identified, with their parameters, supermode indices, coupling length and maximum power transfer.
  - How each swept parameter affects the coupling length and power transfer, compared with coupled-mode theory
    and the literature.
  - What was learned, what worked, and what did not.
  - Which hypotheses were confirmed or rejected, with explicit reference to the supporting data.
  - Any recommendations for future studies or next steps.

You have a maximum of {var.STUDY_BUDGET_HOURS} hours to complete the entire study and make your final report.
Try to make use of the available time.
"""

# OBJECTIVE = "Now just let me know what can you guys search and optimize? what metrics may I ask you to maximize or minimize? Don't do anything else."


if __name__ == "__main__":
    config = load_config(os.path.join("./config", "default.yaml"))
    WORKING_DIRECTORY = var.my_WORKING_DIRECTORY
    assert WORKING_DIRECTORY is not None, "Please set WORKING_DIR in config/default.yaml"
    banner = "##  Working directory: " + WORKING_DIRECTORY + " ##"
    print("#" * len(banner)); print(banner); print("#" * len(banner))

    # ---------------------------------------------------------------- mode
    timeTravelToXFrameBefore = 0
    overwrite = False
    if len(sys.argv) > 1 and sys.argv[1] == "ow":
        print("\n####################\n# Force over write #\n####################\n")
        overwrite = True
    elif not os.path.exists(WORKING_DIRECTORY):
        print("\n# WORKING_DIRECTORY doesn't exist. Start fresh #\n")
        overwrite = True
    elif sum(1 for _ in Path(WORKING_DIRECTORY).iterdir()) == 0:
        print("\n# WORKING_DIRECTORY exists but is empty. Start fresh #\n")
        overwrite = True
    elif len(sys.argv) > 1:
        assert sys.argv[1].isdigit(), \
            "Invalid argument. Use 'ow' for overwrite, a number for time travel, or leave blank to resume."
        timeTravelToXFrameBefore = int(sys.argv[1])
        print(f"\n# Time traveling {timeTravelToXFrameBefore} frames back #\n")
    else:
        print("\n############\n# Resuming #\n############\n")

    if overwrite:
        if os.path.exists(WORKING_DIRECTORY):
            os.system(f"rm -rf {WORKING_DIRECTORY}")
        os.makedirs(WORKING_DIRECTORY, exist_ok=False)

    # ---------------------------------------------------------------- globals
    EXPLOG.init(Path(WORKING_DIRECTORY), disposition_decisions=var.DISPOSITION_DECISIONS,
                target_coupling_length_um=var.TARGET_COUPLING_LENGTH_UM)
    CANVAS.set_working_directory(WORKING_DIRECTORY)
    with open(os.path.join("./config", "fem_available_tools.yaml"), "r") as f:
        Worker_available_tools = yaml.safe_load(f)
    CANVAS.write("Worker_available_tools", Worker_available_tools)

    print("Building agent graph...", flush=True)
    rawGraph = create_planning_graph(config)
    llm_config = {"configurable": {"thread_id": THREAD_ID}, "recursion_limit": 2000}

    print("Start, check the log file for details", flush=True)
    os.makedirs("./log", exist_ok=True)
    log_filename = f"./log/agent_stream_{int(time.time())}.log"
    with open(log_filename, "a") as log_file:
        log_file.write(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        var.startTime = time.time()
        write_history(f"=== Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

        # Our own classes (artifacts, plan steps, tool arg models) must
        # round-trip through checkpoint deserialization.
        try:
            serde = JsonPlusSerializer(
                pickle_fallback=True,
                allowed_msgpack_modules=[("src.myCANVAS",), ("src.planNexe2",), ("src.tools",)],
            )
        except TypeError:  # older langgraph-checkpoint without allowed_msgpack_modules
            serde = JsonPlusSerializer(pickle_fallback=True)
        db_path = f"{WORKING_DIRECTORY}/checkpoints.sqlite"
        checkpointer = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False), serde=serde)
        graph = rawGraph.compile(checkpointer=checkpointer)

        if overwrite:
            inputs = {
                "inputs": OBJECTIVE,
                "plan": [],
                "past_steps": [],
                "draft_response": "",
                "boss_feedback": "",
                "response": "",
                "canvas": CANVAS.canvas,
                "artifacts": CANVAS.result_registry,
                "explog_candidates": EXPLOG.relational_frame.candidates.df,
                "explog_processes": EXPLOG.relational_frame.processes.df,
                "time": 0,
            }
        else:
            # get_state_history() is reverse-chronological: history[0] is the newest.
            history = list(graph.get_state_history(llm_config))
            snap = history[timeTravelToXFrameBefore]

            inputs = None
            var.startTime = time.time() - snap.values["time"]
            print(f"Time since the start of the session: {time.time() - var.startTime} seconds")
            # snap.config only carries configurable{...}; keep our recursion_limit.
            llm_config = {**snap.config, "recursion_limit": 2000}
            CANVAS.canvas = snap.values["canvas"]
            CANVAS.result_registry = snap.values.get("artifacts", {})
            EXPLOG.relational_frame.candidates.df = snap.values["explog_candidates"]
            EXPLOG.relational_frame.processes.df = snap.values["explog_processes"]

            # Tool-level resume: on a plain resume (frame 0) the interrupted round
            # may have fresher state in the inner agent's namespaced checkpoints
            # (StateSyncMiddleware writes canvas/explog/time after every tool call).
            # Overlay it when that checkpoint is NEWER than the round checkpoint.
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
                    print(f"\n# Found mid-round (tool-level) checkpoint in the interrupted step (ns: {row[0]}).")
                    print("# Restoring the fresher state from it; the inner agent resumes after its last tool call.\n")
                    CANVAS.canvas = deepest["canvas"]
                    CANVAS.result_registry = deepest.get("artifacts", CANVAS.result_registry)
                    if deepest.get("explog_candidates") is not None:
                        EXPLOG.relational_frame.candidates.df = deepest["explog_candidates"]
                    if deepest.get("explog_processes") is not None:
                        EXPLOG.relational_frame.processes.df = deepest["explog_processes"]
                    if deepest.get("time"):
                        var.startTime = time.time() - deepest["time"]
                    var.resume_curr_round_result_ids = deepest.get("curr_round_result_ids")

            # The canvas restore reverts Worker_available_tools to the checkpointed
            # list; re-write it so resumed agents see the current tool set.
            CANVAS.write("Worker_available_tools", Worker_available_tools, overwrite=True)
            EXPLOG.update_log()

            CANVAS.print()
            print(EXPLOG.relational_frame.candidates.df.drop(columns=["disposition_record"], errors="ignore"))
            print(EXPLOG.relational_frame.processes.df)
            print("\nResuming with the above snapshot. 'ow' starts fresh; 'python invoke.py x' time-travels x frames back.\n")

        # durability="sync": every checkpoint is committed before execution
        # continues, so a hard kill is resumable from the last completed step.
        for s in graph.stream(inputs, llm_config, durability="sync"):
            if "__end__" not in s:
                print("----")
                write_history("----\n")
                log_file.write(f"{s}\n----\n")
                log_file.flush()
            prune_old_rounds(checkpointer, db_path, THREAD_ID)
        log_file.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        write_history(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    print("End, check the log file for details")
