#!/usr/bin/env python
"""Read-only preview of the resume-time disposition reconciliation, reading the
EXACT source the resume hydrates: the latest parent checkpoint in
checkpoints.sqlite (NOT vasp_calcs/explog.pkl, which dry_run_disposition.py uses).

Loads `explog_candidates` / `explog_processes` from the newest thread-1,
checkpoint_ns="" checkpoint using the SAME SqliteSaver + JsonPlusSerializer
(pickle_fallback=True) construction as invoke.py, assigns them to a THROWAWAY
EXPLOG (in a tempdir), and runs reconcile_dispositions against it. It NEVER writes
to the run dir or the checkpoint -- the only mutation (apply=True, to compute the
post-reconcile Gate-1 backlog) hits the throwaway EXPLOG's tempdir.

Usage:
    python dry_run_reconcile_checkpoint.py [RUN_DIR]
RUN_DIR defaults to config/default.yaml WORKING_DIR. Reads RUN_DIR/checkpoints.sqlite only.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import yaml

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from gnome_dreams_oer_screening.explog.explog import EXPLOG
from src import var
from src.disposition_reconcile import reconcile_dispositions

THREAD_ID = "1"


def _run_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    with open("config/default.yaml") as f:
        return Path(yaml.safe_load(f)["WORKING_DIR"])


def _serde():
    try:
        return JsonPlusSerializer(
            pickle_fallback=True,
            allowed_msgpack_modules=[("src.myCANVAS",), ("src.planNexe2",), ("src.tools",)],
        )
    except TypeError:
        return JsonPlusSerializer(pickle_fallback=True)


def main():
    run_dir = _run_dir()
    db = run_dir / "checkpoints.sqlite"
    print("=" * 72)
    print(f"Reconciliation preview from the CHECKPOINT (READ-ONLY):\n  {db}")
    print("=" * 72)

    saver = SqliteSaver(sqlite3.connect(str(db), check_same_thread=False), serde=_serde())
    tup = saver.get_tuple({"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}})
    if tup is None:
        print("No parent checkpoint found for thread", THREAD_ID)
        return
    ckpt_id = tup.config["configurable"].get("checkpoint_id")
    vals = tup.checkpoint.get("channel_values", {})
    if "explog_candidates" not in vals:
        print("Latest checkpoint has no 'explog_candidates' channel; keys:",
              sorted(vals.keys()))
        return
    cand_df = vals["explog_candidates"]
    proc_df = vals["explog_processes"]
    print(f"latest parent checkpoint_id: {ckpt_id}")
    print(f"candidates: {len(cand_df)}   processes: {len(proc_df)}")

    # Throwaway EXPLOG with the checkpoint frames -- mirrors what invoke.py hydrates.
    tmp = Path(tempfile.mkdtemp(prefix="reconcile_ckpt_dry_"))
    EXPLOG.init(tmp / "vasp_calcs", mode="test",
                disposition_decisions=var.DISPOSITION_DECISIONS,
                legacy_disposition_exempt_ids=var.LEGACY_DISPOSITION_EXEMPT_IDS)
    EXPLOG.relational_frame.candidates.df = cand_df
    EXPLOG.relational_frame.processes.df = proc_df

    if "decision" in cand_df.columns:
        print("\ncurrent 'decision' counts (pre-reconcile):")
        print(cand_df["decision"].value_counts(dropna=False).to_string())

    print("\nGate-1 backlog PRE-reconcile:", len(EXPLOG.candidates_needing_disposition()))

    # 1) Preview (no writes).
    actions = reconcile_dispositions(EXPLOG, apply=False)
    n_del = sum(1 for v in actions.values() if v["deleted"])
    n_ren = sum(1 for v in actions.values() if v["renamed"])
    print("\n" + "=" * 72)
    print(f"RECONCILE PREVIEW (apply=False): {len(actions)} candidate(s) would change")
    print(f"    delete trailing terminal/failed record(s) : {n_del}")
    print(f"    rename 'Investigating' -> '{var.DISPOSITION_DEFAULT_ACTIVE}' : {n_ren}")
    print("=" * 72)
    for cid, info in list(actions.items())[:60]:
        print(f"    {cid}: {info}")
    if len(actions) > 60:
        print(f"    (and {len(actions) - 60} more)")

    # 2) Apply on the throwaway (writes only to the tempdir) -> post-reconcile backlog.
    reconcile_dispositions(EXPLOG, apply=True)
    print("\nGate-1 backlog POST-reconcile "
          "(catch-up dispositions the resume will require):",
          len(EXPLOG.candidates_needing_disposition()))
    print("\nDONE -- read-only: the real checkpoint and run dir were NOT modified.")


if __name__ == "__main__":
    main()
