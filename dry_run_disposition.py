#!/usr/bin/env python
"""Read-only disposition dry run + legacy-exempt-id snapshot for a production run.

Purpose (pre-first-resume cutover for the disposition gate):
  1. SIZE the first-resume catch-up backlog: how many candidates Gate 1 will
     demand a disposition for (the >=1-disposition rule fires once per candidate
     with finalized work, so this is the count regardless of the legacy set).
  2. SURFACE bugs against REAL data with NO agent in the loop -- the gate /
     coverage / self-heal / forgotten-jobs logic runs here, so any problem shows
     up as a traceback you can fix, not an error the live agent spirals on.
  3. EMIT the legacy-exempt process-id set -- the frozen snapshot of process ids
     already FINALIZED before the first resume. Written to
     src/legacy_disposition_exempt_ids.json so it can be wired into
     var.LEGACY_DISPOSITION_EXEMPT_IDS (hard save, used on every resume, never
     updated). The legacy set lets each historical candidate be dispositioned
     with an EMPTY citation instead of citing old ids it never analysed.

Source: vasp_calcs/explog.pkl (the job-handler's last-saved {candidates_df,
processes_df}) -- pandas only, fast, and saved at the same stop as the checkpoint
(so its terminal-id set is a subset of what resume loads -> can only ever
under-exempt, never wrongly exempt). It does NOT touch the production directory.

Usage:
    python dry_run_disposition.py [RUN_DIR]
RUN_DIR defaults to the 27-05 backup. Reads RUN_DIR/vasp_calcs/explog.pkl only.
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd

from gnome_dreams_oer_screening.explog.explog import EXPLOG
from src import var
from src.forgotten_jobs import find_forgotten_jobs

DEFAULT_RUN_DIR = Path(
    "/home/scratch3/matnis/production_run_27-05-2026_backup_11_06_2026"
)
LEGACY_OUT = Path(__file__).resolve().parent / "src" / "legacy_disposition_exempt_ids.json"


def _hr(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    pkl = run_dir / "vasp_calcs" / "explog.pkl"
    _hr(f"Disposition dry run (READ-ONLY) for:\n  {run_dir}")
    print(f"Loading frames from {pkl} ...")
    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    cand_df = payload["candidates_df"]
    proc_df = payload["processes_df"]
    print(f"  candidates: {len(cand_df)}   processes: {len(proc_df)}")

    # ---- legacy-exempt set = process ids already FINALIZED (terminal) --------
    is_terminal = _terminal_predicate()
    term_mask = proc_df["status"].apply(is_terminal)
    legacy_ids = sorted(int(p) for p in proc_df.loc[term_mask, "process_id"].tolist())

    # ---- build a throwaway EXPLOG and assign the real frames (mirrors resume)-
    tmp = Path(tempfile.mkdtemp(prefix="disp_dry_run_"))
    EXPLOG.init(tmp / "vasp_calcs", mode="test",
                disposition_decisions=var.DISPOSITION_DECISIONS,
                legacy_disposition_exempt_ids=set(legacy_ids))
    EXPLOG.relational_frame.candidates.df = cand_df
    EXPLOG.relational_frame.processes.df = proc_df

    # ---- 1) terminal-id snapshot --------------------------------------------
    _hr("1) Legacy-exempt snapshot (finalized process ids)")
    print(f"finalized (terminal) process ids: {len(legacy_ids)}")
    by_type = (proc_df.loc[term_mask].groupby("job_type")["process_id"].count()
               .sort_index())
    for jt, n in by_type.items():
        print(f"    {jt:<20} {n}")
    LEGACY_OUT.write_text(json.dumps(legacy_ids))
    print(f"-> wrote {len(legacy_ids)} ids to {LEGACY_OUT}")

    # ---- 2) Gate-1 backlog (REAL self-heal of the 4 columns happens here) ----
    _hr("2) Gate-1 first-resume backlog (candidates needing a disposition)")
    needing = EXPLOG.candidates_needing_disposition()  # recomputes + self-heals
    print(f"candidates Gate 1 will block on at first resume: {len(needing)}")
    print(f"  (= one forced 'catch-up' disposition each)")
    if needing:
        head = ", ".join(needing[:25])
        more = "" if len(needing) <= 25 else f"  (and {len(needing) - 25} more)"
        print(f"  e.g. {head}{more}")

    # per-stage depth + failed, from the progress projection
    jh = EXPLOG.job_handler
    jh._recompute_candidate_progress()
    cdf = EXPLOG.relational_frame.candidates.df

    def _ge1(col):
        if col not in cdf.columns:
            return 0
        s = pd.to_numeric(cdf[col], errors="coerce")
        return int((s >= 1).sum())

    _hr("   pipeline depth of all candidates (non-exclusive)")
    print(f"    bulk finalized   : {_ge1('n_bulk_finalized')}")
    print(f"    surface finalized: {_ge1('n_surface_finalized')}")
    print(f"    O finalized      : {_ge1('n_O_finalized')}")
    print(f"    OH finalized     : {_ge1('n_OH_finalized')}")
    if "state" in cdf.columns:
        n_failed = int((cdf["state"].astype("string") == "failed").sum())
        print(f"    state == failed  : {n_failed}  (these still need a disposition)")

    # ---- 3) forgotten-jobs smoke (ready-but-unstarted work) ------------------
    _hr("3) find_forgotten_jobs smoke (Gate-2 hint input)")
    forgotten = find_forgotten_jobs(EXPLOG, var.GO_DEV_OH_THRESHOLD)
    print(f"ready-but-unstarted items: {len(forgotten)}")
    kinds = {}
    for it in forgotten:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    for k in ("bulk", "surface", "O", "OH"):
        if k in kinds:
            print(f"    {k:<8} {kinds[k]}")

    _hr("DONE -- no production files were modified")
    print("Next: review the backlog size above, then (if it looks right) wire")
    print("src/legacy_disposition_exempt_ids.json into var.LEGACY_DISPOSITION_EXEMPT_IDS.")


def _terminal_predicate():
    """The same terminal-status rule the job handler uses, without needing an
    initialized handler (status -> bool)."""
    def is_terminal(status):
        if not isinstance(status, str):
            return False
        return status in {"completed", "failed"} or status.startswith("unrecoverable")
    return is_terminal


if __name__ == "__main__":
    main()
