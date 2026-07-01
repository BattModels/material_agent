# Tests for the resume-time disposition reconciliation (Part 3), test-FIRST.
#
# Split like the rest of the disposition code: a PURE classifier (no EXPLOG) that
# decides delete/rename/leave for one candidate's current decision, and an applier
# that walks the trailing records of the real (nested) EXPLOG frame. Neither imports
# src.tools, so this whole file runs in the fast suite.

import pandas as pd

from gnome_dreams_oer_screening.explog.explog import EXPLOG

from src import var
from src.disposition_reconcile import (
    classify_disposition_reconciliation,
    reconcile_dispositions,
)


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------

_T = ("Abandon", "Sufficient")
_DEF = "Medium priority"


def _classify(decision, state, is_forgotten, has_in_flight):
    return classify_disposition_reconciliation(
        decision=decision, state=state, is_forgotten=is_forgotten,
        has_in_flight=has_in_flight, terminal_decisions=_T, default_active=_DEF,
    )


def test_classify_terminal_on_unsettled_deletes():
    assert _classify("Abandon", "surface_relaxation", True, False) == ("delete", None)
    assert _classify("Sufficient", "surface_relaxation", False, True) == ("delete", None)


def test_classify_terminal_on_settled_left():
    assert _classify("Abandon", "surface_relaxation", False, False) == ("leave", None)
    assert _classify("Sufficient", None, False, False) == ("leave", None)


def test_classify_investigating_renamed_when_non_failed():
    # regardless of forgotten / in-flight: it is an active tag, only the name is stale
    assert _classify("Investigating", "surface_relaxation", True, True) == \
        ("rename", "Medium priority")
    assert _classify("Investigating", None, False, False) == ("rename", "Medium priority")


def test_classify_failed_non_abandon_deletes():
    assert _classify("Investigating", "failed", False, False) == ("delete", None)
    assert _classify("Sufficient", "failed", False, False) == ("delete", None)
    assert _classify("Medium priority", "failed", False, False) == ("delete", None)


def test_classify_failed_abandon_left():
    assert _classify("Abandon", "failed", True, True) == ("leave", None)


def test_classify_active_left_when_non_failed():
    assert _classify("Medium priority", "surface_relaxation", True, True) == ("leave", None)
    assert _classify("Low priority", None, False, False) == ("leave", None)


# ---------------------------------------------------------------------------
# Applier (drives the real nested EXPLOG frame)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="reconcile test")


def _inject(cid, job_type, status, termination_index=None, site_index=None):
    table = EXPLOG.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row({"process_id": pid, "candidate_id": cid, "job_type": job_type,
                   "status": status, "termination_index": termination_index,
                   "site_index": site_index}, allow_update=False)
    return pid


def _set_record(cid, decisions):
    # decisions: list of (Decision, [cited_ids]) -> stacked disposition_record
    cdf = EXPLOG.relational_frame.candidates.df
    idx = cdf.index[cdf["candidate_id"] == cid][0]
    rec = [{"Summary": "s", "Summarized_process_id": list(ids),
            "Future_plan": "f", "Decision": d} for d, ids in decisions]
    cdf.at[idx, "disposition_record"] = rec
    cdf.at[idx, "decision"] = rec[-1]["Decision"] if rec else pd.NA


def _val(cid, col):
    cdf = EXPLOG.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def _rec(cid):
    return EXPLOG.job_handler._candidate_dispositions(cid)


def _forgotten_at_O(cid):
    # Realistic frontier honoring the bulk -> surface -> O -> OH workflow order:
    # bulk + surface FINALIZED, O not yet started -> the candidate is a forgotten
    # job (at the O frontier) that still has finished work to cite in a disposition.
    pb = _inject(cid, "bulk_relaxation", "completed")
    ps = _inject(cid, "surface_relaxation", "completed", termination_index=0)
    return [pb, ps]


def test_apply_deletes_premature_abandon_and_flags_needs(tmp_path):
    # forgotten candidate (bulk+surface done, no O) marked Abandon prematurely
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Abandon", ids)])
    actions = reconcile_dispositions(EXPLOG)
    assert actions.get("c", {}).get("deleted") == 1
    assert _rec("c") == []                                 # trailing record popped
    assert pd.isna(_val("c", "decision"))                  # decision cleared
    assert _val("c", "needs_disposition_update") == True   # noqa: E712 uncovered -> re-disposition


def test_apply_renames_investigating_to_medium(tmp_path):
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Investigating", ids)])
    actions = reconcile_dispositions(EXPLOG)
    assert actions.get("c", {}).get("renamed") is True
    assert _val("c", "decision") == "Medium priority"
    assert _rec("c")[-1]["Decision"] == "Medium priority"
    assert _val("c", "needs_disposition_update") == False  # noqa: E712 coverage preserved


def test_apply_leaves_settled_abandon(tmp_path):
    # bulk+surface+O all completed, no competitive-OH gap, nothing in flight -> settled
    _setup(tmp_path); _add_candidate("c")
    pb = _inject("c", "bulk_relaxation", "completed")
    ps = _inject("c", "surface_relaxation", "completed", termination_index=0)
    po = _inject("c", "O_adsorption", "completed", termination_index=0, site_index=0)
    _set_record("c", [("Abandon", [pb, ps, po])])
    actions = reconcile_dispositions(EXPLOG)
    assert "c" not in actions                               # unchanged
    assert _val("c", "decision") == "Abandon"


def test_apply_failed_non_abandon_deleted(tmp_path):
    _setup(tmp_path); _add_candidate("c")
    EXPLOG.mark_candidate_failed("c")
    _set_record("c", [("Sufficient", [])])
    reconcile_dispositions(EXPLOG)
    assert _rec("c") == []
    assert pd.isna(_val("c", "decision"))


def test_apply_failed_abandon_left(tmp_path):
    _setup(tmp_path); _add_candidate("c")
    EXPLOG.mark_candidate_failed("c")
    _set_record("c", [("Abandon", [])])
    actions = reconcile_dispositions(EXPLOG)
    assert "c" not in actions
    assert _val("c", "decision") == "Abandon"


def test_apply_deletes_trailing_stacked_invalids_in_one_pass(tmp_path):
    # forgotten candidate dispositioned terminal twice in a row -> both popped
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Sufficient", ids), ("Abandon", ids)])
    actions = reconcile_dispositions(EXPLOG)
    assert actions.get("c", {}).get("deleted") == 2
    assert _rec("c") == []


def test_apply_reverts_to_prior_valid_active_below_bad_terminal(tmp_path):
    # [active, Abandon(premature)] -> delete only the bad terminal, keep the active
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Medium priority", ids), ("Abandon", ids)])
    reconcile_dispositions(EXPLOG)
    assert _rec("c") == [{"Summary": "s", "Summarized_process_id": ids,
                          "Future_plan": "f", "Decision": "Medium priority"}]
    assert _val("c", "decision") == "Medium priority"


def test_apply_idempotent(tmp_path):
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Investigating", ids)])
    reconcile_dispositions(EXPLOG)
    first = _val("c", "decision")
    actions2 = reconcile_dispositions(EXPLOG)              # second pass -> no-op
    assert "c" not in actions2
    assert _val("c", "decision") == first


def test_apply_dry_run_reports_without_writing(tmp_path):
    _setup(tmp_path); _add_candidate("c")
    ids = _forgotten_at_O("c")
    _set_record("c", [("Abandon", ids)])
    actions = reconcile_dispositions(EXPLOG, apply=False)
    assert actions.get("c", {}).get("deleted") == 1        # reported
    assert _val("c", "decision") == "Abandon"              # but NOT written
    assert _rec("c")[-1]["Decision"] == "Abandon"


def test_apply_deletes_terminal_on_in_flight_candidate(tmp_path):
    # Not forgotten (surface started) but a surface job is RUNNING -> in-flight;
    # a premature terminal is deleted on the has_in_flight branch, not forgotten.
    _setup(tmp_path); _add_candidate("c")
    pb = _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "running", termination_index=0)
    _set_record("c", [("Abandon", [pb])])
    actions = reconcile_dispositions(EXPLOG)
    assert actions.get("c", {}).get("deleted") == 1
    assert _rec("c") == []
    assert pd.isna(_val("c", "decision"))


def test_apply_handles_many_candidates_in_one_pass(tmp_path):
    # One reconcile call mixing delete / rename / leave, each candidate independent.
    _setup(tmp_path)
    for cid in ("del", "ren", "keep"):
        _add_candidate(cid)
    _set_record("del", [("Abandon", _forgotten_at_O("del"))])       # forgotten -> delete
    _set_record("ren", [("Investigating", _forgotten_at_O("ren"))])  # orphaned -> rename
    kb = _inject("keep", "bulk_relaxation", "completed")
    ks = _inject("keep", "surface_relaxation", "completed", termination_index=0)
    ko = _inject("keep", "O_adsorption", "completed", termination_index=0, site_index=0)
    _set_record("keep", [("Abandon", [kb, ks, ko])])                # settled -> leave
    actions = reconcile_dispositions(EXPLOG)
    assert set(actions) == {"del", "ren"}                           # 'keep' untouched
    assert actions["del"]["deleted"] == 1
    assert actions["ren"]["renamed"] is True
    assert pd.isna(_val("del", "decision"))
    assert _val("ren", "decision") == "Medium priority"
    assert _val("keep", "decision") == "Abandon"