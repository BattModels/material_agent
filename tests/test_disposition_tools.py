# Tests for the disposition WRITE operations (Step 3), test-FIRST (TDD).
#
# The logic lives in the nested library as Explog methods (next to Step 2's
# candidate_outstanding), so this imports the INSTALLED nested package directly
# -- never src.tools (which loads the GNoME database on import, ~109s). src.var
# is imported only for the Decision vocabulary (a tiny config module).
#
# These methods return STRUCTURED dicts (so tests assert real fields, never
# substrings). The parent @tool in src/tools.py turns the structured result into
# the agent-facing sentence via src.disposition_messages (tested separately in
# tests/test_disposition_messages.py).
#
# Methods under test (do not exist yet -> RED):
#   EXPLOG.get_disposition_info(cid) -> dict   # candidate_outstanding + sets ready flag
#   EXPLOG.update_disposition_info(cid, summary, summarized_process_id,
#                                  future_plan, decision) -> dict
#       {"ok": bool, "status": "locked"|"invalid_decision"|"non_terminal_ids"
#                              |"incomplete"|"ok",
#        "missing": [int]  (incomplete), "ids": [int] (non_terminal_ids),
#        "decision": str, "candidate_id": str}

import pandas as pd

from gnome_dreams_oer_screening.explog.explog import EXPLOG
from src import var


# ---------------------------------------------------------------------------
# Helpers (operate on the EXPLOG singleton the agent tools use)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")
    # Decision vocabulary is sourced from var.py and lives on the handler.
    EXPLOG.job_handler.disposition_decisions = tuple(var.DISPOSITION_DECISIONS)


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="disposition tool test")


def _inject(cid, job_type, status, termination_index=None, site_index=None):
    table = EXPLOG.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row({"process_id": pid, "candidate_id": cid, "job_type": job_type,
                   "status": status, "termination_index": termination_index,
                   "site_index": site_index}, allow_update=False)
    return pid


def _val(cid, col):
    cdf = EXPLOG.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def _dispositions(cid):
    return EXPLOG.job_handler._candidate_dispositions(cid)


def _update(cid, ids, decision="Investigating", summary="s", future="f"):
    return EXPLOG.update_disposition_info(cid, summary, list(ids), future, decision)


# ---------------------------------------------------------------------------
# get_disposition_info
# ---------------------------------------------------------------------------

def test_get_sets_ready_and_returns_outstanding(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    res = EXPLOG.get_disposition_info("c")
    assert _val("c", "ready_for_disposition_update") == True  # noqa: E712
    assert any(pid_s in set(u["ids"]) for u in res["must_cover"])


# ---------------------------------------------------------------------------
# update_disposition_info — guard rails (structured status, no change)
# ---------------------------------------------------------------------------

def test_update_before_get_is_locked(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    res = _update("c", [pid_s])                  # ready defaults False
    assert res["status"] == "locked"
    assert res["ok"] is False
    assert _dispositions("c") == []
    assert pd.isna(_val("c", "decision"))


def test_update_invalid_decision(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    EXPLOG.get_disposition_info("c")
    res = _update("c", [pid_s], decision="Maybe")
    assert res["status"] == "invalid_decision"
    assert _dispositions("c") == []


def test_update_nonterminal_ids(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    pid_run = _inject("c", "O_adsorption", "running",
                      termination_index=0, site_index=0)
    EXPLOG.get_disposition_info("c")
    res = _update("c", [pid_run])
    assert res["status"] == "non_terminal_ids"
    assert pid_run in res["ids"]
    assert _dispositions("c") == []


# ---------------------------------------------------------------------------
# update_disposition_info — coverage outcomes
# ---------------------------------------------------------------------------

def test_update_complete_records(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    EXPLOG.get_disposition_info("c")
    res = _update("c", [pid_s], decision="Investigating")
    assert res["status"] == "ok"
    recs = _dispositions("c")
    assert len(recs) == 1
    assert recs[0]["Decision"] == "Investigating"
    assert recs[0]["Summarized_process_id"] == [pid_s]
    assert _val("c", "decision") == "Investigating"
    assert _val("c", "ready_for_disposition_update") == False  # noqa: E712
    assert _val("c", "needs_disposition_update") == False      # noqa: E712


def test_update_incomplete_reports_missing(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_a = _inject("c", "surface_relaxation", "completed", termination_index=0)
    pid_b = _inject("c", "O_adsorption", "completed",
                    termination_index=0, site_index=0)
    EXPLOG.get_disposition_info("c")
    res = _update("c", [pid_a])                  # covers only one of two units
    assert res["status"] == "incomplete"
    assert pid_b in res["missing"]
    assert _dispositions("c") == []
    # Nothing was written, so the agent may retry update directly with the
    # missing ids -- no re-get required. The lock only trips after a success.
    assert _val("c", "ready_for_disposition_update") == True   # noqa: E712
    assert _val("c", "needs_disposition_update") == True       # noqa: E712


def test_write_lock_blocks_second_update(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    EXPLOG.get_disposition_info("c")
    _update("c", [pid_s])                         # success -> ready False
    res = _update("c", [pid_s])                   # no intervening get
    assert res["status"] == "locked"


def test_citing_one_oh_subjob_covers_batch(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    p2 = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    EXPLOG.get_disposition_info("c")
    res = _update("c", [p2], decision="Sufficient")   # one sub-job covers the batch
    assert res["status"] == "ok"
    assert _val("c", "needs_disposition_update") == False  # noqa: E712
    assert _val("c", "decision") == "Sufficient"


def test_legacy_exempt_one_disposition_without_citing_ids(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    EXPLOG.job_handler.legacy_disposition_exempt_ids = {pid_s}
    EXPLOG.update_log()                           # >=1-disposition rule -> needs True
    assert _val("c", "needs_disposition_update") == True   # noqa: E712
    EXPLOG.get_disposition_info("c")
    res = _update("c", [], decision="Abandon")    # exempt -> need not cite pid_s
    assert res["status"] == "ok"
    assert _dispositions("c")[0]["Decision"] == "Abandon"
    assert _val("c", "needs_disposition_update") == False  # noqa: E712


def test_latest_disposition_is_the_most_recent_of_several(tmp_path):
    # Dispose the candidate TWICE as new work finishes, then confirm
    # get_disposition_info surfaces the SECOND (most recent) disposition -- this
    # is the part candidate_outstanding's recs[-1] actually decides.
    _setup(tmp_path)
    _add_candidate("c")
    pid_a = _inject("c", "surface_relaxation", "completed", termination_index=0)
    EXPLOG.get_disposition_info("c")
    _update("c", [pid_a], decision="Investigating")          # disposition #1

    pid_b = _inject("c", "O_adsorption", "completed",
                    termination_index=0, site_index=0)        # new finished work
    EXPLOG.get_disposition_info("c")
    _update("c", [pid_b], decision="Sufficient")             # disposition #2

    res = EXPLOG.get_disposition_info("c")
    assert len(_dispositions("c")) == 2
    assert res["latest_disposition"]["Decision"] == "Sufficient"   # most recent
    assert _val("c", "decision") == "Sufficient"             # visible column too


def test_update_unknown_candidate_does_not_crash(tmp_path):
    _setup(tmp_path)                                          # no candidate added
    res = EXPLOG.update_disposition_info("ghost", "s", [], "f", "Investigating")
    assert res["status"] == "unknown_candidate"
    assert res["ok"] is False


def test_get_unknown_candidate_is_flagged(tmp_path):
    _setup(tmp_path)                                          # no candidate added
    res = EXPLOG.get_disposition_info("ghost")
    assert res.get("unknown_candidate") is True              # not a vacuous success


def test_update_rejects_ids_from_another_candidate(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _add_candidate("other")
    pid_self = _inject("c", "surface_relaxation", "completed", termination_index=0)
    pid_foreign = _inject("other", "surface_relaxation", "completed",
                          termination_index=0)              # belongs to 'other'
    EXPLOG.get_disposition_info("c")
    res = _update("c", [pid_self, pid_foreign])
    assert res["status"] == "foreign_ids"
    assert pid_foreign in res["ids"]
    assert pid_self not in res["ids"]                        # only the foreign one
    assert _dispositions("c") == []                          # nothing written


def test_update_self_heals_disposition_columns_on_resume(tmp_path):
    # Simulate a pre-rollout checkpoint: drop the four disposition columns, then
    # call update WITHOUT a prior get/update_log -- it must self-heal and return
    # "locked" (the write-lock), never KeyError.
    _setup(tmp_path)
    _add_candidate("c")
    pid = _inject("c", "surface_relaxation", "completed", termination_index=0)
    cdf = EXPLOG.relational_frame.candidates.df
    cdf.drop(columns=["disposition_record", "decision",
                      "ready_for_disposition_update", "needs_disposition_update"],
             inplace=True, errors="ignore")
    res = EXPLOG.update_disposition_info("c", "s", [pid], "f", "Investigating")
    assert res["status"] == "locked"                         # healed, not crashed
