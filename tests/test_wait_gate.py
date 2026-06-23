# Tests for the wait_for_update gates + batch-aware exit, test-FIRST (TDD).
#
# wait_for_update itself (in src.tools) is a sleep/poll loop that can't be unit
# tested, so its decision logic is extracted into testable pieces that this file
# exercises WITHOUT importing src.tools (which loads the GNoME database, ~109s):
#
#   nested engine (via the installed gnome package -- a fast import):
#       EXPLOG.candidates_needing_disposition() -> list[str]   # Gate 1 input
#       EXPLOG.job_handler.count_pending()       -> int        # Gate 2 input
#       EXPLOG.job_handler.finalized_exit_ids(ids) -> set[int] # batch-aware exit
#   pure decision + prose (via src.disposition_messages, stdlib only):
#       evaluate_wait_entry(...) -> str | None  (None == proceed to wait)
#       format_wait_gate1_refusal / _gate2_refusal / _exit_disposition_hint
#       MSG_NOTHING_TO_WAIT_FOR
#
# Distinctive ids/values (termination 7, pending=5, floor=15) are fed in, so
# asserting they surface genuinely checks rendering (not a coincidental "0").

from gnome_dreams_oer_screening.explog.explog import EXPLOG

from src.disposition_messages import (
    MSG_NOTHING_TO_WAIT_FOR,
    evaluate_wait_entry,
    format_wait_exit_disposition_hint,
    format_wait_gate1_refusal,
    format_wait_gate2_refusal,
)


# ---------------------------------------------------------------------------
# Helpers (operate on the EXPLOG singleton the wait tool uses)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="wait gate test")


def _inject(cid, job_type, status, termination_index=None, site_index=None):
    table = EXPLOG.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row({"process_id": pid, "candidate_id": cid, "job_type": job_type,
                   "status": status, "termination_index": termination_index,
                   "site_index": site_index}, allow_update=False)
    return pid


def _update(cid, ids, decision="Investigating"):
    EXPLOG.job_handler.disposition_decisions = ("Abandon", "Investigating",
                                                "Sufficient")
    EXPLOG.get_disposition_info(cid)
    return EXPLOG.update_disposition_info(cid, "s", list(ids), "f", decision)


# ===========================================================================
# Gate 1 input: candidates_needing_disposition (nested engine)
# ===========================================================================

def test_needing_disposition_lists_candidate_with_uncited_finalized(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    assert EXPLOG.candidates_needing_disposition() == ["c"]


def test_needing_disposition_empty_after_full_disposition(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid = _inject("c", "surface_relaxation", "completed", termination_index=0)
    _update("c", [pid])
    assert EXPLOG.candidates_needing_disposition() == []


def test_needing_disposition_ignores_only_running_work(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "O_adsorption", "running", termination_index=0, site_index=0)
    assert EXPLOG.candidates_needing_disposition() == []   # nothing finalized yet


# ===========================================================================
# Gate 2 input: count_pending (nested engine)
# ===========================================================================

def test_count_pending_counts_only_pending(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "surface_relaxation", "pending", termination_index=0)
    _inject("c", "surface_relaxation", "pending", termination_index=1)
    _inject("c", "surface_relaxation", "running", termination_index=2)
    _inject("c", "surface_relaxation", "completed", termination_index=3)
    assert EXPLOG.job_handler.count_pending() == 2


# ===========================================================================
# Batch-aware exit: finalized_exit_ids (nested engine)
#
# Contract: given the process ids that just reached a terminal status this poll,
# return the FULL id set of every finalized batch unit they belong to. A
# surface/O id maps to itself; ANY one terminal sub-job of a fully-terminal
# bulk/OH batch expands to that whole unit's ids (so the exit report is complete
# regardless of which poll finished which sub-job); a sub-job of a batch that is
# NOT yet fully terminal contributes nothing -> the wait continues.
# ===========================================================================

def test_exit_ids_surface_always_counts(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid = _inject("c", "surface_relaxation", "completed", termination_index=7)
    assert EXPLOG.job_handler.finalized_exit_ids([pid]) == {pid}


def test_exit_ids_partial_bulk_batch_excluded(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    p_done = _inject("c", "bulk_relaxation", "completed")
    _inject("c", "bulk_relaxation", "running")           # batch NOT terminal
    assert EXPLOG.job_handler.finalized_exit_ids([p_done]) == set()


def test_exit_ids_full_bulk_batch_expands_to_whole_unit(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    p0 = _inject("c", "bulk_relaxation", "completed")
    p1 = _inject("c", "bulk_relaxation", "failed")       # all terminal -> finalized
    # one transitioned sub-job returns the WHOLE finalized batch unit:
    assert EXPLOG.job_handler.finalized_exit_ids([p0]) == {p0, p1}


def test_exit_ids_partial_oh_batch_excluded(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    a = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    b = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    _inject("c", "OH_adsorption", "running", termination_index=0, site_index=0)
    assert EXPLOG.job_handler.finalized_exit_ids([a, b]) == set()


def test_exit_ids_full_oh_batch_expands_to_whole_unit(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    a = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    b = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    cc = _inject("c", "OH_adsorption", "completed", termination_index=0, site_index=0)
    # one transitioned sub-job unlocks the whole batch unit (all 3 ids):
    assert EXPLOG.job_handler.finalized_exit_ids([a]) == {a, b, cc}


# ===========================================================================
# Pure prose formatters
# ===========================================================================

def test_gate1_refusal_names_candidates_and_both_tools():
    msg = format_wait_gate1_refusal(["matA", "matB"])
    assert "matA" in msg and "matB" in msg
    assert "get_disposition_info" in msg
    assert "update_disposition_info" in msg


def test_gate2_refusal_states_pending_floor_and_routes_to_supervisor():
    msg = format_wait_gate2_refusal(5, 15)
    assert "5" in msg and "15" in msg
    assert "supervisor" in msg.lower()        # the return-to-supervisor route


def test_exit_hint_names_candidates_else_empty():
    msg = format_wait_exit_disposition_hint(["matA"])
    assert "matA" in msg
    assert "get_disposition_info" in msg
    assert format_wait_exit_disposition_hint([]) == ""   # nothing finalized -> silent


# ===========================================================================
# Pure decision: evaluate_wait_entry (precedence + off-switches)
# ===========================================================================

def test_evaluate_gate1_precedes_gate2():
    # candidates need disposition AND queue is low -> Gate 1 (analysis) wins.
    need = ["matA"]
    msg = evaluate_wait_entry(candidates_need_disposition=need, pending_count=0,
                              has_running=False, enforce_queue_floor=True,
                              queue_min_pending=15)
    assert msg == format_wait_gate1_refusal(need)


def test_evaluate_gate2_when_current_and_queue_low():
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=5,
                              has_running=True, enforce_queue_floor=True,
                              queue_min_pending=15)
    assert msg == format_wait_gate2_refusal(5, 15)


def test_evaluate_floor_off_when_enforce_false():
    # enforce_queue_floor False -> low queue does NOT block (work in flight).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=2,
                              has_running=True, enforce_queue_floor=False,
                              queue_min_pending=15)
    assert msg is None


def test_evaluate_queue_min_zero_is_hard_off_switch():
    # QUEUE_MIN_PENDING <= 0 disables Gate 2 even with enforce True; with nothing
    # in flight we fall through to the nothing-to-wait-for message (not Gate 2).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              has_running=False, enforce_queue_floor=True,
                              queue_min_pending=0)
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_nothing_to_wait_for():
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              has_running=False, enforce_queue_floor=False,
                              queue_min_pending=15)
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_proceeds_when_current_and_queue_full():
    # analysis current, queue at/above floor, work pending -> proceed (None).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=20,
                              has_running=False, enforce_queue_floor=True,
                              queue_min_pending=15)
    assert msg is None
