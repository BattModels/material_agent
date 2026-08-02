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

from src import var
from src.forgotten_jobs import find_forgotten_jobs
from src.disposition_messages import (
    MSG_NOTHING_TO_WAIT_FOR,
    MSG_NOTHING_TO_WAIT_FOR_FINAL_DAYS,
    classify_wait_handback,
    evaluate_wait_entry,
    format_wait_exit_disposition_hint,
    format_wait_gate1_refusal,
    format_wait_gate2_refusal,
)

# Remaining-time inputs for the deadline-relative gates: comfortably before the
# Path B cutoff, and inside the final-days window, respectively.
_REMAIN_MID = var.PATH_B_CUTOFF_SECONDS + 86400
_REMAIN_FINAL = var.PATH_B_CUTOFF_SECONDS - 86400


# ---------------------------------------------------------------------------
# Helpers (operate on the EXPLOG singleton the wait tool uses)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="wait gate test")


def _inject(cid, job_type, status, termination_index=None, site_index=None,
            go_dev=None):
    table = EXPLOG.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    row = {"process_id": pid, "candidate_id": cid, "job_type": job_type,
           "status": status, "termination_index": termination_index,
           "site_index": site_index}
    if go_dev is not None:
        row["G(O) deviation"] = go_dev          # competitive-site signal for OH
    table.add_row(row, allow_update=False)
    return pid


def _update(cid, ids, decision="Medium priority"):
    EXPLOG.job_handler.disposition_decisions = ("Abandon", "Low priority",
                                                "Medium priority", "High priority",
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


def test_gate1_refusal_caps_long_candidate_list():
    # A big backlog (e.g. first post-rollout resume) must not dump every id into
    # one message: show at most 10 + a "(and N more)" remainder.
    cands = [f"mat{i}" for i in range(12)]
    msg = format_wait_gate1_refusal(cands)
    assert "mat0" in msg and "mat9" in msg          # first 10 shown
    assert "mat10" not in msg and "mat11" not in msg  # capped out
    assert "2 more" in msg


def test_gate2_path_b_routes_to_supervisor_with_numbers_and_refill_target():
    # Path B (no ready work): hand back to the supervisor to discuss expansion.
    # The REAL numbers are now shown (the old opacity taught the agents to
    # distrust and disarm the gate), plus the refill goal and the disarm window.
    msg = format_wait_gate2_refusal(running_count=41, pending_count=3)
    assert "supervisor" in msg.lower()               # the return-to-supervisor route
    assert "expand" in msg.lower()                   # discuss expansion
    assert "41 running" in msg and "3 queued" in msg  # real numbers shown
    assert f"floor: {var.QUEUE_MIN_PENDING}" in msg  # the floor itself
    assert f"~{var.QUEUE_REFILL_TARGET}" in msg      # refill goal, beyond the floor
    assert "enforce_queue_floor" in msg              # wind-down option ...
    assert f"final {var.FLOOR_DISARM_WINDOW_DAYS} days" in msg  # ... and its window


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
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID)
    assert msg == format_wait_gate1_refusal(need)


def test_evaluate_gate2_when_current_and_queue_low():
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=5,
                              running_count=1, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID)
    assert msg == format_wait_gate2_refusal(running_count=1, pending_count=5)


def test_evaluate_floor_off_when_enforce_false():
    # enforce_queue_floor False -> low queue does NOT block (work in flight).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=2,
                              running_count=1, enforce_queue_floor=False,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID)
    assert msg is None


def test_evaluate_queue_min_zero_is_hard_off_switch():
    # QUEUE_MIN_PENDING <= 0 disables Gate 2 even with enforce True; with nothing
    # in flight we fall through to the nothing-to-wait-for message (not Gate 2).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=0, remaining_seconds=_REMAIN_MID)
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_nothing_to_wait_for():
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              running_count=0, enforce_queue_floor=False,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID)
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_idle_final_days_steers_to_report():
    # Idle inside the final PATH_B_CUTOFF_DAYS: a fresh batch could not finish,
    # so the idle worker is steered to finalize-and-report, not expansion.
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_FINAL)
    assert msg == MSG_NOTHING_TO_WAIT_FOR_FINAL_DAYS


def test_evaluate_proceeds_when_current_and_queue_full():
    # analysis current, queue at/above floor, work pending -> proceed (None).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=20,
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID)
    assert msg is None


def test_evaluate_nothing_in_flight_no_work_is_not_pushed_to_submit():
    # Nothing pending/running AND no detectable ready work -> "nothing to wait
    # for", NOT the Gate 2 push, even with the floor on (a genuinely-idle/done
    # worker should not be told to submit a batch it cannot see).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                              forgotten_jobs=[])
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_nothing_in_flight_but_forgotten_work_lists_it():
    # Drained queue but detectable ready work -> still surface the actionable
    # forgotten-jobs list rather than the terse nothing-to-wait message.
    jobs = [{"candidate_id": "matZ", "kind": "bulk",
             "termination_index": None, "site_index": None}]
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              running_count=0, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                              forgotten_jobs=jobs)
    assert msg == format_wait_gate2_refusal(jobs, running_count=0, pending_count=0)
    assert "matZ" in msg


def test_evaluate_path_b_disabled_in_final_days_proceeds():
    # Armed floor, work in flight, NOTHING ready, final days -> the wait simply
    # proceeds (no pointless expansion demand while winding down).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=5,
                              running_count=9, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_FINAL,
                              forgotten_jobs=[])
    assert msg is None


def test_evaluate_path_a_still_fires_in_final_days():
    # Path A (ready work exists) is NOT time-gated: while the floor is armed it
    # always fires -- short continuation jobs can still finish in the final days.
    jobs = [{"candidate_id": "matZ", "kind": "bulk",
             "termination_index": None, "site_index": None}]
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=5,
                              running_count=9, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_FINAL,
                              forgotten_jobs=jobs)
    assert msg == format_wait_gate2_refusal(jobs, running_count=9, pending_count=5)


# ===========================================================================
# Supervisor handback path: classify_wait_handback (the flag wait_for_update
# raises, re-derived at supervisor time). Kept in lock-step with
# evaluate_wait_entry. Only "expand" (Path B / idle) is a supervisor handback;
# a Path A refusal (ready work listed) is the WORKER's own job now -- it
# submits the jobs itself and re-calls wait -- so classify returns None for it.
# ===========================================================================

_HB_JOBS = [{"candidate_id": "matZ", "kind": "bulk",
             "termination_index": None, "site_index": None}]


def test_classify_gate1_backlog_is_not_a_handback():
    # A disposition backlog is the worker's own job, not a supervisor handback.
    path = classify_wait_handback(candidates_need_disposition=["matA"], pending_count=0,
                                  running_count=0, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=_HB_JOBS)
    assert path is None


def test_classify_gate2_with_ready_work_is_not_a_handback():
    # Queue below floor AND ready work -> Path A: the worker submits the listed
    # jobs itself (standing duty) -- no supervisor handback.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=5,
                                  running_count=1, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=_HB_JOBS)
    assert path is None


def test_classify_gate2_without_ready_work_is_expand():
    # Queue below floor, work in flight but NOTHING ready to submit -> Path B.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=5,
                                  running_count=1, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=[])
    assert path == "expand"


def test_classify_gate2_without_ready_work_final_days_is_none():
    # Path B is disabled in the final PATH_B_CUTOFF_DAYS -> the wait proceeds,
    # so there is nothing for the supervisor to resolve.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=5,
                                  running_count=1, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_FINAL,
                                  forgotten_jobs=[])
    assert path is None


def test_classify_idle_no_work_is_expand():
    # Nothing pending/running and nothing ready -> idle handback -> Path B
    # (in the final days the worker is steered to finalize-and-report, but the
    # supervisor still has to resolve it -> still "expand").
    for remaining in (_REMAIN_MID, _REMAIN_FINAL):
        path = classify_wait_handback(candidates_need_disposition=[], pending_count=0,
                                      running_count=0, enforce_queue_floor=True,
                                      queue_min_pending=15,
                                      remaining_seconds=remaining, forgotten_jobs=[])
        assert path == "expand"


def test_classify_idle_with_ready_work_is_not_a_handback():
    # Idle but detectable ready work + floor armed -> Path A: worker self-serves.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=0,
                                  running_count=0, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=_HB_JOBS)
    assert path is None


def test_classify_idle_is_expand_even_with_floor_off():
    # A genuinely idle worker still hands back (expand/conclude) even if the floor
    # is disarmed -- the idle branch does not depend on gate2_armed.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=0,
                                  running_count=0, enforce_queue_floor=False,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=[])
    assert path == "expand"


def test_classify_proceeds_none_when_queue_healthy():
    # Analysis current, queue at/above floor, work pending -> proceed, no handback.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=20,
                                  running_count=0, enforce_queue_floor=True,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=_HB_JOBS)
    assert path is None


def test_classify_none_when_floor_off_and_work_in_flight():
    # Floor disarmed with work in flight -> proceed to wait, not a handback.
    path = classify_wait_handback(candidates_need_disposition=[], pending_count=2,
                                  running_count=1, enforce_queue_floor=False,
                                  queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                                  forgotten_jobs=_HB_JOBS)
    assert path is None


def test_classify_parity_with_evaluate_wait_entry():
    # Lock-step invariants across a grid, guarding the two fns drifting apart:
    #   - classify never returns the retired "submit_ready";
    #   - "expand" is only raised for a refusal the SUPERVISOR must resolve
    #     (so never on proceed and never on a Gate 1 backlog);
    #   - a Path A refusal (message tells the worker to re-call wait after
    #     submitting) is never a handback.
    grid = [
        (["matA"], 0, 0, True, 15, _REMAIN_MID, _HB_JOBS),   # gate1
        ([], 5, 1, True, 15, _REMAIN_MID, _HB_JOBS),         # gate2 + jobs (Path A)
        ([], 5, 1, True, 15, _REMAIN_FINAL, _HB_JOBS),       # Path A, final days
        ([], 5, 1, True, 15, _REMAIN_MID, []),               # gate2 no jobs (Path B)
        ([], 5, 1, True, 15, _REMAIN_FINAL, []),             # Path B off -> proceed
        ([], 0, 0, True, 15, _REMAIN_MID, []),               # idle no jobs
        ([], 0, 0, True, 15, _REMAIN_FINAL, []),             # idle, final days
        ([], 0, 0, True, 15, _REMAIN_MID, _HB_JOBS),         # idle + jobs (Path A)
        ([], 0, 0, False, 15, _REMAIN_MID, []),              # idle floor-off
        ([], 20, 0, True, 15, _REMAIN_MID, _HB_JOBS),        # healthy queue -> proceed
        ([], 2, 1, False, 15, _REMAIN_MID, _HB_JOBS),        # floor off, in flight
    ]
    for need, pending, running, floor, floor_min, remaining, jobs in grid:
        kw = dict(candidates_need_disposition=need, pending_count=pending,
                  running_count=running, enforce_queue_floor=floor,
                  queue_min_pending=floor_min, remaining_seconds=remaining,
                  forgotten_jobs=jobs)
        msg = evaluate_wait_entry(**kw)
        path = classify_wait_handback(**kw)
        assert path in (None, "expand"), (kw, msg, path)
        if msg is None or bool(need):
            assert path is None, (kw, msg, path)
        if msg is not None and "call wait_for_update again" in msg:
            assert path is None, (kw, msg, path)   # Path A: worker self-serves


# ===========================================================================
# Forgotten-jobs detection: find_forgotten_jobs (parent src.forgotten_jobs,
# dataframe-only over EXPLOG)
#
# Each item is {candidate_id, kind in {bulk,surface,O,OH}, termination_index,
# site_index}. Categories 1-3 are the per-candidate frontier (mutually
# exclusive); category 4 (OH) is one item per competitive O site without OH.
# ===========================================================================

THR = 0.3        # GO_DEV_OH_THRESHOLD used in these tests


def _kinds(items):
    return [(i["candidate_id"], i["kind"], i["termination_index"],
             i["site_index"]) for i in items]


def test_forgotten_bulk_when_no_bulk_started(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    assert _kinds(find_forgotten_jobs(EXPLOG, THR)) == \
        [("c", "bulk", None, None)]


def test_forgotten_surface_when_bulk_finalized_no_surface(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    assert _kinds(find_forgotten_jobs(EXPLOG, THR)) == \
        [("c", "surface", None, None)]


def test_forgotten_O_when_surface_finalized_no_O(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    assert _kinds(find_forgotten_jobs(EXPLOG, THR)) == \
        [("c", "O", None, None)]


def test_forgotten_OH_for_competitive_O_site_without_OH(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    _inject("c", "O_adsorption", "completed", termination_index=0, site_index=2,
            go_dev=0.1)                                   # |0.1| < 0.3 -> competitive
    assert _kinds(find_forgotten_jobs(EXPLOG, THR)) == \
        [("c", "OH", 0, 2)]


def test_no_forgotten_OH_when_O_not_competitive(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    _inject("c", "O_adsorption", "completed", termination_index=0, site_index=2,
            go_dev=0.9)                                   # |0.9| >= 0.3
    assert find_forgotten_jobs(EXPLOG, THR) == []


def test_no_forgotten_OH_when_oh_already_present(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    _inject("c", "O_adsorption", "completed", termination_index=0, site_index=2,
            go_dev=0.1)
    _inject("c", "OH_adsorption", "pending", termination_index=0, site_index=2)
    assert find_forgotten_jobs(EXPLOG, THR) == []   # OH already at (0,2)


def test_negative_threshold_disables_oh_detection(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    _inject("c", "O_adsorption", "completed", termination_index=0, site_index=2,
            go_dev=0.1)
    assert find_forgotten_jobs(EXPLOG, -1) == []    # OH category off


def test_failed_bulk_candidate_is_not_forgotten(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "failed")            # no winner -> dead end
    assert find_forgotten_jobs(EXPLOG, THR) == []


def test_not_forgotten_after_O_submitted_even_if_failed(tmp_path):
    # Once an O job is submitted the candidate is no longer 'forgotten at O',
    # even if that O FAILED -> the agent is free to submit more O or mark the
    # candidate terminal; a failed O does not re-trap it as ready work.
    _setup(tmp_path)
    _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "completed", termination_index=0)
    _inject("c", "O_adsorption", "failed", termination_index=0, site_index=0)
    assert find_forgotten_jobs(EXPLOG, THR) == []


def test_norow_failed_candidate_is_excluded_from_forgotten(tmp_path):
    # A candidate whose bulk magnetic enumeration failed has NO process rows -- it would
    # otherwise be flagged forever ("start the bulk relaxation"). Marking it state='failed'
    # (what mark_candidate_failed does) must exclude it from the forgotten-jobs hint.
    _setup(tmp_path)
    _add_candidate("c")                                  # no process rows
    assert any(it["candidate_id"] == "c"                 # fresh -> flagged for bulk
               for it in find_forgotten_jobs(EXPLOG, THR))
    EXPLOG.relational_frame.candidates.set_value(
        "c", "state", "failed", allow_new_columns=True)
    assert all(it["candidate_id"] != "c"                 # marked failed -> excluded
               for it in find_forgotten_jobs(EXPLOG, THR))


# ===========================================================================
# Gate 2 message with the forgotten-jobs hint (pure formatter)
# ===========================================================================

def _fitem(cid, kind, t=None, s=None):
    return {"candidate_id": cid, "kind": kind, "termination_index": t,
            "site_index": s}


def test_gate2_path_a_orders_self_service_submit():
    jobs = [_fitem("matA", "bulk"), _fitem("matB", "O")]
    msg = format_wait_gate2_refusal(jobs, running_count=7, pending_count=4)
    assert "matA" in msg and "matB" in msg               # lists the ready jobs
    assert "submit" in msg.lower()                       # the worker submits...
    assert "standing duty" in msg.lower()                # ...under its own task
    assert "call wait_for_update again" in msg.lower()   # then re-waits
    assert "end your turn" not in msg.lower()            # no supervisor round-trip
    assert "7 running" in msg and "only 4 queued" in msg  # real numbers shown
    # deficit is derived from the floor, so read it from var rather than pinning
    # the arithmetic of whatever QUEUE_MIN_PENDING happened to be when this was
    # written (it was 15; tuning it to 25 broke this line, not the code).
    assert f"{var.QUEUE_MIN_PENDING - 4} more queued" in msg
    assert f"~{var.QUEUE_REFILL_TARGET}" in msg          # refill well beyond the floor


def test_gate2_caps_at_ten_and_reports_remainder():
    jobs = [_fitem(f"mat{i}", "bulk") for i in range(12)]
    msg = format_wait_gate2_refusal(jobs, running_count=0, pending_count=0)
    assert "mat0" in msg and "mat9" in msg               # first 10 shown
    assert "mat10" not in msg and "mat11" not in msg     # capped out
    assert "2 more" in msg


def test_gate2_path_a_refusal_is_unconditional():
    # Path A always fires while the floor is armed, regardless of how many jobs
    # are waiting (the old FORGOTTEN_CLOSER_SUPPRESS_ABOVE conditional is gone).
    jobs = [_fitem(f"mat{i}", "bulk") for i in range(35)]
    msg = format_wait_gate2_refusal(jobs, running_count=0, pending_count=0)
    assert "call wait_for_update again" in msg.lower()   # self-service instruction
    assert "25 more" in msg                              # 35 - 10 shown


def test_gate2_path_b_names_expansion_discussion_points():
    msg = format_wait_gate2_refusal([], running_count=1, pending_count=5)
    assert "end your turn" in msg.lower()               # stop, hand back
    assert "supervisor" in msg.lower()
    assert "candidates" in msg.lower()                  # discuss candidates for more calcs
    assert "literature" in msg.lower()                  # ground in literature


def test_nothing_to_wait_for_hands_back_to_discuss_expansion():
    m = MSG_NOTHING_TO_WAIT_FOR
    assert "end your turn" in m.lower()                 # stop polling, hand back
    assert "supervisor" in m.lower()
    assert "candidates" in m.lower()
    assert "literature" in m.lower()
    assert "large batch" in m.lower()                   # target a large batch
    assert str(var.QUEUE_REFILL_TARGET) in m            # sized by the refill goal


def test_nothing_to_wait_for_final_days_steers_to_report():
    m = MSG_NOTHING_TO_WAIT_FOR_FINAL_DAYS
    assert "end your turn" in m.lower()                 # stop polling, hand back
    assert "supervisor" in m.lower()
    assert "final report" in m.lower()                  # finalize, don't expand
    assert "large batch" not in m.lower()


def test_evaluate_forwards_forgotten_jobs_to_gate2():
    jobs = [_fitem("matZ", "surface")]
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=3,
                              running_count=1, enforce_queue_floor=True,
                              queue_min_pending=15, remaining_seconds=_REMAIN_MID,
                              forgotten_jobs=jobs)
    assert msg == format_wait_gate2_refusal(jobs, running_count=1, pending_count=3)
    assert "matZ" in msg
