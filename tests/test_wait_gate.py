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

from src.forgotten_jobs import find_forgotten_jobs
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


def test_gate2_path_b_routes_to_supervisor_and_targets_a_large_batch():
    # Path B (no ready work): hand back to the supervisor to discuss expansion,
    # suggesting a large batch (~40-50) as the target. Anti-gaming -> a batch
    # TARGET is fine, but never a pending count / floor / headroom (gameable deficit).
    msg = format_wait_gate2_refusal()
    assert "supervisor" in msg.lower()         # the return-to-supervisor route
    assert "expand" in msg.lower()             # discuss expansion
    assert "large batch" in msg.lower()        # aim for a large batch
    assert "40-50" in msg                      # a reasonable target, not a deficit
    assert "enforce_queue_floor" in msg        # wind-down option
    assert "headroom" not in msg.lower()       # never a gameable deficit


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
    assert msg == format_wait_gate2_refusal()   # numbers drive the decision, not the prose


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


def test_evaluate_nothing_in_flight_no_work_is_not_pushed_to_submit():
    # Nothing pending/running AND no detectable ready work -> "nothing to wait
    # for", NOT the Gate 2 "submit a large batch" push, even with the floor on
    # (a genuinely-idle/done worker should not be told to submit 40-50 jobs).
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              has_running=False, enforce_queue_floor=True,
                              queue_min_pending=15, forgotten_jobs=[])
    assert msg == MSG_NOTHING_TO_WAIT_FOR


def test_evaluate_nothing_in_flight_but_forgotten_work_lists_it():
    # Drained queue but detectable ready work -> still surface the actionable
    # forgotten-jobs list rather than the terse nothing-to-wait message.
    jobs = [{"candidate_id": "matZ", "kind": "bulk",
             "termination_index": None, "site_index": None}]
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=0,
                              has_running=False, enforce_queue_floor=True,
                              queue_min_pending=15, forgotten_jobs=jobs)
    assert msg == format_wait_gate2_refusal(jobs)
    assert "matZ" in msg


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


def test_gate2_path_a_hands_back_requesting_submit():
    jobs = [_fitem("matA", "bulk"), _fitem("matB", "O")]
    msg = format_wait_gate2_refusal(jobs)
    assert "matA" in msg and "matB" in msg               # lists the ready jobs
    assert "supervisor" in msg.lower()                   # return to the supervisor
    assert "end your turn" in msg.lower()                # stop, hand back
    assert "submit" in msg.lower()                       # request a submit step
    assert "yourself" in msg.lower()                     # do NOT submit it yourself
    assert "large batch" not in msg.lower()              # old self-submit push gone


def test_gate2_caps_at_ten_and_reports_remainder():
    jobs = [_fitem(f"mat{i}", "bulk") for i in range(12)]
    msg = format_wait_gate2_refusal(jobs)
    assert "mat0" in msg and "mat9" in msg               # first 10 shown
    assert "mat10" not in msg and "mat11" not in msg     # capped out
    assert "2 more" in msg
    assert "once these are queued" in msg.lower()


def test_gate2_path_a_handback_is_unconditional():
    # Path A now ALWAYS hands the worker back to the supervisor to request a submit
    # step, regardless of how many jobs are waiting (the old
    # FORGOTTEN_CLOSER_SUPPRESS_ABOVE conditional is gone).
    jobs = [_fitem(f"mat{i}", "bulk") for i in range(35)]
    msg = format_wait_gate2_refusal(jobs)
    assert "supervisor" in msg.lower()                   # handback present
    assert "25 more" in msg                              # 35 - 10 shown


def test_gate2_path_b_names_expansion_discussion_points():
    msg = format_wait_gate2_refusal([])
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


def test_evaluate_forwards_forgotten_jobs_to_gate2():
    jobs = [_fitem("matZ", "surface")]
    msg = evaluate_wait_entry(candidates_need_disposition=[], pending_count=3,
                              has_running=True, enforce_queue_floor=True,
                              queue_min_pending=15, forgotten_jobs=jobs)
    assert msg == format_wait_gate2_refusal(jobs)
    assert "matZ" in msg
