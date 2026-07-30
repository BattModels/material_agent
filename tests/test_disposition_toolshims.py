# Integration tests for the disposition @tool shims + their registration, and the
# per-task enforce_queue_floor flag contract.
#
# SLOW: importing src.tools / src.planNexe2 loads the GNoME database (~110s), so
# this file is intentionally NOT marked `mini` -- run it deliberately, separate
# from the fast nested/formatter suites:
#     python -m pytest tests/test_disposition_toolshims.py -q
#
# Each @tool is a langchain StructuredTool; calling tool.invoke({...}) validates
# the args, runs the wrapped function, and returns its string. These cover the
# parts that only live in src.tools / src.planNexe2 (and so aren't reachable from
# the fast tests): the @tool wiring, the agent-visible column hiding, tool
# registration, and the myStep.enforce_queue_floor contract.

import time
from typing import get_args

import pandas as pd
import yaml

import src.tools as T
import src.planNexe2 as P
from src import var
from gnome_dreams_oer_screening.explog.explog import EXPLOG


# ---------------------------------------------------------------------------
# Helpers (operate on the EXPLOG / CANVAS singletons the tools use)
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")
    EXPLOG.job_handler.disposition_decisions = tuple(var.DISPOSITION_DECISIONS)
    # The wait gates read var.startTime for the deadline-relative windows
    # (remaining time). "Now" = a fresh study, comfortably before every cutoff.
    var.startTime = time.time()


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="tool shim test")


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


# ===========================================================================
# The disposition @tool shims, end-to-end (method -> formatter -> string)
# ===========================================================================

def test_get_tool_shim_sets_ready_and_lists_outstanding(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    # termination 7 is distinctive, so finding it in the message is meaningful.
    _inject("c", "surface_relaxation", "completed", termination_index=7)
    out = T.get_disposition_info.invoke({"candidate_id": "c"})
    assert isinstance(out, str)
    assert "surface relaxation" in out
    assert "termination 7" in out                             # outstanding unit surfaced
    assert _val("c", "ready_for_disposition_update") == True  # noqa: E712


def test_update_tool_shim_records_disposition(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    T.get_disposition_info.invoke({"candidate_id": "c"})
    out = T.update_disposition_info.invoke({
        "candidate_id": "c",
        "Analysis_and_implications": "surface looks promising",
        "Analyzed_process_id": [pid_s],
        "Future_plan": "run OH on the best site",
        "Decision": "Medium priority",
    })
    assert isinstance(out, str)
    recs = _dispositions("c")
    assert len(recs) == 1
    assert recs[0]["Decision"] == "Medium priority"
    assert recs[0]["Summarized_process_id"] == [pid_s]
    assert _val("c", "decision") == "Medium priority"


def test_update_tool_shim_locked_message_when_no_get(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    out = T.update_disposition_info.invoke({          # no get first -> locked
        "candidate_id": "c", "Analysis_and_implications": "x",
        "Analyzed_process_id": [pid_s], "Future_plan": "y",
        "Decision": "Medium priority",
    })
    assert isinstance(out, str) and out
    assert "get_disposition_info" in out              # instructive: how to resolve
    # the rejection must leave ALL state untouched:
    assert _dispositions("c") == []                   # no disposition appended
    assert pd.isna(_val("c", "decision"))             # decision not set
    assert _val("c", "ready_for_disposition_update") == False  # noqa: E712 (lock still closed)


# ---------------------------------------------------------------------------
# Terminal-tag gate wiring (Part 2). The pure rule is unit-tested in
# test_disposition_messages; here we drive the REAL tool so its own computation
# of is_forgotten (via find_forgotten_jobs) / has_in_flight / state is exercised
# end to end, across every branch.
# ---------------------------------------------------------------------------

def _tool_disposition(cid, ids, decision, get_first=True):
    if get_first:
        T.get_disposition_info.invoke({"candidate_id": cid})   # open the write-lock
    return T.update_disposition_info.invoke({
        "candidate_id": cid, "Analysis_and_implications": "x",
        "Analyzed_process_id": list(ids), "Future_plan": "y", "Decision": decision,
    })


def test_gate_rejects_terminal_on_forgotten_bulk(tmp_path):
    # No started work -> forgotten (bulk frontier); Abandon is rejected, nothing
    # is written, and the lock is left OPEN so the agent can retry.
    _setup(tmp_path); _add_candidate("c")
    out = _tool_disposition("c", [], "Abandon")
    assert "Abandon" in out and "settled" in out.lower()
    assert _dispositions("c") == []
    assert pd.isna(_val("c", "decision"))
    assert _val("c", "ready_for_disposition_update") == True  # noqa: E712


def test_gate_rejects_sufficient_on_forgotten_O(tmp_path):
    # bulk+surface finalized, no O started -> forgotten (O frontier); the OTHER
    # terminal tag (Sufficient) is rejected too.
    _setup(tmp_path); _add_candidate("c")
    _inject("c", "bulk_relaxation", "completed")
    ps = _inject("c", "surface_relaxation", "completed", termination_index=0)
    out = _tool_disposition("c", [ps], "Sufficient")
    assert "Sufficient" in out and "settled" in out.lower()
    assert _dispositions("c") == []
    assert _val("c", "ready_for_disposition_update") == True  # noqa: E712


def test_gate_rejects_terminal_while_in_flight(tmp_path):
    # Not forgotten (surface started), but a surface job is RUNNING -> in-flight;
    # the terminal tag is rejected on the has_in_flight branch, not forgotten.
    _setup(tmp_path); _add_candidate("c")
    pb = _inject("c", "bulk_relaxation", "completed")
    _inject("c", "surface_relaxation", "running", termination_index=0)
    out = _tool_disposition("c", [pb], "Abandon")
    assert "settled" in out.lower()
    assert _dispositions("c") == []
    assert pd.isna(_val("c", "decision"))


def test_gate_allows_terminal_when_fully_settled(tmp_path):
    # bulk+surface+O all completed, no competitive-OH gap, nothing in flight ->
    # fully settled: a terminal tag is ALLOWED and written by the engine.
    _setup(tmp_path); _add_candidate("c")
    pb = _inject("c", "bulk_relaxation", "completed")
    ps = _inject("c", "surface_relaxation", "completed", termination_index=0)
    po = _inject("c", "O_adsorption", "completed", termination_index=0, site_index=0)
    out = _tool_disposition("c", [pb, ps, po], "Abandon")
    assert _val("c", "decision") == "Abandon"          # gate allowed -> engine wrote it
    assert _dispositions("c")[-1]["Decision"] == "Abandon"


def test_gate_allows_terminal_after_O_submitted_then_failed(tmp_path):
    # Once an O job has been SUBMITTED the candidate is no longer forgotten at O,
    # even if that O failed -> the tool treats it as settled and the agent is free
    # to mark it terminal (or submit more O). bulk+surface done, one O failed.
    _setup(tmp_path); _add_candidate("c")
    pb = _inject("c", "bulk_relaxation", "completed")
    ps = _inject("c", "surface_relaxation", "completed", termination_index=0)
    po = _inject("c", "O_adsorption", "failed", termination_index=0, site_index=0)
    out = _tool_disposition("c", [pb, ps, po], "Abandon")
    assert _val("c", "decision") == "Abandon"          # gate allowed -> engine wrote it


def test_gate_failed_candidate_rejects_non_abandon(tmp_path):
    # A failed candidate may ONLY be Abandon; an active priority is rejected.
    _setup(tmp_path); _add_candidate("c")
    EXPLOG.mark_candidate_failed("c")
    out = _tool_disposition("c", [], "Medium priority")
    assert "FAILED" in out and "Abandon" in out
    assert _dispositions("c") == []
    assert pd.isna(_val("c", "decision"))
    assert _val("c", "ready_for_disposition_update") == True  # noqa: E712


def test_gate_failed_candidate_allows_abandon(tmp_path):
    # The same failed candidate CAN be Abandoned -> written.
    _setup(tmp_path); _add_candidate("c")
    EXPLOG.mark_candidate_failed("c")
    out = _tool_disposition("c", [], "Abandon")
    assert _val("c", "decision") == "Abandon"
    assert _dispositions("c")[-1]["Decision"] == "Abandon"


def test_gate_allows_active_priority_on_forgotten(tmp_path):
    # An active priority on a forgotten candidate passes the gate and is written
    # (the gate only blocks terminal tags / failed non-Abandon).
    _setup(tmp_path); _add_candidate("c")
    out = _tool_disposition("c", [], "Medium priority")
    assert _val("c", "decision") == "Medium priority"


def test_gate_noop_for_unknown_candidate(tmp_path):
    # The gate's empty-row guard must not crash on an unknown candidate; the
    # engine's own unknown-candidate handling takes over and the tool reports it.
    _setup(tmp_path)                                   # no candidate added
    out = T.update_disposition_info.invoke({
        "candidate_id": "ghost", "Analysis_and_implications": "x",
        "Analyzed_process_id": [], "Future_plan": "y", "Decision": "Abandon",
    })
    assert isinstance(out, str) and out                # no crash
    assert "ghost" in out                              # names the unknown candidate
    assert "experiment log" in out.lower()             # explains it is not logged
    assert "try again" in out.lower()                  # gives a resolution
    assert "Recorded" not in out                       # NOT reported as a success
    assert "Message_ID=" in out                        # tool still registers output
    # the guard must not fabricate a candidate row, and nothing is written:
    assert "ghost" not in EXPLOG.relational_frame.candidates.df["candidate_id"].tolist()
    assert _dispositions("ghost") == []


def test_update_tool_shim_invalid_decision_lists_allowed(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    T.get_disposition_info.invoke({"candidate_id": "c"})
    out = T.update_disposition_info.invoke({
        "candidate_id": "c", "Analysis_and_implications": "x",
        "Analyzed_process_id": [pid_s], "Future_plan": "y",
        "Decision": "Maybe",
    })
    assert all(d in out for d in var.DISPOSITION_DECISIONS)   # built from var.py
    assert _dispositions("c") == []


# ===========================================================================
# Agent-visible column hiding (query_explog / read_explog)
# ===========================================================================

def test_query_explog_hides_internal_columns(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    out = T.query_explog.invoke({"table_name": "candidates", "reason": "check"})
    assert "disposition_record" not in out
    assert "ready_for_disposition_update" not in out
    assert "decision" in out                          # visible
    assert "needs_disposition_update" in out          # visible


def test_read_explog_hides_internal_columns(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    out = T.read_explog.invoke({"candidate_id": "c", "reasons": "check"})
    assert "disposition_record" not in out
    assert "ready_for_disposition_update" not in out


# ===========================================================================
# Tool registration presence (agent tool list + supervisor schema + YAML)
# ===========================================================================

def test_disposition_tools_registered_in_yaml():
    d = yaml.safe_load(open("config/oer_available_tools.yaml"))["OER_Agent"]
    assert "get_disposition_info" in d
    assert "update_disposition_info" in d


def test_disposition_tools_in_mystep_required_tools_literal():
    ann = P.myStep.model_fields["required_tools"].annotation   # List[Literal[...]]
    literal_values = get_args(get_args(ann)[0])
    assert "get_disposition_info" in literal_values
    assert "update_disposition_info" in literal_values


# ===========================================================================
# Deactivated tools (inspect_explog, get_candidate_data) actually absent.
# Functions themselves remain importable (T.inspect_explog, T.get_candidate_data
# still exist -- deactivated, not deleted); these check they're gone from the
# 3 module-level registration surfaces that don't require constructing a real
# agent graph to inspect. NOT covered here (would need to call
# P.create_planning_graph(config), which needs a real LLM config): the
# oer_tools/supervisor_tools lists and the inline ToolList inside
# supervisor_chain_node -- those are local variables inside that function, not
# independently inspectable. If those ever drift from the 3 lists checked
# below, nothing here would catch it.
# ===========================================================================

def test_deactivated_tools_absent_from_yaml():
    d = yaml.safe_load(open("config/oer_available_tools.yaml"))["OER_Agent"]
    assert "inspect_explog" not in d
    assert "get_candidate_data" not in d


def test_deactivated_tools_absent_from_mystep_required_tools_literal():
    ann = P.myStep.model_fields["required_tools"].annotation   # List[Literal[...]]
    literal_values = get_args(get_args(ann)[0])
    assert "inspect_explog" not in literal_values
    assert "get_candidate_data" not in literal_values


def test_deactivated_tools_absent_from_polling_tools():
    assert "inspect_explog" not in P.POLLING_TOOLS
    assert "get_candidate_data" not in P.POLLING_TOOLS


# ===========================================================================
# The per-task enforce_queue_floor flag contract (myStep field + bridge fallback)
# ===========================================================================

def test_mystep_enforce_queue_floor_defaults_true():
    m = P.myStep(step="x", agent="OER_Agent", required_tools=[""])
    assert m.enforce_queue_floor is True


def test_mystep_enforce_queue_floor_explicit_false():
    m = P.myStep(step="x", agent="OER_Agent", required_tools=[""],
                 enforce_queue_floor=False)
    assert m.enforce_queue_floor is False


def test_enforce_queue_floor_bridge_fallback_true():
    # worker_agent_node bridges via getattr(task, "enforce_queue_floor", True):
    # an old plan step lacking the field must default to ENFORCED (resume safety).
    class _OldStep:
        pass
    assert getattr(_OldStep(), "enforce_queue_floor", True) is True


# ===========================================================================
# wait_for_update entry gates, end-to-end (refusals short-circuit before the
# sleep loop, so these exercise the real tool without waiting)
# ===========================================================================

def test_wait_refuses_until_finished_work_is_dispositioned(tmp_path):
    _setup(tmp_path)
    _add_candidate("matX")
    _inject("matX", "surface_relaxation", "completed", termination_index=7)
    out = T.wait_for_update.invoke({"patience": 1})        # Gate 1 fires -> refusal
    assert isinstance(out, str)
    assert "matX" in out                                   # names the candidate
    assert "get_disposition_info" in out and "update_disposition_info" in out


def test_wait_refuses_when_queue_below_floor(tmp_path, monkeypatch):
    _setup(tmp_path)
    var.enforce_queue_floor = True
    monkeypatch.setattr(var, "QUEUE_MIN_PENDING", 12)  # distinctive floor; asserted via var, not a literal
    _add_candidate("matX")
    # bulk+surface finished and dispositioned -> Gate 1 satisfied; O not started
    # (forgotten at O) with a job pending but below the floor -> Gate 2 Path A.
    pb = _inject("matX", "bulk_relaxation", "completed")
    ps = _inject("matX", "surface_relaxation", "completed", termination_index=0)
    T.get_disposition_info.invoke({"candidate_id": "matX"})
    T.update_disposition_info.invoke({
        "candidate_id": "matX", "Analysis_and_implications": "s",
        "Analyzed_process_id": [pb, ps], "Future_plan": "f",
        "Decision": "Medium priority",
    })
    _inject("matX", "surface_relaxation", "pending", termination_index=1)
    out = T.wait_for_update.invoke({"patience": 1})        # Gate 2 fires -> Path A refusal
    assert isinstance(out, str)
    assert "matX" in out                                   # lists matX's forgotten work
    assert "submit" in out.lower()                         # worker submits it itself...
    assert "standing duty" in out.lower()                  # ...as a standing duty
    assert "call wait_for_update again" in out.lower()     # then re-waits
    assert f"hard floor: {var.QUEUE_MIN_PENDING}" in out   # real numbers shown


def test_wait_leaves_handback_flag_false_on_path_a(tmp_path, monkeypatch):
    # A Path A refusal (ready work listed) is now the WORKER's own job -- it
    # submits the jobs itself and re-calls wait -- so the supervisor-handback
    # flag must stay False.
    _setup(tmp_path)
    var.enforce_queue_floor = True
    monkeypatch.setattr(var, "QUEUE_MIN_PENDING", 12)
    var.wait_handback = False
    _add_candidate("matX")
    pb = _inject("matX", "bulk_relaxation", "completed")
    ps = _inject("matX", "surface_relaxation", "completed", termination_index=0)
    T.get_disposition_info.invoke({"candidate_id": "matX"})
    T.update_disposition_info.invoke({
        "candidate_id": "matX", "Analysis_and_implications": "s",
        "Analyzed_process_id": [pb, ps], "Future_plan": "f",
        "Decision": "Medium priority",
    })
    _inject("matX", "surface_relaxation", "pending", termination_index=1)
    T.wait_for_update.invoke({"patience": 1})              # Gate 2 -> Path A refusal
    assert var.wait_handback is False


def test_wait_sets_handback_flag_on_expand_path(tmp_path, monkeypatch):
    # A Path B refusal (queue below floor, NOTHING ready to submit) is the
    # supervisor's to resolve -> the one-shot handback flag must be raised so
    # supervisor_chain_node injects the expand directive on its next turn.
    _setup(tmp_path)
    var.enforce_queue_floor = True
    monkeypatch.setattr(var, "QUEUE_MIN_PENDING", 12)
    var.wait_handback = False
    _add_candidate("matX")
    pb = _inject("matX", "bulk_relaxation", "completed")
    T.get_disposition_info.invoke({"candidate_id": "matX"})
    T.update_disposition_info.invoke({
        "candidate_id": "matX", "Analysis_and_implications": "s",
        "Analyzed_process_id": [pb], "Future_plan": "f",
        "Decision": "Medium priority",
    })
    # Surface already STARTED (pending) -> no forgotten frontier work; queue of 1
    # is below the floor of 12 -> Path B (expand) -> flag raised.
    _inject("matX", "surface_relaxation", "pending", termination_index=1)
    out = T.wait_for_update.invoke({"patience": 1})        # Gate 2 -> Path B refusal
    assert "supervisor" in out.lower()
    assert var.wait_handback is True


def test_wait_leaves_handback_flag_false_on_disposition_backlog(tmp_path):
    # A Gate 1 disposition backlog is the worker's own job, not a supervisor
    # handback -> the flag must stay False (classify_wait_handback returns None).
    _setup(tmp_path)
    var.wait_handback = False
    _add_candidate("matX")
    _inject("matX", "surface_relaxation", "completed", termination_index=7)
    T.wait_for_update.invoke({"patience": 1})              # Gate 1 -> refusal
    assert var.wait_handback is False
