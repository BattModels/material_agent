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
        "Decision": "Investigating",
    })
    assert isinstance(out, str)
    recs = _dispositions("c")
    assert len(recs) == 1
    assert recs[0]["Decision"] == "Investigating"
    assert recs[0]["Summarized_process_id"] == [pid_s]
    assert _val("c", "decision") == "Investigating"


def test_update_tool_shim_locked_message_when_no_get(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    out = T.update_disposition_info.invoke({          # no get first -> locked
        "candidate_id": "c", "Analysis_and_implications": "x",
        "Analyzed_process_id": [pid_s], "Future_plan": "y",
        "Decision": "Investigating",
    })
    assert isinstance(out, str) and out
    assert "get_disposition_info" in out              # instructive: how to resolve
    # the rejection must leave ALL state untouched:
    assert _dispositions("c") == []                   # no disposition appended
    assert pd.isna(_val("c", "decision"))             # decision not set
    assert _val("c", "ready_for_disposition_update") == False  # noqa: E712 (lock still closed)


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


def test_wait_refuses_when_queue_below_floor(tmp_path):
    _setup(tmp_path)
    var.enforce_queue_floor = True
    var.QUEUE_MIN_PENDING = 12        # distinctive floor; asserted via var, not a literal
    _add_candidate("matX")
    # finished work, fully dispositioned -> Gate 1 satisfied; queue then below floor.
    pid = _inject("matX", "surface_relaxation", "completed", termination_index=0)
    T.get_disposition_info.invoke({"candidate_id": "matX"})
    T.update_disposition_info.invoke({
        "candidate_id": "matX", "Analysis_and_implications": "s",
        "Analyzed_process_id": [pid], "Future_plan": "f",
        "Decision": "Investigating",
    })
    _inject("matX", "surface_relaxation", "pending", termination_index=1)
    out = T.wait_for_update.invoke({"patience": 1})        # Gate 2 fires -> refusal
    assert isinstance(out, str)
    assert "queue is running low" in out.lower()           # the Gate 2 header
    assert "matX" in out                                   # lists matX's forgotten work
    assert "supervisor" in out.lower()                     # routes back to supervisor
