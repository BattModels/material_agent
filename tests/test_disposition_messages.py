# Unit tests for the disposition message formatter (Step 3), test-FIRST (TDD).
#
# src/disposition_messages.py holds PURE functions that turn the structured
# result of EXPLOG.get/update_disposition_info into the agent-facing sentence.
# The parent @tool in src/tools.py is then just: call the EXPLOG method, pass the
# result through the formatter, return the string. Keeping the prose here (a) lets
# us unit-test that rejection messages state a reason AND how to resolve, and
# (b) avoids importing src.tools (which loads the GNoME database).
#
# Functions under test (do not exist yet -> RED):
#   format_get_disposition(candidate_id, outstanding) -> str
#   format_update_disposition(candidate_id, result, allowed_decisions) -> str
#
# Distinctive ids (42, 99) are fed as inputs, so asserting they appear in the
# message genuinely checks the formatter renders them (unlike a bare "0").

from src.disposition_messages import (
    format_get_disposition,
    format_update_disposition,
    format_supervisor_handback_directive,
    evaluate_terminal_tag_gate,
)
from src import var

ALLOWED = ("Abandon", "Low priority", "Medium priority", "High priority", "Sufficient")


def test_decision_vocab_partition():
    # Step 1 (TDD): the new vocabulary partitions cleanly into terminal + active,
    # with a neutral active default.
    terminal = set(var.DISPOSITION_TERMINAL_DECISIONS)
    active = set(var.DISPOSITION_ACTIVE_DECISIONS)
    assert terminal | active == set(var.DISPOSITION_DECISIONS)
    assert terminal.isdisjoint(active)
    assert var.DISPOSITION_DEFAULT_ACTIVE in active


# ---------------------------------------------------------------------------
# evaluate_terminal_tag_gate (Step 2): terminal tags require a settled candidate;
# a failed candidate may only be Abandon. Each rejection must also state a reason
# and how to resolve it, so we assert on the message content, not just non-None.
# ---------------------------------------------------------------------------

_TERMINAL = ("Abandon", "Sufficient")
_ACTIVE = ("Low priority", "Medium priority", "High priority")


def _gate(decision, state, is_forgotten, has_in_flight):
    return evaluate_terminal_tag_gate(
        decision=decision, state=state, is_forgotten=is_forgotten,
        has_in_flight=has_in_flight, terminal_decisions=_TERMINAL,
        active_decisions=_ACTIVE,
    )


def test_gate_failed_candidate_only_abandon():
    # failed + non-Abandon -> rejected; message names FAILED, the fix (Abandon),
    # and the offending decision.
    msg = _gate("Medium priority", "failed", False, False)
    assert msg is not None
    assert "FAILED" in msg and "Abandon" in msg
    assert "Medium priority" in msg
    msg2 = _gate("Sufficient", "failed", False, False)
    assert msg2 is not None and "Abandon" in msg2 and "Sufficient" in msg2
    # failed + Abandon -> allowed, regardless of forgotten / in-flight
    assert _gate("Abandon", "failed", True, True) is None


def test_gate_terminal_requires_settled():
    # forgotten -> reject; message names the decision, the reason (not settled),
    # and steers to an active priority.
    msg = _gate("Abandon", "surface_relaxation", True, False)
    assert msg is not None
    assert "Abandon" in msg
    assert "settled" in msg.lower()
    assert all(a in msg for a in _ACTIVE)         # ALL active options offered
    # in-flight -> reject
    msg2 = _gate("Sufficient", "surface_relaxation", False, True)
    assert msg2 is not None and "settled" in msg2.lower() and "Sufficient" in msg2
    # fully settled -> allow (None)
    assert _gate("Abandon", "surface_relaxation", False, False) is None
    assert _gate("Sufficient", None, False, False) is None


def test_gate_active_priority_always_allowed_when_not_failed():
    assert _gate("Medium priority", "surface_relaxation", True, True) is None
    assert _gate("Low priority", None, True, False) is None


# ---------------------------------------------------------------------------
# update messages — each rejection states the reason AND how to resolve
# ---------------------------------------------------------------------------

def test_locked_message_directs_to_get():
    msg = format_update_disposition(
        "matX", {"ok": False, "status": "locked"}, ALLOWED)
    assert "get_disposition_info" in msg          # how to resolve
    assert "matX" in msg


def test_invalid_decision_lists_allowed_values():
    msg = format_update_disposition(
        "matX", {"ok": False, "status": "invalid_decision", "decision": "Maybe"},
        ALLOWED)
    assert "Maybe" in msg                         # what was wrong
    assert all(d in msg for d in ALLOWED)         # the allowed values (from var.py)


def test_nonterminal_message_names_unfinished_ids():
    msg = format_update_disposition(
        "matX", {"ok": False, "status": "non_terminal_ids", "ids": [42, 99]},
        ALLOWED)
    assert "42" in msg and "99" in msg


def test_incomplete_message_names_missing_and_resolution():
    msg = format_update_disposition(
        "matX", {"ok": False, "status": "incomplete", "missing": [42]}, ALLOWED)
    assert "42" in msg                            # still-uncovered id
    assert ("cite" in msg.lower()) or ("get_disposition_info" in msg)  # how to resolve


def test_ok_message_confirms_candidate_and_decision():
    msg = format_update_disposition(
        "matX", {"ok": True, "status": "ok", "decision": "Medium priority"}, ALLOWED)
    assert "matX" in msg
    assert "Medium priority" in msg


# ---------------------------------------------------------------------------
# get message — lists what to summarize
# ---------------------------------------------------------------------------

def test_get_message_lists_outstanding_ids():
    outstanding = {
        "must_cover": [{"job_type": "surface_relaxation", "termination_index": 1,
                        "site_index": None, "ids": [42]}],
        "legacy_optional": [],
        "latest_disposition": None,
        "has_finalized": True,
    }
    msg = format_get_disposition("matX", outstanding)
    assert "42" in msg                            # the id to summarize
    assert "matX" in msg


def test_get_message_when_nothing_outstanding():
    outstanding = {"must_cover": [], "legacy_optional": [],
                   "latest_disposition": None, "has_finalized": False}
    msg = format_get_disposition("matX", outstanding)
    assert isinstance(msg, str) and msg          # non-empty, no crash


def test_get_message_renders_latest_disposition_when_present():
    # NB: the formatter only RENDERS the latest_disposition it is handed --
    # choosing the most recent of several is candidate_outstanding's job (tested
    # in test_disposition_tools.py). Here we just check it is surfaced.
    outstanding = {
        "must_cover": [],
        "legacy_optional": [],
        "latest_disposition": {"Decision": "Medium priority", "Summary": "promising",
                               "Future_plan": "run more OH sites",
                               "Summarized_process_id": [1]},
        "has_finalized": True,
    }
    msg = format_get_disposition("matX", outstanding)
    assert "Medium priority" in msg                # Decision
    assert "promising" in msg                    # Summary
    assert "run more OH sites" in msg            # Future_plan, for context


def test_get_message_states_no_prior_disposition_when_none():
    outstanding = {"must_cover": [], "legacy_optional": [],
                   "latest_disposition": None, "has_finalized": False}
    msg = format_get_disposition("matX", outstanding)
    assert "no prior disposition" in msg.lower()


def test_get_message_never_lists_legacy_ids():
    # legacy_optional present, must_cover empty -> the legacy id must NOT appear
    outstanding = {
        "must_cover": [],
        "legacy_optional": [{"job_type": "surface_relaxation",
                             "termination_index": 0, "site_index": None,
                             "ids": [7]}],
        "latest_disposition": None,
        "has_finalized": True,
    }
    msg = format_get_disposition("matX", outstanding)
    assert "7" not in msg                        # legacy id intentionally omitted


def test_unknown_candidate_update_message():
    msg = format_update_disposition(
        "ghost", {"ok": False, "status": "unknown_candidate"}, ALLOWED)
    assert "ghost" in msg                    # names the offending candidate id
    assert "experiment log" in msg.lower()   # explains it is not in the log
    assert "try again" in msg.lower()        # gives a resolution
    # must NOT read like a success or echo a recorded disposition
    assert "Recorded" not in msg
    assert "Decision:" not in msg


def test_foreign_ids_update_message_names_them():
    msg = format_update_disposition(
        "matX", {"ok": False, "status": "foreign_ids", "ids": [42, 99]}, ALLOWED)
    assert "42" in msg and "99" in msg       # the offending ids
    assert "matX" in msg                     # they do not belong to this candidate
    assert "Recorded" not in msg             # not a success


def test_get_message_unknown_candidate():
    msg = format_get_disposition("ghost", {"unknown_candidate": True})
    assert "ghost" in msg                    # names the offending candidate id
    assert "experiment log" in msg.lower()   # explains it is not in the log
    assert "no prior disposition" not in msg.lower()  # must not read like success


# ---------------------------------------------------------------------------
# format_supervisor_handback_directive: the SUPERVISOR-facing instruction the
# supervisor node injects on a queue-floor handback. Path A (submit_ready) tells
# the supervisor to plan an immediate submit step + lists the ready jobs; Path B
# (expand) tells it to open a discussion with the OER_agent covering four points.
# Neither addresses the worker or tells anyone to "wait".
# ---------------------------------------------------------------------------

def test_handback_directive_submit_ready_plans_submit_and_lists_jobs():
    jobs = [{"candidate_id": "matZ", "kind": "bulk",
             "termination_index": None, "site_index": None},
            {"candidate_id": "matY", "kind": "surface",
             "termination_index": None, "site_index": None}]
    msg = format_supervisor_handback_directive("submit_ready", jobs)
    assert "submit" in msg.lower()               # plan a submit step
    assert "plan step" in msg.lower()            # ... as a plan step
    assert "matZ" in msg and "matY" in msg       # the ready jobs are named
    assert "2 " in msg                           # the count of ready jobs
    assert "effectively free" in msg.lower()     # frames the submit as free
    # It steers the supervisor, not the worker (no worker "end your turn" script).
    assert "END YOUR TURN" not in msg


def test_handback_directive_submit_ready_guards_the_floor_disarm():
    msg = format_supervisor_handback_directive("submit_ready", [
        {"candidate_id": "matZ", "kind": "bulk",
         "termination_index": None, "site_index": None}])
    # Disarming the floor is reserved for a genuine wind-down, not the default.
    assert "enforce_queue_floor=False" in msg
    assert "winding down" in msg.lower()


def test_handback_directive_expand_opens_discussion_with_oer_agent():
    msg = format_supervisor_handback_directive("expand")
    low = msg.lower()
    assert "discussion" in low                   # a discussion, ...
    assert "oer_agent" in low                     # ... with the OER_agent
    assert "not a decision you make alone" in low  # not decided unilaterally
    # The four minimum discussion points are all present.
    assert "(1)" in msg and "(2)" in msg and "(3)" in msg and "(4)" in msg
    assert "current results" in low              # point 1: results
    assert "arxiv" in low                        # point 2: literature
    assert "active candidates" in low            # point 3: which active to push
    assert "aq-gnome" in low                     # point 4: add more candidates
    # Wind-down remains available, but nobody is told to "wait" as an instruction.
    assert "enforce_queue_floor=False" in msg
    assert "END YOUR TURN" not in msg


def test_handback_directive_expand_lists_no_specific_jobs():
    # Path B has no ready work, so it must not fabricate a job listing.
    msg = format_supervisor_handback_directive("expand")
    assert "Ready work to submit" not in msg
