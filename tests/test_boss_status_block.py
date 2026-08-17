# Tests for the facts handed to the BOSS at review time.
#
# BACKGROUND. The 27-05 run was approved as complete at 14.68 of its 30
# study-days, with 0 jobs running and 0 queued. boss_node's prompt carried only
# `Current time: <elapsed>` -- the reviewer deciding whether a study was finished
# was never told the budget, how much of it was left, or whether the cluster was
# doing anything. It could not have weighed the clock; it was never shown one.
#
# format_study_status_block is that missing block, and it is a pure function so
# it can be exercised here without importing planNexe2 (which pulls in the LLM
# stack and the GNoME database). Nothing in it GATES the boss's decision -- the
# boss may still approve -- these tests only pin that the numbers it needs are
# present, correct, and never silently fabricated.

import pytest

from src import var
from src.disposition_messages import (
    MSG_NOTHING_TO_WAIT_FOR,
    format_study_status_block,
    format_supervisor_handback_directive,
    format_wait_gate2_refusal,
)
from src.prompt import boss_prompt, supervisor_prompt


_DAY = 86400


# --- the numbers themselves --------------------------------------------------

def test_reports_remaining_not_just_elapsed():
    """The single fact the boss never had. Elapsed alone is what it always saw."""
    block = format_study_status_block(
        elapsed_seconds=14.68 * _DAY, pending_count=0, running_count=0)
    assert "REMAINING" in block
    # Derived, not hardcoded: STUDY_BUDGET_DAYS is an operator knob (raised
    # 30 -> 200 when the study became a continuing programme), so pinning the
    # arithmetic to one budget just breaks the test when the knob moves.
    assert f"{var.STUDY_BUDGET_DAYS - 14.68:.2f}" in block, block
    assert "14.68" in block


def test_reports_the_budget_and_percentage_spent():
    block = format_study_status_block(
        elapsed_seconds=15 * _DAY, pending_count=3, running_count=4)
    assert f"{var.STUDY_BUDGET_DAYS:.1f}" in block
    assert f"{100 * 15 / var.STUDY_BUDGET_DAYS:.0f}%" in block, block


def test_renders_live_queue_counts_with_the_floor_and_target():
    block = format_study_status_block(
        elapsed_seconds=1 * _DAY, pending_count=7, running_count=11)
    assert "11 running" in block
    assert "7 queued" in block
    assert str(var.QUEUE_MIN_PENDING) in block
    assert str(var.QUEUE_REFILL_TARGET) in block


def test_an_idle_cluster_is_rendered_as_such():
    """0/0 is the exact state the 27-05 run concluded in; it must be visible."""
    block = format_study_status_block(
        elapsed_seconds=14.68 * _DAY, pending_count=0, running_count=0)
    assert "0 running, 0 queued" in block


@pytest.mark.parametrize("pending,running", [(None, None), (None, 4), (3, None)])
def test_unreadable_explog_says_unavailable_and_never_fakes_zero(pending, running):
    """boss_node passes None when EXPLOG raises. Rendering that as 0/0 would read
    as an idle cluster and argue for precisely the wrong conclusion, so the block
    must say it does not know."""
    block = format_study_status_block(
        elapsed_seconds=5 * _DAY, pending_count=pending, running_count=running)
    assert "unavailable" in block.lower()
    assert "0 running" not in block


def test_names_the_too_late_to_start_threshold():
    """The boss needs the criterion, not just the number: below PATH_B_CUTOFF_DAYS
    a new job cannot finish, so concluding is defensible."""
    block = format_study_status_block(
        elapsed_seconds=2 * _DAY, pending_count=1, running_count=1)
    assert str(var.PATH_B_CUTOFF_DAYS) in block


def test_elapsed_beyond_budget_gives_negative_remaining_not_a_crash():
    """An overrun run must still be reviewable."""
    block = format_study_status_block(
        elapsed_seconds=(var.STUDY_BUDGET_DAYS + 2) * _DAY,
        pending_count=0, running_count=0)
    assert "-2.00" in block, block


# --- the prompt that reads them ----------------------------------------------

def test_boss_prompt_ties_approval_to_the_clock():
    """Instruction 2 used to make deliverable coverage SUFFICIENT for approval;
    requirement 2 used to forbid rejecting because more work could be done. Those
    two lines are what approved the 27-05 run at half budget."""
    p = boss_prompt.lower()
    assert "remaining" in p
    assert "budget" in p
    assert "more work could theoretically be done" not in p, (
        "the clause that instructed the boss to approve an early finish is back"
    )


def test_boss_prompt_still_forbids_rejecting_on_style():
    """The narrowing that made the boss useful is kept -- this is a change of
    grounds, not a licence to reject anything."""
    assert "style" in boss_prompt.lower()


def test_supervisor_prompt_makes_coverage_a_standing_objective():
    p = supervisor_prompt.lower()
    assert "coverage" in p
    # Coverage stays a standing objective, but registering candidates is no
    # longer the reflex answer to it: the supervisor must ASK the worker what
    # the log supports first. Registering 169 candidates in one session to hold
    # a queue target is what that reflex produced.
    assert "default action is to register more candidates" not in p
    assert "put the question to your worker" in p


# --- the gate prose ----------------------------------------------------------
# Anti-regression: "or conclude the study if it is genuinely complete" was the
# clause the supervisor walked through at round 745. An empty queue means the
# study needs widening; only the CLOCK justifies ending it.

@pytest.mark.parametrize("msg", [
    MSG_NOTHING_TO_WAIT_FOR,
    format_wait_gate2_refusal([], running_count=1, pending_count=2),
    format_supervisor_handback_directive("expand"),
])
def test_no_conclude_the_study_escape_hatch(msg):
    low = msg.lower()
    assert "genuinely complete" not in low
    assert "conclude the study if" not in low


@pytest.mark.parametrize("msg", [
    MSG_NOTHING_TO_WAIT_FOR,
    format_wait_gate2_refusal([], running_count=1, pending_count=2),
    format_supervisor_handback_directive("expand"),
])
def test_adding_candidates_is_offered_before_deepening(msg):
    """Ordering is the point: these messages used to list new candidates last and
    hedged, after the concrete list of ready continuation work."""
    low = msg.lower()
    assert "candidate" in low
    first_add = min(
        (low.index(k) for k in ("new candidates", "more candidates",
                                "registering more candidates", "adding new candidates")
         if k in low),
        default=None,
    )
    assert first_add is not None, "no new-candidate guidance at all"
    deepen = min((low.index(k) for k in ("termination", "adsorption site") if k in low),
                 default=len(low))
    assert first_add < deepen, "deepening is still offered ahead of widening"


def test_path_a_surfaces_a_shortfall_without_ordering_registration():
    """Path A is the COMMON path and used never to mention new candidates at all,
    so the routine refill loop could only ever go deeper, never wider. It now says
    to REPORT a shortfall -- but must not tell the worker to register candidates
    itself: oer_agent_prompt requirement 15 reserves that for the supervisor, and a
    gate that orders what the system prompt forbids just deadlocks the worker."""
    jobs = [{"candidate_id": "mp-1", "kind": "bulk"}]
    msg = format_wait_gate2_refusal(jobs, running_count=2, pending_count=1)
    low = msg.lower()
    assert "report" in low                       # surface the shortfall...
    assert "supervisor's decision" in low        # ...whose call it is
    assert "do not register new candidates yourself" in low
    assert str(var.QUEUE_REFILL_TARGET) in msg


def test_worker_prompt_keeps_registration_with_the_supervisor():
    """The invariant Path A must not contradict."""
    from src.prompt import oer_agent_prompt
    assert "remains a supervisor decision" in oer_agent_prompt


def test_system_prompts_do_not_assert_a_transient_cluster_state():
    """A system prompt is fixed for the whole run, so it must state POLICY, not a
    fact that expires. "The cluster is under-utilized" is true today and false a
    week in once our own jobs are in flight -- and an agent told that as a
    standing fact will keep submitting against a saturated queue. Live utilization
    belongs in the wait-gate messages, which are built per call from real counts."""
    from src.prompt import oer_agent_prompt, supervisor_prompt
    for p in (oer_agent_prompt, supervisor_prompt, boss_prompt):
        low = p.lower()
        assert "cluster is under-utilized" not in low
        assert "cluster is substantially under-utilized" not in low
        assert "is idle" not in low


def test_worker_prompt_has_no_unanchored_early_finish_nudge():
    """Requirement 16 used to say 'getting close to the time constrain ... report
    back early' with no threshold at all -- the worker-side version of the blind
    spot that let the 27-05 run stop at half budget."""
    from src.prompt import oer_agent_prompt
    assert "getting close to the given time constrain" not in oer_agent_prompt
    assert str(var.PATH_B_CUTOFF_DAYS) in oer_agent_prompt
    assert "not a target to chase for its own sake" not in oer_agent_prompt
