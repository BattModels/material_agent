# Guard tests for the Step 9 prompt edits, test-FIRST (TDD).
#
# These pin the disposition-gate's prose contract so it cannot silently drift:
#   - the old "keep the queue occupied" occupancy-as-target language is gone from
#     BOTH prompts (occupancy is now a consequence of analysis + the wait gates);
#   - the worker prompt teaches the disposition workflow (get -> update before
#     waiting) and names the Decision vocabulary;
#   - the supervisor prompt keeps its enforce_queue_floor guidance.
#
# Importing src.prompt is cheap -- it only pulls the tiny stdlib-only src.var
# (never src.tools / the GNoME database).

from src import var
from src.prompt import supervisor_prompt, oer_agent_prompt


def test_occupancy_as_target_removed_from_both_prompts():
    assert "occupied" not in supervisor_prompt.lower()
    assert "occupied" not in oer_agent_prompt.lower()


def test_worker_prompt_teaches_disposition_workflow():
    assert "get_disposition_info" in oer_agent_prompt
    assert "update_disposition_info" in oer_agent_prompt
    for decision in var.DISPOSITION_DECISIONS:        # the Decision vocabulary, named
        assert decision in oer_agent_prompt


def test_supervisor_prompt_keeps_enforce_queue_floor_guidance():
    assert "enforce_queue_floor" in supervisor_prompt


def test_worker_prompt_has_queue_floor_handback_rule():
    # A refused wait with ready jobs listed (Path A) is the worker's own job:
    # submit them under the current task, then re-call wait. Only a no-ready-work
    # refusal (Path B) is a return-to-supervisor event.
    p = oer_agent_prompt.lower()
    assert "submit them now" in p
    assert "call wait_for_update again" in p
    assert "no supervisor round-trip" in p
    assert "return-to-supervisor event" in p       # Path B still hands back


def test_worker_prompt_defines_the_standing_duty():
    # The pipeline-continuation standing duty must be spelled out: what follows
    # what, disposition-guided priority, and the supervisor-only boundary.
    p = oer_agent_prompt.lower()
    assert "standing duty" in p
    assert "bulk relaxation leads to surface relaxation" in p
    assert "surface relaxation leads to o adsorption" in p
    assert "oh adsorption" in p
    assert "supervisor decision" in p              # new candidates stay with the supervisor


def test_supervisor_prompt_states_floor_semantics_and_disarm_window():
    # The supervisor must be told the floor is on QUEUED jobs, the refill target,
    # that workers self-serve ready work, and the final-days disarm rule.
    s = supervisor_prompt.lower()
    assert "queued" in s
    assert str(var.QUEUE_MIN_PENDING) in s
    assert str(var.QUEUE_REFILL_TARGET) in s
    assert "standing duty" in s
    assert f"final {var.FLOOR_DISARM_WINDOW_DAYS} days" in s


def test_worker_prompt_states_the_terminal_tag_gate():
    # The disposition workflow must mention the gate so the agent doesn't waste
    # rejected attempts: terminal tags need a settled candidate; failed -> Abandon.
    p = oer_agent_prompt.lower()
    assert "may only be abandon" in p
    assert "terminal" in p and "fully settled" in p
