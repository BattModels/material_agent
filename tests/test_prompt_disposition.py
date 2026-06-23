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
