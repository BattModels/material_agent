"""Plan-length rules and routing for the supervisor's Advance/Repeat actions.

These pin the 2026-08-18 05:10 outage. The supervisor answered `NoChange` three
times against a ONE-step plan; the validator refused each time and
supervisor_node hit exit(0). The refusal was right -- that action drops step 0
and dispatches step 1, so a one-step plan would IndexError -- but the action the
supervisor actually wanted ("continue to execute the current step") did not
exist. `repeat` is that action, and test_repeat_is_legal_on_a_one_step_plan plus
test_repeat_routes_on_a_one_step_plan_without_indexerror are the regression.

src/supervisor_actions.py is deliberately pure, so this file never imports
src.planNexe2 (and with it src.tools, which builds the AQ-GNoME cache at import
-- see tests/test_query_explog_tool.py:30 for what that costs).
"""

import pytest

from src.supervisor_actions import (
    ADVANCE,
    LEGACY_KINDS,
    REPEAT,
    action_warning,
    normalize_kind,
    route_action,
)


class _Step:
    """The only thing route_action needs off a plan step."""

    def __init__(self, agent="OER_Agent"):
        self.agent = agent


@pytest.fixture
def two_steps():
    return [_Step("OER_Agent"), _Step("OER_Agent")]


# --- which actions are legal against a plan of length N --------------------

def test_advance_is_legal_with_two_steps(two_steps):
    assert action_warning(ADVANCE, len(two_steps)) is None


def test_advance_is_refused_on_a_one_step_plan():
    """The exact 05:10 death: no 'second' step exists to advance to."""
    assert action_warning(ADVANCE, 1) is not None


def test_advance_refusal_offers_repeat():
    """Before Repeat existed this refusal offered only Plan or Response, neither
    of which was what the supervisor wanted -- so it re-chose the illegal action
    until patience ran out. The way out must be named."""
    warning = action_warning(ADVANCE, 1)
    assert "Repeat" in warning
    assert "Plan" in warning and "Response" in warning


def test_repeat_is_legal_on_a_one_step_plan():
    """THE regression. A one-step plan is precisely when Repeat is needed."""
    assert action_warning(REPEAT, 1) is None


def test_repeat_is_legal_on_a_long_plan():
    assert action_warning(REPEAT, 7) is None


def test_repeat_is_refused_on_an_empty_plan():
    """Nothing to repeat, and route_action would IndexError on plan[0]."""
    assert action_warning(REPEAT, 0) is not None


def test_plan_and_response_are_not_this_helpers_business():
    """Their validations inspect the action payload, not the plan length, and
    stay inline in supervisor_node."""
    for kind in ("plan", "response"):
        assert action_warning(kind, 0) is None
        assert action_warning(kind, 1) is None


# --- what each action does to the plan -------------------------------------

def test_repeat_keeps_the_plan_and_redispatches_the_current_step(two_steps):
    delta = route_action(REPEAT, two_steps)
    assert delta["plan"] == two_steps            # same list contents...
    assert delta["plan"][0] is two_steps[0]      # ...and the same objects
    assert delta["next"] == two_steps[0].agent


def test_repeat_routes_on_a_one_step_plan_without_indexerror():
    """The crash the old validator existed to prevent, now reachable legally."""
    plan = [_Step("OER_Agent")]
    delta = route_action(REPEAT, plan)
    assert len(delta["plan"]) == 1
    assert delta["next"] == "OER_Agent"


def test_advance_drops_the_finished_step_and_dispatches_the_next(two_steps):
    delta = route_action(ADVANCE, two_steps)
    assert delta["plan"] == two_steps[1:]
    assert len(delta["plan"]) == 1
    assert delta["next"] == two_steps[1].agent


def test_advance_dispatches_the_second_steps_agent():
    plan = [_Step("OER_Agent"), _Step("Boss_Agent")]
    assert route_action(ADVANCE, plan)["next"] == "Boss_Agent"


def test_both_actions_clear_boss_feedback(two_steps):
    """boss_feedback is consumed on every exit path from supervisor_node, or a
    one-shot operator directive re-renders for the rest of the run."""
    for kind in (ADVANCE, REPEAT):
        assert route_action(kind, two_steps)["boss_feedback"] == ""


def test_route_action_defers_plan_and_response_to_the_caller(two_steps):
    for kind in ("plan", "response"):
        assert route_action(kind, two_steps) is None


# --- the pre-rename name ---------------------------------------------------

def test_legacy_no_change_kind_maps_onto_advance():
    """Retries append to the previous attempt's conversation in a shared
    checkpoint namespace, so the first resume after the rename can still surface
    the old kind."""
    assert normalize_kind("no_change") == ADVANCE
    assert LEGACY_KINDS["no_change"] == ADVANCE


def test_legacy_kind_is_validated_and_routed_as_advance(two_steps):
    assert action_warning("no_change", 1) is not None
    assert action_warning("no_change", 2) is None
    assert route_action("no_change", two_steps)["next"] == two_steps[1].agent


def test_unknown_kind_is_inert():
    assert action_warning("wibble", 1) is None
    assert route_action("wibble", [_Step()]) is None
