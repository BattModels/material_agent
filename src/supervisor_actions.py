"""Which supervisor actions are legal against the current plan, and what each one
does to it.

PURE -- imports nothing from src.tools -- so the rules that route the graph can be
exercised without building the AQ-GNoME cache at import time (see
tests/test_query_explog_tool.py:30 for what that costs: ~109s, which is why the
GNoME-dependent tests are quarantined behind RUN_SLOW_TESTS=1). Same reason
src/forgotten_jobs.py and src/disposition_messages.py are pure.

Extracted after the 2026-08-18 05:10 outage. The supervisor answered `NoChange`
three times against a ONE-step plan; the validator refused each time -- correctly,
because that action drops step 0 and dispatches step 1, so a one-step plan would
IndexError -- and supervisor_node hit `exit(0)` on exhausted patience. What the
supervisor actually wanted was written in its own comment:

    "No change to the plan needed; continue to execute the current step."

...which the action set could not express. `repeat` is that action. `advance` is
the old `NoChange`, renamed: it does change the plan (it removes the completed
step), so "no change" described the wrong half and invited exactly this misreading.
"""

from typing import Any, Dict, List, Optional

# The `kind` discriminators of Act.action in src/planNexe2.py. Routing dispatches
# on these strings rather than on the Pydantic classes, which is what keeps this
# module free of the heavy import.
PLAN = "plan"
ADVANCE = "advance"
REPEAT = "repeat"
RESPONSE = "response"

# Accepted on input only, mapped to ADVANCE. `no_change` was this action's kind
# before the rename; the supervisor's inner agent shares one checkpoint namespace
# across retry iterations (planNexe2.py:816-819), so the first resume after the
# rename restores a conversation still carrying it.
LEGACY_KINDS = {"no_change": ADVANCE}

_ADVANCE_NEEDS_TWO_STEPS = (
    "\n\nWARNING: there are fewer than 2 steps left in the current plan, so there "
    "is no 'second' step to execute and you cannot choose 'Advance'. If the worker "
    "should execute the CURRENT step again, choose 'Repeat' -- that leaves the plan "
    "untouched and is the right choice for a standing-duty step that keeps cycling. "
    "Otherwise review what has been done and either 'Plan' more steps (the first "
    "step of the new plan you write will be executed next), or 'Response' with a "
    "draft final answer for the boss review."
)

_REPEAT_NEEDS_A_STEP = (
    "\n\nWARNING: the current plan is empty, so there is no current step to repeat "
    "and you cannot choose 'Repeat'. Either 'Plan' more steps, or 'Response' with a "
    "draft final answer for the boss review."
)


def normalize_kind(kind: str) -> str:
    """Map a legacy `kind` onto its current name; pass anything else through."""
    return LEGACY_KINDS.get(kind, kind)


def action_warning(kind: str, n_steps: int) -> Optional[str]:
    """The retry warning for an action that is illegal against a plan of `n_steps`,
    or None if it is legal.

    Covers only the plan-length rules for `advance` / `repeat`. The other two
    validations in supervisor_node -- unknown required_tools on a step, and an
    empty `Plan` -- inspect the action's own payload rather than the plan, and stay
    where they are.
    """
    kind = normalize_kind(kind)
    if kind == ADVANCE and n_steps <= 1:
        return _ADVANCE_NEEDS_TWO_STEPS
    if kind == REPEAT and n_steps == 0:
        return _REPEAT_NEEDS_A_STEP
    return None


def route_action(kind: str, plan: List[Any]) -> Optional[Dict[str, Any]]:
    """The state delta for `advance` / `repeat`, or None for the kinds whose
    delta supervisor_node builds itself (`plan`, `response`).

    `plan` is only duck-typed: each step needs a `.agent`. Callers must have run
    action_warning() first -- routing an `advance` on a one-step plan raises
    IndexError, which is the crash the warning exists to prevent.
    """
    kind = normalize_kind(kind)
    if kind == ADVANCE:
        return {"plan": plan[1:], "boss_feedback": "", "next": plan[1].agent}
    if kind == REPEAT:
        # The plan is handed back UNCHANGED and the current step re-dispatched.
        return {"plan": plan, "boss_feedback": "", "next": plan[0].agent}
    return None
