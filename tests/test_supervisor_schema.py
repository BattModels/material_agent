"""The Act action union: Advance / Repeat parsing, coercion, and self-description.

Imported lazily -- src.planNexe2 pulls in src.tools and the GNoME database (see
tests/test_artifact_store.py for the same pattern). The plan-length RULES are
tested without that cost in tests/test_supervisor_actions.py; what needs the real
module is the Pydantic union and its _coerce_action validator.

test_action_description_states_the_mechanism is the anti-regression on the actual
root cause of the 2026-08-18 outage: the prompt body and the class docstrings both
described NoChange precisely, but the description attached to the `action` FIELD
-- the string in front of the model as it fills that field -- gave only a
condition, "if the team can continue to execute the original plan without any
change". The supervisor echoed it almost verbatim and died on it.
"""

import json

import pytest


@pytest.fixture(scope="module")
def schema():
    pytest.importorskip("gnome_dreams_oer_screening")
    from src.planNexe2 import Act, Advance, Repeat
    return Act, Advance, Repeat


# --- the two new variants --------------------------------------------------

def test_repeat_parses_from_its_kind(schema):
    Act, _, Repeat = schema
    act = Act(action={"kind": "repeat", "comment": "keep cycling"})
    assert isinstance(act.action, Repeat)
    assert act.action.kind == "repeat"


def test_advance_parses_from_its_kind(schema):
    Act, Advance, _ = schema
    act = Act(action={"kind": "advance", "comment": "step done"})
    assert isinstance(act.action, Advance)
    assert act.action.kind == "advance"


def test_plan_and_response_are_untouched(schema):
    Act, _, _ = schema
    assert Act(action={"kind": "response", "response": "draft"}).action.kind == "response"
    assert Act(action={"kind": "plan", "steps": []}).action.kind == "plan"


def test_an_unknown_kind_is_rejected(schema):
    Act, _, _ = schema
    with pytest.raises(Exception):
        Act(action={"kind": "wibble", "comment": "x"})


# --- _coerce_action, case 1: action arrives as a JSON string ---------------

def test_json_string_action_still_coerces(schema):
    Act, _, Repeat = schema
    act = Act(action=json.dumps({"kind": "repeat", "comment": "x"}))
    assert isinstance(act.action, Repeat)


# --- _coerce_action, case 2: model wraps the variant in its own name -------

def test_wrapped_repeat_coerces(schema):
    Act, _, Repeat = schema
    assert isinstance(Act(action={"Repeat": {"comment": "x"}}).action, Repeat)


def test_wrapped_advance_coerces(schema):
    Act, Advance, _ = schema
    assert isinstance(Act(action={"Advance": {"comment": "x"}}).action, Advance)


def test_wrapped_plan_and_response_still_coerce(schema):
    Act, _, _ = schema
    assert Act(action={"Plan": {"steps": []}}).action.kind == "plan"
    assert Act(action={"Response": {"response": "r"}}).action.kind == "response"


# --- _coerce_action, case 3: the pre-rename name --------------------------

def test_legacy_no_change_kind_becomes_advance(schema):
    """A checkpoint written before the rename can still surface this: the
    supervisor's inner agent appends retries to the previous attempt's
    conversation in a shared checkpoint namespace."""
    Act, Advance, _ = schema
    act = Act(action={"kind": "no_change", "comment": "old"})
    assert isinstance(act.action, Advance)
    assert act.action.kind == "advance"


def test_legacy_wrapped_nochange_becomes_advance(schema):
    Act, Advance, _ = schema
    assert isinstance(Act(action={"NoChange": {"comment": "x"}}).action, Advance)


def test_legacy_wrapped_nochange_with_inner_kind_becomes_advance(schema):
    """Case 2 leaves an inner `kind` alone, so case 3 must run after it."""
    Act, Advance, _ = schema
    act = Act(action={"NoChange": {"kind": "no_change", "comment": "x"}})
    assert isinstance(act.action, Advance)
    assert act.action.kind == "advance"


# --- what the model is told at the point of choice ------------------------

def test_action_description_states_the_mechanism(schema):
    """Not just the condition. Each branch must say what it does to the plan --
    this is the string the supervisor followed off a cliff."""
    Act, _, _ = schema
    desc = Act.model_fields["action"].description
    assert "Advance" in desc and "Repeat" in desc
    # Advance: says the step is removed and that two are required
    assert "remove" in desc.lower()
    assert "two steps" in desc.lower()
    # Repeat: says the plan is NOT modified
    assert "again" in desc.lower()
    assert "not modified" in desc.lower()


def test_docstrings_distinguish_the_two_variants(schema):
    _, Advance, Repeat = schema
    assert "REMOVED" in Advance.__doc__
    assert "AGAIN" in Repeat.__doc__ and "NOT modified" in Repeat.__doc__
