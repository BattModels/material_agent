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


# --- myStep.required_tools --------------------------------------------------
#
# Field()'s first positional parameter is `default`, not `description`. This
# field was declared Field(f"must-use tools for this step, ...") -- so the prose
# became the DEFAULT VALUE and the field carried no description at all. Two
# consequences, both live for the whole study until 2026-08-18:
#
#   * the model was never told what required_tools means, only the Literal enum;
#   * omitting the field yielded a str, and `set(str)` in the supervisor's
#     wrongTools check iterated it CHARACTER BY CHARACTER --
#     wrongTools: {'-', 'W', 'u', 'C', ' ', 'l', 'd', ...}
#
# That phantom rejection was then silently discarded by an `else: sup_good = True`
# which has now been removed, so these tests are what keep the un-defanged check
# from refusing real plans.


@pytest.fixture(scope="module")
def step_model():
    pytest.importorskip("gnome_dreams_oer_screening")
    from src.planNexe2 import myStep
    return myStep


def test_required_tools_has_a_description(step_model):
    """It had none: the text was consumed as the default instead."""
    desc = step_model.model_fields["required_tools"].description
    assert desc, "required_tools must describe itself to the model"
    assert "subset of the tools available" in desc


def test_required_tools_defaults_to_an_empty_list(step_model):
    """Not to its own description string."""
    default = step_model(step="s", agent="OER_Agent").required_tools
    assert default == []
    assert isinstance(default, list)


def test_the_default_cannot_produce_a_phantom_wrongtools(step_model):
    """The exact expression from supervisor_node. With a str default this
    returned 33 single characters and refused a perfectly good plan."""
    step = step_model(step="s", agent="OER_Agent")
    assert set(step.required_tools) - {"wait_for_update", ""} == set()


def test_an_unknown_tool_name_is_rejected_by_the_literal(step_model):
    """Pydantic is the primary guard; wrongTools is only a backstop for values
    that bypassed validation. This is what makes removing the else-reset safe."""
    with pytest.raises(Exception):
        step_model(step="s", agent="OER_Agent", required_tools=["not_a_real_tool"])


def test_valid_tool_names_are_accepted(step_model):
    step = step_model(step="s", agent="OER_Agent",
                      required_tools=["wait_for_update", "query_explog"])
    assert step.required_tools == ["wait_for_update", "query_explog"]


def test_empty_required_tools_reads_as_no_required_tools(step_model):
    """supervisor_node's worker path tests `not task.required_tools`, which must
    catch the new [] default as well as the legacy ""."""
    assert not step_model(step="s", agent="OER_Agent").required_tools


def test_every_validated_tool_name_survives_the_wrongtools_check(step_model):
    """The invariant that matters: anything the schema ACCEPTS must also pass the
    supervisor's check. supervisor_node used to validate against a hand-copied
    list; the two were identical, but nothing enforced it -- and now that a
    rejection is no longer discarded, a drift would refuse a tool the schema
    accepts and exit(0) after three retries.

    Asserted behaviourally rather than by recomputing the production expression,
    which could only ever agree with itself."""
    from src.planNexe2 import _REQUIRED_TOOL_NAMES

    for tool in _REQUIRED_TOOL_NAMES:
        step = step_model(step="s", agent="OER_Agent", required_tools=[tool])
        assert set(step.required_tools) - set(_REQUIRED_TOOL_NAMES) == set(), (
            f"{tool!r} validates but the supervisor would reject it"
        )


def test_tool_list_did_not_collapse_to_empty(step_model):
    """The dangerous failure mode: if the annotation shape changes, get_args
    nests differently and the list silently becomes [] -- which rejects EVERY
    tool rather than none. planNexe2 asserts this at import; pin it here too."""
    from src.planNexe2 import _REQUIRED_TOOL_NAMES

    assert len(_REQUIRED_TOOL_NAMES) > 10
    for tool in ("submit_dft_job", "wait_for_update", "query_explog", "write_report"):
        assert tool in _REQUIRED_TOOL_NAMES


def test_a_realistic_plan_step_passes_the_wrongtools_check(step_model):
    """End-to-end on the armed check: the expression from supervisor_node must
    return empty for a step the schema accepted."""
    from src.planNexe2 import _REQUIRED_TOOL_NAMES

    step = step_model(step="s", agent="OER_Agent",
                      required_tools=["wait_for_update", "submit_dft_job"])
    assert set(step.required_tools) - set(_REQUIRED_TOOL_NAMES) == set()
