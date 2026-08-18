"""LiveVisualizer parsing of the supervisor's Act variants.

The bug these exist for: _parse_structured used to discriminate the action by
FIELD PRESENCE --

    elif "comment" in action:
        result["action_type"] = "NoChange"

-- and Advance and Repeat BOTH carry `comment`. Every Repeat would have been
reported as an Advance in the live view, silently. It now discriminates on
`kind`, which every variant already carries.

The other half is backward compatibility: `NoChange` was Advance's name before
2026-08-18, and 400+ MB of hist/ plus the nohup logs say so. Prior sessions are
re-parsed on every restart (1961 steps on the last launch), so the old spelling
must keep parsing forever.
"""

import pytest

from src import var
from src.live_visualizer import SUPERVISOR_TOOL, LiveVisualizer


@pytest.fixture
def viz(tmp_path, monkeypatch):
    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 999)
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 999)
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))
    return v


def _parse(viz, payload=None, content=""):
    return viz._parse_structured(SUPERVISOR_TOOL, content, payload)


# --- structured payloads: discriminate on `kind` ---------------------------

def test_repeat_payload_is_not_reported_as_advance(viz):
    """THE regression. Both variants carry `comment`."""
    parsed = _parse(viz, {"action": {"kind": "repeat", "comment": "keep cycling"}})
    assert parsed["action_type"] == "Repeat"
    assert parsed["comment"] == "keep cycling"


def test_advance_payload_is_reported_as_advance(viz):
    parsed = _parse(viz, {"action": {"kind": "advance", "comment": "step done"}})
    assert parsed["action_type"] == "Advance"
    assert parsed["comment"] == "step done"


def test_plan_payload_still_parses(viz):
    parsed = _parse(viz, {"action": {"kind": "plan",
                                     "steps": [{"step": "s", "agent": "OER_Agent"}]}})
    assert parsed["action_type"] == "Plan"
    assert parsed["steps"] == [{"step": "s", "agent": "OER_Agent"}]


def test_response_payload_still_parses(viz):
    parsed = _parse(viz, {"action": {"kind": "response", "response": "draft"}})
    assert parsed["action_type"] == "Response"
    assert parsed["response"] == "draft"


def test_legacy_no_change_payload_parses_as_advance(viz):
    parsed = _parse(viz, {"action": {"kind": "no_change", "comment": "old"}})
    assert parsed["action_type"] == "Advance"
    assert parsed["comment"] == "old"


def test_payload_without_a_kind_still_falls_back_on_field_presence(viz):
    """Older checkpoints carry no `kind` at all; a bare `comment` is an Advance."""
    assert _parse(viz, {"action": {"comment": "no kind here"}})["action_type"] == "Advance"
    assert _parse(viz, {"action": {"steps": []}})["action_type"] == "Plan"
    assert _parse(viz, {"action": {"response": "r"}})["action_type"] == "Response"


def test_unrecognisable_payload_is_marked_unknown(viz):
    parsed = _parse(viz, {"action": {"wibble": 1}})
    assert parsed["action_type"] == "Unknown"


# --- regex fallback over raw log text --------------------------------------

def test_historical_nochange_line_still_parses(viz):
    """400+ MB of existing logs use this spelling and are replayed on restart."""
    parsed = _parse(viz, content="Returning structured response: "
                                 "action=NoChange(kind='no_change', comment='old text')")
    assert parsed["action_type"] == "Advance"
    assert parsed["comment"] == "old text"


def test_repeat_line_parses_from_raw_text(viz):
    parsed = _parse(viz, content="Returning structured response: "
                                 "action=Repeat(kind='repeat', comment='again')")
    assert parsed["action_type"] == "Repeat"
    assert parsed["comment"] == "again"


def test_advance_line_parses_from_raw_text(viz):
    parsed = _parse(viz, content="Returning structured response: "
                                 "action=Advance(kind='advance', comment='onward')")
    assert parsed["action_type"] == "Advance"
    assert parsed["comment"] == "onward"


def test_plan_and_response_lines_still_parse_from_raw_text(viz):
    plan = _parse(viz, content="action=Plan(steps=[myStep(step='do it', agent='OER_Agent')])")
    assert plan["action_type"] == "Plan"
    assert plan["steps"] == [{"step": "do it", "agent": "OER_Agent"}]

    resp = _parse(viz, content="action=Response(response='the answer')")
    assert resp["action_type"] == "Response"
    assert resp["response"] == "the answer"
