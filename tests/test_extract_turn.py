# Tests for src.workflow_audit.extract_turn._find_round_boundaries / extract_turns.
#
# Regression coverage for a bug found while auditing a live run: requesting
# exactly the LAST round of a his.txt/hist file (still open -- no next-round
# marker has appeared yet, e.g. because the run is still in progress) used to
# raise ValueError even though the round's content was already captured.

import pytest

from src.workflow_audit.extract_turn import extract_turns

# 3 rounds: supervisor, worker, boss. The file ends mid-round-3 -- no 4th
# round marker -- exactly the "still-open last round" shape of a live run.
_HIS_TXT = """supervisor is processing!!!!! Current time: 0:00:01.
line r1a
line r1b
Agent OER_Agent is processing!!!!!
line r2a
line r2b
Boss_Agent is processing!!!!! Current time: 0:00:03.
line r3a
line r3b
"""


def _make_his(tmp_path):
    p = tmp_path / "his.txt"
    p.write_text(_HIS_TXT)
    return p


def test_extract_last_open_round_succeeds(tmp_path):
    src = _make_his(tmp_path)
    out = tmp_path / "turn_3.txt"
    extract_turns(src, 3, 3, out)  # must not raise
    text = out.read_text()
    assert "Boss_Agent is processing" in text
    assert "line r3a" in text
    assert "line r3b" in text


def test_extract_beyond_last_round_still_raises(tmp_path):
    src = _make_his(tmp_path)
    out = tmp_path / "turn_4.txt"
    with pytest.raises(ValueError, match="turn 4 not found"):
        extract_turns(src, 4, 4, out)
    assert not out.exists()


def test_extract_range_past_end_still_raises(tmp_path):
    # start_turn (2) exists, but end_turn (4) does not -- must still raise,
    # not silently truncate to whatever rounds happen to exist.
    src = _make_his(tmp_path)
    out = tmp_path / "turn_2_4.txt"
    with pytest.raises(ValueError, match="turn 4 not found"):
        extract_turns(src, 2, 4, out)
    assert not out.exists()


def test_extract_non_last_round_still_works(tmp_path):
    src = _make_his(tmp_path)
    out = tmp_path / "turn_2.txt"
    extract_turns(src, 2, 2, out)
    text = out.read_text()
    assert "Agent OER_Agent is processing" in text
    assert "line r2a" in text
    assert "line r2b" in text
    assert "line r3a" not in text
