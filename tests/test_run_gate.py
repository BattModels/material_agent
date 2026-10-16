"""Pause / step-through gate: parsing, budgeting, blocking and resume."""
import json
import threading
import time

import pytest

from src import run_gate as G
from src import var


# ---------------------------------------------------------------- parse_tokens
@pytest.mark.parametrize("content, expected", [
    ("1312", [1, 3, 1, 2]),                       # bare digits: one grant per char
    ("1212410", [1, 2, 1, 2, 4, 1, 0]),
    ("1,3,1,2,10", [1, 3, 1, 2, 10]),             # separated: multi-digit grants
    ("1, 3 , 1,2, 10", [1, 3, 1, 2, 10]),         # whitespace around separators
    ("1 3 1 2 10", [1, 3, 1, 2, 10]),             # whitespace alone separates
    ("  1312\n", [1, 3, 1, 2]),                   # surrounding whitespace ignored
    ("", []),
    ("   \n ", []),
    ("7", [7]),
])
def test_parse_tokens(content, expected):
    assert G.parse_tokens(content) == expected


@pytest.mark.parametrize("bad", ["12a4", "1,-3", "1,,x", "three", "1.5"])
def test_parse_tokens_rejects_non_numbers(bad):
    with pytest.raises(G.MalformedStepLog):
        G.parse_tokens(bad)


def test_the_two_notations_are_interchangeable_mid_run():
    # the log may switch form; what matters is the token list, not the text
    assert G.parse_tokens("1312") == G.parse_tokens("1,3,1,2") == [1, 3, 1, 2]
    grant, seen, warn = G.reconcile(G.parse_tokens("1312"), G.parse_tokens("1,3,1,2,10"))
    assert (grant, seen, warn) == (10, [1, 3, 1, 2, 10], None)


# ---------------------------------------------------------------- reconcile
def test_reconcile_grants_only_the_appended_tokens():
    assert G.reconcile([1, 3, 1, 2], [1, 3, 1, 2, 4]) == (4, [1, 3, 1, 2, 4], None)
    assert G.reconcile([], [1]) == (1, [1], None)                      # fresh run
    assert G.reconcile([1, 3], [1, 3, 2, 5]) == (7, [1, 3, 2, 5], None)  # two at once


def test_reconcile_grants_nothing_when_unchanged():
    assert G.reconcile([1, 3, 1, 2], [1, 3, 1, 2]) == (0, [1, 3, 1, 2], None)


def test_reconcile_refuses_a_rewrite_and_rebaselines():
    grant, seen, warn = G.reconcile([1, 3, 1, 2], [1, 3, 1, 5])
    assert grant == 0 and seen == [1, 3, 1, 5]
    assert warn and "append-only" in warn
    # re-baselined, so appending to the NEW content works normally next time
    assert G.reconcile(seen, [1, 3, 1, 5, 2]) == (2, [1, 3, 1, 5, 2], None)


# ---------------------------------------------------------------- files
@pytest.fixture
def wd(tmp_path, monkeypatch):
    monkeypatch.setattr(var, "my_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(var, "RUN_GATE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(var, "RUN_GATE_HEARTBEAT_SECONDS", 0.05)
    return tmp_path


def test_init_creates_defaults_and_never_overwrites(wd):
    G.init_control_files()
    assert (wd / "status.txt").read_text() == var.RUN_GATE_INITIAL_MODE == "num"
    assert (wd / "step_num.txt").read_text() == "1"
    # a resume must keep the operator's current mode and the grant history
    (wd / "status.txt").write_text("run")
    (wd / "step_num.txt").write_text("1312")
    G.init_control_files()
    assert (wd / "status.txt").read_text() == "run"
    assert (wd / "step_num.txt").read_text() == "1312"


def test_read_mode_strips_whitespace_and_recreates_a_missing_file(wd):
    (wd / "status.txt").write_text("  STOP\n")      # trailing newline used to defeat this
    assert G.read_mode() == "stop"
    (wd / "status.txt").unlink()
    assert G.read_mode() == var.RUN_GATE_INITIAL_MODE
    assert (wd / "status.txt").exists()


# ---------------------------------------------------------------- the gate
def test_run_mode_never_blocks(wd):
    G.init_control_files(mode="run")
    for _ in range(3):
        G.wait_for_turn("FEM_Agent")                # returns immediately
    assert not (wd / "step_gate_state.json").exists()   # no budget touched


def test_num_mode_spends_one_step_per_crossing_then_blocks(wd):
    G.init_control_files()                          # mode "num", log "1"
    G.wait_for_turn("Supervisor")                   # the initial "1" grants one step
    state = G.read_state()
    assert state == {"seen": [1], "remaining": 0}

    # budget exhausted -> the next crossing blocks until the log grows
    released = threading.Event()

    def grant_later():
        time.sleep(0.05)
        (wd / "step_num.txt").write_text("13")      # append a grant of 3
        released.set()

    t = threading.Thread(target=grant_later, daemon=True)
    t.start()
    G.wait_for_turn("FEM_Agent")                    # blocks, then consumes 1 of 3
    t.join()
    assert released.is_set()
    assert G.read_state() == {"seen": [1, 3], "remaining": 2}

    G.wait_for_turn("FEM_Agent")
    G.wait_for_turn("Boss")
    assert G.read_state()["remaining"] == 0


def test_stop_blocks_until_the_mode_changes(wd):
    G.init_control_files(mode="stop")

    def release():
        time.sleep(0.05)
        (wd / "status.txt").write_text("run")

    threading.Thread(target=release, daemon=True).start()
    t0 = time.monotonic()
    G.wait_for_turn("Supervisor")
    assert time.monotonic() - t0 >= 0.04


def test_an_unrecognised_mode_pauses_rather_than_running(wd):
    G.init_control_files()
    (wd / "status.txt").write_text("gibberish")

    def release():
        time.sleep(0.05)
        (wd / "status.txt").write_text("run")

    threading.Thread(target=release, daemon=True).start()
    t0 = time.monotonic()
    G.wait_for_turn("FEM_Agent")                    # fail safe: did not run
    assert time.monotonic() - t0 >= 0.04


def test_resume_does_not_regrant_the_whole_history(wd):
    """The point of step_gate_state.json: a restart with '13124' in the log must
    not hand the agent 13124 turns."""
    G.init_control_files()
    (wd / "step_num.txt").write_text("13124")
    for _ in range(11):                             # 1+3+1+2+4 = 11 granted
        G.wait_for_turn("FEM_Agent")
    assert G.read_state() == {"seen": [1, 3, 1, 2, 4], "remaining": 0}

    # simulate a process restart: nothing in memory, state read back from disk
    G.init_control_files()
    assert G.read_state() == {"seen": [1, 3, 1, 2, 4], "remaining": 0}

    def release():
        time.sleep(0.05)
        # switching notation means rewriting the WHOLE log that way
        (wd / "step_num.txt").write_text("1,3,1,2,4,10")
    threading.Thread(target=release, daemon=True).start()
    G.wait_for_turn("FEM_Agent")
    assert G.read_state() == {"seen": [1, 3, 1, 2, 4, 10], "remaining": 9}


def test_half_switching_notation_is_refused_instead_of_granting_thousands(wd):
    """Appending ',10' to the bare log '13124' re-reads it as [13124, 10]. That
    must NOT be honored, or one keystroke buys 13124 agent turns."""
    G.init_control_files()
    (wd / "step_num.txt").write_text("13124")
    G.wait_for_turn("FEM_Agent")
    assert G.read_state()["remaining"] == 10

    grant, seen, warn = G.reconcile([1, 3, 1, 2, 4], G.parse_tokens("13124,10"))
    assert grant == 0 and seen == [13124, 10]
    assert "whole log" in warn.lower() and "1,3,1,2,4,10" in warn


def test_a_zero_grant_does_not_release_the_gate(wd):
    G.init_control_files()
    G.wait_for_turn("Supervisor")                   # spends the initial 1

    def release():
        time.sleep(0.04)
        (wd / "step_num.txt").write_text("10")      # bare digits: grants 1 then 0
        time.sleep(0.04)
        (wd / "status.txt").write_text("run")

    threading.Thread(target=release, daemon=True).start()
    G.wait_for_turn("FEM_Agent")                    # released by the "1"
    assert G.read_state()["remaining"] == 0         # the "0" granted nothing


def test_a_malformed_or_missing_log_pauses_and_recovers(wd):
    G.init_control_files()
    G.wait_for_turn("Supervisor")
    (wd / "step_num.txt").write_text("1oops")

    def repair():
        time.sleep(0.05)
        (wd / "step_num.txt").write_text("12")
    threading.Thread(target=repair, daemon=True).start()
    G.wait_for_turn("FEM_Agent")                    # paused on garbage, then granted 2
    assert G.read_state() == {"seen": [1, 2], "remaining": 1}


def test_state_file_is_valid_json_and_survives_corruption(wd):
    G.init_control_files()
    G.wait_for_turn("Supervisor")
    assert json.loads((wd / "step_gate_state.json").read_text())["seen"] == [1]
    (wd / "step_gate_state.json").write_text("{not json")
    assert G.read_state() == {"seen": [], "remaining": 0}   # degrades, never raises
