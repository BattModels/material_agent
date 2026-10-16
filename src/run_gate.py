"""Pause / step-through control for the agent loop.

Two plain-text files in the working directory drive it, so a human can steer a
running study from a shell without touching the process:

    status.txt     "run" | "stop" | "num"
    step_num.txt   the step-grant log, only consulted when status is "num"

Every GATE CROSSING counts as one step: ``wait_for_turn`` is called at the top of
each agent node (supervisor, worker, boss), so a supervisor turn followed by a
worker turn consumes two.

    run    proceed, and re-read the file at the next crossing
    stop   block until the file says something else
    num    proceed while the step budget lasts, otherwise block

THE STEP-GRANT LOG. ``step_num.txt`` is append-only: each token you add grants
that many further steps. Two notations, interchangeable at any time:

    1312        bare digits -- one grant per character  (1, 3, 1, 2)
    1,3,1,2,10  separated   -- one grant per token, so grants above 9 are possible

Appending "4" to either form grants 4 more steps; appending "10" in the separated
form grants 10. Tokens already honored are remembered, so the file is a history,
not a countdown: it only ever grows.

RESUME. The honored tokens and the unused budget are persisted to
``step_gate_state.json`` beside the other two files. Without it, resuming a run
whose log reads "13124" would re-grant every token in the history at once. A
working directory with no state file is a fresh run, so an initial "1" grants
exactly one step and then pauses -- which is the documented starting behaviour.

DOCTRINE: stdlib-only, like src/history_log.py. The parsing is pure
(``parse_tokens`` / ``reconcile``) so it is testable without touching the clock
or the filesystem.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src import var

STATUS_FILE = "status.txt"
STEP_NUM_FILE = "step_num.txt"
STATE_FILE = "step_gate_state.json"

MODE_RUN = "run"
MODE_STOP = "stop"
MODE_NUM = "num"
VALID_MODES = (MODE_RUN, MODE_STOP, MODE_NUM)

# Token separators for the "1, 3, 1, 2, 10" form: commas, semicolons, any whitespace.
_SEPARATORS = re.compile(r"[,;\s]+")


class MalformedStepLog(ValueError):
    """step_num.txt is not a readable grant log."""


# ---------------------------------------------------------------------------
# pure parsing
# ---------------------------------------------------------------------------
def parse_tokens(content: str) -> list[int]:
    """The grant log as a list of non-negative ints.

    Bare digits are one grant per character ("1312" -> [1, 3, 1, 2]); anything
    containing a separator is split on it, so multi-digit grants are possible
    ("1, 3, 1, 2, 10" -> [1, 3, 1, 2, 10]). Surrounding whitespace is ignored.
    Raises MalformedStepLog on anything else.
    """
    s = (content or "").strip()
    if not s:
        return []
    if _SEPARATORS.search(s):
        parts = [p for p in _SEPARATORS.split(s) if p]
    else:
        parts = list(s)
    out = []
    for p in parts:
        if not p.isdigit():
            raise MalformedStepLog(
                f"{p!r} is not a non-negative integer. Write the log as bare digits "
                f"('1312') or separated numbers ('1, 3, 1, 2, 10')."
            )
        out.append(int(p))
    return out


def reconcile(seen: list[int], tokens: list[int]) -> tuple[int, list[int], str | None]:
    """Compare the honored tokens with the log's current tokens.

    Returns ``(grant, new_seen, warning)``:
      * the log grew by one or more tokens -> grant their sum;
      * unchanged                          -> grant 0;
      * changed in place (not an extension) -> grant 0 and re-baseline, because
        honoring a rewrite would turn a typo into hundreds of real agent turns.
    """
    if tokens == seen:
        return 0, seen, None
    if tokens[:len(seen)] == seen:
        return sum(tokens[len(seen):]), list(tokens), None
    hint = ""
    if len(tokens) and len(seen) and tokens[0] != seen[0]:
        # The classic trap: the log was bare digits ("13124" -> 1,3,1,2,4) and a
        # separator was introduced ("13124,10" -> 13124,10), which re-reads the
        # whole history as one huge token. Refusing here is what stops that from
        # becoming thousands of real agent turns.
        hint = (" If you are switching to the separated notation, write the WHOLE "
                "log that way (e.g. '1,3,1,2,4,10'), not just the new token.")
    return 0, list(tokens), (
        f"step log was rewritten (was {seen}, now {tokens}); no steps granted. "
        f"The log is append-only -- add tokens to the end instead.{hint}"
    )


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def _dir(working_directory=None) -> Path:
    return Path(working_directory if working_directory is not None
                else var.my_WORKING_DIRECTORY)


def read_mode(working_directory=None) -> str:
    """The current mode, self-healing: a missing status.txt is recreated with
    the configured initial mode rather than deadlocking the run."""
    path = _dir(working_directory) / STATUS_FILE
    try:
        return path.read_text().strip().lower()
    except FileNotFoundError:
        mode = getattr(var, "RUN_GATE_INITIAL_MODE", MODE_NUM)
        print(f"[gate] {STATUS_FILE} is missing; recreating it as '{mode}'.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mode)
        return mode


def read_state(working_directory=None) -> dict:
    """``{'seen': [...], 'remaining': int}``; an absent/corrupt file means a
    fresh run (nothing honored yet, no budget)."""
    try:
        data = json.loads((_dir(working_directory) / STATE_FILE).read_text())
        return {
            "seen": [int(t) for t in data.get("seen", [])],
            "remaining": int(data.get("remaining", 0)),
        }
    except (OSError, ValueError, TypeError):
        return {"seen": [], "remaining": 0}


def write_state(state: dict, working_directory=None) -> None:
    d = _dir(working_directory)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps({"seen": state["seen"], "remaining": state["remaining"]}))
    tmp.replace(d / STATE_FILE)


def init_control_files(working_directory=None, mode: str | None = None) -> None:
    """Create the control files if absent. Never overwrites: a resume keeps the
    mode, the grant log and the honored-token state exactly as they were."""
    d = _dir(working_directory)
    d.mkdir(parents=True, exist_ok=True)
    mode = mode or getattr(var, "RUN_GATE_INITIAL_MODE", MODE_NUM)
    if mode not in VALID_MODES:
        raise ValueError(f"initial mode must be one of {VALID_MODES}, got {mode!r}")
    status = d / STATUS_FILE
    if not status.exists():
        status.write_text(mode)
    steps = d / STEP_NUM_FILE
    if not steps.exists():
        steps.write_text(str(getattr(var, "RUN_GATE_INITIAL_STEPS", "1")))
    print(f"[gate] mode '{read_mode(working_directory)}' "
          f"(edit {d / STATUS_FILE} to change; append to {d / STEP_NUM_FILE} to grant steps)")


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _try_consume(label: str, working_directory) -> bool:
    """One non-blocking attempt in 'num' mode: fold in any new grants, then
    spend one step if the budget allows. True when the caller may proceed."""
    state = read_state(working_directory)
    try:
        tokens = parse_tokens((_dir(working_directory) / STEP_NUM_FILE).read_text())
    except FileNotFoundError:
        print(f"[gate] {STEP_NUM_FILE} is missing; paused until it is restored.")
        return False
    except MalformedStepLog as e:
        print(f"[gate] {STEP_NUM_FILE} is malformed: {e}")
        return False

    grant, new_seen, warning = reconcile(state["seen"], tokens)
    if warning:
        print(f"[gate] {warning}")
    if grant or new_seen != state["seen"]:
        state = {"seen": new_seen, "remaining": state["remaining"] + grant}
        write_state(state, working_directory)
        if grant:
            print(f"[gate] granted {grant} step(s); budget is now {state['remaining']}.")

    if state["remaining"] <= 0:
        return False
    state["remaining"] -= 1
    write_state(state, working_directory)
    print(f"[gate] {label} proceeding; {state['remaining']} step(s) left in the budget.")
    return True


def wait_for_turn(label: str, working_directory=None, poll_seconds=None) -> None:
    """Block until this agent turn may run. Counts as one step in 'num' mode.

    Called at the top of every agent node, replacing the copy-pasted
    status.txt polling loops.
    """
    poll = poll_seconds if poll_seconds is not None else getattr(var, "RUN_GATE_POLL_SECONDS", 5)
    heartbeat = getattr(var, "RUN_GATE_HEARTBEAT_SECONDS", 60)
    waiting_since = None
    last_beat = 0.0

    while True:
        mode = read_mode(working_directory)
        if mode == MODE_RUN:
            return
        if mode == MODE_NUM and _try_consume(label, working_directory):
            return
        if mode not in VALID_MODES:
            # Fail safe: an unrecognised value pauses rather than running.
            reason = f"unrecognised status {mode!r} (expected one of {VALID_MODES})"
        elif mode == MODE_STOP:
            reason = "status is 'stop'"
        else:
            reason = "no step budget left"

        now = time.monotonic()
        if waiting_since is None:
            waiting_since = now
            last_beat = now
            print(f"[gate] {label} paused: {reason}. "
                  f"cwd: {_dir(working_directory)}")
        elif now - last_beat >= heartbeat:
            last_beat = now
            print(f"[gate] {label} still paused ({reason}); "
                  f"{int(now - waiting_since)}s so far.")
        time.sleep(poll)
