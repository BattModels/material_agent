"""Parse a DREAMS his.txt dialogue log into a per-tool-call table.

One row per real tool call: the round it occurred in, which agent made it
(supervisor / worker / boss), the round's elapsed start-time, and -- for
submit_dft_job -- the calculation type and success/error outcome.

A "round" is one agent turn: the round number increments whenever the active
agent changes (supervisor -> worker -> supervisor -> boss -> ...). Consecutive
markers for the same agent (the worker banner is printed twice per turn, and
more on a retry) do NOT start a new round. Only supervisor and boss turns carry
a timestamp in his.txt; worker turns have none, so their time fields are left
blank. The round counter never resets on a restart (a "Session started" banner
does not close the round it lands in), so restarts show up as the `session`
column changing while `round` stays put.

The parser is a small state machine over the lines of his.txt; rule S and
rules 1-6 are marked inline below. his.txt holds the agent dialogue only (no
@@@ redundant-polling echoes), so every tool call appears exactly once.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                      # progress bar is optional
    tqdm = None

from src.history_log import list_hist_files

# Structured-output "tools" that are control flow (a supervisor decision, a
# worker return, a boss verdict), not real tools -- excluded from the counts.
CONTROL_TOOLS = {"Act", "wokerResponse", "BossReview"}

# Line patterns, validated against production_run_27-05-2026/his.txt
RE_SUP  = re.compile(r"^supervisor is processing!!!!! Current time:\s*(.+?)\s*$")
RE_WORK = re.compile(r"^Agent\s+\S+\s+is processing!!!!!")
RE_BOSS = re.compile(r"^(\S+) is processing!!!!! Current time:\s*(.+?)\s*$")
RE_TOOL = re.compile(r"^\s{2}([A-Za-z_]\w*) \(toolu")     # pretty "Tool Calls" line
RE_CALC = re.compile(r"^\s+calculation_type:\s*(\w+)")    # pretty Args block
RE_NAME = re.compile(r"^Name:\s*(\w+)")                   # Tool Message header

# wait_for_update's two "genuinely waited" return paths (src/tools.py) embed
# fixed, unambiguous phrasing -- neither appears in any of its refusal
# messages (src/disposition_messages.py), so matching these two and falling
# back to "refused" needs no enumeration of refusal text.
RE_WAIT_DONE    = re.compile(r"time waited:\s*(\d+)\s*hours?\s*and\s*(\d+)\s*minutes")
RE_WAIT_TIMEOUT = re.compile(r"waiting for\s*(\d+)\s*minutes with no update")

# invoke.py writes this unconditionally on every launch (fresh, "ow", digit
# time-travel, "replay", or plain resume) -- see write_history() call right
# before the main graph.stream() loop. Session numbering starts at 1 and
# bumps on each occurrence, so it survives crash/relaunch cycles without any
# new logging: the marker was already there, this just stops discarding it.
RE_SESSION = re.compile(r"^=== Session started at (.+) ===\s*$")

# timedelta repr: "8 days, 23:07:17.605402" or "0:10:06.123456"
_ELAPSED = re.compile(r"(?:(\d+)\s+days?,\s*)?(\d+):(\d{2}):(\d{2})(?:\.\d+)?\s*$")


def parse_elapsed_hours(text: str):
    """'8 days, 23:07:17' / '0:10:06' -> float hours (None if unparseable)."""
    m = _ELAPSED.match(text.strip().rstrip("."))
    if not m:
        return None
    days, hh, mm, ss = m.groups()
    return (int(days or 0) * 86400 + int(hh) * 3600 + int(mm) * 60 + int(ss)) / 3600.0


def _parse_wait_result(line: str):
    """Classify a wait_for_update result line: ("waited", minutes),
    ("timeout", minutes), or ("refused", None).

    `timeout`'s minutes is the requested `patience` value baked into the log
    text, not an independently re-measured elapsed time -- the only number
    the tool's own log ever records for that case."""
    m = RE_WAIT_DONE.search(line)
    if m:
        return "waited", int(m.group(1)) * 60 + int(m.group(2))
    m = RE_WAIT_TIMEOUT.search(line)
    if m:
        return "timeout", int(m.group(1))
    return "refused", None


def _resolve_sources(path) -> list[Path]:
    """Accept either a single his.txt (a run with no hist/ dir yet) or a
    hist/ directory (detected via is_dir()). For the latter, also pick up
    a sibling legacy his.txt (WORKING_DIR/his.txt, frozen at the resume that
    created hist/) as the chronologically-first source, if present -- the
    old flat file predates everything under hist/, never the reverse."""
    p = Path(path)
    if p.is_dir():
        files = list_hist_files(p)
        legacy = p.parent / "his.txt"
        if legacy.is_file():
            files = [legacy] + files
        if not files:
            raise ValueError(f"No his.txt or his_<N>.txt files found for {p}")
        return files
    return [p]


def parse_history(path, progress: bool = True) -> pd.DataFrame:
    """Parse a run's dialogue log into one row per real tool call.

    path: a single his.txt file (a run with no hist/ dir yet), or a hist/
    directory (a run migrated to the rotating writer) -- in the latter case
    its legacy sibling his.txt, if any, is included as the chronologically
    -first source (see _resolve_sources).
    progress: show a byte-accurate tqdm progress bar while streaming.

    Returns a DataFrame with one row per real tool call (plus one
    placeholder row per tool-call-free round), columns: round, session,
    agent, tool, calc_type, outcome, wait_status, wait_minutes,
    round_start_h, round_start_raw, line_no -- see the module docstring /
    README for what each means.
    `session` counts restarts: 1 for the first session (everything up to and
    including the first "Session started" banner, if any -- most his.txt
    files open with one, since invoke.py writes it as literally its first
    line), +1 each time a LATER banner appears (df.groupby("session")["round"]
    .min() gives the round each session resumed into).
    `wait_status`/`wait_minutes` are populated only for wait_for_update rows
    (blank/NaN elsewhere, like `calc_type`): "waited"/"timeout" with the
    minutes actually waited, or "refused" (no wait happened, minutes is NaN)
    -- see _parse_wait_result.
    """
    files = _resolve_sources(path)
    rows = []
    round_idx = 0            # rules 1-3: increments when the active agent changes
    session = 1               # rule S: increments on each "Session started" banner
                              # that follows real content (round_idx > 0) -- a
                              # banner before any round has even started is just
                              # session 1 announcing itself, not a restart
    agent = None             # supervisor / worker / boss
    start_h = None           # current round's elapsed start-time, hours (blank on worker turns)
    start_raw = ""           # ... and its raw string (blank on worker turns)
    open_call = None         # most recent tool-call row, awaiting calc/outcome
    expect_result = False    # True after the matching "Name: <tool>" header
    round_has_rows = False   # did the current round emit any real tool-call row?
    round_marker_line = 0    # his.txt line of the marker that started the round

    def flush_empty_round():
        """If the round that just ended produced no real tool calls, emit one
        placeholder row (blank `tool`) so the round still appears in the table
        with a zero tool-call count instead of being missing entirely."""
        if agent is not None and not round_has_rows:
            rows.append(dict(round=round_idx, session=session, agent=agent,
                             tool="", calc_type="", outcome="",
                             wait_status="", wait_minutes=None,
                             round_start_h=start_h, round_start_raw=start_raw,
                             line_no=round_marker_line))

    total = sum(f.stat().st_size for f in files)
    bar = tqdm(total=total, unit="B", unit_scale=True,
               desc="parsing his.txt") if (progress and tqdm) else None

    lineno = 0
    for fpath in files:
        with open(fpath, "rb") as fh:
            for raw in fh:
                lineno += 1
                if bar:
                    bar.update(len(raw))
                line = raw.decode("utf-8", "replace").rstrip("\n")

                # rule S -- session-restart banner: bump the session counter,
                #           unless no round has started yet (that's session 1
                #           announcing itself, not a restart). Independent of
                #           the round/agent state machine below.
                if RE_SESSION.match(line):
                    if round_idx > 0:
                        session += 1
                    continue

                # rule 1 -- supervisor marker: new round only if the agent changed;
                #           carries the timestamp.
                m = RE_SUP.match(line)
                if m:
                    if agent != "supervisor":
                        flush_empty_round()          # close the previous round
                        round_idx += 1
                        round_has_rows = False
                        round_marker_line = lineno
                    agent = "supervisor"
                    start_raw = m.group(1)
                    start_h = parse_elapsed_hours(start_raw)
                    open_call, expect_result = None, False
                    continue

                # rule 2 -- worker marker: new round only if the agent changed. Worker
                #           turns have NO timestamp, so the time fields are left blank.
                if RE_WORK.match(line):
                    if agent != "worker":
                        flush_empty_round()
                        round_idx += 1
                        round_has_rows = False
                        round_marker_line = lineno
                    agent = "worker"
                    start_raw = ""
                    start_h = None
                    open_call, expect_result = None, False
                    continue

                # rule 3 -- boss marker: new round only if the agent changed (not the
                #           supervisor); carries the timestamp.
                m = RE_BOSS.match(line)
                if m and m.group(1) != "supervisor":
                    if agent != "boss":
                        flush_empty_round()
                        round_idx += 1
                        round_has_rows = False
                        round_marker_line = lineno
                    agent = "boss"
                    start_raw = m.group(2)        # group(1) is the agent name, group(2) the time
                    start_h = parse_elapsed_hours(start_raw)
                    open_call, expect_result = None, False
                    continue

                # rule 4 -- tool-call invocation: emit a row (skip control tools)
                m = RE_TOOL.match(line)
                if m:
                    tool = m.group(1)
                    expect_result = False
                    if tool in CONTROL_TOOLS:
                        open_call = None
                        continue
                    open_call = dict(round=round_idx, session=session, agent=agent,
                                     tool=tool, calc_type="", outcome="",
                                     wait_status="", wait_minutes=None,
                                     round_start_h=start_h, round_start_raw=start_raw,
                                     line_no=lineno)
                    rows.append(open_call)
                    round_has_rows = True
                    continue

                # rule 5 -- calculation_type for an open submit_dft_job call
                m = RE_CALC.match(line)
                if (m and open_call is not None
                        and open_call["tool"] == "submit_dft_job"
                        and not open_call["calc_type"]):
                    open_call["calc_type"] = m.group(1)
                    continue

                # rule 6a -- result header matching the open call -> next line decides
                m = RE_NAME.match(line)
                if (m and open_call is not None
                        and m.group(1) == open_call["tool"]
                        and open_call["outcome"] == ""):
                    expect_result = True
                    continue

                # rule 6b -- first non-empty line after that header = the result
                if expect_result and open_call is not None and line.strip():
                    open_call["outcome"] = ("error" if line.lstrip().startswith("Tool error")
                                            else "success")
                    if open_call["tool"] == "wait_for_update":
                        open_call["wait_status"], open_call["wait_minutes"] = _parse_wait_result(line)
                    expect_result = False
                    continue

    flush_empty_round()   # the final round, if it had no tool calls

    if bar:
        bar.close()

    return pd.DataFrame(rows, columns=["round", "session", "agent", "tool",
                                       "calc_type", "outcome", "wait_status",
                                       "wait_minutes", "round_start_h",
                                       "round_start_raw", "line_no"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="his.txt -> per-tool-call CSV")
    ap.add_argument("history", help="path to his.txt (old runs) or the hist/ directory (new runs)")
    ap.add_argument("-o", "--output", default="outputs/tool_calls.csv")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    df = parse_history(args.history, progress=not args.no_progress)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nwrote {len(df)} tool-call rows -> {out}")
    print(df.groupby("agent").size().rename("calls").to_string())
    sub = df[df.tool == "submit_dft_job"]
    if len(sub):
        print("submit_dft_job calc_type:", dict(sub.calc_type.value_counts()))
        print("submit_dft_job outcome :", dict(sub.outcome.value_counts()))
    wf = df[df.tool == "wait_for_update"]
    if len(wf):
        print("wait_for_update status:", dict(wf.wait_status.value_counts()))
        print("wait_for_update avg minutes by status:",
              wf.groupby("wait_status")["wait_minutes"].mean().round(1).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
