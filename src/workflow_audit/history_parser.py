"""Parse a DREAMS his.txt dialogue log into a per-tool-call table.

One row per real tool call: the round it occurred in, which agent made it
(supervisor / worker / boss), the round's elapsed start-time, and -- for
submit_dft_job -- the calculation type and success/error outcome.

A "round" is one agent turn: the round number increments whenever the active
agent changes (supervisor -> worker -> supervisor -> boss -> ...). Consecutive
markers for the same agent (the worker banner is printed twice per turn, and
more on a retry) do NOT start a new round. Only supervisor and boss turns carry
a timestamp in his.txt; worker turns have none, so their time fields are left
blank.

The parser is a small state machine over the lines of his.txt; the 6 rules
are marked inline below. his.txt holds the agent dialogue only (no @@@
redundant-polling echoes), so every tool call appears exactly once.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                      # progress bar is optional
    tqdm = None

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

# timedelta repr: "8 days, 23:07:17.605402" or "0:10:06.123456"
_ELAPSED = re.compile(r"(?:(\d+)\s+days?,\s*)?(\d+):(\d{2}):(\d{2})(?:\.\d+)?\s*$")


def parse_elapsed_hours(text: str):
    """'8 days, 23:07:17' / '0:10:06' -> float hours (None if unparseable)."""
    m = _ELAPSED.match(text.strip().rstrip("."))
    if not m:
        return None
    days, hh, mm, ss = m.groups()
    return (int(days or 0) * 86400 + int(hh) * 3600 + int(mm) * 60 + int(ss)) / 3600.0


def parse_history(path, progress: bool = True) -> pd.DataFrame:
    """Stream his.txt once; return one row per real tool call."""
    rows = []
    round_idx = 0            # rules 1-3: increments when the active agent changes
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
            rows.append(dict(round=round_idx, agent=agent, tool="",
                             calc_type="", outcome="",
                             round_start_h=start_h, round_start_raw=start_raw,
                             line_no=round_marker_line))

    total = os.path.getsize(path)
    bar = tqdm(total=total, unit="B", unit_scale=True,
               desc="parsing his.txt") if (progress and tqdm) else None

    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            if bar:
                bar.update(len(raw))
            line = raw.decode("utf-8", "replace").rstrip("\n")

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
                open_call = dict(round=round_idx, agent=agent, tool=tool,
                                 calc_type="", outcome="",
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
                expect_result = False
                continue

    flush_empty_round()   # the final round, if it had no tool calls

    if bar:
        bar.close()

    return pd.DataFrame(rows, columns=["round", "agent", "tool", "calc_type",
                                       "outcome", "round_start_h",
                                       "round_start_raw", "line_no"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="his.txt -> per-tool-call CSV")
    ap.add_argument("history", help="path to his.txt")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
