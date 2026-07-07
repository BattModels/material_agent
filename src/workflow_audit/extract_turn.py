"""Extract one or more turns (rounds) of raw his.txt text into a small file.

his.txt is a cumulative, multi-GB dialogue log -- too big to read directly.
This tool streams it once, finds the round boundaries using the exact same
markers as history_parser.py (so a "turn" number here lines up 1:1 with the
`round` column in tool_calls.csv), and copies the raw lines of the requested
turn(s) verbatim into a new, much smaller txt file.

A "turn" == a "round" as defined in history_parser.py: it increments on every
agent change (supervisor -> worker -> supervisor -> boss -> ...), numbered
cumulatively across the whole file (a restart's "=== Session started ==="
banner does not reset the count). Turn N's content runs from round N's marker
line (inclusive) to the line right before round N+1's marker (exclusive), or
EOF if N is the last round in the file. Content before round 1 belongs to no
round and is never extracted.

Usage:
    python -m src.workflow_audit.extract_turn his.txt 842
    python -m src.workflow_audit.extract_turn his.txt 12 15 -o range.txt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:                      # progress bar is optional
    tqdm = None

from .history_parser import RE_BOSS, RE_SUP, RE_WORK, parse_elapsed_hours


def _find_round_boundaries(path, start_turn: int, end_turn: int,
                            progress: bool = True):
    """Stream the file once; return a list of (round, agent, start_line,
    start_raw, lines) dicts for each round in [start_turn, end_turn]. Stops
    reading as soon as the round after end_turn is seen -- no need to scan
    the whole file unless the target range is near the end.

    Raises ValueError if end_turn (and therefore possibly start_turn) falls
    outside the rounds actually found in the file.
    """
    round_idx = 0
    agent = None
    current = None      # dict for the round currently being captured
    found = []
    range_closed = False   # True once a round after end_turn is seen

    total = os.path.getsize(path)
    bar = tqdm(total=total, unit="B", unit_scale=True,
               desc=f"scanning for turn(s) {start_turn}-{end_turn}"
               ) if (progress and tqdm) else None
    if progress and not tqdm:
        print(f"scanning for turn(s) {start_turn}-{end_turn} "
              f"({total / 1e9:.2f} GB)...")

    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            if bar:
                bar.update(len(raw))
            line = raw.decode("utf-8", "replace").rstrip("\n")

            new_agent = None
            start_raw = ""
            m = RE_SUP.match(line)
            if m:
                new_agent, start_raw = "supervisor", m.group(1)
            elif RE_WORK.match(line):
                new_agent = "worker"
            else:
                m = RE_BOSS.match(line)
                if m and m.group(1) != "supervisor":
                    new_agent, start_raw = "boss", m.group(2)

            if new_agent is not None and new_agent != agent:
                if current is not None:
                    found.append(current)
                round_idx += 1
                agent = new_agent
                if bar:
                    bar.set_postfix(round=round_idx, agent=agent, refresh=False)
                if round_idx > end_turn:
                    range_closed = True
                    break
                current = None
                if round_idx >= start_turn:
                    current = dict(round=round_idx, agent=agent,
                                    start_line=lineno, start_raw=start_raw,
                                    lines=[])
                    if progress and not tqdm:
                        print(f"  found turn {round_idx} (agent={agent}) "
                              f"at line {lineno}")

            if current is not None:
                current["lines"].append(raw)

    if bar:
        bar.close()

    if not range_closed:
        if current is not None:
            found.append(current)
        raise ValueError(
            f"turn {end_turn} not found -- only {round_idx} round(s) "
            f"in {path}")

    return found


def extract_turns(path, start_turn: int, end_turn: int, out_path,
                   progress: bool = True):
    """Write the raw text of rounds [start_turn, end_turn] to out_path,
    each preceded by a one-line '# turn ...' header. Raises ValueError
    (without creating out_path) if the requested turns don't exist."""
    rounds = _find_round_boundaries(path, start_turn, end_turn,
                                     progress=progress)

    if progress:
        print(f"writing {len(rounds)} turn(s) -> {out_path}")

    with open(out_path, "wb") as out:
        for rnd in rounds:
            n_lines = len(rnd["lines"])
            end_line = rnd["start_line"] + n_lines - 1
            start_h = (parse_elapsed_hours(rnd["start_raw"])
                       if rnd["start_raw"] else None)
            header = (
                f"# turn {rnd['round']} | agent={rnd['agent']} | "
                f"{path} lines {rnd['start_line']}-{end_line} | "
                f"round_start="
                f"{rnd['start_raw'] or '-'}"
                f"{f' ({start_h:.3f}h)' if start_h is not None else ''}\n"
            )
            out.write(header.encode("utf-8"))
            out.writelines(rnd["lines"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract one turn (round), or an inclusive range of "
                    "turns, of raw his.txt text into a small new file.")
    ap.add_argument("history", help="path to his.txt")
    ap.add_argument("turn", type=int, help="turn (round) number")
    ap.add_argument("end_turn", type=int, nargs="?", default=None,
                     help="if given, extract the inclusive range "
                          "[turn, end_turn] into one merged file")
    ap.add_argument("-o", "--output", default=None,
                     help="output path (default: his_turn_<N>.txt or "
                          "his_turns_<N>-<M>.txt next to the source file)")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    start_turn = args.turn
    end_turn = args.end_turn if args.end_turn is not None else args.turn
    if start_turn > end_turn:
        ap.error(f"start turn {start_turn} is after end turn {end_turn}")
    if start_turn < 1:
        ap.error("turn numbers are 1-indexed")

    src = Path(args.history)
    if args.output:
        out_path = Path(args.output)
    elif start_turn == end_turn:
        out_path = src.with_name(f"his_turn_{start_turn}.txt")
    else:
        out_path = src.with_name(f"his_turns_{start_turn}-{end_turn}.txt")

    try:
        extract_turns(src, start_turn, end_turn, out_path,
                       progress=not args.no_progress)
    except ValueError as exc:
        ap.error(str(exc))

    print(f"wrote turn(s) {start_turn}-{end_turn} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
