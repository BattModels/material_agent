# workflow_audit

Simple, transparent audit of a DREAMS agentic run. Reads the dialogue log
`his.txt` and writes one CSV row per **real tool call**, resolved by **round**
and **agent** (supervisor / worker / boss), with calculation type + outcome for
DFT submissions and the round's elapsed time. A notebook reads only the CSV.

Lives under `src/` so it ships with the repo, but **nothing in the workflow
runtime imports it** (`invoke.py` imports only `src.planNexe2`, `src.tools`,
`src.myCANVAS`, `src.var`) — so it cannot affect a running study.

## How to run

From the `material_agent` directory:

```bash
python -m src.workflow_audit.history_parser \
    production_run_27-05-2026/his.txt \
    -o src/workflow_audit/outputs/tool_calls.csv
```

A byte-accurate progress bar advances over the (multi-GB) file and a per-agent
summary prints at the end. Then open `tool_calls.ipynb` (it reads
`outputs/tool_calls.csv`; needs only `pandas`, `matplotlib`, `jupyter`).

## How the parser works (6 rules)

A **round is one agent turn**: the round number increments whenever the active
agent changes (supervisor → worker → supervisor → boss → …). Consecutive markers
for the *same* agent do not start a new round — this matters because the worker
banner `Agent … is processing!!!!!` is printed **twice per worker turn** (once
before its retry loop, once inside), so counting raw worker markers would double
every worker turn.

The parser streams `his.txt` line by line, holding the current **round**,
**agent**, **round start-time**, and the most recent **open tool call**:

1. `supervisor is processing!!!!! Current time: T` → if agent changed, round += 1; agent = supervisor; round start-time = T.
2. `Agent <name> is processing!!!!!` → if agent changed, round += 1; agent = worker. **Worker turns carry no timestamp, so the round's time is left blank.**
3. `<name> is processing!!!!! Current time: T` (not supervisor) → if agent changed, round += 1; agent = boss; round start-time = T.
4. `  <tool> (toolu…` → one tool call: skip the 3 control tools (`Act`, `wokerResponse`, `BossReview`); otherwise emit a row and hold it open.
5. `    calculation_type: <type>` → if the open call is `submit_dft_job`, record its calc type.
6. `Name: <tool>` matching the open call, then the next non-empty line → outcome = `error` if it starts with `Tool error`, else `success`.

Parallel tool calls are disabled in the workflow, so only one call is ever open
— the pairing of a call with its args/result is unambiguous. `his.txt` has no
`@@@` redundant-polling echo blocks, so each call appears exactly once.

## CSV columns

| column | meaning |
|---|---|
| `round` | round number; increments on each agent turn (supervisor / worker / boss) |
| `agent` | supervisor / worker / boss (constant within a round) |
| `tool` | tool name (control tools excluded) |
| `calc_type` | submit_dft_job only: bulk/surface/O/OH adsorption (else blank) |
| `outcome` | success / error (blank if no result line was seen) |
| `round_start_h` | round start, hours since project start (float); **blank for worker rounds** |
| `round_start_raw` | e.g. `8 days, 23:07:17`; **blank for worker rounds** |
| `line_no` | line in his.txt, for traceability |

## Limitations

- **outcome** detects framework *exceptions* (the uniform `Tool error:` wrapper).
  A tool that *returns* an error string without raising counts as `success`.
- A **blank `outcome`** on a non-placeholder row means the call was seen but no
  result/error line followed — i.e. the run **crashed mid-call**. On resume the
  workflow usually re-invokes that call, so the same work can appear twice: the
  crashed attempt (blank outcome) and the replay (often `error`, rejected by the
  duplicate guard). Treat blank-outcome rows as interrupted, not completed.
- **time**: only supervisor and boss turns carry a timestamp in his.txt, so
  `round_start_h` / `round_start_raw` are **blank for every worker round** (left
  empty rather than guessed). Worker timing would have to be recovered from the
  `total time elapsed…` that the worker's own tools (`check_time`,
  `wait_for_update`) emit mid-turn — out of scope for this audit.
- `his.txt` is **cumulative across all runs** (including resume replays), so
  rounds are campaign-cumulative.
- A round with **no real tool calls** (a boss turn — the boss has no tools; or a
  worker turn that only returns a `wokerResponse`) is represented by a single
  **placeholder row** with an empty `tool` (and blank `calc_type`/`outcome`), so
  every round stays present in the table with a zero tool-call count and the
  `round` column has no gaps. Filter `tool != ""` when counting real tool calls.
