---
name: retarget-dreams-domain
description: Retarget the DREAMS agent framework from one scientific domain to another - replacing the experiment log schema, the simulation backend, the tools and the prompts while keeping the supervisor/worker/boss loop, CANVAS provenance, context compaction and checkpointing intact. Use when asked to point the agent at a new kind of calculation (FEM, MD, a different DFT target, a surrogate), to add a new simulation backend, or to understand which parts of this repo are domain-specific. Derived from the OER/VASP to FEMWell conversion.
---

# Retargeting DREAMS to a new domain

This repo is a plan-and-execute research agent. Roughly a third of it is domain-specific
and the rest is machinery you must not touch. This skill is the procedure for swapping
the domain, written from an actual conversion (OER electrocatalysis on VASP to
directional-coupler photonics on FEMWell) including every trap that conversion hit.

**Read `references/pitfalls.md` before writing code.** Most of the cost of the original
conversion was five bugs, three of which are invisible to the obvious tests and one of
which silently destroyed a production run. They are all preventable if you know them.

## The one-paragraph mental model

A **candidate** is one thing you are studying. A **process** is one calculation run on a
candidate. `EXPLOG` is a two-table relational frame holding both, and it is the agent's
memory of what has been tried. `CANVAS` is a key-value scratchpad plus an artifact
registry that gives every tool output an 8-character reference id, so claims chain back
to the tool call that produced them. A supervisor plans, a worker executes one plan step
with tools, a boss reviews the final answer. Everything is checkpointed to SQLite so a
30-day run survives being killed.

## Procedure

Work in this order. Each phase ends in something runnable, and phases 1-3 need no LLM.

### Phase 0 - decide the scope, and keep it small

Write down, before any code:

- **What is a candidate?** One parameter combination. Give it a chronological id
  (`dc_0001`), not a hash or an external database key, unless you genuinely have a
  candidate pool to screen.
- **What is a process?** Prefer **one job type and one process per candidate.**
- **Is the calculation synchronous?** If it finishes in seconds, run it in-process
  (via a subprocess, see Phase 5) and record the result immediately. Delete every
  queue, poll, wait and handback mechanism.

> **The single most important scoping rule.** The OER pipeline has four job types in a
> dependency cascade, a queue floor, a wait tool, forgotten-job detection and a
> supervisor handback protocol. All of that exists because a DFT job takes hours on
> SLURM. If your calculation takes seconds, **none of it should survive.** The first two
> conversion plans were rejected for mirroring that structure. Collapsing to one
> synchronous job type removed about 290 lines from the graph driver alone.

Pick your objective metrics now too (Phase 6) - they shape the explog schema.

### Phase 1 - map generic against domain-specific

See `references/architecture-map.md` for the full table. The short version:

| Never touch | Retarget | Delete |
|---|---|---|
| `relational_frame.py`, `myCANVAS.py`, `past_steps.py`, `history_log.py`, `run_gate.py`, `dag_visualizer.py`, `utils.py` | `var.py`, `tools.py`, `prompt.py`, `planNexe2.py`, `invoke.py`, `disposition_messages.py` | every domain module and its tests |
| `live_visualizer.py` (generic, but see pitfall 6) | the explog (`fem_explog.py`) and the backend (`fem_calculator.py`) are new files | |

### Phase 2 - the experiment log

Copy `relational_frame.py` unchanged. Write `src/<domain>_explog.py` with:

- a **candidates** table: design columns, hypothesis, notes, progress counters, `state`,
  the results copied up from the latest completed process, derived metrics, and the four
  disposition columns;
- a **processes** table: one row per calculation, status, note, error, results, metrics;
- `add_candidate`, `record_analysis`, `update_log`, the progress projection, and the
  disposition engine.

**Port the disposition engine verbatim in meaning.** It is the mechanism that forces the
agent to analyze results before moving on, and the prompts depend on its exact semantics.
See `references/explog-schema.md` for the contract and the self-healing rule.

### Phase 3 - the tools layer

Split `tools.py` into generic and domain halves. Generic: the CANVAS trio, `write_report`,
`search_artifacts`, `check_time`, `arXiv_search`. Domain: enter a candidate, run the
calculation, query and read the explog, the two disposition tools.

Every tool registers its output on CANVAS and returns a reference id. Tools that consume
a prior result take that id as an argument and reject unknown ones. See
`references/tools-layer.md`.

### Phase 4 - prompts, graph, driver

Three prompts: supervisor, worker, boss. The worker prompt carries the domain methodology.
In `planNexe2.py` change only the seams listed in `references/prompts-and-graph.md`, and
note that the tool vocabulary appears in **two** places that must agree.

### Phase 5 - the simulation backend

Define the contract first, implement a placeholder that returns random numbers in
plausible ranges, wire the whole agent end to end against it, and only then write the
real solver. See `references/calculator-contract.md`, which also contains the
threading trap that must be designed around rather than discovered.

### Phase 6 - objective metrics

Raw solver outputs are not an objective. Add derived, smaller-is-better metrics on both
tables, driven by a single target knob in `var.py`, and propagate them to all nine
consumers. See `references/metrics.md`.

### Phase 7 - tests

A fast pytest tier plus an executable notebook. The suite **must** include a case that
drives a tool through a checkpointed LangGraph with `durability="sync"`, or you will
not catch the worker-thread class of bug. See `references/testing.md`.

### Phase 8 - a live smoke run, then read the log

Point `WORKING_DIR` at a fresh directory, set `RUN_GATE_INITIAL_MODE` to `"run"` for an
unattended run, and execute one round. Then **read `hist/his_1.txt` and the explog
yourself.** Do not accept the agent's own account of what happened: in the original
conversion the agent correctly reported that every design failed but attributed it to
"a technical limitation of the simulation environment," which is true and useless. The
root cause took a reproduction script and four probes to localize.

## Reference files

| File | What it answers |
|---|---|
| `references/pitfalls.md` | **What goes wrong. Read this first.** |
| `references/architecture-map.md` | Which file is generic, which to retarget, which to delete |
| `references/explog-schema.md` | Table shapes, the derived projections, the disposition engine |
| `references/tools-layer.md` | Which tools to keep, the provenance and rejection rules |
| `references/calculator-contract.md` | The backend contract, the subprocess hop, the exception taxonomy |
| `references/prompts-and-graph.md` | The three prompts and every seam in the graph driver |
| `references/metrics.md` | Turning raw outputs into objectives, and the nine consumers |
| `references/testing.md` | The two test tiers, and the worker-thread regression test |
| `references/worked-example-oer-to-fem.md` | The original conversion end to end, with what each step cost |

## Verification checklist

- [ ] `grep -rn "OER\|VASP\|G(O)\|overpotential"` over the retargeted modules is empty
- [ ] `python -c "import src.planNexe2"` succeeds and does not import the old domain package
- [ ] the graph compiles and reports the expected node names
- [ ] a tool invoked through a checkpointed graph with `durability="sync"` succeeds
- [ ] an environment fault leaves the candidate untouched and retryable
- [ ] a resumed pre-metric checkpoint self-heals rather than raising
- [ ] the full pytest suite and the notebook both pass
