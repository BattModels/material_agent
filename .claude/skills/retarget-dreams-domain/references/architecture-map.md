# Architecture map: what is generic, what is domain-specific

## The three layers

1. **Orchestration** - the supervisor/worker/boss LangGraph loop, structured outputs,
   checkpointing, context compaction, the pause gate. Domain-blind except for a handful
   of named seams.
2. **Memory** - CANVAS (scratchpad + artifact provenance registry) and EXPLOG (the
   two-table relational frame). The *engine* is generic; the *schema* is domain-specific.
3. **Domain** - the experiment-log schema, the simulation backend, the tools, the prompts.

## File-by-file

| File | Verdict | Notes |
|---|---|---|
| `relational_frame.py` | **generic** | Copy unchanged. Tables with a primary key, declared parent/child relations, `RowView`/`RelatedView`. Originally lived in the nested OER package; copied to `src/` so the new domain does not depend on it. |
| `myCANVAS.py` | **generic** | Key-value canvas plus `result_registry` of artifacts. `register_tool_output` returns the 8-char id every tool hands back. |
| `past_steps.py` | **generic** | Bounded step ledger and compaction. Only `CLEARABLE_EXCLUDE_TOOLS` needs your side-effecting tool names. |
| `history_log.py` | **generic** | Rotating `hist/his_N.txt` dialogue log. |
| `run_gate.py` | **generic** | `status.txt` / `step_num.txt` pause and step-through control. |
| `dag_visualizer.py` | **generic** | Per-step provenance DAG HTML. |
| `utils.py` | **generic** | Config loader, the `Filter`/`SortSpec`/`df_query` grammar, numeric-token helpers for `search_artifacts`. |
| `live_visualizer.py` | **generic, but** | Reads the relational frame by attribute name, auto-detects any node ending in `Agent`. See pitfall 6 before trusting it. |
| `disposition_messages.py` | **retarget** | Pure formatters. Only `_JOB_LABEL` and `_unit_desc` know the job vocabulary. |
| `var.py` | **retarget** | Strip the old domain's knobs; add yours. Keep the disposition vocabulary and the history-rotation knobs. |
| `tools.py` | **retarget** | Split generic from domain. |
| `prompt.py` | **retarget** | Three prompts. |
| `planNexe2.py` | **retarget** | Change only the named seams. |
| `invoke.py` | **retarget** | Objective text, `EXPLOG.init`, remove old reconciliation steps. |
| `<domain>_explog.py` | **new** | |
| `<domain>_calculator.py` | **new** | |
| `<domain>_materials.py` etc. | **new** | Any static domain reference data. |

## What the OER conversion deleted

23 modules and 9 test files, including: the whole nested `GNoME_DREAMS_OER_screening`
dependency, `forgotten_jobs.py`, `disposition_reconcile.py`, `aq_gnome_candidate_sync.py`,
`explog_mode_guard.py`, the DFT/HPC/MD agents and their graphs, and several abandoned
prototypes (`newCANVAS.py`, `planNexe2-oer.py`, `stubbed_functions*.py`, `tmp.py`).

Deleting aggressively is worth it: `from src.tools import *` in the graph driver means
anything left behind stays reachable and can be silently referenced by a stale prompt.

## What survived untouched and why it matters

- **Checkpoint/resume**, including mid-round tool-level resume via `StateSyncMiddleware`.
- **Context compaction**, the bounded `past_steps` ledger with report-driven digests.
- **The disposition gate**, which forces analysis before the agent can move on.
- **CANVAS artifact provenance**, the chain from a claim back to the tool call.
- **The boss review gate.**

These are the parts that make a long autonomous run survivable. None of them needed to
change, and changing them is almost never the right answer to a domain problem.
