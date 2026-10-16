# Prompts, the graph driver, and the entry point

## The three prompts (`src/prompt.py`)

**Supervisor** - domain-light. It plans rounds, delegates, and submits a draft answer
for boss review. Two things to retarget: the sentence about how to pace the study given
the cost of a calculation, and one requirement telling it how to find the current best
candidates (name the metrics and the sort order).

**Worker** - carries the domain. Sections: Role, Objective, Your Capability (what EXPLOG
records, what the tools do, the static reference table rendered inline), Requirements
(numbered, see below), Methodology.

**Boss** - domain-free. One addition is worth making: tell it to spot-check quoted
numbers against EXPLOG with `query_explog` rather than trusting the draft.

### Worker requirements worth keeping

Renumber freely, but keep the substance:

- check EXPLOG first, then decide;
- the hypothesis behind every candidate goes into EXPLOG at entry;
- notes go on CANVAS **with the supporting reference id**;
- **the disposition workflow spelled out as read, analyze, interpret, hypothesize** -
  read the actual numbers from the processes table, then `get_disposition_info`, then
  `update_disposition_info` with an analysis that interprets rather than restates;
- the terminal-tag rules (failed means Abandon only);
- sweep deliberately, one parameter at a time;
- call `check_time` and wind down as the budget runs out;
- the reporting template.

### Methodology block

Give the worker the physics it needs to interpret results: what each output means, the
governing relation, and the trends it should expect to see. This is what lets it form
falsifiable hypotheses instead of guessing. Include known failure modes
("a 10 nm mismatch can drop transfer below 0.1") so a surprising number prompts a check
rather than a wrong conclusion.

Prompts are f-strings over `var`, so a knob like the target value appears in the prompt,
the objective and the metric from one definition.

## `planNexe2.py` - change only these seams

| Line region | Change |
|---|---|
| imports | `from src.prompt import ...`, `from src.<domain>_explog import EXPLOG` |
| `members = [...]` | the worker node name, e.g. `["FEM_Agent"]` |
| `myStep.required_tools` | the `Literal[...]` tool vocabulary |
| runtime `ToolList = [...]` inside the supervisor retry loop | **the same list again** |
| `POLLING_TOOLS` | tools the redundant-polling guard watches |
| `boss_tools`, `supervisor_tools`, `<domain>_tools` | the three tool lists |
| graph assembly | one `add_node` per agent |
| the HARD-path summariser prompt | names the quantities that live in EXPLOG |

> **The tool vocabulary appears twice** - once in the `myStep` pydantic `Literal` and
> once in the runtime validation list. They must agree or the supervisor will be told
> its own valid plan uses unknown tools.

Delete anything the new domain does not need. The OER conversion removed the queue-floor
flag on `myStep`, the wait-handback block in the supervisor node, the floor-coercion
block in the worker node, and the dead capability strings: about 290 lines.

Read the model name from config rather than hard-coding it.

## `invoke.py`

- `OBJECTIVE` as a module-level f-string over `var`, so budgets and targets cannot drift
  from the metrics.
- `EXPLOG.init(...)` with the domain's arguments.
- Delete the old domain's resume-time reconciliation steps.
- Keep: checkpoint pruning, the serde allowlist, fresh/resume/time-travel modes, the
  mid-round tool-level resume overlay, `durability="sync"`.
- `init_control_files(WORKING_DIRECTORY)` creates the pause-gate files without
  clobbering them on resume.

## Operational controls

`run_gate.wait_for_turn(name)` at the top of each agent node is the pause and
step-through gate: `status.txt` holds `run`, `stop` or `num`, and in `num` mode the
append-only `step_num.txt` grants further steps. One gate crossing is one step. State
persists to `step_gate_state.json` so a restart does not re-grant history.

For an unattended run set `var.RUN_GATE_INITIAL_MODE = "run"`.
