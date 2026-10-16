# Objective metrics

## Why raw outputs are not enough

The solver returns physical quantities. The study has a goal. Without derived metrics
the agent has to do the arithmetic in its head every time it ranks candidates, and it
has nothing to sort on. In the OER domain these were the overpotentials and the
deviation of an adsorption energy from its ideal value; in the FEM domain, the deviation
of a coupling length from a target and an excess loss in dB.

## Design rules

**Smaller is better, always.** Make every metric a distance or a loss so the agent can
sort ascending without thinking about direction. Say "smaller is better" in the column
documentation and in the prompt.

**One target knob, one definition.** Put the target in `var.py`. The metric function,
the objective text and the prompt all read it, so prose and arithmetic cannot drift.

```python
TARGET_COUPLING_LENGTH_UM = 20.0
```

**A pure function, separately testable:**

```python
METRIC_COLUMNS = ("coupling_length_deviation_um", "excess_loss_dB")

def derived_metrics(result: dict, target: float) -> dict:
    ...
```

**Guard the domain of the maths.** `-10*log10(p)` needs `p` clamped away from zero, or
one perfect-failure row produces `inf` and poisons every aggregate. Clamp to a floor
(`1e-12` gives 120 dB, which reads as "catastrophic" and stays finite).

**Watch the sign of zero.** `-10*log10(1.0)` is `-0.0`, which renders as `-0.0` in
tables and plots. Take the absolute value at the display or analysis boundary.

**Recompute, do not copy.** Metrics land on the process row at record time *and* are
recomputed for every completed process on each `update_log`, then the latest is copied
onto the candidate. This makes a changed target back-fill and an old checkpoint
self-heal.

## The nine places a metric must reach

Adding a metric is not one edit. The OER-to-FEM work touched all of these:

1. `var.py` - the target knob
2. `<domain>_explog.py` - `METRIC_COLUMNS`, `derived_metrics`, both table schemas, the
   recompute, the copy-up, the self-heal
3. `invoke.py` - `EXPLOG.init(target=...)` and the objective text
4. `tools.py` - the result formatter the agent reads after each run
5. `tools.py` - the `query_explog` docstring, with direction and sort idiom
6. `prompt.py` - the worker's capability list and methodology block
7. `prompt.py` - the supervisor's "how to find the best candidates" requirement
8. `config/<domain>_available_tools.yaml` - the tool guide written to CANVAS
9. `live_visualizer.py` - threshold styling, and `planNexe2.py`'s summariser prompt

A checklist is cheaper than discovering at review time that the supervisor cannot see
the number the whole study optimises.

## Group-level metrics stay out of the schema

Bandwidth, fabrication tolerance, a Pareto front: these need several candidates and
belong in the agent's analysis and reports, not as columns. Only per-candidate scalars
go in the table.
