# Autonomous OER catalyst screening

A multi-agent system that runs DFT catalyst screening unattended for months: it
picks candidate materials, launches and supervises VASP jobs on an HPC cluster,
reads the results, and decides what to compute next — forming and discarding its
own hypotheses as it goes.

The target is the oxygen evolution reaction in acid, the stability bottleneck in
green-hydrogen electrolysis. Candidates come from
[AQ-GNoME](https://github.com/Matminator/AQ-GNoME), a Pourbaix-stability screen
over Google DeepMind's GNoME structures.

**[Browse the results](https://matminator.github.io/OER-GNoME/)** — every
candidate considered, every calculation launched, and the agent's reasoning for
each decision, including the paths it abandoned.
**Data (frozen paper snapshot):** [doi.org/10.11583/DTU.33265158](https://doi.org/10.11583/DTU.33265158)

## How it works

A supervisor plans, a worker executes against the cluster, and a third agent
gates the conclusion — approving the study as finished or sending it back, which
stops it declaring victory with budget unspent. The plan is mutable state, and
the standing objective is a file the operator can rewrite between rounds.

**EXPLOG** is relational: one row per candidate with its hypothesis and decision
history, one per calculation with its site and free energies — the agent's
memory and the published dataset both. **CANVAS** is a provenance graph over tool
outputs, so claims in a report must cite the artifact IDs behind them. State
lives on disk, so a campaign survives driver restarts and cluster reboots.

LangGraph and LangChain; VASP via pysqa on SLURM, with partition routing by
system size and convergence recovery. The DFT pipeline is a fixed ladder (bulk →
surface → O\* → OH\*); the agent chooses which materials and sites to spend it
on. Entry point `invoke.py`, agent layer `src/`; the chemistry engine is a
separate package, [GNoME_DREAMS_OER_screening](https://github.com/Matminator/GNoME_DREAMS_OER_screening).

## Provenance

A divergent branch of [DREAMS](https://arxiv.org/abs/2507.14267) (Wang et al.),
which supplies the plan–execute loop and CANVAS. Added here: the exploration log
and its decision model, the OER and VASP domain layer, HPC scheduling and queue
management, the durable-state rework behind month-long resumability, operator
steering, and the published record.
