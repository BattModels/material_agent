# Worked example: OER/VASP to FEMWell directional couplers

The conversion this skill was written from, in order, with what each step cost.

## Before

Screening oxide catalysts for the oxygen evolution reaction. Candidates came from the
AQ-GNoME database; each ran a four-stage DFT cascade on SLURM through VASP
(bulk relaxation, surface relaxation, O adsorption, OH adsorption), hours per job.
Objectives were adsorption free energies and overpotentials. Around 45,000 lines in
`src/`, plus a nested `GNoME_DREAMS_OER_screening` package.

## After

Screening two-waveguide directional couplers. A candidate is one parameter combination
(gap, two widths, wavelength, core and cladding material). One synchronous FEM
coupled-mode analysis per candidate, seconds each. Objectives are the deviation of the
coupling length from a target and an excess loss in dB.

## The steps

**1. Read the code base and classify it.** Two sessions of reading produced the
generic/retarget/delete table now in `architecture-map.md`.

**2. Scope, rejected twice.** Plan A proposed a mesh/mode-solve/CMT pipeline with
wavelength as a second index dimension. Plan B kept an 800-row enumerated design-space
table. Both were rejected as too complicated. Plan C - one job type, one process per
candidate, chronological ids, no pool, no queue - was accepted. **This was the single
highest-leverage decision in the conversion.**

**3. Foundations.** `relational_frame.py` copied from the nested package. `var.py`
rewritten (60 lines, from 156). `utils.py` trimmed to the config loader, the query
grammar and the numeric helpers. `fem_materials.py` added for the refractive-index
table. `disposition_messages.py` trimmed to the new job vocabulary.

**4. `fem_explog.py`.** Two tables, the progress projection, the disposition engine
ported in meaning. About 400 lines.

**5. `tools.py` rebuilt.** Generic blocks extracted and re-concatenated under a new
header - which is where the missing `json` and `sys` imports came from. Fourteen tools,
down from about forty.

**6. Prompts, graph, driver.** `planNexe2.py` lost 290 lines net. `invoke.py` rewritten
lean. 23 modules and 9 test files deleted.

**7. First green:** imports clean, graph compiles, 72 pytest cases pass. No physics yet -
the calculator was still returning random numbers.

**8. Notebook suite** written as a generated, executable `@case` notebook. It
immediately found the two missing imports.

**9. The real calculator.** FEMWell and gmsh installed, the reference example
reproduced (coupling length 35.306 um against the documented 35.31). Then three
robustness problems found by sweeping other platforms: the wrong mode family on thick
cores, truncated weakly-confined modes, and the need to fail loudly below cutoff.

**10. Objective metrics** added and propagated to all nine consumers.

**11. The live visualizer** audited against a populated explog; two bugs fixed.

**12. The pause gate** refactored out of three copy-pasted blocks into `run_gate.py`.

**13. First live run - total failure.** Every design died with
`signal only works in main thread of the main interpreter`. The agent abandoned nine
good designs. Diagnosis took a reproduction script and four probes; the decisive one
showed that a LangGraph node runs on `MainThread` without a checkpointer and on
`ThreadPoolExecutor-2_0` with one.

**14. The fix:** subprocess execution with a timeout, the `DesignNotViable` versus
`CalculationError` split, and `test_worker_thread.py`.

**15. Second live run - success.** 29 designs over 7 rounds, 23 computed, 4 correctly
rejected as not viable, a best design 1.33 um from target at 0 dB.

**16. Analysis of that run** surfaced the last problem: a symmetric coupler reporting
17.5 dB of loss at a 1 um gap. Mesh asymmetry fakes a detuning that swamps a weak
coupling coefficient. The agent had already drawn a false conclusion from it. Documented
as issue 1 in `KNOWN_ISSUES.md`, deliberately not yet fixed.

## Final shape

16 modules in `src/`, 10 test files plus the notebook, 132 pytest cases and 101 notebook
cases passing.

## If you are doing this again

The order that worked: foundations, explog, tools, prompts and graph, placeholder
backend, **end-to-end green**, real backend, metrics, live run, analysis. Getting to
end-to-end green on a placeholder before writing any physics is what kept the plumbing
bugs and the physics bugs separable.

The two days that were avoidable: the threading bug, which pitfall 1 now prevents, and
the two rejected plans, which Phase 0 now prevents.
