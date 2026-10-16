# Pitfalls

Every one of these was hit for real during the OER to FEM conversion. They are ordered
by how much damage they did. The first one destroyed a production run.

---

## 1. LangGraph runs tool calls on worker threads, and C libraries hate that

**Symptom.** Every calculation fails with the same error, here
`ValueError: signal only works in main thread of the main interpreter`. The agent
dutifully records nine failed designs and abandons all of them.

**Cause.** `invoke.py` compiles the graph with a `SqliteSaver` and streams with
`durability="sync"`. Under those conditions LangGraph executes node tasks, and therefore
the inner agent's tool calls, on a `ThreadPoolExecutor`. The mesher called
`gmsh.initialize()`, which defaults to `interruptible=True` and calls
`signal.signal(SIGINT, SIG_DFL)`. CPython forbids `signal.signal` off the main thread.

**Why it is easy to miss.** A bare LangGraph node runs on `MainThread`. The thread pool
only appears once a checkpointer is attached. So a quick probe graph, every unit test,
and every notebook cell will all pass while production fails 100 % of the time. Verify
with:

```python
inner = StateGraph(S); inner.add_node("tools", ToolNode([probe]))
ic = inner.compile(checkpointer=MemorySaver())          # <- the checkpointer is the trigger
for _ in ic.stream({...}, {"configurable": {"thread_id": "1"}}, durability="sync"): pass
# probe reports ThreadPoolExecutor-2_0, is_main=False
```

**Fix.** Run the solver in a child process. The child works on its own main thread, so
the problem disappears, and you also contain crashes and get a timeout. Measured cost in
the real case: about 2 s of interpreter and import startup on top of a 3 to 25 s solve.

An in-process monkeypatch of `gmsh.initialize` to default `interruptible=False` also
works and costs nothing, but a segfault or a hang in the C library then kills the whole
study. Prefer the subprocess; keep the monkeypatch in mind if startup cost ever matters.

**Do not** try to disable this by setting a module attribute. `interruptible` is a
keyword parameter with a default, not a module global, so `gmsh.interruptible = False`
silently does nothing. Check the actual signature with `inspect.signature`.

**Guard against regressions:** `tests/test_worker_thread.py`.

---

## 2. An environment fault recorded as a scientific failure

**Symptom.** Nine physically fine designs permanently marked `state=failed` and
`decision=Abandon`, with that false conclusion written into the agent's round report.

**Cause.** The tool wrapped the solver in a bare `except Exception` and recorded every
failure as a failed process. The terminal-tag gate then enforces, correctly, that a
failed candidate may only be dispositioned `Abandon`. So one threading bug laundered
itself into nine scientific verdicts.

**Fix.** Two distinct exception families, and the tool branches on the type:

| exception | meaning | tool behaviour |
|---|---|---|
| `DesignNotViable(ValueError)` | the design is bad: no guided mode, no contrast, bad parameters | record a failed process; agent dispositions it |
| `CalculationError(RuntimeError)` | the environment is broken: crash, no result | record **nothing**; candidate stays untouched and retryable |
| `CalculationTimeout(CalculationError)` | exceeded the wall-clock limit | same as above |

The agent-facing message for the second family must say explicitly that this says
nothing about the design, that nothing was recorded, and that a repeated failure should
be reported to the supervisor rather than retried forever.

**General principle.** Anywhere the agent's memory records a verdict, ask what happens
when the infrastructure fails. A tool that cannot distinguish "the answer is no" from
"I could not get an answer" will poison the experiment log.

---

## 3. Reassembling a module loses its imports, and only runtime notices

**Symptom.** `NameError: name 'json' is not defined`, then `name 'sys'`, from deep inside
a tool, long after import succeeded.

**Cause.** `tools.py` was rebuilt by concatenating extracted blocks under a freshly
written header. The header did not import everything the pasted bodies referenced. Python
does not check this at import time.

**Fix.** After assembling a module from parts, grep for bare module references and
reconcile against the header:

```bash
grep -oE "\b(sys|json|copy|uuid|np|os|re|math|time|random|pickle|subprocess)\.[a-zA-Z_]+" src/tools.py \
  | cut -d. -f1 | sort -u
```

Both of these were caught by the executable notebook, not by pytest, because the pytest
tool tests stubbed the affected code path out.

---

## 4. Moving a seam silently disables every stub that patched the old one

**Symptom.** After switching the tool from calling `compute_coupled_modes` directly to
calling `solve_in_subprocess`, four tool tests started running the real solver. They got
slower and returned different numbers.

**Cause.** The tests monkeypatched `tools.compute_coupled_modes`. That name was still
imported and still existed, so the patch succeeded and did nothing.

**Fix.** When you move the call seam, grep for every `monkeypatch.setattr` naming the old
seam and update them together. A stub that no longer intercepts fails loudly only if the
real thing is slow or networked; if it is fast and local, the test may still pass while
testing nothing.

---

## 5. Hard-coding an expected value that the tool derives differently

**Symptom.** An end-to-end test asserted `coupling_length == 23.3539...` and got
`23.2466...`.

**Cause.** The test used the physical constants from the upstream documentation, while
the tool resolves refractive indices from the project's own material library by
interpolation. Both were right; the comparison was wrong.

**Fix.** In end-to-end tests, derive the expectation from the same source the tool used
(`EXPLOG.design_of(cid)`) rather than from a literature constant. This turns a brittle
assertion into a genuinely stronger one: it proves the out-of-process result is identical
to the in-process one.

---

## 6. Generic code that is nonetheless wrong for your topology

Two bugs in the live visualizer, neither OER-specific, both invisible until exercised:

- **Worker steps attributed to the supervisor.** The driver unwraps the
  `{node_name: update}` envelope *before* calling `viz.on_event`, so the worker's node
  name never reaches the visualizer and the stale "current agent" guess wins. Fixed by
  learning the worker's name from the `agent` field of the supervisor's own plan steps,
  which keeps it domain-agnostic.
- **Internal columns shipped to the browser.** `hide_columns` was applied at render time
  only, so the full disposition record, including every analysis paragraph, was
  serialized into `live_data.js` and embedded in the saved HTML on every flush. Drop
  hidden columns server-side in `_collect_explog`.

**Principle.** "Generic" means it does not mention your domain. It does not mean it is
correct for your agent topology or your schema. Exercise it with real data.

---

## 7. Numerical artifacts the agent will confidently misinterpret

**Symptom.** A perfectly symmetric coupler reports 17.5 dB of excess loss. The agent
abandons it and reports that large gaps are lossy.

**Cause.** The two waveguides are meshed independently, so geometrically identical cores
get effective indices differing by about 1e-4. The physics code reads that as a real
detuning. The noise is constant while the coupling coefficient decays exponentially, so
past a threshold the result is pure artifact.

**Lessons that generalize.**

- **Know your backend's noise floor and where it dominates.** Sweep a case whose answer
  you know analytically (here: identical waveguides must give exactly 0 dB) and find the
  parameter value where the model breaks.
- **Do not silently clamp.** `min(1.0, raw)` turned an unphysical 1.0002 into a clean 1.0,
  which is benign, but it would equally hide a raw 1.5 that signals a real bug. Clamp
  quietly within tolerance, flag beyond it.
- **If the objective tells the agent to sweep into the bad region, it will,** and it will
  draw a false conclusion with full confidence. Either fix the model, or bound the
  objective, or teach the tool to mark results as untrustworthy.

Tracked as issue 1 in `KNOWN_ISSUES.md` at the repo root.

---

## 8. Mode-family and domain-correctness traps in the solver itself

Specific to eigenmode solvers, but the shape recurs in any physics backend:

- **The lowest eigenvalue is not the mode you want.** On thick or square cores the TM
  family becomes near-degenerate with TE, so the two lowest modes of the pair were the
  TE and TM symmetric supermodes rather than the TE pair. Solve a few extra modes and
  filter by a polarization fraction.
- **A truncated domain silently changes the answer.** Weakly confined modes hit the
  simulation window boundary. Add a sanity invariant (the supermodes must bracket the
  isolated modes) and retry with a larger window before trusting the result.
- **Fail loudly below cutoff.** A design with no guided mode must raise `DesignNotViable`,
  not return a meaningless number.

---

## 9. Cost is per-design, not per-position, and varies wildly

An early claim that the solver had a 20 s warm-up followed by 3 s runs was wrong. It came
from comparing three *different* designs. The same design repeated is uniformly slow.
Profiling showed meshing at 0.17 s and the eigensolver at 0.96 s or 11.9 s depending on
the design, a 10x spread with no warning.

**Consequences.** Always bound a solve with a timeout. Never estimate study throughput
from one design. When you quote a cost, say which design produced it.

---

## 10. Scope creep in the plan, twice

The first conversion plan proposed a multi-stage pipeline, a per-candidate wavelength
index and a job handler. The second kept an enumerated design-space table. Both were
rejected as too complicated for the actual goal. The accepted plan was: one job type,
one process per candidate, chronological ids, no pool, no queue.

**Principle.** Mirror the old domain's *structure* only where the new domain's physics
demands it. Hours-long SLURM jobs justify a queue; a three-second local solve does not.
