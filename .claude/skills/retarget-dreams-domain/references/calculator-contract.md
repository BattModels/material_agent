# The simulation backend

## The contract is the only thing the rest of the code may depend on

```python
RESULT_KEYS = ("...", ...)                      # must equal the explog's RESULT_COLUMNS

def compute(params: dict) -> dict:              # pure, in-process, main-thread
    """Every RESULT_KEYS entry as a finite float. Raises DesignNotViable or
    CalculationError."""

def solve_in_subprocess(params: dict, timeout_seconds=None) -> dict:
    """What the TOOL calls. Same return, same exceptions, out of process."""
```

Keep `compute` importable and working in-process: the child entry point, the unit tests
and the notebook all call it directly. Only the tool goes through the subprocess.

## Build it in two passes

**Pass 1, a placeholder.** Return random numbers in physically plausible ranges, with
any internal consistency the domain demands (in the FEM case the coupling length had to
agree with the supermode splitting, so the placeholder derived one from the other). Wire
the entire agent end to end against this. It costs an hour and it decouples "is the
agent plumbing right" from "is the physics right", which are very different debugging
problems.

**Pass 2, the real solver.** Replace only this file. Prove it against a published
reference case with a regression test that pins the numbers.

## Exception taxonomy - get this right or corrupt the log

```python
class DesignNotViable(ValueError):   """The DESIGN is bad. A real, reportable result."""
class CalculationError(RuntimeError): """The ENVIRONMENT failed. Says nothing about the design."""
class CalculationTimeout(CalculationError): """Exceeded the wall-clock limit."""
```

Raise `DesignNotViable` from parameter validation and from every physics sanity check
(below cutoff, no guided mode, wrong mode family available). Raise `CalculationError`
for non-finite results and anything the environment did. See pitfall 2 for what happens
if you conflate them.

## The subprocess hop

Why, in one line: LangGraph runs tool calls on worker threads once a checkpointer is
attached, and C libraries that install signal handlers cannot do that. See pitfall 1.

Shape that works:

- parent writes `params.json` into a `TemporaryDirectory`, runs
  `[sys.executable, "-m", "src.<domain>_calculator", params, result]` with `cwd` at the
  repo root and `PYTHONPATH` prefixed with it, under `subprocess.run(timeout=...)`;
- the child **always exits 0** when it managed to write an envelope
  `{"ok": true, "result": {...}}` or `{"ok": false, "kind": "not_viable"|"error", "message": "..."}`;
- the parent re-raises the right exception type from `kind`;
- **no result file** means the child died (segfault, OOM) - raise `CalculationError`
  carrying the return code and the stderr tail;
- `subprocess.TimeoutExpired` becomes `CalculationTimeout`.

Timeout lives in `var.<DOMAIN>_SOLVE_TIMEOUT_SECONDS`; 0 disables it. Always set one:
solve cost varies by 10x with no warning (pitfall 9).

A `__main__` block that reads params and writes the envelope doubles as a standalone
CLI and as a future batch or SLURM entry point.

## Internal knobs stay internal

Meshing parameters, tolerances, retry policy: derive them from the physical inputs and
keep them out of the tool signature. The agent should reason about the design, not about
discretization. In the FEM case the simulation window and mesh resolution are computed
from the index contrast and the core size, with a named policy function
(`_mesh_policy`) that is unit-tested directly.

Expose them only if the study is *about* them.

## Sanity invariants belong in the solver

Add cheap physical checks the solver itself can verify, and act on them:

- a derived quantity must bracket, bound or otherwise agree with another;
- if it does not, retry with a larger domain or a finer discretization before trusting it;
- if it still does not, raise `DesignNotViable` with a message that names the numbers.

The FEM version retries with progressively larger simulation windows
(`PAD_RETRY_SCALES`) and only then gives up. This is also where you discover your noise
floor (pitfall 7): sweep a case with a known analytic answer and find where the model
stops reproducing it.
