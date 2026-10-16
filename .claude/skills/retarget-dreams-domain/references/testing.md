# Testing a retargeted domain

Two tiers, both runnable without an API key or an LLM call.

## Tier 1 - pytest

| File | Covers |
|---|---|
| `test_<domain>_explog.py` | schema, chronological ids, duplicate refusal, progress projection, metric propagation and self-heal, every disposition status path |
| `test_<domain>_calculator.py` | the contract, a published reference case, physical consistency, the exception taxonomy, the subprocess hop, timeout, a dead child |
| `test_<domain>_tools.py` | each tool's happy path and every rejection, provenance chaining, the disposition read-out, query/sort/paginate |
| `test_worker_thread.py` | **the regression test that matters most, below** |
| `test_live_visualizer.py` | metrics reach the browser, internal columns do not, step attribution, resume |
| carried over | `test_past_steps.py`, `test_history_log.py`, `test_run_gate.py`, `test_disposition_messages.py` |

Mark anything that runs the real solver `@pytest.mark.slow` and register the marker in
`pytest.ini`.

## The worker-thread test

Without this you will not catch pitfall 1, and it is the one that kills a run.

```python
g = StateGraph(S)
g.add_node("tools", ToolNode([tools.run_<analysis>]))
g.add_edge(START, "tools"); g.add_edge("tools", END)
graph = g.compile(checkpointer=MemorySaver())          # the checkpointer is the trigger
for chunk in graph.stream({"messages": [call]},
                          {"configurable": {"thread_id": "1"}},
                          durability="sync"):
    out = chunk
```

Three assertions, all necessary:

1. the tool **actually ran off the main thread** - record
   `threading.current_thread() is threading.main_thread()` in a wrapper around the
   solver seam and assert it is `False`, so the test cannot silently stop exercising
   the bug;
2. the tool message says completed;
3. the number equals an in-process solve of the same **stored** design (pitfall 5).

Also keep a test asserting the in-process solve still fails off the main thread. If a
future library version makes it pass, that documents the subprocess hop as optional.

## Tier 2 - an executable notebook

`tests/<domain>_test_suite.ipynb`, generated from a builder script so it is
regenerable and reviewable as a diff. One section per module, one subsection per
function. Every check is a `@case` decorator that runs on definition, prints PASS or
FAIL, and is tallied in a final cell that asserts nothing failed, so it runs headlessly:

```python
import nbformat
from nbclient import NotebookClient
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": "."}}, allow_errors=True).execute()
```

Keep every fixture in one setup cell so the case cells are assertions only.

The notebook is not redundant with pytest: both `NameError`s from pitfall 3 were caught
here, because the pytest tool tests stubbed the affected path out.

## Habits that paid off

**Patch the seam the tool actually calls.** When the call path changes, update every
stub together (pitfall 4).

**Derive expectations from the same source the code used**, not from a constant you
believe is equivalent (pitfall 5).

**Test the infrastructure-fault path explicitly:** an environment error must leave the
candidate with no process row, `state == 'active'`, no disposition flag, and must be
retryable immediately afterwards.

**Test self-heal** by dropping the metric columns from a populated frame and calling
`update_log`.

**Assert on the agent-facing strings**, not just on state. The prompts tell the agent
to look for particular words ("NOT VIABLE", "could NOT BE RUN", "already has a completed
analysis"); if the wording drifts, the behaviour drifts with it.
