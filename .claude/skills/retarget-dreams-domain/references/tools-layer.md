# The tools layer

## Keep these generic tools unchanged

`inspect_my_canvas`, `read_my_canvas`, `write_my_canvas`, `write_report`,
`search_artifacts`, `check_time`, `arXiv_search`.

Two caveats when you move them into a fresh module: reconcile the imports (pitfall 3),
and make `arXiv_search` degrade to a clear message if the optional `ursa` package is
absent rather than failing at import.

## Domain tools

| Tool | Job |
|---|---|
| `list_<reference_data>` | Expose the static domain table (materials, potentials, ...) and return a reference id the agent can cite. |
| `enter_<candidate>_in_log` | Validate parameters, resolve derived inputs, refuse duplicates, create the candidate. |
| `run_<analysis>` | Run the calculation, record the process, format the result. |
| `query_explog` | Filter/sort/paginate either table. Its docstring is the schema documentation the agent reads. |
| `read_explog` | Everything about one candidate, including its latest disposition. |
| `get_disposition_info` / `update_disposition_info` | The analysis gate. |

## Rules

**Every tool registers its output and returns an 8-character reference id.** Use
`CANVAS.register_tool_output(tool_name, args, value, description, reasons,
parent_result_ids, metadata)`. The `parent_result_ids` argument is what makes the
provenance DAG work.

**Tools that act on a prior result require its id and verify it:**

```python
if ref and CANVAS.get_artifact(ref) is None:
    return f"Error: Reference ID {ref} not found in CANVAS."
```

This is how a design entry chains back to the literature search or the data table that
motivated it.

**Return a string, not an exception, for user-level problems.** Bad parameters, unknown
materials, duplicates, missing references: all return an explanatory message. Reserve
raised exceptions for genuine faults, which the middleware wraps.

**Every rejection message states both the reason and the fix.** "Error: wavelength 2.0
is outside the study window 1.3-1.6 um" is useful; "invalid input" is not.

**Put the domain schema in the `query_explog` docstring.** The agent reads it as its data
dictionary. Document every column, its dtype, its meaning, and for metrics say
explicitly which direction is better and how to sort on it. This docstring is long on
purpose.

**Validate before mutating.** `enter_design_in_log` checks references, positivity,
range, physical consistency and duplicates before touching EXPLOG, so a rejected call
leaves no trace.

**Do not recompute completed work.** `run_<analysis>` returns early if the candidate
already has a completed process, but *allows* a retry when only failures exist.

## The standing-duty reminder

After a successful calculation, append a reminder listing *other* candidates with
unanalyzed results:

```python
others = [c for c in EXPLOG.candidates_needing_disposition() if c != candidate_id]
outStr += format_needs_disposition_reminder(others)
```

Cheap, and it keeps the agent's analysis current without a blocking gate.

## `search_artifacts` and dict-valued results

If a tool stores a dict as its artifact value, make sure the search categoriser indexes
it as key-value atoms rather than one opaque string, or those results become unfindable
by content. The fix is one branch in `_category_atoms` routing dicts through
`_atoms_from_args`.
