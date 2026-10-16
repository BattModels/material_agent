# The experiment log

`src/<domain>_explog.py` defines a `RelationalFrame` with two related tables and the
derived projections the agent and the prompts depend on.

## Candidates table

```
candidate_id          str, chronological: dc_0001, dc_0002, ...
<design columns>      the parameters that IDENTIFY a candidate
reason_or_hypothesis  str, why this candidate, what it tests
notes                 str
n_started             Int64, calculations begun
n_finalized           Int64, of those, terminal (failures included)
state                 str, 'active' | 'failed'
<result columns>      copied up from the latest COMPLETED process
<metric columns>      derived, smaller-is-better (see metrics.md)
disposition_record    object, append-only list of dicts        [hidden from the agent]
decision              str, the latest Decision
ready_for_disposition_update  boolean, the write lock          [hidden from the agent]
needs_disposition_update      boolean, derived
```

## Processes table

```
process_id    Int64, globally unique and monotonic across candidates
candidate_id  str
job_type      str
status        str, 'completed' | 'failed'   (add pending/running only if async)
processNote   str
error         str, the failure message
<result columns>, <metric columns>
```

## Rules that the rest of the system depends on

**Design columns identify a candidate.** `find_duplicate_design` compares them with a
numeric tolerance and a case-insensitive match on names, and `add_candidate` refuses an
exact duplicate by returning the existing id. Without this the agent re-enters the same
design every round.

**Counts are per logical unit, not per raw sub-job.** If one logical step launches three
sub-jobs, it still counts as one started and one finalized once all three are terminal.
The prompts quote these counters and the agent filters on them.

**`n_X_started == 0` must mean "ready for stage X".** In a multi-stage pipeline, gate a
counter to `<NA>` until the candidate is eligible, so the filter idiom works. With a
single stage this collapses to a plain 0/1 and you should not invent stages to preserve it.

**`state` is derived, and 'failed' means every attempt failed.** Recompute it from the
processes table; a later success flips it back to active. (The OER version latched
`failed` permanently because a failed bulk relaxation really is terminal. Decide which
your domain needs.)

**Results are copied up, metrics are recomputed.** Copy the latest completed process's
raw results onto the candidate row, but *recompute* the derived metrics rather than
copying them, so changing the target knob back-fills correctly.

## The disposition engine - port this verbatim in meaning

It is what stops the agent accumulating unexamined results. Four moving parts:

1. **Units.** `_candidate_finalized_units` enumerates terminal work as
   `{job_type, ids:[...]}`, collapsing batches so citing any one sub-job covers the unit.
2. **Coverage.** `candidate_outstanding` returns the units not yet cited by any
   disposition record. A unit is covered iff a record's `Summarized_process_id`
   intersects its ids.
3. **The write lock.** `get_disposition_info` sets `ready_for_disposition_update = True`
   and returns what needs analyzing; `update_disposition_info` refuses with `locked`
   unless it is set, and **only a successful write clears it**, so a rejected attempt
   can be retried without re-reading.
4. **The terminal-tag gate** (in the tool layer, not the engine): a failed candidate may
   only be `Abandon`; a terminal tag on a live candidate is refused.

Status vocabulary returned by `update_disposition_info`: `ok`, `locked`,
`invalid_decision`, `foreign_ids`, `non_terminal_ids`, `incomplete`, `unknown_candidate`.
`disposition_messages.py` turns each into one agent-facing sentence that states both the
reason and the fix. Keep all seven; the prompts and tests reference them.

## Self-healing is mandatory

A run resumed from a checkpoint written before you added a column must not crash. Every
derived column is created if absent on each recompute:

```python
@staticmethod
def _ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
```

Call it before any read of those columns, including inside `record_analysis`, because a
resumed process may write before it ever recomputes.

## Persistence

Pickle both dataframes atomically (temp file plus `os.replace`) to
`<working_dir>/explog.pkl` after every mutation. This is a human-inspectable side
record; the authoritative state for resume is the LangGraph checkpoint, which carries
both dataframes as state channels.
