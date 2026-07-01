"""Resume-time disposition reconciliation (Part 3).

The interrupted production run recorded dispositions under the OLD vocabulary and
WITHOUT the terminal-tag gate, so its checkpoint can contain decisions that are no
longer valid:
  - a terminal tag (Abandon/Sufficient) on a candidate that is NOT fully settled
    (it is a forgotten job, or still has in-flight work) -- it was abandoned/called
    done prematurely;
  - a failed candidate whose current decision is not Abandon;
  - the orphaned legacy tag ``Investigating`` (dropped from the vocabulary).

``reconcile_dispositions`` is called once on every resume (idempotent) to bring the
loaded candidates frame into line with the new rules. Two mechanics:
  - DELETE the trailing disposition record(s) whose current Decision is invalid, so
    the candidate's finished work becomes uncovered again and it is re-flagged for a
    fresh (honest) disposition (needs_disposition_update -> True);
  - RENAME an orphaned ``Investigating`` (on a non-failed candidate) to the neutral
    active default, preserving its coverage (needs stays False).

This lives in the parent (not the nested engine) because it reuses
``find_forgotten_jobs``; it imports only pandas + the stdlib-only ``src.var`` +
``src.forgotten_jobs`` (no ``src.tools``), so it is fast to unit-test.
"""

import pandas as pd

from src import var
from src.forgotten_jobs import find_forgotten_jobs


def classify_disposition_reconciliation(
    *,
    decision,
    state,
    is_forgotten: bool,
    has_in_flight: bool,
    terminal_decisions,
    default_active: str,
):
    """Decide what to do with ONE candidate's CURRENT (trailing) disposition.

    Returns ``(action, new_value)`` where action is ``"delete"`` (pop the trailing
    record -> re-evaluate the next), ``"rename"`` (rewrite the trailing Decision to
    ``new_value``), or ``"leave"`` (valid; stop). PURE -- the caller supplies the
    candidate facts. Mirror of the Part 2 gate, but for an existing decision.
    """
    if str(state) == "failed":
        # A failed candidate may ONLY be Abandon; anything else is invalid.
        return ("leave", None) if decision == "Abandon" else ("delete", None)
    # Non-failed:
    if decision == "Investigating":
        # Orphaned legacy active tag -> map to the new neutral active default.
        return ("rename", default_active)
    if decision in tuple(terminal_decisions):
        settled = not (is_forgotten or has_in_flight)
        return ("leave", None) if settled else ("delete", None)
    # An active priority (or any already-valid decision) stands.
    return ("leave", None)


def reconcile_dispositions(explog, *, go_dev_oh_threshold=None, apply: bool = True) -> dict:
    """Reconcile every candidate's current disposition against the new rules.

    Idempotent: once no invalid trailing decision remains, a re-run is a no-op.
    Returns ``{candidate_id: {deleted, renamed, new_decision}}`` for every candidate
    that changed (this is also the dry-run report). With ``apply=False`` nothing is
    written to the frame or persisted -- the report is computed read-only.
    """
    threshold = (
        go_dev_oh_threshold if go_dev_oh_threshold is not None
        else var.GO_DEV_OH_THRESHOLD
    )
    # Refreshes the derived progress/state columns as a side effect, so 'state'
    # (used below) is current before we read it.
    forgotten = {it["candidate_id"] for it in find_forgotten_jobs(explog, threshold)}
    cdf = explog.relational_frame.candidates.df
    pdf = explog.relational_frame.processes.df
    jh = explog.job_handler
    if "disposition_record" not in cdf.columns or len(cdf) == 0:
        return {}

    actions: dict = {}
    for cid in cdf["candidate_id"].tolist():
        idx = cdf.index[cdf["candidate_id"] == cid][0]
        rec = cdf.at[idx, "disposition_record"]
        if not isinstance(rec, list) or not rec:
            continue
        state = cdf.at[idx, "state"] if "state" in cdf.columns else None
        statuses = pdf.loc[pdf["candidate_id"] == cid, "status"].tolist()
        has_in_flight = any(
            not jh._is_terminal_process_status(s) for s in statuses
        )
        is_forgotten = cid in forgotten

        new_rec = list(rec)          # shallow copy; inner dicts never mutated
        deleted, renamed = 0, False
        while new_rec:
            action, new_value = classify_disposition_reconciliation(
                decision=new_rec[-1].get("Decision"),
                state=state,
                is_forgotten=is_forgotten,
                has_in_flight=has_in_flight,
                terminal_decisions=var.DISPOSITION_TERMINAL_DECISIONS,
                default_active=var.DISPOSITION_DEFAULT_ACTIVE,
            )
            if action == "delete":
                new_rec.pop()
                deleted += 1
                continue
            if action == "rename":
                new_rec[-1] = {**new_rec[-1], "Decision": new_value}
                renamed = True
            break

        if deleted or renamed:
            actions[cid] = {
                "deleted": deleted,
                "renamed": renamed,
                "new_decision": (new_rec[-1]["Decision"] if new_rec else None),
            }
            if apply:
                cdf.at[idx, "disposition_record"] = new_rec
                cdf.at[idx, "decision"] = (
                    new_rec[-1]["Decision"] if new_rec else pd.NA
                )

    if apply and actions:
        # Deleting a covering record uncovers finalized work -> needs recompute.
        jh._recompute_disposition_state()
        explog._save_pickle()
    return actions