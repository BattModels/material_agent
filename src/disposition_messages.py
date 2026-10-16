"""Pure formatters for the disposition tools.

These turn the STRUCTURED result of ``EXPLOG.get_disposition_info`` /
``EXPLOG.update_disposition_info`` into the single agent-facing sentence that the
``@tool`` in ``src/tools.py`` hands back to the worker. The prose lives here
(rather than in ``src/tools.py``) so it can be unit-tested without importing
``src.tools``. Imports only the stdlib plus the stdlib-only ``src.var``.
"""

from typing import Any, Dict, Iterable, List

from src import var


# Machine job_type -> readable label used in agent messages.
_JOB_LABEL = {
    "coupled_mode_analysis": "coupled-mode analysis",
}


def _unit_desc(unit: Dict[str, Any]) -> str:
    """Render ONE finalized unit as a fragment for the agent.

    A "unit" is one piece of finished work from ``candidate_outstanding`` --
    ``{job_type, ids}``. The fragment names the calculation and which process
    ids belong to it, e.g.

        coupled-mode analysis: process id 4
    """
    label = _JOB_LABEL.get(unit.get("job_type"), unit.get("job_type"))
    ids = list(unit.get("ids", []))
    if len(ids) == 1:
        ids_part = f"process id {ids[0]}"
    else:
        ids_part = "process ids " + ", ".join(str(i) for i in ids) + " (cite any one)"
    return f"{label}: {ids_part}"


def format_get_disposition(candidate_id: str, outstanding: Dict[str, Any]) -> str:
    """Build the get_disposition_info message: tell the agent what it must
    review and analyze for ``candidate_id``.

    ``outstanding`` is the dict returned by ``candidate_outstanding`` --
    ``{must_cover, latest_disposition, has_finalized}`` -- or
    ``{"unknown_candidate": True}`` when the candidate is not in the log.
    """
    if outstanding.get("unknown_candidate"):
        return (
            f"No candidate '{candidate_id}' exists in the experiment log. "
            f"Check the candidate id and try again."
        )

    must = outstanding.get("must_cover", []) or []
    latest = outstanding.get("latest_disposition")

    parts: List[str] = []
    if latest is not None:
        parts.append(
            f"Latest disposition on record for {candidate_id}: "
            f"Decision={latest.get('Decision')}; "
            f"Analysis_and_implications={latest.get('Summary')}; "
            f"Latest Future_plan when this candidate was dispositioned last time={latest.get('Future_plan')}."
        )
    else:
        parts.append(f"No prior disposition for {candidate_id}.")

    if must:
        listing = "; ".join(_unit_desc(u) for u in must)
        parts.append(
            f"Candidate {candidate_id} has finished results that still need a "
            f"disposition: {listing}. Review and analyze these results, base your "
            f"Analysis_and_implications on what they show and imply, and pass the "
            f"process ids you used as Analyzed_process_id to update_disposition_info."
        )
    else:
        parts.append(
            f"Candidate {candidate_id} has no finished results awaiting a "
            f"disposition; you may still record one disposition for it."
        )

    return " ".join(parts)


def format_update_disposition(
    candidate_id: str,
    result: Dict[str, Any],
    allowed_decisions: Iterable[str],
) -> str:
    """Build the update_disposition_info message from the structured result.

    On rejection the sentence states the reason AND how to resolve it; on
    success it confirms. ``result`` is the dict returned by
    ``EXPLOG.update_disposition_info`` (keys: ``status`` and, per status,
    ``decision`` / ``ids`` / ``missing``).
    """
    status = result.get("status")

    if status == "ok":
        return (
            f"Recorded disposition for candidate {candidate_id} "
            f"(Decision: {result.get('decision')}). All finished results for "
            f"this candidate are now dispositioned."
        )

    if status == "locked":
        return (
            f"Cannot record a disposition for {candidate_id} yet: call "
            f"get_disposition_info('{candidate_id}') first to review its "
            f"outstanding results, analyze latest results and previous findings,"
            f"then call update_disposition_info again."
        )

    if status == "invalid_decision":
        allowed = ", ".join(allowed_decisions)
        return (
            f"'{result.get('decision')}' is not a valid Decision. Choose one of: "
            f"{allowed}. Then call update_disposition_info again."
        )

    if status == "foreign_ids":
        ids = ", ".join(str(i) for i in result.get("ids", []))
        return (
            f"These process ids do not belong to candidate {candidate_id}: "
            f"{ids}. Cite only {candidate_id}'s own finished results (see "
            f"get_disposition_info), then call update_disposition_info again."
        )

    if status == "non_terminal_ids":
        ids = ", ".join(str(i) for i in result.get("ids", []))
        return (
            f"These process ids are not finished yet and cannot be analyzed: "
            f"{ids}. Omit them, then call update_disposition_info again."
        )

    if status == "incomplete":
        missing = ", ".join(str(i) for i in result.get("missing", []))
        return (
            f"Disposition not recorded: candidate {candidate_id} still has "
            f"finished results you have not cited: {missing}. Cite at least one "
            f"process id from each, then call update_disposition_info again."
        )

    if status == "unknown_candidate":
        return (
            f"No candidate '{candidate_id}' exists in the experiment log. "
            f"Check the candidate id and try again."
        )

    return f"Unexpected disposition result for {candidate_id}: {status!r}."


def evaluate_terminal_tag_gate(
    *,
    decision: str,
    state: Any,
    has_in_flight: bool,
    terminal_decisions: Iterable[str],
    active_decisions: Iterable[str],
) -> str | None:
    """Decide whether a NEW disposition ``Decision`` is allowed for a candidate,
    given the candidate's state. Returns a rejection sentence, or ``None`` to allow
    the write. PURE (no EXPLOG): the caller passes the facts in.

    Rules:
      - A FAILED candidate may ONLY be "Abandon" -- any other decision is rejected.
      - A terminal tag (Abandon/Sufficient) on a non-failed candidate is allowed
        only when nothing is in flight for it.
      - An active priority on a non-failed candidate is always allowed (None).
    """
    if str(state) == "failed":
        if decision != "Abandon":
            return (
                f"Candidate has FAILED, so its only valid disposition is 'Abandon' "
                f"(you gave '{decision}'). Re-record this disposition with "
                f"Decision='Abandon'."
            )
        return None
    if decision in tuple(terminal_decisions) and has_in_flight:
        active = ", ".join(active_decisions)
        return (
            f"'{decision}' is a terminal decision, but this candidate still has "
            f"a calculation in flight. Let it finish first, or record an active "
            f"priority instead ({active})."
        )
    return None


def format_needs_disposition_reminder(candidate_ids: Iterable[str]) -> str:
    """Appended to compute-tool results when other candidates have finished
    results the agent has not yet analyzed. Advisory, never a refusal."""
    ids = sorted(str(c) for c in candidate_ids)
    if not ids:
        return ""
    return (
        "\nREMINDER: the following candidates have finished results that are not "
        f"yet dispositioned: {', '.join(ids)}. Before you end your turn, review "
        "each (query_explog / read_explog), then call get_disposition_info and "
        "update_disposition_info to record what the results show and imply."
    )
