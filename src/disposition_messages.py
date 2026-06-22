"""Pure formatters for the disposition tools.

These turn the STRUCTURED result of ``EXPLOG.get_disposition_info`` /
``EXPLOG.update_disposition_info`` into the single agent-facing sentence that the
``@tool`` in ``src/tools.py`` hands back to the worker. The prose lives here
(rather than in ``src/tools.py``) so it can be unit-tested without importing
``src.tools`` -- which loads the GNoME database on import. Only stdlib imports.
"""

from typing import Any, Dict, Iterable, List


# Machine job_type -> readable label used in agent messages.
_JOB_LABEL = {
    "bulk_relaxation": "bulk relaxation",
    "surface_relaxation": "surface relaxation",
    "O_adsorption": "O adsorption",
    "OH_adsorption": "OH adsorption",
}


def _unit_desc(unit: Dict[str, Any]) -> str:
    """Render ONE finalized unit as a fragment for the agent.

    A "unit" is one batch-collapsed piece of finished work from
    ``candidate_outstanding`` -- ``{job_type, termination_index, site_index,
    ids}``. The fragment names the calculation, where it sits, and which process
    ids belong to it. Shape examples:

        bulk relaxation: process ids 0, 1 (cite any one)
        surface relaxation (termination 1): process id 2
        O adsorption (termination 0, site 3): process id 4
        OH adsorption (termination 0, site 3): process ids 5, 6, 7 (cite any one)
    """
    label = _JOB_LABEL.get(unit.get("job_type"), unit.get("job_type"))

    # Location: bulk has neither termination nor site (it is a whole-material
    # magnetic batch); surface has a termination; O/OH have termination + site.
    where = []
    if unit.get("termination_index") is not None:
        where.append(f"termination {unit['termination_index']}")
    if unit.get("site_index") is not None:
        where.append(f"site {unit['site_index']}")
    location = f" ({', '.join(where)})" if where else ""

    # Ids: a single-job unit reads "process id X"; a batch (bulk/OH) lists all
    # its sub-job ids and reminds the agent that citing any one covers the batch.
    ids = list(unit.get("ids", []))
    if len(ids) == 1:
        ids_part = f"process id {ids[0]}"
    else:
        ids_part = "process ids " + ", ".join(str(i) for i in ids) + " (cite any one)"

    return f"{label}{location}: {ids_part}"


def format_get_disposition(candidate_id: str, outstanding: Dict[str, Any]) -> str:
    """Build the get_disposition_info message: tell the agent what it must
    summarize for ``candidate_id``.

    ``outstanding`` is the dict returned by ``candidate_outstanding`` --
    ``{must_cover, legacy_optional, latest_disposition, has_finalized}``.
    """
    must = outstanding.get("must_cover", []) or []
    latest = outstanding.get("latest_disposition")

    parts: List[str] = []

    # 1) Always surface the latest disposition so the agent can read where it
    #    left off -- or state explicitly that there is none yet.
    if latest is not None:
        parts.append(
            f"Latest disposition on record for {candidate_id}: "
            f"Decision={latest.get('Decision')}; Summary={latest.get('Summary')}."
        )
    else:
        parts.append(f"No prior disposition for {candidate_id}.")

    # 2) The required work: every must_cover unit, listed via _unit_desc, plus
    #    the instruction on how to record the disposition. Only ids that NEED
    #    citing are listed -- legacy-exempt results are intentionally NOT shown.
    if must:
        listing = "; ".join(_unit_desc(u) for u in must)
        parts.append(
            f"Candidate {candidate_id} has finished results that still need a "
            f"disposition: {listing}. Base your Summary on these results and pass "
            f"the process ids you used (any one id per batch) as "
            f"Summarized_process_id to update_disposition_info."
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

    # Success: the disposition was recorded.
    if status == "ok":
        return (
            f"Recorded disposition for candidate {candidate_id} "
            f"(Decision: {result.get('decision')}). All finished results for "
            f"this candidate are now dispositioned."
        )

    # Write-lock: the agent must read the outstanding results before writing.
    if status == "locked":
        return (
            f"Cannot record a disposition for {candidate_id} yet: call "
            f"get_disposition_info('{candidate_id}') first to review its "
            f"outstanding results, then call update_disposition_info again."
        )

    # Bad Decision: name the offending value and list the allowed vocabulary.
    if status == "invalid_decision":
        allowed = ", ".join(allowed_decisions)
        return (
            f"'{result.get('decision')}' is not a valid Decision. Choose one of: "
            f"{allowed}. Then call update_disposition_info again."
        )

    # Cited ids that are not finished yet: name them; the agent waits or omits.
    if status == "non_terminal_ids":
        ids = ", ".join(str(i) for i in result.get("ids", []))
        return (
            f"These process ids are not finished yet and cannot be summarized: "
            f"{ids}. Wait for them to finish (or omit them), then call "
            f"update_disposition_info again."
        )

    # Incomplete coverage: name the still-uncovered ids; nothing was written.
    if status == "incomplete":
        missing = ", ".join(str(i) for i in result.get("missing", []))
        return (
            f"Disposition not recorded: candidate {candidate_id} still has "
            f"finished results you have not cited: {missing}. Cite at least one "
            f"process id from each, then call update_disposition_info again."
        )

    # The candidate id was not found in the experiment log.
    if status == "unknown_candidate":
        return (
            f"No candidate '{candidate_id}' exists in the experiment log. "
            f"Check the candidate id and try again."
        )

    # Defensive fallback for an unrecognised status.
    return f"Unexpected disposition result for {candidate_id}: {status!r}."
