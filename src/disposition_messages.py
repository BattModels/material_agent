"""Pure formatters for the disposition tools.

These turn the STRUCTURED result of ``EXPLOG.get_disposition_info`` /
``EXPLOG.update_disposition_info`` into the single agent-facing sentence that the
``@tool`` in ``src/tools.py`` hands back to the worker. The prose lives here
(rather than in ``src/tools.py``) so it can be unit-tested without importing
``src.tools`` -- which loads the GNoME database on import. Imports only the stdlib
plus the tiny stdlib-only ``src.var`` config module (never ``src.tools``).
"""

from typing import Any, Dict, Iterable, List

from src import var


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
    ``{must_cover, legacy_optional, latest_disposition, has_finalized}`` -- or
    ``{"unknown_candidate": True}`` when the candidate is not in the log.
    """
    # The candidate id was not found in the experiment log.
    if outstanding.get("unknown_candidate"):
        return (
            f"No candidate '{candidate_id}' exists in the experiment log. "
            f"Check the candidate id and try again."
        )

    must = outstanding.get("must_cover", []) or []
    latest = outstanding.get("latest_disposition")

    parts: List[str] = []

    # 1) Always surface the latest disposition so the agent can read where it
    #    left off -- or state explicitly that there is none yet.
    if latest is not None:
        parts.append(
            f"Latest disposition on record for {candidate_id}: "
            f"Decision={latest.get('Decision')}; Summary={latest.get('Summary')}; "
            f"Future_plan={latest.get('Future_plan')}."
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

    # Cited ids that belong to a different candidate (or do not exist): name them.
    if status == "foreign_ids":
        ids = ", ".join(str(i) for i in result.get("ids", []))
        return (
            f"These process ids do not belong to candidate {candidate_id}: "
            f"{ids}. Cite only {candidate_id}'s own finished results (see "
            f"get_disposition_info), then call update_disposition_info again."
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


# ===========================================================================
# wait_for_update gate prose + the pure entry decision
#
# wait_for_update (src/tools.py) is a sleep/poll loop, so its entry-gate
# DECISION is factored out here as a pure function over scalars + the prose it
# selects. This keeps the gate logic testable without importing src.tools (which
# loads the GNoME database). The loop itself only gathers the inputs (from the
# nested EXPLOG helpers + var) and returns whatever message this hands back.
# ===========================================================================

# Returned (unchanged from the original tool) when there is genuinely nothing to
# wait for: no pending and no running work, and no analysis owed.
MSG_NOTHING_TO_WAIT_FOR = (
    "No pending or running jobs found in the EXPLOG. Please check the EXPLOG "
    "and see if there is anything you can do to move the study forward, instead "
    "of waiting for updates."
)


def format_wait_gate1_refusal(candidate_ids: Iterable[str]) -> str:
    """Gate 1 refusal: finished results are not yet tied back to a disposition.

    Names the offending candidates (capped at 10 + a remainder count, so a large
    backlog -- e.g. the first post-rollout resume -- cannot dump hundreds of ids
    into one message) and the exact two-step workflow to clear them. Gate 1 is
    ALWAYS enforced -- analysis must be current before waiting.
    """
    ids_list = [str(c) for c in candidate_ids]
    shown = ids_list[:10]
    listing = ", ".join(shown)
    extra = len(ids_list) - len(shown)
    if extra > 0:
        listing += f" (and {extra} more)"
    return (
        "You cannot wait yet: these candidates have finished results you have "
        f"not yet summarized into a disposition: {listing}. For each, call "
        "get_disposition_info(candidate_id) to see what needs summarizing, then "
        "update_disposition_info(...) to record your reading of the results. "
        "Once every finished result is dispositioned you may wait."
    )


def _forgotten_job_line(item: Dict[str, Any]) -> str:
    """One agent-facing line for a forgotten-work item from find_forgotten_jobs."""
    cid = item.get("candidate_id")
    kind = item.get("kind")
    if kind == "bulk":
        return f"{cid}: no bulk relaxation started -- start the bulk relaxation."
    if kind == "surface":
        return (f"{cid}: bulk finished but no surface relaxation started -- "
                "start surface relaxations.")
    if kind == "O":
        return (f"{cid}: surface finished but no O adsorption started -- "
                "start O adsorption jobs.")
    if kind == "OH":
        return (f"{cid}: competitive O site (termination "
                f"{item.get('termination_index')}, site {item.get('site_index')}) "
                "has no OH adsorption -- start OH there.")
    return f"{cid}: ready unstarted work ({kind})."


def format_wait_gate2_refusal(forgotten_jobs: Iterable[Dict[str, Any]] = ()) -> str:
    """Gate 2 refusal: analysis is current but the HPC queue is below its floor.

    DELIBERATELY states no pending count / floor / headroom: a precise deficit is
    a target the agent would game (submit just enough to clear the gate). Instead
    it advises submitting a LARGE batch and, if available, lists up to 10 concrete
    ready-but-unstarted items (from find_forgotten_jobs) to launch; otherwise it
    points at the candidates df's per-stage progress columns.

    When there are <= var.FORGOTTEN_CLOSER_SUPPRESS_ABOVE forgotten jobs it also
    routes the worker back to the supervisor to discuss expanding submissions and
    to ground decisions in a literature review; above that there is plainly plenty
    of work, so that closer is dropped to keep the worker moving.
    """
    header = (
        "The HPC queue is running low -- submit a large batch of new jobs now to "
        "keep the cluster well fed. Aim to queue many jobs (on the order of 40-50 "
        "or more), not just the bare minimum, so utilization stays high while "
        "results come in."
    )

    jobs = list(forgotten_jobs)
    if jobs:
        shown = jobs[:10]
        listing = "\n".join("  - " + _forgotten_job_line(j) for j in shown)
        body = "Ready, unstarted work you can launch right now:\n" + listing
        extra = len(jobs) - len(shown)
        if extra > 0:
            body += (
                f"\n... and {extra} more forgotten job(s), which will be listed "
                "once these 10 are taken care of."
            )
    else:
        body = (
            "No ready-but-unstarted work was detected automatically. To find "
            "more, filter the candidates table with query_explog on the per-stage "
            "progress columns -- n_surface_started, n_O_started, n_OH_started "
            "(and the n_*_finalized variants): for each, <NA> means the candidate "
            "is not yet eligible for that stage while == 0 means eligible but not "
            "yet started (so n_O_started == 0 lists candidates whose surface "
            "finished but no O job was launched, and n_OH_started == 0 those "
            "ready for OH). Also revisit your reports and summaries, review the "
            "decision notes, and consider adding new AQ-GNoME candidates."
        )

    parts = [header, body]
    if len(jobs) <= var.FORGOTTEN_CLOSER_SUPPRESS_ABOVE:
        parts.append(
            "Also return to the supervisor to discuss how best to expand the "
            "submissions, and consider a literature review (e.g. arXiv_search) to "
            "inform which candidates and adsorption sites are most worth pursuing "
            "-- let both your and the supervisor's decisions be guided by that "
            "literature."
        )
    return " ".join(parts)


def format_wait_exit_disposition_hint(candidate_ids: Iterable[str]) -> str:
    """Trailing line for a wait EXIT: the candidates whose work just finalized.

    Empty when nothing finalized (the wait exited on timeout) -> returns "" so
    the caller can append unconditionally.
    """
    ids = [str(c) for c in candidate_ids]
    if not ids:
        return ""
    return (
        f"\nFinished work belongs to candidate(s): {', '.join(ids)}. Before "
        "waiting again, summarize their finished results with "
        "get_disposition_info then update_disposition_info."
    )


def evaluate_wait_entry(
    *,
    candidates_need_disposition: List[str],
    pending_count: int,
    has_running: bool,
    enforce_queue_floor: bool,
    queue_min_pending: int,
    forgotten_jobs: Iterable[Dict[str, Any]] = (),
) -> str:
    """Decide whether wait_for_update may proceed. Returns a refusal message, or
    ``None`` to proceed into the wait loop. Precedence:

      1. Gate 1 (always): finished work must be dispositioned first.
      2. Nothing in flight (no pending, no running): "nothing to wait for" --
         UNLESS the floor is armed AND there is detectable ready work, in which
         case the (more actionable) Gate 2 refill message wins. A genuinely idle
         worker with no detectable work is therefore NOT pushed to submit a large
         batch, even with the floor on (that case routes to the supervisor).
      3. Something in flight but the queue is below its floor (and the floor is
         armed) -> Gate 2 refill.
      4. Otherwise -> proceed (None).

    The numeric inputs (pending_count, queue_min_pending) drive the Gate 2
    DECISION only; they are never shown to the agent (the Gate 2 message states no
    deficit, to avoid making the floor a gameable target). ``forgotten_jobs``
    (from find_forgotten_jobs) is passed through to the Gate 2 message. A
    non-positive floor or enforce_queue_floor=False disarms Gate 2 entirely.
    """
    if candidates_need_disposition:
        return format_wait_gate1_refusal(candidates_need_disposition)

    jobs = list(forgotten_jobs)
    gate2_armed = (
        enforce_queue_floor
        and queue_min_pending > 0
        and pending_count < queue_min_pending
    )

    if pending_count == 0 and not has_running:
        # Nothing to wait for. Only push to refill if the floor is armed AND we
        # actually have ready work to point the worker at.
        if gate2_armed and jobs:
            return format_wait_gate2_refusal(jobs)
        return MSG_NOTHING_TO_WAIT_FOR

    if gate2_armed:
        return format_wait_gate2_refusal(jobs)
    return None
