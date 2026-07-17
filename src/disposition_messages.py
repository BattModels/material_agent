"""Pure formatters for the disposition tools.

These turn the STRUCTURED result of ``EXPLOG.get_disposition_info`` /
``EXPLOG.update_disposition_info`` into the single agent-facing sentence that the
``@tool`` in ``src/tools.py`` hands back to the worker. The prose lives here
(rather than in ``src/tools.py``) so it can be unit-tested without importing
``src.tools`` -- which loads the GNoME database on import. Imports only the stdlib
plus the tiny stdlib-only ``src.var`` config module (never ``src.tools``).
"""

from typing import Any, Dict, Iterable, List, Optional

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
    review and analyze for ``candidate_id``.

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
            f"Decision={latest.get('Decision')}; "
            f"Analysis_and_implications={latest.get('Summary')}; "
            f"Latest Future_plan when this candidate was dispositioned last time={latest.get('Future_plan')}."
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
            f"disposition: {listing}. Review and analyze these results, base your "
            f"Analysis_and_implications on what they show and imply, and pass the "
            f"process ids you used (any one id per batch) as Analyzed_process_id "
            f"to update_disposition_info."
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
            f"outstanding results, analyze latest results and previous findings,"
            f"then call update_disposition_info again."
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
            f"These process ids are not finished yet and cannot be analyzed: "
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

# Returned when there is genuinely nothing to wait for: no pending and no
# running work, and no analysis owed. Batch size comes from the refill target.
MSG_NOTHING_TO_WAIT_FOR = (
    "Nothing is pending or running and there is nothing ready to submit. Do NOT "
    "keep polling: END YOUR TURN and return to the supervisor to discuss how to "
    "move the study forward and queue a large batch of new jobs (on the order of "
    f"{var.QUEUE_REFILL_TARGET}) -- which candidates, if any, warrant more "
    "calculations (more surfaces / sites, O/OH jobs), whether to add more "
    "candidates, and how to ground these choices in your current findings and "
    "the literature -- or conclude the study if it is genuinely complete."
)

# Final-days variant of the above: with less than PATH_B_CUTOFF_DAYS remaining,
# a fresh batch could not finish, so the idle worker is steered to finalization
# instead of expansion.
MSG_NOTHING_TO_WAIT_FOR_FINAL_DAYS = (
    "Nothing is pending or running and the study is in its final days -- a new "
    "batch of jobs could not finish in time. Do NOT keep polling: END YOUR TURN "
    "and return to the supervisor to finalize the results and produce the final "
    "report."
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
        f"not yet reviewed and analyzed into a disposition: {listing}. For each, "
        "call get_disposition_info(candidate_id) to see what needs analyzing, then "
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


def format_wait_gate2_refusal(
    forgotten_jobs: Iterable[Dict[str, Any]] = (),
    *,
    running_count: int = 0,
    pending_count: int = 0,
) -> str:
    """Gate 2 refusal: analysis is current but the HPC queue is below its floor.

    States the REAL numbers (running / queued / floor / deficit) plus, in short
    terms, why QUEUED jobs are the target -- the old message hid them and the
    agents responded to the opacity by distrusting and disarming the gate. Two
    paths:
      - Path A (ready/forgotten work exists): the worker submits the listed jobs
        ITSELF under its current task (standing duty), then re-calls wait. No
        supervisor round-trip.
      - Path B (no ready work): end the turn and hand back to the supervisor to
        expand the study or wind down (enforce_queue_floor=False, honored only
        in the final FLOOR_DISARM_WINDOW_DAYS).
    """
    jobs = list(forgotten_jobs)
    deficit = max(var.QUEUE_MIN_PENDING - pending_count, 1)

    # Path A: ready continuation work -> the worker submits it NOW, then re-waits.
    if jobs:
        shown = jobs[:10]
        listing = "\n".join("  - " + _forgotten_job_line(j) for j in shown)
        extra = len(jobs) - len(shown)
        more = f"\n  ... and {extra} more." if extra > 0 else ""
        return (
            f"HPC queue below floor: {running_count} running but only "
            f"{pending_count} queued (hard floor: {var.QUEUE_MIN_PENDING} queued "
            "-- queued jobs feed nodes the instant they free up, and a near-empty "
            "queue under fair-share means the cluster is absorbing all we give "
            "it: free capacity). While under-utilized, submissions are "
            "effectively free: submit the most valuable ready continuation jobs "
            "below NOW under your current task (standing duty), guided by your "
            f"dispositions' priorities. Clear the floor at minimum ({deficit} "
            "more queued), and prefer refilling toward "
            f"~{var.QUEUE_REFILL_TARGET} queued where the work justifies it; "
            "then call wait_for_update again.\n\n"
            "Ready-but-unstarted continuation work:\n" + listing + more
        )

    # Path B: no ready work -> end the turn and return to the supervisor to DISCUSS
    # how to expand the study (or wind down).
    return (
        f"HPC queue below floor: {running_count} running, {pending_count} queued "
        f"(floor: {var.QUEUE_MIN_PENDING}) and no ready continuation work "
        "remains. Do NOT wait: END YOUR TURN and return to the supervisor to "
        "discuss expanding the study so the queue can be refilled toward "
        f"~{var.QUEUE_REFILL_TARGET} queued jobs -- under-utilization makes new "
        "submissions effectively free. Discuss which candidates deserve more "
        "surfaces/terminations/sites and whether to add new candidates (e.g. a "
        "fresh AQ-GNoME query), grounded in your findings and the literature "
        "(arXiv_search). If the study is genuinely winding down instead, ask the "
        "supervisor to set enforce_queue_floor=False (honored only in the final "
        f"{var.FLOOR_DISARM_WINDOW_DAYS} days). Do NOT re-call wait_for_update."
    )


def format_supervisor_handback_directive(
    path: str, forgotten_jobs: Iterable[Dict[str, Any]] = ()
) -> str:
    """SUPERVISOR-facing directive injected into the supervisor prompt when a
    worker handed back off the queue floor (``path`` from classify_wait_handback).

    Unlike ``format_wait_gate2_refusal`` -- addressed to the worker, telling it to
    end its turn -- this tells the SUPERVISOR which plan step to make, so the
    handback is acted on rather than discounted as worker prose. Only the
    ``"expand"`` path remains (Path B / idle): ready continuation work is now the
    worker's standing duty, submitted under its own task without a handback.
    ``forgotten_jobs`` is kept in the signature for call-site compatibility.
    """
    # path == "expand" (also the idle "nothing to wait for" handback)
    return (
        "HANDBACK -- QUEUE FLOOR: a worker returned to you because the HPC queue "
        "is under-utilized and there is no ready continuation work left to submit. "
        "Plan a step that opens a DISCUSSION with the OER_agent on how best to "
        "expand the study (or whether to wind it down) -- not a hand-off for it to "
        "decide alone, and not a decision you make alone: the OER_agent is the "
        "only agent that can explore the current results directly and search the "
        "literature (arXiv), so its input is essential while the direction is "
        "settled together. The discussion must cover, at a minimum, these four "
        "points: (1) review the current results of the active candidates; (2) "
        "consult the literature (e.g. arXiv_search) for relevant guidance; (3) "
        "decide which active candidates, if any, are interesting enough to explore "
        "further (more surfaces / terminations, more O or OH adsorption sites); "
        "and (4) whether more candidates should be added (e.g. a fresh AQ-GNoME "
        "query) if that is the best course of action -- the goal is a queue "
        f"refilled toward ~{var.QUEUE_REFILL_TARGET} queued jobs of the most "
        "valuable work. Only if the study is genuinely winding down (the "
        "remaining time is too short for newly-submitted jobs to finish) set "
        "enforce_queue_floor=False on the next step so the worker may instead "
        "wait for and finalize the in-flight results (note: honored only within "
        f"the final {var.FLOOR_DISARM_WINDOW_DAYS} days of the "
        f"{var.STUDY_BUDGET_DAYS}-day budget), or conclude the study if it is "
        "genuinely complete."
    )


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
        "waiting again, review and analyze their finished results with "
        "get_disposition_info then update_disposition_info."
    )


def evaluate_wait_entry(
    *,
    candidates_need_disposition: List[str],
    pending_count: int,
    running_count: int,
    enforce_queue_floor: bool,
    queue_min_pending: int,
    remaining_seconds: float,
    forgotten_jobs: Iterable[Dict[str, Any]] = (),
) -> str:
    """Decide whether wait_for_update may proceed. Returns a refusal message, or
    ``None`` to proceed into the wait loop. Precedence:

      1. Gate 1 (always): finished work must be dispositioned first.
      2. Nothing in flight (no pending, no running): "nothing to wait for" --
         UNLESS the floor is armed AND there is detectable ready work, in which
         case the (more actionable) Gate 2 refill message wins. A genuinely idle
         worker with no detectable work is routed to the supervisor -- to expand,
         or (in the final PATH_B_CUTOFF_DAYS) to finalize and report.
      3. Something in flight but the queue is below its floor (and the floor is
         armed) -> Gate 2: Path A (ready work listed; the worker submits it
         itself then re-waits) whenever ready work exists; Path B (hand back to
         the supervisor to expand) only while more than PATH_B_CUTOFF_DAYS
         remain -- inside that window a wait with nothing ready simply proceeds.
      4. Otherwise -> proceed (None).

    The numeric inputs (pending_count, running_count, queue_min_pending) drive
    the Gate 2 decision AND are shown to the agent with the deficit -- the old
    opacity taught agents to distrust the gate. ``forgotten_jobs`` (from
    find_forgotten_jobs) is passed through to the Gate 2 message. A non-positive
    floor or enforce_queue_floor=False disarms Gate 2 entirely.
    """
    if candidates_need_disposition:
        return format_wait_gate1_refusal(candidates_need_disposition)

    jobs = list(forgotten_jobs)
    gate2_armed = (
        enforce_queue_floor
        and queue_min_pending > 0
        and pending_count < queue_min_pending
    )
    path_b_active = remaining_seconds >= var.PATH_B_CUTOFF_SECONDS

    if pending_count == 0 and running_count == 0:
        # Nothing to wait for. Only push to refill if the floor is armed AND we
        # actually have ready work to point the worker at.
        if gate2_armed and jobs:
            return format_wait_gate2_refusal(
                jobs, running_count=running_count, pending_count=pending_count
            )
        if path_b_active:
            return MSG_NOTHING_TO_WAIT_FOR
        return MSG_NOTHING_TO_WAIT_FOR_FINAL_DAYS

    if gate2_armed and (jobs or path_b_active):
        return format_wait_gate2_refusal(
            jobs, running_count=running_count, pending_count=pending_count
        )
    return None


def classify_wait_handback(
    *,
    candidates_need_disposition: List[str],
    pending_count: int,
    running_count: int,
    enforce_queue_floor: bool,
    queue_min_pending: int,
    remaining_seconds: float,
    forgotten_jobs: Iterable[Dict[str, Any]] = (),
) -> Optional[str]:
    """Classify a wait_for_update refusal into the SUPERVISOR handback path it
    warrants -- the path token behind the directive the supervisor injects, as
    opposed to ``evaluate_wait_entry``'s worker-facing message.

      - ``None``     -> not a supervisor handback: a Gate 1 disposition backlog
                        (the worker clears it itself), a Path A refusal (the
                        worker submits the listed ready work ITSELF -- standing
                        duty, no supervisor round-trip), or "proceed to wait".
      - ``"expand"`` -> Path B: queue below its floor with no ready work (while
                        more than PATH_B_CUTOFF_DAYS remain), OR nothing in
                        flight at all (idle) -> the supervisor should expand the
                        study, wind it down, or (idle in the final days)
                        finalize and report.

    Kept in lock-step with ``evaluate_wait_entry`` (same precedence + off-switches)
    so the flag the wait tool raises and the path the supervisor re-derives from
    live EXPLOG never disagree.
    """
    if candidates_need_disposition:
        return None

    jobs = list(forgotten_jobs)
    gate2_armed = (
        enforce_queue_floor
        and queue_min_pending > 0
        and pending_count < queue_min_pending
    )
    path_b_active = remaining_seconds >= var.PATH_B_CUTOFF_SECONDS

    if pending_count == 0 and running_count == 0:
        if gate2_armed and jobs:
            return None    # Path A: worker self-serves
        return "expand"    # idle: expansion or (final days) finalize-and-report

    if gate2_armed and not jobs and path_b_active:
        return "expand"
    return None


def evaluate_terminal_tag_gate(
    *,
    decision: str,
    state: Any,
    is_forgotten: bool,
    has_in_flight: bool,
    terminal_decisions: Iterable[str],
    active_decisions: Iterable[str],
) -> str | None:
    """Decide whether a NEW disposition ``Decision`` is allowed for a candidate,
    given the candidate's state. Returns a rejection sentence, or ``None`` to allow
    the write. PURE (no EXPLOG): the caller passes the three facts in, so this is
    fast-testable without importing ``src.tools``.

    Rules (the Part 2 terminal-tag gate):
      - A FAILED candidate may ONLY be "Abandon" -- any other decision is rejected.
      - A terminal tag (Abandon/Sufficient) on a non-failed candidate is allowed
        only when the candidate is "fully settled" = NOT a forgotten job AND no
        in-flight (pending/running) process. Otherwise it still has pipeline work to
        finish, so it is steered to continue or record an active priority.
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
    if decision in tuple(terminal_decisions):
        if is_forgotten or has_in_flight:
            active = ", ".join(active_decisions)
            return (
                f"'{decision}' is a terminal decision, but this candidate is not "
                f"fully settled yet -- it still has ready or in-flight pipeline work. "
                f"Let that work finish first (its ready jobs will be submitted), or "
                f"record an active priority instead ({active})."
            )
        return None
    return None
