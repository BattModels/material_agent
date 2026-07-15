# Unit tests for the bounded step ledger (src/past_steps.py).
#
# src/past_steps.py holds the PURE half of the context-management fix: rendering,
# token estimation, the per-step cap, and -- the important one -- plan_compaction,
# which decides what to evict. Keeping it pure means these tests never import
# src.tools (which loads the GNoME database, ~109 s) and never call an LLM.
#
# The numbers in the fixtures are the MEASURED ones from 85 real runs:
#   median step 1,481 chars | mean 2,813 | worst single step 33,209
#   growth ~666 tokens/step, linear, no plateau
#   worst run: 109 steps -> ~77,000 tokens in one prompt
# test_steady_state_ledger_plateaus is the money test: it is the literal
# statement of "linear growth, no plateau" -> "plateau".

import random

import pytest

from src.past_steps import (
    CHARS_PER_TOKEN,
    CLEARABLE_EXCLUDE_TOOLS,
    DIGEST_PREFIX,
    HARD_TOKENS,
    K_VERBATIM,
    MIN_EVICT,
    SOFT_TOKENS,
    STEP_CHAR_CAP,
    build_report_digest,
    build_summary_digest,
    worst_post_compaction_tokens,
    count_leading_digests,
    estimate_tokens,
    first_verbatim_index,
    fold_digests,
    is_digest,
    plan_compaction,
    render_past_steps,
    should_request_report,
    steps_completed,
    truncate_step_text,
)


class FakeStep:
    """Duck-typed stand-in for myPastStep -- the module must never need the real
    pydantic model (it lives in planNexe2, which imports src.tools)."""

    def __init__(self, step, agent="OER_Agent", timeStamp="1:02:03.456789", timeSpent="0:10:00"):
        self.step = step
        self.agent = agent
        self.timeStamp = timeStamp
        self.timeSpent = timeSpent


def make_steps(n, chars=100, start=0):
    return [FakeStep("x" * chars + f"#{start + i}") for i in range(n)]


# --- render: byte-for-byte parity with the f-string it replaces ----------------

def test_render_is_byte_identical_to_the_original_fstring():
    # The exact expression from planNexe2.py:493 / :575 / :794, which this
    # function replaces in all three places. If this drifts, the prompts change.
    steps = [
        FakeStep("did A", agent="OER_Agent", timeStamp="1:02:03.456789", timeSpent="0:10:00"),
        FakeStep("did B", agent="OER_Agent", timeStamp="2:04:06.999999", timeSpent="1:02:03"),
    ]
    original = "\n".join(
        f"{i+1}. {step.agent}: {step.step} "
        f"[total time elapsed since project start: {str(step.timeStamp).split('.')[0]}, "
        f"time spent on step {i+1}: {step.timeSpent}]"
        for i, step in enumerate(steps)
    )
    assert render_past_steps(steps) == original
    # the fractional seconds must be dropped, exactly as the original did
    assert "1:02:03," in render_past_steps(steps)
    assert "456789" not in render_past_steps(steps)


def test_render_without_timing_matches_the_compaction_variant():
    # planNexe2.py:890 used a second, timing-less render to build the archive text.
    steps = [FakeStep("did A"), FakeStep("did B")]
    original = "\n".join(f"{i+1}. {step.agent}: {step.step}" for i, step in enumerate(steps))
    assert render_past_steps(steps, with_timing=False) == original


def test_render_uses_absolute_indices_after_compaction():
    # After a compaction the ledger holds only the tail, but the supervisor must
    # not be told the study restarted at step 1.
    steps = make_steps(3)
    out = render_past_steps(steps, start_index=18)
    assert out.startswith("18. ")
    assert "\n19. " in out
    assert "time spent on step 20:" in out  # the per-line index tracks too


def test_digests_render_verbatim_and_consume_no_index():
    digest = FakeStep(f"{DIGEST_PREFIX} Steps 1-8 -> report 'r1'", agent="system")
    steps = [digest] + make_steps(2)
    out = render_past_steps(steps, start_index=9)
    lines = out.split("\n")
    assert lines[0] == f"{DIGEST_PREFIX} Steps 1-8 -> report 'r1'"
    assert lines[1].startswith("9. ")   # numbering resumes at the first real step
    assert lines[2].startswith("10. ")


def test_empty_ledger_renders_empty():
    assert render_past_steps([]) == ""
    assert estimate_tokens("") == 0


# --- token estimation ---------------------------------------------------------

def test_estimate_tokens_uses_the_anthropic_ratio():
    assert CHARS_PER_TOKEN == 3.3
    assert estimate_tokens("a" * 330) == 100
    assert estimate_tokens("a" * 331) == 101   # ceil, never round down


def test_estimate_tokens_is_monotone():
    assert estimate_tokens("a" * 500) > estimate_tokens("a" * 100)


# --- per-step cap: kills the measured fat tail --------------------------------

def test_short_step_is_untouched():
    text = "x" * (STEP_CHAR_CAP - 1)
    out, truncated = truncate_step_text(text)
    assert out == text
    assert truncated is False


def test_the_real_33209_char_step_is_clamped():
    # The single worst step actually observed in production.
    out, truncated = truncate_step_text("x" * 33_209, archive_ref="step_77")
    assert truncated is True
    assert len(out) <= STEP_CHAR_CAP
    assert "step_77" in out          # the archive pointer survives the clip
    assert estimate_tokens(out) < 1000   # was ~8,300 tokens on its own


def test_truncation_is_idempotent():
    once, _ = truncate_step_text("x" * 33_209, archive_ref="step_77")
    twice, again = truncate_step_text(once, archive_ref="step_77")
    assert twice == once
    assert again is False


# --- plan_compaction: the decision table --------------------------------------

def test_no_compaction_when_small_and_no_report():
    assert plan_compaction(make_steps(12, chars=100)).should is False


def test_no_compaction_below_the_anti_thrash_floor_even_when_over_hard():
    # THE thrash bug: one fat step can push a short ledger over HARD with nothing
    # worth evicting. Without this floor the node fires an LLM summary every turn,
    # forever, and never gets under the limit.
    steps = make_steps(K_VERBATIM + MIN_EVICT - 1, chars=20_000)
    assert estimate_tokens(render_past_steps(steps)) > HARD_TOKENS
    assert plan_compaction(steps).should is False


def test_hard_watermark_forces_compaction_without_a_report():
    steps = make_steps(40, chars=4_000)
    assert estimate_tokens(render_past_steps(steps)) >= HARD_TOKENS
    plan = plan_compaction(steps)
    assert plan.should is True
    assert plan.reason == "hard"
    assert plan.evict_lo == 0
    assert plan.evict_hi == 40 - K_VERBATIM   # everything but the tail


def test_a_report_compacts_for_free_even_below_hard():
    # The whole point: the report already distilled these steps onto CANVAS, so
    # keeping them in context is pure waste. No LLM needed.
    steps = make_steps(20, chars=100)
    assert estimate_tokens(render_past_steps(steps)) < HARD_TOKENS
    plan = plan_compaction(steps, report_written=True)
    assert plan.should is True
    assert plan.reason == "report"
    assert plan.n_evicted == 20 - K_VERBATIM


def test_existing_digests_are_never_evicted_only_stepped_over():
    digests = [FakeStep(f"{DIGEST_PREFIX} old", agent="system") for _ in range(2)]
    steps = digests + make_steps(20)
    plan = plan_compaction(steps, report_written=True)
    assert plan.evict_lo == 2          # skips the digest block
    assert plan.evict_hi == 22 - K_VERBATIM


def test_plan_is_deterministic_the_replay_guarantee():
    # worker_agent_node re-executes from the top on a mid-round resume, so the
    # eviction decision must be identical the second time or the archive and the
    # ledger disagree.
    steps = make_steps(40, chars=4_000)
    a, b = plan_compaction(steps), plan_compaction(steps)
    assert (a.should, a.evict_lo, a.evict_hi, a.reason) == (b.should, b.evict_lo, b.evict_hi, b.reason)


# --- the SOFT nudge (supervisor-side, no compaction) --------------------------

def test_should_request_report_fires_only_at_soft():
    assert should_request_report(make_steps(5, chars=100)) is False
    big = make_steps(30, chars=2_000)
    assert estimate_tokens(render_past_steps(big)) >= SOFT_TOKENS
    assert should_request_report(big) is True


def test_soft_is_below_hard_with_room_to_comply():
    # The agent needs slack between "you should report" and "I'm compacting for
    # you": at the measured 666 tok/step that is ~9 steps of grace.
    assert SOFT_TOKENS < HARD_TOKENS
    assert (HARD_TOKENS - SOFT_TOKENS) / 666 >= 5


# --- digests ------------------------------------------------------------------

def test_report_digest_points_at_canvas_and_is_recognized():
    d = build_report_digest(report_name="round3_report", report_id="abc123", step_lo=25, step_hi=38)
    assert is_digest(FakeStep(d))
    assert "round3_report" in d and "abc123" in d
    assert "25-38" in d
    assert "read_my_canvas" in d          # the recovery path is stated
    assert estimate_tokens(d) < 150       # cheap: ~50-100 tokens replacing thousands


def test_report_digest_uses_the_worker_gist_as_its_body():
    # The write_report tool now asks the worker for a one-sentence gist; it becomes
    # the human-readable body of the compacted step, so the digest says WHAT happened
    # at a glance, not just where to look.
    gist = "Screened 12 Ru/Ir oxides; IrO2 rutile is the best candidate, Co3O4 abandoned."
    d = build_report_digest(report_name="round3", report_id="i", step_lo=25, step_hi=38, gist=gist)
    assert gist in d
    assert d.startswith(f"{DIGEST_PREFIX} (steps 25-38) Screened 12 Ru/Ir oxides")
    # still self-consistent for the step-count derivation (the canonical head is intact)
    from src.past_steps import _digest_hi
    assert _digest_hi(d) == 38
    # a missing/blank gist just omits the body, no stray double-space artifact
    assert "(steps 1-2)  " not in build_report_digest(
        report_name="r", report_id=None, step_lo=1, step_hi=2, gist="")


def test_summary_digest_is_recognized():
    d = build_summary_digest(summary="tried X, it failed", step_lo=1, step_hi=9)
    assert is_digest(FakeStep(d))
    assert "tried X, it failed" in d


def test_digests_point_at_the_raw_steps_canvas_key_when_given():
    # Raw evicted steps now live on CANVAS; the digest must tell the agent the key.
    r = build_report_digest(report_name="r3", report_id="i3", step_lo=1, step_hi=9,
                            raw_steps_key="r3__steps")
    assert "read_my_canvas(key='r3__steps')" in r
    s = build_summary_digest(summary="s", step_lo=1, step_hi=9,
                             raw_steps_key="compacted_ru_oxides")
    assert "read_my_canvas(key='compacted_ru_oxides')" in s
    # and without a key (back-compat) they simply omit the raw pointer
    assert "read_my_canvas(key='None')" not in build_report_digest(
        report_name="r", report_id=None, step_lo=1, step_hi=2)


def test_count_leading_digests_stops_at_the_first_real_step():
    steps = [FakeStep(f"{DIGEST_PREFIX} a"), FakeStep(f"{DIGEST_PREFIX} b"), FakeStep("real"),
             FakeStep(f"{DIGEST_PREFIX} c")]
    assert count_leading_digests(steps) == 2


def test_fold_digests_bounds_the_block_and_keeps_every_report_reachable():
    # A digest costs ~115 chars (~35 tokens), so it takes ~70 of them -- i.e. a
    # ~700-step campaign -- before the block itself is worth folding.
    n = 82
    digests = [
        build_report_digest(report_name=f"r{i}", report_id=f"id{i}", step_lo=i * 10, step_hi=i * 10 + 9)
        for i in range(n)
    ]
    assert len("\n".join(digests)) > 8_000
    folded = fold_digests(digests)
    assert len("\n".join(folded)) <= 8_000
    assert len(folded) < len(digests)
    assert all(is_digest(FakeStep(d)) for d in folded)
    # nothing becomes unreachable, and the merged head must still span the old range
    # so the step count is preserved: it starts at step 0, and the highest step
    # covered by the folded block equals the highest before folding (digests cover
    # steps 0..10n-1, no verbatim steps here).
    assert folded[0].startswith(f"{DIGEST_PREFIX} (steps 0-")
    fake_before = [FakeStep(d, agent="system") for d in digests]
    fake_after = [FakeStep(d, agent="system") for d in folded]
    assert steps_completed(fake_after) == steps_completed(fake_before) == n * 10 - 1


def test_fold_is_a_noop_when_under_cap():
    digests = [build_report_digest(report_name="r1", report_id="i1", step_lo=1, step_hi=9)]
    assert fold_digests(digests) == digests


# --- steps_completed: the monotone counter, DERIVED not stored --------------------
# This is the crux of the resume-safety fix: no `steps_completed` state channel is
# added to PlanExecute, so the count must be recoverable from the ledger alone.

def test_steps_completed_with_no_digests_is_just_len():
    # An OLD checkpoint (resumed) has no [COMPACTED] digest, so this MUST equal
    # len(past_steps) -- byte-identical to the pre-change behaviour.
    assert steps_completed(make_steps(7)) == 7
    assert steps_completed([]) == 0


def test_steps_completed_after_one_compaction():
    digest = build_report_digest(report_name="r1", report_id="i1", step_lo=1, step_hi=30)
    steps = [FakeStep(digest, agent="system")] + make_steps(10)
    assert steps_completed(steps) == 40   # 30 in the digest + 10 verbatim


def test_steps_completed_after_two_compactions():
    d1 = build_report_digest(report_name="r1", report_id="i1", step_lo=1, step_hi=30)
    d2 = build_summary_digest(summary="unreported work", step_lo=31, step_hi=50)
    steps = [FakeStep(d1, agent="system"), FakeStep(d2, agent="system")] + make_steps(10)
    assert steps_completed(steps) == 60   # highest digest hi (50) + 10 verbatim


def test_steps_completed_takes_the_max_digest_hi_after_a_fold():
    # After folding, a merged digest sits before newer ones; the count must follow
    # the HIGHEST hi across the block, not the first digest's.
    merged = build_report_digest(report_name="merged", report_id="m", step_lo=1, step_hi=40)
    recent = build_summary_digest(summary="s", step_lo=41, step_hi=70)
    steps = [FakeStep(merged, agent="system"), FakeStep(recent, agent="system")] + make_steps(5)
    assert steps_completed(steps) == 75   # 70 + 5


def test_digest_hi_round_trips_through_both_builders():
    from src.past_steps import _digest_hi
    assert _digest_hi(build_report_digest(report_name="r", report_id="i", step_lo=12, step_hi=34)) == 34
    assert _digest_hi(build_summary_digest(summary="x", step_lo=5, step_hi=9)) == 9
    # a non-digest string yields 0 (safe default, never negative)
    assert _digest_hi("2. OER_Agent: did a thing") == 0


def test_first_verbatim_index_derives_from_the_digest_block():
    # A real digest carrying the canonical head for steps 1-32, then 10 verbatim
    # steps -> the first verbatim step is step 33. No steps_completed arg any more.
    digest = build_report_digest(report_name="r1", report_id="i1", step_lo=1, step_hi=32)
    steps = [FakeStep(digest, agent="system")] + make_steps(10)
    assert first_verbatim_index(steps) == 33
    assert steps_completed(steps) == 42   # 32 covered by the digest + 10 verbatim


# --- Phase 2 safety fence -----------------------------------------------------

def test_side_effecting_tools_are_never_clearable():
    # If ClearToolUsesEdit blanks a submit_dft_job result, the agent can lose the
    # evidence it already submitted and RE-SUBMIT real DFT jobs. Burned compute,
    # not just a context bug. This fence must never regress.
    for tool in ("submit_dft_job", "enter_candidate_in_log", "update_disposition_info",
                 "write_report", "write_my_canvas", "wait_for_update"):
        assert tool in CLEARABLE_EXCLUDE_TOOLS


# --- the knob guard -----------------------------------------------------------

def test_knobs_satisfy_the_thrash_constraint():
    """The knobs are NOT independent. If HARD is not above the worst possible
    post-compaction size, then compacting leaves the ledger still over HARD, so we
    compact again next turn -- an LLM call every turn, forever, that never
    converges. This test is the fence: raise K_VERBATIM or STEP_CHAR_CAP too far
    and it fails HERE rather than silently in a 30-day run."""
    worst = worst_post_compaction_tokens()
    assert HARD_TOKENS > worst, (
        f"thrash: worst post-compaction ledger is {worst} tokens but HARD_TOKENS is "
        f"{HARD_TOKENS}. Raise HARD_TOKENS, or lower K_VERBATIM / STEP_CHAR_CAP / "
        f"DIGEST_BLOCK_CHAR_CAP."
    )
    # not required for correctness, but keeps the supervisor from being nagged for
    # a report it has just satisfied
    assert SOFT_TOKENS > worst, (
        f"SOFT_TOKENS ({SOFT_TOKENS}) is below the worst post-compaction ledger "
        f"({worst}), so the report nudge would fire again immediately after a report."
    )


def test_the_documented_failure_case_really_does_trip_the_fence():
    # The concrete trap called out in past_steps.py: K=20 at the current step cap.
    assert worst_post_compaction_tokens(k=20) > HARD_TOKENS


# --- the money test -----------------------------------------------------------

def test_steady_state_ledger_plateaus():
    """Replay 250 steps drawn from the MEASURED size distribution and assert the
    ledger stops growing. Before this change: 666 tok/step, linear, 109 steps ->
    77k tokens. After: bounded."""
    rng = random.Random(1234)
    ledger: list[FakeStep] = []
    compactions = 0
    peak = 0

    for i in range(250):
        # measured: median 1,481 / mean 2,813 / occasional 33,209 monster
        if rng.random() < 0.04:
            size = 33_209
        else:
            size = int(rng.lognormvariate(7.3, 0.6))
        text, _ = truncate_step_text("y" * size, archive_ref=f"step_{i+1}")
        ledger.append(FakeStep(text))

        # the running total is derived from the ledger, never tracked separately
        assert steps_completed(ledger) == i + 1

        # a report every ~12 steps once the nudge would have fired (the agent complies)
        report = should_request_report(ledger) and rng.random() < 0.35

        plan = plan_compaction(ledger, report_written=report)
        if plan.should:
            compactions += 1
            lo_abs = first_verbatim_index(ledger)
            digest_text = (
                build_report_digest(report_name=f"r{compactions}", report_id=f"id{compactions}",
                                    step_lo=lo_abs, step_hi=lo_abs + plan.n_evicted - 1)
                if plan.reason == "report"
                else build_summary_digest(summary="stub summary of un-reported work",
                                          step_lo=lo_abs, step_hi=lo_abs + plan.n_evicted - 1)
            )
            kept_digests = [s.step for s in ledger[: plan.evict_lo]] + [digest_text]
            kept_digests = fold_digests(kept_digests)
            ledger = [FakeStep(d, agent="system") for d in kept_digests] + list(ledger[plan.evict_hi:])

        tokens = estimate_tokens(render_past_steps(ledger))
        peak = max(peak, tokens)

    # (1) the bound actually holds: never more than HARD plus the one step that
    #     tipped it over (which cannot exceed the per-step cap).
    assert peak <= HARD_TOKENS + estimate_tokens("y" * STEP_CHAR_CAP) + 500, f"peak={peak}"

    # (2) no thrash -- we are not compacting on nearly every turn
    assert compactions < 250 / 5, f"compactions={compactions}"

    # (3) and it genuinely plateaued rather than just growing slower: the old
    #     behaviour would be ~250 * 666 = 166,000 tokens by now.
    final = estimate_tokens(render_past_steps(ledger))
    assert final < HARD_TOKENS, f"final={final}"


# --- integration: the duck-typing must hold against the REAL model shape -------
# src/past_steps.py never imports myPastStep (that would drag in src.tools and its
# 500k-row DB). So the one real coupling risk is: does it work on an actual pydantic
# object whose timeStamp is a timedelta, not a string? Rebuild myPastStep's exact
# shape here (planNexe2.py:90-98) and prove it does.

from datetime import timedelta
from typing import Any as _Any

from pydantic import BaseModel, Field


class RealShapePastStep(BaseModel):
    step: str = Field(description="Step to perform.")
    agent: str = Field(description="Agent to perform the step.")
    timeStamp: _Any = Field(description="The time when the step is completed.")
    timeSpent: str = Field(description="The time spent on this step.")


def test_render_handles_a_pydantic_step_with_a_timedelta_timestamp():
    steps = [
        RealShapePastStep(step="did A", agent="OER_Agent",
                          timeStamp=timedelta(seconds=3723, microseconds=456789),
                          timeSpent="0:10:00"),
    ]
    out = render_past_steps(steps)
    assert out == (
        "1. OER_Agent: did A [total time elapsed since project start: 1:02:03, "
        "time spent on step 1: 0:10:00]"
    )
    assert "456789" not in out   # timedelta's microseconds are stripped, as before


def test_compaction_pipeline_end_to_end_on_pydantic_steps():
    ledger = [
        RealShapePastStep(step=f"step {i}" * 50, agent="OER_Agent",
                          timeStamp=timedelta(seconds=i * 600), timeSpent="0:10:00")
        for i in range(1, 21)
    ]
    plan = plan_compaction(ledger, report_written=True)
    assert plan.should and plan.n_evicted == 20 - K_VERBATIM

    digest = build_report_digest(report_name="round3_report", report_id="abc123",
                                 step_lo=1, step_hi=plan.n_evicted)
    compacted = [
        RealShapePastStep(step=digest, agent="system",
                          timeStamp=timedelta(seconds=0), timeSpent="0:00:00")
    ] + ledger[plan.evict_hi:]

    assert len(compacted) == 1 + K_VERBATIM
    assert is_digest(compacted[0])
    # numbering must resume at the right ABSOLUTE step, not restart at 1
    rendered = render_past_steps(compacted, start_index=first_verbatim_index(compacted))
    assert rendered.startswith(digest)
    assert "\n11. OER_Agent:" in rendered
