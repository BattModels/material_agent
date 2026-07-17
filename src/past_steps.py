"""Pure helpers for the bounded step ledger (`PlanExecute.past_steps`).

The ledger used to grow without bound: measured across 85 real runs it climbed
~666 tokens per step, linearly, and the worst run reached 109 steps / ~77k
tokens -- injected verbatim into the worker, supervisor AND boss prompts every
round. The only compaction fired when a report was written, and 82 of those 85
runs never wrote one, so it effectively never ran.

The ledger is now kept to the invariant

    past_steps == [digest steps ...] + [last K verbatim steps]

where a *digest* is a short pointer standing in for a run of steps that has
already been distilled elsewhere (normally into a report on CANVAS). Digests are
ordinary step objects whose text starts with a canonical, machine-parseable head
("[COMPACTED] (steps LO-HI) ..."). That is deliberate: it keeps the LangGraph
state schema (PlanExecute) BYTE-UNCHANGED -- no new channel -- so an old
checkpoint still resumes, and it lets us recover the monotone total step count
(steps_completed()) and the absolute step numbering straight out of `past_steps`
itself instead of a dedicated field. An old checkpoint has no such digest, so the
count falls back to len(past_steps), exactly the pre-change behaviour.

Three cooperating thresholds, all recomputed from the ledger itself every turn
(no process-global state, so this is resume-safe by construction):

    SOFT  -- the supervisor notices its own history is too long and plans a
             `write_report` step. Steering only; no compaction here.
    report written -- compact for free: the report is already on CANVAS, so the
             evicted steps collapse to a ~50-token pointer. No LLM call.
    HARD  -- the agent ignored the nudge. Compact anyway, paying for an LLM
             summary of the un-reported steps. This is the backstop that makes
             the bound unconditional.

DOCTRINE: this module is pure and stdlib-only, like src/disposition_messages.py
and src/forgotten_jobs.py. It must never import src.tools (which loads a
~500k-row database at module scope, ~109 s) nor src.planNexe2 (which imports
src.tools). In particular it is *duck-typed* on step objects -- it reads
.step/.agent/.timeStamp/.timeSpent and never constructs a `myPastStep`; the
caller in planNexe2 builds the models. That keeps the fast test tier fast and
keeps zero serialization risk.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

# --- Tuning knobs -------------------------------------------------------------
# Sized against the measured distribution (median step 1,481 chars, mean 2,813,
# worst single step 33,209) and a 200k-token Sonnet 4.5 window whose fixed worker
# floor is ~9k tokens (system prompt 4,980 + 17 tool schemas ~3k).
#
# THESE ARE NOT INDEPENDENT. Before changing K_VERBATIM or STEP_CHAR_CAP, read the
# thrash constraint below -- violating it makes the agent compact on every single
# turn, forever, burning an LLM call each time and never getting under the limit.
# `worst_post_compaction_tokens()` computes the bound and
# tests/test_past_steps.py::test_knobs_satisfy_the_thrash_constraint enforces it,
# so a bad combination fails the test suite rather than the campaign.

K_VERBATIM = 0          # recent steps always kept in full please keep 0!
SOFT_TOKENS = 11_000     # supervisor asks for a report at/above this
HARD_TOKENS = 16_000     # compact unconditionally at/above this
STEP_CHAR_CAP = 3_000    # per-step ingestion cap; clips the fat tail only
MIN_EVICT = 5            # never compact unless this many steps are evictable

# Per-rendered-line boilerplate: "12. OER_Agent: " plus the
# "[total time elapsed since project start: ..., time spent on step 12: ...]" tail.
_RENDER_OVERHEAD_CHARS = 100

# Anthropic's own approximate ratio (what langchain's count_tokens_approximately
# uses for anthropic-chat). Deliberately NOT importing that helper: it takes
# *messages*, not a string, and would drag langchain into this stdlib-pure module.
CHARS_PER_TOKEN = 3.3

# Total size of the leading digest block. Digests are cheap (~50-100 tokens each)
# but a 500-step campaign could accumulate ~40 of them, so fold the oldest once
# the block gets big. Folding is LLM-free -- it just lists the report keys.
DIGEST_BLOCK_CHAR_CAP = 8_000

# A step whose text starts with this is a digest, not a real step.
DIGEST_PREFIX = "[COMPACTED]"

DIGEST_AGENT = "system"

# --- Phase 2 (inner ReAct loop) -----------------------------------------------
# Tools whose ToolMessage bodies must NEVER be cleared by ClearToolUsesEdit.
# The first two are the dangerous ones: they are SIDE-EFFECTING, and blanking
# their results can make the agent believe the action never happened and REDO it
# -- i.e. re-submit real DFT jobs, or re-enter a candidate. The rest gate the
# workflow (disposition coverage, report bookkeeping, the wait gate) and the
# agent reasons directly off their text.
CLEARABLE_EXCLUDE_TOOLS: tuple[str, ...] = (
    "submit_dft_job",
    "enter_candidate_in_log",
    "update_disposition_info",
    "get_disposition_info",
    "write_report",
    "write_my_canvas",
    "wait_for_update",
)


def estimate_tokens(text: str) -> int:
    """Approximate Anthropic token count for a rendered string."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def worst_post_compaction_tokens(
    *,
    k: int = K_VERBATIM,
    step_char_cap: int = STEP_CHAR_CAP,
    digest_block_char_cap: int = DIGEST_BLOCK_CHAR_CAP,
) -> int:
    """Largest the ledger can be immediately AFTER a compaction: a full digest
    block plus `k` steps that each hit the per-step cap.

    THE constraint the knobs must satisfy is

        HARD_TOKENS > worst_post_compaction_tokens()

    Otherwise compaction leaves the ledger still over HARD, so the next turn
    compacts again -- an LLM call every turn, forever, that never converges. That
    is why K_VERBATIM and STEP_CHAR_CAP cannot be raised in isolation: at the
    current STEP_CHAR_CAP, K_VERBATIM=20 alone would blow past HARD_TOKENS.

    Keeping SOFT_TOKENS above this too is not required for correctness, but it is
    what stops the supervisor from being nagged for a report it just satisfied.
    """
    return estimate_tokens(
        "x" * (k * (step_char_cap + _RENDER_OVERHEAD_CHARS) + digest_block_char_cap)
    )


def is_digest(step: Any) -> bool:
    return str(getattr(step, "step", "")).startswith(DIGEST_PREFIX)


def count_leading_digests(steps: Sequence[Any]) -> int:
    n = 0
    for s in steps:
        if not is_digest(s):
            break
        n += 1
    return n


# Every digest starts with this exact, machine-parseable head, e.g.
# "[COMPACTED] (steps 25-38) ...". It is the ONLY place the absolute step range is
# read back out of a digest -- which is how the monotone step counter survives
# compaction WITHOUT a dedicated state channel (see steps_completed()). Human prose
# follows the head; the head itself is never free text.
_DIGEST_HEAD_RE = re.compile(re.escape(DIGEST_PREFIX) + r" \(steps (\d+)-(\d+)\)")


def _digest_head(step_lo: int, step_hi: int) -> str:
    return f"{DIGEST_PREFIX} (steps {step_lo}-{step_hi})"


def _digest_range(text: Any) -> tuple[int, int] | None:
    """(lo, hi) parsed from a digest's canonical head, or None if it has none."""
    m = _DIGEST_HEAD_RE.match(str(text))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _digest_hi(text: Any) -> int:
    r = _digest_range(text)
    return r[1] if r else 0


def _max_digest_hi(steps: Sequence[Any]) -> int:
    """Highest absolute step number recorded by any digest in the ledger (0 if
    none). Digests all lead, so this is the last real step the digest block covers."""
    return max((_digest_hi(s.step) for s in steps if is_digest(s)), default=0)


def steps_completed(steps: Sequence[Any]) -> int:
    """Total steps EVER completed, recovered from the ledger alone.

    = the last step the digest block accounts for, plus the verbatim steps kept
    after it. This replaces a would-be `steps_completed` state channel: the count
    is already latent in `past_steps` (a checkpointed channel), so no schema change
    is needed and resume from an old checkpoint is unaffected -- an old ledger has
    no `[COMPACTED]` digest, so `_max_digest_hi == 0` and this is exactly
    `len(steps)`, the pre-change behaviour.
    """
    n_verbatim = sum(1 for s in steps if not is_digest(s))
    return _max_digest_hi(steps) + n_verbatim


def render_past_steps(
    steps: Sequence[Any],
    *,
    with_timing: bool = True,
    start_index: int = 1,
) -> str:
    """Render the ledger for injection into a prompt.

    With no digests and start_index=1 this is byte-identical to the f-string it
    replaces (planNexe2's boss/supervisor/worker renders), so existing prompts
    are unchanged until a compaction actually happens.

    `start_index` is the ABSOLUTE index of the first verbatim step, so that after
    a compaction the supervisor sees "27." rather than the list restarting at
    "1." and silently implying the study is younger than it is.

    Digest lines are self-describing (they carry their own step range) and are
    emitted without a number, so they never consume an index.
    """
    lines: list[str] = []
    idx = start_index
    for step in steps:
        if is_digest(step):
            lines.append(str(step.step))
            continue
        if with_timing:
            lines.append(
                f"{idx}. {step.agent}: {step.step} "
                f"[total time elapsed since project start: {str(step.timeStamp).split('.')[0]}, "
                f"time spent on step {idx}: {step.timeSpent}]"
            )
        else:
            lines.append(f"{idx}. {step.agent}: {step.step}")
        idx += 1
    return "\n".join(lines)


def first_verbatim_index(steps: Sequence[Any]) -> int:
    """Absolute 1-based index of the first non-digest step in `steps`, derived from
    the digest block (the step after the last one the digests account for)."""
    return _max_digest_hi(steps) + 1


_TRUNC_MARK = "\n...[step text truncated at {cap} chars"


def truncate_step_text(
    text: str,
    cap: int = STEP_CHAR_CAP,
    archive_ref: str | None = None,
) -> tuple[str, bool]:
    """Clip one step's summary to `cap` chars.

    Exists purely to kill the fat tail: the median step is 1,481 chars but the
    worst one measured was 33,209 (~8,300 tokens), i.e. a single step able to eat
    a third of the history budget. Idempotent -- truncating twice is a no-op.
    """
    text = text or ""
    if len(text) <= cap:
        return text, False
    mark = _TRUNC_MARK.format(cap=cap)
    if archive_ref:
        mark += f"; full text at {archive_ref}"
    mark += "]"
    keep = max(0, cap - len(mark))
    return text[:keep] + mark, True


@dataclass(frozen=True)
class CompactionPlan:
    """What to evict, decided purely positionally so it is replay-deterministic."""

    should: bool
    evict_lo: int = 0   # inclusive index into `steps`
    evict_hi: int = 0   # exclusive
    reason: str = ""    # "" | "report" | "hard"

    @property
    def n_evicted(self) -> int:
        return max(0, self.evict_hi - self.evict_lo)


def plan_compaction(
    steps: Sequence[Any],
    *,
    k: int = K_VERBATIM,
    hard: int = HARD_TOKENS,
    report_written: bool = False,
    min_evict: int = MIN_EVICT,
) -> CompactionPlan:
    """Decide whether (and what) to compact. No LLM, no EXPLOG, no I/O.

    Evictable = the verbatim steps that are neither part of the leading digest
    block nor part of the last `k`. Compact when either:
      * a report was just written  -> free compaction, the report IS the summary; or
      * the rendered ledger is at/over `hard` -> forced, pay for an LLM summary.

    SOFT is deliberately absent: it drives the supervisor's "go write a report"
    nudge (see `should_request_report`), never a compaction.

    The `min_evict` floor is the anti-thrash guard. Without it a single fat step
    can push the ledger over `hard` while leaving nothing worth evicting, and the
    node would then fire an LLM summary every single turn, forever, achieving
    nothing.
    """
    n = len(steps)
    lo = count_leading_digests(steps)
    hi = n - k
    if hi - lo < min_evict:
        return CompactionPlan(should=False)

    if report_written:
        return CompactionPlan(should=True, evict_lo=lo, evict_hi=hi, reason="report")

    if estimate_tokens(render_past_steps(steps)) >= hard:
        return CompactionPlan(should=True, evict_lo=lo, evict_hi=hi, reason="hard")

    return CompactionPlan(should=False)


def should_request_report(steps: Sequence[Any], *, soft: int = SOFT_TOKENS) -> bool:
    """True when the supervisor should plan a `write_report` step now.

    Called by the supervisor node on its own `state["past_steps"]` -- it needs no
    flag from the worker, which is why the whole scheme carries no process-global
    state and survives resume for free.
    """
    the_estimate_tokens = estimate_tokens(render_past_steps(steps))
    print("===================")
    print("token estimation: ", the_estimate_tokens)
    print("===================")
    return the_estimate_tokens >= soft


def build_report_digest(
    *,
    report_name: str,
    report_id: str | None,
    step_lo: int,
    step_hi: int,
    gist: str | None = None,
    raw_steps_key: str | None = None,
) -> str:
    """The free, LLM-less digest: a pointer at a report already on CANVAS.

    `gist` is the worker's own one-sentence summary of the report (from the
    write_report tool); it becomes the human-readable body of the digest, so the
    compacted steps still say WHAT was accomplished at a glance, not just where to
    look."""
    ref = f" (ID={report_id})" if report_id else ""
    lead = f" {gist.strip()}" if gist and gist.strip() else ""
    raw = (
        f" Raw step text: read_my_canvas(key='{raw_steps_key}')."
        if raw_steps_key else ""
    )
    return (
        f"{_digest_head(step_lo, step_hi)}{lead} Distilled into report "
        f"'{report_name}'{ref} on CANVAS -- read_my_canvas(key='{report_name}') for "
        f"full detail.{raw}"
    )


def build_summary_digest(
    *, summary: str, step_lo: int, step_hi: int, raw_steps_key: str | None = None,
) -> str:
    """The HARD-path digest: an LLM summary of steps no report ever covered."""
    raw = (
        f" Full raw step text: read_my_canvas(key='{raw_steps_key}')."
        if raw_steps_key else ""
    )
    return (
        f"{_digest_head(step_lo, step_hi)} summary (compacted to keep the context "
        f"bounded; no report covered these).{raw}\n{summary}"
    )


def fold_digests(digests: Sequence[str], *, cap: int = DIGEST_BLOCK_CHAR_CAP) -> list[str]:
    """Keep the leading digest block itself bounded, LLM-free.

    Digests are cheap, but a 500-step campaign accumulates ~40 of them. Once the
    block exceeds `cap`, merge the oldest half into ONE digest that spans their whole
    range -- so nothing becomes unreachable (detail is still on CANVAS), it just costs
    a read_my_canvas lookup instead of sitting in context. The merged digest
    keeps a canonical head covering [min lo, max hi] of what it absorbed, so
    steps_completed() still reads the right number back out.
    """
    block = "\n".join(digests)
    if len(block) <= cap or len(digests) < 2:
        return list(digests)

    half = len(digests) // 2
    old, keep = digests[:half], digests[half:]
    ranges = [r for d in old if (r := _digest_range(d))]
    lo = min((r[0] for r in ranges), default=0)
    hi = max((r[1] for r in ranges), default=0)
    merged = (
        f"{_digest_head(lo, hi)} earlier work, already compacted across {len(old)} "
        f"report(s). Detail is on CANVAS -- inspect_my_canvas lists the report and "
        f"raw-step keys."
    )
    return [merged] + list(keep)
