"""
Safety / verification tools for the supervisor + worker LangGraph framework.

This revision aligns the safety guard with the migrated tools:

* Canonical provenance: every science tool now writes
  `artifact.parent_result_ids_w_args: Dict[str, str | List[str]]` — a per-
  parameter source map keyed by the same names that appear in `artifact.args`.
  Scalar-shaped params get a scalar source ID; list-shaped params get a list
  of source IDs aligned by index. The verifier reads this field directly
  instead of parsing argument-name conventions.

* Reasons are always dict-shaped. Tools that have a single logical input
  register their rationale as `{"reasons": "<text>"}`; per-parameter lookups
  fall through to that key when the parameter name is not present.

* The per-parameter judge reads the merged `Context:` line at the start of
  each `reason` to determine whether the artifact is part of production or
  sub-study work, and adjusts its scrutiny accordingly. The overall study
  goal and target quantity are no longer threaded directly into judge
  prompts — the `Context:` prefix already encodes the local purpose, and
  the judge does not need the report's top-level claim to evaluate a
  specific upstream parameter. The judge call accepts only what's locally
  necessary: tool name and description for grounding, parameter name and
  value, the rationale (with merged context), the source artifact summary
  if any, and the rule string for the current branch.

* List-shaped parameters are verified element-by-element for value match,
  then judged collectively (one judge call asking whether the N sources
  are appropriate for the list-valued parameter as a whole).

* Per-claim verification scope: each numerical_results entry verifies its
  subtree under that claim's own `varied_parameters`. The same upstream
  artifact verified for two different claims produces two artifact-result
  records, distinguishable by the `verified_for_quantity` field.

Authority model
---------------
* `CANVAS.result_registry` is the only source of authoritative values. Every
  numerical or textual value used by these tools is read from
  `artifact.value` after the registry hands the artifact back; equality is
  checked through `CANVAS.verify_artifact(expected_value, result_id)`.
* `CANVAS.canvas` is an agent scratchpad. The verifier never reads a value
  from canvas; the only thing it ever loads from canvas is the
  report-as-note itself.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Literal, Set, Annotated, Tuple
import os

from pydantic import BaseModel, Field
from langchain.tools import tool
from src.myCANVAS import CANVAS
from src import var
from src.verify_dag_visualizer import VerifyVisualizer


from langchain_anthropic import ChatAnthropic


# =========================================================
# Debug helper
# =========================================================

_DBG_SLEEP = 2  # seconds to sleep after each debug print


def _dbg(msg: str) -> None:
    """Print a debug message and sleep so logs are easy to follow live."""
    sleep_time = int(len(f"[DBG] {msg}") * 0.01 * _DBG_SLEEP)
    print(f"[DBG][{sleep_time}] {msg}", flush=True)
    # time.sleep(sleep_time)  # longer sleep for longer messages

# =========================================================
# Module constants
# =========================================================

EXTRACTION_TOOL_NAMES: Set[str] = {
    "extract_numeric_from_tool_output",
    "extract_text_from_tool_output",
}
# Tools that gather information rather than do science. Empty `reasons`
# entries on non-sensitive parameters of these tools are non-blocking.
INFO_TOOLS: Set[str] = {
    "inspect_my_canvas",
    "read_my_canvas",
    "write_my_canvas",
    "find_pseudopotential",
    "get_convergence_suggestions",
    "submit_and_monitor_job",
    "submit_single_job",
    "add_resource_suggestion",
    "read_single_output",
}

_CLAIM_TAG = re.compile(r"\{claim:([A-Za-z0-9_\-]+)\}")
_LIT_TAG = re.compile(r"\{lit:([^}]+)\}")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])")


# =========================================================
# Issue categorization for supervisor-facing output
# =========================================================
#
# The verifier emits two kinds of failures:
#
#   (1) per-parameter checks generated inside `verify_artifact_parameterization`,
#       `_verify_sourced_param`, `_verify_optional_source_match`, and
#       `_verify_extraction_behavior`. Each of these sites tags its
#       `ParamCheckResult` with a category at the moment of emission.
#
#   (2) report-/claim-level checks generated directly inside
#       `verify_structured_report` (empty overall_goal, claim value-mismatch,
#       artifact lookup failures, etc.) and `_verify_one_artifact_recursive`
#       (recursion failures). Same: the site supplies the category.
#
# The post-processor `summarize_verification_for_supervisor` then assembles
# a numbered, structured issue list keyed by these categories, attaches
# remediation options, and returns the supervisor-facing format.

class IssueCategory(str):
    """Closed set of issue category strings.

    Using a str subclass instead of an Enum keeps these values trivially
    JSON-serializable and avoids unnecessary friction at LLM boundaries
    (the LLM consumes the string value, not an enum object).
    """
    # Claim-level: the claim's value doesn't match its registered artifact.
    VALUE_MISMATCH_CLAIM = "value_mismatch_claim"
    # Per-parameter: a parameter's value doesn't match the upstream artifact
    # it was sourced from (deterministic check, before any judge involvement).
    VALUE_MISMATCH_PARAM = "value_mismatch_param"
    # Sensitive parameter has no upstream source and the per-parameter judge
    # decided this is a real failure (production context, or sub-study cover
    # that the judge found incoherent).
    UNSOURCED_SENSITIVE = "unsourced_sensitive"
    # Sensitive parameter has an upstream source but the judge said the
    # source is semantically inappropriate for this parameter or the
    # source's conditions don't match the current use.
    CROSS_WIRED_SOURCE = "cross_wired_source"
    # Sensitive parameter is sourced via a dotted reference into an
    # earlier tool call's input AND the current call is production
    # work AND the upstream input was itself uncharacterized (held by
    # design within its origin call, not drawn from an upstream
    # characterization). Production cannot legitimately consume an
    # uncharacterized value; a dotted ref to one launders the missing
    # characterization.
    DOTTED_SOURCE_INAPPROPRIATE = "dotted_source_inappropriate"
    # Sensitive-and-varied parameter where the judge flagged the value as
    # not making sense as a sweep point for the stated study.
    UNDER_JUSTIFIED_SWEEP = "under_justified_sweep"
    # Non-sensitive parameter where the judge flagged the rationale as
    # missing workflow context.
    UNDER_JUSTIFIED_CHOICE = "under_justified_choice"
    # Extraction-tool output (numeric or text) doesn't appear as a complete
    # token in its source.
    EXTRACTION_TOKEN_MISMATCH = "extraction_token_mismatch"
    # Extraction judge flagged the extraction as unjustified or wrong.
    EXTRACTION_JUDGE_FAIL = "extraction_judge_fail"
    # Extraction has no recoverable source text — surfaces as warning.
    EXTRACTION_NO_SOURCE = "extraction_no_source"
    # Report has empty overall_goal or other report-generation-time
    # schema violation.
    SCHEMA_VIOLATION = "schema_violation"
    # An artifact result_id couldn't be loaded from the registry.
    ARTIFACT_LOOKUP_FAILED = "artifact_lookup_failed"
    # The recursive descent into a child artifact raised an exception.
    RECURSION_FAILURE = "recursion_failure"
    # Catch-all for ParamCheckResult.verdict in {fail, warning} that
    # somehow didn't get a more specific category. Should not appear in
    # practice; a debug aid if a new code path forgets to tag.
    UNCATEGORIZED = "uncategorized"


# Sentinel used as ParamCheckResult.rule_applied when the deterministic
# value-match check fails. Distinguishes the value-mismatch failure mode
# from the LLM-judge failure mode that uses the actual rule string.
VALUE_MATCH_SENTINEL = "VALUE_MATCH"


# Per-category remediation hints. Each value is a list of legitimate
# fix paths the supervisor can choose between. The post-processor attaches
# the relevant list to each issue's `remediation_options` field.
#
# Two cross-cutting reminders are appended where relevant:
#   * "DO NOT silently patch by attaching some other artifact's ID..."
#     (anti-ref-patching guidance for source-related failures)
#   * "Regenerating an artifact invalidates downstream consumers..."
#     (the regeneration cascade reminder, for any fix that involves
#     recreating an artifact)
# These are expressed as standalone constants and concatenated into the
# remediation lists below so the wording stays consistent.

_NO_REF_PATCHING_WARNING = (
    "DO NOT resolve this by finding some other artifact's `result_id` and "
    "patching it in as the missing ref. The verifier will catch a "
    "value-mismatch (the patched artifact's value won't match what was "
    "actually used) or a cross-wiring (the patched artifact may be "
    "semantically inappropriate). The legitimate fixes are listed above."
)

_REGEN_CASCADE_WARNING = (
    "IMPORTANT: regenerating an artifact invalidates every artifact "
    "downstream that referenced it. After fixing this artifact, you MUST "
    "re-run every dependent calculation (production runs, ensemble "
    "calculations, downstream extractions) using the new artifact's "
    "`result_id`. The verifier will surface stale references on the next "
    "verification pass."
)


REMEDIATION_OPTIONS: Dict[str, List[str]] = {
    IssueCategory.VALUE_MISMATCH_CLAIM: [
        "The numerical value in the claim does not match what the "
        "registered artifact actually produced. Re-read the artifact's "
        "value via CANVAS and update the claim's `value` field to match, "
        "OR verify that the claim is pointing at the correct `result_id`.",
        "If the claim is pointing at the wrong artifact (e.g. the agent "
        "captured the result_id of an intermediate calculation instead of "
        "the final one), update the `result_id` to the correct artifact.",
    ],
    IssueCategory.VALUE_MISMATCH_PARAM: [
        "The parameter's value does not match the upstream artifact it "
        "was sourced from. Either update the parameter's value to match "
        "the upstream artifact's actual output, OR — if the value is "
        "correct — find the actual upstream artifact whose value matches "
        "and update the ref. The current ref is incorrect.",
        _NO_REF_PATCHING_WARNING,
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.UNSOURCED_SENSITIVE: [
        "Characterize the parameter upstream by running a sub-study (e.g. "
        "a convergence test via "
        "generate_convergence_test/submit_and_monitor_job/find_optimal_parameter), "
        "then re-create the offending artifact with the appropriate "
        "`*_ref` pointing at the optimization result.",
        "If this parameter is being deliberately swept for the claim, "
        "add it to the claim's `varied_parameters` list and re-generate "
        "the report.",
        "If this artifact is misclassified (i.e. it is actually part of a "
        "sub-study and not production work), update the tool call's "
        "`context` argument to clearly identify the sub-study purpose "
        "(e.g. 'convergence test for ecutwfc'), re-create the artifact, "
        "and re-generate the report.",
        _NO_REF_PATCHING_WARNING,
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.CROSS_WIRED_SOURCE: [
        "The judge determined the upstream source is semantically wrong "
        "for this parameter (e.g. an ecutwfc convergence result used as a "
        "kspacing pin), or the source's characterization conditions don't "
        "match this artifact's use. Find or create the correct upstream "
        "artifact and re-create this artifact with the proper `*_ref`.",
        "If the source IS semantically correct but the judge raised a "
        "conditions-mismatch concern, examine the per-parameter rationale "
        "and add justification for why the source's conditions are still "
        "appropriate for this use; re-create the artifact.",
        _NO_REF_PATCHING_WARNING,
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.DOTTED_SOURCE_INAPPROPRIATE: [
        "The current call is production work and is consuming a value "
        "that was never characterized. Run the appropriate sub-study "
        "(convergence test, sensitivity sweep) that PRODUCES an "
        "output artifact for this value, then re-create the current "
        "artifact with a bare `*_ref` pointing at that output. Do "
        "not point at the upstream input via a dotted ref — the "
        "value still wouldn't be characterized.",
        # "If the current artifact is misclassified as production (it "
        # "is actually part of a sub-study or exploration), update the "
        # "tool call's `context` argument to identify the sub-study "
        # "purpose and re-create the artifact. In a sub-study context, "
        # "the dotted ref is acceptable.",
        _NO_REF_PATCHING_WARNING,
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.UNDER_JUSTIFIED_SWEEP: [
        "The judge flagged this swept value as not making sense for the "
        "stated study. Either correct the value to a sensible sweep point "
        "and re-create the artifact, or strengthen the per-parameter "
        "rationale to explain why this specific value is a coherent sweep "
        "point for the study named in the `Context:` line.",
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.UNDER_JUSTIFIED_CHOICE: [
        "The judge flagged this parameter's rationale as missing workflow "
        "context. Update the per-parameter rationale to explicitly "
        "identify the study and the parameter's role in it; re-create "
        "the artifact.",
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.EXTRACTION_TOKEN_MISMATCH: [
        "The extracted value does not appear as a complete token in the "
        "source. Re-run the extraction tool with the correct value/text, "
        "or correct the `evidence_snippet` to point at the actual region "
        "of the source where the value appears.",
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.EXTRACTION_JUDGE_FAIL: [
        "The extraction-judge LLM determined the extraction is "
        "unjustified or semantically wrong. Re-examine the source "
        "artifact and either re-extract with a corrected value/snippet "
        "or strengthen the per-extraction rationale.",
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.EXTRACTION_NO_SOURCE: [
        "The extraction's source text could not be recovered from the "
        "registry. Confirm the source artifact is properly registered "
        "with retrievable text, or re-run the extraction with a "
        "different (recoverable) source.",
    ],
    IssueCategory.SCHEMA_VIOLATION: [
        "The report does not satisfy a generation-time schema "
        "requirement. Re-call `generate_structured_report` with the "
        "required field corrected (e.g. provide a meaningful "
        "`overall_goal`, fix duplicate quantity_names, add citations "
        "for all numerical mentions in prose, etc.).",
    ],
    IssueCategory.ARTIFACT_LOOKUP_FAILED: [
        "An artifact `result_id` referenced by the report or by the "
        "value-flow chain could not be loaded from CANVAS. The id may be "
        "malformed or may refer to an artifact that was never registered. "
        "Confirm the result_id is correct and that the upstream tool "
        "call actually produced an artifact.",
    ],
    IssueCategory.RECURSION_FAILURE: [
        "The recursive verification descent raised an exception, likely "
        "due to a malformed artifact in the chain. Inspect the artifact "
        "registry for the offending id, then either repair its metadata "
        "or re-create the affected upstream artifact.",
        _REGEN_CASCADE_WARNING,
    ],
    IssueCategory.UNCATEGORIZED: [
        "This issue could not be automatically categorized. Inspect the "
        "raw `judge_reasoning` and `problem` fields and consult the "
        "verifier's debug log for diagnostic context.",
    ],
}


def _problem_text_for_param_check(
    *,
    category: str,
    parameter_name: str,
    parameter_value: Any,
    tool_name: str,
    verdict: str,
) -> str:
    """Build a one-sentence `problem` string for a per-parameter issue.

    Specialized per category so the supervisor sees an immediately
    actionable summary without having to read the longer `judge_reasoning`.
    """
    val_repr = repr(parameter_value)
    if category == IssueCategory.VALUE_MISMATCH_PARAM:
        return (
            f"Parameter '{parameter_name}' on a `{tool_name}` artifact has "
            f"value {val_repr}, which does not match the upstream artifact "
            f"the agent referenced as its source. The deterministic value-"
            f"match check failed."
        )
    if category == IssueCategory.UNSOURCED_SENSITIVE:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact is sensitive but has no upstream "
            f"source artifact. The per-parameter judge — reading the "
            f"`Context:` line of the rationale — found this missing source "
            f"unjustified."
        )
    if category == IssueCategory.CROSS_WIRED_SOURCE:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact has an upstream source whose "
            f"value matches numerically, but the per-parameter judge "
            f"flagged the source as semantically inappropriate or "
            f"obtained under conditions that don't match the current use."
        )
    if category == IssueCategory.DOTTED_SOURCE_INAPPROPRIATE:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact is sourced via a dotted reference "
            f"into an earlier tool call's INPUT. The current call is "
            f"production work, and the per-parameter judge determined "
            f"the referenced upstream input was itself uncharacterized "
            f"(a deliberately-held sub-study value, not a "
            f"characterized result). Production work cannot inherit "
            f"a value that was never characterized."
        )
    if category == IssueCategory.UNDER_JUSTIFIED_SWEEP:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact is being swept, but the per-parameter "
            f"judge flagged the value as not making sense as a sweep point "
            f"for the study described in the rationale's `Context:` line."
        )
    if category == IssueCategory.UNDER_JUSTIFIED_CHOICE:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact has a rationale the per-parameter "
            f"judge found incomplete — typically missing identification "
            f"of the study and the parameter's role in it."
        )
    if category == IssueCategory.EXTRACTION_TOKEN_MISMATCH:
        return (
            f"Extraction artifact (`{tool_name}`) value {val_repr} does "
            f"not appear as a complete token in the source text — likely "
            f"a substring extraction (e.g. extracting '123' out of "
            f"'1234.56') or a value the agent computed rather than read."
        )
    if category == IssueCategory.EXTRACTION_JUDGE_FAIL:
        return (
            f"Extraction artifact (`{tool_name}`) extracted value "
            f"{val_repr} but the extraction judge flagged the extraction "
            f"as semantically wrong or unjustified."
        )
    if category == IssueCategory.EXTRACTION_NO_SOURCE:
        return (
            f"Extraction artifact (`{tool_name}`) cannot be verified: "
            f"no source text was recoverable from the artifact's args or "
            f"from any registered parent artifact."
        )
    return (
        f"Parameter '{parameter_name}' (value {val_repr}) on a "
        f"`{tool_name}` artifact has verdict={verdict!r} but no specific "
        f"problem template matched the issue category."
    )


# --- Rule strings used inside verify_artifact_parameterization ---------------
#
# Every rule begins by asking the judge to read the `Context:` line that
# opens the parameter's `reason` (the per-call context is merged in by
# `_merge_context` in tools.py before the artifact is registered, so every
# rationale starts with "Context: <study description>"). The judge uses the
# context to determine whether this artifact is part of the report's
# production lineage or part of an earlier sub-study (a convergence test, an
# EOS sweep, a sensitivity scan, etc.) and adjusts its scrutiny accordingly.

R1_RULE = (
    "Parameter is sensitive, not being varied for the current target "
    "quantity, and is sourced from the OUTPUT of another tool call (a "
    "bare 8-character result_id ref). The source's value has already "
    "been confirmed to match by deterministic check; your job is to "
    "weigh whether that output is an appropriate origin for this "
    "parameter's value. "
    "Begin by reading the `Context:` line at the start of the current "
    "parameter's rationale. Determine whether the current artifact is "
    "part of the report's production lineage or part of an earlier "
    "sub-study. Then judge: does the source artifact's tool, "
    "description, and produced value make semantic sense as the origin "
    "of this parameter's value, given the study context? "
    "Reject when the source is plausibly cross-wired (e.g. an ecutwfc "
    "convergence result used as a kspacing pin), when the source's "
    "characterization conditions clearly don't match the current use "
    "(e.g. a kspacing converged at low ecutwfc used at high ecutwfc), "
    "or when the rationale fails to identify the workflow context."
)

R1_DOTTED_RULE = (
    "Parameter is sensitive, not being varied for the current target "
    "quantity, and is sourced from an INPUT PARAMETER of an earlier "
    "tool call (a dotted reference of the form '<id>.<field>'). The "
    "agent is asserting: 'this value should be the same value that was "
    "used as the named input to that earlier call.' The deterministic "
    "value-match has already confirmed the values agree; your job is "
    "to weigh whether that earlier input is an appropriate origin for "
    "the current parameter's value. "
    "Read TWO contexts: "
    "  (1) the `Context:` line at the start of the CURRENT parameter's "
    "      rationale (what study is the current call part of?), and "
    "  (2) the `Context:` line at the start of the source input's "
    "      rationale, which is provided as `input_parameter_rationale` "
    "      in the source summary (what study was the earlier call part "
    "      of, and what role did that input play there?). "
    "Then judge whether sourcing the current value from that earlier "
    "input is coherent: "
    "  • If the current call is production work, the source input must have "
    "    been characterized with a bare ref — production work cannot "
    "    legitimately consume an uncharacterized value. A dotted ref to an "
    "    uncharacterized input launders the missing characterization and is "
    "    therefore unacceptable. Reject when the source input was "
    "    uncharacterized, or when the rationale fails to identify the "
    "    workflow context of either the current call or the source input."
)



R1_NO_SOURCE_RULE = (
    "Parameter is sensitive, not being varied for the current target "
    "quantity, and has NO upstream source artifact. The verdict depends "
    "entirely on the workflow context the agent has recorded. "
    "Begin by reading the `Context:` line at the start of the parameter's "
    "rationale. The context names the study or exploration this tool call "
    "is part of. Identify which kind of work it describes, and judge "
    "accordingly: "
    "  • PRODUCTION work — phrases like 'production run for adsorption "
    "    energy', 'final relaxation', 'BEEF ensemble for adsorption', or "
    "    similar terminal calculations whose output is being claimed. "
    "    Production calls require every sensitive parameter to be either "
    "    upstream-characterized or being varied. A missing source here is "
    "    a real failure — return fail. "
    "  • SUB-STUDY work — phrases like 'convergence test for ecutwfc', "
    "    'EOS sweep', 'sensitivity scan over n_fixed_layers', 'one-off "
    "    exploration'. Sub-studies legitimately hold parameters at "
    "    deliberately-chosen values (without upstream characterization) "
    "    while the sub-study characterizes some other parameter. Pass if "
    "    all of the following hold: "
    "    (a) the rationale clearly identifies the sub-study purpose; "
    "    (b) the held value is a reasonable choice for that sub-study "
    "        (e.g. a tight kspacing while ecutwfc is being characterized); "
    "    (c) the sub-study's purpose is consistent with the artifact's "
    "        tool name and the parameter being held — i.e. the parameter "
    "        being held is plausibly held FOR THE PURPOSE the context "
    "        names. "
    "    Return fail if the rationale is incoherent, the held value is "
    "    unusual without justification, or the context-tool mismatch "
    "    suggests the agent is fabricating a sub-study cover for what "
    "    was actually a forgotten reference. "
    "Do not default to either verdict. The context decides."
)

R2_RULE = (
    "Parameter is sensitive and intentionally varied for the current "
    "target quantity. Its value must be sensible as a sweep point. "
    "Begin by reading the `Context:` line at the start of the parameter's "
    "rationale. Determine whether the current artifact is part of the "
    "report's production lineage or part of an earlier sub-study. Then "
    "judge: is this specific value a coherent sweep point for the study "
    "the context names? "
    "Atypical values (e.g. very low ecutwfc) are acceptable when the "
    "study context justifies them — for example a convergence sweep "
    "deliberately includes low values to characterize the convergence "
    "curve. Reject only when the value is incoherent for the stated "
    "study, when the rationale fails to identify the sweep's purpose, or "
    "when the context-tool mismatch suggests fabrication."
)

R3_RULE = (
    "Parameter is not under the hard provenance rule (it is not in the "
    "sensitive-parameters list). Its value must still be sensible based "
    "on the recorded rationale. "
    "Begin by reading the `Context:` line at the start of the parameter's "
    "rationale. Determine whether the current artifact is part of the "
    "report's production lineage or part of an earlier sub-study. Then "
    "judge: does the rationale situate the parameter coherently in that "
    "study? "
    "Approve rationales that identify the study and the role of the "
    "parameter — even when the value itself looks atypical, as long as "
    "the context justifies it. Reject rationales that lack workflow "
    "context (e.g. 'standard value' with no further detail)."
)

PARAM_JUDGE_GUIDANCE = """
Guidance:
- Always begin by reading the `Context:` line of the rationale. It tells
  you whether this artifact is part of the production lineage of the
  claim or part of an earlier sub-study, and you must adjust scrutiny
  accordingly.
- A rationale that explicitly situates the parameter within a study
  (production run / convergence test / sensitivity sweep / etc.) and
  explains the parameter's role in that study generally deserves PASS,
  even if the value itself looks atypical out of context.
- A rationale that lists only "intent and effect" without identifying
  the study or the parameter's role in it deserves WARNING — the value
  may be fine, but the reviewer cannot confirm without that context.
- A rationale that gives no workflow context at all, and the value would
  be questionable in a default production setting, deserves FAIL.
- Watch for context-tool mismatches: if the context says "convergence
  test for ecutwfc" but the artifact's tool is `analyze_BEEF_result` (a
  production-stage tool), the rationale is suspicious and the verdict
  should reflect that.
"""

# =========================================================
# Shared schemas
# =========================================================

class QuantitySpec(BaseModel):
    """One numerical claim the agent wants to put in the report."""
    quantity_name: str = Field(
        description=(
            "Unique name for this quantity across the report. Also the "
            "citation handle: use {claim:<quantity_name>} in narrative text."
        )
    )
    value: float = Field(
        description=(
            "The numerical value being claimed. Must match the artifact "
            "identified by `result_id`."
        )
    )
    result_id: str = Field(
        description="ID of the artifact that produced `value`."
    )
    varied_parameters: List[str] = Field(
        default_factory=list,
        description=(
            "Parameters whose values were intentionally swept to obtain this "
            "quantity. Used by the verifier to decide which sensitive "
            "parameters in the supporting chain need provenance vs. may be "
            "set directly. Include only parameters actually being varied."
        ),
    )
    unit: Optional[str] = Field(
        default=None, description="Unit string, e.g. 'eV', 'Angstrom', 'GPa'."
    )
    note: str = Field(
        default="",
        description="Optional one-sentence context for this claim.",
    )


class ReportNumericalClaim(BaseModel):
    quantity_name: str
    value: float
    unit: Optional[str] = None
    result_id: str
    varied_parameters: List[str] = Field(default_factory=list)
    note: str = ""


class StructuredScientificReport(BaseModel):
    overall_goal: str
    quantities_sought: List[Dict[str, Any]]
    numerical_results: List[ReportNumericalClaim]
    qualitative_findings: str = ""
    conclusion: str = ""
    rendered_markdown: str = ""


class ParamCheckResult(BaseModel):
    parameter_name: str
    parameter_value: Any
    verdict: Literal["pass", "fail", "warning", "info"]
    rule_applied: str
    source_result_id: Optional[Any] = None  # str or List[str]
    reasoning: str
    # Category tag set at the emitting site so the post-processor can
    # build a categorized issue list without inferring from prose.
    # Optional because pass-verdict checks don't need a category.
    category: Optional[str] = None


class ArtifactVerificationResult(BaseModel):
    result_id: str
    tool_name: str
    artifact_description: str
    overall_verdict: Literal["pass", "fail", "warning"]
    summary: str
    checks: List[ParamCheckResult] = Field(default_factory=list)
    recursive_children_checked: List[str] = Field(default_factory=list)
    verified_for_quantity: str = ""


class ReportVerificationIssue(BaseModel):
    """Structured, supervisor-facing description of one verification failure.

    Designed to be self-contained: the supervisor can act on a single
    issue without needing to consult the artifact tree or the per-claim
    walks. Numbering is assigned by the post-processor.
    """
    issue_number: int = 0  # 1-indexed; assigned by summarize_verification_for_supervisor
    category: str  # one of IssueCategory.* values
    severity: Literal["fail", "warning"]
    where: Dict[str, Any] = Field(default_factory=dict)
    # `where` keys (subset present per issue):
    #   "claim_name":   str
    #   "claim_value":  float
    #   "claim_unit":   str
    #   "result_id":    str  (the artifact at fault, when applicable)
    #   "tool_name":    str  (the tool that produced the offending artifact)
    #   "parameter":    str  (the parameter name, when at param level)
    #   "parameter_value": Any
    #   "child_id":     str  (when the issue is a recursion failure on a child)
    #   "field":        str  (e.g. "overall_goal" for schema violations)
    context_at_site: Optional[str] = None
    # The merged "Context:" line from the rationale at the offending site,
    # extracted when available. Helps the supervisor see how the agent
    # framed the call.
    problem: str
    # One-sentence statement of what's wrong, in plain prose.
    judge_reasoning: Optional[str] = None
    # The judge's verdict text when the issue came from an LLM judge call.
    remediation_options: List[str] = Field(default_factory=list)
    # Filled by the post-processor from REMEDIATION_OPTIONS[category].


class ReportVerificationResult(BaseModel):
    overall_verdict: Literal["pass", "fail", "warning"]
    checked_result_ids: List[str]
    issues: List[ReportVerificationIssue]
    artifact_results: List[ArtifactVerificationResult]


class LLMParameterJudgement(BaseModel):
    verdict: Literal["pass", "fail", "warning"]
    reasoning: str


# =========================================================
# Helpers
# =========================================================

def _get_artifact(result_id: str) -> Any:
    _dbg(f"_get_artifact: ENTER result_id={result_id!r}")
    artifact = CANVAS.get_artifact(result_id)
    if artifact is None:
        _dbg(f"_get_artifact: registry returned None for {result_id!r} — about to raise")
        raise ValueError(f"Result id '{result_id}' not found in result_registry.")
    n_args = len(getattr(artifact, "args", {}) or {})
    has_pria = bool(getattr(artifact, "parent_result_ids_w_args", {}) or {})
    _dbg(
        f"_get_artifact: OK tool_name={artifact.tool_name!r} "
        f"n_args={n_args} parent_result_ids_w_args_nonempty={has_pria}"
    )
    return artifact


def _flatten_listed_value(artifact: Any) -> Any:
    v = artifact.value
    if isinstance(v, list):
        out = [getattr(item, "value", item) for item in v]
        _dbg(f"_flatten_listed_value: list-shaped len={len(out)}")
        return out
    _dbg(f"_flatten_listed_value: scalar-shaped type={type(v).__name__}")
    return v


def _is_listed(artifact: Any) -> bool:
    return isinstance(artifact.value, list)


def _param_source_ids(artifact: Any) -> Dict[str, Any]:
    """Per-parameter source map.

    Scalar entries are str; list entries are List[str], aligned by index
    with the corresponding entry in `artifact.args`.
    """
    pria = getattr(artifact, "parent_result_ids_w_args", {}) or {}
    shapes = {k: ("list" if isinstance(v, list) else type(v).__name__) for k, v in pria.items()}
    _dbg(f"_param_source_ids: keys={list(pria.keys())} shapes={shapes}")
    return pria


def _collect_recursive_source_ids(artifact: Any) -> List[str]:
    """All upstream BARE result_ids referenced by this artifact, flattened.

    Empty strings are filtered out — they mark "no source declared", not a
    real upstream artifact, so the recursive walker must not try to
    dereference them.

    Dotted refs ('<id>.<field>') are EXCLUDED here and are NOT recursed
    into at all. The dotted-source judge call at the immediate
    downstream parameter produces the verdict for the cross-reference;
    the walker does not descend through the dotted ref to verify the
    source artifact's other parameters or its named field's deeper
    chain. This keeps the dotted-source semantics narrow: the agent's
    claim is about that one value, and the verifier's findings stay
    scoped to that one value.
    """
    source_ids: Set[str] = set(
        s for s in (artifact.parent_result_ids or [])
        if s and not _is_dotted_ref(s)
    )
    for v in _param_source_ids(artifact).values():
        if isinstance(v, list):
            source_ids.update(
                s for s in v
                if isinstance(s, str) and s and not _is_dotted_ref(s)
            )
        elif isinstance(v, str) and v and not _is_dotted_ref(v):
            source_ids.add(v)
    out = list(source_ids)
    _dbg(
        f"_collect_recursive_source_ids: result_id={artifact.result_id!r} "
        f"collected n={len(out)} bare ids={out}"
    )
    return out


def _summarize_artifact(artifact: Any) -> Dict[str, Any]:
    """Compact, JSON-serializable view used inside LLM judge prompts."""
    value_repr = repr(_flatten_listed_value(artifact))
    _dbg(
        f"_summarize_artifact: result_id={artifact.result_id!r} "
        f"tool_name={artifact.tool_name!r} value_repr_len={len(value_repr)}"
    )
    return {
        "result_id": artifact.result_id,
        "tool_name": artifact.tool_name,
        "description": artifact.description,
        "args": artifact.args,
        "reasons": artifact.reasons,
        "parent_result_ids": artifact.parent_result_ids,
        "metadata": artifact.metadata,
        "value_repr": value_repr,
    }


def _is_dotted_ref(ref: Any) -> bool:
    """A ref is dotted if it's a non-empty string containing a single dot
    with both halves non-empty. The CANVAS-level validator handles strict
    parsing; this helper is used only for branching decisions in the
    verifier, so it's intentionally permissive."""
    if not isinstance(ref, str) or not ref:
        return False
    return "." in ref


def _strip_dotted_field(ref: Any) -> Any:
    """Return the bare result_id portion of a ref. For a dotted ref
    '<id>.<field>' return '<id>'. For a bare ref return it unchanged.
    For empty / non-string inputs return as-is. Used by the chain
    walker when it needs to descend into the parent artifact regardless
    of which form was passed."""
    if isinstance(ref, str) and "." in ref:
        return ref.split(".", 1)[0]
    return ref


def _any_dotted(source: Any) -> bool:
    """For a per-parameter source entry (scalar string or list of
    strings), return True if any element is a dotted ref."""
    if isinstance(source, str):
        return _is_dotted_ref(source)
    if isinstance(source, list):
        return any(_is_dotted_ref(s) for s in source)
    return False


def _summarize_artifact_dotted(ref: str) -> Dict[str, Any]:
    """Source summary for a dotted ref. The agent is referencing a
    specific INPUT PARAMETER of a past tool call, not the call's
    output. The summary surfaces:
      - the source artifact's tool name and description (so the judge
        knows what kind of call this input belonged to);
      - the specific input parameter name being referenced;
      - the value of that input as recorded in args;
      - the RATIONALE the agent gave for that input at registration time
        (which embeds its own `Context:` line, telling the judge whether
        the source call was production or sub-study work).
    The `kind` field flags the dotted case so the judge does not
    confuse this with an output-shaped source.
    """
    bare_id, field = ref.split(".", 1)
    artifact = _get_artifact(bare_id)
    args = artifact.args or {}
    reasons = artifact.reasons or {}
    field_value = args.get(field)
    field_rationale = reasons.get(field) or reasons.get("reasons") or ""
    return {
        "kind": "input_parameter_reference",
        "source_artifact_id": bare_id,
        "source_tool_name": artifact.tool_name,
        "source_tool_description": artifact.description or "",
        "input_parameter_name": field,
        "input_parameter_value": field_value,
        "input_parameter_rationale": field_rationale,
    }


def _build_source_summary(ref: Any) -> Any:
    """Dispatcher used by the per-parameter judge call sites. Builds the
    right summary shape based on whether the ref is bare (output
    reference) or dotted (input-parameter reference). Handles
    list-shaped refs element-by-element, so a mix of bare and dotted
    refs in one list-shaped source produces a list of heterogeneously
    shaped summary dicts."""
    if isinstance(ref, list):
        return [_build_source_summary(r) for r in ref]
    if isinstance(ref, str) and _is_dotted_ref(ref):
        return _summarize_artifact_dotted(ref)
    # Bare ref path. Empty / non-string refs should not reach here.
    return _summarize_artifact(_get_artifact(ref))


def _fold_verdict(current: str, incoming: str) -> str:
    """`fail` > `warning` > `pass`. `info` is non-blocking."""
    if current == "fail" or incoming == "fail":
        folded = "fail"
    elif current == "warning" or incoming == "warning":
        folded = "warning"
    else:
        folded = "pass"
    _dbg(f"_fold_verdict: ({current!r}, {incoming!r}) -> {folded!r}")
    return folded


def _is_complete_numeric_token(needle: str, haystack: str) -> bool:
    try:
        target = float(needle)
    except (TypeError, ValueError):
        _dbg(f"_is_complete_numeric_token: needle={needle!r} not parseable as float -> False")
        return False
    for m in _NUMBER_RE.finditer(haystack):
        try:
            if math.isclose(float(m.group()), target,
                            rel_tol=1e-12, abs_tol=1e-12):
                _dbg(f"_is_complete_numeric_token: needle={needle!r} matched token={m.group()!r} -> True")
                return True
        except ValueError:
            continue
    _dbg(f"_is_complete_numeric_token: needle={needle!r} no complete-token match -> False")
    return False


def _looks_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False


def _is_complete_text_token(needle: str, haystack: str) -> bool:
    n = str(needle).strip()
    h = str(haystack)
    result = bool(n) and n in h
    _dbg(
        f"_is_complete_text_token: needle_len={len(n)} haystack_len={len(h)} -> {result}"
    )
    return result


def _normalize_to_list_pair(
    param_value: Any,
    source: Any,
    param_name: str,
) -> Tuple[bool, List[Any], List[str], Optional[str]]:
    """Decide list-vs-scalar shape and validate consistency.

    Returns
    -------
    (is_list, values, sources, error_message)
        * is_list   — True iff both sides are lists.
        * values    — list of bare values (singleton when scalar).
        * sources   — list of source result_ids (singleton when scalar).
        * error_message — non-None iff there's a structural mismatch the
          caller should fail on.
    """
    val_is_list = isinstance(param_value, list)
    src_is_list = isinstance(source, list)
    _dbg(
        f"_normalize_to_list_pair: param={param_name!r} "
        f"val_is_list={val_is_list} src_is_list={src_is_list}"
    )

    if val_is_list != src_is_list:
        _dbg(f"_normalize_to_list_pair: SHAPE MISMATCH for param={param_name!r}")
        return False, [], [], (
            f"Parameter '{param_name}': value is "
            f"{'a list' if val_is_list else 'a scalar'} but source is "
            f"{'a list' if src_is_list else 'a scalar'}. List-shape "
            "mismatch — the producing tool registered inconsistent "
            "provenance."
        )

    if val_is_list:
        if len(param_value) != len(source):
            _dbg(
                f"_normalize_to_list_pair: LENGTH MISMATCH param={param_name!r} "
                f"value_len={len(param_value)} source_len={len(source)}"
            )
            return False, [], [], (
                f"Parameter '{param_name}': value list has length "
                f"{len(param_value)} but source list has length "
                f"{len(source)}. List-shape mismatch — the producing tool "
                "registered inconsistent provenance."
            )
        _dbg(
            f"_normalize_to_list_pair: list-shape OK param={param_name!r} "
            f"len={len(param_value)}"
        )
        return True, list(param_value), list(source), None

    _dbg(f"_normalize_to_list_pair: scalar-shape OK param={param_name!r}")
    return False, [param_value], [source], None


# =========================================================
# LLM judge calls
# =========================================================

def _call_param_judge_llm(
    *,
    tool_name: str,
    tool_description: str,
    parameter_name: str,
    parameter_value: Any,
    reason: str,
    source_artifact_summary: Optional[Any],   # dict or list-of-dicts
    rule_to_apply: str,
    judge,
) -> Dict[str, Any]:
    """Slim per-parameter judge call.

    Inputs intentionally minimal:
      - `tool_name` and `tool_description` ground the judge in what kind
        of artifact we're judging (e.g. "this is a `find_optimal_parameter`
        artifact, which selects the best parameter from a sweep").
      - `parameter_name` / `parameter_value`: the parameter under review.
      - `reason`: includes the merged "Context: <study description>"
        prefix from the agent's tool call. The judge reads this to
        determine whether the artifact is production or sub-study work
        and adjusts scrutiny accordingly.
      - `source_artifact_summary`: when present (R1 with source), the
        judge can weigh whether the source is the right one.
      - `rule_to_apply`: the specific rule string for this branch
        (R1, R1-no-source, R2, R3) — the rule itself instructs the judge
        on what to look for.

    Dropped on purpose: overall_goal (already in `reason` via
    _merge_context), target_quantity (judge doesn't need to know the
    top-level claim to judge a specific parameter — the parameter's own
    Context line tells it what study this call is part of), varied/
    sensitive parameters (already factored into rule selection upstream),
    artifact_summary (the bits the judge needs are tool_name/description).
    """
    if source_artifact_summary is None:
        source_block = "None"
    else:
        source_block = json.dumps(source_artifact_summary, indent=2, default=str)

    prompt = f"""
You are verifying whether one tool parameter was set correctly in a scientific
agent workflow.

Tool that produced this artifact:
  name:        {tool_name}
  description: {tool_description}

Parameter under review:
  name:  {parameter_name}
  value: {repr(parameter_value)}
  reason given by the agent: {reason}

Source artifact(s) for this parameter, if any:
{source_block}

Rule that must be applied:
{rule_to_apply}

{PARAM_JUDGE_GUIDANCE}

Return:
- pass     : parameter setting is sensible and consistent with the rule
- fail     : parameter setting is not acceptable
- warning  : plausible but under-justified or ambiguous
"""
    _dbg(
        f"_call_param_judge_llm: INVOKE param={parameter_name!r} "
        f"value={parameter_value!r} tool={tool_name!r} "
        f"prompt_len={len(prompt)}"
    )
    t0 = time.time()
    result = judge.invoke(prompt)
    elapsed = time.time() - t0
    reasoning_preview = str(result.get("reasoning", ""))[:120]
    _dbg(
        f"_call_param_judge_llm: RETURN param={parameter_name!r} "
        f"verdict={result.get('verdict')!r} elapsed={elapsed:.2f}s "
        f"reasoning_preview={reasoning_preview!r}"
    )
    return result


def _call_extraction_judge_llm(
    *,
    overall_goal: str,
    source_description: str,
    source_tool: str,
    source_args_json: str,
    source_text: str,
    extracted_value: Any,
    extraction_rationale: str,
    judge,
) -> Dict[str, Any]:
    overall_goal_text = overall_goal if overall_goal.strip() else "(not specified)"
    prompt = f"""
You are auditing an extraction step by a scientific agent.

Overall study goal:
{overall_goal_text}

Source artifact (where the extraction came from):
  producing tool: {source_tool}
  description:    {source_description}
  args:           {source_args_json}

Source content:
\"\"\"
{source_text}
\"\"\"

The agent extracted: {repr(extracted_value)}
Stated purpose of the extraction: {extraction_rationale}

Decide TWO things:

1. VALUE CORRECTNESS
   - Is the extracted value actually present in the source?
   - For numeric values: is it a COMPLETE number, not a substring of a longer
     one (e.g. '123' pulled from '123456789')?
   - For text values: is it a clean, verbatim span of the source that matches
     the stated purpose, not a fragment of a different statement?
   - Does it semantically correspond to what the agent claims to be extracting?

2. SOURCE APPROPRIATENESS
   - Is this source the right place to extract from, given the stated purpose?
   - If the agent is recording a FINAL result, the source MUST NOT be a test
     run, scratch file, debug output, draft, stale or earlier version, or the
     output of a different calculation.
   - If the agent is recording an INTERMEDIATE value, the source must still
     match that stage (e.g. a converged run, not a non-converged one).

Return:
- pass     : both the value is correct AND the source is appropriate
- fail     : the value is wrong OR the source is inappropriate for the purpose
- warning  : ambiguous
"""
    _dbg(
        f"_call_extraction_judge_llm: INVOKE extracted={extracted_value!r} "
        f"source_tool={source_tool!r} prompt_len={len(prompt)}"
    )
    t0 = time.time()
    result = judge.invoke(prompt)
    elapsed = time.time() - t0
    reasoning_preview = str(result.get("reasoning", ""))[:120]
    _dbg(
        f"_call_extraction_judge_llm: RETURN extracted={extracted_value!r} "
        f"verdict={result.get('verdict')!r} elapsed={elapsed:.2f}s "
        f"reasoning_preview={reasoning_preview!r}"
    )
    return result


# =========================================================
# Tool 1 — Structured report generation
# =========================================================
@tool
def generate_structured_report(
    report_name: Annotated[str, "Name for this report."],
    overall_goal: Annotated[str, "Overall goal of the study. Required and non-empty."],
    quantity_specs: Annotated[
        List[QuantitySpec],
        "List of (value, result_id) claims plus presentation metadata.",
    ],
    qualitative_findings: Annotated[str, "Prose findings."] = "",
    conclusion: Annotated[str, "Prose conclusion."] = "",
    strict: Annotated[bool, "Strict mode for orphan-number detection."] = True,
) -> Dict[str, Any]:
    """
    Build a structured scientific report from numerical claims, each backed
    by a registered artifact. 

    Always at the end of a study, before declaring the work done.
    Optionally mid-project, to lock down an intermediate finding before
    moving on — a mid-project report is verified just like a final one and
    must stand on its own. 

    YOU ARE RESPONSIBLE FOR EVERY NUMBER YOU CLAIM.
      * Every QuantitySpec must point at a real `result_id`. If you want
        a number you don't have an artifact for, run the tool that produces
        it first.
      * Do not paraphrase numbers from memory or from raw tool output. If
        a tool printed a value but it isn't yet a registered artifact, use
        the extraction tools to register it, then cite the new artifact.
      * Never do math/calculations yourself. Use the math tool, which produces 
        artifacts you can cite.
      * In prose: numbers that are measurements must be cited via
        `{claim:<quantity_name>}`. Non-measurement numbers (run counts,
        indices, dates, figure refs) must be wrapped as `{lit:<number>}`.
      * `overall_goal` is required and non-empty. It is threaded into every
        per-parameter check; a vague goal weakens the whole verification.

    The function raises (not silently degrades) on: empty overall_goal,
    duplicate quantity_name, value/artifact mismatch, undeclared citations,
    or — when strict=True — un-cited numbers in prose. Treat each as a real
    failure to fix at the source, not to work around.
    """
    _dbg(
        f"generate_structured_report: ENTER report_name={report_name!r} "
        f"overall_goal: {overall_goal} n_specs={len(quantity_specs)} "
        f"strict={strict}"
    )

    if report_name in var.All_Report_Names:
        _dbg(
            f"generate_structured_report: name collision — existing names: "
            f"{sorted(var.All_Report_Names)}"
        )
        raise ValueError(f"Report name '{report_name}' already exists. "
                         "Choose a unique name for each report.")

    if not overall_goal or not overall_goal.strip():
        _dbg("generate_structured_report: overall_goal empty — about to raise")
        raise ValueError(
            "overall_goal is required and must be non-empty. It describes the "
            "study's purpose and is used by the verifier to provide context "
            "to per-parameter judgments throughout the call chain."
        )

    parsed_specs = [QuantitySpec.model_validate(x) for x in quantity_specs]
    _dbg(f"generate_structured_report: parsed {len(parsed_specs)} QuantitySpec(s)")

    names = [s.quantity_name for s in parsed_specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        _dbg(f"generate_structured_report: duplicate quantity_names found: {dupes}")
        raise ValueError(
            f"Duplicate quantity_name(s): {dupes}. Each must be unique — "
            "it is the citation handle."
        )
    _dbg("generate_structured_report: duplicate-name check passed")

    quantities_sought: List[Dict[str, Any]] = []
    numerical_results: List[ReportNumericalClaim] = []

    for spec in parsed_specs:
        _dbg(
            f"generate_structured_report: verifying spec quantity={spec.quantity_name!r} "
            f"value={spec.value!r} result_id={spec.result_id!r}"
        )
        _get_artifact(spec.result_id)
        _dbg(
            f"generate_structured_report: calling CANVAS.verify_artifact "
            f"value={spec.value!r} result_id={spec.result_id!r}"
        )
        ok, msg = CANVAS.verify_artifact(spec.value, spec.result_id)
        _dbg(
            f"generate_structured_report: verify_artifact -> ok={ok} msg={msg!r}"
        )
        if not ok:
            raise ValueError(
                f"Quantity '{spec.quantity_name}': claimed value {spec.value!r} "
                f"is not backed by artifact '{spec.result_id}'. {msg}"
            )
        quantities_sought.append({
            "quantity_name": spec.quantity_name,
            "varied_parameters": spec.varied_parameters,
            "result_id": spec.result_id,
            "unit": spec.unit,
            "note": spec.note,
        })
        numerical_results.append(ReportNumericalClaim(
            quantity_name=spec.quantity_name,
            value=spec.value,
            unit=spec.unit,
            result_id=spec.result_id,
            varied_parameters=spec.varied_parameters,
            note=spec.note,
        ))

    by_name = {c.quantity_name: c for c in numerical_results}
    _dbg(
        f"generate_structured_report: all specs verified — "
        f"by_name keys={list(by_name.keys())}"
    )

    for field_name, text in (("qualitative_findings", qualitative_findings),
                             ("conclusion", conclusion)):
        used = set(_CLAIM_TAG.findall(text))
        missing = used - set(by_name)
        _dbg(
            f"generate_structured_report: tag-check field={field_name!r} "
            f"used_tags={sorted(used)} missing={sorted(missing)}"
        )
        if missing:
            raise ValueError(
                f"{field_name} references undeclared quantity_name(s): "
                f"{sorted(missing)}. Valid: {sorted(by_name)}."
            )

    if strict:
        for field_name, text in (("qualitative_findings", qualitative_findings),
                                 ("conclusion", conclusion)):
            stripped = _CLAIM_TAG.sub("", text)
            stripped = _LIT_TAG.sub("", stripped)
            orphans = _NUMBER_RE.findall(stripped)
            _dbg(
                f"generate_structured_report: orphan-number check field={field_name!r} "
                f"n_orphans={len(orphans)} orphans={orphans}"
            )
            if orphans:
                raise ValueError(
                    f"{field_name} contains un-cited numbers: {orphans}. "
                    "Wrap each measurement in `{claim:<quantity_name>}`, or "
                    "wrap non-measurement numbers (run counts, indices, "
                    "dates, figure refs) in `{lit:<number>}`. Use "
                    "strict=False only as a last resort."
                )

    _dbg("generate_structured_report: rendering markdown")
    rendered = _render_report_markdown(
        overall_goal=overall_goal,
        quantities=numerical_results,
        qualitative_findings=qualitative_findings,
        conclusion=conclusion,
    )
    _dbg(f"generate_structured_report: rendered markdown len={len(rendered)}")
    
    parent_result_ids = list({c.result_id for c in numerical_results})
    _dbg(
        f"generate_structured_report: parent_result_ids fingerprint "
        f"n={len(parent_result_ids)} ids={parent_result_ids}"
    )
    
    id = CANVAS.register_tool_output(
        tool_name="generate_structured_report",
        description=f"Structured scientific report with {len(numerical_results)} numerical claims.",
        value=rendered,
        args={
            "overall_goal": overall_goal,
            "quantity_specs": [s.model_dump() for s in parsed_specs],
            "qualitative_findings": qualitative_findings,
            "conclusion": conclusion,
        },
        reasons={},
        parent_result_ids=parent_result_ids,
        metadata={"report": True},    
    )
    _dbg(f"generate_structured_report: register_tool_output returned id={id!r}")
    
    var.reportName = report_name
    
    tmpReport = StructuredScientificReport(
        overall_goal=overall_goal,
        quantities_sought=quantities_sought,
        numerical_results=numerical_results,
        qualitative_findings=qualitative_findings,
        conclusion=conclusion,
        rendered_markdown=rendered,
    ).model_dump()
    
    CANVAS.canvas[report_name] = tmpReport
    print(tmpReport)
    _dbg(
        f"generate_structured_report: canvas[{report_name!r}] written. "
        f"All canvas keys now: {list(CANVAS.canvas.keys())}"
    )

    _dbg(
        f"generate_structured_report: RETURN n_quantities={len(numerical_results)} "
        f"report_name={report_name!r}"
    )
    return tmpReport


def _render_report_markdown(
    *,
    overall_goal: str,
    quantities: List[ReportNumericalClaim],
    qualitative_findings: str,
    conclusion: str,
) -> str:
    by_name = {c.quantity_name: c for c in quantities}
    _dbg(
        f"_render_report_markdown: ENTER n_quantities={len(quantities)} "
        f"qf_len={len(qualitative_findings)} conclusion_len={len(conclusion)}"
    )

    sub_count = {"claim": 0, "lit": 0}

    def _sub(m: "re.Match[str]") -> str:
        sub_count["claim"] += 1
        c = by_name[m.group(1)]
        val = f"{c.value:g}"
        if c.unit:
            val += f" {c.unit}"
        return f"{val} [^{c.quantity_name}]"

    def _unwrap_lit(text: str) -> str:
        def _u(m: "re.Match[str]") -> str:
            sub_count["lit"] += 1
            return m.group(1)
        return _LIT_TAG.sub(_u, text)

    lines: List[str] = [
        "# Scientific Report", "",
        "## Overall Goal", "", overall_goal, "",
        "## Quantities Sought", "",
    ]
    for c in quantities:
        varied = ", ".join(c.varied_parameters) if c.varied_parameters else "(none)"
        unit = f" {c.unit}" if c.unit else ""
        tail = f"  — {c.note}" if c.note else ""
        lines.append(
            f"- **{c.quantity_name}** = {c.value:g}{unit}  "
            f"(varied: {varied}){tail} [^{c.quantity_name}]"
        )
    lines.append("")

    if qualitative_findings.strip():
        rendered = _unwrap_lit(_CLAIM_TAG.sub(_sub, qualitative_findings))
        lines.extend(["## Qualitative Findings", "", rendered, ""])
    if conclusion.strip():
        rendered = _unwrap_lit(_CLAIM_TAG.sub(_sub, conclusion))
        lines.extend(["## Conclusion", "", rendered, ""])

    lines.extend(["## References", ""])
    for c in quantities:
        note = f" — {c.note}" if c.note else ""
        unit = f" {c.unit}" if c.unit else ""
        lines.append(
            f"[^{c.quantity_name}]: **{c.quantity_name}** = {c.value:g}{unit}, "
            f"source artifact `{c.result_id}`{note}"
        )
    final = "\n".join(lines).rstrip() + "\n"
    _dbg(
        f"_render_report_markdown: RETURN line_count={len(lines)} "
        f"final_char_len={len(final)} claim_subs={sub_count['claim']} "
        f"lit_unwraps={sub_count['lit']}"
    )
    return final


# =========================================================
# Tool 3 — Local per-artifact verification
# =========================================================

def verify_artifact_parameterization(
    target_quantity: Annotated[str, "Quantity currently being verified."],
    overall_goal: Annotated[str, "Overall study goal, threaded into every judge call."],
    varied_parameters: Annotated[List[str], "Parameters intentionally varied for this quantity."],
    sensitive_parameters: Annotated[List[str], "User-specified sensitive parameters."],
    result_id: Annotated[str, "Artifact result_id to verify."],
    judge,
) -> Dict[str, Any]:
    """Verify one artifact locally."""
    _dbg(
        f"verify_artifact_parameterization: ENTER result_id={result_id!r} "
        f"target_quantity={target_quantity!r} "
        f"varied={varied_parameters} sensitive={sensitive_parameters}"
    )
    artifact = _get_artifact(result_id)
    args = artifact.args or {}
    reasons = artifact.reasons or {}
    param_source_ids = _param_source_ids(artifact)
    is_info_tool = artifact.tool_name in INFO_TOOLS
    _dbg(
        f"verify_artifact_parameterization: tool_name={artifact.tool_name!r} "
        f"is_info_tool={is_info_tool} n_args={len(args)} "
        f"n_reasons={len(reasons)} n_param_sources={len(param_source_ids)}"
    )

    checks: List[ParamCheckResult] = []
    overall: str = "pass"

    # R0 — extraction tool special case.
    if artifact.tool_name in EXTRACTION_TOOL_NAMES:
        _dbg(
            f"verify_artifact_parameterization: R0 path — extraction tool "
            f"{artifact.tool_name!r}"
        )
        for c in _verify_extraction_behavior(
            artifact=artifact, args=args,
            param_source_ids=param_source_ids,
            overall_goal=overall_goal, judge=judge,
        ):
            checks.append(c)
            overall = _fold_verdict(overall, c.verdict)

    # Per-parameter checks.
    for param_name, param_value in args.items():
        _dbg(
            f"verify_artifact_parameterization: --- param loop --- "
            f"name={param_name!r} value={param_value!r}"
        )
        # Reason lookup: per-param key first, then the singleton "reasons"
        # fallback used by tools whose `reasons: str` was wrapped as
        # `{"reasons": "..."}` at registration time.
        if param_name in reasons:
            reason_source_key = param_name
        elif "reasons" in reasons:
            reason_source_key = "reasons (fallback)"
        else:
            reason_source_key = "(none — empty reason)"
        reason = reasons.get(param_name, "") or reasons.get("reasons", "")
        _dbg(
            f"verify_artifact_parameterization: reason lookup for "
            f"{param_name!r} -> source_key={reason_source_key!r} "
            f"reason_len={len(reason)}"
        )
        source = param_source_ids.get(param_name)

        # Detect "no source declared" — any of these states means the
        # agent did not point at an upstream artifact for this parameter:
        #   - source is None (param has no entry in param_source_ids)
        #   - source is "" (empty string)
        #   - source is a list with any empty / non-string entry
        is_missing_source = (
            source is None
            or (isinstance(source, str) and not source)
            or (isinstance(source, list) and any(
                not (isinstance(s, str) and s) for s in source
            ))
        )

        # Match either bare `<param>` (applies to every tool) or scoped
        # `<tool>.<param>` (applies only to this tool).
        is_sensitive = (
            param_name in sensitive_parameters
            or f"{artifact.tool_name}.{param_name}" in sensitive_parameters
        )
        is_varied = param_name in varied_parameters
        # Dotted-source detection. When the source contains any
        # '<id>.<field>' entry, the agent is referencing an INPUT of a
        # past tool call rather than the call's output, and we route to
        # the dedicated R1_DOTTED rule.
        is_dotted_source = (not is_missing_source) and _any_dotted(source)
        if is_sensitive and not is_varied and is_missing_source:
            _branch_label = "R1-no-source"
        elif is_sensitive and not is_varied and is_dotted_source:
            _branch_label = "R1-dotted"
        elif is_sensitive and not is_varied:
            _branch_label = "R1"
        elif is_sensitive and is_varied:
            _branch_label = "R2"
        else:
            _branch_label = "R3"
        _dbg(
            f"verify_artifact_parameterization: branch decision for "
            f"{param_name!r}: is_sensitive={is_sensitive} is_varied={is_varied} "
            f"is_missing_source={is_missing_source} "
            f"is_dotted_source={is_dotted_source} "
            f"is_info_tool={is_info_tool} "
            f"has_reason={bool(reason.strip())} -> rule={_branch_label}"
        )

        # R1-no-source — sensitive, not varied, no upstream artifact.
        # Hand to the judge with R1_NO_SOURCE_RULE; the judge reads the
        # `Context:` line in the rationale and decides production-fail
        # vs. sub-study-conditional-pass.
        if is_sensitive and not is_varied and is_missing_source:
            _dbg(
                f"verify_artifact_parameterization: R1-no-source path — "
                f"calling judge for {param_name!r}"
            )
            judgement = _call_param_judge_llm(
                tool_name=artifact.tool_name,
                tool_description=artifact.description or "",
                parameter_name=param_name,
                parameter_value=param_value,
                reason=reason,
                source_artifact_summary=None,
                rule_to_apply=R1_NO_SOURCE_RULE,
                judge=judge,
            )
            _dbg(
                f"verify_artifact_parameterization: judge verdict for "
                f"{param_name!r} -> {judgement['verdict']!r}"
            )
            checks.append(ParamCheckResult(
                parameter_name=param_name, parameter_value=param_value,
                verdict=judgement["verdict"],
                rule_applied=R1_NO_SOURCE_RULE,
                source_result_id=source,
                reasoning=judgement["reasoning"],
                category=(IssueCategory.UNSOURCED_SENSITIVE
                          if judgement["verdict"] != "pass" else None),
            ))
            overall = _fold_verdict(overall, judgement["verdict"])
            continue

        # R1 — sensitive, not varied, source IS present. Two flavors:
        #   * Dotted source: agent is referencing an INPUT of an earlier
        #     tool call. Use R1_DOTTED_RULE; on judge fail the category
        #     is DOTTED_SOURCE_INAPPROPRIATE.
        #   * Bare source: agent is referencing the OUTPUT of an earlier
        #     tool call. Use R1_RULE; on judge fail the category is
        #     CROSS_WIRED_SOURCE.
        # In both cases value-match against the source happens
        # deterministically inside `_verify_sourced_param` (the CANVAS
        # layer handles dotted refs in `verify_artifact`).
        if is_sensitive and not is_varied:
            if is_dotted_source:
                _dbg(
                    f"verify_artifact_parameterization: R1-dotted path — "
                    f"calling _verify_sourced_param for {param_name!r}"
                )
                rule_for_r1 = R1_DOTTED_RULE
                fail_category_for_r1 = IssueCategory.DOTTED_SOURCE_INAPPROPRIATE
            else:
                _dbg(
                    f"verify_artifact_parameterization: R1 path — calling "
                    f"_verify_sourced_param for {param_name!r}"
                )
                rule_for_r1 = R1_RULE
                fail_category_for_r1 = IssueCategory.CROSS_WIRED_SOURCE
            checks_to_add, branch_verdict = _verify_sourced_param(
                param_name=param_name, param_value=param_value,
                source=source, rule=rule_for_r1,
                tool_name=artifact.tool_name,
                tool_description=artifact.description or "",
                reason=reason, judge=judge,
                fail_category=fail_category_for_r1,
            )
            checks.extend(checks_to_add)
            overall = _fold_verdict(overall, branch_verdict)
            continue

        # R2 — sensitive and varied.
        if is_sensitive and is_varied:
            rule = R2_RULE
        # R3 — non-sensitive.
        else:
            rule = R3_RULE

        # R3 lenient path for information-only tools with empty reasons.
        if (not is_sensitive) and is_info_tool and not reason.strip():
            _dbg(
                f"verify_artifact_parameterization: R3 lenient info-tool path "
                f"for {param_name!r} on tool {artifact.tool_name!r} -> info"
            )
            checks.append(ParamCheckResult(
                parameter_name=param_name, parameter_value=param_value,
                verdict="info", rule_applied=rule,
                source_result_id=source,
                reasoning=(
                    f"Tool '{artifact.tool_name}' is an information-gathering "
                    "tool; per-parameter rationale is not required."
                ),
            ))
            continue

        # Belt-and-braces source check. If a source is declared (non-empty
        # ref or non-empty list of refs), verify it element-by-element
        # (list shape) or directly (scalar shape). Empty / missing sources
        # are treated the same as "no source declared" — skip the match,
        # judge with no source summary.
        source_is_real = (
            (isinstance(source, str) and bool(source))
            or (isinstance(source, list) and source and all(
                isinstance(s, str) and s for s in source
            ))
        )
        source_summary: Optional[Any] = None
        if source_is_real:
            _dbg(
                f"verify_artifact_parameterization: real source declared "
                f"for {param_name!r} — running belt-and-braces match"
            )
            checks_to_add, branch_verdict, source_summary = (
                _verify_optional_source_match(
                    param_name=param_name, param_value=param_value,
                    source=source, rule=rule,
                )
            )
            _dbg(
                f"verify_artifact_parameterization: belt-and-braces result "
                f"for {param_name!r}: branch_verdict={branch_verdict!r} "
                f"n_extra_checks={len(checks_to_add)}"
            )
            if checks_to_add:
                checks.extend(checks_to_add)
                if branch_verdict == "fail":
                    overall = _fold_verdict(overall, "fail")
                    continue
                overall = _fold_verdict(overall, branch_verdict)

        # LLM judgment.
        _dbg(
            f"verify_artifact_parameterization: invoking judge for "
            f"{param_name!r} under rule={'R2' if rule is R2_RULE else 'R3'}"
        )
        judgement = _call_param_judge_llm(
            tool_name=artifact.tool_name,
            tool_description=artifact.description or "",
            parameter_name=param_name,
            parameter_value=param_value,
            reason=reason,
            source_artifact_summary=source_summary,
            rule_to_apply=rule,
            judge=judge,
        )
        _dbg(
            f"verify_artifact_parameterization: judge verdict for "
            f"{param_name!r} -> {judgement['verdict']!r}"
        )
        if judgement["verdict"] == "pass":
            check_category: Optional[str] = None
        elif rule is R2_RULE:
            check_category = IssueCategory.UNDER_JUSTIFIED_SWEEP
        else:  # rule is R3_RULE
            check_category = IssueCategory.UNDER_JUSTIFIED_CHOICE
        checks.append(ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict=judgement["verdict"], rule_applied=rule,
            source_result_id=source,
            reasoning=judgement["reasoning"],
            category=check_category,
        ))
        overall = _fold_verdict(overall, judgement["verdict"])

    _dbg(
        f"verify_artifact_parameterization: DONE result_id={result_id!r} "
        f"overall={overall!r} n_checks={len(checks)}"
    )

    return ArtifactVerificationResult(
        result_id=result_id,
        tool_name=artifact.tool_name,
        artifact_description=artifact.description,
        overall_verdict=overall,
        summary=(
            f"Local verification for artifact {result_id} → '{overall}' "
            f"(in scope of quantity '{target_quantity}')."
        ),
        checks=checks,
        recursive_children_checked=[],
        verified_for_quantity=target_quantity,
    ).model_dump()


def _verify_sourced_param(
    *,
    param_name: str,
    param_value: Any,
    source: Any,
    rule: str,
    tool_name: str,
    tool_description: str,
    reason: str,
    judge,
    fail_category: str = IssueCategory.CROSS_WIRED_SOURCE,
) -> Tuple[List[ParamCheckResult], str]:
    """R1 path: must be sourced. Element-by-element value match, then a
    single LLM judge call evaluating the source(s) collectively.

    Source summaries are built via `_build_source_summary`, which
    dispatches between the bare-ref shape (the source artifact's full
    summary) and the dotted-ref shape (the specific input parameter's
    value + rationale on its parent artifact). The judge sees the
    appropriate shape and applies the rule accordingly.

    `fail_category` selects the IssueCategory tag attached to a judge-
    failure verdict. Bare R1 -> CROSS_WIRED_SOURCE; dotted R1 ->
    DOTTED_SOURCE_INAPPROPRIATE. Value-mismatch failures always tag
    VALUE_MISMATCH_PARAM regardless.
    """
    _dbg(
        f"_verify_sourced_param: ENTER param={param_name!r} "
        f"value={param_value!r} source={source!r} "
        f"fail_category={fail_category!r}"
    )
    is_list, values, sources, err = _normalize_to_list_pair(
        param_value, source, param_name
    )
    if err is not None:
        _dbg(f"_verify_sourced_param: shape error -> FAIL ({err!r})")
        return [ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict="fail", rule_applied=rule,
            source_result_id=source, reasoning=err,
            category=IssueCategory.UNCATEGORIZED,
        )], "fail"

    # Value-match each element. CANVAS.verify_artifact handles both
    # bare and dotted refs transparently (dotted refs are validated
    # against the upstream artifact's args field).
    for i, (v, ref) in enumerate(zip(values, sources)):
        _dbg(
            f"_verify_sourced_param: value-match element idx={i} "
            f"value={v!r} ref={ref!r}"
        )
        ok, msg = CANVAS.verify_artifact(v, ref)
        _dbg(
            f"_verify_sourced_param: verify_artifact -> ok={ok} msg={msg!r}"
        )
        if not ok:
            label = f"{param_name}[{i}]" if is_list else param_name
            _dbg(f"_verify_sourced_param: element {label} FAILED value-match")
            return [ParamCheckResult(
                parameter_name=label, parameter_value=v,
                verdict="fail", rule_applied=VALUE_MATCH_SENTINEL,
                source_result_id=ref, reasoning=msg,
                category=IssueCategory.VALUE_MISMATCH_PARAM,
            )], "fail"

    # Single judge call over the collective source set. Summaries are
    # heterogeneous-shape-tolerant: a list of refs can mix bare and
    # dotted entries, and each is summarized independently.
    if is_list:
        source_summary: Any = [_build_source_summary(ref) for ref in sources]
        _dbg(
            f"_verify_sourced_param: built list source_summary "
            f"n_entries={len(source_summary)}"
        )
    else:
        source_summary = _build_source_summary(sources[0])
        _dbg("_verify_sourced_param: built scalar source_summary")

    judgement = _call_param_judge_llm(
        tool_name=tool_name,
        tool_description=tool_description,
        parameter_name=param_name,
        parameter_value=param_value,
        reason=reason,
        source_artifact_summary=source_summary,
        rule_to_apply=rule,
        judge=judge,
    )
    _dbg(
        f"_verify_sourced_param: RETURN param={param_name!r} "
        f"verdict={judgement['verdict']!r}"
    )
    return [ParamCheckResult(
        parameter_name=param_name, parameter_value=param_value,
        verdict=judgement["verdict"], rule_applied=rule,
        source_result_id=source,
        reasoning=judgement["reasoning"],
        category=(fail_category
                  if judgement["verdict"] != "pass" else None),
    )], judgement["verdict"]


def _verify_optional_source_match(
    *,
    param_name: str,
    param_value: Any,
    source: Any,
    rule: str,
) -> Tuple[List[ParamCheckResult], str, Optional[Any]]:
    """R2/R3 belt-and-braces. Returns checks (failures only), branch verdict,
    and the source summary to pass to the judge if value-match passes.

    The caller is responsible for not invoking this helper when `source` is
    None / empty / a list with empty entries — those cases mean "no source
    declared" and should be handled upstream (R1-no-source for sensitive
    params; skip belt-and-braces for R2/R3).
    """
    _dbg(
        f"_verify_optional_source_match: ENTER param={param_name!r} "
        f"value={param_value!r} source={source!r}"
    )
    is_list, values, sources, err = _normalize_to_list_pair(
        param_value, source, param_name
    )
    if err is not None:
        _dbg(f"_verify_optional_source_match: shape error -> FAIL ({err!r})")
        return [ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict="fail", rule_applied=rule,
            source_result_id=source, reasoning=err,
            category=IssueCategory.UNCATEGORIZED,
        )], "fail", None

    for i, (v, ref) in enumerate(zip(values, sources)):
        _dbg(
            f"_verify_optional_source_match: value-match element idx={i} "
            f"value={v!r} ref={ref!r}"
        )
        ok, msg = CANVAS.verify_artifact(v, ref)
        _dbg(
            f"_verify_optional_source_match: verify_artifact -> ok={ok} "
            f"msg={msg!r}"
        )
        if not ok:
            label = f"{param_name}[{i}]" if is_list else param_name
            _dbg(
                f"_verify_optional_source_match: element {label} FAILED "
                "value-match"
            )
            return [ParamCheckResult(
                parameter_name=label, parameter_value=v,
                verdict="fail", rule_applied=VALUE_MATCH_SENTINEL,
                source_result_id=ref, reasoning=msg,
                category=IssueCategory.VALUE_MISMATCH_PARAM,
            )], "fail", None

    if is_list:
        summary: Any = [_build_source_summary(ref) for ref in sources]
        _dbg(
            f"_verify_optional_source_match: PASS list-shape "
            f"n_entries={len(summary)}"
        )
        return [], "pass", summary
    summary = _build_source_summary(sources[0])
    _dbg("_verify_optional_source_match: PASS scalar-shape")
    return [], "pass", summary


def _verify_extraction_behavior(
    *,
    artifact: Any,
    args: Dict[str, Any],
    param_source_ids: Dict[str, Any],
    overall_goal: str,
    judge,
) -> List[ParamCheckResult]:
    """Verify an extraction artifact (numeric, text, or list-valued).

    The canonical source key after the extract-tool migration is
    `param_source_ids["source_tool_call_id"]`. We check that first, then
    fall back to other recovery paths for hypothetical future extraction
    tools.
    """
    _dbg(
        f"_verify_extraction_behavior: ENTER result_id={artifact.result_id!r} "
        f"tool={artifact.tool_name!r}"
    )
    extraction_rationale = artifact.description

    source_text: Optional[str] = (
        args.get("source_text") or args.get("text") or args.get("content")
    )
    _dbg(
        f"_verify_extraction_behavior: source_text from args? "
        f"{source_text is not None}"
    )

    source_id_for_ref: Optional[str] = None
    canonical = param_source_ids.get("source_tool_call_id")
    if isinstance(canonical, str) and canonical:
        source_id_for_ref = canonical
        _dbg(
            f"_verify_extraction_behavior: source recovery — canonical key "
            f"hit -> {source_id_for_ref!r}"
        )
    if source_id_for_ref is None:
        for candidate in ("source", "source_file", "file", "filename",
                          "file_path", "input_file", "path"):
            v = param_source_ids.get(candidate)
            _dbg(
                f"_verify_extraction_behavior: source recovery — fallback try "
                f"{candidate!r} -> {v!r}"
            )
            if isinstance(v, str) and v:
                source_id_for_ref = v
                _dbg(
                    f"_verify_extraction_behavior: source recovery — fallback "
                    f"hit on {candidate!r} -> {source_id_for_ref!r}"
                )
                break
    if source_id_for_ref is None and len(param_source_ids) == 1:
        only = next(iter(param_source_ids.values()))
        if isinstance(only, str):
            source_id_for_ref = only
            _dbg(
                f"_verify_extraction_behavior: source recovery — single-entry "
                f"fallback -> {source_id_for_ref!r}"
            )

    source_description = ""
    source_tool = ""
    source_args_json = "{}"

    if source_id_for_ref:
        try:
            src = _get_artifact(source_id_for_ref)
            if not source_text:
                source_text = (
                    src.metadata.get("text")
                    or getattr(src, "text", None)
                    or str(_flatten_listed_value(src))
                )
            source_description = src.description
            source_tool = src.tool_name
            source_args_json = json.dumps(src.args, indent=2, default=str)
            _dbg(
                f"_verify_extraction_behavior: parent artifact loaded — "
                f"tool={source_tool!r} src_text_len="
                f"{len(source_text) if source_text else 0}"
            )
        except Exception as e:
            _dbg(
                f"_verify_extraction_behavior: parent artifact load FAILED — "
                f"{type(e).__name__}: {e}"
            )
            pass

    if not source_text:
        _dbg(
            "_verify_extraction_behavior: no source text recoverable — "
            "emitting WARNING"
        )
        return [ParamCheckResult(
            parameter_name="<extracted_value>",
            parameter_value=_flatten_listed_value(artifact),
            verdict="warning",
            rule_applied="Extraction must have a recoverable source.",
            source_result_id=source_id_for_ref,
            reasoning=(
                "No source text recoverable from args or a parent artifact; "
                "cannot verify extraction syntactically or semantically."
            ),
            category=IssueCategory.EXTRACTION_NO_SOURCE,
        )]

    if _is_listed(artifact):
        elements: List[Tuple[str, Any]] = [
            (f"<extracted_value[{i}]>", getattr(item, "value", item))
            for i, item in enumerate(artifact.value)
        ]
    else:
        elements = [("<extracted_value>", artifact.value)]
    _dbg(
        f"_verify_extraction_behavior: element count = {len(elements)} "
        f"(listed={_is_listed(artifact)})"
    )

    out: List[ParamCheckResult] = []
    source_text_str = str(source_text)

    for label, value in elements:
        _dbg(
            f"_verify_extraction_behavior: checking element {label} "
            f"value={value!r}"
        )
        if _looks_numeric(value):
            if not _is_complete_numeric_token(str(value), source_text_str):
                _dbg(
                    f"_verify_extraction_behavior: numeric token check "
                    f"FAILED for {label}"
                )
                out.append(ParamCheckResult(
                    parameter_name=label, parameter_value=value,
                    verdict="fail",
                    rule_applied=(
                        "Numeric extractions must appear as a complete "
                        "numeric token in the source."
                    ),
                    source_result_id=source_id_for_ref,
                    reasoning=(
                        f"Extracted value {value!r} does not appear as a "
                        "complete numeric token in the source — possible "
                        "substring extraction (e.g. '123' from '123456789')."
                    ),
                    category=IssueCategory.EXTRACTION_TOKEN_MISMATCH,
                ))
                continue
        else:
            if not _is_complete_text_token(str(value), source_text_str):
                _dbg(
                    f"_verify_extraction_behavior: text token check "
                    f"FAILED for {label}"
                )
                out.append(ParamCheckResult(
                    parameter_name=label, parameter_value=value,
                    verdict="fail",
                    rule_applied=(
                        "Text extractions must appear verbatim in the source."
                    ),
                    source_result_id=source_id_for_ref,
                    reasoning=(
                        f"Extracted text {value!r} does not appear in the "
                        "source. Extraction must be verbatim."
                    ),
                    category=IssueCategory.EXTRACTION_TOKEN_MISMATCH,
                ))
                continue

        _dbg(
            f"_verify_extraction_behavior: token check PASSED for {label} — "
            "invoking extraction judge"
        )
        judgement = _call_extraction_judge_llm(
            overall_goal=overall_goal,
            source_description=source_description,
            source_tool=source_tool,
            source_args_json=source_args_json,
            source_text=source_text_str,
            extracted_value=value,
            extraction_rationale=extraction_rationale,
            judge=judge,
        )
        _dbg(
            f"_verify_extraction_behavior: judge verdict for {label} -> "
            f"{judgement['verdict']!r}"
        )
        out.append(ParamCheckResult(
            parameter_name=label, parameter_value=value,
            verdict=judgement["verdict"],
            rule_applied=(
                "Extracted value must semantically match its stated purpose, "
                "and the source must be the correct file/run/artifact for "
                "that purpose (not a test/draft/stale version)."
            ),
            source_result_id=source_id_for_ref,
            reasoning=judgement["reasoning"],
            category=(IssueCategory.EXTRACTION_JUDGE_FAIL
                      if judgement["verdict"] != "pass" else None),
        ))

    _dbg(
        f"_verify_extraction_behavior: RETURN n_results={len(out)} "
        f"verdicts={[c.verdict for c in out]}"
    )
    return out


# =========================================================
# Tool 2 — Recursive structured report verification
# =========================================================



def _verify_one_artifact_recursive(
    *,
    result_id: str,
    target_quantity: str,
    overall_goal: str,
    varied_parameters: List[str],
    sensitive_parameters: List[str],
    visited: Set[Tuple[str, frozenset]],
    artifact_results: List[ArtifactVerificationResult],
    issues: List[ReportVerificationIssue],
    judge,
    depth: int = 0,
    viz
) -> None:
    indent = "  " * depth
    _dbg(
        f"{indent}_verify_one_artifact_recursive: ENTER depth={depth} "
        f"result_id={result_id!r} target_quantity={target_quantity!r}"
    )
    # Cache key: (result_id, frozenset of varied parameters). Two claims
    # with the SAME varied_parameters share cache entries — the second
    # claim's walk skips artifacts the first already verified. Two
    # claims with DIFFERENT varied_parameters cache separately, since
    # R1 vs R2 branch selection (and the resulting verdicts) depend on
    # the varied set.
    visit_key = (result_id, frozenset(varied_parameters))
    if visit_key in visited:
        _dbg(
            f"{indent}_verify_one_artifact_recursive: already visited "
            f"{result_id!r} under varied={sorted(varied_parameters)} — "
            f"skipping (cache hit)"
        )
        return
    visited.add(visit_key)

    artifact = _get_artifact(result_id)
    viz.begin_artifact(result_id=result_id, target_quantity=target_quantity,
                       depth=depth, artifact=artifact)

    # Post-order traversal: verify children first, then the current node.
    # The output `artifact_results` reads bottom-up, so a reader scrolling
    # through it sees the leaf verdicts before the parent that depends on
    # them — matching how a human auditor would walk the chain.
    children = _collect_recursive_source_ids(artifact)
    children_total = len(children)
    children_failed = 0
    _dbg(
        f"{indent}_verify_one_artifact_recursive: result_id={result_id!r} "
        f"has {children_total} children: {children}"
    )
    children_checked: List[str] = []
    for child_id in children:
        viz.begin_child_descent(parent_id=result_id, child_id=child_id)
        _dbg(
            f"{indent}_verify_one_artifact_recursive: descending into child "
            f"{child_id!r} (parent={result_id!r})"
        )
        try:
            _verify_one_artifact_recursive(
                result_id=child_id,
                target_quantity=target_quantity,
                overall_goal=overall_goal,
                varied_parameters=varied_parameters,
                sensitive_parameters=sensitive_parameters,
                visited=visited,
                artifact_results=artifact_results,
                issues=issues,
                judge=judge,
                depth=depth + 1,
                viz=viz
            )
            children_checked.append(child_id)
            viz.end_child_descent(parent_id=result_id, child_id=child_id)
            _dbg(
                f"{indent}_verify_one_artifact_recursive: returned from child "
                f"{child_id!r}"
            )
        except Exception as e:                        # noqa: BLE001
            children_failed += 1
            viz.end_child_descent(parent_id=result_id, child_id=child_id,
                                  error=str(e))
            _dbg(
                f"{indent}_verify_one_artifact_recursive: child {child_id!r} "
                f"raised {type(e).__name__}: {e}"
            )
            issues.append(ReportVerificationIssue(
                category=IssueCategory.RECURSION_FAILURE,
                severity="fail",
                where={"result_id": result_id, "child_id": child_id},
                problem=(
                    f"Recursive verification of child artifact "
                    f"'{child_id}' (a parent of '{result_id}') raised an "
                    f"exception during descent."
                ),
                judge_reasoning=str(e),
            ))

    _dbg(
        f"{indent}_verify_one_artifact_recursive: post-order summary for "
        f"{result_id!r} — children_total={children_total} "
        f"children_checked={len(children_checked)} "
        f"children_failed={children_failed}"
    )

    # Dotted refs are NOT recursed into. The dotted-source judge call
    # at the immediate downstream parameter produces the verdict for
    # the cross-reference (R1_DOTTED), and the walker stops there. We
    # deliberately do not descend into the source artifact's siblings
    # or chase the named field's deeper chain — the agent's claim is
    # narrow to the one referenced value, so the verifier's findings
    # are scoped to match.


    local_dict = verify_artifact_parameterization(
        target_quantity=target_quantity,
        overall_goal=overall_goal,
        varied_parameters=varied_parameters,
        sensitive_parameters=sensitive_parameters,
        result_id=result_id,
        judge=judge,
    )
    local_result = ArtifactVerificationResult.model_validate(local_dict)
    local_result.recursive_children_checked = children_checked
    artifact_results.append(local_result)
    _dbg(
        f"{indent}_verify_one_artifact_recursive: appended local result for "
        f"{result_id!r} verdict={local_result.overall_verdict!r}"
    )

    if local_result.overall_verdict != "pass":
        _dbg(
            f"{indent}_verify_one_artifact_recursive: emitting per-check "
            f"issues for {result_id!r} "
            f"verdict={local_result.overall_verdict!r}"
        )

        # Look up the artifact once to extract per-parameter `Context:`
        # lines from its rationale. The merged context is the segment of
        # `reasons[param]` (or `reasons["reasons"]` for str-shape tools)
        # between "Context:" and "\n\nRationale:".
        try:
            _artifact_for_ctx = _get_artifact(result_id)
            _all_reasons: Dict[str, str] = _artifact_for_ctx.reasons or {}
            _tool_name_for_issue = _artifact_for_ctx.tool_name
        except Exception:                                # noqa: BLE001
            _all_reasons = {}
            _tool_name_for_issue = local_result.tool_name

        for check in local_result.checks:
            if check.verdict == "pass" or check.verdict == "info":
                continue

            cat = check.category or IssueCategory.UNCATEGORIZED

            # Extract the Context: line from this parameter's rationale.
            _reason_text = (
                _all_reasons.get(check.parameter_name)
                or _all_reasons.get("reasons")
                or ""
            )
            ctx_at_site: Optional[str] = None
            if "Context:" in _reason_text:
                _ctx_after = _reason_text.split("Context:", 1)[1]
                # Strip at the next "Rationale:" delimiter or newline-newline.
                for _delim in ("\n\nRationale:", "\nRationale:", "\n\n"):
                    if _delim in _ctx_after:
                        _ctx_after = _ctx_after.split(_delim, 1)[0]
                        break
                ctx_at_site = _ctx_after.strip() or None

            problem = _problem_text_for_param_check(
                category=cat,
                parameter_name=check.parameter_name,
                parameter_value=check.parameter_value,
                tool_name=_tool_name_for_issue,
                verdict=check.verdict,
            )

            issues.append(ReportVerificationIssue(
                category=cat,
                severity=("fail" if check.verdict == "fail" else "warning"),
                where={
                    "result_id": result_id,
                    "tool_name": _tool_name_for_issue,
                    "parameter": check.parameter_name,
                    "parameter_value": check.parameter_value,
                },
                context_at_site=ctx_at_site,
                problem=problem,
                judge_reasoning=check.reasoning,
            ))

    viz.end_artifact(result_id=result_id, verification_result=local_result)


def summarize_verification_for_supervisor(
    raw_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Reorganize the raw verifier output into a numbered, categorized list
    of issues suitable for the supervisor agent to act on.

    Input
    -----
    `raw_result` is the dict produced by `ReportVerificationResult.model_dump()`
    — i.e. has keys: overall_verdict, checked_result_ids, issues,
    artifact_results.

    Output
    ------
    {
        "overall_verdict": str,
        "n_fails": int,
        "n_warnings": int,
        "summary": str,                 # one-liner for the supervisor
        "issues": List[Dict],           # numbered, with remediation_options filled
        "checked_result_ids": List[str],
        "artifact_results": List[Dict], # kept for diagnostic tooling
    }

    The numbered `issues` list is the supervisor's primary surface. Each
    entry includes:
        issue_number:          1-indexed
        category:              IssueCategory.* string
        severity:              "fail" | "warning"
        where:                 structured dict (claim_name / result_id /
                               tool_name / parameter / parameter_value / etc.)
        context_at_site:       the merged Context: line if extractable
        problem:               one-sentence problem description
        judge_reasoning:       judge verdict text where applicable
        remediation_options:   list of fix-paths from REMEDIATION_OPTIONS
    """
    raw_issues: List[Dict[str, Any]] = list(raw_result.get("issues") or [])
    out_issues: List[Dict[str, Any]] = []

    n_fails = 0
    n_warnings = 0

    for i, issue in enumerate(raw_issues, start=1):
        cat = issue.get("category") or IssueCategory.UNCATEGORIZED
        severity = issue.get("severity", "fail")
        if severity == "fail":
            n_fails += 1
        elif severity == "warning":
            n_warnings += 1

        remediation = list(REMEDIATION_OPTIONS.get(
            cat, REMEDIATION_OPTIONS[IssueCategory.UNCATEGORIZED]
        ))

        enriched = dict(issue)
        enriched["issue_number"] = i
        enriched["remediation_options"] = remediation
        out_issues.append(enriched)

    overall = raw_result.get("overall_verdict", "pass")
    n_total = n_fails + n_warnings
    if n_total == 0:
        summary = "All checks passed."
    else:
        # Count distinct claims and artifacts touched by issues.
        claims_with_issues = {
            i["where"].get("claim_name")
            for i in out_issues
            if i.get("where", {}).get("claim_name")
        }
        artifacts_with_issues = {
            i["where"].get("result_id")
            for i in out_issues
            if i.get("where", {}).get("result_id")
        }
        bits = [f"{n_fails} fail" + ("s" if n_fails != 1 else "")]
        if n_warnings:
            bits.append(f"{n_warnings} warning" + ("s" if n_warnings != 1 else ""))
        loc_bits = []
        if claims_with_issues:
            loc_bits.append(
                f"{len(claims_with_issues)} claim"
                + ("s" if len(claims_with_issues) != 1 else "")
            )
        if artifacts_with_issues:
            loc_bits.append(
                f"{len(artifacts_with_issues)} artifact"
                + ("s" if len(artifacts_with_issues) != 1 else "")
            )
        if loc_bits:
            summary = f"{' and '.join(bits)} across {' / '.join(loc_bits)}."
        else:
            summary = f"{' and '.join(bits)}."

    return {
        "overall_verdict": overall,
        "n_fails": n_fails,
        "n_warnings": n_warnings,
        "summary": summary,
        "issues": out_issues,
        "checked_result_ids": raw_result.get("checked_result_ids", []),
        "artifact_results": raw_result.get("artifact_results", []),
    }


# =========================================================
# Debug tool: investigate a surprising result by walking the
# artifact chain with a user-supplied question.
# =========================================================
#
# Same chain-walking infrastructure as the verifier (parent_result_ids_w_args,
# _get_artifact, _summarize_artifact). Different semantics: instead of
# checking parameters against R1/R2/R3 rules, the per-parameter judge is
# asked whether each parameter could plausibly explain the user's question.
#
# Output mirrors the verifier's supervisor-facing format. The supervisor
# reads the debug output the same way they read verifier output: a
# numbered list of findings with structured `where`, `problem`,
# `judge_reasoning`, and `remediation_options` per finding. Plus a
# synthesis paragraph that names the most likely culprit.

# Categories specific to debug findings.
class DebugCategory(str):
    """Closed set of debug-finding categories. String subclass for the
    same reason as IssueCategory — JSON serializability and LLM
    boundary friendliness."""
    # The parameter's own value is a plausible cause of the surprise
    # (e.g. low ecutwfc on a production run when the user asks why their
    # adsorption energy looks too high).
    PARAMETER_VALUE_SUSPECT = "parameter_value_suspect"
    # The upstream source the value came from might itself be the
    # problem (e.g. kspacing came from a convergence test that didn't
    # converge tightly enough to transfer to the current use).
    SOURCE_SUSPECT = "source_suspect"


# Per-parameter debug rule. Read by the LLM judge for every parameter
# encountered during the debug walk.
DEBUG_PARAM_RULE = """
You are helping the user diagnose a surprising or unexpected result by
investigating one specific parameter on one specific tool call along the
chain that produced the result.

The user's investigation question:
  {user_question}

Investigation history (what the user has already tested and ruled out):
  {investigation_history}

If the parameter under review (or its source) has already been ruled out
by the investigation history, return verdict "not_relevant" with a brief
explanation that this was previously tested. Do NOT re-flag a previously
ruled-out hypothesis.

Otherwise, judge whether this parameter could plausibly contribute to or
explain the user's question:

  - "parameter_value_suspect" : the parameter's chosen value, given its
                                role in this tool call and the broader
                                study, is a plausible cause of the
                                surprise. The user should consider
                                investigating this parameter (likely by
                                re-running with a different value to see
                                if the result changes).
  - "source_suspect"          : the parameter's value comes from an
                                upstream artifact, and that upstream
                                artifact is itself a plausible source of
                                the problem (e.g. an upstream convergence
                                that may not have been tight enough for
                                the current use). The user should
                                investigate the upstream artifact rather
                                than this parameter directly.
  - "neutral"                 : the parameter is set in a way consistent
                                with normal practice and unlikely to
                                explain the surprise, OR you don't have
                                enough information to tell.
  - "not_relevant"            : the parameter has no plausible connection
                                to the user's question, or has already
                                been ruled out by the investigation
                                history.

Be honest. If you don't know, say "neutral" with a one-sentence
explanation of why you can't tell. Do not invent suspicion to look
helpful — false leads waste the user's calculation time.
"""


# Synthesis rule. One call after the walk completes, given all per-parameter
# findings. Produces a one-paragraph human-readable summary that the
# supervisor reads alongside the numbered findings list.
DEBUG_SYNTHESIS_RULE = """
The user is investigating a surprising or unexpected result.

  Question: {user_question}

  Investigation history (already ruled out): {investigation_history}

You have walked the value-flow chain of the artifact in question. For
each parameter on each artifact in the chain, a per-parameter judge
produced one of four verdicts: "parameter_value_suspect",
"source_suspect", "neutral", or "not_relevant".

Below is the complete walk:

{walk_dump}

Your job: synthesize a one-paragraph diagnostic summary for the user.

  - If one or more parameters were flagged as suspects, name them in
    PRIORITY ORDER (most likely cause first). Briefly explain why each
    is suspicious.
  - If multiple suspects share a single root cause (e.g. several
    parameters all depend on the same un-converged sub-study), name the
    root cause once and group the dependents under it.
  - If NO parameters were flagged as suspect — which can happen when
    the investigation history has already ruled everything out, or when
    the chain looks clean — say so explicitly. In that case, suggest
    the surprise may originate OUTSIDE the value-flow chain: in the
    original input data (geometry, pseudopotentials), in a physical
    assumption (functional choice, smearing scheme), in an external
    benchmark, or in the user's expectation. Do not invent suspicion
    to fill space.
  - DO NOT recommend changing multiple parameters at once. If multiple
    suspects exist, the user must vary them ONE AT A TIME in priority
    order — varying several at once makes attribution impossible.

Begin your response with the disclaimer:
  "These findings are LLM-generated hypotheses about what could explain
  the surprise. Each must be confirmed by re-running the relevant
  calculation with the suggested change; until then, treat them as
  starting points, not conclusions. Vary one parameter at a time in
  priority order."

Then provide the synthesis paragraph.
"""


# Per-category remediation options for debug findings. These describe
# what to investigate next, not how to "fix" something — debug findings
# are hypotheses, not failures.
DEBUG_REMEDIATION_OPTIONS: Dict[str, List[str]] = {
    DebugCategory.PARAMETER_VALUE_SUSPECT: [
        "Re-run the relevant calculation with a different value of this "
        "parameter to test whether it changes the surprising result. If "
        "the result moves substantially, this parameter is the cause. "
        "If not, mark this parameter as ruled out and pass that to the "
        "next debug call via `investigation_history`.",
        "Check whether the parameter's value is consistent with the role "
        "it plays in the study (i.e. read its `Context:` line and "
        "rationale alongside the value).",
    ],
    DebugCategory.SOURCE_SUSPECT: [
        "The upstream source artifact named in `where.source_result_id` "
        "is itself a candidate cause. Re-run this debug tool with that "
        "upstream `result_id` as the new root and re-ask the same "
        "question to walk further back into the chain.",
        "Check whether the upstream source's own characterization was "
        "tight enough for the current production use (e.g. kspacing "
        "converged at low ecutwfc may not transfer to high-ecutwfc "
        "production). If not, the upstream sub-study may need to be "
        "re-run with tighter conditions.",
    ],
}


def _call_debug_param_judge_llm(
    *,
    tool_name: str,
    tool_description: str,
    parameter_name: str,
    parameter_value: Any,
    reason: str,
    source_artifact_summary: Optional[Any],
    user_question: str,
    investigation_history: str,
    judge,
) -> Dict[str, Any]:
    """Per-parameter debug judge call. Returns a dict with keys
    `verdict` (one of "parameter_value_suspect", "source_suspect",
    "neutral", "not_relevant") and `reasoning`."""
    if source_artifact_summary is None:
        source_block = "None (no upstream artifact for this parameter)"
    else:
        source_block = json.dumps(source_artifact_summary, indent=2, default=str)

    history_block = (
        investigation_history.strip()
        if investigation_history and investigation_history.strip()
        else "(none — this is the first investigation pass)"
    )

    rule_text = DEBUG_PARAM_RULE.format(
        user_question=user_question,
        investigation_history=history_block,
    )

    prompt = f"""
{rule_text}

Tool that produced the artifact under review:
  name:        {tool_name}
  description: {tool_description}

Parameter under review:
  name:  {parameter_name}
  value: {repr(parameter_value)}
  reason given by the agent: {reason}

Source artifact for this parameter, if any:
{source_block}

Return one of: "parameter_value_suspect", "source_suspect", "neutral",
"not_relevant", with a brief reasoning.
"""
    _dbg(
        f"_call_debug_param_judge_llm: INVOKE param={parameter_name!r} "
        f"value={parameter_value!r} tool={tool_name!r} "
        f"prompt_len={len(prompt)}"
    )
    t0 = time.time()
    result = judge.invoke(prompt)
    elapsed = time.time() - t0
    _dbg(
        f"_call_debug_param_judge_llm: RETURN param={parameter_name!r} "
        f"verdict={result.get('verdict')!r} elapsed={elapsed:.2f}s"
    )
    return result


def _call_debug_synthesis_llm(
    *,
    user_question: str,
    investigation_history: str,
    walk_findings: List[Dict[str, Any]],
    judge,
) -> str:
    """Single synthesis call after the walk. Returns one paragraph of
    free-text diagnostic for the supervisor."""
    history_block = (
        investigation_history.strip()
        if investigation_history and investigation_history.strip()
        else "(none — this is the first investigation pass)"
    )

    # Compact dump of every finding for the synthesis prompt. Includes
    # the "neutral" / "not_relevant" findings so the synthesis can
    # reason about what was ruled out, not just what was flagged.
    walk_dump = json.dumps(walk_findings, indent=2, default=str)

    prompt = DEBUG_SYNTHESIS_RULE.format(
        user_question=user_question,
        investigation_history=history_block,
        walk_dump=walk_dump,
    )
    _dbg(
        f"_call_debug_synthesis_llm: INVOKE n_findings={len(walk_findings)} "
        f"prompt_len={len(prompt)}"
    )
    t0 = time.time()
    result = judge.invoke(prompt)
    elapsed = time.time() - t0
    # The judge contract returns dicts with `reasoning`. For the
    # synthesis we want free-text; we read `reasoning` since that's
    # where the prose lives.
    text = (
        result.get("reasoning")
        or result.get("verdict")
        or json.dumps(result)
    )
    _dbg(
        f"_call_debug_synthesis_llm: RETURN text_len={len(str(text))} "
        f"elapsed={elapsed:.2f}s"
    )
    return str(text)


def _walk_artifact_for_debug(
    *,
    result_id: str,
    user_question: str,
    investigation_history: str,
    judge,
    visited: Set[str],
    depth: int,
    max_depth: int,
    max_judge_calls: int,
    judge_call_counter: List[int],   # mutable single-element list
) -> List[Dict[str, Any]]:
    """Recursively walk the value-flow chain (parent_result_ids_w_args
    only) starting at `result_id`. For each parameter on each artifact,
    invoke the debug judge and record a finding.

    Cycle / depth / budget protection:
      - `visited` prevents reprocessing the same artifact.
      - `max_depth` caps the walk depth from the root.
      - `max_judge_calls` caps the total number of judge invocations
        across the whole walk; further parameters get a synthetic
        "neutral / budget_exceeded" finding so the synthesis still
        sees them but doesn't pay for them.

    Returns a list of finding dicts, one per parameter visited.
    """
    out: List[Dict[str, Any]] = []

    if result_id in visited:
        return out
    if depth > max_depth:
        return out
    visited = visited | {result_id}

    try:
        artifact = _get_artifact(result_id)
    except Exception as e:                                # noqa: BLE001
        out.append({
            "depth": depth,
            "result_id": result_id,
            "tool_name": "<artifact_lookup_failed>",
            "parameter_name": "<n/a>",
            "parameter_value": None,
            "context_at_site": None,
            "verdict": "not_relevant",
            "reasoning": (
                f"Could not load artifact {result_id} during debug walk: {e}"
            ),
            "source_result_id": None,
        })
        return out

    sources = _param_source_ids(artifact)
    args = artifact.args or {}
    reasons = artifact.reasons or {}

    for param_name, param_value in args.items():
        # Extract Context: from this parameter's reason for the
        # supervisor-facing finding.
        reason_text = reasons.get(param_name) or reasons.get("reasons") or ""
        ctx_at_site: Optional[str] = None
        if "Context:" in reason_text:
            _ctx_after = reason_text.split("Context:", 1)[1]
            for _delim in ("\n\nRationale:", "\nRationale:", "\n\n"):
                if _delim in _ctx_after:
                    _ctx_after = _ctx_after.split(_delim, 1)[0]
                    break
            ctx_at_site = _ctx_after.strip() or None

        # Resolve the source artifact summary if a real upstream ref
        # exists for this parameter. Empty / missing sources mean the
        # parameter wasn't sourced — pass None to the judge. Bare and
        # dotted refs are handled uniformly via _build_source_summary.
        source = sources.get(param_name)
        source_summary: Optional[Any] = None
        source_id_for_finding: Optional[Any] = None
        if isinstance(source, str) and source:
            try:
                source_summary = _build_source_summary(source)
                source_id_for_finding = source
            except Exception:                              # noqa: BLE001
                source_summary = None
                source_id_for_finding = source
        elif isinstance(source, list) and source and all(
            isinstance(s, str) and s for s in source
        ):
            try:
                source_summary = [_build_source_summary(s) for s in source]
                source_id_for_finding = source
            except Exception:                              # noqa: BLE001
                source_summary = None
                source_id_for_finding = source

        # Budget check. If we've burned through the judge budget, emit
        # a synthetic finding so the synthesis still sees this parameter
        # exists in the walk.
        if judge_call_counter[0] >= max_judge_calls:
            out.append({
                "depth": depth,
                "result_id": result_id,
                "tool_name": artifact.tool_name,
                "parameter_name": param_name,
                "parameter_value": param_value,
                "context_at_site": ctx_at_site,
                "verdict": "neutral",
                "reasoning": (
                    "Judge-call budget exceeded for this debug walk; "
                    "this parameter was not individually examined. "
                    "Increase max_judge_calls or scope the investigation "
                    "to a subtree to examine it."
                ),
                "source_result_id": source_id_for_finding,
            })
            continue

        judgement = _call_debug_param_judge_llm(
            tool_name=artifact.tool_name,
            tool_description=artifact.description or "",
            parameter_name=param_name,
            parameter_value=param_value,
            reason=reason_text,
            source_artifact_summary=source_summary,
            user_question=user_question,
            investigation_history=investigation_history,
            judge=judge,
        )
        judge_call_counter[0] += 1

        out.append({
            "depth": depth,
            "result_id": result_id,
            "tool_name": artifact.tool_name,
            "parameter_name": param_name,
            "parameter_value": param_value,
            "context_at_site": ctx_at_site,
            "verdict": judgement.get("verdict", "neutral"),
            "reasoning": judgement.get("reasoning", ""),
            "source_result_id": source_id_for_finding,
        })

    # Recurse into upstream BARE refs only. Dotted refs are NOT
    # recursed into — the dotted-source judge call at the immediate
    # downstream parameter has already produced its finding, and the
    # debug walker stops there. This mirrors the verifier's narrow
    # scope on dotted-ref claims.
    bare_upstream_ids: Set[str] = set()
    for source in sources.values():
        if isinstance(source, list):
            for s in source:
                if isinstance(s, str) and s and not _is_dotted_ref(s):
                    bare_upstream_ids.add(s)
        elif isinstance(source, str) and source and not _is_dotted_ref(source):
            bare_upstream_ids.add(source)

    for parent_id in bare_upstream_ids:
        out.extend(_walk_artifact_for_debug(
            result_id=parent_id,
            user_question=user_question,
            investigation_history=investigation_history,
            judge=judge,
            visited=visited,
            depth=depth + 1,
            max_depth=max_depth,
            max_judge_calls=max_judge_calls,
            judge_call_counter=judge_call_counter,
        ))

    return out



def _problem_text_for_debug_finding(
    *,
    verdict: str,
    parameter_name: str,
    parameter_value: Any,
    tool_name: str,
) -> str:
    """One-sentence problem text per debug-finding category."""
    val_repr = repr(parameter_value)
    if verdict == DebugCategory.PARAMETER_VALUE_SUSPECT:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact is a plausible cause of the "
            f"surprising result based on its value and role in the "
            f"study."
        )
    if verdict == DebugCategory.SOURCE_SUSPECT:
        return (
            f"Parameter '{parameter_name}' (value {val_repr}) on a "
            f"`{tool_name}` artifact draws from an upstream source that "
            f"is itself a plausible cause; investigate upstream rather "
            f"than this parameter directly."
        )
    return (
        f"Parameter '{parameter_name}' (value {val_repr}) on a "
        f"`{tool_name}` artifact has unrecognized debug verdict "
        f"{verdict!r}."
    )


def summarize_debug_for_supervisor(
    *,
    user_question: str,
    root_result_id: str,
    walk_findings: List[Dict[str, Any]],
    synthesis_text: str,
    judge_calls_used: int,
    max_judge_calls: int,
) -> Dict[str, Any]:
    """Filter the walk findings to potential-causes-only and assemble
    the supervisor-facing output dict.

    The walk's full contents are NOT returned — they were used internally
    by the synthesis. The supervisor sees only the potential-causes list
    (numbered) plus the synthesis paragraph plus a one-line summary.
    """
    potential_causes: List[Dict[str, Any]] = []
    issue_number = 0

    for finding in walk_findings:
        verdict = finding.get("verdict", "neutral")
        if verdict not in (
            DebugCategory.PARAMETER_VALUE_SUSPECT,
            DebugCategory.SOURCE_SUSPECT,
        ):
            continue

        issue_number += 1
        cat = verdict
        problem = _problem_text_for_debug_finding(
            verdict=verdict,
            parameter_name=finding["parameter_name"],
            parameter_value=finding["parameter_value"],
            tool_name=finding["tool_name"],
        )
        remediation = list(DEBUG_REMEDIATION_OPTIONS.get(cat, []))

        potential_causes.append({
            "issue_number": issue_number,
            "category": cat,
            "where": {
                "result_id": finding["result_id"],
                "tool_name": finding["tool_name"],
                "parameter": finding["parameter_name"],
                "parameter_value": finding["parameter_value"],
                "source_result_id": finding.get("source_result_id"),
            },
            "context_at_site": finding.get("context_at_site"),
            "problem": problem,
            "judge_reasoning": finding.get("reasoning"),
            "remediation_options": remediation,
        })

    n_potential = len(potential_causes)
    if n_potential == 0:
        summary = (
            "No potential causes identified in the value-flow chain. "
            "See the synthesis for guidance on where else the surprise "
            "may originate."
        )
    else:
        n_artifacts = len({c["where"]["result_id"] for c in potential_causes})
        summary = (
            f"{n_potential} potential cause"
            f"{'s' if n_potential != 1 else ''} identified across "
            f"{n_artifacts} artifact{'s' if n_artifacts != 1 else ''}."
        )

    return {
        "investigation_question": user_question,
        "root_result_id": root_result_id,
        "n_potential_causes": n_potential,
        "summary": summary,
        "potential_causes": potential_causes,
        "synthesis": synthesis_text,
        "judge_calls_used": judge_calls_used,
        "max_judge_calls": max_judge_calls,
        "budget_exceeded": judge_calls_used >= max_judge_calls,
    }


@tool
def debug_artifact_chain(
    result_id: Annotated[
        str,
        "The result_id of the artifact whose value is surprising or "
        "unexpected. The walk traces back through this artifact's "
        "value-flow chain (parent_result_ids_w_args).",
    ],
    user_question: Annotated[
        str,
        "The investigation question. Phrase as the user would: 'why is "
        "the adsorption energy 0.5 eV higher than literature?', 'why "
        "does the lattice constant change with ecutwfc above 80 Ry?', "
        "etc. The judge uses this to decide whether each parameter "
        "could plausibly contribute to the surprise.",
    ],
    judge,
    investigation_history: Annotated[
        str,
        "Free-text summary of what has already been tested and ruled "
        "out as the cause. Pass an empty string on the first debug "
        "call. On subsequent calls, describe what was tested and what "
        "the result was, so the judge does not re-flag previously "
        "ruled-out hypotheses. Example: 'Re-ran the production calc "
        "with ecutwfc doubled (40 -> 80) and the result was unchanged; "
        "ecutwfc is not the cause.'",
    ] = "",
    max_depth: Annotated[
        int,
        "Maximum depth of the chain walk from the root. Default 10.",
    ] = 10,
    max_judge_calls: Annotated[
        int,
        "Maximum number of per-parameter judge invocations across the "
        "whole walk. Default 50; raise for deep chains, lower to keep "
        "costs bounded. Parameters past the budget receive a synthetic "
        "'neutral' verdict noting the budget was exceeded.",
    ] = 50,
) -> Dict[str, Any]:
    """Investigate a surprising or unexpected artifact value.

    Walks the artifact's value-flow chain (parent_result_ids_w_args
    only) and asks an LLM judge, per parameter, whether that parameter
    could plausibly explain the user's question. Filters to
    potential-causes-only and returns a supervisor-facing dict in the
    same shape as `verify_structured_report`'s output: numbered
    findings with `where`, `problem`, `judge_reasoning`,
    `remediation_options`, plus a synthesis paragraph naming the most
    likely culprit.

    Output dict keys:
      investigation_question  : str  — echoes the user's question
      root_result_id          : str  — the artifact investigated
      n_potential_causes      : int
      summary                 : str  — one-line tally
      potential_causes        : List[Dict]  — numbered findings (only
                                "parameter_value_suspect" and
                                "source_suspect" verdicts surface here;
                                "neutral" / "not_relevant" findings are
                                used internally by the synthesis but not
                                returned)
      synthesis               : str  — one-paragraph diagnostic with a
                                fixed disclaimer prefix
      judge_calls_used        : int
      max_judge_calls         : int
      budget_exceeded         : bool

    Notes for the supervisor:

      * The output is HYPOTHESES, not conclusions. Each potential cause
        must be confirmed by re-running the relevant calculation with
        the suggested change.
      * Vary ONE parameter at a time in priority order from the
        synthesis; varying multiple at once makes attribution
        impossible.
      * If the first call's potential causes are exhausted (re-run and
        ruled out), call this tool AGAIN with the same `result_id` and
        `user_question`, but pass `investigation_history` describing
        what was ruled out. The judge will skip those and surface new
        candidates.
      * If the tool returns no potential causes AND the investigation
        history rules out the obvious candidates, the synthesis will
        say so explicitly — the surprise may originate outside the
        value-flow chain (in input data, physical assumptions, or in
        the user's expectation).
    """
    _dbg(
        f"debug_artifact_chain: ENTER root={result_id!r} "
        f"question_len={len(user_question)} "
        f"history_len={len(investigation_history)} "
        f"max_depth={max_depth} max_judge_calls={max_judge_calls}"
    )

    judge_call_counter = [0]
    walk_findings = _walk_artifact_for_debug(
        result_id=result_id,
        user_question=user_question,
        investigation_history=investigation_history,
        judge=judge,
        visited=set(),
        depth=0,
        max_depth=max_depth,
        max_judge_calls=max_judge_calls,
        judge_call_counter=judge_call_counter,
    )

    _dbg(
        f"debug_artifact_chain: walk complete — n_findings={len(walk_findings)} "
        f"judge_calls_used={judge_call_counter[0]}"
    )

    synthesis_text = _call_debug_synthesis_llm(
        user_question=user_question,
        investigation_history=investigation_history,
        walk_findings=walk_findings,
        judge=judge,
    )

    out = summarize_debug_for_supervisor(
        user_question=user_question,
        root_result_id=result_id,
        walk_findings=walk_findings,
        synthesis_text=synthesis_text,
        judge_calls_used=judge_call_counter[0],
        max_judge_calls=max_judge_calls,
    )

    _dbg(
        f"debug_artifact_chain: RETURN n_potential_causes="
        f"{out['n_potential_causes']} budget_exceeded={out['budget_exceeded']}"
    )
    return out


def verify_structured_report(
    reportName: Annotated[str, "Canvas key where the report was stored."],
    sensitive_parameters: Annotated[List[str], "User-specified sensitive parameters."],
    judge,
) -> Dict[str, Any]:
    """Verify a structured report end-to-end."""
    
    viz = VerifyVisualizer(html_path=os.path.join(var.my_WORKING_DIRECTORY, f"verify_{reportName}.html"))

    _dbg(
        f"verify_structured_report: ENTER reportName={reportName!r} "
        f"sensitive_parameters={sensitive_parameters}"
    )
    if reportName not in CANVAS.canvas:
        _dbg(
            f"verify_structured_report: canvas key not found. "
            f"Available keys: {list(CANVAS.canvas.keys())}"
        )
        raise ValueError(f"Canvas key '{reportName}' not found.")

    raw = CANVAS.canvas[reportName]
    if isinstance(raw, StructuredScientificReport):
        _dbg(
            "verify_structured_report: canvas entry was already a "
            "StructuredScientificReport instance"
        )
        parsed_report = raw
    else:
        _dbg(
            f"verify_structured_report: canvas entry is "
            f"{type(raw).__name__} — re-validating into "
            "StructuredScientificReport"
        )
        try:
            parsed_report = StructuredScientificReport.model_validate(raw)
        except Exception as e:
            _dbg(
                f"verify_structured_report: model_validate FAILED — "
                f"{type(e).__name__}: {e}"
            )
            raise ValueError(
                f"Canvas entry '{reportName}' is not a valid "
                f"StructuredScientificReport: {e}"
            ) from e

    issues: List[ReportVerificationIssue] = []
    artifact_results: List[ArtifactVerificationResult] = []
    # Flat set of artifact IDs visited at least once across the whole
    # report, for the `checked_result_ids` output field.
    all_visited: Set[str] = set()
    # Cross-claim verification cache. Keyed by
    # (artifact_result_id, frozenset(varied_parameters)). Two claims
    # with the same varied set share cache entries; claims with
    # different varied sets cache separately because R1 vs R2 branch
    # selection (and the verdict that follows) depends on which
    # parameters are being varied.
    visited_cache: Set[Tuple[str, frozenset]] = set()

    overall_goal = parsed_report.overall_goal
    _dbg(
        f"verify_structured_report: parsed report — overall_goal_len="
        f"{len(overall_goal)} n_claims="
        f"{len(parsed_report.numerical_results)}"
    )
    if not overall_goal or not overall_goal.strip():
        _dbg(
            "verify_structured_report: overall_goal empty — emitting "
            "report-level FAIL and short-circuiting"
        )
        issues.append(ReportVerificationIssue(
            category=IssueCategory.SCHEMA_VIOLATION,
            severity="fail",
            where={"field": "overall_goal"},
            problem=(
                "Report has empty `overall_goal`. The verifier requires a "
                "non-empty goal so per-parameter judgments have study "
                "context."
            ),
        ))
        raw_result = ReportVerificationResult(
            overall_verdict="fail",
            checked_result_ids=[],
            issues=issues,
            artifact_results=artifact_results,
        ).model_dump()
        return summarize_verification_for_supervisor(raw_result)
        
    viz.begin_report(
        report_name=reportName,
        overall_goal=overall_goal,
        sensitive_parameters=sensitive_parameters,
        claims=parsed_report.numerical_results
    )

    for claim in parsed_report.numerical_results:
        viz.begin_claim(claim)
        _dbg(
            f"verify_structured_report: ===== claim "
            f"{claim.quantity_name!r} ===== value={claim.value!r} "
            f"result_id={claim.result_id!r} varied={claim.varied_parameters}"
        )
        try:
            _get_artifact(claim.result_id)
        except Exception as e:                        # noqa: BLE001
            _dbg(
                f"verify_structured_report: claim {claim.quantity_name!r} "
                f"artifact lookup FAILED — {type(e).__name__}: {e}"
            )
            issues.append(ReportVerificationIssue(
                category=IssueCategory.ARTIFACT_LOOKUP_FAILED,
                severity="fail",
                where={
                    "claim_name": claim.quantity_name,
                    "result_id": claim.result_id,
                },
                problem=(
                    f"Claim '{claim.quantity_name}' points at "
                    f"`result_id={claim.result_id}` but that id could not "
                    f"be loaded from the CANVAS registry."
                ),
                judge_reasoning=str(e),
            ))
            continue

        ok, msg = CANVAS.verify_artifact(claim.value, claim.result_id)
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} "
            f"value-match -> ok={ok} msg={msg!r}"
        )
        if not ok:
            issues.append(ReportVerificationIssue(
                category=IssueCategory.VALUE_MISMATCH_CLAIM,
                severity="fail",
                where={
                    "claim_name": claim.quantity_name,
                    "claim_value": claim.value,
                    "claim_unit": claim.unit,
                    "result_id": claim.result_id,
                },
                problem=(
                    f"Claim '{claim.quantity_name}' asserts value "
                    f"{claim.value!r} (unit={claim.unit!r}) against "
                    f"`result_id={claim.result_id}`, but the registered "
                    f"artifact's value does not match."
                ),
                judge_reasoning=msg,
            ))
            continue

        # Per-claim descent. Reuse the cross-claim `visited_cache` so the
        # second claim that reaches a shared upstream artifact under the
        # SAME varied set short-circuits. Claims with different varied
        # sets cache separately and re-judge that artifact.
        cache_size_before = len(visited_cache)
        ids_before = {k[0] for k in visited_cache}
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"starting recursive descent (cache size before="
            f"{cache_size_before})"
        )
        _verify_one_artifact_recursive(
            result_id=claim.result_id,
            target_quantity=claim.quantity_name,
            overall_goal=overall_goal,
            varied_parameters=claim.varied_parameters,
            sensitive_parameters=sensitive_parameters,
            visited=visited_cache,
            artifact_results=artifact_results,
            issues=issues,
            judge=judge,
            viz=viz
        )
        ids_after = {k[0] for k in visited_cache}
        new_ids_for_this_claim = ids_after - ids_before
        all_visited.update(ids_after)
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"recursive descent complete (cache size after="
            f"{len(visited_cache)}, "
            f"new artifacts touched this claim={len(new_ids_for_this_claim)})"
        )
        if any(i.severity == "fail" for i in issues):
            overall = "fail"
        elif any(i.severity == "warning" for i in issues):
            overall = "warning"
        else:
            overall = "pass"
        # Per-claim viz hook gets a short status string (since the old
        # `issues[-1].message` reference is no longer valid — issues[-1]
        # is not necessarily related to this claim, and `.message` no
        # longer exists in the new schema).
        viz.end_claim(
            claim.quantity_name, overall,
            f"claim verdict={overall} "
            f"(issues so far: {sum(1 for i in issues if i.severity == 'fail')} "
            f"fails, "
            f"{sum(1 for i in issues if i.severity == 'warning')} warnings)"
        )

    if any(i.severity == "fail" for i in issues):
        overall = "fail"
    elif any(i.severity == "warning" for i in issues):
        overall = "warning"
    else:
        overall = "pass"

    # Issue breakdown by category for log readability.
    category_counts: Dict[str, int] = {}
    for i in issues:
        category_counts[i.category] = category_counts.get(i.category, 0) + 1
    n_pass = sum(1 for r in artifact_results if r.overall_verdict == "pass")
    n_warn = sum(1 for r in artifact_results if r.overall_verdict == "warning")
    n_fail = sum(1 for r in artifact_results if r.overall_verdict == "fail")
    _dbg(
        f"verify_structured_report: AGGREGATE overall={overall!r} "
        f"n_artifacts_checked={len(all_visited)} "
        f"artifact_verdicts(pass/warn/fail)={n_pass}/{n_warn}/{n_fail} "
        f"issues_by_category={category_counts}"
    )

    raw_result = ReportVerificationResult(
        overall_verdict=overall,
        checked_result_ids=sorted(all_visited),
        issues=issues,
        artifact_results=artifact_results,
    ).model_dump()
    return summarize_verification_for_supervisor(raw_result)