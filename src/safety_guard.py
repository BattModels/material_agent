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

* `overall_goal` is now threaded into every judge call. The judge sees both
  the immediate target quantity and the overall study purpose, so it can
  weigh atypical-looking values that are sensible inside an exploration.

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
from typing import Any, Dict, List, Optional, Literal, Set, Annotated, Tuple

from pydantic import BaseModel, Field
from langchain.tools import tool
from src.myCANVAS import CANVAS
from src import var

from langchain_anthropic import ChatAnthropic


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


# --- Rule strings used inside verify_artifact_parameterization ---------------

R1_RULE = (
    "Parameter is sensitive and not being varied for the current target "
    "quantity. It must be obtained from another tool output; the source's "
    "value must match, and the source must make semantic sense for this "
    "parameter. "
    "A good rationale identifies the study this call is part of, the role "
    "the parameter plays in that study, and the basis for the specific "
    "value. Reject rationales that don't situate the parameter in the "
    "workflow (e.g. 'standard value' or 'reasonable choice' without "
    "context)."
)

R2_RULE = (
    "Parameter is sensitive but intentionally varied for the current target "
    "quantity. Its value must still be sensible for exploring the quantity. "
    "A good rationale identifies the study this call is part of (e.g. "
    "convergence sweep, sensitivity test, EOS scan), the role this "
    "parameter plays in that study (which one is being swept, which others "
    "are pinned), and why this specific value was chosen as a sweep point "
    "rather than another. Atypical values (e.g. very low ecutwfc) are "
    "acceptable when the study context justifies them — for example a "
    "convergence sweep deliberately includes low values to characterize "
    "the convergence curve. Reject only when the rationale fails to "
    "situate the value within a coherent exploration."
)

R3_RULE = (
    "Parameter is not under the hard provenance rule. Its value must still "
    "be sensible based on the recorded reason. "
    "A good rationale identifies the study this call is part of, the role "
    "this parameter plays in it (e.g. fixed default, inherited from upstream, "
    "placeholder before a real value is obtained), and the basis for the "
    "specific value. Approve rationales that situate the parameter in the "
    "workflow — even when the value itself looks atypical, as long as the "
    "context justifies it. Reject rationales that lack workflow context "
    "(e.g. 'standard value' with no further detail)."
)

PARAM_JUDGE_GUIDANCE = """
Guidance:
- A rationale that explicitly situates the parameter within a study
  (convergence test / production run / sensitivity sweep / etc.) and
  explains the parameter's role in that study generally deserves PASS,
  even if the value itself looks atypical out of context.
- A rationale that lists only "intent and effect" without identifying
  the study or the parameter's role in it deserves WARNING — the value
  may be fine, but the reviewer cannot confirm without that context.
- A rationale that gives no workflow context at all, and the value would
  be questionable in a default production setting, deserves FAIL.
- When the parameter is intentionally being varied (R2), accept values
  outside the typical "production" range as long as they are coherent
  sweep points for the stated study.
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
    level: Literal["report", "claim", "artifact", "parameter"]
    location: str
    verdict: Literal["pass", "fail", "warning"]
    message: str


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
    artifact = CANVAS.get_artifact(result_id)
    if artifact is None:
        raise ValueError(f"Result id '{result_id}' not found in result_registry.")
    return artifact


def _flatten_listed_value(artifact: Any) -> Any:
    v = artifact.value
    if isinstance(v, list):
        return [getattr(item, "value", item) for item in v]
    return v


def _is_listed(artifact: Any) -> bool:
    return isinstance(artifact.value, list)


def _param_source_ids(artifact: Any) -> Dict[str, Any]:
    """Per-parameter source map.

    Scalar entries are str; list entries are List[str], aligned by index
    with the corresponding entry in `artifact.args`.
    """
    return getattr(artifact, "parent_result_ids_w_args", {}) or {}


def _collect_recursive_source_ids(artifact: Any) -> List[str]:
    """All upstream result_ids referenced by this artifact, flattened."""
    source_ids: Set[str] = set(artifact.parent_result_ids or [])
    for v in _param_source_ids(artifact).values():
        if isinstance(v, list):
            source_ids.update(s for s in v if isinstance(s, str))
        elif isinstance(v, str):
            source_ids.add(v)
    return list(source_ids)


def _summarize_artifact(artifact: Any) -> Dict[str, Any]:
    """Compact, JSON-serializable view used inside LLM judge prompts."""
    return {
        "result_id": artifact.result_id,
        "tool_name": artifact.tool_name,
        "description": artifact.description,
        "args": artifact.args,
        "reasons": artifact.reasons,
        "parent_result_ids": artifact.parent_result_ids,
        "metadata": artifact.metadata,
        "value_repr": repr(_flatten_listed_value(artifact)),
    }


def _fold_verdict(current: str, incoming: str) -> str:
    """`fail` > `warning` > `pass`. `info` is non-blocking."""
    if current == "fail" or incoming == "fail":
        return "fail"
    if current == "warning" or incoming == "warning":
        return "warning"
    return "pass"


def _is_complete_numeric_token(needle: str, haystack: str) -> bool:
    try:
        target = float(needle)
    except (TypeError, ValueError):
        return False
    for m in _NUMBER_RE.finditer(haystack):
        try:
            if math.isclose(float(m.group()), target,
                            rel_tol=1e-12, abs_tol=1e-12):
                return True
        except ValueError:
            continue
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
    return bool(n) and n in h


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

    if val_is_list != src_is_list:
        return False, [], [], (
            f"Parameter '{param_name}': value is "
            f"{'a list' if val_is_list else 'a scalar'} but source is "
            f"{'a list' if src_is_list else 'a scalar'}. List-shape "
            "mismatch — the producing tool registered inconsistent "
            "provenance."
        )

    if val_is_list:
        if len(param_value) != len(source):
            return False, [], [], (
                f"Parameter '{param_name}': value list has length "
                f"{len(param_value)} but source list has length "
                f"{len(source)}. List-shape mismatch — the producing tool "
                "registered inconsistent provenance."
            )
        return True, list(param_value), list(source), None

    return False, [param_value], [source], None


# =========================================================
# LLM judge calls
# =========================================================

def _call_param_judge_llm(
    *,
    overall_goal: str,
    target_quantity: str,
    varied_parameters: List[str],
    sensitive_parameters: List[str],
    artifact_summary: Dict[str, Any],
    parameter_name: str,
    parameter_value: Any,
    reason: str,
    source_artifact_summary: Optional[Any],   # dict or list-of-dicts
    rule_to_apply: str,
    judge,
) -> Dict[str, Any]:
    overall_goal_text = overall_goal if overall_goal.strip() else "(not specified)"
    if source_artifact_summary is None:
        source_block = "None"
    else:
        source_block = json.dumps(source_artifact_summary, indent=2, default=str)

    prompt = f"""
You are verifying whether one tool parameter was set correctly in a scientific
agent workflow.

Overall study goal:
{overall_goal_text}

Target quantity being sought (current verification scope):
{target_quantity}

Parameters intentionally varied while seeking this quantity:
{varied_parameters}

Sensitive parameters:
{sensitive_parameters}

Current artifact:
{json.dumps(artifact_summary, indent=2, default=str)}

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
    return judge.invoke(prompt)


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
    return judge.invoke(prompt)


# =========================================================
# Tool 1 — Structured report generation
# =========================================================
@tool
def generate_structured_report(
    overall_goal: Annotated[str, "Overall goal of the study. Required and non-empty."],
    quantity_specs: Annotated[
        List[QuantitySpec],
        "List of (value, result_id) claims plus presentation metadata.",
    ],
    qualitative_findings: Annotated[str, "Prose findings."] = "",
    conclusion: Annotated[str, "Prose conclusion."] = "",
    strict: Annotated[bool, "Strict mode for orphan-number detection."] = True,
) -> Dict[str, Any]:
    """Generate a structured report from (value, result_id) claims."""

    if not overall_goal or not overall_goal.strip():
        raise ValueError(
            "overall_goal is required and must be non-empty. It describes the "
            "study's purpose and is used by the verifier to provide context "
            "to per-parameter judgments throughout the call chain."
        )

    parsed_specs = [QuantitySpec.model_validate(x) for x in quantity_specs]

    names = [s.quantity_name for s in parsed_specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"Duplicate quantity_name(s): {dupes}. Each must be unique — "
            "it is the citation handle."
        )

    quantities_sought: List[Dict[str, Any]] = []
    numerical_results: List[ReportNumericalClaim] = []

    for spec in parsed_specs:
        _get_artifact(spec.result_id)
        ok, msg = CANVAS.verify_artifact(spec.value, spec.result_id)
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

    for field_name, text in (("qualitative_findings", qualitative_findings),
                             ("conclusion", conclusion)):
        used = set(_CLAIM_TAG.findall(text))
        missing = used - set(by_name)
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
            if orphans:
                raise ValueError(
                    f"{field_name} contains un-cited numbers: {orphans}. "
                    "Wrap each measurement in `{claim:<quantity_name>}`, or "
                    "wrap non-measurement numbers (run counts, indices, "
                    "dates, figure refs) in `{lit:<number>}`. Use "
                    "strict=False only as a last resort."
                )

    rendered = _render_report_markdown(
        overall_goal=overall_goal,
        quantities=numerical_results,
        qualitative_findings=qualitative_findings,
        conclusion=conclusion,
    )

    return StructuredScientificReport(
        overall_goal=overall_goal,
        quantities_sought=quantities_sought,
        numerical_results=numerical_results,
        qualitative_findings=qualitative_findings,
        conclusion=conclusion,
        rendered_markdown=rendered,
    ).model_dump()


def _render_report_markdown(
    *,
    overall_goal: str,
    quantities: List[ReportNumericalClaim],
    qualitative_findings: str,
    conclusion: str,
) -> str:
    by_name = {c.quantity_name: c for c in quantities}

    def _sub(m: "re.Match[str]") -> str:
        c = by_name[m.group(1)]
        val = f"{c.value:g}"
        if c.unit:
            val += f" {c.unit}"
        return f"{val} [^{c.quantity_name}]"

    def _unwrap_lit(text: str) -> str:
        return _LIT_TAG.sub(lambda m: m.group(1), text)

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
    return "\n".join(lines).rstrip() + "\n"


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
    artifact = _get_artifact(result_id)
    args = artifact.args or {}
    reasons = artifact.reasons or {}
    param_source_ids = _param_source_ids(artifact)
    artifact_summary = _summarize_artifact(artifact)
    is_info_tool = artifact.tool_name in INFO_TOOLS

    checks: List[ParamCheckResult] = []
    overall: str = "pass"

    # R0 — extraction tool special case.
    if artifact.tool_name in EXTRACTION_TOOL_NAMES:
        for c in _verify_extraction_behavior(
            artifact=artifact, args=args,
            param_source_ids=param_source_ids,
            overall_goal=overall_goal, judge=judge,
        ):
            checks.append(c)
            overall = _fold_verdict(overall, c.verdict)

    # Per-parameter checks.
    for param_name, param_value in args.items():
        # Reason lookup: per-param key first, then the singleton "reasons"
        # fallback used by tools whose `reasons: str` was wrapped as
        # `{"reasons": "..."}` at registration time.
        reason = reasons.get(param_name, "") or reasons.get("reasons", "")
        source = param_source_ids.get(param_name)

        is_sensitive = param_name in sensitive_parameters
        is_varied = param_name in varied_parameters

        # R1 — sensitive and not varied: must be sourced.
        if is_sensitive and not is_varied:
            if source is None:
                checks.append(ParamCheckResult(
                    parameter_name=param_name, parameter_value=param_value,
                    verdict="fail", rule_applied=R1_RULE,
                    source_result_id=None,
                    reasoning=(
                        f"Parameter '{param_name}' is sensitive and it is not intentionally varied "
                        "but there's no recorded source for Parameter '{param_name}'"
                        "indicating potentially hallucinated input."
                    ),
                ))
                overall = _fold_verdict(overall, "fail")
                continue

            checks_to_add, branch_verdict = _verify_sourced_param(
                param_name=param_name, param_value=param_value,
                source=source, rule=R1_RULE,
                overall_goal=overall_goal,
                target_quantity=target_quantity,
                varied_parameters=varied_parameters,
                sensitive_parameters=sensitive_parameters,
                artifact_summary=artifact_summary,
                reason=reason, judge=judge,
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

        # Belt-and-braces source check. If a source is declared, verify it
        # element-by-element (list shape) or directly (scalar shape).
        source_summary: Optional[Any] = None
        if source is not None:
            checks_to_add, branch_verdict, source_summary = (
                _verify_optional_source_match(
                    param_name=param_name, param_value=param_value,
                    source=source, rule=rule,
                )
            )
            if checks_to_add:
                checks.extend(checks_to_add)
                if branch_verdict == "fail":
                    overall = _fold_verdict(overall, "fail")
                    continue
                overall = _fold_verdict(overall, branch_verdict)

        # LLM judgment.
        judgement = _call_param_judge_llm(
            overall_goal=overall_goal,
            target_quantity=target_quantity,
            varied_parameters=varied_parameters,
            sensitive_parameters=sensitive_parameters,
            artifact_summary=artifact_summary,
            parameter_name=param_name, 
            parameter_value=param_value,
            reason=reason, 
            source_artifact_summary=source_summary,
            rule_to_apply=rule, 
            judge=judge,
        )
        checks.append(ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict=judgement["verdict"], rule_applied=rule,
            source_result_id=source,
            reasoning=judgement["reasoning"],
        ))
        overall = _fold_verdict(overall, judgement["verdict"])

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
    overall_goal: str,
    target_quantity: str,
    varied_parameters: List[str],
    sensitive_parameters: List[str],
    artifact_summary: Dict[str, Any],
    reason: str,
    judge,
) -> Tuple[List[ParamCheckResult], str]:
    """R1 path: must be sourced. Element-by-element value match, then a
    single LLM judge call evaluating the source(s) collectively."""
    is_list, values, sources, err = _normalize_to_list_pair(
        param_value, source, param_name
    )
    if err is not None:
        return [ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict="fail", rule_applied=rule,
            source_result_id=source, reasoning=err,
        )], "fail"

    # Value-match each element.
    for i, (v, ref) in enumerate(zip(values, sources)):
        ok, msg = CANVAS.verify_artifact(v, ref)
        if not ok:
            label = f"{param_name}[{i}]" if is_list else param_name
            return [ParamCheckResult(
                parameter_name=label, parameter_value=v,
                verdict="fail", rule_applied=rule,
                source_result_id=ref, reasoning=msg,
            )], "fail"

    # Single judge call over the collective source set.
    if is_list:
        source_summary: Any = [
            _summarize_artifact(_get_artifact(ref)) for ref in sources
        ]
    else:
        source_summary = _summarize_artifact(_get_artifact(sources[0]))

    judgement = _call_param_judge_llm(
        overall_goal=overall_goal,
        target_quantity=target_quantity,
        varied_parameters=varied_parameters,
        sensitive_parameters=sensitive_parameters,
        artifact_summary=artifact_summary,
        parameter_name=param_name, parameter_value=param_value,
        reason=reason, source_artifact_summary=source_summary,
        rule_to_apply=rule, judge=judge,
    )
    return [ParamCheckResult(
        parameter_name=param_name, parameter_value=param_value,
        verdict=judgement["verdict"], rule_applied=rule,
        source_result_id=source,
        reasoning=judgement["reasoning"],
    )], judgement["verdict"]


def _verify_optional_source_match(
    *,
    param_name: str,
    param_value: Any,
    source: Any,
    rule: str,
) -> Tuple[List[ParamCheckResult], str, Optional[Any]]:
    """R2/R3 belt-and-braces. Returns checks (failures only), branch verdict,
    and the source summary to pass to the judge if value-match passes."""
    is_list, values, sources, err = _normalize_to_list_pair(
        param_value, source, param_name
    )
    if err is not None:
        return [ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict="fail", rule_applied=rule,
            source_result_id=source, reasoning=err,
        )], "fail", None

    for i, (v, ref) in enumerate(zip(values, sources)):
        ok, msg = CANVAS.verify_artifact(v, ref)
        if not ok:
            label = f"{param_name}[{i}]" if is_list else param_name
            return [ParamCheckResult(
                parameter_name=label, parameter_value=v,
                verdict="fail", rule_applied=rule,
                source_result_id=ref, reasoning=msg,
            )], "fail", None

    if is_list:
        return [], "pass", [
            _summarize_artifact(_get_artifact(ref)) for ref in sources
        ]
    return [], "pass", _summarize_artifact(_get_artifact(sources[0]))


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
    extraction_rationale = artifact.description

    source_text: Optional[str] = (
        args.get("source_text") or args.get("text") or args.get("content")
    )

    source_id_for_ref: Optional[str] = None
    canonical = param_source_ids.get("source_tool_call_id")
    if isinstance(canonical, str) and canonical:
        source_id_for_ref = canonical
    if source_id_for_ref is None:
        for candidate in ("source", "source_file", "file", "filename",
                          "file_path", "input_file", "path"):
            v = param_source_ids.get(candidate)
            if isinstance(v, str) and v:
                source_id_for_ref = v
                break
    if source_id_for_ref is None and len(param_source_ids) == 1:
        only = next(iter(param_source_ids.values()))
        if isinstance(only, str):
            source_id_for_ref = only

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
        except Exception:
            pass

    if not source_text:
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
        )]

    if _is_listed(artifact):
        elements: List[Tuple[str, Any]] = [
            (f"<extracted_value[{i}]>", getattr(item, "value", item))
            for i, item in enumerate(artifact.value)
        ]
    else:
        elements = [("<extracted_value>", artifact.value)]

    out: List[ParamCheckResult] = []
    source_text_str = str(source_text)

    for label, value in elements:
        if _looks_numeric(value):
            if not _is_complete_numeric_token(str(value), source_text_str):
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
                ))
                continue
        else:
            if not _is_complete_text_token(str(value), source_text_str):
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
                ))
                continue

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
        ))

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
    visited: Set[str],
    artifact_results: List[ArtifactVerificationResult],
    issues: List[ReportVerificationIssue],
    judge,
) -> None:
    if result_id in visited:
        return
    visited.add(result_id)

    artifact = _get_artifact(result_id)

    # Post-order traversal: verify children first, then the current node.
    # The output `artifact_results` reads bottom-up, so a reader scrolling
    # through it sees the leaf verdicts before the parent that depends on
    # them — matching how a human auditor would walk the chain.
    children = _collect_recursive_source_ids(artifact)
    children_checked: List[str] = []
    for child_id in children:
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
            )
            children_checked.append(child_id)
        except Exception as e:                        # noqa: BLE001
            issues.append(ReportVerificationIssue(
                level="artifact",
                location=f"result_id={result_id} -> child={child_id}",
                verdict="fail",
                message=str(e),
            ))

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

    if local_result.overall_verdict != "pass":
        issues.append(ReportVerificationIssue(
            level="artifact",
            location=f"result_id={result_id}",
            verdict=local_result.overall_verdict,
            message=local_result.summary,
        ))


def verify_structured_report(
    reportName: Annotated[str, "Canvas key where the report was stored."],
    sensitive_parameters: Annotated[List[str], "User-specified sensitive parameters."],
    judge,
) -> Dict[str, Any]:
    """Verify a structured report end-to-end."""
    if reportName not in CANVAS.canvas:
        raise ValueError(f"Canvas key '{reportName}' not found.")

    raw = CANVAS.canvas[reportName]
    if isinstance(raw, StructuredScientificReport):
        parsed_report = raw
    else:
        try:
            parsed_report = StructuredScientificReport.model_validate(raw)
        except Exception as e:
            raise ValueError(
                f"Canvas entry '{reportName}' is not a valid "
                f"StructuredScientificReport: {e}"
            ) from e

    issues: List[ReportVerificationIssue] = []
    artifact_results: List[ArtifactVerificationResult] = []
    all_visited: Set[str] = set()

    overall_goal = parsed_report.overall_goal
    if not overall_goal or not overall_goal.strip():
        issues.append(ReportVerificationIssue(
            level="report",
            location="overall_goal",
            verdict="fail",
            message=(
                "Report has empty overall_goal. The verifier requires a "
                "non-empty goal to provide context to per-parameter "
                "judgments. Regenerate the report with a meaningful "
                "overall_goal."
            ),
        ))
        return ReportVerificationResult(
            overall_verdict="fail",
            checked_result_ids=[],
            issues=issues,
            artifact_results=artifact_results,
        ).model_dump()

    for claim in parsed_report.numerical_results:
        try:
            _get_artifact(claim.result_id)
        except Exception as e:                        # noqa: BLE001
            issues.append(ReportVerificationIssue(
                level="claim",
                location=f"quantity={claim.quantity_name}",
                verdict="fail",
                message=str(e),
            ))
            continue

        ok, msg = CANVAS.verify_artifact(claim.value, claim.result_id)
        if not ok:
            issues.append(ReportVerificationIssue(
                level="claim",
                location=f"quantity={claim.quantity_name}, result_id={claim.result_id}",
                verdict="fail",
                message=msg,
            ))
            continue

        # Per-claim visited scope: the same upstream artifact reachable from
        # two different claims gets verified twice, under each claim's own
        # `varied_parameters`.
        visited: Set[str] = set()
        _verify_one_artifact_recursive(
            result_id=claim.result_id,
            target_quantity=claim.quantity_name,
            overall_goal=overall_goal,
            varied_parameters=claim.varied_parameters,
            sensitive_parameters=sensitive_parameters,
            visited=visited,
            artifact_results=artifact_results,
            issues=issues,
            judge=judge,
        )
        all_visited.update(visited)

    if any(i.verdict == "fail" for i in issues):
        overall = "fail"
    elif any(i.verdict == "warning" for i in issues):
        overall = "warning"
    else:
        overall = "pass"

    return ReportVerificationResult(
        overall_verdict=overall,
        checked_result_ids=sorted(all_visited),
        issues=issues,
        artifact_results=artifact_results,
    ).model_dump()