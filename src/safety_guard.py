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
import time
from typing import Any, Dict, List, Optional, Literal, Set, Annotated, Tuple

from pydantic import BaseModel, Field
from langchain.tools import tool
from src.myCANVAS import CANVAS
from src import var

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

# Sentinel used in *_ref / *_w_ref arguments and in `parent_result_ids_w_args`
# to indicate that the agent has explicitly chosen the value as a placeholder
# at the moment of the tool call. No upstream artifact is required; the
# verifier judges only whether the chosen placeholder is a reasonable default.
PLACEHOLDER_REF = "PLACEHOLDER"

R1_PLACEHOLDER_RULE = (
    "Parameter is sensitive, not being varied for the current target "
    "quantity, and the agent has explicitly marked it as a placeholder "
    "(no upstream source artifact yet exists — typically because the "
    "study has not characterized this parameter yet). "
    "Provenance is not required. The judgment is: is the chosen "
    "placeholder value a reasonable default to hold while other "
    "parameters are being characterized? "
    "Approve placeholders that are commonly-used safe defaults (e.g. a "
    "tight kspacing while ecutwfc is being swept) and whose rationale "
    "clearly identifies the value as provisional. Reject placeholders "
    "that are unusual choices for a default, or whose rationale fails "
    "to acknowledge the value as provisional."
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
    acknowledged_placeholders: List[str] = Field(
        default_factory=list,
        description=(
            "Parameters that the agent explicitly acknowledges were held at "
            "a placeholder value (PLACEHOLDER_REF) somewhere in the value-"
            "flow chain to this claim, and that the agent does NOT intend "
            "to characterize before this claim is reported. The claim is "
            "thereby declared to be conditional on those held values. "
            "By contrast, a placeholder on a parameter NOT listed here AND "
            "not in `varied_parameters` AND not pinned to a real ref by an "
            "artifact closer to this claim in the chain will fail the "
            "report. Use this for claims that are themselves the answer to "
            "characterizing one parameter while other sensitive parameters "
            "are deliberately held — e.g. an `optimal_ecutwfc` claim that "
            "holds kspacing at a tight default. For claims that should "
            "stand alone (production measurements, final results), leave "
            "this list empty and ensure every sensitive parameter has been "
            "characterized upstream. Must not overlap with `varied_parameters`."
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
    acknowledged_placeholders: List[str] = Field(default_factory=list)
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
    """All upstream result_ids referenced by this artifact, flattened.

    The PLACEHOLDER_REF sentinel is filtered out — it marks an explicit
    placeholder declaration, not a real upstream artifact, so the recursive
    walker must not try to dereference it.
    """
    source_ids: Set[str] = set(
        s for s in (artifact.parent_result_ids or []) if s != PLACEHOLDER_REF
    )
    for v in _param_source_ids(artifact).values():
        if isinstance(v, list):
            source_ids.update(
                s for s in v
                if isinstance(s, str) and s != PLACEHOLDER_REF
            )
        elif isinstance(v, str) and v != PLACEHOLDER_REF:
            source_ids.add(v)
    out = list(source_ids)
    _dbg(
        f"_collect_recursive_source_ids: result_id={artifact.result_id!r} "
        f"collected n={len(out)} ids={out}"
    )
    return out


def _walk_placeholders_for_claim(
    result_id: str,
    seen_real_for_param: Set[str],
    varied_parameters: Set[str],
    acknowledged_placeholders: Set[str],
    visited: Set[str],
) -> List[Tuple[str, str, str, str]]:
    """Walk the value-flow chain (parent_result_ids_w_args only) from a claim
    and surface unresolved placeholders.

    A placeholder on parameter X encountered in artifact A is **resolved** iff:
      (a) X has been pinned by a real ref on the path closer to the claim
          (i.e. X is in `seen_real_for_param`), OR
      (b) X is in the claim's `varied_parameters` (X is being characterized
          by the claim itself), OR
      (c) X is in the claim's `acknowledged_placeholders` (the agent has
          declared the claim conditional on X being held provisional).

    Otherwise it is unresolved.

    Walks only `parent_result_ids_w_args` — not the broad `parent_result_ids`
    flat list. Artifacts that were upstream-but-not-value-providing (e.g.
    convergence-sweep input scripts whose values never flowed forward) are
    not reached and their placeholders are not flagged.

    `seen_real_for_param` is passed by value (a copy is forked at each
    recursive call), so diamond chains track resolution per traversal path:
    if any path from the claim to an ancestor passes through an unresolved
    placeholder, that path's complaint is recorded.

    `visited` is used only to break cycles; it does NOT prevent the same
    artifact from being judged multiple times under different paths.

    Returns
    -------
    List of (artifact_id, param_name, source_repr, message) tuples.
    Empty list means every placeholder reachable from the claim was resolved.
    """
    out: List[Tuple[str, str, str, str]] = []

    if result_id in visited:
        return out
    visited = visited | {result_id}

    try:
        artifact = _get_artifact(result_id)
    except Exception as e:                            # noqa: BLE001
        out.append((
            result_id, "<artifact_lookup_failed>", "",
            f"Could not load artifact '{result_id}' while walking value-flow "
            f"chain for placeholder resolution: {e}"
        ))
        return out

    sources = _param_source_ids(artifact)

    # Phase 1: identify parameters that THIS artifact pins to real refs.
    # A list-shaped source counts as pinning only if every element is real;
    # any PLACEHOLDER_REF inside a list means the parameter as a whole is
    # not pinned by this artifact.
    newly_pinned: Set[str] = set()
    for param, source in sources.items():
        if isinstance(source, str) and source and source != PLACEHOLDER_REF:
            newly_pinned.add(param)
        elif isinstance(source, list) and source and all(
            isinstance(s, str) and s and s != PLACEHOLDER_REF for s in source
        ):
            newly_pinned.add(param)
            
    _dbg(f"newly pinned: {newly_pinned!r}")

    # Resolution context that applies to placeholders found AT THIS artifact:
    # a placeholder on X here is resolved if X is already pinned closer to
    # the claim (seen_real_for_param), OR X is varied / acknowledged on the
    # claim. Note: `newly_pinned` is NOT included — pinning at this artifact
    # cannot resolve a placeholder also at this artifact (that would be a
    # contradiction in the same source map).
    resolved_here = (
        seen_real_for_param | varied_parameters | acknowledged_placeholders
    )
    
    _dbg(f"resolved_here: {resolved_here!r}")

    # Phase 2: surface unresolved placeholders at this artifact.
    for param, source in sources.items():
        is_placeholder = source == PLACEHOLDER_REF or (
            isinstance(source, list) and PLACEHOLDER_REF in source
        )
        if is_placeholder and param not in resolved_here:
            out.append((
                result_id, param, repr(source),
                f"Unresolved placeholder: parameter '{param}' is set to "
                f"PLACEHOLDER on artifact '{result_id}', and no artifact "
                f"closer to the claim in the value-flow chain has pinned "
                f"it to a real upstream artifact. The claim has not "
                f"declared this parameter under `varied_parameters` or "
                f"`acknowledged_placeholders`, so the placeholder leaks "
                f"into the claim."
            ))

    # Phase 3: recurse upstream via real-ref sources only.
    # Each branch sees `seen_real_for_param | newly_pinned` — the parameters
    # this artifact pinned are now resolved for any deeper placeholder of
    # the same name.
    seen_below = seen_real_for_param | newly_pinned

    upstream_ids: Set[str] = set()
    for source in sources.values():
        if isinstance(source, list):
            upstream_ids.update(
                s for s in source
                if isinstance(s, str) and s and s != PLACEHOLDER_REF
            )
        elif isinstance(source, str) and source and source != PLACEHOLDER_REF:
            upstream_ids.add(source)

    for parent_id in upstream_ids:
        out.extend(_walk_placeholders_for_claim(
            result_id=parent_id,
            seen_real_for_param=seen_below,
            varied_parameters=varied_parameters,
            acknowledged_placeholders=acknowledged_placeholders,
            visited=visited,
        ))

    _dbg(
        f"_walk_placeholders_for_claim: result_id={result_id!r} "
        f"newly_pinned={sorted(newly_pinned)} "
        f"n_unresolved_here={sum(1 for u in out if u[0] == result_id)} "
        f"n_unresolved_total={len(out)}"
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
    _dbg(
        f"_call_param_judge_llm: INVOKE param={parameter_name!r} "
        f"value={parameter_value!r} target={target_quantity!r} "
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

    # Per-spec validation: varied_parameters and acknowledged_placeholders
    # are mutually exclusive scopes. A parameter cannot simultaneously be
    # being varied and being held at a placeholder for the same claim.
    for spec in parsed_specs:
        overlap = set(spec.varied_parameters) & set(spec.acknowledged_placeholders)
        if overlap:
            _dbg(
                f"generate_structured_report: overlap between varied_parameters "
                f"and acknowledged_placeholders for quantity={spec.quantity_name!r}: "
                f"{sorted(overlap)}"
            )
            raise ValueError(
                f"Quantity '{spec.quantity_name}': parameters "
                f"{sorted(overlap)} appear in both `varied_parameters` and "
                "`acknowledged_placeholders`. These are mutually exclusive — "
                "a parameter cannot be both intentionally swept AND held at "
                "a placeholder for the same claim. Place each parameter in "
                "exactly one list."
            )
    _dbg("generate_structured_report: varied/acknowledged overlap check passed")

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
            "acknowledged_placeholders": spec.acknowledged_placeholders,
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
            acknowledged_placeholders=spec.acknowledged_placeholders,
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
    artifact_summary = _summarize_artifact(artifact)
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

        # Detect explicit placeholder declaration. The agent declares a
        # placeholder by passing PLACEHOLDER_REF as the *_ref / *_w_ref
        # value at the producing tool call. For list-shaped sources, any
        # PLACEHOLDER_REF entry triggers the placeholder branch for the
        # whole parameter (partial placeholders are not modelled).
        is_placeholder = source == PLACEHOLDER_REF or (
            isinstance(source, list) and PLACEHOLDER_REF in source
        )

        # Match either bare `<param>` (applies to every tool) or scoped
        # `<tool>.<param>` (applies only to this tool).
        is_sensitive = (
            param_name in sensitive_parameters
            or f"{artifact.tool_name}.{param_name}" in sensitive_parameters
        )        
        is_varied = param_name in varied_parameters
        if is_sensitive and not is_varied and is_placeholder:
            _branch_label = "R1-placeholder"
        elif is_sensitive and not is_varied:
            _branch_label = "R1"
        elif is_sensitive and is_varied:
            _branch_label = "R2"
        else:
            _branch_label = "R3"
        _dbg(
            f"verify_artifact_parameterization: branch decision for "
            f"{param_name!r}: is_sensitive={is_sensitive} is_varied={is_varied} "
            f"is_placeholder={is_placeholder} is_info_tool={is_info_tool} "
            f"has_reason={bool(reason.strip())} -> rule={_branch_label}"
        )

        # R1-placeholder — sensitive, not varied, agent acknowledged
        # placeholder. No upstream source required; judgment is on whether
        # the chosen placeholder is a reasonable provisional default.
        if is_sensitive and not is_varied and is_placeholder:
            _dbg(
                f"verify_artifact_parameterization: R1-placeholder path — "
                f"calling judge for {param_name!r} (no source recovery)"
            )
            judgement = _call_param_judge_llm(
                overall_goal=overall_goal,
                target_quantity=target_quantity,
                varied_parameters=varied_parameters,
                sensitive_parameters=sensitive_parameters,
                artifact_summary=artifact_summary,
                parameter_name=param_name, parameter_value=param_value,
                reason=reason, source_artifact_summary=None,
                rule_to_apply=R1_PLACEHOLDER_RULE, judge=judge,
            )
            _dbg(
                f"verify_artifact_parameterization: judge verdict for "
                f"{param_name!r} -> {judgement['verdict']!r}"
            )
            checks.append(ParamCheckResult(
                parameter_name=param_name, parameter_value=param_value,
                verdict=judgement["verdict"],
                rule_applied=R1_PLACEHOLDER_RULE,
                source_result_id=source,
                reasoning=judgement["reasoning"],
            ))
            overall = _fold_verdict(overall, judgement["verdict"])
            continue

        # R1 — sensitive and not varied: must be sourced.
        if is_sensitive and not is_varied:
            if source is None:
                _dbg(
                    f"verify_artifact_parameterization: R1 — no source "
                    f"recorded for {param_name!r} -> FAIL"
                )
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

            _dbg(
                f"verify_artifact_parameterization: R1 path — calling "
                f"_verify_sourced_param for {param_name!r}"
            )
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

        # Belt-and-braces source check. If a source is declared, verify it
        # element-by-element (list shape) or directly (scalar shape).
        source_summary: Optional[Any] = None
        if source is not None:
            _dbg(
                f"verify_artifact_parameterization: optional source declared "
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
        _dbg(
            f"verify_artifact_parameterization: judge verdict for "
            f"{param_name!r} -> {judgement['verdict']!r}"
        )
        checks.append(ParamCheckResult(
            parameter_name=param_name, parameter_value=param_value,
            verdict=judgement["verdict"], rule_applied=rule,
            source_result_id=source,
            reasoning=judgement["reasoning"],
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
    _dbg(
        f"_verify_sourced_param: ENTER param={param_name!r} "
        f"value={param_value!r} source={source!r}"
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
        )], "fail"

    # Value-match each element.
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
                verdict="fail", rule_applied=rule,
                source_result_id=ref, reasoning=msg,
            )], "fail"

    # Single judge call over the collective source set.
    if is_list:
        source_summary: Any = [
            _summarize_artifact(_get_artifact(ref)) for ref in sources
        ]
        _dbg(
            f"_verify_sourced_param: built list source_summary "
            f"n_entries={len(source_summary)}"
        )
    else:
        source_summary = _summarize_artifact(_get_artifact(sources[0]))
        _dbg("_verify_sourced_param: built scalar source_summary")

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
    _dbg(
        f"_verify_sourced_param: RETURN param={param_name!r} "
        f"verdict={judgement['verdict']!r}"
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
                verdict="fail", rule_applied=rule,
                source_result_id=ref, reasoning=msg,
            )], "fail", None

    if is_list:
        summary: Any = [
            (None if ref == PLACEHOLDER_REF
             else _summarize_artifact(_get_artifact(ref)))
            for ref in sources
        ]
        _dbg(
            f"_verify_optional_source_match: PASS list-shape "
            f"n_entries={len(summary)} "
            f"n_placeholders={sum(1 for s in summary if s is None)}"
        )
        return [], "pass", summary
    if sources[0] == PLACEHOLDER_REF:
        _dbg("_verify_optional_source_match: PASS scalar-shape (placeholder)")
        return [], "pass", None
    summary = _summarize_artifact(_get_artifact(sources[0]))
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
    visited: Set[str],
    artifact_results: List[ArtifactVerificationResult],
    issues: List[ReportVerificationIssue],
    judge,
    depth: int = 0,
) -> None:
    indent = "  " * depth
    _dbg(
        f"{indent}_verify_one_artifact_recursive: ENTER depth={depth} "
        f"result_id={result_id!r} target_quantity={target_quantity!r}"
    )
    if result_id in visited:
        _dbg(
            f"{indent}_verify_one_artifact_recursive: already visited "
            f"{result_id!r} — skipping"
        )
        return
    visited.add(result_id)

    artifact = _get_artifact(result_id)

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
            )
            children_checked.append(child_id)
            _dbg(
                f"{indent}_verify_one_artifact_recursive: returned from child "
                f"{child_id!r}"
            )
        except Exception as e:                        # noqa: BLE001
            children_failed += 1
            _dbg(
                f"{indent}_verify_one_artifact_recursive: child {child_id!r} "
                f"raised {type(e).__name__}: {e}"
            )
            issues.append(ReportVerificationIssue(
                level="artifact",
                location=f"result_id={result_id} -> child={child_id}",
                verdict="fail",
                message=str(e),
            ))

    _dbg(
        f"{indent}_verify_one_artifact_recursive: post-order summary for "
        f"{result_id!r} — children_total={children_total} "
        f"children_checked={len(children_checked)} "
        f"children_failed={children_failed}"
    )

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
            f"{indent}_verify_one_artifact_recursive: appending issue for "
            f"{result_id!r} verdict={local_result.overall_verdict!r}"
        )
        
        nonpassMSG = f"{local_result.summary}\n"
        for check in local_result.checks:
            if check.verdict != "pass":
                nonpassMSG += (
                    f"- Parameter '{check.parameter_name}': "
                    f"{check.verdict.upper()} (reason: {check.reasoning})\n"
                )
        nonpassMSG += "--------------------------------------\n"
        
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
    all_visited: Set[str] = set()

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
                level="claim",
                location=f"quantity={claim.quantity_name}",
                verdict="fail",
                message=str(e),
            ))
            continue

        ok, msg = CANVAS.verify_artifact(claim.value, claim.result_id)
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} "
            f"value-match -> ok={ok} msg={msg!r}"
        )
        if not ok:
            issues.append(ReportVerificationIssue(
                level="claim",
                location=f"quantity={claim.quantity_name}, result_id={claim.result_id}",
                verdict="fail",
                message=msg,
            ))
            continue

        # Placeholder-resolution check. Walk the value-flow chain (only via
        # parent_result_ids_w_args) and surface any placeholder source that
        # is not resolved by:
        #   (a) a real ref pinning the same parameter closer to the claim,
        #   (b) the parameter being in claim.varied_parameters, or
        #   (c) the parameter being in claim.acknowledged_placeholders.
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"running placeholder-resolution walk "
            f"(varied={list(claim.varied_parameters)} "
            f"acknowledged={list(claim.acknowledged_placeholders)})"
        )
        unresolved = _walk_placeholders_for_claim(
            result_id=claim.result_id,
            seen_real_for_param=set(),
            varied_parameters=set(claim.varied_parameters),
            acknowledged_placeholders=set(claim.acknowledged_placeholders),
            visited=set(),
        )
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"placeholder walk found n={len(unresolved)} unresolved entries"
        )
        for artifact_id, param_name, source_repr, message in unresolved:
            issues.append(ReportVerificationIssue(
                level="claim",
                location=(
                    f"quantity={claim.quantity_name}, "
                    f"artifact={artifact_id}, param={param_name}"
                ),
                verdict="fail",
                message=message,
            ))

        # Per-claim visited scope: the same upstream artifact reachable from
        # two different claims gets verified twice, under each claim's own
        # `varied_parameters`.
        visited: Set[str] = set()
        visited_before = len(visited)
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"starting recursive descent (visited size before={visited_before})"
        )
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
        _dbg(
            f"verify_structured_report: claim {claim.quantity_name!r} — "
            f"recursive descent complete (blast radius={len(visited)} "
            f"artifacts)"
        )
        all_visited.update(visited)

    if any(i.verdict == "fail" for i in issues):
        overall = "fail"
    elif any(i.verdict == "warning" for i in issues):
        overall = "warning"
    else:
        overall = "pass"

    # Issue breakdown by level for log readability.
    level_counts: Dict[str, int] = {}
    for i in issues:
        level_counts[i.level] = level_counts.get(i.level, 0) + 1
    n_pass = sum(1 for r in artifact_results if r.overall_verdict == "pass")
    n_warn = sum(1 for r in artifact_results if r.overall_verdict == "warning")
    n_fail = sum(1 for r in artifact_results if r.overall_verdict == "fail")
    _dbg(
        f"verify_structured_report: AGGREGATE overall={overall!r} "
        f"n_artifacts_checked={len(all_visited)} "
        f"artifact_verdicts(pass/warn/fail)={n_pass}/{n_warn}/{n_fail} "
        f"issues_by_level={level_counts}"
    )

    return ReportVerificationResult(
        overall_verdict=overall,
        checked_result_ids=sorted(all_visited),
        issues=issues,
        artifact_results=artifact_results,
    ).model_dump()