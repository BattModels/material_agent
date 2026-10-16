"""Agent tools for the directional-coupler screening study.

Three groups:
  * generic memory tools -- CANVAS (inspect/read/write), reports, artifact search,
    elapsed time, arXiv literature search;
  * EXPLOG tools -- query/read the experiment log, disposition read/write;
  * FEM tools -- material library, enter a design, run the coupled-mode analysis
    (in-process, returns instantly; see src/fem_calculator.py).

Every tool registers its output on CANVAS and hands back an 8-character reference
id, which downstream tools require as provenance for their inputs.
"""
import asyncio
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from pydantic import StrictFloat, StrictStr

from src import var
from src.myCANVAS import CANVAS, ListedArtifact
from src.utils import (
    Filter,
    SortSpec,
    df_query,
    is_floatable,
    util_find_all_substring_spans,
    util_numeric_matches_in_region,
)
from src.fem_explog import (
    EXPLOG,
    DESIGN_COLUMNS,
    HIDDEN_CANDIDATE_COLUMNS,
    METRIC_COLUMNS,
    RESULT_COLUMNS,
    derived_metrics,
)
from src.fem_materials import (
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
    default_core_height_um,
    known_materials,
    materials_table,
    refractive_index,
)
from src.fem_calculator import (
    CalculationError,
    CalculationTimeout,
    DesignNotViable,
    compute_coupled_modes,
    solve_in_subprocess,
)
from src.disposition_messages import (
    evaluate_terminal_tag_gate,
    format_get_disposition,
    format_needs_disposition_reminder,
    format_update_disposition,
)

try:
    from ursa.agents import ArxivAgent
except Exception:  # ursa is optional; the arXiv tool degrades to a clear message
    ArxivAgent = None

import asyncio  # If needed for defining async_func

async def _arXiv_search(arxiv_search_query, context):  # Your async operation
    if ArxivAgent is None:
        return ("arXiv search is unavailable in this environment (the 'ursa' package is not "
                "installed). Proceed from your own knowledge and say so in your notes.")
    config = var.OTHER_GLOBAL_VARIABLES
    ursaWorkspace = Path(os.path.join(var.my_WORKING_DIRECTORY, "ursa_workspace"))
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
    agent = ArxivAgent(llm=llm, process_images=False, max_results=5, workspace=ursaWorkspace)
    result = await agent.ainvoke(
        arxiv_search_query=arxiv_search_query, 
        context=context + "When writing chemical formulas, never use Unicode subscripts or superscripts."
    )
    os.makedirs(ursaWorkspace/"arxiv_papers_used", exist_ok=True)
    # move all files under ursaWorkspace / "arxiv_papers" into ursaWorkspace/"arxiv_papers_used"
    for file in os.listdir(ursaWorkspace/"arxiv_papers"):
        os.rename(ursaWorkspace/"arxiv_papers"/file, ursaWorkspace/"arxiv_papers_used"/file)
    
    return result["final_summary"]

# FOR DEMO 
# async def _arXiv_search(arxiv_search_query, context):  # Your async operation
#     config = var.OTHER_GLOBAL_VARIABLES
#     llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=config['ANTHROPIC_API_KEY'],temperature=0.0)
#     result = await llm.ainvoke(
#         [("user", f"Please answer the question: {arxiv_search_query} with context: {context}.")]
#     )
    
#     return result.content


@tool
def arXiv_search(
    arxiv_search_query: Annotated[str, "Keyword search query used to retrieve papers from arXiv."],
    context: Annotated[str, "The specific question or focus that the summary should address."]
    ) -> str:
    """
    Perform an arXiv search for papers with a given arxiv_search_query and context and provide a summary.
    Only 10 papers will be considered in the search. If you want to consider more papers, you will need 
    to refine your search arguments.
    """
    # Only 10 papers will be considered in the search. If you want to consider more papers, you will need to refine
    # your search arguments.

    result = asyncio.run(_arXiv_search(arxiv_search_query, context))

    id = CANVAS.register_tool_output(
        tool_name="arXiv_search",
        args={
            "arxiv_search_query": arxiv_search_query,
            "context": context,
        },
        value=result,
        description="Summary of arXiv search results for the query: {arxiv_search_query} with context: {context}",
        parent_result_ids=[],
        metadata={
            "arxiv_search_query": arxiv_search_query,
            "context": context,
        }
    )
    
    return f"{result}\nLiterature_result_ID={id}. Please extract the numerical values if you need to use numerical values from the result to make decisions or conclusions."

def format_time_budget(elapsed_seconds, budget_seconds) -> str:
    """Elapsed / remaining / percent-used prose for check_time.

    Pure, so it is testable without a clock. Nothing here ENFORCES the budget --
    an over-budget study keeps running; the message just tells the agent to stop
    opening new lines of work and deliver.
    """
    elapsed = max(0.0, float(elapsed_seconds or 0.0))
    el = str(timedelta(seconds=int(elapsed)))
    if not budget_seconds or float(budget_seconds) <= 0:
        return (f"Total time elapsed since project start: {el}. "
                f"No time budget is set for this study.")

    budget = float(budget_seconds)
    bd = str(timedelta(seconds=int(budget)))
    pct = 100.0 * elapsed / budget
    remaining = budget - elapsed
    head = (f"Total time elapsed since project start: {el} of the {bd} budget "
            f"({pct:.1f}% used)")

    if remaining < 0:
        return (f"{head}; OVER BUDGET by {timedelta(seconds=int(-remaining))}. "
                f"Do NOT start new designs or calculations: finish analysing the results "
                f"you already have, record their dispositions, and deliver the final "
                f"report now.")

    msg = f"{head}; {timedelta(seconds=int(remaining))} remaining."
    warn_at = float(getattr(var, "TIME_BUDGET_WARN_FRACTION", 0.8)) * 100.0
    if pct >= warn_at:
        msg += (" The budget is nearly exhausted: stop opening new lines of "
                "investigation, finish analysing what you already have, and get the "
                "final report written.")
    return msg


@tool
def check_time():
    """
    Check how much of the study's time budget has been used.

    Reports the time elapsed since the project start, the budget, the percentage
    used and the time remaining. Once the budget is nearly exhausted -- or already
    exceeded -- the message says so and tells you to wrap up: finish analysing the
    results you already have and report back, rather than starting new calculations.
    """
    start = var.startTime if isinstance(var.startTime, (int, float)) else None
    elapsed = (time.time() - start) if start else 0.0
    return format_time_budget(elapsed, getattr(var, "STUDY_BUDGET_SECONDS", None))

@tool
def inspect_my_canvas():
    """Inspect the working canvas to get available keys"""
    # get all keys in myCANVAS and return them as a list [key1, key2, ...]
    # print(CANVAS)
    _ = CANVAS.register_tool_output(
        tool_name="inspect_my_canvas",
        args={},
        value="",
        description=f"Inspecting the working canvas to get available keys.",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.inspect()

@tool
def read_my_canvas(key: Annotated[str, "key"]):
    """Read a value from the working canvas"""
    # read a value from myCANVAS given a key
    _ = CANVAS.register_tool_output(
        tool_name="read_my_canvas",
        args={
            "key": key,
        },
        value="",
        description=f"Reading value from canvas with key '{key}'",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.read(key)

@tool
def write_my_canvas(key: Annotated[str, "key"],
                    value: Annotated[Any, "value"],
                    overwrite: Annotated[bool, "True to overwrite if key already exist. only set to True if you are certain you want to overwrite the existing value"] = False):
    """Write a value to the working canvas. If the key already exists, it will not overwrite unless specified."""
    # write a value to myCANVAS given a key and a value
    
    _ = CANVAS.register_tool_output(
        tool_name="write_my_canvas",
        args={
            "key": key,
            "value": value,
            "overwrite": overwrite,
        },
        value=value,
        description=f"Writing value to canvas with key '{key}'",
        parent_result_ids=[],
        metadata={},
    )
    
    return CANVAS.write(key, value, overwrite)

@tool
def write_report(
    report: Annotated[str, "Intermediate/final report content in markdown format."],
    report_name: Annotated[str, "Name of the report."],
    report_gist: Annotated[str, "A short one-sentence, plain-language summary of what this report covers and concludes (e.g. 'Screened 12 Ru/Ir oxides; IrO2 rutile is the most promising candidate so far, Co3O4 abandoned'). Used verbatim to label the compacted step history once these steps roll out of context, so make it self-contained and specific."],
    ref_list: Annotated[List[str], "Please provide the list of all 8-char reference IDs that you used when writing this report"],
    ):
    """Note down your report on CANVAS and let the supervisor know you've generated a report"""
    
    for ref in ref_list:
        art = CANVAS.get_artifact(ref)
        if art is None:
            return f"Error: Reference ID {ref} not found in CANVAS. Please check the canvas and try again, or regenerate the reference result with corresponding tool."
    
    outStr = CANVAS.write(report_name, report, False)
    if outStr == f"Key '{report_name}' already exists. Please choose a different key. If you want to overwrite the value, set the 'overwrite' flag to True.":
        # The name collided, so CANVAS refused the write and NO report was stored.
        # Bail out here. This used to fall through and still register an artifact,
        # handing back a Report_ID for a report that does not exist -- so the agent
        # believed it had reported, while var.reportName stayed unset and the
        # context compaction that a report is supposed to trigger silently never
        # happened. Fail loudly instead.
        return (
            f"ERROR: no report was written. The name '{report_name}' is already taken, and "
            f"reports are never overwritten. Call write_report again with a NEW, unique "
            f"report_name (e.g. append the round number). Nothing has been recorded yet."
        )

    var.reportName = report_name

    id = CANVAS.register_tool_output(
        tool_name="write_report",
        args={
            "report": report,
            "report_name": report_name,
            "report_gist": report_gist,
        },
        value=report,
        description=f"Writing report to canvas with key '{report_name}'",
        parent_result_ids=ref_list,
        metadata={},
    )
    # Handed to worker_agent_node together with reportName so the compaction digest
    # can cite the report's artifact id and label the compacted steps with the
    # worker's own one-line gist, not just the canvas key.
    var.reportId = id
    var.reportGist = report_gist

    return outStr + f"\nReport_ID: {id}. Please refer to this ID if you want to reference this report later or use the information in the report for further analysis or decision making."

# @tool
# def write_my_canvas(key: Annotated[str, "key"],
#                     value: Annotated[Any, "value"],
#                     entry_type: Annotated[Literal["note", "numerical_result"], "entry type. 'note' is for general note or text, not allowed to be use to generate final report. 'numerical_result' is verifiable with tools output and will be verified. Must provide source_result_id if entry_type is 'numerical_result'."],
#                     overwrite: Annotated[bool, "True to overwrite if key already exist. only set to True if you are certain you want to overwrite the existing value"] = False,
#                     source_result_id: Annotated[Optional[str], "the result_id of the tool output that this numerical canvas entry is based on."] = None
#                     ):
#     """Write a value to the working canvas. If the key already exists, it will not overwrite unless specified."""
#     # write a value to myCANVAS given a key and a value
#     if entry_type == "numerical_result":
#         assert source_result_id is not None, "source_result_id must be provided for numerical_result entry type."
#     return CANVAS.write(
#         key=key,
#         value=value,
#         entry_type=entry_type,
#         overwrite=overwrite,
#         source_result_id=source_result_id
#         )
    
# @tool
# def register_parameter_choice_by_LLM_agent(
#     value: Annotated[float, "The value of the parameter chosen by the LLM agent. It should be a numeric value that can be used for calculations."],
#     reason: Annotated[str, "A brief reason of why a certain parameter was chosen to be this value. You must clarify the unit of the value."]
# ):
#     """Register the choice of a specific parameter by the LLM agent, along with the reason for the choice with unit clarified. """

    
#     result_id = CANVAS.register_tool_output(
#         tool_name="register_parameter_choice_by_LLM_agent",
#         value=value,
#         numerical_result=True,
#         parent_result_ids=[],
#         metadata={
#             "reason": reason,
#         },
#     )
    
#     return f"value: {value} is registered with result_id: {result_id}."


def _norm_query_term(term: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single query term (str, float, or list of str) into a
    matcher dict. Returns None if the term is empty / not searchable.

    Matcher dict shape:
      {"kind": "str" | "float" | "list_str",
       "value": <original term>,
       "str_form": <lowercase string>,    # for "str" only
       "float_value": <float>,            # for "float" only
       "list_forms": [<lowercase str>, ...]}   # for "list_str" only

    For "float" matching against a string-valued category, we DO NOT
    treat the query as a substring — instead we extract numeric tokens
    from the category text via the shared util_numeric_matches_in_region
    helper, parse each, and compare numerically with tolerance. This
    avoids the classic false positive where '60' matches inside '160'.
    """
    if term is None:
        return None
    if isinstance(term, bool):
        # bool is a subclass of int; reject explicitly to avoid surprises
        print("BOOL!")
        return None
    if isinstance(term, (int, float)):
        return {
            "kind": "float",
            "value": term,
            "float_value": float(term),
        }
    if isinstance(term, str):
        if not term.strip():
            return None
        return {
            "kind": "str",
            "value": term,
            "str_form": term.lower(),
        }
    if isinstance(term, list):
        cleaned = [s.lower() for s in term if isinstance(s, str) and s.strip()]
        if not cleaned:
            return None
        return {
            "kind": "list_str",
            "value": term,
            "list_forms": cleaned,
        }
    return None


def _atoms_from_args(args: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """Decompose an args dict into (string_atoms, numeric_atoms).

    - For each `(key, value)` pair we always include `"<key>=<value_repr>"`
      as a string atom so list-of-strings queries (which need all terms
      in one atom) can match key+value pairs naturally (e.g. searching
      `["ecutwfc", "80"]` against `args["ecutwfc"]=80` finds the atom
      "ecutwfc=80").
    - Numeric values (int / float, excluding bool) ALSO contribute
      directly to numeric_atoms so float queries match by numeric
      equality without going through stringification or regex
      tokenization (avoids edge cases like numpy scalars whose repr
      differs from the standard form).
    - When a value is itself a list/tuple, we descend ONE level and
      apply the same rule per element (string element → its own
      string atom; numeric element → numeric atom; otherwise → repr
      as string atom).
    """
    string_atoms: List[str] = []
    numeric_atoms: List[float] = []
    if not args:
        return string_atoms, numeric_atoms

    def _ingest_value(value: Any) -> None:
        if isinstance(value, bool):
            string_atoms.append(repr(value))
            return
        if isinstance(value, (int, float)):
            numeric_atoms.append(float(value))
            return
        if isinstance(value, str):
            string_atoms.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, bool):
                    string_atoms.append(repr(item))
                elif isinstance(item, (int, float)):
                    numeric_atoms.append(float(item))
                elif isinstance(item, str):
                    string_atoms.append(item)
                else:
                    string_atoms.append(repr(item))
            return
        string_atoms.append(repr(value))

    for k, v in args.items():
        # Pair-atom keeps key and value together for list-of-strings
        # queries that need both terms in one atom.
        string_atoms.append(f"{k}={v!r}")
        # And the value (or its elements) contribute independently for
        # numeric matching and for individual-element string matching.
        _ingest_value(v)
    return string_atoms, numeric_atoms


def _category_atoms(
    artifact: Any, category: str
) -> Tuple[List[str], List[float]]:
    """Return (string_atoms, numeric_atoms) for the requested category
    on `artifact`. String matching iterates string_atoms; numeric
    matching checks numeric_atoms directly and (for text categories)
    also tokenizes inside each string atom via the regex helper.

    Behaviors per category:
      - tool_name / description / context: one string atom (the field
        itself). Numeric matching also tokenizes inside the text.
      - args: see `_atoms_from_args` — key=value pair atoms plus
        per-value atoms (numeric values go to numeric_atoms;
        list-valued args expand one level).
      - value: if numeric, one numeric atom. If string, one string
        atom. If list/tuple, each element is an atom of the
        appropriate kind. If anything else (dict, custom object),
        one string atom (its repr).
      - Note: this function is NEVER called on a ListedArtifact, so
        `value` will never be the nested-artifact-list case.

    Missing or empty categories yield ([], []).
    """
    if category == "tool_name":
        s = str(getattr(artifact, "tool_name", "") or "")
        return ([s] if s else [], [])
    if category == "description":
        s = str(getattr(artifact, "description", "") or "")
        return ([s] if s else [], [])
    if category == "context":
        s = str(getattr(artifact, "context", "") or "")
        return ([s] if s else [], [])
    if category == "args":
        return _atoms_from_args(getattr(artifact, "args", {}) or {})
    if category == "value":
        v = getattr(artifact, "value", None)
        if v is None:
            return ([], [])
        if isinstance(v, bool):
            return ([repr(v)], [])
        if isinstance(v, (int, float)):
            return ([], [float(v)])
        if isinstance(v, str):
            return ([v], [])
        if isinstance(v, (list, tuple)):
            string_atoms: List[str] = []
            numeric_atoms: List[float] = []
            for item in v:
                if isinstance(item, bool):
                    string_atoms.append(repr(item))
                elif isinstance(item, (int, float)):
                    numeric_atoms.append(float(item))
                elif isinstance(item, str):
                    string_atoms.append(item)
                else:
                    string_atoms.append(repr(item))
            return (string_atoms, numeric_atoms)
        if isinstance(v, dict):
            # e.g. run_coupled_mode_analysis stores {result_key: float, metric: float}:
            # index it like args, so "excess_loss_dB=0.13" and the bare number both match.
            return _atoms_from_args(v)
        return ([repr(v)], [])
    return ([], [])


def _matches_category(
    matcher: Dict[str, Any],
    string_atoms: List[str],
    numeric_atoms: List[float],
) -> bool:
    """Apply a normalized matcher to one category's atom view.

    String matching: at least one string_atom (lowercased) contains
    the query.

    List-of-strings matching: at least one string_atom (lowercased)
    contains EVERY listed term. The terms must all be in the same
    atom — not split across atoms — so searching `['ecutwfc', '80']`
    requires a single atom like "ecutwfc=80" containing both terms.

    Float matching: numeric_atoms checked first by equality with
    tolerance 1e-6. If no numeric match, each string_atom is also
    tokenized via the regex helper so numbers embedded in narrative
    text ("Energy from ecutwfc=80 calculation") are still found,
    while substrings like '60' inside '160' are correctly rejected
    by the regex lookarounds.
    """
    lowered: List[str] = [s.lower() for s in string_atoms]
    if matcher["kind"] == "str":
        target = matcher["str_form"]
        return any(target in a for a in lowered)
    if matcher["kind"] == "list_str":
        terms = matcher["list_forms"]
        return any(all(t in a for t in terms) for a in lowered)
    if matcher["kind"] == "float":
        target = matcher["float_value"]
        tol = 1e-6
        # Direct numeric equality on the category's numeric atoms.
        for n in numeric_atoms:
            if abs(n - target) <= tol:
                return True
        # Regex-tokenized numeric matching inside string atoms (for
        # numbers embedded in description / context / etc).
        for atom in string_atoms:
            if not atom:
                continue
            try:
                hits = util_numeric_matches_in_region(
                    atom, 0, len(atom), target, tol
                )
                if hits:
                    return True
            except Exception:                                   # noqa: BLE001
                # Defensive: if the helper raises for any reason, skip
                # this atom and move on.
                continue
        return False
    return False


_SEARCHABLE_CATEGORIES = ("tool_name", "args", "value", "description", "context")


def _build_result_dict(
    artifact: Any,
    parent_result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the per-result dict surfaced to the agent.

    When the match is on a component of a ListedArtifact, pass
    `parent_result_id` so the surfaced result_id is the parent's
    (the agent uses the parent's id downstream and does not care
    which specific component matched).
    """
    v_preview = repr(getattr(artifact, "value", None))
    if len(v_preview) > 100:
        v_preview = v_preview[:97] + "..."
    rid = parent_result_id if parent_result_id is not None else getattr(
        artifact, "result_id", ""
    )
    return {
        "result_id": rid,
        "tool_name": getattr(artifact, "tool_name", ""),
        "value_preview": v_preview,
        "args": getattr(artifact, "args", {}) or {},
        "description": getattr(artifact, "description", ""),
        "context": getattr(artifact, "context", ""),
    }


@tool
def search_artifacts(
    query: Annotated[
        Optional[Union[StrictStr, StrictFloat, List[StrictStr]]],
        "Search query: a string (case-insensitive substring), a float "
        "(numeric match with 1e-6 tolerance; word-boundary aware — "
        "60.0 won't match the '60' inside '160'), or a list of "
        "strings (all terms must appear in the SAME atom). Matches if "
        "any category of the artifact satisfies it (if a list of string is specified"
        "that one category must satisfies all element in the list)."
        "Pass None to use filters only.",
    ],
    filters: Annotated[
        Optional[Dict[str, Union[StrictStr, StrictFloat, List[StrictStr]]]],
        "Per-category AND filters. Keys: 'tool_name', 'args', 'value', "
        "'description', 'context'. Each value matched the same way as "
        "`query`. Pass None to use `query` only. Combinable with "
        "`query` (both must match).",
    ],
    order: Annotated[
        Literal["ascending", "descending"],
        "Result order by creation timestamp. 'descending' = newest "
        "first (typical); 'ascending' = oldest first.",
    ],
):
    """
    Find a past artifact's result_id by content. 
    
    Examples:
      - query="some_key_word" → substring across any category
      - query=80.0 → numeric match (value, args including nested
        lists, or numbers embedded in description/context)
      - filters={"tool_name": "example_tool_name"} → all results generated with example_tool_name
      - query=["keyword1", "keyword12"], filters={"tool_name":
        "example_tool_name"} → all results generated with example_tool_name that have at least one category matches both terms: keyword1, keyword2
        

    Categories: tool_name, args (each key=value pair is one atom;
    numeric values also matched directly), value (per-element for
    lists), description, context.

    Returns a dict with `results`, each entry having: result_id,
    tool_name, value_preview, args, description, context.

    At least one of `query` or `filters` must be supplied.
    """
    max_results = 20
    # Normalize the broad query into a matcher.
    query_matcher = _norm_query_term(query) if query is not None else None

    # Normalize filters into per-category matchers.
    filter_matchers: Dict[str, Dict[str, Any]] = {}
    if filters:
        for cat, term in filters.items():
            if cat not in _SEARCHABLE_CATEGORIES:
                return (
                    f"Unknown filter category {cat!r}. Allowed categories: "
                    f"{list(_SEARCHABLE_CATEGORIES)}."
                )
            m = _norm_query_term(term)
            if m is None:
                return (
                    f"Filter for category {cat!r} is empty or has an "
                    f"unsupported type (got {term!r}). Supply a non-empty "
                    f"string, a float, or a non-empty list of strings."
                )
            filter_matchers[cat] = m
            
    
    print(f"query is {query!r} → matcher {query_matcher}")
    print(f"filters are {filters!r} → matchers {filter_matchers}")
    if query_matcher is None and not filter_matchers:
        return (
            "Empty search rejected. Supply at least one of `query` "
            "(broad search) or `filters` (per-category structured "
            "search). Use `inspect_my_canvas` to browse CANVAS keys, "
            "or `list_referenceable_inputs` to enumerate a specific "
            "artifact's input parameters."
        )

    if max_results <= 0:
        return f"max_results must be positive; got {max_results}."

    # Build sorted iteration order.
    registry = getattr(CANVAS, "result_registry", {}) or {}
    if not registry:
        return "Artifact registry is empty — no artifacts have been registered yet."

    items = list(registry.items())
    items.sort(
        key=lambda kv: getattr(kv[1], "timeStamp", 0.0),
        reverse=(order == "descending"),
    )

    results: List[Dict[str, Any]] = []
    total_matched_before_cap = 0

    def _check_artifact_level(art: Any) -> bool:
        """Apply query + filters to an artifact (or a list-component)
        and return True if it matches. Per-category atom views are
        computed once per artifact."""
        # Precompute category atom views once.
        cat_views: Dict[str, Tuple[List[str], List[float]]] = {
            c: _category_atoms(art, c) for c in _SEARCHABLE_CATEGORIES
        }
        # Broad query: at least one category must match.
        if query_matcher is not None:
            broad_hit = False
            for c in _SEARCHABLE_CATEGORIES:
                str_atoms, num_atoms = cat_views[c]
                if _matches_category(query_matcher, str_atoms, num_atoms):
                    broad_hit = True
                    break
            if not broad_hit:
                return False
        # Filters: every named category must match its filter.
        for c, m in filter_matchers.items():
            str_atoms, num_atoms = cat_views[c]
            if not _matches_category(m, str_atoms, num_atoms):
                return False
        return True

    for rid, artifact in items:
        if artifact is None:
            continue
        if isinstance(artifact, ListedArtifact):
            # ListedArtifact contributes ONLY through its components.
            # _category_atoms is never invoked on a ListedArtifact, so
            # the components' own fields (their own tool_name, args,
            # value, description, context) are what get matched. If no
            # component matches, this ListedArtifact contributes
            # nothing — even if the parent's own description/context
            # would have matched. This is by design: matches surface
            # at the granularity the agent can act on.
            for i, comp in enumerate(artifact.value or []):
                if not _check_artifact_level(comp):
                    continue
                total_matched_before_cap += 1
                if len(results) < max_results:
                    # Surface under the PARENT's result_id — the agent
                    # always references the parent ref when using this
                    # artifact downstream.
                    results.append(
                        _build_result_dict(
                            comp,
                            parent_result_id=artifact.result_id,
                        )
                    )
        else:
            # Non-ListedArtifact: standard parent-level match.
            if _check_artifact_level(artifact):
                total_matched_before_cap += 1
                if len(results) < max_results:
                    results.append(_build_result_dict(artifact))

    truncated = total_matched_before_cap > max_results

    out: Dict[str, Any] = {
        "n_matches": total_matched_before_cap,
        "n_returned": len(results),
        "order": order,
        "results": results,
    }
    if truncated:
        out["note"] = (
            f"More than max_results ({max_results}) matched; "
            f"{total_matched_before_cap - max_results} additional matches "
            f"are not shown. Refine `query` or `filters` to narrow the "
            f"search."
        )
    
    json.dump(out, sys.stdout, indent=2)    
    
    return out

##################################################################################################
##                                          DFT tools                                           ##
##################################################################################################

# @tool
# def get_my_WORKING_DIRECTORY() -> str:
#     """Get the working directory."""
#     return var.my_WORKING_DIRECTORY




##################################################################################################
# EXPLOG tools
##################################################################################################

def _candidates_view() -> pd.DataFrame:
    cdf = EXPLOG.relational_frame.candidates.df.copy()
    return cdf.drop(columns=list(HIDDEN_CANDIDATE_COLUMNS), errors="ignore")


@tool
def query_explog(
    table_name: Annotated[str, "Table to query: 'candidates' (one row per design, with its parameters and latest results) or 'processes' (one row per calculation)."],
    reason: Annotated[str, "Reason behind the query. Why are you using such filters and sort? What are you looking for?"],
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    start_idx: Annotated[int, "Row index to start reading from (inclusive). The tool always returns up to 50 rows from this position. Use 0 for the first page, 50 for the second, etc."] = 0,
) -> str:
    """Query the experiment log (EXPLOG) with optional filter and sort criteria.
    Returns the filtered table as a string with row index.

    candidates table (one row per coupler DESIGN):
        candidate_id (str, e.g. dc_0003; assigned in order of entry),
        gap_um, wg1_width_um, wg2_width_um, core_height_um, wavelength_um (Float64, the geometry and operating wavelength),
        core_material, clad_material (str), n_core, n_clad (Float64, the refractive indices actually used),
        reason_or_hypothesis (str, why this design was entered / what it tests),
        notes (str, any notes you've added),
        n_started, n_finalized (Int64, calculations run for this design and how many are finished -- they are equal
            because calculations return instantly; n_started == 0 means the design has not been computed yet),
        state (str, 'active' or 'failed'; 'failed' means every calculation attempted for this design failed),
        n_eff_wg1, n_eff_wg2, n_eff_symmetric, n_eff_antisymmetric, kappa_12_per_um, kappa_21_per_um,
            coupling_length_um, max_power_transfer (Float64, results of the latest completed analysis; <NA> until computed),
        coupling_length_deviation_um (Float64, |coupling_length_um - target coupling length|; SMALLER IS BETTER --
            the primary ranking metric, sort ascending),
        excess_loss_dB (Float64, -10 log10(max_power_transfer): the fraction of light that never crosses because the
            cores are detuned; 0 dB for a symmetric coupler, SMALLER IS BETTER),
        decision (str, the design's CURRENT disposition tag -- one of Abandon, Low priority, Medium priority,
            High priority, Sufficient; <NA> until first dispositioned),
        needs_disposition_update (boolean, True when the design has finished results not yet analyzed/dispositioned)
    processes table (one row per CALCULATION):
        process_id (int), candidate_id (str), job_type (str, always coupled_mode_analysis),
        status (str, completed or failed), processNote (str), error (str, the failure message if status is failed),
        n_eff_wg1, n_eff_wg2 (Float64, fundamental TE mode index of each waveguide in isolation),
        n_eff_symmetric, n_eff_antisymmetric (Float64, supermode indices of the coupled pair),
        kappa_12_per_um, kappa_21_per_um (Float64, coupling coefficients, 1/um),
        coupling_length_um (Float64, propagation length for maximum power transfer),
        max_power_transfer (Float64, peak fraction of power transferred, 0..1),
        coupling_length_deviation_um, excess_loss_dB (Float64, the same two ranking metrics as on the candidates table)
    """
    EXPLOG.update_log()
    if table_name == "candidates":
        df = _candidates_view()
    elif table_name == "processes":
        df = EXPLOG.relational_frame.processes.df.copy()
    else:
        return "table_name must be either 'candidates' or 'processes'"

    filtered = df_query(df, filters, sort)
    total_rows = len(filtered)
    end_idx = min(start_idx + 50, total_rows)
    page = filtered.iloc[start_idx:end_idx]
    footer = f"\nShowing rows {start_idx}-{max(end_idx - 1, 0)} of {total_rows} total."
    if end_idx < total_rows:
        footer += f" Call again with start_idx={end_idx} to see the next rows."
    result = page.to_string(index=True) + footer

    id = CANVAS.register_tool_output(
        tool_name="query_explog",
        args={"table_name": table_name, "reason": reason},
        value=result,
        description=f"Result of querying the EXPLOG {table_name} table with reason: {reason}, filters: {filters}, and sort: {sort}.",
        reasons={"reason": reason},
        parent_result_ids=[],
        metadata={"table_name": table_name, "reason": reason},
    )
    return f"{result}\nQuery_result_ID={id}. Please refer to this ID if you want to use the query result for further analysis or decision making."


@tool
def read_explog(
    candidate_id: Annotated[str, "Id of the design to read the experiment log for (e.g. dc_0003)."],
    reasons: Annotated[str, "Why are you interested in this design? What do you want to find out from the experiment log?"],
) -> str:
    """Get everything the experiment log holds about one design: its parameters and
    hypothesis, every calculation run for it with results, and its latest disposition.
    """
    EXPLOG.update_log()
    cdf = _candidates_view()
    row = cdf[cdf["candidate_id"] == candidate_id]
    if row.empty:
        return f"No candidate '{candidate_id}' exists in the experiment log."
    pdf = EXPLOG.relational_frame.processes.df
    procs = pdf[pdf["candidate_id"] == candidate_id]
    latest = EXPLOG.candidate_outstanding(candidate_id)["latest_disposition"]
    answer = (
        f"Candidate information:\n{row.T.to_string(header=False)}\n\n"
        f"Related processes information:\n{procs.to_string(index=False) if len(procs) else '(no calculations yet)'}\n"
    )
    if latest is not None:
        answer += (
            f"\nLatest disposition: Decision={latest.get('Decision')}; "
            f"Analysis_and_implications={latest.get('Summary')}; "
            f"Future_plan={latest.get('Future_plan')}\n"
        )
    id = CANVAS.register_tool_output(
        tool_name="read_explog",
        args={"candidate_id": candidate_id},
        value=answer,
        description=f"Summary of the experiment log for candidate {candidate_id} with reason: {reasons}.",
        reasons={"reasons": reasons},
        parent_result_ids=[],
        metadata={"candidate_id": candidate_id, "reasons": reasons},
    )
    return f"{answer}\nQuery_result_ID={id}. Please refer to this ID if you want to use this query result for further analysis or decision making."


def _outstanding_results_table(outstanding: dict) -> str:
    """Render the process rows that still need a disposition (results + the two
    ranking metrics), so the agent analyzes real numbers without a second query."""
    ids = sorted({int(i) for u in (outstanding.get("must_cover") or []) for i in u["ids"]})
    if not ids:
        return ""
    pdf = EXPLOG.relational_frame.processes.df
    rows = pdf[pdf["process_id"].isin(ids)].sort_values("process_id")
    cols = ["process_id", "status", *RESULT_COLUMNS, *METRIC_COLUMNS, "error"]
    cols = [c for c in cols if c in rows.columns]
    return (f"\nResults awaiting analysis (target coupling length "
            f"{EXPLOG.target_coupling_length_um:g} um; both metrics smaller is better):\n"
            + rows[cols].to_string(index=False))


@tool
def get_disposition_info(
    candidate_id: Annotated[str, "Id of the design whose disposition status you want to read."],
) -> str:
    """
    Return a design's disposition status: its latest disposition (if any) and the
    finished results that still need to be reviewed and analyzed for a disposition.

    You MUST call this for a candidate before update_disposition_info -- it unlocks
    the disposition write for that candidate.
    """
    res = EXPLOG.get_disposition_info(candidate_id)
    msg = format_get_disposition(candidate_id, res)
    msg += _outstanding_results_table(res)
    id = CANVAS.register_tool_output(
        tool_name="get_disposition_info",
        args={"candidate_id": candidate_id},
        value=msg,
        description=f"Outstanding disposition information for candidate {candidate_id}.",
        parent_result_ids=[],
        metadata={"candidate_id": candidate_id},
    )
    return f"{msg}\nMessage_ID={id}. Refer to this ID if you need to reference this review."


@tool
def update_disposition_info(
    candidate_id: Annotated[str, "Id of the design to disposition."],
    Analysis_and_implications: Annotated[str, "Review and analyze the design's current and historical finished results, then record what they show AND what they imply -- e.g. which hypothesis was confirmed or rejected. Interpret the numbers, do not just list them."],
    Analyzed_process_id: Annotated[list[int], "The finished process ids your Analysis_and_implications is based on. Get the outstanding ids from get_disposition_info first."],
    Future_plan: Annotated[str, "The hypothesis you want to test next in relation to this design and the specific calculation(s) that would test it (free text)."],
    Decision: Annotated[str, f"Where to take this design next. Must be exactly one of: {', '.join(var.DISPOSITION_DECISIONS)}."],
) -> str:
    """
    Record a disposition for a design: tie its finished results back to an
    Analysis_and_implications, a Future_plan (the hypothesis to test next), and a
    Decision.

    Call get_disposition_info(candidate_id) first to see what needs analyzing.
    The disposition is recorded only once every outstanding finished result is
    cited; otherwise the returned message explains why and how to fix it.

    Decisions: Abandon and Sufficient are TERMINAL (no further compute). A FAILED
    design may ONLY be Abandon. A rejected Decision leaves the disposition
    unrecorded and explains why.
    """
    cdf = EXPLOG.relational_frame.candidates.df
    crow = cdf.loc[cdf["candidate_id"] == candidate_id]
    gate_msg = None
    if not crow.empty:
        gate_msg = evaluate_terminal_tag_gate(
            decision=Decision,
            state=crow["state"].iloc[0],
            has_in_flight=False,  # calculations return instantly; nothing is ever in flight
            terminal_decisions=var.DISPOSITION_TERMINAL_DECISIONS,
            active_decisions=var.DISPOSITION_ACTIVE_DECISIONS,
        )
    if gate_msg is not None:
        msg, status = gate_msg, "gate_rejected"
    else:
        res = EXPLOG.update_disposition_info(
            candidate_id, Analysis_and_implications, Analyzed_process_id, Future_plan, Decision
        )
        msg = format_update_disposition(candidate_id, res, var.DISPOSITION_DECISIONS)
        status = res.get("status")
    id = CANVAS.register_tool_output(
        tool_name="update_disposition_info",
        args={
            "candidate_id": candidate_id,
            "Analysis_and_implications": Analysis_and_implications,
            "Analyzed_process_id": Analyzed_process_id,
            "Future_plan": Future_plan,
            "Decision": Decision,
        },
        value=msg,
        description=f"Disposition update for candidate {candidate_id} (status: {status}).",
        reasons={"Analysis_and_implications": Analysis_and_implications},
        parent_result_ids=[],
        metadata={"candidate_id": candidate_id, "status": status},
    )
    return f"{msg}\nMessage_ID={id}. Refer to this ID for this disposition update."


##################################################################################################
# FEM tools
##################################################################################################

@tool
def list_materials() -> str:
    """
    List the material library available for the study: each material's role
    (core or cladding), its refractive index at several wavelengths in the
    1.3-1.6 um window, and the default core height of its platform.
    Use it to choose core/cladding materials and to reason about index contrast.
    """
    table = materials_table()
    id = CANVAS.register_tool_output(
        tool_name="list_materials",
        args={},
        value=table,
        description="Material library: refractive index vs wavelength and default core heights.",
        parent_result_ids=[],
        metadata={},
    )
    return f"{table}\nReference_ID={id}. Refer to this ID when a design is motivated by this table."


def _resolve_index(material: str, wavelength_um: float, override: float | None, what: str) -> float:
    if override is not None:
        return float(override)
    if material not in known_materials() and material.lower() not in [m.lower() for m in known_materials()]:
        raise ValueError(
            f"Unknown {what} material '{material}'. Known materials: {known_materials()}. "
            f"Pass an explicit refractive index (n_{what}) to use a material outside the library."
        )
    return refractive_index(material, wavelength_um)


@tool
def enter_design_in_log(
    reason_or_hypothesis: Annotated[str, "Detailed reason and hypothesis for this design: what question it answers, and what result you expect. Used later for analysis and the report."],
    gap_um: Annotated[float, "Edge-to-edge gap between the two waveguide cores, in um."],
    wg1_width_um: Annotated[float, "Width of waveguide 1, in um."],
    wg2_width_um: Annotated[float, "Width of waveguide 2, in um. Equal to wg1_width_um for a symmetric coupler."],
    wavelength_um: Annotated[float, f"Operating (vacuum) wavelength in um, within {WAVELENGTH_MIN_UM}-{WAVELENGTH_MAX_UM}."],
    core_material: Annotated[str, "Core material name from the material library (see list_materials), e.g. 'Si', 'SiN'."],
    clad_material: Annotated[str, "Cladding material name from the material library, e.g. 'SiO2', 'Air'."],
    design_ref: Annotated[str, "Reference ID of the result that motivated this design (a list_materials table, an arXiv_search summary, a query_explog result, ...). Used for provenance."],
    core_height_um: Annotated[float | None, "Core (film) height in um. Defaults to the platform default for the core material."] = None,
    n_core: Annotated[float | None, "Explicit core refractive index. Overrides the library value; required if core_material is not in the library."] = None,
    n_clad: Annotated[float | None, "Explicit cladding refractive index. Overrides the library value; required if clad_material is not in the library."] = None,
    note: Annotated[str | None, "Any notes you want to add."] = None,
) -> str:
    """
    Enter a directional-coupler design (one parameter combination) into the
    experiment log (EXPLOG) so it can be computed with run_coupled_mode_analysis.

    The design is identified by its parameters; entering an identical design twice
    is refused and the existing candidate id is returned instead. A new design
    gets the next chronological id (dc_0001, dc_0002, ...).
    """
    if design_ref:
        if CANVAS.get_artifact(design_ref) is None:
            return f"Error: Reference ID {design_ref} not found in CANVAS."

    for name, v in (("gap_um", gap_um), ("wg1_width_um", wg1_width_um),
                    ("wg2_width_um", wg2_width_um), ("wavelength_um", wavelength_um)):
        if v is None or not float(v) > 0:
            return f"Error: {name} must be a positive number (got {v!r})."
    if not (WAVELENGTH_MIN_UM <= float(wavelength_um) <= WAVELENGTH_MAX_UM):
        return (f"Error: wavelength_um={wavelength_um} is outside the study window "
                f"{WAVELENGTH_MIN_UM}-{WAVELENGTH_MAX_UM} um.")

    try:
        n_core_val = _resolve_index(core_material, wavelength_um, n_core, "core")
        n_clad_val = _resolve_index(clad_material, wavelength_um, n_clad, "clad")
    except (KeyError, ValueError) as e:
        return f"Error: {e}"
    if not n_core_val > n_clad_val:
        return (f"Error: core index ({n_core_val:.4f}) must exceed cladding index "
                f"({n_clad_val:.4f}) for a guided mode.")
    if core_height_um is None:
        try:
            core_height_um = default_core_height_um(core_material)
        except KeyError:
            core_height_um = None
        if core_height_um is None:
            return "Error: core_height_um is required for this core material (no platform default)."
    if not float(core_height_um) > 0:
        return f"Error: core_height_um must be positive (got {core_height_um!r})."

    design = {
        "gap_um": float(gap_um),
        "wg1_width_um": float(wg1_width_um),
        "wg2_width_um": float(wg2_width_um),
        "core_height_um": float(core_height_um),
        "wavelength_um": float(wavelength_um),
        "core_material": core_material,
        "n_core": float(n_core_val),
        "clad_material": clad_material,
        "n_clad": float(n_clad_val),
    }
    try:
        cid = EXPLOG.add_candidate(design, reason_or_hypothesis=reason_or_hypothesis, notes=note)
    except ValueError as e:
        return f"Not entered: {e}"

    message = (
        f"Design {cid} added to the experiment log: "
        + ", ".join(f"{k}={design[k]}" for k in DESIGN_COLUMNS)
        + f". Reason: {reason_or_hypothesis}. It can now be computed with run_coupled_mode_analysis."
    )
    id = CANVAS.register_tool_output(
        tool_name="enter_design_in_log",
        args={"candidate_id": cid, **design, "reason_or_hypothesis": reason_or_hypothesis, "note": note},
        value=cid,
        description=f"Entry of design {cid} into the experiment log with reason: {reason_or_hypothesis}.",
        reasons={"reason_or_hypothesis": reason_or_hypothesis},
        parent_result_ids=[design_ref] if design_ref else [],
        metadata={"candidate_id": cid, **design, "design_ref": design_ref, "note": note},
    )
    return f"{message}\nMessage_ID={id}. Refer to this ID as the candidate_id_ref when you run the analysis for {cid}."


def _format_result(result: dict) -> str:
    metrics = derived_metrics(result, EXPLOG.target_coupling_length_um)
    lines = [f"  {k} = {float(result[k]):.6g}" for k in RESULT_COLUMNS]
    lines.append(f"  coupling_length_deviation_um = {metrics['coupling_length_deviation_um']:.6g}"
                 f"   (target {EXPLOG.target_coupling_length_um:g} um; smaller is better)")
    lines.append(f"  excess_loss_dB = {metrics['excess_loss_dB']:.6g}   (smaller is better)")
    return "\n".join(lines)


@tool
def run_coupled_mode_analysis(
    candidate_id: Annotated[str, "Id of the design to compute (e.g. dc_0003)."],
    candidate_id_ref: Annotated[str, "Reference ID of the result where you obtained this candidate id (the enter_design_in_log message, or a query_explog result)."],
    note: Annotated[str, "Short note: why compute this design now and what hypothesis the result will test."],
) -> str:
    """
    Run the finite-element coupled-mode analysis for a design already in the
    experiment log and record the results in the EXPLOG.

    The analysis meshes the coupler cross-section, solves the fundamental TE mode
    of each waveguide in isolation and the two lowest supermodes of the coupled
    pair, and evaluates the coupling coefficients, the coupling length and the
    maximum power transfer with coupled-mode theory. It returns when the solve
    finishes, typically in a few seconds.

    Two kinds of unhappy outcome, and they mean different things:
      * the DESIGN is not viable (no guided mode, no index contrast, too weakly
        confined) -- recorded as a failed process, and the design should be
        dispositioned as Abandon;
      * the CALCULATION could not be run (solver crash or timeout) -- nothing is
        recorded against the design, which stays untouched and can be retried.
        Report a repeated environment failure to the supervisor.

    A design that already has a completed analysis is not recomputed.
    """
    if candidate_id_ref and CANVAS.get_artifact(candidate_id_ref) is None:
        return f"Error: Reference ID {candidate_id_ref} not found in CANVAS."
    cdf = EXPLOG.relational_frame.candidates.df
    if not (cdf["candidate_id"] == candidate_id).any():
        return f"Error: no candidate '{candidate_id}' in the experiment log. Enter it first with enter_design_in_log."

    existing = EXPLOG.has_completed_analysis(candidate_id)
    if existing is not None:
        return (f"Design {candidate_id} already has a completed analysis (process_id {existing}); "
                f"it is not recomputed. Read the results with read_explog('{candidate_id}').")

    design = EXPLOG.design_of(candidate_id)
    t0 = time.time()
    pid = None
    try:
        result = solve_in_subprocess(design)
        pid = EXPLOG.record_analysis(candidate_id, note, result=result, error=None)
        status = "completed"
        outStr = (f"Coupled-mode analysis for design {candidate_id} completed (process_id {pid}, "
                  f"{time.time() - t0:.1f} s):\n{_format_result(result)}")
        value = {**{k: float(result[k]) for k in RESULT_COLUMNS},
                 **derived_metrics(result, EXPLOG.target_coupling_length_um)}
    except DesignNotViable as e:
        # A property of the DESIGN: record it, the agent dispositions it.
        err = f"DesignNotViable: {e}"
        pid = EXPLOG.record_analysis(candidate_id, note, result=None, error=err)
        status = "failed"
        outStr = (f"Design {candidate_id} is NOT VIABLE (process_id {pid}): {e} "
                  f"This is a real result about the design, not a tool problem. It is recorded "
                  f"in the EXPLOG; disposition {candidate_id} as 'Abandon' and move on to a "
                  f"different geometry or material.")
        value = outStr
    except (CalculationTimeout, CalculationError) as e:
        # The ENVIRONMENT failed, not the design. Record NOTHING against the
        # candidate: it must stay re-runnable and must not be forced to Abandon.
        status = "error"
        outStr = (f"Coupled-mode analysis for design {candidate_id} could NOT BE RUN: {e} "
                  f"Nothing has been recorded against {candidate_id} and it is unchanged, so you "
                  f"may retry it. This says NOTHING about the design -- do NOT disposition it as "
                  f"failed. If this keeps happening for every design, stop retrying and report "
                  f"the environment failure to the supervisor.")
        value = outStr

    others = [c for c in EXPLOG.candidates_needing_disposition() if c != candidate_id]
    outStr += format_needs_disposition_reminder(others)

    id = CANVAS.register_tool_output(
        tool_name="run_coupled_mode_analysis",
        args={"candidate_id": candidate_id, "note": note, **design},
        value=value,
        description=f"Coupled-mode analysis of design {candidate_id} ({status}). Note: {note}",
        reasons={"note": note},
        parent_result_ids=[candidate_id_ref] if candidate_id_ref else [],
        metadata={"candidate_id": candidate_id, "process_id": pid, "status": status, **design},
    )
    return f"{outStr}\nResult_ID={id}. Refer to this ID when you cite these numbers or disposition {candidate_id}."
