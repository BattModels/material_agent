# Integration tests for the ACTUAL query_explog @tool in src/tools.py (not the
# pure src/aq_gnome_candidate_sync.py logic it delegates to -- see
# test_aq_gnome_candidate_sync.py for that fast, mocked-free coverage).
#
# SLOW AND OPT-IN: importing src.tools triggers _STABILITY_CACHE = _StabilityCache()
# at module scope, which loads the full ~500k-row AQ-GNoME database (CSVs +
# H5PY files) and takes roughly 109s -- see the comment atop
# tests/test_disposition_tools.py for the established precedent of keeping
# that import out of the default (fast) test run. This file is SKIPPED unless
# RUN_SLOW_TESTS=1 is set:
#
#   RUN_SLOW_TESTS=1 pytest tests/test_query_explog_tool.py -v
#
# What this file proves that the fast pure-module tests cannot (they never
# touch the real @tool wrapper): the actual query_explog function correctly
# calls EXPLOG.update_log(), syncs reduced_formula onto the real EXPLOG frame,
# builds the candidates view via build_candidates_view, dispatches
# candidates/processes/invalid table_name, paginates, and formats output --
# i.e. that the refactor in tools.py didn't drop or misuse any of the pure
# logic it now delegates to.

import os

import pandas as pd
import pytest

if not os.environ.get("RUN_SLOW_TESTS"):
    pytest.skip(
        "Slow: imports src.tools, which loads the full AQ-GNoME database "
        "(~109s). Run with RUN_SLOW_TESTS=1 to include this file.",
        allow_module_level=True,
    )

from gnome_dreams_oer_screening.explog.explog import EXPLOG
from src.tools import query_explog, _STABILITY_CACHE
from src.aq_gnome_candidate_sync import sync_reduced_formula


def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="query_explog tool test")


def _real_material_id():
    """A MaterialId actually present in the loaded AQ-GNoME lookup, so
    reduced_formula/material-property assertions exercise a real "found"
    row rather than only the not-found/<NA> path."""
    return _STABILITY_CACHE.candidate_lookup.index[0]


def test_candidates_table_always_shows_reduced_formula_never_internal_columns(tmp_path):
    _setup(tmp_path)
    _add_candidate(_real_material_id())
    for flag in (False, True):
        out = query_explog.invoke({
            "table_name": "candidates",
            "reason": "check reduced_formula visibility",
            "include_material_properties": flag,
        })
        assert "Reduced Formula" in out
        for internal_col in ("study_obj", "disposition_record", "ready_for_disposition_update"):
            assert internal_col not in out


def test_candidates_table_reduced_formula_resolves_for_real_candidate(tmp_path):
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    expected_formula = _STABILITY_CACHE.candidate_lookup.loc[mid, "Reduced Formula"]
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "check reduced_formula value resolves for a real candidate",
    })
    assert expected_formula in out


def test_include_material_properties_false_hides_columns(tmp_path):
    _setup(tmp_path)
    _add_candidate(_real_material_id())
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "default flag should hide material properties",
    })
    for col in ("Bandgap", "Disorder Probability", "average_HHI_P", "Crystal System"):
        assert col not in out


def test_include_material_properties_true_shows_columns(tmp_path):
    _setup(tmp_path)
    _add_candidate(_real_material_id())
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "explicit flag should show material properties",
        "include_material_properties": True,
    })
    for col in ("Bandgap", "Disorder Probability", "average_HHI_P", "Crystal System", "Elements"):
        assert col in out


def test_filter_on_material_property_column_works_without_flag(tmp_path):
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    elements = _STABILITY_CACHE.candidate_lookup.loc[mid, "Elements"]
    an_element = elements[0]
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "filter candidates by an element they actually contain",
        "filters": [{"column": "Elements", "op": "contains_any", "value": [an_element]}],
    })
    assert mid in out
    # Auto-joined only to filter; flag was omitted (default False) -> dropped again.
    assert "Bandgap" not in out


def test_processes_table_still_dispatches(tmp_path):
    _setup(tmp_path)
    _add_candidate(_real_material_id())
    out = query_explog.invoke({
        "table_name": "processes",
        "reason": "smoke-test the processes branch is unaffected by the candidates refactor",
    })
    assert "VASP_dir" not in out


def test_invalid_table_name_returns_error_message(tmp_path):
    _setup(tmp_path)
    out = query_explog.invoke({
        "table_name": "bogus",
        "reason": "invalid table name should be rejected",
    })
    assert "table_name must be either 'candidates' or 'processes'" in out


def test_invoke_py_resume_reconciliation_sequence(tmp_path):
    # Mirrors invoke.py's Part 4 (added after the code-review finding that
    # enter_candidate_in_log could crash on a resumed pre-feature checkpoint):
    #   EXPLOG.job_handler._ensure_reduced_formula_column()
    #   sync_reduced_formula(EXPLOG.relational_frame.candidates.df, _STABILITY_CACHE.candidate_lookup)
    # Exercises the exact composition against the real EXPLOG + real
    # _STABILITY_CACHE -- each piece is unit tested alone (test_reduced_formula.py,
    # test_aq_gnome_candidate_sync.py), this proves they compose correctly.
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    expected_formula = _STABILITY_CACHE.candidate_lookup.loc[mid, "Reduced Formula"]

    # Simulate a resumed pre-feature checkpoint: the column existed (from
    # add_candidate) but pretend it never did, as a real old checkpoint would.
    cdf = EXPLOG.relational_frame.candidates.df
    EXPLOG.relational_frame.candidates.df = cdf.drop(columns=["Reduced Formula"])

    EXPLOG.job_handler._ensure_reduced_formula_column()
    sync_reduced_formula(EXPLOG.relational_frame.candidates.df,
                         _STABILITY_CACHE.candidate_lookup)

    cdf = EXPLOG.relational_frame.candidates.df
    assert cdf.loc[cdf["candidate_id"] == mid, "Reduced Formula"].iloc[0] == expected_formula
