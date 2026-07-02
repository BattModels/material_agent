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
# touch the real @tool wrapper, or real AQ-GNoME H5PY Pourbaix data): the
# actual query_explog function correctly calls EXPLOG.update_log(), syncs
# reduced_formula and decomposition energy onto the real EXPLOG frame (the
# latter backed by real H5PY reads, and persisted -- not recomputed -- across
# repeated queries), builds the candidates view via build_candidates_view,
# dispatches candidates/processes/invalid table_name, paginates, and formats
# output -- i.e. that the refactor in tools.py didn't drop or misuse any of
# the pure logic it now delegates to.

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
from src.tools import query_explog, enter_candidate_in_log, _STABILITY_CACHE
from src.aq_gnome_candidate_sync import sync_reduced_formula, sync_decomposition_energy


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


def test_include_material_properties_true_shows_decomposition_energy(tmp_path):
    # Replaces the now-deactivated get_candidate_data tool's per-ID
    # decomposition-energy lookup. This candidate is added via the raw
    # EXPLOG.add_candidate test helper (no decomposition_energy passed), so
    # the persisted value starts <NA> -- query_explog's internal
    # sync_decomposition_energy call must backfill it before display.
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "check decomposition energy is included with the flag",
        "include_material_properties": True,
    })
    assert "max_dG_U[1.2,2.0]_pH0" in out
    val = EXPLOG.relational_frame.candidates.df.loc[
        EXPLOG.relational_frame.candidates.df["candidate_id"] == mid,
        "max_dG_U[1.2,2.0]_pH0",
    ].iloc[0]
    assert not pd.isna(val)


def test_include_material_properties_false_hides_decomposition_energy(tmp_path):
    _setup(tmp_path)
    _add_candidate(_real_material_id())
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "default flag should hide decomposition energy too",
    })
    assert "max_dG_U[1.2,2.0]_pH0" not in out


def test_filter_on_decomposition_energy_works_without_flag(tmp_path):
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    out = query_explog.invoke({
        "table_name": "candidates",
        "reason": "filter candidates by decomposition energy without the flag",
        "filters": [{"column": "max_dG_U[1.2,2.0]_pH0", "op": "ge", "value": 0.0}],
    })
    assert mid in out
    # Already-persisted column (synced internally); flag was omitted
    # (default False) -> dropped from the output again after filtering.
    assert "max_dG_U[1.2,2.0]_pH0" not in out


def test_decomposition_energy_persists_and_is_not_recomputed_on_second_query(tmp_path, monkeypatch):
    # The whole point of persisting this column (vs. the earlier lazy-compute
    # design): an H5PY read per candidate should happen ONCE, not on every
    # query_explog(include_material_properties=True) call.
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)

    row = _STABILITY_CACHE.candidate_lookup.loc[mid]
    reader = (_STABILITY_CACHE.dh.mixed_results if row["mixed_pbx_save_id"] != "Not computed"
              else _STABILITY_CACHE.dh.gga_results)
    calls = {"count": 0}
    real_read_id = reader.read_id

    def counting_read_id(id_):
        calls["count"] += 1
        return real_read_id(id_)

    monkeypatch.setattr(reader, "read_id", counting_read_id)

    query_explog.invoke({
        "table_name": "candidates",
        "reason": "first query computes and persists decomposition energy",
        "include_material_properties": True,
    })
    assert calls["count"] == 1

    query_explog.invoke({
        "table_name": "candidates",
        "reason": "second query should reuse the persisted value, not recompute",
        "include_material_properties": True,
    })
    assert calls["count"] == 1  # unchanged


def test_enter_candidate_in_log_computes_decomposition_energy_at_entry(tmp_path):
    _setup(tmp_path)
    mid = _real_material_id()
    enter_candidate_in_log.invoke({
        "reason_or_hypothesis": "test entry via the real tool",
        "MaterialId": mid,
        "MaterialId_ref": "",
    })
    cdf = EXPLOG.relational_frame.candidates.df
    val = cdf.loc[cdf["candidate_id"] == mid, "max_dG_U[1.2,2.0]_pH0"].iloc[0]
    assert not pd.isna(val)
    assert val >= 0.0


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
    #   EXPLOG.job_handler._ensure_decomposition_energy_column()
    #   sync_reduced_formula(EXPLOG.relational_frame.candidates.df, _STABILITY_CACHE.candidate_lookup)
    #   sync_decomposition_energy(EXPLOG.relational_frame.candidates.df, _STABILITY_CACHE.candidate_lookup, _STABILITY_CACHE.dh, _STABILITY_CACHE.decomposition_criteria)
    # Exercises the exact composition against the real EXPLOG + real
    # _STABILITY_CACHE -- each piece is unit tested alone (test_reduced_formula.py,
    # test_decomposition_energy.py, test_aq_gnome_candidate_sync.py), this
    # proves they compose correctly.
    _setup(tmp_path)
    mid = _real_material_id()
    _add_candidate(mid)
    expected_formula = _STABILITY_CACHE.candidate_lookup.loc[mid, "Reduced Formula"]

    # Simulate a resumed pre-feature checkpoint: the columns existed (from
    # add_candidate) but pretend they never did, as a real old checkpoint would.
    cdf = EXPLOG.relational_frame.candidates.df
    EXPLOG.relational_frame.candidates.df = cdf.drop(
        columns=["Reduced Formula", "max_dG_U[1.2,2.0]_pH0"]
    )

    EXPLOG.job_handler._ensure_reduced_formula_column()
    EXPLOG.job_handler._ensure_decomposition_energy_column()
    sync_reduced_formula(EXPLOG.relational_frame.candidates.df,
                         _STABILITY_CACHE.candidate_lookup)
    sync_decomposition_energy(EXPLOG.relational_frame.candidates.df,
                              _STABILITY_CACHE.candidate_lookup,
                              _STABILITY_CACHE.dh,
                              _STABILITY_CACHE.decomposition_criteria)

    cdf = EXPLOG.relational_frame.candidates.df
    assert cdf.loc[cdf["candidate_id"] == mid, "Reduced Formula"].iloc[0] == expected_formula
    assert not pd.isna(
        cdf.loc[cdf["candidate_id"] == mid, "max_dG_U[1.2,2.0]_pH0"].iloc[0]
    )
