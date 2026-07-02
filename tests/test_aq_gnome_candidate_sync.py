# Tests for src/aq_gnome_candidate_sync.py, written test-FIRST (TDD).
#
# This module holds the PURE logic behind two query_explog features so it can
# be unit-tested fast, without importing src.tools (which loads the ~500k-row
# GNoME database at module scope and takes ~109s -- see test_disposition_tools.py
# for the established precedent of avoiding that import in tests).
#
# Contract pinned here:
#   MATERIAL_PROPERTY_COLUMNS: the fixed list of AQ-GNoME columns Part 2 can
#     surface (Elements, Crystal System, Bandgap, Disorder Probability, and
#     the six HHI variants). Does NOT include "Reduced Formula" -- that's a
#     separately persisted EXPLOG column (Part 1), always present.
#
#   sync_reduced_formula(cdf, lookup_df) -> None
#     Mutates cdf's "reduced_formula" column in place from
#     lookup_df["Reduced Formula"] (indexed by MaterialId). Self-heals (creates
#     the column if missing) and unconditionally refreshes every row; a
#     candidate_id absent from lookup_df gets <NA>.
#
#   columns_needed_for_query(filters, sort, known_columns) -> set[str]
#     Pure set-intersection helper: which of `known_columns` do the given
#     filters/sort actually reference.
#
#   attach_material_properties(df, lookup_df, columns) -> pd.DataFrame
#     Joins exactly `columns` (a subset of lookup_df's columns) onto df by
#     candidate_id -> lookup_df's MaterialId index. Missing candidates get NA.
#
#   apply_candidate_query(df, filters, sort, include_material_properties,
#                          lookup_df) -> pd.DataFrame
#     Orchestrates Part 2 end-to-end: joins MATERIAL_PROPERTY_COLUMNS in
#     transiently whenever a filter/sort references one of them OR the flag is
#     True, runs df_query (existing filter/sort machinery), then drops those
#     columns again unless the flag was True.
#
#   INTERNAL_CANDIDATE_COLUMNS: the agent-internal candidates.df columns
#     query_explog/read_explog must never display (study_obj, disposition_record,
#     ready_for_disposition_update).
#
#   build_candidates_view(cdf, filters, sort, include_material_properties,
#                          lookup_df) -> pd.DataFrame
#     The exact column pipeline query_explog's candidates branch runs: drop
#     INTERNAL_CANDIDATE_COLUMNS, then apply_candidate_query. Does NOT call
#     sync_reduced_formula (a side-effecting mutation of the real EXPLOG frame
#     -- the caller does that first). Pinning this as a named, tested function
#     (rather than inline code in tools.py) is what lets a fast test prove
#     "reduced_formula is always visible, internal columns never are" without
#     importing src.tools.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import Filter, SortSpec
from src.aq_gnome_candidate_sync import (
    MATERIAL_PROPERTY_COLUMNS,
    INTERNAL_CANDIDATE_COLUMNS,
    sync_reduced_formula,
    columns_needed_for_query,
    attach_material_properties,
    apply_candidate_query,
    build_candidates_view,
)


def _lookup_df():
    return pd.DataFrame(
        {
            "Reduced Formula": ["RuO2", "IrO2"],
            "Elements": [["Ru", "O"], ["Ir", "O"]],
            "Crystal System": ["tetragonal", "cubic"],
            "Bandgap": [0.0, 1.5],
            "Disorder Probability": [0.1, 0.9],
            "average_HHI_P": [2000, 3000],
            "average_HHI_R": [1500, 2500],
            "average_HHI_P_excluding_O_H": [2100, 3100],
            "average_HHI_R_excluding_O_H": [1600, 2600],
            "max_HHI_P": [2500, 3500],
            "max_HHI_R": [2000, 3000],
        },
        index=pd.Index(["mp-1", "mp-2"], name="MaterialId"),
    )


def _candidates_df(ids):
    return pd.DataFrame({
        "candidate_id": ids,
        "state": ["active"] * len(ids),
    })


# ---------------------------------------------------------------------------
# sync_reduced_formula
# ---------------------------------------------------------------------------

def test_sync_fills_matching_candidates():
    cdf = _candidates_df(["mp-1", "mp-2"])
    sync_reduced_formula(cdf, _lookup_df())
    assert cdf.loc[cdf["candidate_id"] == "mp-1", "reduced_formula"].iloc[0] == "RuO2"
    assert cdf.loc[cdf["candidate_id"] == "mp-2", "reduced_formula"].iloc[0] == "IrO2"


def test_sync_leaves_na_for_not_found_candidate():
    cdf = _candidates_df(["mp-1", "mp-999"])
    sync_reduced_formula(cdf, _lookup_df())
    assert pd.isna(cdf.loc[cdf["candidate_id"] == "mp-999", "reduced_formula"].iloc[0])


def test_sync_creates_column_if_missing():
    cdf = _candidates_df(["mp-1"])
    assert "reduced_formula" not in cdf.columns
    sync_reduced_formula(cdf, _lookup_df())
    assert "reduced_formula" in cdf.columns
    assert cdf["reduced_formula"].iloc[0] == "RuO2"


def test_sync_overwrites_stale_value():
    cdf = _candidates_df(["mp-1"])
    cdf["reduced_formula"] = "WrongFormula"
    sync_reduced_formula(cdf, _lookup_df())
    assert cdf["reduced_formula"].iloc[0] == "RuO2"


def test_sync_empty_frame_no_raise():
    cdf = _candidates_df([])
    sync_reduced_formula(cdf, _lookup_df())
    assert "reduced_formula" in cdf.columns
    assert len(cdf) == 0


# ---------------------------------------------------------------------------
# columns_needed_for_query
# ---------------------------------------------------------------------------

def test_columns_needed_empty_when_no_filters_or_sort():
    assert columns_needed_for_query([], [], MATERIAL_PROPERTY_COLUMNS) == set()


def test_columns_needed_detects_filter_reference():
    filters = [Filter(column="Elements", op="contains_any", value=["Ru"])]
    assert columns_needed_for_query(filters, [], MATERIAL_PROPERTY_COLUMNS) == {"Elements"}


def test_columns_needed_detects_sort_reference():
    sort = [SortSpec(column="Bandgap", ascending=True)]
    assert columns_needed_for_query([], sort, MATERIAL_PROPERTY_COLUMNS) == {"Bandgap"}


def test_columns_needed_ignores_non_enrichment_columns():
    filters = [Filter(column="state", op="eq", value="active")]
    assert columns_needed_for_query(filters, [], MATERIAL_PROPERTY_COLUMNS) == set()


def test_columns_needed_union_of_filter_and_sort():
    filters = [Filter(column="Bandgap", op="gt", value=1.0)]
    sort = [SortSpec(column="max_HHI_P", ascending=False)]
    result = columns_needed_for_query(filters, sort, MATERIAL_PROPERTY_COLUMNS)
    assert result == {"Bandgap", "max_HHI_P"}


# ---------------------------------------------------------------------------
# attach_material_properties
# ---------------------------------------------------------------------------

def test_attach_joins_requested_columns_only():
    df = _candidates_df(["mp-1", "mp-2"])
    out = attach_material_properties(df, _lookup_df(), ["Bandgap", "Crystal System"])
    assert list(out["Bandgap"]) == [0.0, 1.5]
    assert list(out["Crystal System"]) == ["tetragonal", "cubic"]
    assert "average_HHI_P" not in out.columns  # not requested


def test_attach_preserves_original_columns():
    df = _candidates_df(["mp-1"])
    out = attach_material_properties(df, _lookup_df(), ["Bandgap"])
    assert "candidate_id" in out.columns
    assert "state" in out.columns


def test_attach_na_for_not_found_candidate():
    df = _candidates_df(["mp-999"])
    out = attach_material_properties(df, _lookup_df(), ["Bandgap", "Elements"])
    assert pd.isna(out["Bandgap"].iloc[0])
    assert pd.isna(out["Elements"].iloc[0])


# ---------------------------------------------------------------------------
# apply_candidate_query (orchestration)
# ---------------------------------------------------------------------------

def test_flag_false_no_filters_hides_material_properties():
    df = _candidates_df(["mp-1", "mp-2"])
    out = apply_candidate_query(df, [], [], False, _lookup_df())
    for col in MATERIAL_PROPERTY_COLUMNS:
        assert col not in out.columns


def test_flag_true_includes_all_material_properties():
    df = _candidates_df(["mp-1", "mp-2"])
    out = apply_candidate_query(df, [], [], True, _lookup_df())
    for col in MATERIAL_PROPERTY_COLUMNS:
        assert col in out.columns
    assert out.loc[out["candidate_id"] == "mp-1", "Bandgap"].iloc[0] == 0.0


def test_flag_true_appends_columns_after_originals():
    df = _candidates_df(["mp-1"])
    out = apply_candidate_query(df, [], [], True, _lookup_df())
    original_cols = list(df.columns)
    assert list(out.columns)[:len(original_cols)] == original_cols


def test_filter_on_enrichment_column_auto_joins_without_flag():
    df = _candidates_df(["mp-1", "mp-2"])
    filters = [Filter(column="Elements", op="contains_any", value=["Ir"])]
    out = apply_candidate_query(df, filters, [], False, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2"]
    # Auto-joined only to filter; flag was False, so it must be dropped again.
    for col in MATERIAL_PROPERTY_COLUMNS:
        assert col not in out.columns


def test_sort_on_enrichment_column_auto_joins_without_flag():
    df = _candidates_df(["mp-1", "mp-2"])
    sort = [SortSpec(column="Bandgap", ascending=False)]
    out = apply_candidate_query(df, [], sort, False, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2", "mp-1"]  # mp-2 has higher Bandgap
    for col in MATERIAL_PROPERTY_COLUMNS:
        assert col not in out.columns


def test_filter_on_enrichment_column_with_flag_true_keeps_columns():
    df = _candidates_df(["mp-1", "mp-2"])
    filters = [Filter(column="Bandgap", op="gt", value=1.0)]
    out = apply_candidate_query(df, filters, [], True, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2"]
    assert "Bandgap" in out.columns
    assert out["Bandgap"].iloc[0] == 1.5


def test_filter_on_base_column_never_joins_material_properties():
    df = _candidates_df(["mp-1", "mp-2"])
    filters = [Filter(column="state", op="eq", value="active")]
    out = apply_candidate_query(df, filters, [], False, _lookup_df())
    assert len(out) == 2
    for col in MATERIAL_PROPERTY_COLUMNS:
        assert col not in out.columns


def test_not_found_candidate_kept_with_na_properties_when_flag_true():
    df = _candidates_df(["mp-1", "mp-999"])
    out = apply_candidate_query(df, [], [], True, _lookup_df())
    assert len(out) == 2
    row = out.loc[out["candidate_id"] == "mp-999"]
    assert pd.isna(row["Bandgap"].iloc[0])


# ---------------------------------------------------------------------------
# build_candidates_view -- the exact pipeline query_explog's candidates
# branch runs. These pin the claim that reduced_formula is ALWAYS visible
# (regardless of include_material_properties or filters) while the
# agent-internal columns are NEVER visible.
# ---------------------------------------------------------------------------

def _raw_candidates_df(ids):
    df = _candidates_df(ids)
    df["reduced_formula"] = ["RuO2", "IrO2"][: len(ids)]
    df["study_obj"] = [object() for _ in ids]
    df["disposition_record"] = [[] for _ in ids]
    df["ready_for_disposition_update"] = [False for _ in ids]
    return df


def test_internal_columns_never_visible():
    for flag in (False, True):
        out = build_candidates_view(_raw_candidates_df(["mp-1"]), [], [], flag, _lookup_df())
        for col in INTERNAL_CANDIDATE_COLUMNS:
            assert col not in out.columns


def test_reduced_formula_always_visible_regardless_of_flag():
    for flag in (False, True):
        out = build_candidates_view(_raw_candidates_df(["mp-1", "mp-2"]), [], [], flag, _lookup_df())
        assert "reduced_formula" in out.columns
        assert list(out["reduced_formula"]) == ["RuO2", "IrO2"]


def test_reduced_formula_survives_material_property_filtering():
    # Filtering on an enrichment column (flag False) must not disturb
    # reduced_formula's visibility, even though MATERIAL_PROPERTY_COLUMNS
    # get joined-then-dropped in the same call.
    filters = [Filter(column="Elements", op="contains_any", value=["Ir"])]
    out = build_candidates_view(_raw_candidates_df(["mp-1", "mp-2"]), filters, [], False, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2"]
    assert list(out["reduced_formula"]) == ["IrO2"]


def test_build_candidates_view_does_not_mutate_input():
    cdf = _raw_candidates_df(["mp-1"])
    original_cols = list(cdf.columns)
    build_candidates_view(cdf, [], [], True, _lookup_df())
    assert list(cdf.columns) == original_cols