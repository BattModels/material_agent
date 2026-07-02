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
#     Mutates cdf's "Reduced Formula" column in place from
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
#
#   DECOMPOSITION_ENERGY_COLUMN: like "Reduced Formula", this is a PERSISTED
#     EXPLOG column (not a MATERIAL_PROPERTY_COLUMNS-style transient join) --
#     computed once at add_candidate time and reconciled on resume, since
#     (unlike the other material properties) computing it is an H5PY read per
#     candidate, not a cheap in-memory lookup. apply_candidate_query only
#     drops it from the OUTPUT when the flag is False; it never joins or
#     computes it (that's sync_decomposition_energy's job, called by the
#     caller before this function runs, same division of labor as
#     sync_reduced_formula/build_candidates_view).
#
#   sync_decomposition_energy(cdf, pbx_lookup_df, dh, sc) -> None
#     Mutates cdf's DECOMPOSITION_ENERGY_COLUMN in place. Self-heals (creates
#     the column if missing) and fills ONLY rows currently <NA> -- the
#     opposite of sync_reduced_formula's unconditional-overwrite, because
#     recomputing an already-known value here means a wasted H5PY read.

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import Filter, SortSpec
from src.aq_gnome_candidate_sync import (
    MATERIAL_PROPERTY_COLUMNS,
    INTERNAL_CANDIDATE_COLUMNS,
    DECOMPOSITION_ENERGY_COLUMN,
    sync_reduced_formula,
    columns_needed_for_query,
    attach_material_properties,
    compute_decomposition_energy,
    sync_decomposition_energy,
    apply_candidate_query,
    build_candidates_view,
)


# Lightweight fakes for the AQ-GNoME reader/criteria objects
# compute_decomposition_energy needs (dh.mixed_results/gga_results.read_id(id),
# sc.max_dG_in_region(decom_G), sc.col_name). Deliberately NOT importing the
# real aq_gnome.Stability_Criteria here -- importing it alone (no database
# load) still costs ~10s, transitively pulling in pymatgen/torch/matplotlib,
# which would tax this file's "fast" guarantee for every test, not just the
# ones that need it.
class _FakeReader:
    def __init__(self, values_by_id):
        self._values_by_id = values_by_id

    def read_id(self, id_):
        return self._values_by_id[id_]


class _FakeDataHandler:
    def __init__(self, mixed_values_by_id=None, gga_values_by_id=None):
        self.mixed_results = _FakeReader(mixed_values_by_id or {})
        self.gga_results = _FakeReader(gga_values_by_id or {})


class _FakeStabilityCriteria:
    col_name = DECOMPOSITION_ENERGY_COLUMN

    def max_dG_in_region(self, decom_G):
        # The fake reader already returns the "region max" scalar directly,
        # so this just passes it through -- real Stability_Criteria's grid
        # indexing is pre-existing, tested-elsewhere code, not under test here.
        return decom_G


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
    assert cdf.loc[cdf["candidate_id"] == "mp-1", "Reduced Formula"].iloc[0] == "RuO2"
    assert cdf.loc[cdf["candidate_id"] == "mp-2", "Reduced Formula"].iloc[0] == "IrO2"


def test_sync_leaves_na_for_not_found_candidate():
    cdf = _candidates_df(["mp-1", "mp-999"])
    sync_reduced_formula(cdf, _lookup_df())
    assert pd.isna(cdf.loc[cdf["candidate_id"] == "mp-999", "Reduced Formula"].iloc[0])


def test_sync_creates_column_if_missing():
    cdf = _candidates_df(["mp-1"])
    assert "Reduced Formula" not in cdf.columns
    sync_reduced_formula(cdf, _lookup_df())
    assert "Reduced Formula" in cdf.columns
    assert cdf["Reduced Formula"].iloc[0] == "RuO2"


def test_sync_overwrites_stale_value():
    cdf = _candidates_df(["mp-1"])
    cdf["Reduced Formula"] = "WrongFormula"
    sync_reduced_formula(cdf, _lookup_df())
    assert cdf["Reduced Formula"].iloc[0] == "RuO2"


def test_sync_empty_frame_no_raise():
    cdf = _candidates_df([])
    sync_reduced_formula(cdf, _lookup_df())
    assert "Reduced Formula" in cdf.columns
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


def test_attach_keeps_integer_columns_nullable_int_with_na_present():
    # _lookup_df()'s HHI columns are plain int64 (matching production, where
    # they come straight from pd.read_csv). A bare .map() would upcast the
    # WHOLE column to float64 the moment any row is unmapped (NaN), silently
    # contradicting the documented Int64 dtype. mp-999 isn't in the lookup,
    # so this must trigger that path.
    df = _candidates_df(["mp-1", "mp-999"])
    out = attach_material_properties(df, _lookup_df(), ["average_HHI_P"])
    assert out["average_HHI_P"].dtype == pd.Int64Dtype()
    assert out.loc[out["candidate_id"] == "mp-1", "average_HHI_P"].iloc[0] == 2000
    assert pd.isna(out.loc[out["candidate_id"] == "mp-999", "average_HHI_P"].iloc[0])


def test_attach_leaves_non_integer_columns_untouched():
    # Bandgap (float) and Crystal System (str) must not be coerced to Int64.
    df = _candidates_df(["mp-1", "mp-999"])
    out = attach_material_properties(df, _lookup_df(), ["Bandgap", "Crystal System"])
    assert pd.api.types.is_float_dtype(out["Bandgap"].dtype)
    assert out["Bandgap"].dtype != pd.Int64Dtype()


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
    df["Reduced Formula"] = ["RuO2", "IrO2"][: len(ids)]
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
        assert "Reduced Formula" in out.columns
        assert list(out["Reduced Formula"]) == ["RuO2", "IrO2"]


def test_reduced_formula_survives_material_property_filtering():
    # Filtering on an enrichment column (flag False) must not disturb
    # reduced_formula's visibility, even though MATERIAL_PROPERTY_COLUMNS
    # get joined-then-dropped in the same call.
    filters = [Filter(column="Elements", op="contains_any", value=["Ir"])]
    out = build_candidates_view(_raw_candidates_df(["mp-1", "mp-2"]), filters, [], False, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2"]
    assert list(out["Reduced Formula"]) == ["IrO2"]


def test_build_candidates_view_does_not_mutate_input():
    cdf = _raw_candidates_df(["mp-1"])
    original_cols = list(cdf.columns)
    build_candidates_view(cdf, [], [], True, _lookup_df())
    assert list(cdf.columns) == original_cols


# ---------------------------------------------------------------------------
# compute_decomposition_energy -- replaces get_candidate_data's per-ID
# decomposition-energy lookup (now deactivated as a tool). Mirrors its exact
# algorithm: prefer mixed (GGA/GGA(+U)/r2SCAN) results, fall back to GGA-only
# when mixed_pbx_save_id == "Not computed", floor at 0.0.
# ---------------------------------------------------------------------------

def _pbx_lookup_df():
    return pd.DataFrame(
        {
            "mixed_pbx_save_id": ["mixed-1", "Not computed"],
            "gga_only_pbx_save_id": ["gga-1", "gga-2"],
        },
        index=pd.Index(["mp-1", "mp-2"], name="MaterialId"),
    )


def test_compute_decomposition_energy_uses_mixed_when_available():
    df = _candidates_df(["mp-1"])
    dh = _FakeDataHandler(mixed_values_by_id={"mixed-1": 0.42})
    out = compute_decomposition_energy(df, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert out[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.42


def test_compute_decomposition_energy_falls_back_to_gga_when_mixed_not_computed():
    df = _candidates_df(["mp-2"])
    dh = _FakeDataHandler(gga_values_by_id={"gga-2": 0.13})
    out = compute_decomposition_energy(df, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert out[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.13


def test_compute_decomposition_energy_clamps_negative_to_zero():
    df = _candidates_df(["mp-1"])
    dh = _FakeDataHandler(mixed_values_by_id={"mixed-1": -0.7})
    out = compute_decomposition_energy(df, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert out[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.0


def test_compute_decomposition_energy_na_for_not_found_candidate():
    df = _candidates_df(["mp-999"])
    dh = _FakeDataHandler()
    out = compute_decomposition_energy(df, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert pd.isna(out[DECOMPOSITION_ENERGY_COLUMN].iloc[0])


def test_compute_decomposition_energy_does_not_mutate_input():
    df = _candidates_df(["mp-1"])
    original_cols = list(df.columns)
    dh = _FakeDataHandler(mixed_values_by_id={"mixed-1": 0.1})
    compute_decomposition_energy(df, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# sync_decomposition_energy -- unlike sync_reduced_formula's unconditional
# overwrite, this fills ONLY rows currently <NA> (recomputing an already-known
# value would waste an H5PY read for no benefit, since AQ-GNoME data is
# static within a run).
# ---------------------------------------------------------------------------

def test_sync_decomposition_energy_creates_column_if_missing():
    cdf = _candidates_df(["mp-1"])
    assert DECOMPOSITION_ENERGY_COLUMN not in cdf.columns
    dh = _FakeDataHandler(mixed_values_by_id={"mixed-1": 0.42})
    sync_decomposition_energy(cdf, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert DECOMPOSITION_ENERGY_COLUMN in cdf.columns
    assert cdf[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.42


def test_sync_decomposition_energy_fills_only_missing_rows():
    cdf = _candidates_df(["mp-1", "mp-2"])
    cdf[DECOMPOSITION_ENERGY_COLUMN] = pd.array([0.99, pd.NA], dtype="Float64")
    # Only mp-2's lookup is provided; if mp-1 were recomputed this would
    # KeyError on "mixed-1", proving the already-filled row was skipped.
    dh = _FakeDataHandler(gga_values_by_id={"gga-2": 0.05})
    sync_decomposition_energy(cdf, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert cdf.loc[cdf["candidate_id"] == "mp-1", DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.99
    assert cdf.loc[cdf["candidate_id"] == "mp-2", DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.05


def test_sync_decomposition_energy_noop_when_nothing_missing():
    cdf = _candidates_df(["mp-1"])
    cdf[DECOMPOSITION_ENERGY_COLUMN] = pd.array([0.99], dtype="Float64")
    dh = _FakeDataHandler()  # empty -- would KeyError if anything were (re)computed
    sync_decomposition_energy(cdf, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert cdf[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.99


def test_sync_decomposition_energy_na_for_candidate_not_in_lookup():
    cdf = _candidates_df(["mp-999"])
    dh = _FakeDataHandler()
    sync_decomposition_energy(cdf, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert pd.isna(cdf[DECOMPOSITION_ENERGY_COLUMN].iloc[0])


def test_sync_decomposition_energy_empty_frame_no_raise():
    cdf = _candidates_df([])
    dh = _FakeDataHandler()
    sync_decomposition_energy(cdf, _pbx_lookup_df(), dh, _FakeStabilityCriteria())
    assert DECOMPOSITION_ENERGY_COLUMN in cdf.columns
    assert len(cdf) == 0


# ---------------------------------------------------------------------------
# apply_candidate_query / build_candidates_view -- decomposition energy is a
# PERSISTED column (like "Reduced Formula"), not a MATERIAL_PROPERTY_COLUMNS-
# style transient join: when it's already present on the input df (as it will
# be after sync_decomposition_energy has run), these functions only decide
# whether to keep or drop it based on the flag -- no join step, and filtering/
# sorting on it works through plain df_query with no special-casing at all.
# ---------------------------------------------------------------------------

def _candidates_df_with_decomp(ids, values):
    df = _candidates_df(ids)
    df[DECOMPOSITION_ENERGY_COLUMN] = pd.array(values, dtype="Float64")
    return df


def test_flag_true_keeps_decomposition_energy_already_on_df():
    df = _candidates_df_with_decomp(["mp-1"], [0.42])
    out = apply_candidate_query(df, [], [], True, _lookup_df())
    assert DECOMPOSITION_ENERGY_COLUMN in out.columns
    assert out[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.42


def test_flag_false_drops_decomposition_energy_already_on_df():
    df = _candidates_df_with_decomp(["mp-1"], [0.42])
    out = apply_candidate_query(df, [], [], False, _lookup_df())
    assert DECOMPOSITION_ENERGY_COLUMN not in out.columns


def test_flag_false_drop_is_safe_when_column_absent():
    # apply_candidate_query must not assume the column is always present
    # (e.g. a candidate predating this feature, not yet reconciled).
    df = _candidates_df(["mp-1"])
    out = apply_candidate_query(df, [], [], False, _lookup_df())
    assert DECOMPOSITION_ENERGY_COLUMN not in out.columns


def test_filter_on_decomposition_energy_works_without_flag_no_join_needed():
    df = _candidates_df_with_decomp(["mp-1", "mp-2"], [0.42, 0.05])
    filters = [Filter(column=DECOMPOSITION_ENERGY_COLUMN, op="lt", value=0.1)]
    out = apply_candidate_query(df, filters, [], False, _lookup_df())
    assert list(out["candidate_id"]) == ["mp-2"]
    # Filtered fine with no pbx_lookup_df/dh/sc passed at all; flag False -> dropped after.
    assert DECOMPOSITION_ENERGY_COLUMN not in out.columns


def test_build_candidates_view_keeps_decomposition_energy_when_flagged():
    cdf = _raw_candidates_df(["mp-1"])
    cdf[DECOMPOSITION_ENERGY_COLUMN] = pd.array([0.42], dtype="Float64")
    out = build_candidates_view(cdf, [], [], True, _lookup_df())
    assert DECOMPOSITION_ENERGY_COLUMN in out.columns
    assert out[DECOMPOSITION_ENERGY_COLUMN].iloc[0] == 0.42