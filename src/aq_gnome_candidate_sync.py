# Pure logic behind two query_explog features, kept out of src/tools.py so it
# can be unit-tested fast (src.tools loads the ~500k-row GNoME database at
# module scope, which takes ~109s -- see tests/test_disposition_tools.py for
# the established precedent of keeping such logic importable without that
# cost). src/tools.py wires this module's functions to the real EXPLOG and
# _STABILITY_CACHE.
#
# Part 1 -- "Reduced Formula" reconciliation:
#   sync_reduced_formula refreshes the candidates table's persisted
#   "Reduced Formula" column (named to match AQ-GNoME's own field name
#   exactly, for consistency with OER_data_analasis_v2/browse_df) from the
#   AQ-GNoME database, healing candidates entered before this feature
#   existed (or resumed from a pre-feature checkpoint).
#
# Part 2 -- optional material-property enrichment for query_explog:
#   apply_candidate_query joins HHI / disorder / bandgap / crystal-system /
#   elements columns onto the candidates table on demand -- either because
#   include_material_properties=True was passed, or because a filter/sort
#   references one of them (in which case the columns are dropped again
#   after filtering, since the agent didn't ask to see them). The Pourbaix
#   decomposition energy (DECOMPOSITION_ENERGY_COLUMN) follows the same
#   on-demand contract but needs a separate code path: it isn't a static
#   column, it's computed per candidate from H5PY decomposition-energy-grid
#   data (compute_decomposition_energy, mirroring the now-deactivated
#   get_candidate_data tool's exact algorithm) -- so apply_candidate_query
#   only computes it when the caller supplies pbx_lookup_df/dh/sc.

from typing import Iterable, List

import pandas as pd

from src.utils import Filter, SortSpec, df_query

# AQ-GNoME columns Part 2 can surface, using their exact source names (see
# `dataset_description` in src/prompt.py, already in the OER agent's system
# prompt, for field semantics). "Reduced Formula" is intentionally excluded:
# it is the separately persisted "Reduced Formula" EXPLOG column (Part 1),
# always present regardless of this flag.
MATERIAL_PROPERTY_COLUMNS: List[str] = [
    "Elements",
    "Crystal System",
    "Bandgap",
    "Disorder Probability",
    "average_HHI_P",
    "average_HHI_R",
    "average_HHI_P_excluding_O_H",
    "average_HHI_R_excluding_O_H",
    "max_HHI_P",
    "max_HHI_R",
]

# Maximum (worst-case) Pourbaix decomposition energy in eV/atom across
# pH = 0 and U in [1.2, 2.0] V vs. SHE (matches _StabilityCache.PHS/US in
# src/tools.py). Named to match aq_gnome.Stability_Criteria's own
# dynamically-generated col_name for these fixed pHs/Us -- fixed here as a
# constant since PHS/US never change at runtime. Governed by the same
# include_material_properties contract as MATERIAL_PROPERTY_COLUMNS, but
# computed separately (see compute_decomposition_energy) since it isn't a
# static column.
DECOMPOSITION_ENERGY_COLUMN = "max_dG_U[1.2,2.0]_pH0"


def sync_reduced_formula(cdf: pd.DataFrame, lookup_df: pd.DataFrame) -> None:
    """Refresh cdf's "Reduced Formula" column in place from
    lookup_df["Reduced Formula"] (lookup_df indexed by MaterialId). Source
    and destination share the same column name -- both match AQ-GNoME's own
    field name exactly, for consistency across the whole repo.

    Self-heals (creates the column if entirely missing) and unconditionally
    overwrites every row from the lookup -- cheap given candidate counts are
    small, and simpler/more robust than only-fill-if-missing. A candidate_id
    not found in lookup_df gets <NA>.
    """
    if "Reduced Formula" not in cdf.columns:
        cdf["Reduced Formula"] = pd.Series(
            [pd.NA] * len(cdf), index=cdf.index, dtype="string"
        )
    if len(cdf) == 0:
        return
    cdf["Reduced Formula"] = (
        cdf["candidate_id"].map(lookup_df["Reduced Formula"]).astype("string")
    )


def columns_needed_for_query(
    filters: Iterable[Filter],
    sort: Iterable[SortSpec],
    known_columns: Iterable[str],
) -> set:
    """Which of `known_columns` are actually referenced by `filters`/`sort`."""
    known = set(known_columns)
    needed = {f.column for f in filters if f.column in known}
    needed |= {s.column for s in sort if s.column in known}
    return needed


def attach_material_properties(
    df: pd.DataFrame, lookup_df: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    """Join exactly `columns` (a subset of lookup_df's columns) onto df by
    candidate_id -> lookup_df's MaterialId index. A candidate_id not found in
    lookup_df gets NA for every requested column.

    lookup_df's HHI columns are plain (non-nullable) int64, as loaded by
    pd.read_csv. A bare .map() silently upcasts the WHOLE result to float64
    the moment any row is unmapped (NaN can't live in int64) -- so an
    integer-typed source column is cast to pandas' nullable Int64 afterward,
    which holds both real integers and <NA> without that downcast.
    """
    out = df.copy()
    for col in columns:
        mapped = out["candidate_id"].map(lookup_df[col])
        if pd.api.types.is_integer_dtype(lookup_df[col].dtype):
            mapped = mapped.astype("Int64")
        out[col] = mapped
    return out


def compute_decomposition_energy(
    df: pd.DataFrame, pbx_lookup_df: pd.DataFrame, dh, sc
) -> pd.DataFrame:
    """Compute DECOMPOSITION_ENERGY_COLUMN for each row in df, by
    candidate_id -> pbx_lookup_df's MaterialId-indexed mixed_pbx_save_id /
    gga_only_pbx_save_id. Mirrors the now-deactivated get_candidate_data
    tool's exact per-row algorithm: prefer mixed (GGA/GGA(+U)/r2SCAN)
    results, fall back to GGA-only when mixed_pbx_save_id == "Not computed",
    floor at 0.0 (a material can't be "more stable than most stable").

    dh: a Data_Handler-like object (dh.mixed_results / dh.gga_results, each
        with .read_id(id) -> decomposition-energy grid).
    sc: a Stability_Criteria-like object (.max_dG_in_region(decom_G) -> float).

    Computed per row rather than for the whole AQ-GNoME database, so cost is
    bounded by len(df) (EXPLOG's candidate count, typically small) rather
    than the full screened universe -- unlike the other material properties,
    this isn't a cheap static lookup; each row is an H5PY read. A
    candidate_id not found in pbx_lookup_df gets <NA>.
    """
    out = df.copy()
    values = []
    for cid in out["candidate_id"]:
        if cid not in pbx_lookup_df.index:
            values.append(pd.NA)
            continue
        row = pbx_lookup_df.loc[cid]
        mixed_id = row["mixed_pbx_save_id"]
        if mixed_id != "Not computed":
            decom_G = dh.mixed_results.read_id(mixed_id)
        else:
            decom_G = dh.gga_results.read_id(row["gga_only_pbx_save_id"])
        values.append(max(0.0, sc.max_dG_in_region(decom_G)))
    out[sc.col_name] = values
    return out


def sync_decomposition_energy(
    cdf: pd.DataFrame, pbx_lookup_df: pd.DataFrame, dh, sc
) -> None:
    """Fill DECOMPOSITION_ENERGY_COLUMN for rows currently <NA>, in place.

    Self-heals (creates the column, NA-filled, if entirely missing). Unlike
    sync_reduced_formula's unconditional overwrite, only computes rows that
    don't already have a value: computing this column is an H5PY read per
    candidate (compute_decomposition_energy), not a cheap lookup, so
    recomputing an already-known value would waste work for no benefit (the
    underlying AQ-GNoME data is static within a run). A candidate_id not
    found in pbx_lookup_df gets <NA> and stays eligible for a future resync.
    """
    if DECOMPOSITION_ENERGY_COLUMN not in cdf.columns:
        cdf[DECOMPOSITION_ENERGY_COLUMN] = pd.array(
            [pd.NA] * len(cdf), dtype="Float64"
        )
    if len(cdf) == 0:
        return
    missing_mask = cdf[DECOMPOSITION_ENERGY_COLUMN].isna()
    if not missing_mask.any():
        return
    computed = compute_decomposition_energy(cdf.loc[missing_mask], pbx_lookup_df, dh, sc)
    cdf.loc[missing_mask, DECOMPOSITION_ENERGY_COLUMN] = computed[DECOMPOSITION_ENERGY_COLUMN].values


def apply_candidate_query(
    df: pd.DataFrame,
    filters: List[Filter],
    sort: List[SortSpec],
    include_material_properties: bool,
    lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate Part 2 for the candidates table.

    Joins MATERIAL_PROPERTY_COLUMNS in transiently whenever a filter/sort
    references one of them, or permanently when include_material_properties
    is True; runs the existing filter/sort machinery (df_query); then drops
    those columns again unless the flag was True.

    DECOMPOSITION_ENERGY_COLUMN is different: it's a PERSISTED EXPLOG column
    (like "Reduced Formula"), not joined here at all -- the caller runs
    sync_decomposition_energy on the real EXPLOG frame first (same division
    of labor as sync_reduced_formula/build_candidates_view). If it's already
    present on `df`, filtering/sorting on it works for free through df_query
    with no special-casing; this function's only job for it is dropping it
    from the output when the flag is False (a no-op, via errors="ignore", if
    the column isn't there -- e.g. a candidate not yet reconciled).
    """
    needed = columns_needed_for_query(filters, sort, MATERIAL_PROPERTY_COLUMNS)
    should_join = include_material_properties or bool(needed)

    if should_join:
        df = attach_material_properties(df, lookup_df, MATERIAL_PROPERTY_COLUMNS)

    df = df_query(df, filters, sort)

    if should_join and not include_material_properties:
        df = df.drop(columns=MATERIAL_PROPERTY_COLUMNS, errors="ignore")

    if not include_material_properties:
        df = df.drop(columns=[DECOMPOSITION_ENERGY_COLUMN], errors="ignore")

    return df


# Agent-internal candidates.df columns query_explog/read_explog must never
# display: study_obj (a live Python object, not display-safe), disposition_record
# (raw list-of-dicts; read via get_disposition_info instead), and
# ready_for_disposition_update (an internal write-lock).
INTERNAL_CANDIDATE_COLUMNS: List[str] = [
    "study_obj",
    "disposition_record",
    "ready_for_disposition_update",
]


def build_candidates_view(
    cdf: pd.DataFrame,
    filters: List[Filter],
    sort: List[SortSpec],
    include_material_properties: bool,
    lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    """The exact column pipeline query_explog's candidates branch runs: drop
    INTERNAL_CANDIDATE_COLUMNS, then apply_candidate_query.

    Does NOT call sync_reduced_formula or sync_decomposition_energy -- those
    mutate the real EXPLOG frame in place (side effects the caller must
    trigger separately, before this function runs) and are a distinct
    concern from building a display view. "Reduced Formula" itself is never
    touched here: it isn't in INTERNAL_CANDIDATE_COLUMNS or
    MATERIAL_PROPERTY_COLUMNS, so it always survives both the internal-column
    drop and Part 2's join/drop.
    """
    df = cdf.drop(columns=INTERNAL_CANDIDATE_COLUMNS, errors="ignore")
    return apply_candidate_query(df, filters, sort, include_material_properties, lookup_df)