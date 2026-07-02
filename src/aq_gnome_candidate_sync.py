# Pure logic behind two query_explog features, kept out of src/tools.py so it
# can be unit-tested fast (src.tools loads the ~500k-row GNoME database at
# module scope, which takes ~109s -- see tests/test_disposition_tools.py for
# the established precedent of keeping such logic importable without that
# cost). src/tools.py wires this module's functions to the real EXPLOG and
# _STABILITY_CACHE.
#
# Part 1 -- reduced_formula reconciliation:
#   sync_reduced_formula refreshes the candidates table's persisted
#   reduced_formula column from the AQ-GNoME database, healing candidates
#   entered before this feature existed (or resumed from a pre-feature
#   checkpoint).
#
# Part 2 -- optional material-property enrichment for query_explog:
#   apply_candidate_query joins HHI / disorder / bandgap / crystal-system /
#   elements columns onto the candidates table on demand -- either because
#   include_material_properties=True was passed, or because a filter/sort
#   references one of them (in which case the columns are dropped again
#   after filtering, since the agent didn't ask to see them).

from typing import Iterable, List

import pandas as pd

from src.utils import Filter, SortSpec, df_query

# AQ-GNoME columns Part 2 can surface, using their exact source names (see
# `dataset_description` in src/prompt.py, already in the OER agent's system
# prompt, for field semantics). "Reduced Formula" is intentionally excluded:
# it is the separately persisted `reduced_formula` EXPLOG column (Part 1),
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


def sync_reduced_formula(cdf: pd.DataFrame, lookup_df: pd.DataFrame) -> None:
    """Refresh cdf's reduced_formula column in place from
    lookup_df["Reduced Formula"] (lookup_df indexed by MaterialId).

    Self-heals (creates the column if entirely missing) and unconditionally
    overwrites every row from the lookup -- cheap given candidate counts are
    small, and simpler/more robust than only-fill-if-missing. A candidate_id
    not found in lookup_df gets <NA>.
    """
    if "reduced_formula" not in cdf.columns:
        cdf["reduced_formula"] = pd.Series(
            [pd.NA] * len(cdf), index=cdf.index, dtype="string"
        )
    if len(cdf) == 0:
        return
    cdf["reduced_formula"] = (
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
    """
    needed = columns_needed_for_query(filters, sort, MATERIAL_PROPERTY_COLUMNS)
    should_join = include_material_properties or bool(needed)

    if should_join:
        df = attach_material_properties(df, lookup_df, MATERIAL_PROPERTY_COLUMNS)

    df = df_query(df, filters, sort)

    if should_join and not include_material_properties:
        df = df.drop(columns=MATERIAL_PROPERTY_COLUMNS, errors="ignore")

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

    Does NOT call sync_reduced_formula -- that mutates the real EXPLOG frame
    in place (a side effect callers must trigger separately, before this
    function runs) and is a distinct concern from building a display view.
    reduced_formula itself is never touched here: it isn't in
    INTERNAL_CANDIDATE_COLUMNS or MATERIAL_PROPERTY_COLUMNS, so it always
    survives both the internal-column drop and Part 2's join/drop.
    """
    df = cdf.drop(columns=INTERNAL_CANDIDATE_COLUMNS, errors="ignore")
    return apply_candidate_query(df, filters, sort, include_material_properties, lookup_df)