import math
import os
import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src import var


def load_config(path: str):
    """Load config/default.yaml, mirror every key into var.OTHER_GLOBAL_VARIABLES
    and set the working directory."""
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in config.items():
        var.OTHER_GLOBAL_VARIABLES[key] = value
    var.my_WORKING_DIRECTORY = config["WORKING_DIR"]
    return config


def save_graph_to_file(graph, path: str, name: str):
    try:
        im = graph.get_graph(xray=True).draw_mermaid_png()
        with open(os.path.join(path, f"{name}.png"), "wb") as f:
            f.write(im)
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    return


# ---------------------------------------------------------------------------
# Generic dataframe filter / sort grammar shared by query_explog.
# ---------------------------------------------------------------------------
Op = Literal[
    "eq", "ne", "gt", "ge", "lt", "le",
    "in", "not_in",
    "contains", "not_contains",
    "regex", "not_regex",
    "isnull", "notnull",
    "contains_any", "not_contains_any", "contains_all", "not_contains_all",
]


class Filter(BaseModel):
    column: str = Field(description="Column name to filter on.")
    op: Op = Field(description="Filter operation.")
    value: Optional[Any] = Field(
        default=None,
        description="Value for the operation.\n"
            "Expected types by op:\n"
            "- isnull, notnull: no value needed (omit or set null).\n"
            "- eq, ne: scalar matching the column dtype (e.g., str, int, float, bool).\n"
            "- gt, ge, lt, le: numeric scalar (int/float) or other comparable scalar matching the column dtype.\n"
            "- in, not_in: list-like (e.g., list[str], list[int], list[float]); pass a list even for one item.\n"
            "- contains, not_contains: str substring/pattern (uses pandas string contains).\n"
            "- regex, not_regex: str regex pattern.\n"
            "- contains_any, not_contains_any, contains_all, not_contains_all: list[str] "
            "(for list-like columns).",
    )


class SortSpec(BaseModel):
    column: str = Field(description="Column name to sort by.")
    ascending: bool = Field(default=True, description="Sort ascending if True else descending.")


def _apply_filter(frame: pd.DataFrame, f) -> pd.DataFrame:
    allowed_cols = set(frame.columns)
    if f.column not in allowed_cols:
        raise ValueError(f"Unknown column: {f.column}. Allowed: {sorted(allowed_cols)}")

    s = frame[f.column]

    if f.op in ("isnull", "notnull"):
        mask = s.isna() if f.op == "isnull" else s.notna()
        return frame[mask]

    if f.value is None:
        raise ValueError(f"Filter op '{f.op}' requires a value.")

    val = f.value

    if f.op in ("contains_any", "not_contains_any", "contains_all", "not_contains_all"):
        if not isinstance(val, list):
            raise ValueError(f"'{f.op}' requires a list value.")
        target = set(val)

        def ok(x):
            if not isinstance(x, list):
                return False
            xs = set(x)
            if f.op in ("contains_any", "not_contains_any"):
                return len(xs & target) > 0
            return target.issubset(xs)

        mask = s.apply(ok)
        if f.op.startswith("not_"):
            mask = ~mask
        return frame[mask]

    if f.op == "contains":
        return frame[s.astype(str).str.contains(str(val), na=False)]
    if f.op == "not_contains":
        return frame[~s.astype(str).str.contains(str(val), na=False)]
    if f.op == "regex":
        pattern = re.compile(str(val))
        return frame[s.astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]
    if f.op == "not_regex":
        pattern = re.compile(str(val))
        return frame[~s.astype(str).apply(lambda x: bool(pattern.search(x)) if pd.notna(x) else False)]

    if f.op == "in":
        if not isinstance(val, list):
            raise ValueError("'in' requires a list value.")
        return frame[s.isin(val)]
    if f.op == "not_in":
        if not isinstance(val, list):
            raise ValueError("'not_in' requires a list value.")
        return frame[~s.isin(val)]

    # NA-safe scalar comparisons: a missing value is excluded from eq/gt/ge/lt/le
    # and kept for ne. The .fillna is required so nullable columns don't raise
    # on boolean masking.
    if f.op == "eq":
        return frame[(s == val).fillna(False)]
    if f.op == "ne":
        return frame[(s != val).fillna(True)]
    if f.op == "gt":
        return frame[(s > val).fillna(False)]
    if f.op == "ge":
        return frame[(s >= val).fillna(False)]
    if f.op == "lt":
        return frame[(s < val).fillna(False)]
    if f.op == "le":
        return frame[(s <= val).fillna(False)]

    raise ValueError(f"Unsupported op: {f.op}")


def df_query(
    frame: pd.DataFrame,
    filters: List[Filter] = [],
    sort: List[SortSpec] = [],
    select: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter (AND semantics), then sort, then optionally select columns."""
    allowed_cols = set(frame.columns)
    for f in filters:
        frame = _apply_filter(frame, f)
    if sort:
        for s in sort:
            if s.column not in allowed_cols:
                raise ValueError(f"Unknown sort column: {s.column}")
        frame = frame.sort_values(
            by=[s.column for s in sort],
            ascending=[s.ascending for s in sort],
            kind="mergesort",  # stable
        )
    if select is not None:
        unknown = [c for c in select if c not in allowed_cols]
        if unknown:
            raise ValueError(f"Unknown columns in select: {unknown}")
        frame = frame[select]
    return frame


# ---------------------------------------------------------------------------
# Numeric-token helpers used by search_artifacts.
# ---------------------------------------------------------------------------
_NUMERIC_RE = re.compile(
    r"""
    (?<![\w.])                       # avoid letters/underscore immediately before
    [+-]?                            # optional sign
    (?:
        (?:\d+\.\d*|\.\d+|\d+)       # int or decimal
        (?:[eE][+-]?\d+)?            # optional scientific notation
    )
    (?![\w.])                        # avoid letters/underscore immediately after
    """,
    re.VERBOSE,
)


def util_find_all_substring_spans(text: str, snippet: str) -> List[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        idx = text.find(snippet, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(snippet)))
        start = idx + 1
    return spans


def util_numeric_matches_in_region(
    text: str,
    region_start: int,
    region_end: int,
    target_value: float,
    abs_tol: float,
) -> List[Dict[str, Any]]:
    matches = []
    region_text = text[region_start:region_end]
    for m in _NUMERIC_RE.finditer(region_text):
        token = m.group(0)
        try:
            parsed = float(token)
        except ValueError:
            continue
        if math.isclose(parsed, float(target_value), rel_tol=0.0, abs_tol=abs_tol):
            matches.append(
                {
                    "token": token,
                    "parsed_value": parsed,
                    "start": region_start + m.start(),
                    "end": region_start + m.end(),
                }
            )
    return matches


def is_floatable(x):
    try:
        float(x)
        return True
    except Exception:
        return False
