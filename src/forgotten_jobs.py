"""Forgotten-jobs detection for the wait-tool's queue-floor (Gate 2) hint.

``find_forgotten_jobs`` reports ready-but-unstarted work derived ENTIRELY from
the EXPLOG dataframes (the per-candidate progress projection on the candidates
df + the processes df). It lives here -- a small parent module that imports only
pandas + the EXPLOG dataframes -- rather than in ``src/tools.py`` so it can be
unit-tested in the parent fast suite without paying that module's ~109 s GNoME
database import. The wait tool calls it and feeds the result to the Gate 2
message; it is deliberately NOT registered as a standalone @tool.

Each returned item is a dict
``{candidate_id, kind in {"bulk","surface","O","OH"}, termination_index,
site_index, go_dev, formula}`` (termination/site/go_dev are None except for OH;
``formula`` is the candidate's "Reduced Formula", None until it is backfilled).

ORDERING IS PART OF THE CONTRACT. The list comes back BEST FIRST: OH items
ranked by measured G(O) deviation ascending, then the unranked frontier stages
depth-first (O, surface, bulk). See ``_KIND_RANK`` for why bulk must come last.
"""

import pandas as pd

# Sort ranks for the returned list. OH items carry a MEASURED number, so they
# are ordered against each other on evidence and come first. Frontier items have
# no per-item number and are ranked by pipeline depth instead -- deliberately
# NOT by an invented pseudo-deviation, which would just relocate a
# fabricated-threshold failure into the code.
#
# bulk MUST rank last, and this is load-bearing: a candidate-registration round
# adds dozens of bulk items at once, and under the old registration-order
# listing those would flood the top of the (truncated) Gate 2 message and hide
# the competitive OH work behind "... and N more" -- the exact failure that
# deadlocked the 02-08 run for 88 consecutive steps.
_KIND_RANK = {"OH": 0, "O": 1, "surface": 2, "bulk": 3}


def _sort_key(item):
    dev = item.get("go_dev")
    return (_KIND_RANK.get(item.get("kind"), 99),
            float(dev) if dev is not None else 0.0)


def _na_to_none_int(v):
    """pandas NA/NaN -> None; otherwise a plain int (so (term, site) keys and
    the rendered message are clean Python ints, never <NA>)."""
    if v is None or pd.isna(v):
        return None
    return int(v)


def _na_to_none_str(v):
    """pandas NA/NaN/blank -> None; otherwise a plain str. "Reduced Formula" is
    <NA> until sync_reduced_formula backfills it, and the renderer must be able
    to tell "no formula known" from a literal "<NA>" in the agent's message."""
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _forgotten_oh_sites(processes_df, candidate_id, go_dev_oh_threshold):
    """Competitive O sites of one candidate that have no OH adsorption yet.

    From the processes df only: O_adsorption rows that are ``completed`` with
    ``|G(O) deviation| < go_dev_oh_threshold`` and no OH_adsorption row at the
    same (termination, site). One item per such site.
    """
    sub = processes_df[processes_df["candidate_id"] == candidate_id]
    if sub.empty:
        return []
    oh_sites = {
        (_na_to_none_int(r["termination_index"]),
         _na_to_none_int(r["site_index"]))
        for _, r in sub[sub["job_type"] == "OH_adsorption"].iterrows()
    }
    out = []
    o_rows = sub[(sub["job_type"] == "O_adsorption")
                 & (sub["status"] == "completed")]
    for _, r in o_rows.iterrows():
        dev = r.get("G(O) deviation")
        if dev is None or pd.isna(dev):
            continue
        if abs(float(dev)) >= float(go_dev_oh_threshold):
            continue
        term = _na_to_none_int(r["termination_index"])
        site = _na_to_none_int(r["site_index"])
        if (term, site) in oh_sites:
            continue
        # abs() is defensive: the backend already stores |G(O) - 2.46|, and the
        # threshold check above applies abs() as well, so the rendered message
        # can never show a negative "deviation".
        out.append({"candidate_id": candidate_id, "kind": "OH",
                    "termination_index": term, "site_index": site,
                    "go_dev": float(abs(float(dev))), "formula": None})
    return out


def find_forgotten_jobs(explog, go_dev_oh_threshold):
    """Ready-but-unstarted work, derived ENTIRELY from the EXPLOG dataframes.

    Refreshes the per-candidate progress projection (cheap; no HPC poll), then
    for each candidate reports the single frontier stage that is eligible but
    unstarted (bulk -> surface -> O) or, once O results exist, one item per
    competitive O site that still has no OH.

    The ``<NA>`` gating in the progress columns means a FAILED-bulk candidate
    yields nothing (``n_surface_started`` stays ``<NA>``, not 0); a
    ``state == "failed"`` candidate is also skipped outright. OH detection
    (category 4) is disabled when ``go_dev_oh_threshold < 0``.
    """
    # Refresh the derived progress columns from the current processes df so the
    # frontier reflects the latest known log state (matches the pattern of
    # candidates_needing_disposition refreshing the disposition state).
    explog.job_handler._recompute_candidate_progress()
    cdf = explog.relational_frame.candidates.df
    pdf = explog.relational_frame.processes.df
    if len(cdf) == 0:
        return []
    oh_enabled = go_dev_oh_threshold is not None and go_dev_oh_threshold >= 0

    def is0(v):
        return pd.notna(v) and int(v) == 0

    def eq1(v):
        return pd.notna(v) and int(v) == 1

    def ge1(v):
        return pd.notna(v) and int(v) >= 1

    items = []
    for _, row in cdf.iterrows():
        if str(row.get("state")) == "failed":
            continue
        cid = row["candidate_id"]
        formula = _na_to_none_str(row.get("Reduced Formula"))
        # Frontier (mutually exclusive): bulk -> surface -> O. go_dev is None
        # for all three -- they have no completed O site to measure. The key is
        # still present so every item has the same shape.
        if is0(row.get("n_bulk_started")):
            items.append({"candidate_id": cid, "kind": "bulk",
                          "termination_index": None, "site_index": None,
                          "go_dev": None, "formula": formula})
            continue
        if eq1(row.get("n_bulk_finalized")) and is0(row.get("n_surface_started")):
            items.append({"candidate_id": cid, "kind": "surface",
                          "termination_index": None, "site_index": None,
                          "go_dev": None, "formula": formula})
            continue
        if ge1(row.get("n_surface_finalized")) and is0(row.get("n_O_started")):
            items.append({"candidate_id": cid, "kind": "O",
                          "termination_index": None, "site_index": None,
                          "go_dev": None, "formula": formula})
            continue
        # Competitive O sites without OH (one item per site). _forgotten_oh_sites
        # works off the processes df alone and cannot see the candidates row, so
        # the formula is stamped on here rather than widening its signature.
        if oh_enabled:
            oh = _forgotten_oh_sites(pdf, cid, go_dev_oh_threshold)
            for it in oh:
                it["formula"] = formula
            items.extend(oh)
    # BEST FIRST -- see _KIND_RANK. list.sort is stable, so candidate
    # registration order survives as a deterministic tie-break within a rank.
    items.sort(key=_sort_key)
    return items
