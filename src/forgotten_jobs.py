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
site_index}`` (termination/site are None except for OH).
"""

import pandas as pd


def _na_to_none_int(v):
    """pandas NA/NaN -> None; otherwise a plain int (so (term, site) keys and
    the rendered message are clean Python ints, never <NA>)."""
    if v is None or pd.isna(v):
        return None
    return int(v)


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
        out.append({"candidate_id": candidate_id, "kind": "OH",
                    "termination_index": term, "site_index": site})
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
        # Frontier (mutually exclusive): bulk -> surface -> O.
        if is0(row.get("n_bulk_started")):
            items.append({"candidate_id": cid, "kind": "bulk",
                          "termination_index": None, "site_index": None})
            continue
        if eq1(row.get("n_bulk_finalized")) and is0(row.get("n_surface_started")):
            items.append({"candidate_id": cid, "kind": "surface",
                          "termination_index": None, "site_index": None})
            continue
        if ge1(row.get("n_surface_finalized")) and is0(row.get("n_O_started")):
            items.append({"candidate_id": cid, "kind": "O",
                          "termination_index": None, "site_index": None})
            continue
        # Competitive O sites without OH (one item per site).
        if oh_enabled:
            items.extend(_forgotten_oh_sites(pdf, cid, go_dev_oh_threshold))
    return items
