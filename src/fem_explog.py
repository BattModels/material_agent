"""Experiment log (EXPLOG) for the directional-coupler screening study.

Two related tables on a ``RelationalFrame``:

  candidates  -- one row per coupler DESIGN (a parameter combination): geometry,
                 wavelength, materials, the agent's hypothesis, a per-candidate
                 progress projection, the best result copied up, and the
                 disposition columns.
  processes   -- one row per CALCULATION run for a candidate. The calculator runs
                 in-process and returns instantly, so a process is recorded as
                 'completed' or 'failed' the moment it is entered; there is no
                 pending/running state to poll.

The disposition engine (coverage of finalized work by disposition records, the
write-lock, the ``needs_disposition_update`` flag) is kept identical in meaning
to the OER study so the disposition tools and their prompts carry over unchanged.

Persistence: pickled atomically to ``<working_dir>/explog.pkl`` after every
mutation. The two dataframes are ALSO checkpointed LangGraph channels (see
src/planNexe2.py), which is what resume/time-travel actually restore from; the
pickle is a human-inspectable side record.
"""
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.relational_frame import RelationalFrame


JOB_TYPE = "coupled_mode_analysis"
TERMINAL_STATUSES = ("completed", "failed")

# Design (input) columns, in display order. These identify a candidate: two
# candidates with identical values here are the same design.
DESIGN_COLUMNS = (
    "gap_um", "wg1_width_um", "wg2_width_um", "core_height_um", "wavelength_um",
    "core_material", "n_core", "clad_material", "n_clad",
)

# Result columns produced by the calculator, copied onto the process row and
# (for the latest completed run) onto the candidate row.
RESULT_COLUMNS = (
    "n_eff_wg1", "n_eff_wg2", "n_eff_symmetric", "n_eff_antisymmetric",
    "kappa_12_per_um", "kappa_21_per_um", "coupling_length_um", "max_power_transfer",
)

# Metrics derived from a result and the study target (see derived_metrics). Both
# are "smaller is better" so the agent can rank on them directly.
METRIC_COLUMNS = (
    "coupling_length_deviation_um",   # |coupling_length_um - target|
    "excess_loss_dB",                 # -10 log10(max_power_transfer): light that never crosses
)
_MIN_POWER_FOR_LOG = 1e-12


def derived_metrics(result: dict, target_coupling_length_um: float) -> dict:
    """Pure: the METRIC_COLUMNS for one completed result."""
    lc = float(result["coupling_length_um"])
    p = max(float(result["max_power_transfer"]), _MIN_POWER_FOR_LOG)
    return {
        "coupling_length_deviation_um": abs(lc - float(target_coupling_length_um)),
        "excess_loss_dB": -10.0 * math.log10(p),
    }


# Candidate columns that are agent-internal and hidden by the query tools.
HIDDEN_CANDIDATE_COLUMNS = ("disposition_record", "ready_for_disposition_update")


def _f64():
    return pd.Series(dtype=pd.Float64Dtype())


def _str():
    return pd.Series(dtype=pd.StringDtype())


def _i64():
    return pd.Series(dtype=pd.Int64Dtype())


def _bool():
    return pd.Series(dtype=pd.BooleanDtype())


class FemExplog:
    """Manually initialised singleton (call ``init`` once per process)."""

    def init(self, working_directory: Path | str, disposition_decisions=None,
             target_coupling_length_um: float = 20.0) -> None:
        rf = RelationalFrame(root="candidates")

        candidates = pd.DataFrame({
            "candidate_id": _str(),
            **{c: (_str() if c.endswith("_material") else _f64()) for c in DESIGN_COLUMNS},
            "reason_or_hypothesis": _str(),
            "notes": _str(),
            # --- progress projection (derived on every _recompute) ---
            "n_started": _i64(),      # calculations entered for this design
            "n_finalized": _i64(),    # of those, terminal (completed or failed)
            "state": _str(),          # 'active' | 'failed'
            # --- latest completed result, copied up, plus its derived metrics ---
            **{c: _f64() for c in RESULT_COLUMNS},
            **{c: _f64() for c in METRIC_COLUMNS},
            # --- disposition gate ---
            "disposition_record": pd.Series(dtype="object"),  # append-only list of dicts
            "decision": _str(),
            "ready_for_disposition_update": _bool(),
            "needs_disposition_update": _bool(),
        })
        rf.add_table("candidates", candidates, pk="candidate_id")

        processes = pd.DataFrame({
            "process_id": _i64(),
            "candidate_id": _str(),
            "job_type": _str(),
            "status": _str(),
            "processNote": _str(),
            "error": _str(),
            **{c: _f64() for c in RESULT_COLUMNS},
            **{c: _f64() for c in METRIC_COLUMNS},
        })
        rf.add_table("processes", processes, pk="process_id")
        rf.candidates.add_relation(name="processes", child_table="processes",
                                   fk_in_child="candidate_id")

        self.relational_frame = rf
        self.working_directory = Path(working_directory)
        self.disposition_decisions = tuple(disposition_decisions or ())
        self.target_coupling_length_um = float(target_coupling_length_um)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _save_pickle(self) -> None:
        payload = {
            "candidates_df": self.relational_frame.candidates.df,
            "processes_df": self.relational_frame.processes.df,
        }
        self.working_directory.mkdir(parents=True, exist_ok=True)
        tmp = self.working_directory / "explog.pkl.tmp"
        dest = self.working_directory / "explog.pkl"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        try:
            os.replace(tmp, dest)
        except PermissionError:
            if dest.exists():
                dest.unlink()
            tmp.rename(dest)

    # ------------------------------------------------------------------
    # candidates
    # ------------------------------------------------------------------
    def _next_candidate_id(self) -> str:
        cdf = self.relational_frame.candidates.df
        return f"dc_{len(cdf) + 1:04d}"

    def find_duplicate_design(self, design: dict) -> str | None:
        """candidate_id of an existing design with identical inputs, else None."""
        cdf = self.relational_frame.candidates.df
        if len(cdf) == 0:
            return None
        mask = pd.Series(True, index=cdf.index)
        for c in DESIGN_COLUMNS:
            v = design.get(c)
            col = cdf[c]
            if c.endswith("_material"):
                mask &= (col.astype("string").str.lower() == str(v).lower()).fillna(False)
            else:
                mask &= (col.astype("Float64").sub(float(v)).abs() < 1e-9).fillna(False)
        hits = cdf.loc[mask, "candidate_id"].tolist()
        return str(hits[0]) if hits else None

    def add_candidate(self, design: dict, reason_or_hypothesis: str | None = None,
                      notes: str | None = None) -> str:
        """Enter one design. Raises ValueError on an exact duplicate."""
        dup = self.find_duplicate_design(design)
        if dup is not None:
            raise ValueError(
                f"An identical design is already in the log as candidate {dup}. "
                f"Use that id instead of entering it again."
            )
        cid = self._next_candidate_id()
        row = {
            "candidate_id": cid,
            **{c: design[c] for c in DESIGN_COLUMNS},
            "reason_or_hypothesis": reason_or_hypothesis,
            "notes": notes,
            "n_started": 0,
            "n_finalized": 0,
            "state": "active",
            "disposition_record": [],
            "decision": None,
            "ready_for_disposition_update": False,
            "needs_disposition_update": False,
        }
        self.relational_frame.candidates.add_row(row, allow_update=False)
        self._recompute()
        self._save_pickle()
        return cid

    def design_of(self, candidate_id: str) -> dict:
        row = self.relational_frame.candidates[candidate_id].series
        out = {c: row[c] for c in DESIGN_COLUMNS}
        for c in DESIGN_COLUMNS:
            if not c.endswith("_material"):
                out[c] = float(out[c])
        return out

    def has_completed_analysis(self, candidate_id: str) -> int | None:
        """process_id of a completed analysis for this candidate, else None."""
        pdf = self.relational_frame.processes.df
        hits = pdf[(pdf["candidate_id"] == candidate_id) & (pdf["status"] == "completed")]
        return int(hits["process_id"].iloc[0]) if len(hits) else None

    # ------------------------------------------------------------------
    # processes (instant compute: entered and finalized in one call)
    # ------------------------------------------------------------------
    def record_analysis(self, candidate_id: str, note: str | None,
                        result: dict | None, error: str | None) -> int:
        """Append one process row for ``candidate_id``.

        Exactly one of ``result`` (a dict with every RESULT_COLUMNS key) or
        ``error`` must be given; the row is 'completed' or 'failed' accordingly.
        Returns the new process_id.
        """
        pdf = self.relational_frame.processes.df
        pid = int(pdf["process_id"].max()) + 1 if len(pdf) else 0
        row: dict[str, Any] = {
            "process_id": pid,
            "candidate_id": candidate_id,
            "job_type": JOB_TYPE,
            "processNote": note,
        }
        if result is not None:
            missing = [k for k in RESULT_COLUMNS if k not in result]
            if missing:
                raise ValueError(f"Calculator result is missing keys: {missing}")
            row["status"] = "completed"
            row.update({k: float(result[k]) for k in RESULT_COLUMNS})
            row.update(derived_metrics(result, self.target_coupling_length_um))
        else:
            row["status"] = "failed"
            row["error"] = str(error)
        # Self-heal: a checkpoint that predates METRIC_COLUMNS lacks them.
        self._ensure_columns(self.relational_frame.processes.df, METRIC_COLUMNS)
        self.relational_frame.processes.add_row(row, allow_update=False)
        self._recompute()
        self._save_pickle()
        return pid

    def update_log(self) -> dict:
        """Kept for interface symmetry with a job-queue backend. Nothing is
        asynchronous here, so this only refreshes the derived columns."""
        self._recompute()
        return {}

    # ------------------------------------------------------------------
    # derived projections
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols) -> None:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())

    @staticmethod
    def _is_terminal_process_status(status) -> bool:
        return pd.notna(status) and str(status) in TERMINAL_STATUSES

    def _recompute(self) -> None:
        self._recompute_candidate_progress()
        self._recompute_disposition_state()

    def _recompute_candidate_progress(self) -> None:
        cdf = self.relational_frame.candidates.df
        pdf = self.relational_frame.processes.df
        if len(cdf) == 0:
            return
        cid = cdf["candidate_id"]
        terminal = pdf["status"].isin(TERMINAL_STATUSES)
        completed = pdf["status"] == "completed"

        def per_cand(mask) -> pd.Series:
            counts = pdf.loc[mask, "candidate_id"].value_counts()
            return cid.map(counts).fillna(0).astype("Int64")

        started = per_cand(pd.Series(True, index=pdf.index))
        finalized = per_cand(terminal)
        n_completed = per_cand(completed)
        cdf["n_started"] = started
        cdf["n_finalized"] = finalized
        # failed = tried at least once, everything terminal, nothing succeeded.
        failed = (started >= 1) & (finalized == started) & (n_completed == 0)
        state = pd.Series("active", index=cdf.index, dtype="string")
        state[failed.fillna(False).astype(bool)] = "failed"
        cdf["state"] = state

        # Copy the LATEST completed result (and its derived metrics) up to the
        # candidate row. Metrics are recomputed from the result rather than copied,
        # so a resumed pre-metric checkpoint (or a changed target) self-heals.
        self._ensure_columns(cdf, RESULT_COLUMNS + METRIC_COLUMNS)
        self._ensure_columns(pdf, METRIC_COLUMNS)
        done = pdf[completed].sort_values("process_id")
        if len(done):
            metrics = done.apply(lambda r: pd.Series(derived_metrics(r, self.target_coupling_length_um)), axis=1)
            for c in METRIC_COLUMNS:
                pdf.loc[done.index, c] = metrics[c].astype(float)
                done[c] = metrics[c].astype(float)
        latest = done.groupby("candidate_id").tail(1).set_index("candidate_id")
        for c in RESULT_COLUMNS + METRIC_COLUMNS:
            cdf[c] = cid.map(latest[c]).astype(pd.Float64Dtype())

    # ------------------------------------------------------------------
    # disposition engine
    # ------------------------------------------------------------------
    def _candidate_dispositions(self, candidate_id) -> list:
        cdf = self.relational_frame.candidates.df
        mask = cdf["candidate_id"] == candidate_id
        if not mask.any():
            return []
        rec = cdf.loc[mask, "disposition_record"].iloc[0]
        return rec if isinstance(rec, list) else []

    def _candidate_finalized_units(self, candidate_id) -> list:
        """One unit per terminal process row: {job_type, ids:[pid]}."""
        pdf = self.relational_frame.processes.df
        sub = pdf[pdf["candidate_id"] == candidate_id]
        return [
            {"job_type": str(r["job_type"]), "ids": [int(r["process_id"])]}
            for _, r in sub.iterrows()
            if self._is_terminal_process_status(r["status"])
        ]

    def candidate_outstanding(self, candidate_id) -> dict:
        """{must_cover, latest_disposition, has_finalized} -- the single source
        of truth for needs_disposition_update and the disposition tools. A unit
        is COVERED iff some disposition's Summarized_process_id intersects its ids."""
        units = self._candidate_finalized_units(candidate_id)
        recs = self._candidate_dispositions(candidate_id)
        covered: set = set()
        for rec in recs:
            for pid in (rec.get("Summarized_process_id") or []):
                covered.add(int(pid))
        must_cover = [u for u in units if not (set(u["ids"]) & covered)]
        return {
            "must_cover": must_cover,
            "latest_disposition": recs[-1] if recs else None,
            "has_finalized": len(units) > 0,
        }

    def _recompute_disposition_state(self) -> None:
        cdf = self.relational_frame.candidates.df
        if len(cdf) == 0:
            return
        needs = []
        for cid in cdf["candidate_id"].tolist():
            out = self.candidate_outstanding(cid)
            needs.append(bool(out["must_cover"]) or (
                out["latest_disposition"] is None and out["has_finalized"]))
        cdf["needs_disposition_update"] = pd.Series(needs, index=cdf.index, dtype="boolean")

    def candidates_needing_disposition(self) -> list:
        self._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        if len(cdf) == 0:
            return []
        mask = cdf["needs_disposition_update"].astype("boolean").fillna(False)
        return [str(c) for c in cdf.loc[mask, "candidate_id"].tolist()]

    def get_disposition_info(self, candidate_id: str) -> dict:
        """Unlock the disposition write for ``candidate_id`` and return its
        outstanding-coverage info (or {"unknown_candidate": True})."""
        self._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        mask = cdf["candidate_id"] == candidate_id
        if not mask.any():
            return {"unknown_candidate": True, "candidate_id": candidate_id}
        cdf.loc[mask, "ready_for_disposition_update"] = True
        self._save_pickle()
        return self.candidate_outstanding(candidate_id)

    def update_disposition_info(self, candidate_id: str, summary: str,
                                summarized_process_id, future_plan: str,
                                decision: str) -> dict:
        """Record (or reject) a disposition. Returns
        {ok, status, candidate_id, decision, missing, ids}; only a successful
        write clears the write-lock."""
        self._recompute_disposition_state()
        cdf = self.relational_frame.candidates.df
        mask = cdf["candidate_id"] == candidate_id
        result = {"ok": False, "status": None, "candidate_id": candidate_id,
                  "decision": decision, "missing": [], "ids": []}
        if not mask.any():
            result["status"] = "unknown_candidate"
            return result
        ready_val = cdf.loc[mask, "ready_for_disposition_update"].iloc[0]
        if not (pd.notna(ready_val) and bool(ready_val)):
            result["status"] = "locked"
            return result
        if self.disposition_decisions and decision not in self.disposition_decisions:
            result["status"] = "invalid_decision"
            return result

        pdf = self.relational_frame.processes.df
        cited = [int(i) for i in (summarized_process_id or [])]
        own_ids = set(int(x) for x in
                      pdf.loc[pdf["candidate_id"] == candidate_id, "process_id"].tolist())
        foreign = [pid for pid in cited if pid not in own_ids]
        if foreign:
            result["status"] = "foreign_ids"
            result["ids"] = foreign
            return result
        non_terminal = [pid for pid in cited
                        if not self._is_terminal_process_status(
                            pdf.loc[pdf["process_id"] == pid, "status"].iloc[0])]
        if non_terminal:
            result["status"] = "non_terminal_ids"
            result["ids"] = non_terminal
            return result
        out = self.candidate_outstanding(candidate_id)
        cited_set = set(cited)
        uncovered = [u for u in out["must_cover"] if not (set(u["ids"]) & cited_set)]
        if uncovered:
            result["status"] = "incomplete"
            result["missing"] = sorted({int(i) for u in uncovered for i in u["ids"]})
            return result

        idx = cdf.index[mask][0]
        existing = cdf.at[idx, "disposition_record"]
        if not isinstance(existing, list):
            existing = []
        cdf.at[idx, "disposition_record"] = existing + [{
            "Summary": summary,
            "Summarized_process_id": cited,
            "Future_plan": future_plan,
            "Decision": decision,
        }]
        cdf.loc[mask, "decision"] = decision
        cdf.loc[mask, "ready_for_disposition_update"] = False
        self._recompute_disposition_state()
        self._save_pickle()
        result["ok"] = True
        result["status"] = "ok"
        return result


EXPLOG = FemExplog()
