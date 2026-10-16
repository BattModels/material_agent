"""Fast tests for the FEM experiment log: candidate entry, instant analysis
recording, the progress projection, and the disposition engine."""
import pandas as pd
import pytest

from src import var
from src.fem_explog import FemExplog, RESULT_COLUMNS, DESIGN_COLUMNS, METRIC_COLUMNS, derived_metrics


def _design(**over):
    d = {
        "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.46, "core_height_um": 0.22,
        "wavelength_um": 1.55, "core_material": "Si", "n_core": 3.4777,
        "clad_material": "SiO2", "n_clad": 1.444,
    }
    d.update(over)
    return d


def _result(**over):
    r = {k: 1.0 for k in RESULT_COLUMNS}
    r.update({"n_eff_symmetric": 2.371, "n_eff_antisymmetric": 2.349,
              "coupling_length_um": 35.31, "max_power_transfer": 0.99})
    r.update(over)
    return r


@pytest.fixture
def log(tmp_path):
    e = FemExplog()
    e.init(tmp_path, disposition_decisions=var.DISPOSITION_DECISIONS, target_coupling_length_um=20.0)
    return e


def test_derived_metrics_formulas():
    m = derived_metrics({"coupling_length_um": 35.31, "max_power_transfer": 0.5}, 20.0)
    assert m["coupling_length_deviation_um"] == pytest.approx(15.31)
    assert m["excess_loss_dB"] == pytest.approx(3.0103, abs=1e-3)
    assert derived_metrics({"coupling_length_um": 20.0, "max_power_transfer": 1.0}, 20.0) == \
        {"coupling_length_deviation_um": 0.0, "excess_loss_dB": 0.0}
    assert derived_metrics({"coupling_length_um": 1.0, "max_power_transfer": 0.0}, 20.0)["excess_loss_dB"] == pytest.approx(120.0)


def test_metrics_land_on_process_and_candidate_rows_and_track_the_latest(log):
    cid = log.add_candidate(_design(), "h")
    p0 = log.record_analysis(cid, "n", result=_result(coupling_length_um=35.31, max_power_transfer=0.5), error=None)
    prow = log.relational_frame.processes[p0].series
    assert float(prow["coupling_length_deviation_um"]) == pytest.approx(15.31)
    assert float(prow["excess_loss_dB"]) == pytest.approx(3.0103, abs=1e-3)
    row = log.relational_frame.candidates[cid].series
    assert float(row["coupling_length_deviation_um"]) == pytest.approx(15.31)
    log.record_analysis(cid, "n", result=_result(coupling_length_um=19.0, max_power_transfer=1.0), error=None)
    row = log.relational_frame.candidates[cid].series
    assert float(row["coupling_length_deviation_um"]) == pytest.approx(1.0) and float(row["excess_loss_dB"]) == 0.0
    # a failed run carries no metrics
    pf = log.record_analysis(cid, "n", result=None, error="x")
    assert pd.isna(log.relational_frame.processes[pf].series["excess_loss_dB"])


def test_metrics_self_heal_on_a_pre_metric_frame(log):
    cid = log.add_candidate(_design(), "h")
    log.record_analysis(cid, "n", result=_result(coupling_length_um=25.0), error=None)
    for tbl in (log.relational_frame.candidates, log.relational_frame.processes):
        tbl.df = tbl.df.drop(columns=list(METRIC_COLUMNS))          # simulate an old checkpoint
    log.update_log()
    assert float(log.relational_frame.candidates[cid].series["coupling_length_deviation_um"]) == pytest.approx(5.0)
    assert float(log.relational_frame.processes[0].series["coupling_length_deviation_um"]) == pytest.approx(5.0)
    # and a new record on the healed frame still works
    log.record_analysis(cid, "n", result=_result(coupling_length_um=21.0), error=None)
    assert float(log.relational_frame.candidates[cid].series["coupling_length_deviation_um"]) == pytest.approx(1.0)


def test_ids_are_chronological_and_duplicates_refused(log):
    a = log.add_candidate(_design(), "h1")
    b = log.add_candidate(_design(gap_um=0.5), "h2")
    assert (a, b) == ("dc_0001", "dc_0002")
    with pytest.raises(ValueError) as ei:
        log.add_candidate(_design(), "again")
    assert "dc_0001" in str(ei.value)
    # material name comparison is case-insensitive
    assert log.find_duplicate_design(_design(core_material="si")) == "dc_0001"


def test_fresh_candidate_projection(log):
    cid = log.add_candidate(_design(), "h")
    row = log.relational_frame.candidates[cid].series
    assert int(row["n_started"]) == 0 and int(row["n_finalized"]) == 0
    assert row["state"] == "active"
    assert pd.isna(row["coupling_length_um"])
    assert bool(row["needs_disposition_update"]) is False


def test_completed_analysis_is_copied_up_and_needs_disposition(log):
    cid = log.add_candidate(_design(), "h")
    pid = log.record_analysis(cid, "note", result=_result(), error=None)
    assert pid == 0
    row = log.relational_frame.candidates[cid].series
    assert int(row["n_started"]) == 1 and int(row["n_finalized"]) == 1
    assert row["state"] == "active"
    assert float(row["coupling_length_um"]) == pytest.approx(35.31)
    assert bool(row["needs_disposition_update"]) is True
    assert log.candidates_needing_disposition() == [cid]
    assert log.has_completed_analysis(cid) == 0


def test_failed_analysis_marks_candidate_failed_until_a_success(log):
    cid = log.add_candidate(_design(), "h")
    log.record_analysis(cid, "n", result=None, error="boom")
    assert log.relational_frame.candidates[cid].series["state"] == "failed"
    assert log.has_completed_analysis(cid) is None
    # a later success flips it back to active (no latch: the design is fine)
    log.record_analysis(cid, "n", result=_result(), error=None)
    assert log.relational_frame.candidates[cid].series["state"] == "active"


def test_design_of_round_trips_floats(log):
    cid = log.add_candidate(_design(), "h")
    d = log.design_of(cid)
    assert set(d) == set(DESIGN_COLUMNS)
    assert isinstance(d["gap_um"], float) and d["core_material"] == "Si"


def test_disposition_write_lock_and_coverage(log):
    cid = log.add_candidate(_design(), "h")
    pid = log.record_analysis(cid, "n", result=_result(), error=None)
    # locked until get_disposition_info
    r = log.update_disposition_info(cid, "s", [pid], "f", "Medium priority")
    assert r["status"] == "locked"
    info = log.get_disposition_info(cid)
    assert info["must_cover"] == [{"job_type": "coupled_mode_analysis", "ids": [pid]}]
    # invalid vocabulary
    assert log.update_disposition_info(cid, "s", [pid], "f", "Investigating")["status"] == "invalid_decision"
    # foreign id
    assert log.update_disposition_info(cid, "s", [99], "f", "Medium priority")["status"] == "foreign_ids"
    # incomplete coverage (cite nothing)
    r = log.update_disposition_info(cid, "s", [], "f", "Medium priority")
    assert r["status"] == "incomplete" and r["missing"] == [pid]
    # success clears the flag and records the decision
    r = log.update_disposition_info(cid, "s", [pid], "f", "Medium priority")
    assert r["ok"] and r["status"] == "ok"
    row = log.relational_frame.candidates[cid].series
    assert row["decision"] == "Medium priority"
    assert bool(row["needs_disposition_update"]) is False
    assert log.candidates_needing_disposition() == []
    # a second analysis re-opens the need
    pid2 = log.record_analysis(cid, "n", result=_result(), error=None)
    assert log.candidates_needing_disposition() == [cid]
    assert log.get_disposition_info(cid)["must_cover"] == [{"job_type": "coupled_mode_analysis", "ids": [pid2]}]


def test_unknown_candidate(log):
    assert log.get_disposition_info("dc_9999") == {"unknown_candidate": True, "candidate_id": "dc_9999"}
    assert log.update_disposition_info("dc_9999", "s", [], "f", "Abandon")["status"] == "unknown_candidate"


def test_pickle_written(log, tmp_path):
    log.add_candidate(_design(), "h")
    assert (tmp_path / "explog.pkl").exists()
