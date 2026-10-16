"""End-to-end tests of the FEM tools against a fresh EXPLOG + CANVAS, with the
calculator monkeypatched so the numbers are deterministic (and a failure path
is exercised). No LLM involved."""
import time

import pytest

import src.tools as tools
from src import var
from src.fem_calculator import CalculationError, CalculationTimeout, DesignNotViable
from src.fem_explog import RESULT_COLUMNS
from src.myCANVAS import CANVAS


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    tools.EXPLOG.init(tmp_path, disposition_decisions=var.DISPOSITION_DECISIONS, target_coupling_length_um=20.0)
    CANVAS.set_working_directory(str(tmp_path))
    CANVAS.canvas.clear()
    CANVAS.result_registry.clear()
    CANVAS.curr_round_result_ids.clear()
    monkeypatch.setattr(var, "my_WORKING_DIRECTORY", str(tmp_path))
    yield


def _fixed_result(params, *a, **kw):
    """Stands in for solve_in_subprocess: deterministic, and no real solver."""
    return {**{k: 1.0 for k in RESULT_COLUMNS},
            "coupling_length_um": 20.0 * params["gap_um"] / 0.4, "max_power_transfer": 0.97}


def _ref_from(msg, label):
    tok = [t for t in msg.replace("\n", " ").split() if t.startswith(label + "=")]
    assert tok, msg
    return tok[0].split("=", 1)[1].rstrip(".")


def test_enter_and_run_happy_path(monkeypatch):
    monkeypatch.setattr(tools, "solve_in_subprocess", _fixed_result)
    mat = tools.list_materials.invoke({})
    mat_ref = _ref_from(mat, "Reference_ID")
    assert mat_ref in CANVAS.result_registry

    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "baseline symmetric SOI coupler",
        "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.45, "wavelength_um": 1.55,
        "core_material": "Si", "clad_material": "SiO2", "design_ref": mat_ref,
    })
    assert "dc_0001" in msg
    cid_ref = _ref_from(msg, "Message_ID")
    cdf = tools.EXPLOG.relational_frame.candidates.df
    assert cdf.loc[0, "core_height_um"] == pytest.approx(0.22)      # platform default
    assert cdf.loc[0, "n_core"] > cdf.loc[0, "n_clad"]

    out = tools.run_coupled_mode_analysis.invoke({
        "candidate_id": "dc_0001", "candidate_id_ref": cid_ref, "note": "baseline"})
    assert "completed" in out and "coupling_length_um = 20" in out
    assert "coupling_length_deviation_um = 0" in out and "target 20 um" in out
    assert "excess_loss_dB = 0.132" in out                      # -10 log10(0.97)
    pdf = tools.EXPLOG.relational_frame.processes.df
    assert len(pdf) == 1 and pdf.loc[0, "status"] == "completed"
    assert pdf.loc[0, "excess_loss_dB"] == pytest.approx(0.1323, abs=1e-3)
    assert tools.EXPLOG.relational_frame.candidates.df.loc[0, "coupling_length_deviation_um"] == 0.0
    # the artifact chain points back at the design entry, which points at the table
    res_ref = _ref_from(out, "Result_ID")
    assert CANVAS.get_artifact(res_ref).parent_result_ids == [cid_ref]
    assert CANVAS.get_artifact(cid_ref).parent_result_ids == [mat_ref]

    # not recomputed
    again = tools.run_coupled_mode_analysis.invoke({
        "candidate_id": "dc_0001", "candidate_id_ref": cid_ref, "note": "again"})
    assert "already has a completed analysis" in again
    assert len(tools.EXPLOG.relational_frame.processes.df) == 1


def test_enter_rejections():
    bad_ref = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "x", "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
        "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": "nope1234"})
    assert bad_ref.startswith("Error: Reference ID")
    common = {"reason_or_hypothesis": "x", "gap_um": 0.4, "wg1_width_um": 0.45,
              "wg2_width_um": 0.45, "design_ref": ""}
    assert "outside the study window" in tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 2.0, "core_material": "Si", "clad_material": "SiO2"})
    assert "must exceed cladding" in tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 1.55, "core_material": "SiO2", "clad_material": "Si"})
    assert "Unknown core material" in tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 1.55, "core_material": "Unobtainium", "clad_material": "SiO2"})
    # explicit indices allow an off-library material, height then required
    msg = tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 1.55, "core_material": "Unobtainium", "clad_material": "SiO2",
         "n_core": 2.0})
    assert "core_height_um is required" in msg
    ok = tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 1.55, "core_material": "Unobtainium", "clad_material": "SiO2",
         "n_core": 2.0, "core_height_um": 0.3})
    assert "dc_0001" in ok
    dup = tools.enter_design_in_log.invoke(
        {**common, "wavelength_um": 1.55, "core_material": "Unobtainium", "clad_material": "SiO2",
         "n_core": 2.0, "core_height_um": 0.3})
    assert dup.startswith("Not entered") and "dc_0001" in dup
    assert len(tools.EXPLOG.relational_frame.candidates.df) == 1


def test_an_unviable_design_is_recorded_and_only_abandon_allowed(monkeypatch):
    def boom(params, *a, **kw):
        raise DesignNotViable("waveguide 1 has no guided mode at 1.55 um")
    monkeypatch.setattr(tools, "solve_in_subprocess", boom)
    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "x", "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
        "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
    ref = _ref_from(msg, "Message_ID")
    out = tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": ref, "note": "n"})
    assert "NOT VIABLE" in out and "no guided mode" in out
    cdf = tools.EXPLOG.relational_frame.candidates.df
    assert cdf.loc[0, "state"] == "failed"
    tools.get_disposition_info.invoke({"candidate_id": "dc_0001"})
    rej = tools.update_disposition_info.invoke({
        "candidate_id": "dc_0001", "Analysis_and_implications": "a", "Analyzed_process_id": [0],
        "Future_plan": "f", "Decision": "Medium priority"})
    assert "FAILED" in rej and "Abandon" in rej
    ok = tools.update_disposition_info.invoke({
        "candidate_id": "dc_0001", "Analysis_and_implications": "a", "Analyzed_process_id": [0],
        "Future_plan": "f", "Decision": "Abandon"})
    assert "Recorded disposition" in ok


def test_time_budget_reports_remaining_and_percent_used():
    f = tools.format_time_budget
    assert "No time budget is set" in f(300, None) and "No time budget is set" in f(300, 0)
    under = f(900, 1800)
    assert "0:15:00 of the 0:30:00 budget" in under and "50.0% used" in under
    assert "0:15:00 remaining" in under and "nearly exhausted" not in under


def test_time_budget_warns_near_the_limit_and_reports_overrun():
    f = tools.format_time_budget
    warn = f(1440, 1800)                                   # 80 %: the warn threshold
    assert "80.0% used" in warn and "0:06:00 remaining" in warn and "nearly exhausted" in warn
    assert "nearly exhausted" in f(1800, 1800)             # exactly at the limit
    over = f(2400, 1800)
    assert "OVER BUDGET by 0:10:00" in over and "133.3% used" in over
    assert "Do NOT start new designs" in over and "remaining" not in over
    assert "OVER BUDGET by 1:30:00" in f(7200, 1800)       # far over, still sane


def test_time_budget_tolerates_a_missing_or_negative_clock():
    f = tools.format_time_budget
    assert "0:00:00" in f(-5, 1800) and "0.0% used" in f(-5, 1800)
    assert "0:00:00" in f(None, 1800)


def test_check_time_tool_uses_the_live_budget(monkeypatch):
    monkeypatch.setattr(var, "STUDY_BUDGET_SECONDS", 1800)
    monkeypatch.setattr(var, "startTime", time.time() - 900)
    out = tools.check_time.invoke({})
    assert "of the 0:30:00 budget" in out and "50.0% used" in out
    # an un-started clock must not raise
    monkeypatch.setattr(var, "startTime", "")
    assert "0:00:00" in tools.check_time.invoke({})


def test_disposition_readout_shows_results_and_metrics(monkeypatch):
    monkeypatch.setattr(tools, "solve_in_subprocess", _fixed_result)
    msg, ref = None, None
    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "x", "gap_um": 0.2, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
        "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
    ref = _ref_from(msg, "Message_ID")
    # nothing finished yet -> no table
    assert "Results awaiting analysis" not in tools.get_disposition_info.invoke({"candidate_id": "dc_0001"})
    tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": ref, "note": "n"})
    g = tools.get_disposition_info.invoke({"candidate_id": "dc_0001"})
    assert "Results awaiting analysis" in g and "target coupling length 20 um" in g
    assert "coupling_length_deviation_um" in g and "excess_loss_dB" in g and "10.0" in g   # |10-20| = 10


def test_search_finds_run_artifacts_by_metric_value(monkeypatch):
    monkeypatch.setattr(tools, "solve_in_subprocess", _fixed_result)
    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "x", "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
        "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
    out = tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": _ref_from(msg, "Message_ID"), "note": "n"})
    run_ref = _ref_from(out, "Result_ID")
    by_key = tools.search_artifacts.invoke({"query": ["excess_loss_dB", "0.13"], "filters": {"tool_name": "run_coupled_mode_analysis"}, "order": "descending"})
    assert [r["result_id"] for r in by_key["results"]] == [run_ref]
    by_num = tools.search_artifacts.invoke({"query": 20.0, "filters": {"tool_name": "run_coupled_mode_analysis"}, "order": "descending"})
    assert [r["result_id"] for r in by_num["results"]] == [run_ref]          # coupling_length_um == 20.0


@pytest.mark.parametrize("exc", [
    CalculationError("the solver process exited with code -11 without producing a result"),
    CalculationTimeout("the solve exceeded its 120 s limit and was cancelled"),
])
def test_an_environment_fault_records_nothing_and_leaves_the_design_retryable(monkeypatch, exc):
    """The 2026-10-03 regression: a threading fault was recorded as nine failed
    designs, which the terminal-tag gate then forced the agent to Abandon."""
    def boom(params, *a, **kw):
        raise exc
    monkeypatch.setattr(tools, "solve_in_subprocess", boom)
    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "x", "gap_um": 0.4, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
        "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
    ref = _ref_from(msg, "Message_ID")
    out = tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": ref, "note": "n"})

    assert "could NOT BE RUN" in out and "do NOT disposition it as" in out
    assert "NOT VIABLE" not in out
    # nothing written: no process row, candidate still active and un-flagged
    assert len(tools.EXPLOG.relational_frame.processes.df) == 0
    cdf = tools.EXPLOG.relational_frame.candidates.df
    assert cdf.loc[0, "state"] == "active" and not bool(cdf.loc[0, "needs_disposition_update"])
    assert tools.EXPLOG.candidates_needing_disposition() == []
    # and it can simply be retried once the environment is healthy again
    monkeypatch.setattr(tools, "solve_in_subprocess", _fixed_result)
    retry = tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": ref, "note": "retry"})
    assert "completed" in retry
    assert tools.EXPLOG.relational_frame.candidates.df.loc[0, "state"] == "active"


def test_query_and_read_and_reminder(monkeypatch):
    monkeypatch.setattr(tools, "solve_in_subprocess", _fixed_result)
    refs = []
    for gap in (0.2, 0.4):
        msg = tools.enter_design_in_log.invoke({
            "reason_or_hypothesis": f"gap {gap}", "gap_um": gap, "wg1_width_um": 0.45, "wg2_width_um": 0.45,
            "wavelength_um": 1.55, "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
        refs.append(_ref_from(msg, "Message_ID"))
    tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0001", "candidate_id_ref": refs[0], "note": "n"})
    out2 = tools.run_coupled_mode_analysis.invoke({"candidate_id": "dc_0002", "candidate_id_ref": refs[1], "note": "n"})
    # the second run reminds about the first, still-undispositioned design
    reminder = out2.split("REMINDER")[1].split("\n")[0]
    assert "dc_0001" in reminder and "dc_0002" not in reminder

    q = tools.query_explog.invoke({
        "table_name": "candidates", "reason": "sorted by L_c",
        "filters": [{"column": "n_started", "op": "ge", "value": 1}],
        "sort": [{"column": "coupling_length_um", "ascending": False}]})
    assert "disposition_record" not in q and "ready_for_disposition_update" not in q
    assert q.index("dc_0002") < q.index("dc_0001")       # 20*0.4/0.4=20 > 10
    assert "Showing rows 0-1 of 2 total" in q

    r = tools.read_explog.invoke({"candidate_id": "dc_0001", "reasons": "check"})
    assert "gap_um" in r and "coupling_mode_analysis" not in r and "coupled_mode_analysis" in r
    assert "No candidate" in tools.read_explog.invoke({"candidate_id": "dc_0099", "reasons": "x"})
