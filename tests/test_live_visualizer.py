"""Live visualizer compatibility with the FEM explog + agent topology.

Drives LiveVisualizer with the event shape planNexe2.print_stream actually
produces (the {node: update} envelope is already unwrapped there) and checks the
payload it writes for the browser.
"""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src import var
from src.fem_explog import FemExplog, METRIC_COLUMNS, RESULT_COLUMNS
from src.live_visualizer import LiveVisualizer


def _result(**over):
    r = {k: 1.0 for k in RESULT_COLUMNS}
    r.update(coupling_length_um=23.4, max_power_transfer=0.97)
    r.update(over)
    return r


def _design():
    return {"gap_um": 0.2, "wg1_width_um": 0.45, "wg2_width_um": 0.45, "core_height_um": 0.22,
            "wavelength_um": 1.55, "core_material": "Si", "n_core": 3.4777,
            "clad_material": "SiO2", "n_clad": 1.444}


@pytest.fixture
def populated(tmp_path):
    """An explog with one computed, dispositioned design."""
    log = FemExplog()
    log.init(tmp_path, disposition_decisions=var.DISPOSITION_DECISIONS, target_coupling_length_um=20.0)
    cid = log.add_candidate(_design(), "baseline")
    pid = log.record_analysis(cid, "n", result=_result(), error=None)
    log.get_disposition_info(cid)
    log.update_disposition_info(cid, "SECRETTEXT " * 50, [pid], "sweep", "High priority")
    return log


def _payload(tmp_path):
    js = (tmp_path / "live_data.js").read_text()
    return json.loads(js[js.index("{"): js.rindex(";\nwindow.__LIVE_TICK__")])


def _drive(viz):
    """One full round: supervisor plan -> worker tool + response -> boss approve."""
    def ev(msg, DAG=None):
        viz.on_event({"messages": [msg]}, DAG=DAG)

    def structured(name, args, tid, inp, outp):
        ev(AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": tid}],
                     usage_metadata={"input_tokens": inp, "output_tokens": outp, "total_tokens": inp + outp}))
        ev(ToolMessage(content="ok", tool_call_id=tid, name=name))

    structured("Act", {"action": {"kind": "plan", "steps": [
        {"step": "Sweep the gap", "agent": "FEM_Agent", "required_tools": ["run_coupled_mode_analysis"]}]}},
        "t1", 1200, 300)
    ev(AIMessage(content="running", tool_calls=[{"name": "run_coupled_mode_analysis", "args": {"candidate_id": "dc_0001"}, "id": "t2"}],
                 usage_metadata={"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020}))
    ev(ToolMessage(content="completed (process_id 0)", tool_call_id="t2", name="run_coupled_mode_analysis"))
    structured("wokerResponse", {"answer": "a", "summary": "s", "success": True}, "t3", 1000, 80)
    structured("BossReview", {"decision": "approve", "feedback": ""}, "t4", 500, 40)
    viz.close()


def test_explog_tables_carry_the_metrics_and_hide_internal_columns(populated, tmp_path):
    viz = LiveVisualizer(explog_obj=populated)
    viz.set_working_directory(str(tmp_path))
    data = _payload(tmp_path)
    assert data["explog"].get("_error") is None
    cand = data["explog"]["candidates"][0]
    proc = data["explog"]["processes"][0]
    for col in METRIC_COLUMNS:
        assert isinstance(cand[col], float) and isinstance(proc[col], float)
    assert cand["coupling_length_deviation_um"] == pytest.approx(3.4)
    assert cand["decision"] == "High priority"
    # agent-internal columns are dropped server-side, not merely hidden in the UI
    assert not [k for k in cand if k in ("disposition_record", "ready_for_disposition_update")]
    assert "SECRETTEXT" not in (tmp_path / "live_data.js").read_text()
    assert "SECRETTEXT" not in (tmp_path / "live_visualization.html").read_text()
    # every cell is JSON-native (no pandas NA / numpy scalars leaking through)
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in cand.values())


def test_steps_are_attributed_to_the_right_agent(populated, tmp_path):
    viz = LiveVisualizer(explog_obj=populated)
    viz.set_working_directory(str(tmp_path))
    _drive(viz)
    data = _payload(tmp_path)
    assert [(s["agent"], s["structured_response"]["type"]) for s in data["steps"]] == [
        ("Supervisor", "Act"), ("FEM_Agent", "wokerResponse"), ("Boss", "BossReview")]
    # the worker's name is learned from the plan, never hard-coded
    assert viz.last_planned_agent == "FEM_Agent" and "FEM_Agent" in viz.known_agents
    assert data["steps"][1]["title"].startswith("FEM_Agent")
    assert data["tokens_total"] == {"input": 3600, "output": 540}


def test_resume_restores_history_and_relearns_the_worker(populated, tmp_path):
    first = LiveVisualizer(explog_obj=populated)
    first.set_working_directory(str(tmp_path))
    _drive(first)

    resumed = LiveVisualizer(explog_obj=populated)
    resumed.set_working_directory(str(tmp_path))
    assert len(resumed.steps) == 3
    assert resumed.last_planned_agent == "FEM_Agent"
    assert resumed.tokens_total == {"input": 3600, "output": 540}


def test_a_broken_explog_degrades_instead_of_killing_the_run(tmp_path):
    class Broken:
        @property
        def relational_frame(self):
            raise RuntimeError("explog exploded")

    viz = LiveVisualizer(explog_obj=Broken())
    viz.set_working_directory(str(tmp_path))
    assert "explog exploded" in _payload(tmp_path)["explog"]["_error"]
