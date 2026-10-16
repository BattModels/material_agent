"""Regression tests for the bug that killed the 2026-10-03 run.

Every design failed with ``ValueError: signal only works in main thread of the
main interpreter``. Chain: invoke.py compiles the graph with a checkpointer and
streams ``durability="sync"``, so LangGraph runs node tasks (and therefore tool
calls) on ThreadPoolExecutor workers -> pygmsh calls ``gmsh.initialize()``, which
defaults to ``interruptible=True`` and installs a SIGINT handler -> CPython allows
``signal.signal`` only on the main thread.

The existing suite missed it because everything ran on the main thread and the
tool tests stubbed the calculator out. These tests exercise the real thing from a
worker thread.
"""
import operator
import threading
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

import src.tools as tools
from src import var
from src.fem_calculator import compute_coupled_modes, solve_in_subprocess
from src.myCANVAS import CANVAS

DESIGN = dict(gap_um=0.2, wg1_width_um=0.45, wg2_width_um=0.45, core_height_um=0.22,
              wavelength_um=1.55, n_core=3.4777, n_clad=1.444)
EXPECTED_LC = 23.353969664            # the value this design yields on the main thread


def _in_thread(fn):
    """Run fn() on a non-main thread; return (result, exception, was_main)."""
    box = {}

    def body():
        box["main"] = threading.current_thread() is threading.main_thread()
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert box["main"] is False, "the probe did not actually run off the main thread"
    return box.get("result"), box.get("error")


def test_in_process_solve_still_fails_off_the_main_thread():
    """Documents WHY the solve is out-of-process. If a future gmsh makes this
    pass, the subprocess hop becomes optional rather than required."""
    result, error = _in_thread(lambda: compute_coupled_modes(DESIGN))
    assert result is None
    assert isinstance(error, ValueError)
    assert "signal only works in main thread" in str(error)


@pytest.mark.slow
def test_subprocess_solve_succeeds_off_the_main_thread():
    """The fix: the child does its work on its own main thread."""
    result, error = _in_thread(lambda: solve_in_subprocess(DESIGN))
    assert error is None, f"unexpected {type(error).__name__}: {error}"
    assert result["coupling_length_um"] == pytest.approx(EXPECTED_LC, abs=1e-6)


@pytest.mark.slow
def test_the_tool_works_through_a_checkpointed_langgraph(tmp_path, monkeypatch):
    """End-to-end in the shape that actually broke: ToolNode inside a graph with
    a checkpointer and durability="sync", which is what invoke.py builds."""
    tools.EXPLOG.init(tmp_path, disposition_decisions=var.DISPOSITION_DECISIONS,
                      target_coupling_length_um=20.0)
    CANVAS.set_working_directory(str(tmp_path))
    CANVAS.canvas.clear(); CANVAS.result_registry.clear(); CANVAS.curr_round_result_ids.clear()
    monkeypatch.setattr(var, "my_WORKING_DIRECTORY", str(tmp_path))

    msg = tools.enter_design_in_log.invoke({
        "reason_or_hypothesis": "worker-thread regression", "gap_um": 0.2,
        "wg1_width_um": 0.45, "wg2_width_um": 0.45, "wavelength_um": 1.55,
        "core_material": "Si", "clad_material": "SiO2", "design_ref": ""})
    ref = [t for t in msg.split() if t.startswith("Message_ID=")][0].split("=", 1)[1].rstrip(".")

    threads = []
    real = tools.solve_in_subprocess

    def recording_solve(params, *a, **kw):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(params, *a, **kw)

    monkeypatch.setattr(tools, "solve_in_subprocess", recording_solve)

    class S(TypedDict):
        messages: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("tools", ToolNode([tools.run_coupled_mode_analysis]))
    g.add_edge(START, "tools"); g.add_edge("tools", END)
    graph = g.compile(checkpointer=MemorySaver())

    call = AIMessage(content="", tool_calls=[{
        "name": "run_coupled_mode_analysis", "id": "t1",
        "args": {"candidate_id": "dc_0001", "candidate_id_ref": ref, "note": "regression"}}])
    out = None
    for chunk in graph.stream({"messages": [call]}, {"configurable": {"thread_id": "1"}},
                              durability="sync"):
        out = chunk
    text = out["tools"]["messages"][-1].content

    assert threads == [False], "the tool did not run on a worker thread; test is not exercising the bug"
    assert "completed" in text and "signal only works in main thread" not in text
    pdf = tools.EXPLOG.relational_frame.processes.df
    assert len(pdf) == 1 and pdf.loc[0, "status"] == "completed"

    # The number must not depend on HOW it was produced: solving the same stored
    # design directly, in-process and on the main thread, has to agree exactly.
    reference = compute_coupled_modes(tools.EXPLOG.design_of("dc_0001"))
    assert float(pdf.loc[0, "coupling_length_um"]) == pytest.approx(
        reference["coupling_length_um"], abs=1e-9)
    assert float(pdf.loc[0, "n_eff_symmetric"]) == pytest.approx(
        reference["n_eff_symmetric"], abs=1e-9)
