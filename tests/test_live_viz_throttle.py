"""LiveVisualizer write throttle.

_flush() rewrites BOTH live_data.js and live_visualization.html in full, and
on_event() calls it for EVERY streamed agent event -- at least once per model
call and once per tool call. Measured on the 27-05 run that is ~180 MB per
event (80 MB js + 99 MB self-contained html), i.e. MORE I/O than the
checkpointer, for a dashboard nobody watches during an unattended run.

Both files are views rebuilt from in-memory state, so skipping a write loses
nothing. These tests pin that: throttled in the hot path, forced at the
boundaries, and always reflecting the latest state once a write does happen.
"""

import time

import pytest

from src import var
from src.live_visualizer import LiveVisualizer


@pytest.fixture
def viz(tmp_path, monkeypatch):
    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 999)
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 999)
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))   # forces the first paint
    return v


def _mtimes(v):
    import os
    return (os.path.getmtime(v.data_path), os.path.getmtime(v.html_path))


def test_first_paint_is_forced(viz, tmp_path):
    """Both files must exist immediately, even with the throttle wide open."""
    assert (tmp_path / "live_data.js").exists()
    assert (tmp_path / "live_visualization.html").exists()


def test_hot_path_is_throttled(viz):
    """Many events inside the interval -> no rewrites."""
    before = _mtimes(viz)
    for i in range(50):
        viz.on_event({"messages": [("user", f"event {i}")]})
    assert _mtimes(viz) == before, "throttle did not suppress the hot path"
    assert viz._skipped_flushes == 50


def test_close_always_writes(viz):
    """The final state must land regardless of the throttle."""
    before = _mtimes(viz)
    for i in range(5):
        viz.on_event({"messages": [("user", f"e{i}")]})
    assert _mtimes(viz) == before
    time.sleep(0.01)
    viz.close()
    assert _mtimes(viz) != before, "close() must force a final write"


def test_data_and_html_throttle_independently(tmp_path, monkeypatch):
    """live_data.js is what the page polls, so it refreshes far more often than
    the ~99 MB self-contained html."""
    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 0)     # always
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 999)   # never
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))
    d0, h0 = _mtimes(v)
    time.sleep(0.01)
    v.on_event({"messages": [("user", "x")]})
    d1, h1 = _mtimes(v)
    assert d1 != d0, "live_data.js should have been rewritten"
    assert h1 == h0, "the heavy html should have been skipped"


def test_disable_switch(tmp_path, monkeypatch):
    """LIVE_VIZ_ENABLED=False stops ALL dashboard writes -- including the forced
    ones -- without breaking the run. 'Off' must mean off."""
    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 0)
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))
    monkeypatch.setattr(var, "LIVE_VIZ_ENABLED", False)
    before = _mtimes(v)
    time.sleep(0.01)
    for i in range(10):
        v.on_event({"messages": [("user", f"e{i}")]})
    assert _mtimes(v) == before, "disabled visualiser still wrote"
    v.close()                                   # force=True, but disabled
    assert _mtimes(v) == before, "kill switch must beat force=True"


def test_disabled_from_the_start_writes_nothing(tmp_path, monkeypatch):
    """With it off before the run starts, the files are never created at all."""
    monkeypatch.setattr(var, "LIVE_VIZ_ENABLED", False)
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))      # first paint is forced
    assert not (tmp_path / "live_data.js").exists()
    assert not (tmp_path / "live_visualization.html").exists()
    v.on_event({"messages": [("user", "x")]})   # must not raise
    v.close()


def test_state_is_not_lost_by_throttling(tmp_path, monkeypatch):
    """A skipped write must not drop data: the next write emits the LATEST
    state, because the files are rebuilt from memory rather than appended to."""
    # NB: _handle_message deliberately SKIPS tuples, so a real message object is
    # required for the event to be recorded at all.
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 999)
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 999)
    v = LiveVisualizer()
    v.set_working_directory(str(tmp_path))
    for i in range(20):
        v.on_event({"messages": [HumanMessage(content=f"marker-{i}")]})
    v.close()                                   # forced
    content = (tmp_path / "live_data.js").read_text(encoding="utf-8")
    assert "marker-19" in content, "latest event missing after a throttled run"


def test_errors_in_flush_never_break_the_run(viz, monkeypatch):
    """on_event swallows visualiser errors by design -- the agent must not die
    because a dashboard write failed."""
    monkeypatch.setattr(var, "LIVE_VIZ_DATA_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(var, "LIVE_VIZ_HTML_MIN_INTERVAL_S", 0)
    monkeypatch.setattr(viz, "_build_data",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    viz.on_event({"messages": [("user", "x")]})   # must not raise
