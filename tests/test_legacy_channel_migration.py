# Dropping a channel from the state schema does NOT remove it from a checkpoint.
#
# Commits 87cb597/06d25fb/3bbf619 moved CANVAS, the artifact registry and both
# EXPLOG frames out of PlanExecute and into their own files, cutting the
# per-tool-call checkpoint from ~375 MB to ~25 KB. But the 27-05 head checkpoint
# was written BEFORE that, so it still carries all four -- measured at 139.8 MB,
# 99.9% of the checkpoint -- and LangGraph copies channel_values it does not
# recognise straight into every new checkpoint. Resuming would therefore have
# rewritten 140 MB of dead 22-07 state on every parent super-step, under
# durability="sync", quietly undoing the whole point of those three commits.
#
# The fix: keep the four declared in PlanExecute as VESTIGIAL channels so
# `invoke.py continue` can write empties into them once.
#
# These tests pin the two LangGraph behaviours the fix depends on, both verified
# against langgraph 1.0.9. Either could change under an upgrade, and the failure
# is silent -- no error, just a slow run and a growing database.

from typing import TypedDict

import pickle
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

_BLOB = "x" * 10_000


class _Old(TypedDict):
    inputs: str
    boss_feedback: str
    next: str
    canvas: dict
    artifacts: dict


class _NewDropped(TypedDict):
    """What removing the channels outright looks like."""
    inputs: str
    boss_feedback: str
    next: str


class _NewVestigial(TypedDict):
    """What we actually ship: still declared, so they can be emptied."""
    inputs: str
    boss_feedback: str
    next: str
    canvas: dict
    artifacts: dict


def _graph(schema, saver):
    def supervisor(state):
        return {"next": "FINISH"}
    g = StateGraph(schema)
    g.add_node("Supervisor", supervisor)
    g.add_conditional_edges("Supervisor", lambda s: s["next"],
                            {"FINISH": END, "Supervisor": "Supervisor"})
    g.add_edge(START, "Supervisor")
    return g.compile(checkpointer=saver)


def _seed_old_run(saver, cfg):
    """A finished run whose checkpoint carries the fat legacy channels."""
    for _ in _graph(_Old, saver).stream(
        {"inputs": "obj", "boss_feedback": "", "next": "",
         "canvas": {"big": _BLOB}, "artifacts": {"a": _BLOB}},
        cfg, durability="sync",
    ):
        pass


def _sizes(saver, cfg):
    ch = saver.get_tuple(cfg).checkpoint["channel_values"]
    return {k: len(pickle.dumps(v)) for k, v in ch.items()}


@pytest.fixture
def seeded():
    saver = InMemorySaver()
    cfg = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
    _seed_old_run(saver, cfg)
    assert _sizes(saver, cfg)["canvas"] > 10_000
    return saver, cfg


def test_dropping_a_channel_does_not_drop_it_from_the_checkpoint(seeded):
    """The premise. If this ever fails, LangGraph started pruning unknown
    channels and the vestigial declarations could be removed."""
    saver, cfg = seeded
    graph = _graph(_NewDropped, saver)
    graph.update_state(cfg, {"next": "Supervisor", "boss_feedback": "d"},
                       as_node="Supervisor")
    for _ in graph.stream(None, cfg, durability="sync"):
        pass
    sizes = _sizes(saver, cfg)
    assert sizes.get("canvas", 0) > 10_000, (
        "LangGraph now drops unrecognised channels -- the vestigial channels in "
        "PlanExecute are no longer needed and this migration can be deleted"
    )


def test_writing_an_out_of_schema_channel_is_a_silent_no_op(seeded):
    """Why the fix is 'declare then empty' rather than 'just write empties'.
    update_state raises nothing here -- it accepts the write and ignores it."""
    saver, cfg = seeded
    graph = _graph(_NewDropped, saver)
    graph.update_state(
        cfg, {"next": "Supervisor", "boss_feedback": "d", "canvas": {}, "artifacts": {}},
        as_node="Supervisor")
    assert _sizes(saver, cfg)["canvas"] > 10_000, (
        "out-of-schema writes now take effect; the simpler fix is available"
    )


def test_vestigial_channels_can_be_emptied_and_stay_empty(seeded):
    """The shipped path: what `invoke.py continue` does."""
    saver, cfg = seeded
    graph = _graph(_NewVestigial, saver)
    graph.update_state(
        cfg, {"next": "Supervisor", "boss_feedback": "d", "canvas": {}, "artifacts": {}},
        as_node="Supervisor")

    after = _sizes(saver, cfg)
    assert after["canvas"] < 100, after["canvas"]
    assert after["artifacts"] < 100, after["artifacts"]

    # and they must not come back once the graph runs for real
    for _ in graph.stream(None, cfg, durability="sync"):
        pass
    later = _sizes(saver, cfg)
    assert later["canvas"] < 100, "legacy blob reappeared after a super-step"
    assert later["artifacts"] < 100


def test_planexecute_still_declares_the_vestigial_channels():
    """Guards the actual shipped schema, not just the model of it above.
    Deleting these 'unused' fields silently reintroduces the 140 MB leak."""
    from src.planNexe2 import PlanExecute
    keys = set(PlanExecute.__annotations__)
    assert {"canvas", "artifacts", "explog_candidates", "explog_processes"} <= keys, (
        "vestigial channels removed from PlanExecute -- a resumed run will carry "
        "the pre-3bbf619 CANVAS/EXPLOG blobs in every checkpoint again"
    )
