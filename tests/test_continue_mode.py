"""The `python invoke.py continue "<directive>"` contract.

When the boss approves, whos_next routes to END and the head checkpoint has
next=() -- a plain resume then executes ZERO super-steps, so there is no way to
restart the study without putting a task back on the graph.

`continue` does that by writing the operator's directive as if the boss had
REJECTED the draft. These tests pin the LangGraph behaviour that makes it work,
on a graph mirroring create_planning_graph's routing:

    START -> Supervisor -> (conditional on state["next"]) -> OER_Agent
                                                           | Boss_Agent
                                                           | END
    OER_Agent  -> Supervisor
    Boss_Agent -> (conditional) -> Supervisor | END

Fast: InMemorySaver, no LLM, no GNoME import. Guards against a langgraph
upgrade quietly changing update_state(as_node=...) semantics.
"""

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class _S(TypedDict):
    inputs: str
    draft_response: str
    boss_feedback: str
    response: str
    next: str


def _build(boss_decision="approve"):
    """Mirror of the real routing. The boss approves by default, so the graph
    runs straight to END -- the state `continue` has to rescue."""
    def supervisor(state):
        return {"draft_response": "final report", "next": "Boss_Agent"}

    def oer(state):
        return {}

    def boss(state):
        if boss_decision == "approve":
            return {"response": state["draft_response"], "boss_feedback": "",
                    "next": "FINISH"}
        return {"boss_feedback": "needs work", "next": "Supervisor"}

    g = StateGraph(_S)
    g.add_node("OER_Agent", oer)
    g.add_node("Boss_Agent", boss)
    g.add_node("Supervisor", supervisor)
    g.add_edge("OER_Agent", "Supervisor")
    g.add_conditional_edges("Supervisor", lambda s: s["next"],
                            {"OER_Agent": "OER_Agent", "Boss_Agent": "Boss_Agent",
                             "FINISH": END, "Supervisor": "Supervisor"})
    g.add_conditional_edges("Boss_Agent", lambda s: s["next"],
                            {"FINISH": END, "Supervisor": "Supervisor"})
    g.add_edge(START, "Supervisor")
    graph = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
    return graph, cfg


def _run_to_finish(graph, cfg):
    for _ in graph.stream({"inputs": "obj", "draft_response": "", "boss_feedback": "",
                           "response": "", "next": ""}, cfg, durability="sync"):
        pass


def test_finished_run_is_a_dead_end_for_plain_resume():
    """The premise: this is why `continue` has to exist at all."""
    graph, cfg = _build()
    _run_to_finish(graph, cfg)
    assert graph.get_state(cfg).next == ()

    steps = sum(1 for _ in graph.stream(None, cfg, durability="sync"))
    assert steps == 0, "a finished run must not resume on its own"


def test_continue_rearms_the_supervisor():
    graph, cfg = _build()
    _run_to_finish(graph, cfg)

    directive = "OPERATOR: budget remains; expand the study."
    newcfg = graph.update_state(
        cfg, {"next": "Supervisor", "boss_feedback": directive},
        as_node="Boss_Agent")

    st = graph.get_state(cfg)
    assert st.next == ("Supervisor",)
    assert st.values["boss_feedback"] == directive
    # a NEW checkpoint, so nothing is rewound and the finished run is preserved
    assert newcfg["configurable"]["checkpoint_id"] != \
        cfg["configurable"].get("checkpoint_id")


def test_continue_actually_executes_on_resume():
    graph, cfg = _build()
    _run_to_finish(graph, cfg)
    graph.update_state(cfg, {"next": "Supervisor", "boss_feedback": "go on"},
                       as_node="Boss_Agent")

    steps = sum(1 for _ in graph.stream(None, cfg, durability="sync"))
    assert steps > 0, "supervisor did not run after the continue injection"


def test_finished_response_is_preserved():
    """The approved final answer must survive -- `continue` appends, never rewinds."""
    graph, cfg = _build()
    _run_to_finish(graph, cfg)
    assert graph.get_state(cfg).values["response"] == "final report"

    graph.update_state(cfg, {"next": "Supervisor", "boss_feedback": "more"},
                       as_node="Boss_Agent")
    assert graph.get_state(cfg).values["response"] == "final report"


def test_continue_is_refused_on_an_unfinished_run():
    """invoke.py guards on next != (); this pins the condition it checks."""
    graph, cfg = _build(boss_decision="revise")
    # boss keeps rejecting -> recursion limit, so just take one super-step
    it = graph.stream({"inputs": "obj", "draft_response": "", "boss_feedback": "",
                       "response": "", "next": ""}, cfg, durability="sync")
    next(it)
    it.close()
    assert graph.get_state(cfg).next != (), (
        "an in-flight run must report a pending task, which is what invoke.py "
        "refuses `continue` on"
    )
