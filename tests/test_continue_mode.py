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


def _build(boss_decision="approve", supervisor_action="respond"):
    """Mirror of the real routing. The boss approves by default, so the graph
    runs straight to END -- the state `continue` has to rescue.

    supervisor_action="plan" models the supervisor's NORMAL behaviour: it hands
    work to the worker instead of proposing a final answer, so the boss does not
    run at all. Both real paths consume boss_feedback themselves (planNexe2's
    supervisor_chain_node returns boss_feedback="" on every exit), which is what
    makes the directive genuinely one-shot -- reaching the boss is NOT what
    clears it."""
    def supervisor(state):
        # "plan" mode: delegate to the worker whenever there is feedback to act
        # on, and only propose a draft once there is none left. That is the real
        # shape in miniature -- work first, answer later -- and it keeps the mock
        # terminating instead of ping-ponging with the worker forever.
        if supervisor_action == "plan" and state["boss_feedback"].strip():
            return {"boss_feedback": "", "next": "OER_Agent"}
        return {"draft_response": "final report", "boss_feedback": "",
                "next": "Boss_Agent"}

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


def test_continue_can_replace_the_objective():
    """`inputs` is a plain LastValue channel, so the same update_state that
    injects the directive can swap the objective outright."""
    graph, cfg = _build()
    _run_to_finish(graph, cfg)
    assert graph.get_state(cfg).values["inputs"] == "obj"

    new_objective = "Cover the database broadly; do not stop while budget remains."
    graph.update_state(
        cfg,
        {"next": "Supervisor", "boss_feedback": "resumed", "inputs": new_objective},
        as_node="Boss_Agent")

    st = graph.get_state(cfg)
    assert st.next == ("Supervisor",)
    assert st.values["inputs"] == new_objective


def test_the_objective_outlives_the_directive():
    """THE reason continue_objective.md exists as a separate file.

    `inputs` is re-rendered into every supervisor and boss prompt for the rest of
    the run; boss_feedback is consumed by the supervisor round that reads it.
    Standing policy put in the directive is heard once; standing policy put in the
    objective is heard always. This pins that asymmetry, so a refactor that made
    `inputs` consumable (or boss_feedback sticky again) fails loudly.

    NOTE: the supervisor here PLANS rather than proposing a draft, which is what
    the real one does for weeks at a time. An earlier version of this test used a
    supervisor that answered immediately, so the boss ran and cleared the field --
    and the test passed while the real graph leaked the directive into every
    round for the rest of the run, because nothing on the planning path cleared
    it. The clearing must be the SUPERVISOR's, not a side effect of reaching the
    boss."""
    graph, cfg = _build(supervisor_action="plan")
    _run_to_finish(graph, cfg)

    new_objective = "Broad coverage is a primary goal."
    graph.update_state(
        cfg,
        {"next": "Supervisor", "boss_feedback": "one-shot text", "inputs": new_objective},
        as_node="Boss_Agent")

    # one supervisor round that PLANS -- the boss never runs
    it = graph.stream(None, cfg, durability="sync")
    next(it)
    it.close()

    st = graph.get_state(cfg)
    assert st.values["boss_feedback"] == "", (
        "the supervisor must consume boss_feedback when it plans; otherwise the "
        "directive is re-shown every round until the boss next approves"
    )
    assert st.values["inputs"] == new_objective, "the objective must persist"


def test_directive_does_not_survive_into_later_planning_rounds():
    """The failure this guards against is silent: the supervisor keeps being told
    'your previous draft final answer has been rejected' with one-time marching
    orders attached, every round, for the rest of the study."""
    graph, cfg = _build(supervisor_action="plan")
    _run_to_finish(graph, cfg)
    graph.update_state(
        cfg, {"next": "Supervisor", "boss_feedback": "INGEST THE FINISHED RESULTS"},
        as_node="Boss_Agent")

    seen = []
    it = graph.stream(None, cfg, durability="sync")
    for _ in range(4):
        try:
            next(it)
        except StopIteration:
            break
        seen.append(graph.get_state(cfg).values["boss_feedback"])
    it.close()

    assert seen, "the graph did not advance"
    assert all(f == "" for f in seen[1:]), (
        f"directive persisted across rounds: {seen}"
    )


def test_omitting_the_objective_leaves_it_untouched():
    """No continue_objective.md -> invoke.py omits the key entirely, and the run
    keeps the objective it started with."""
    graph, cfg = _build()
    _run_to_finish(graph, cfg)

    graph.update_state(cfg, {"next": "Supervisor", "boss_feedback": "resumed"},
                       as_node="Boss_Agent")

    assert graph.get_state(cfg).values["inputs"] == "obj"


def test_continue_is_refused_on_an_unfinished_run():
    """invoke.py guards on next != (); this pins the condition it checks.

    Without `--force` that guard is a hard SystemExit -- a plain resume is the
    supported way to carry on an interrupted run."""
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


def test_forced_continue_supersedes_an_interrupted_rounds_pending_task():
    """`continue --force` on a run killed MID-ROUND.

    This is the assumption Change 3 rests on: update_state APPENDS a checkpoint
    rather than rewinding, so writing next="Supervisor" as the boss leaves the
    interrupted round's pending task behind in a no-longer-head checkpoint --
    superseded, never executed twice -- while the directive and the replacement
    objective both land. Without this property, forcing the guard would corrupt
    the run instead of restarting the round.
    """
    graph, cfg = _build(boss_decision="revise")
    it = graph.stream({"inputs": "obj", "draft_response": "", "boss_feedback": "",
                       "response": "", "next": ""}, cfg, durability="sync")
    next(it)
    it.close()
    interrupted = graph.get_state(cfg)
    assert interrupted.next != ()          # mid-round: a task is pending

    graph.update_state(
        cfg,
        {"next": "Supervisor", "boss_feedback": "operator directive",
         "inputs": "replacement objective"},
        as_node="Boss_Agent",
    )

    after = graph.get_state(cfg)
    assert tuple(after.next) == ("Supervisor",), (
        "the forced write must leave exactly one pending Supervisor task -- the "
        "restarted round -- not the interrupted one plus a new one"
    )
    assert after.values["boss_feedback"] == "operator directive"
    assert after.values["inputs"] == "replacement objective", (
        "replacing the objective is the whole reason --force exists: a plain "
        "resume reads neither continue file"
    )
    assert after.config["configurable"]["checkpoint_id"] != \
        interrupted.config["configurable"]["checkpoint_id"], (
        "update_state must APPEND a new checkpoint, not rewrite the head"
    )
