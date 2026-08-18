"""_stream_agent: recover when LangGraph swallows the prompt on a resume.

The 2026-08-18 outage. On the first super-step after a resume LangGraph stamps
CONFIG_KEY_RESUMING ("__pregel_resuming") into every node config
(pregel/_loop.py:727-730). Node configs are what planNexe2 forwards verbatim as
`inner_cfg = {**config, ...}`, so the inner agent sees the flag too -- and the
subgraph's resume branch and apply-input branch are mutually exclusive
(pregel/_loop.py:682-693). The resume branch wins and the prompt is DISCARDED.

That is fine while the replayed checkpoint still has something runnable (this is
how a worker killed mid-round picks up after its last tool call). It is fatal
once the checkpoint is exhausted: no tasks, tick() returns False immediately,
the stream yields NOTHING and raises NOTHING, and the caller's loop variable is
never bound:

    UnboundLocalError: cannot access local variable 'agent_response'

_stream_agent turns a zero-update stream into one retry with the flag cleared.
The flag must NOT be cleared unconditionally -- the worker's mid-round resume
depends on it -- which test_flag_is_left_alone_on_the_first_attempt pins.

Only a fake agent is needed: the helper just calls .stream().
"""

import pytest

RESUMING = "__pregel_resuming"


@pytest.fixture(scope="module")
def stream_agent(): 
    pytest.importorskip("gnome_dreams_oer_screening")
    from src import planNexe2
    return planNexe2


@pytest.fixture
def helper(stream_agent, monkeypatch):
    """_stream_agent with its two output side effects silenced."""
    monkeypatch.setattr(stream_agent, "print_stream", lambda *a, **k: None)
    monkeypatch.setattr(stream_agent, "write_history", lambda *a, **k: None)
    return stream_agent._stream_agent


class FakeAgent:
    """Yields one batch of chunks per .stream() call, recording each call."""

    def __init__(self, *batches):
        self.batches = [list(b) for b in batches]
        self.calls = []

    def stream(self, payload, cfg, **kw):
        self.calls.append({"payload": payload, "cfg": cfg, "kw": kw})
        for chunk in (self.batches.pop(0) if self.batches else []):
            yield chunk


def _chunk(value):
    """LangGraph 'updates' mode hands back {node_name: update}."""
    return {"model": value}


CFG = {"configurable": {"thread_id": "1", RESUMING: True, "checkpoint_ns": "Supervisor:x"}}


# --- the normal path is unchanged -------------------------------------------

def test_returns_the_last_chunk_and_calls_once(helper):
    agent = FakeAgent([_chunk({"structured_response": "first"}),
                       _chunk({"structured_response": "last"})])
    out = helper(agent, "prompt", CFG)
    assert out == {"structured_response": "last"}
    assert len(agent.calls) == 1


def test_flag_is_left_alone_on_the_first_attempt(helper):
    """THE safety property. The worker's mid-round resume depends on this flag,
    so a healthy stream must never have it stripped."""
    agent = FakeAgent([_chunk({"structured_response": "ok"})])
    helper(agent, "prompt", CFG)
    assert agent.calls[0]["cfg"]["configurable"][RESUMING] is True


def test_stream_mode_and_durability_are_pinned(helper):
    """Without stream_mode the propagated parent config flips chunks to
    'values' mode, which breaks every caller's ['structured_response'] access."""
    agent = FakeAgent([_chunk({"structured_response": "ok"})])
    helper(agent, "prompt", CFG)
    assert agent.calls[0]["kw"]["stream_mode"] == "updates"
    assert agent.calls[0]["kw"]["durability"] == "sync"


def test_prompt_is_delivered_as_a_user_message(helper):
    agent = FakeAgent([_chunk({"structured_response": "ok"})])
    helper(agent, "hello supervisor", CFG)
    assert agent.calls[0]["payload"] == {"messages": [("user", "hello supervisor")]}


# --- the recovery path -------------------------------------------------------

def test_zero_updates_retries_once_with_the_flag_cleared(helper):
    """THE regression: first stream yields nothing (prompt swallowed), so the
    helper resends it without __pregel_resuming."""
    agent = FakeAgent([], [_chunk({"structured_response": "recovered"})])
    out = helper(agent, "prompt", CFG)
    assert out == {"structured_response": "recovered"}
    assert len(agent.calls) == 2
    assert RESUMING in agent.calls[0]["cfg"]["configurable"]
    assert RESUMING not in agent.calls[1]["cfg"]["configurable"]


def test_the_retry_keeps_every_other_config_key(helper):
    """Only the resume flag is dropped -- thread_id and checkpoint_ns must
    survive or the retry would address a different checkpoint entirely."""
    agent = FakeAgent([], [_chunk({"structured_response": "recovered"})])
    helper(agent, "prompt", CFG)
    retry = agent.calls[1]["cfg"]["configurable"]
    assert retry["thread_id"] == "1"
    assert retry["checkpoint_ns"] == "Supervisor:x"


def test_the_caller_config_is_not_mutated(helper):
    agent = FakeAgent([], [_chunk({"structured_response": "recovered"})])
    helper(agent, "prompt", CFG)
    assert CFG["configurable"][RESUMING] is True


def test_two_empty_streams_raise_a_named_error(helper):
    """Better than UnboundLocalError pointing at the wrong line."""
    agent = FakeAgent([], [])
    with pytest.raises(RuntimeError, match=RESUMING):
        helper(agent, "prompt", CFG)
    assert len(agent.calls) == 2


def test_print_kwargs_are_forwarded(helper, stream_agent, monkeypatch):
    """The worker passes DAG=step_no through to print_stream."""
    seen = []
    monkeypatch.setattr(stream_agent, "print_stream",
                        lambda chunk, **kw: seen.append(kw))
    agent = FakeAgent([_chunk({"structured_response": "ok"})])
    stream_agent._stream_agent(agent, "prompt", CFG, DAG=7)
    assert seen == [{"DAG": 7}]


def test_a_config_without_configurable_still_works(helper):
    """inner_cfg is built from an arbitrary parent config; don't assume shape."""
    agent = FakeAgent([], [_chunk({"structured_response": "ok"})])
    out = helper(agent, "prompt", {"recursion_limit": 1000})
    assert out == {"structured_response": "ok"}
