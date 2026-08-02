# The '#' header on continue_directive.md / continue_objective.md must never
# reach the agents.
#
# invoke.py used to do a plain read().strip(), so the objective the supervisor
# was handed began: "The overall goal is: # REPLACEMENT USER PROMPT
# (state["inputs"]) for a resumed run # # This is the OBJECTIVE ..." -- 21 lines
# of notes about LangGraph checkpoints presented as the research goal, repeated
# in every prompt for the rest of the run (the objective, unlike the directive,
# is never consumed).
#
# These tests cover read_operator_message AND the two real files, so a header
# added later without re-reading this cannot silently leak.

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _load_read_operator_message():
    """Import the helper WITHOUT executing invoke.py's __main__ block (which
    would load the GNoME database and build the graph)."""
    src = (_REPO / "invoke.py").read_text()
    start = src.index("def read_operator_message(")
    end = src.index("\nMAX_CHECKPOINT_DB_BYTES")
    ns = {}
    exec(compile(src[start:end], "invoke.py", "exec"), ns)
    return ns["read_operator_message"]


read_operator_message = _load_read_operator_message()


def _write(tmp_path, text):
    p = tmp_path / "msg.md"
    p.write_text(text)
    return str(p)


def test_strips_the_leading_comment_header(tmp_path):
    path = _write(tmp_path, "# a note\n# another note\n\nReal message.\n")
    assert read_operator_message(path) == "Real message."


def test_keeps_hashes_inside_the_body(tmp_path):
    """Only the CONTIGUOUS LEADING block goes -- a '#' further down is content."""
    path = _write(tmp_path, "# header\n\nDo the thing.\n\n# Section two\nMore.\n")
    out = read_operator_message(path)
    assert out.startswith("Do the thing.")
    assert "# Section two" in out


def test_a_header_only_file_reads_as_empty(tmp_path):
    """invoke.py turns this into a SystemExit rather than blanking the objective."""
    path = _write(tmp_path, "# just a header\n#\n# nothing else\n")
    assert read_operator_message(path) == ""


def test_no_header_is_fine(tmp_path):
    path = _write(tmp_path, "Straight to the point.\n")
    assert read_operator_message(path) == "Straight to the point."


def test_blank_lines_before_the_header_are_tolerated(tmp_path):
    path = _write(tmp_path, "\n\n# header\n\nBody.\n")
    assert read_operator_message(path) == "Body."


# --- the real files ----------------------------------------------------------

@pytest.mark.parametrize("name", ["continue_directive.md", "continue_objective.md"])
def test_real_file_delivers_a_body_and_no_header(name):
    path = _REPO / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this working tree")
    body = read_operator_message(str(path))
    assert body, f"{name} would deliver nothing"
    assert not body.lstrip().startswith("#"), f"{name} leaks its header"
    # the specific strings that used to leak
    assert "state[" not in body.split("\n")[0]
    assert "invoke.py" not in body[:400]


def test_the_objective_opens_as_a_user_request():
    """It is rendered as 'The overall goal is: <this>', so it has to read like an
    instruction from the user -- not like feedback, and not like a file header."""
    path = _REPO / "continue_objective.md"
    if not path.is_file():
        pytest.skip("continue_objective.md not present in this working tree")
    body = read_operator_message(str(path))
    assert body.startswith("Please conduct"), body[:120]
