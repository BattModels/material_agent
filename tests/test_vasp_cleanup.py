"""VASP restart-file cleanup: what it deletes, and that it runs on failures too.

Measured 2026-08-18: 51 failed job directories were holding 172 GiB of
WAVECAR/CHG that nothing would ever remove. `clean_up_vasp_directory` was the
last statement inside update_log's `try`, so any exception -- non-convergence
raises in read_vasp_results, a truncated vasprun.xml, a NaN energy -- jumped to
`except` and skipped it. The fix moves the call into a `finally`, which is
required rather than stylistic: both `except` blocks end in `continue`, so a
call placed after the try/except would be skipped on exactly those failures.

Two cheap angles: what the function does (tmp dirs, no VASP), and where it is
called from (AST, no import).
"""

import ast
from pathlib import Path

import pytest

JOB_HANDLER = (Path(__file__).resolve().parents[1] /
               "GNoME_DREAMS_OER_screening/src/gnome_dreams_oer_screening"
               "/vasp/job_handler.py")


def _make_vasp_dir(tmp_path):
    """A finished job directory: results plus the big restart files."""
    d = tmp_path / "20260818_120000_abc123_bulk_relaxation_1"
    d.mkdir()
    for name in ("INCAR", "KPOINTS", "POSCAR", "CONTCAR", "OSZICAR",
                 "vasprun.xml", "OUTCAR", "vasp.out", "custodian.json",
                 "WAVECAR", "CHG", "CHGCAR", "PROCAR", "DOSCAR"):
        (d / name).write_text(name)
    return d


@pytest.fixture
def clean_up():
    pytest.importorskip("gnome_dreams_oer_screening")
    from gnome_dreams_oer_screening.vasp.vasp_calculation import (
        clean_up_vasp_directory,
    )
    return clean_up_vasp_directory


# --- what it deletes -------------------------------------------------------

def test_restart_files_go_and_results_stay(tmp_path, clean_up):
    d = _make_vasp_dir(tmp_path)
    clean_up(str(d))
    left = {f.name for f in d.iterdir()}
    assert {"WAVECAR", "CHG", "CHGCAR", "PROCAR", "DOSCAR"} & left == set()
    assert {"vasprun.xml", "OUTCAR", "CONTCAR", "INCAR", "POSCAR",
            "KPOINTS", "OSZICAR", "vasp.out", "custodian.json"} <= left


def test_it_is_idempotent(tmp_path, clean_up):
    """update_log can revisit a directory; a second pass must not blow up."""
    d = _make_vasp_dir(tmp_path)
    clean_up(str(d))
    before = {f.name for f in d.iterdir()}
    clean_up(str(d))
    assert {f.name for f in d.iterdir()} == before


def test_missing_directory_raises(tmp_path, clean_up):
    """Why the finally wraps the call in its own try/except: this propagates
    out of update_log otherwise."""
    with pytest.raises(FileNotFoundError):
        clean_up(str(tmp_path / "does_not_exist"))


# --- where it is called from ----------------------------------------------

def _cleanup_calls(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "clean_up_vasp_directory"]


def test_cleanup_is_always_called_from_a_finally():
    """THE regression. A call as the last statement of a `try` is skipped for
    every failed job, and a call after the try/except is skipped too because
    both `except` blocks end in `continue`. Only `finally` runs on both paths."""
    if not JOB_HANDLER.exists():          # GNoME_DREAMS_OER_screening is a
        pytest.skip("GNoME package not checked out")   # separate repo
    tree = ast.parse(JOB_HANDLER.read_text())

    in_finally = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                in_finally.update(id(c) for c in _cleanup_calls(stmt))

    calls = _cleanup_calls(tree)
    assert calls, "no clean_up_vasp_directory call found in job_handler.py"
    assert all(id(c) in in_finally for c in calls), (
        f"{sum(id(c) not in in_finally for c in calls)} of {len(calls)} "
        "clean_up_vasp_directory call(s) are outside a finally block"
    )
