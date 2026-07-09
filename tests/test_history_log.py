# Tests for src.history_log's his.txt -> hist/his_<N>.txt rotation.
#
# Pure-stdlib module (os, re, pathlib + src.var only) -- no heavy imports,
# so this runs in the fast tier alongside test_wait_gate.py etc.

import os

from src import var
from src.history_log import hist_dir, list_hist_files, write_history


def _reset(monkeypatch, tmp_path, rotate_bytes=1024**3, save_dialogue=True):
    monkeypatch.setattr(var, "my_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(var, "my_SAVE_DIALOGUE", save_dialogue)
    monkeypatch.setattr(var, "HIST_ROTATE_BYTES", rotate_bytes)
    monkeypatch.setattr(var, "hist_active_index", None)
    monkeypatch.setattr(var, "hist_active_bytes", None)


def test_fresh_dir_creates_his_1(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    write_history("hello\n")
    files = list_hist_files(hist_dir())
    assert [f.name for f in files] == ["his_1.txt"]
    assert (hist_dir() / "his_1.txt").read_text() == "hello\n"


def test_rotation_at_threshold(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path, rotate_bytes=50)
    write_history("a" * 30)
    write_history("b" * 30)  # pushes his_1.txt to 60 >= 50 -> rotates
    write_history("c" * 10)  # lands on his_2.txt
    files = list_hist_files(hist_dir())
    assert [f.name for f in files] == ["his_1.txt", "his_2.txt"]
    assert os.path.getsize(files[0]) == 60
    assert os.path.getsize(files[1]) == 10


def test_resume_recomputes_from_disk(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path, rotate_bytes=50)
    write_history("a" * 30)
    write_history("b" * 30)  # rotates to his_2.txt
    write_history("c" * 10)  # his_2.txt now has 10 bytes on disk

    # Simulate a fresh process (killed + relaunched via nohup): the
    # in-memory cache is gone, must be recomputed from disk, not assumed.
    monkeypatch.setattr(var, "hist_active_index", None)
    monkeypatch.setattr(var, "hist_active_bytes", None)

    write_history("d" * 5)
    assert var.hist_active_index == 2
    assert var.hist_active_bytes == 15
    assert os.path.getsize(hist_dir() / "his_2.txt") == 15


def test_numeric_sort_order(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path)
    d = hist_dir()
    d.mkdir(parents=True)
    for n in (1, 2, 10, 11, 9):
        (d / f"his_{n}.txt").write_text("x")
    files = list_hist_files(d)
    assert [f.name for f in files] == [
        "his_1.txt", "his_2.txt", "his_9.txt", "his_10.txt", "his_11.txt",
    ]


def test_save_dialogue_false_is_noop(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path, save_dialogue=False)
    write_history("should not be written")
    assert not hist_dir().exists()


def test_migration_rollover(tmp_path, monkeypatch):
    _reset(monkeypatch, tmp_path, rotate_bytes=50)
    d = hist_dir()
    d.mkdir(parents=True)
    (d / "his_1.txt").write_text("x" * 100)  # simulates a migrated, already-oversized flat his.txt

    write_history("new content")
    assert var.hist_active_index == 2
    files = list_hist_files(d)
    assert [f.name for f in files] == ["his_1.txt", "his_2.txt"]
    assert os.path.getsize(files[0]) == 100  # untouched, never appended to
    assert (d / "his_2.txt").read_text() == "new content"
