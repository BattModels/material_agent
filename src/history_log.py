"""Centralized, rotating writer for the dialogue history log.

his.txt used to be a single ever-growing flat file. It is now a *directory*
of size-capped files, WORKING_DIR/hist/his_1.txt, his_2.txt, ... . Rotation
happens once the active file's size reaches var.HIST_ROTATE_BYTES.

Resume-safety: which file is "active" and how many bytes it holds is
recomputed from disk on the first write_history() call in a fresh process --
never assumed to be his_1.txt / 0 bytes, since a run is killed and relaunched
via nohup repeatedly over a 30-day campaign.
"""
import os
import re
from pathlib import Path

from src import var

_HIST_FILE_RE = re.compile(r"^his_(\d+)\.txt$")


def hist_dir() -> Path:
    return Path(var.my_WORKING_DIRECTORY) / "hist"


def list_hist_files(directory) -> list[Path]:
    """his_<N>.txt files in `directory`, sorted by their NUMERIC index (not
    lexicographic -- his_2.txt must sort before his_10.txt). Shared by the
    writer and by workflow_audit's readers so the sort order is defined once."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = [
        (int(m.group(1)), p)
        for p in directory.iterdir()
        if (m := _HIST_FILE_RE.match(p.name))
    ]
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _rotate_if_full() -> None:
    if var.hist_active_bytes >= var.HIST_ROTATE_BYTES:
        var.hist_active_index += 1
        var.hist_active_bytes = 0


def _init_state() -> None:
    """Once per process: figure out the real active file + its real size."""
    d = hist_dir()
    d.mkdir(parents=True, exist_ok=True)
    files = list_hist_files(d)
    if files:
        var.hist_active_index = int(_HIST_FILE_RE.match(files[-1].name).group(1))
        var.hist_active_bytes = os.path.getsize(files[-1])
    else:
        var.hist_active_index, var.hist_active_bytes = 1, 0
    # Handles both: (a) a resume where the last file already crossed the
    # threshold mid-session, and (b) a migrated flat his.txt renamed to
    # his_1.txt that's already oversized -- either way, don't append to an
    # already-full file, roll to a fresh one immediately.
    _rotate_if_full()


def _active_path() -> Path:
    return hist_dir() / f"his_{var.hist_active_index}.txt"


def write_history(text: str) -> None:
    """Append `text` to the active hist/his_<N>.txt, rotating once it
    reaches var.HIST_ROTATE_BYTES. No-op if var.my_SAVE_DIALOGUE is False."""
    if not var.my_SAVE_DIALOGUE:
        return
    if var.hist_active_index is None:
        _init_state()

    with open(_active_path(), "a", encoding="utf-8") as f:
        f.write(text)
    var.hist_active_bytes += len(text.encode("utf-8", "replace"))
    _rotate_if_full()
