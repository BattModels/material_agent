"""Rehydrate EXPLOG from ``vasp_calcs/explog.pkl`` on resume.

WHY THIS EXISTS. EXPLOG persists itself: every mutation calls ``_save_pickle()``,
which atomically writes ``{"candidates_df": ..., "processes_df": ...}``. But the
class has NO counterpart loader -- ``init()`` always builds empty frames. The
pickle was write-only, a backup nothing read.

Until commit 3bbf619 that was fine, because invoke.py restored the frames from
the LangGraph checkpoint instead::

    EXPLOG.relational_frame.candidates.df = snap.values["explog_candidates"]
    EXPLOG.relational_frame.processes.df  = snap.values["explog_processes"]

3bbf619 removed those two lines (they cost ~42 MB per checkpoint) on the premise
that ``init`` hydrates from the pickle. It does not. The first resume therefore
came up with 0 candidates and 0 processes against 139 candidates / 1107
processes of real DFT work; the worker reported "CRITICAL DATA LOSS", the
supervisor planned to re-register everything from CANVAS, and the next
``_save_pickle()`` overwrote the 43 MB pickle with a 5.7 KB empty one.

So this module is the missing half of ``_save_pickle``. It reads what that
writes, and nothing else.

The failure mode to design against is SILENCE: an empty EXPLOG is a completely
valid object, so nothing downstream objects to it -- it just looks like a study
that has not started. Hence ``restore_explog`` reports what it did and
``assert_restored`` turns "I expected work and found none" into a hard stop.
"""

import pickle
import shutil
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

PICKLE_NAME = "explog.pkl"
# One-generation undo, refreshed on every successful non-empty restore. Cheap
# insurance (~43 MB) against exactly what happened here: a process that starts
# with an empty log and then persists it over the real one.
BACKUP_NAME = "explog.pkl.startup_bak"


class ExplogRestoreError(RuntimeError):
    """The pickle exists but could not be turned into usable frames."""


def load_explog_payload(vasp_dir) -> dict:
    """Read and validate the pickle written by ``EXPLOG._save_pickle``.

    Raises ExplogRestoreError on anything unexpected rather than returning
    something empty-but-plausible -- a malformed payload must not be
    indistinguishable from a study that has done no work.
    """
    path = Path(vasp_dir) / PICKLE_NAME
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ExplogRestoreError(f"{path} could not be unpickled: {exc!r}") from exc

    if not isinstance(payload, dict):
        raise ExplogRestoreError(
            f"{path} holds {type(payload).__name__}, expected a dict with "
            "'candidates_df' and 'processes_df' (see EXPLOG._save_pickle)."
        )
    missing = {"candidates_df", "processes_df"} - set(payload)
    if missing:
        raise ExplogRestoreError(
            f"{path} is missing {sorted(missing)}; keys present: {sorted(payload)}."
        )
    for key in ("candidates_df", "processes_df"):
        if not isinstance(payload[key], pd.DataFrame):
            raise ExplogRestoreError(
                f"{path}['{key}'] is {type(payload[key]).__name__}, expected DataFrame."
            )
    return payload


def restore_explog(explog: Any, vasp_dir) -> Tuple[int, int]:
    """Assign the persisted frames onto ``explog``. Returns (n_candidates, n_processes).

    Returns (0, 0) when the pickle is absent -- correct for a genuinely fresh
    run. The caller decides whether absence is acceptable (see assert_restored);
    this function does not guess.
    """
    vasp_dir = Path(vasp_dir)
    try:
        payload = load_explog_payload(vasp_dir)
    except FileNotFoundError:
        return (0, 0)

    candidates = payload["candidates_df"]
    processes = payload["processes_df"]

    # Back up BEFORE handing control back: from this point the process may
    # mutate EXPLOG and call _save_pickle, which overwrites the source.
    if len(candidates) or len(processes):
        try:
            shutil.copy2(vasp_dir / PICKLE_NAME, vasp_dir / BACKUP_NAME)
        except OSError:
            pass  # a missing backup must never block a resume

    explog.relational_frame.candidates.df = candidates
    explog.relational_frame.processes.df = processes
    return (len(candidates), len(processes))


def assert_restored(n_candidates: int, n_processes: int, vasp_dir) -> None:
    """Hard-stop when a pickle promising real work produced an empty EXPLOG.

    The bug this guards was silent for a full startup: the log printed an empty
    frame and the run continued into an agent that had to discover the loss for
    itself. Anything that leaves EXPLOG empty while explog.pkl says otherwise
    must kill the process, not warn.
    """
    path = Path(vasp_dir) / PICKLE_NAME
    if not path.exists():
        return
    if n_candidates or n_processes:
        return
    size = path.stat().st_size
    raise ExplogRestoreError(
        f"EXPLOG is EMPTY after restore, but {path} exists ({size} bytes).\n"
        "Refusing to continue: an empty EXPLOG looks like a study that has not "
        "started, so the agents would be told to re-register everything from "
        "scratch and the next _save_pickle would overwrite this file.\n"
        f"A pre-restore copy may be available at {path.parent / BACKUP_NAME}."
    )
