# Tests for the EXPLOG rehydration that commit 3bbf619 removed.
#
# THE BUG THIS EXISTS FOR. EXPLOG persists itself via _save_pickle() but has no
# loader -- init() always builds empty frames. invoke.py used to restore the
# frames from the checkpoint; 3bbf619 deleted those two lines believing init()
# read the pickle. The first resume came up with 0 candidates against 139
# candidates / 1107 processes of real DFT work, and the next _save_pickle()
# overwrote the 43 MB pickle with a 5.7 KB empty one.
#
# What made it dangerous was SILENCE: an empty EXPLOG is a perfectly valid
# object. Nothing downstream complained -- it simply looked like a study that
# had not begun. So these tests care as much about the loud-failure paths as
# about the happy one.

import pickle

import pandas as pd
import pytest

from src.explog_restore import (
    BACKUP_NAME,
    PICKLE_NAME,
    ExplogRestoreError,
    assert_restored,
    load_explog_payload,
    restore_explog,
)


class _Table:
    def __init__(self):
        self.df = pd.DataFrame()


class _RF:
    def __init__(self):
        self.candidates = _Table()
        self.processes = _Table()


class _Explog:
    """Stand-in for EXPLOG: only the two attributes the restore touches."""
    def __init__(self):
        self.relational_frame = _RF()


def _write_pickle(d, candidates=2, processes=5):
    payload = {
        "candidates_df": pd.DataFrame({"candidate_id": [f"c{i}" for i in range(candidates)]}),
        "processes_df": pd.DataFrame({"process_id": list(range(processes))}),
    }
    with open(d / PICKLE_NAME, "wb") as f:
        pickle.dump(payload, f)
    return payload


# --- the happy path ---------------------------------------------------------

def test_restores_both_frames_and_reports_counts(tmp_path):
    _write_pickle(tmp_path, candidates=139, processes=1107)
    ex = _Explog()
    assert len(ex.relational_frame.candidates.df) == 0

    n_c, n_p = restore_explog(ex, tmp_path)

    assert (n_c, n_p) == (139, 1107)
    assert len(ex.relational_frame.candidates.df) == 139
    assert len(ex.relational_frame.processes.df) == 1107


def test_a_backup_is_taken_before_the_process_can_overwrite_the_source(tmp_path):
    """The pickle is overwritten by the first _save_pickle after startup, so the
    only safe moment to copy it is during the restore itself."""
    _write_pickle(tmp_path)
    restore_explog(_Explog(), tmp_path)
    backup = tmp_path / BACKUP_NAME
    assert backup.exists()
    assert backup.read_bytes() == (tmp_path / PICKLE_NAME).read_bytes()


def test_no_backup_for_an_empty_log(tmp_path):
    """Backing up an empty log would destroy a good earlier backup."""
    _write_pickle(tmp_path, candidates=0, processes=0)
    restore_explog(_Explog(), tmp_path)
    assert not (tmp_path / BACKUP_NAME).exists()


def test_a_missing_pickle_is_not_an_error(tmp_path):
    """Correct for a genuinely fresh run; the caller decides what it means."""
    assert restore_explog(_Explog(), tmp_path) == (0, 0)


# --- the loud-failure paths -------------------------------------------------

def test_assert_raises_when_a_pickle_exists_but_nothing_loaded(tmp_path):
    """THE regression guard. This is the exact shape of the 02-08 failure:
    explog.pkl on disk holding 139 candidates, EXPLOG empty, run continuing."""
    _write_pickle(tmp_path, candidates=139, processes=1107)
    with pytest.raises(ExplogRestoreError, match="EMPTY after restore"):
        assert_restored(0, 0, tmp_path)


def test_assert_is_silent_when_work_was_restored(tmp_path):
    _write_pickle(tmp_path)
    assert_restored(139, 1107, tmp_path) is None


def test_assert_is_silent_with_no_pickle_at_all(tmp_path):
    """A run that has genuinely never registered anything must still start."""
    assert_restored(0, 0, tmp_path) is None


@pytest.mark.parametrize("payload", [
    ["not", "a", "dict"],
    {"candidates_df": pd.DataFrame()},                       # processes_df missing
    {"candidates_df": None, "processes_df": pd.DataFrame()},  # wrong type
])
def test_malformed_payloads_raise_instead_of_looking_empty(tmp_path, payload):
    """A malformed pickle must never be indistinguishable from 'no work yet' --
    that ambiguity is what let the original bug run for a full startup."""
    with open(tmp_path / PICKLE_NAME, "wb") as f:
        pickle.dump(payload, f)
    with pytest.raises(ExplogRestoreError):
        load_explog_payload(tmp_path)


def test_unreadable_pickle_raises_explog_restore_error(tmp_path):
    (tmp_path / PICKLE_NAME).write_bytes(b"not a pickle at all")
    with pytest.raises(ExplogRestoreError, match="could not be unpickled"):
        load_explog_payload(tmp_path)


def test_restore_propagates_malformed_pickles(tmp_path):
    """restore_explog swallows only FileNotFoundError -- never a corrupt file."""
    with open(tmp_path / PICKLE_NAME, "wb") as f:
        pickle.dump(["wrong shape"], f)
    with pytest.raises(ExplogRestoreError):
        restore_explog(_Explog(), tmp_path)


def test_matches_the_shape_explog_actually_writes():
    """Pins the contract against EXPLOG._save_pickle. If that ever changes its
    payload keys, this fails here rather than silently restoring nothing."""
    pytest.importorskip("gnome_dreams_oer_screening")
    import inspect
    from gnome_dreams_oer_screening.explog.explog import EXPLOG
    src = inspect.getsource(EXPLOG._save_pickle)
    assert '"candidates_df"' in src
    assert '"processes_df"' in src
    assert 'explog.pkl' in src
