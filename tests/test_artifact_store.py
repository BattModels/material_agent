"""Tests for the durable artifact registry (src/artifact_store.py + myCANVAS wiring).

Fast suite: imports only src.myCANVAS / src.artifact_store, never src.tools, so
it does not pay the ~109 s GNoME database import.

What matters here is that the registry survives a process restart from the
SQLite table ALONE -- once `artifacts` is dropped from the LangGraph graph state
this table is the only copy of the provenance record that `verify_artifact` and
every report citation depend on.
"""

import pytest

from src.artifact_store import ArtifactStore
from src.myCANVAS import myCANVAS


@pytest.fixture
def canvas(tmp_path):
    c = myCANVAS()
    c.set_working_directory(str(tmp_path))
    return c


def _numeric(c, value=42.0, tool="submit_dft_job"):
    return c.register_tool_output(
        tool_name=tool, args={"a": 1}, value=value,
        description="d", reasons={"r": "why"},
    )


def test_registering_persists_to_sqlite(canvas):
    rid = _numeric(canvas)
    assert canvas.result_registry[rid].value == 42.0      # in memory
    assert canvas._artifact_store.count() == 1            # and on disk
    assert canvas._artifact_store.get(rid).value == 42.0


def test_registry_rehydrates_from_disk_alone(tmp_path):
    """The restart path: a fresh CANVAS must recover the full registry."""
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    rid = _numeric(c1, 3.5)
    other = c1.register_tool_output(
        tool_name="query_explog", args={}, value="a rendered table",
        description="d", reasons={},
    )

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    assert set(c2.result_registry) == {rid, other}
    assert c2.get_artifact(rid).value == 3.5
    assert c2.get_artifact(other).tool_name == "query_explog"


def test_verify_artifact_works_after_reload(tmp_path):
    """verify_artifact is the anti-hallucination gate -- it must still accept a
    genuine citation, and still reject a wrong one, after a restart."""
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    rid = _numeric(c1, 7.25)

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    assert c2.verify_artifact(7.25, rid)[0] is True
    assert c2.verify_artifact(9.99, rid)[0] is False
    assert c2.verify_artifact(7.25, "nosuchid")[0] is False


def test_listed_artifact_nested_values_survive(tmp_path):
    """submit_dft_job returns a ListedArtifact whose NumericArtifacts are nested
    one level down; verify_artifact matches against any of them."""
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    rid = c1.register_tool_output(
        tool_name="submit_dft_job", args={}, value=[11.0, 12.0],
        description="d", reasons={}, listed_value=True,
    )

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    assert c2.verify_artifact(12.0, rid)[0] is True
    assert c2.verify_artifact(13.0, rid)[0] is False


def test_open_is_idempotent(tmp_path):
    """invoke.py calls set_working_directory twice; the second call must not
    re-open the store or re-load ~100 MB of artifacts."""
    c = myCANVAS(); c.set_working_directory(str(tmp_path))
    _numeric(c)
    store = c._artifact_store
    c.set_working_directory(str(tmp_path))
    assert c._artifact_store is store
    assert len(c.result_registry) == 1


def test_curr_round_ids_track_registrations(canvas):
    """check_required_tool_use reads curr_round_result_ids; the single write
    path must keep appending to it."""
    a = _numeric(canvas, tool="submit_dft_job")
    b = _numeric(canvas, tool="query_explog")
    assert canvas.curr_round_result_ids == [a, b]
    canvas.rest_curr_round_result_ids()
    assert canvas.curr_round_result_ids == []


def test_store_survives_reopen_without_canvas(tmp_path):
    """ArtifactStore on its own round-trips -- the audit notebooks read it
    directly without going through CANVAS."""
    c = myCANVAS(); c.set_working_directory(str(tmp_path))
    _numeric(c, 1.5)
    s = ArtifactStore(str(tmp_path / "artifacts.sqlite"))
    assert s.count() == 1
    (rid, art), = s.load_all().items()
    assert art.value == 1.5
    assert art.tool_name == "submit_dft_job"


def test_registration_works_without_a_store():
    """The module-level CANVAS singleton exists before any run directory is
    known (_artifact_store is None until set_working_directory). Registering
    then must still work, memory-only -- otherwise importing CANVAS in a test
    or a helper script would blow up."""
    c = myCANVAS()
    assert c._artifact_store is None
    rid = _numeric(c, 5.0)
    assert c.result_registry[rid].value == 5.0
    assert c.verify_artifact(5.0, rid)[0] is True


def test_new_artifacts_append_to_an_existing_table(tmp_path):
    """Reopening must not truncate: a resumed run keeps adding to the log it
    just rehydrated."""
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    first = _numeric(c1, 1.0)

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    second = _numeric(c2, 2.0)

    assert c2._artifact_store.count() == 2
    c3 = myCANVAS(); c3.set_working_directory(str(tmp_path))
    assert set(c3.result_registry) == {first, second}


def test_reregistering_same_id_is_idempotent(tmp_path):
    """Crash-replay: a resumed run may re-run the tool call that was in flight.
    Re-putting the same result_id must overwrite, not raise or duplicate."""
    c = myCANVAS(); c.set_working_directory(str(tmp_path))
    rid = _numeric(c, 4.0)
    art = c.result_registry[rid]
    c._artifact_store.put(rid, art)
    c._artifact_store.put(rid, art)
    assert c._artifact_store.count() == 1


def test_put_many_matches_put(tmp_path):
    """The migration path (migrate_artifacts.py) uses put_many; it must produce
    rows indistinguishable from the live put() path."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    src = myCANVAS(); src.set_working_directory(str(tmp_path / "a"))
    ids = [_numeric(src, float(i)) for i in range(5)]

    dest = ArtifactStore(str(tmp_path / "b" / "artifacts.sqlite"))
    assert dest.put_many(src.result_registry.items()) == 5
    assert dest.count() == 5
    reloaded = dest.load_all()
    for rid in ids:
        assert reloaded[rid].__dict__ == src.result_registry[rid].__dict__


def test_get_missing_id_returns_none(tmp_path):
    s = ArtifactStore(str(tmp_path / "artifacts.sqlite"))
    assert s.get("nosuchid") is None
    assert s.count() == 0
    assert s.load_all() == {}


def test_missing_directory_fails_clearly(tmp_path):
    """A mis-set WORKING_DIR should say so, not surface sqlite's opaque
    'unable to open database file'."""
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        ArtifactStore(str(tmp_path / "no" / "such" / "dir" / "artifacts.sqlite"))


def test_state_snapshot_carries_no_bulk_state():
    """REGRESSION GUARD. canvas / artifacts / explog_* must never be put back
    into the per-tool-call state sync: each is persisted by its own owner, and
    carrying them here cost ~376 MB per tool call (measured on the 27-05 run)
    to record a ~12 KB delta.

    NOTE on PlanExecute: this used to also assert the four names were absent
    from the PARENT schema. They are back, deliberately, as VESTIGIAL channels
    -- deleting them from the schema does not delete them from an existing
    checkpoint, and LangGraph copies unrecognised channel_values forward
    forever, so they have to stay declared long enough for `invoke.py continue`
    to write empties into them (see tests/test_legacy_channel_migration.py).
    Declaring them is harmless; POPULATING them is the thing that costs 140 MB
    a checkpoint, and that is what the two assertions below actually prevent:
    nothing reaches those channels except the one-shot clearing write.

    Imported lazily -- src.planNexe2 pulls in src.tools and the GNoME database.
    """
    pytest.importorskip("gnome_dreams_oer_screening")
    from src.planNexe2 import full_state_snapshot, SyncedAgentState
    from src import var

    # var.startTime is "" until invoke.py sets it; the snapshot subtracts it.
    var.startTime = 0.0

    banned = {"canvas", "artifacts", "explog_candidates", "explog_processes"}
    assert banned.isdisjoint(SyncedAgentState.__annotations__)
    assert set(full_state_snapshot()) == {"curr_round_result_ids", "time"}


# --- canvas.pickle: atomic write + startup restore --------------------------

def test_canvas_persists_and_restores(tmp_path):
    """The notes board must survive a restart from disk alone -- it is no
    longer carried in the checkpoint."""
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    c1.write("round1_report", "findings ...")
    c1.write("hypothesis", {"H1": "Ir-Rh 1:2 optimal"})

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    assert c2.read("round1_report") == "findings ..."
    assert c2.read("hypothesis") == {"H1": "Ir-Rh 1:2 optimal"}


def test_canvas_overwrite_is_persisted(tmp_path):
    c1 = myCANVAS(); c1.set_working_directory(str(tmp_path))
    c1.write("k", "v1")
    c1.write("k", "v2", overwrite=True)

    c2 = myCANVAS(); c2.set_working_directory(str(tmp_path))
    assert c2.read("k") == "v2"


def test_canvas_write_is_atomic(tmp_path):
    """No .tmp left behind, and the destination is only ever replaced whole."""
    c = myCANVAS(); c.set_working_directory(str(tmp_path))
    c.write("k", "v")
    assert (tmp_path / "canvas.pickle").exists()
    assert not (tmp_path / "canvas.pickle.tmp").exists()


def test_wiped_run_dir_does_not_resurrect_stale_state(tmp_path):
    """invoke.py's 'ow' path rm -rf's the run dir BETWEEN its two
    set_working_directory calls. Neither the canvas nor the artifact registry
    may survive that."""
    import shutil
    run = tmp_path / "run"; run.mkdir()
    c = myCANVAS(); c.set_working_directory(str(run))
    c.write("old_report", "from the previous run")
    _numeric(c, 1.0)
    assert len(c.result_registry) == 1 and c.read("old_report") is not None

    shutil.rmtree(str(run)); run.mkdir()          # what 'ow' does
    c.set_working_directory(str(run))             # second call

    assert c.canvas == {}, "stale canvas survived a wiped run directory"
    assert c.result_registry == {}, "stale artifacts survived a wiped run directory"
    # and the reopened store must be usable, not pointing at the unlinked file
    rid = _numeric(c, 2.0)
    assert c._artifact_store.count() == 1
    assert c._artifact_store.get(rid).value == 2.0
