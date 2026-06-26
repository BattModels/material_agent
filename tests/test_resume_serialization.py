# Resume safety: a recorded disposition must survive the REAL checkpoint
# serializer round-trip, not just the column-drop / df-reassignment that the
# other resume tests simulate.
#
# Production path (invoke.py): the candidates/processes DataFrames are carried as
# the `explog_candidates` / `explog_processes` LangGraph channels; SqliteSaver
# serializes every channel value with the configured serde and, on resume,
# invoke.py assigns the deserialized frames straight back onto the EXPLOG
# singleton:
#     EXPLOG.relational_frame.candidates.df = snap.values["explog_candidates"]
#     EXPLOG.relational_frame.processes.df  = snap.values["explog_processes"]
# A DataFrame is not a msgpack-native type, so it travels via pickle_fallback --
# which is exactly how the existing `study_obj` object column already round-trips
# in production. This test proves the new `disposition_record` (a list[dict]
# object column) + `decision` + the derived `needs` survive that same trip, and
# that Gate 1 reads correctly afterwards.
#
# Fast tier: imports the nested package + langgraph serde only (NOT src.tools).

from gnome_dreams_oer_screening.explog.explog import EXPLOG
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# ---------------------------------------------------------------------------
# Harness (mirrors tests/test_disposition_tools.py) + the production serde
# ---------------------------------------------------------------------------

def _setup(tmp_path):
    EXPLOG.init(tmp_path, mode="test")
    EXPLOG.job_handler.disposition_decisions = ("Abandon", "Investigating",
                                                "Sufficient")


def _add_candidate(cid):
    EXPLOG.add_candidate(candidate_id=cid, study_obj=object(),
                         reason_or_hypothesis="resume serialization test")


def _inject(cid, job_type, status, termination_index=None, site_index=None):
    table = EXPLOG.relational_frame.processes
    pid = (int(table.df["process_id"].max()) + 1) if len(table.df) else 0
    table.add_row({"process_id": pid, "candidate_id": cid, "job_type": job_type,
                   "status": status, "termination_index": termination_index,
                   "site_index": site_index}, allow_update=False)
    return pid


def _val(cid, col):
    cdf = EXPLOG.relational_frame.candidates.df
    return cdf.loc[cdf["candidate_id"] == cid, col].iloc[0]


def _dispositions(cid):
    return EXPLOG.job_handler._candidate_dispositions(cid)


def _record_disposition(cid, ids, decision="Investigating"):
    EXPLOG.get_disposition_info(cid)
    return EXPLOG.update_disposition_info(cid, "surface looks promising",
                                          list(ids), "run OH on the best site",
                                          decision)


def _production_serde():
    """The exact serde invoke.py hands to SqliteSaver (see invoke.py:626-636).
    The allowed_msgpack_modules are string module paths only -- constructing the
    serde does NOT import src.tools, so this stays in the fast tier."""
    try:
        return JsonPlusSerializer(
            pickle_fallback=True,
            allowed_msgpack_modules=[("src.myCANVAS",), ("src.planNexe2",),
                                     ("src.tools",)],
        )
    except TypeError:  # older langgraph-checkpoint without the kwarg
        return JsonPlusSerializer(pickle_fallback=True)


def _resume_via_serde(serde):
    """Round-trip both EXPLOG frames through the serde and reassign them onto the
    singleton -- faithfully mirroring the invoke.py resume restore, but exercising
    the real serialize -> deserialize instead of an in-memory df handoff."""
    rf = EXPLOG.relational_frame
    rf.candidates.df = serde.loads_typed(serde.dumps_typed(rf.candidates.df))
    rf.processes.df = serde.loads_typed(serde.dumps_typed(rf.processes.df))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recorded_disposition_survives_checkpoint_serde(tmp_path):
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    res = _record_disposition("c", [pid_s], decision="Investigating")
    assert res["status"] == "ok"

    _resume_via_serde(_production_serde())          # <-- the real round-trip

    # the disposition_record object column (list[dict]) came back intact:
    recs = _dispositions("c")
    assert len(recs) == 1
    assert recs[0]["Decision"] == "Investigating"
    assert recs[0]["Summarized_process_id"] == [pid_s]
    assert recs[0]["Summary"] == "surface looks promising"
    # the visible `decision` column survived too:
    assert _val("c", "decision") == "Investigating"
    # and Gate 1 reads it correctly post-resume: nothing left to disposition.
    assert EXPLOG.candidates_needing_disposition() == []


def test_resume_reblocks_gate1_on_work_finished_after_checkpoint(tmp_path):
    # Continuity: after a resume the prior disposition is restored AND the gate
    # still demands a disposition for work that finalized after the checkpoint.
    _setup(tmp_path)
    _add_candidate("c")
    pid_s = _inject("c", "surface_relaxation", "completed", termination_index=0)
    _record_disposition("c", [pid_s], decision="Investigating")

    _resume_via_serde(_production_serde())

    # an O job finalizes "after" the resume -> a new uncited unit appears.
    pid_o = _inject("c", "O_adsorption", "completed",
                    termination_index=0, site_index=1)
    assert EXPLOG.candidates_needing_disposition() == ["c"]      # Gate 1 re-blocks

    # get surfaces BOTH the restored latest disposition and the new must_cover id.
    out = EXPLOG.get_disposition_info("c")
    assert out["latest_disposition"]["Decision"] == "Investigating"
    assert any(pid_o in set(u["ids"]) for u in out["must_cover"])
