"""One-off backfill: lift the CANVAS artifact registry out of a run's LangGraph
checkpoint and into that run's `artifacts.sqlite`. Not part of the app.

Run this BEFORE the code stops carrying `artifacts` in graph state. Until the
table is populated and verified, the checkpoint is still the only copy of the
provenance/citation record that `verify_artifact` and every report citation
(e.g. "EXPLOG Query References: gosq33kz, ...") depend on.

Read-only with respect to checkpoints.sqlite. Idempotent (INSERT OR REPLACE),
so it is safe to re-run.

Usage:
    ml Python/3.11.3-GCCcore-12.3.0
    source venv2/bin/activate
    python migrate_artifacts.py [RUN_DIR]
"""

import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.artifact_store import ArtifactStore

DEFAULT_RUN_DIR = (
    "/home/energy/matnis/projects/dreams_colab/v2/material_agent/"
    "production_run_27-05-2026_fork_02_08_2026"
)


def newest_artifacts_blob(ckpt_path):
    """The most recent `artifacts` channel write, across ALL namespaces.

    The registry only ever grows, and checkpoint_ids are UUIDv6 (lexicographically
    time-ordered), so the newest write is the complete registry. Inner
    (tool-level) namespaces can be newer than the parent round, hence no
    checkpoint_ns filter.
    """
    conn = sqlite3.connect(f"file:{ckpt_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT checkpoint_ns, checkpoint_id, type, value FROM writes "
            "WHERE channel = 'artifacts' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"No 'artifacts' writes found in {ckpt_path}")
    ns, ckpt_id, type_, value = row
    print(f"  source: checkpoint_id={ckpt_id}  ns={ns or '<parent>'}  "
          f"({len(value) / 1024**2:.1f} MB)")
    return JsonPlusSerializer(pickle_fallback=True).loads_typed((type_, value))


def main(run_dir):
    ckpt = os.path.join(run_dir, "checkpoints.sqlite")
    dest = os.path.join(run_dir, "artifacts.sqlite")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"No checkpoints.sqlite in {run_dir}")

    print(f"run dir : {run_dir}")
    print("reading artifacts out of the checkpoint ...")
    registry = newest_artifacts_blob(ckpt)
    print(f"  loaded {len(registry)} artifacts from the checkpoint")

    print(f"writing  {dest} ...")
    store = ArtifactStore(dest)
    n = store.put_many(registry.items())
    print(f"  inserted {n} rows")

    # ---- verification gates -------------------------------------------------
    print("\nVERIFYING")
    ok = True

    count = store.count()
    print(f"  [{'ok ' if count == len(registry) else 'FAIL'}] row count "
          f"{count} == {len(registry)} source artifacts")
    ok &= count == len(registry)

    # Round-trip a random sample: the rehydrated object must match field-for-field.
    reloaded = store.load_all()
    sample = random.sample(sorted(registry), min(50, len(registry)))
    bad = []
    for rid in sample:
        a, b = registry[rid], reloaded.get(rid)
        if b is None or getattr(a, "__dict__", a) != getattr(b, "__dict__", b):
            bad.append(rid)
    print(f"  [{'ok ' if not bad else 'FAIL'}] round-trip {len(sample)} sampled "
          f"artifacts field-identical" + (f" -- BAD: {bad[:5]}" if bad else ""))
    ok &= not bad

    # The whole point of the registry: verify_artifact -- the gate that stops
    # the agent asserting numbers it never computed -- must still accept real
    # citations when the registry is rehydrated from sqlite ALONE.
    # NOTE: NumericArtifacts are usually nested inside a ListedArtifact.value,
    # so a top-level isinstance check finds nothing; walk one level down.
    from src.myCANVAS import CANVAS

    def numeric_value(a):
        v = getattr(a, "value", None)
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, list):
            for sub in v:
                sv = getattr(sub, "value", None)
                if isinstance(sv, (int, float)):
                    return sv
        return None

    saved_registry = CANVAS.result_registry
    CANVAS.result_registry = reloaded          # sqlite is the ONLY source here
    try:
        citable = [(rid, numeric_value(a)) for rid, a in reloaded.items()]
        citable = [(r, v) for r, v in citable if v is not None]
        print(f"  ---- verify_artifact against a sqlite-only registry "
              f"({len(citable)} citable numeric artifacts) ----")
        for rid, val in citable[:5]:
            res = CANVAS.verify_artifact(val, rid)
            good = res[0]
            print(f"  [{'ok ' if good else 'FAIL'}] verify({val!r}, {rid}) accepted")
            ok &= good
        if citable:                            # negative control
            rid, val = citable[0]
            rejected = not CANVAS.verify_artifact(val + 12345.0, rid)[0]
            print(f"  [{'ok ' if rejected else 'FAIL'}] wrong value rejected")
            ok &= rejected
        unknown = not CANVAS.verify_artifact(1.0, "zzzznope")[0]
        print(f"  [{'ok ' if unknown else 'FAIL'}] unknown result_id rejected")
        ok &= unknown
    finally:
        CANVAS.result_registry = saved_registry

    tools = sqlite3.connect(dest).execute(
        "SELECT tool_name, count(*) FROM artifacts GROUP BY tool_name "
        "ORDER BY count(*) DESC LIMIT 5"
    ).fetchall()
    print("\n  top tools in the new table:")
    for t, c in tools:
        print(f"    {c:6d}  {t}")
    print(f"\n  {store!r}")

    store.close()
    print("\n" + ("MIGRATION OK" if ok else "MIGRATION FAILED -- do not proceed"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR))
