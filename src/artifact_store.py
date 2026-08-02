"""Durable, append-only store for CANVAS tool-output artifacts.

WHY THIS EXISTS
---------------
``CANVAS.result_registry`` is an append-only, immutable log: every tool call
adds exactly one ~12 KB artifact and nothing is ever mutated afterwards. It
used to be persisted by being carried in the LangGraph graph state, which meant
appending one artifact re-serialized the WHOLE registry (95.7 MB measured at
7,969 artifacts) into every checkpoint -- and, because the inner agents are
checkpointed subgraphs, that happened after EVERY tool call.

Storing it here instead makes the cost proportional to what actually changed:
one INSERT per tool call. The in-memory ``result_registry`` dict is unchanged --
every reader (``get_artifact``, ``verify_artifact``, ``search_artifacts``,
``check_required_tool_use``, the DAG builder) still reads that dict. This class
only changes how the dict is *persisted* and *rehydrated*.

It also makes the provenance log queryable, which is what
``src/workflow_audit/`` currently reconstructs by regex-parsing the multi-GB
``hist/his_<N>.txt`` dialogue logs:

    select tool_name, count(*) from artifacts group by tool_name;

DURABILITY
----------
This is an AUTHORITATIVE copy: once the graph state stops carrying artifacts,
this file is the only place they live. So writes use WAL + synchronous=FULL --
each artifact is fsync'd before the tool returns. The rows are ~12 KB, so the
fsync cost is negligible next to the DFT jobs this run schedules.

WAL is proven on this NFS mount (the LangGraph SqliteSaver already runs WAL
against the same filesystem) and lets the audit notebooks read the table while
a run is writing to it.
"""

import os
import pickle
import sqlite3
import threading


_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    result_id  TEXT PRIMARY KEY,
    tool_name  TEXT,
    created_at REAL,
    blob       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_tool_name ON artifacts(tool_name);
"""


class ArtifactStore:
    """SQLite-backed persistence for the CANVAS artifact registry.

    Append-only by construction: ``put`` is the only writer and artifacts are
    immutable, so there is no update or compaction path. ``INSERT OR REPLACE``
    is used purely for idempotency -- re-running a tool call after a crash
    rewrites the identical row instead of raising.
    """

    def __init__(self, path):
        self.path = str(path)
        # Fail with something actionable instead of sqlite3's "unable to open
        # database file". Deliberately does NOT mkdir: a missing parent means a
        # mis-set WORKING_DIR, and silently creating it would hide that.
        parent = os.path.dirname(os.path.abspath(self.path))
        if not os.path.isdir(parent):
            raise FileNotFoundError(
                f"Cannot open artifact store -- directory does not exist: {parent}"
            )
        # check_same_thread=False: tool calls can be dispatched from a worker
        # thread while the LangGraph background executor holds the main one.
        # All access goes through self._lock, so this stays single-writer.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Authoritative store -> fsync every commit. Rows are ~12 KB.
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- write ---------------------------------------------------------------

    def put(self, result_id, artifact):
        """Persist one artifact. Called once per tool call."""
        blob = pickle.dumps(artifact)
        tool_name = getattr(artifact, "tool_name", None)
        created_at = getattr(artifact, "timestamp", None)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts "
                "(result_id, tool_name, created_at, blob) VALUES (?, ?, ?, ?)",
                (str(result_id), tool_name, created_at, blob),
            )
            self._conn.commit()

    def put_many(self, items):
        """Bulk insert (result_id, artifact) pairs in ONE transaction.

        Only for migration/backfill -- the live path uses ``put`` so each
        artifact is durable before its tool call returns.
        """
        rows = [
            (
                str(rid),
                getattr(a, "tool_name", None),
                getattr(a, "timestamp", None),
                pickle.dumps(a),
            )
            for rid, a in items
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO artifacts "
                "(result_id, tool_name, created_at, blob) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    # -- read ----------------------------------------------------------------

    def load_all(self):
        """Rebuild the full ``{result_id: artifact}`` dict.

        Called once at startup. Unpickling requires the artifact classes to be
        importable (``src.myCANVAS``), which is true wherever CANVAS is.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT result_id, blob FROM artifacts"
            ).fetchall()
        return {rid: pickle.loads(blob) for rid, blob in rows}

    def count(self):
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM artifacts").fetchone()[0]

    def get(self, result_id):
        """Single-artifact lookup. Not used by the live path (the in-memory
        dict serves those), but handy for audit tooling."""
        with self._lock:
            row = self._conn.execute(
                "SELECT blob FROM artifacts WHERE result_id = ?", (str(result_id),)
            ).fetchone()
        return pickle.loads(row[0]) if row is not None else None

    def close(self):
        with self._lock:
            self._conn.close()

    def __repr__(self):
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        return f"<ArtifactStore {self.path} ({size / 1024**2:.1f} MB)>"
