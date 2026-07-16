"""Guard against a run directory's EXPLOG mode silently changing between
resumes -- e.g. a stray EXPLOG_MODE = "test" edit landing on top of a real
production run, which once fabricated a batch of job "completions" and
required a full checkpoint rollback to fix.

Pure and stdlib-only (like src/history_log.py, src/disposition_messages.py):
no src.tools import, so this stays in the fast test tier.
"""
from pathlib import Path

MODE_MARKER_NAME = ".explog_mode"


def check_or_record_explog_mode(working_directory, mode: str) -> None:
    """First call for a run directory records `mode` in a marker file.
    Every later call must pass the same `mode`, or this raises RuntimeError
    instead of silently letting the run start in a different mode."""
    marker = Path(working_directory) / MODE_MARKER_NAME
    if marker.exists():
        recorded = marker.read_text().strip()
        if recorded != mode:
            raise RuntimeError(
                f"EXPLOG_MODE is '{mode}', but {marker} says this run "
                f"directory was previously initialized with mode "
                f"'{recorded}'. Refusing to start. If switching modes for "
                f"this run directory is really intended, delete {marker}."
            )
    else:
        marker.write_text(mode)
