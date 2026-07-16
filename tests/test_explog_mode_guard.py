# Tests for src.explog_mode_guard -- the marker file that stops a run
# directory's EXPLOG mode from silently changing between resumes.

import pytest

from src.explog_mode_guard import (
    MODE_MARKER_NAME,
    check_or_record_explog_mode,
    read_recorded_mode,
    refuse_overwrite_of_production_run,
)


def test_first_call_records_the_mode(tmp_path):
    marker = tmp_path / MODE_MARKER_NAME
    assert not marker.exists()

    check_or_record_explog_mode(tmp_path, "production")

    assert marker.exists()
    assert marker.read_text().strip() == "production"


def test_matching_mode_on_later_call_is_a_noop(tmp_path):
    check_or_record_explog_mode(tmp_path, "production")
    marker = tmp_path / MODE_MARKER_NAME
    before = marker.read_text()

    check_or_record_explog_mode(tmp_path, "production")

    assert marker.read_text() == before


def test_mismatched_mode_raises(tmp_path):
    check_or_record_explog_mode(tmp_path, "production")

    with pytest.raises(RuntimeError, match="production"):
        check_or_record_explog_mode(tmp_path, "test")

    # refusing to start must not silently overwrite the recorded mode
    marker = tmp_path / MODE_MARKER_NAME
    assert marker.read_text().strip() == "production"


def test_mismatched_mode_error_names_both_modes_and_the_marker_path(tmp_path):
    check_or_record_explog_mode(tmp_path, "test")

    with pytest.raises(RuntimeError) as exc_info:
        check_or_record_explog_mode(tmp_path, "production")

    msg = str(exc_info.value)
    assert "'production'" in msg
    assert "'test'" in msg
    assert str(tmp_path / MODE_MARKER_NAME) in msg


def test_marker_file_content_is_whitespace_tolerant(tmp_path):
    # a hand-edited or newline-terminated marker file should still compare
    # correctly against the requested mode
    marker = tmp_path / MODE_MARKER_NAME
    marker.write_text("production\n")

    check_or_record_explog_mode(tmp_path, "production")  # must not raise

    with pytest.raises(RuntimeError):
        check_or_record_explog_mode(tmp_path, "test")


def test_read_recorded_mode_is_none_before_any_init(tmp_path):
    assert read_recorded_mode(tmp_path) is None


def test_read_recorded_mode_returns_what_was_recorded(tmp_path):
    check_or_record_explog_mode(tmp_path, "production")
    assert read_recorded_mode(tmp_path) == "production"


def test_refuse_overwrite_is_a_noop_when_never_initialized(tmp_path):
    refuse_overwrite_of_production_run(tmp_path)  # must not raise -- nothing to protect


def test_refuse_overwrite_is_a_noop_for_a_recorded_test_mode(tmp_path):
    check_or_record_explog_mode(tmp_path, "test")
    refuse_overwrite_of_production_run(tmp_path)  # must not raise


def test_refuse_overwrite_raises_for_a_recorded_production_run(tmp_path):
    check_or_record_explog_mode(tmp_path, "production")

    with pytest.raises(RuntimeError, match="production"):
        refuse_overwrite_of_production_run(tmp_path)
