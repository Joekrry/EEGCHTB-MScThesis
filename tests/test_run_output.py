"""Tests for the atomic, self-cleaning run-output utility."""

from __future__ import annotations

import pytest

from eegchtb.run_output import (
    COMPLETION_MARKER,
    atomic_write,
    is_complete,
    new_run_id,
    run_directory,
    sweep_incomplete,
)


# atomic_write
def test_atomic_write_creates_file_with_no_temp_left(tmp_path):
    target = tmp_path / "out" / "data.bin"
    with atomic_write(target) as f:
        f.write(b"hello")
    assert target.read_bytes() == b"hello"
    # no leftover temp files in the directory
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_removes_temp_and_leaves_no_target_on_error(tmp_path):
    target = tmp_path / "data.bin"
    with pytest.raises(RuntimeError):
        with atomic_write(target) as f:
            f.write(b"partial")
            raise RuntimeError("boom")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_does_not_corrupt_existing_file_on_error(tmp_path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with atomic_write(target) as f:
            f.write(b"new-but-doomed")
            raise RuntimeError("boom")
    # existing file is untouched, no temp remains
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_text_mode(tmp_path):
    target = tmp_path / "note.txt"
    with atomic_write(target, mode="w", encoding="utf-8") as f:
        f.write("line\n")
    assert target.read_text(encoding="utf-8") == "line\n"


# run_directory
def test_run_directory_success_writes_completion_marker(tmp_path):
    with run_directory(tmp_path, "run-1") as d:
        (d / "result.txt").write_text("done", encoding="utf-8")
    assert (tmp_path / "run-1").is_dir()
    assert is_complete(tmp_path / "run-1")
    assert (tmp_path / "run-1" / COMPLETION_MARKER).exists()


def test_run_directory_deletes_everything_on_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with run_directory(tmp_path, "run-2") as d:
            (d / "partial.txt").write_text("half", encoding="utf-8")
            raise RuntimeError("boom")
    assert not (tmp_path / "run-2").exists()


def test_run_directory_deletes_on_keyboard_interrupt(tmp_path):
    # Ctrl+C raises KeyboardInterrupt, which is not an Exception subclass
    with pytest.raises(KeyboardInterrupt):
        with run_directory(tmp_path, "run-3") as d:
            (d / "partial.txt").write_text("half", encoding="utf-8")
            raise KeyboardInterrupt
    assert not (tmp_path / "run-3").exists()


def test_run_directory_can_keep_output_on_failure_when_disabled(tmp_path):
    with pytest.raises(RuntimeError):
        with run_directory(tmp_path, "run-4", cleanup_on_failure=False) as d:
            (d / "partial.txt").write_text("half", encoding="utf-8")
            raise RuntimeError("boom")
    assert (tmp_path / "run-4").exists()
    assert not is_complete(tmp_path / "run-4")


def test_run_directory_refuses_existing_directory(tmp_path):
    (tmp_path / "run-5").mkdir()
    with pytest.raises(FileExistsError):
        with run_directory(tmp_path, "run-5"):
            pass


# sweep_incomplete
def test_sweep_removes_incomplete_keeps_complete(tmp_path):
    with run_directory(tmp_path, "good"):
        pass  # completes and gets a marker
    incomplete = tmp_path / "bad"
    incomplete.mkdir()
    (incomplete / "leftover.txt").write_text("x", encoding="utf-8")

    removed = sweep_incomplete(tmp_path)

    assert incomplete in removed
    assert not incomplete.exists()
    assert (tmp_path / "good").exists()


def test_sweep_on_missing_root_returns_empty(tmp_path):
    assert sweep_incomplete(tmp_path / "does-not-exist") == []


def test_new_run_id_is_unique_and_prefixed():
    a = new_run_id("cell")
    b = new_run_id("cell")
    assert a.startswith("cell-")
    assert a != b
