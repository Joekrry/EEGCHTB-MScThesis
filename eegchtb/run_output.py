"""Atomic, self-cleaning run outputs: a failed or interrupted run leaves no partial files."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

# marker file written into a run directory upon complete run
COMPLETION_MARKER = ".run_complete"


def new_run_id(prefix: str = "run") -> str:
    """Return a unique run id of the form "<prefix>-YYYYmmdd-HHMMSS-ffffff"."""
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


@contextmanager
def atomic_write(path, mode: str = "wb", encoding: str | None = None) -> Iterator:
    """Write to path atomically via a temp file and os.replace; on failure path is left untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, encoding=encoding) as handle:
            yield handle
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_complete(directory) -> bool:
    """Return True if directory holds the completion marker."""
    return (Path(directory) / COMPLETION_MARKER).exists()


@contextmanager
def run_directory(
    root, run_id: str, *, cleanup_on_failure: bool = True
) -> Iterator[Path]:
    """Per-run output directory: writes a completion marker on clean exit, deleted whole on failure."""
    directory = Path(root) / run_id
    if directory.exists():
        raise FileExistsError(f"run directory already exists: {directory}")
    directory.mkdir(parents=True)
    try:
        yield directory
    except BaseException:
        if cleanup_on_failure:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    else:
        (directory / COMPLETION_MARKER).write_text("ok\n", encoding="utf-8")


def sweep_incomplete(root) -> list[Path]:
    """Delete run directories under root that lack a completion marker; returns those removed."""
    root = Path(root)
    removed: list[Path] = []
    if not root.exists():
        return removed
    for child in sorted(root.iterdir()):
        if child.is_dir() and not is_complete(child):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed
