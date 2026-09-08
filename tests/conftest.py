"""Shared pytest configuration: locates the BCI IV datasets for the data-gated tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA_DIR = REPO_ROOT / "eegchtb" / "data"


def _dataset_dir(env_var: str, signal_glob: str) -> Path | None:
    """Resolve a dataset dir: env-var override, else the in-repo eegchtb/data.

    Only returns a directory that actually holds the dataset's signal files, so
    tests skip cleanly when the (gitignored) GDF files are not present. The
    search is recursive because the files may be flat or in the train/test
    subdirectories the loaders also accept.
    """
    candidates: list[Path] = []
    value = os.environ.get(env_var)
    if value:
        candidates.append(Path(value))
    candidates.append(PACKAGE_DATA_DIR)
    for path in candidates:
        if path.exists() and next(path.rglob(signal_glob), None) is not None:
            return path
    return None


def requires_mne():
    """Skip marker for when MNE-Python is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec("mne") is None,
        reason="mne not installed",
    )


def requires_torch():
    """Skip marker for when PyTorch is not installed."""
    return pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )


@pytest.fixture
def bci_iv_2a_dir() -> Path:
    path = _dataset_dir("BCI_IV_2A_DIR", "A*.gdf")
    if path is None:
        pytest.skip("BCI IV-2a GDF files not found (set BCI_IV_2A_DIR or place under eegchtb/data)")
    return path


@pytest.fixture
def bci_iv_2b_dir() -> Path:
    path = _dataset_dir("BCI_IV_2B_DIR", "B*.gdf")
    if path is None:
        pytest.skip("BCI IV-2b GDF files not found (set BCI_IV_2B_DIR or place under eegchtb/data)")
    return path
