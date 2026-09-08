"""Evaluation layer: PyTorch adapters, splits, and metrics over EpochBundles."""

from __future__ import annotations

from .dataset import EpochBundleDataset
from .manifest import SplitManifest, collect_provenance, write_run_manifest
from .metrics import METRIC_NAMES, compute_metrics
from .results import RunResult, config_fingerprint
from .splits import Split, subject_specific

__all__ = [
    "EpochBundleDataset",
    "Split",
    "subject_specific",
    "METRIC_NAMES",
    "compute_metrics",
    "RunResult",
    "config_fingerprint",
    "SplitManifest",
    "collect_provenance",
    "write_run_manifest",
]
