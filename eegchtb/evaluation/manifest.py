"""Split manifests and run provenance: persist exactly what is needed to reproduce a run."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..run_output import atomic_write
from .splits import Split


@dataclass
class SplitManifest:
    """Exact train/val/test trial membership for one split, persistable and reconstructable."""

    protocol: str
    seed: int
    fold: str
    train: list[int]
    val: list[int]
    test: list[int]

    @classmethod
    def from_split(cls, split: Split, fold: str | None = None) -> "SplitManifest":
        meta = split.meta
        if fold is None:
            fold = str(meta.get("subject", ""))
        return cls(
            protocol=str(meta.get("protocol", "")),
            seed=int(meta.get("seed", 0)),
            fold=fold,
            train=[int(i) for i in split.train],
            val=[int(i) for i in split.val],
            test=[int(i) for i in split.test],
        )

    def to_split(self) -> Split:
        import numpy as np

        return Split(
            np.asarray(self.train, dtype=np.int64),
            np.asarray(self.val, dtype=np.int64),
            np.asarray(self.test, dtype=np.int64),
            {"protocol": self.protocol, "seed": self.seed, "fold": self.fold},
        )

    def verify_disjoint(self) -> None:
        """Raise if train/val/test share any trial index (automated leakage check)."""
        sets = {"train": set(self.train), "val": set(self.val), "test": set(self.test)}
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            if sets[a] & sets[b]:
                raise ValueError(f"manifest partitions overlap: {a} and {b}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SplitManifest":
        return cls(**data)

    def save(self, path) -> None:
        with atomic_write(path, mode="w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path) -> "SplitManifest":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def _package_version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:
        return None


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _device_info() -> dict:
    info: dict = {"cuda_available": False}
    try:
        import torch
    except ImportError:
        return info
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def collect_provenance() -> dict:
    """Environment, software versions, device, and git revision needed to reproduce a run."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "scikit_learn": _package_version("sklearn"),
            "mne": _package_version("mne"),
        },
        "git_revision": _git_revision(),
        "device": _device_info(),
    }


def write_run_manifest(
    directory,
    *,
    config: dict,
    split: Split,
    fold: str | None = None,
    channels_used: list[str] | None = None,
    preprocessing_provenance: dict | None = None,
) -> Path:
    """Write split membership, config fingerprint, provenance and (optionally) channels_used / preprocessing_provenance to directory/manifest.json."""
    from .results import config_fingerprint

    manifest = {
        "config": config,
        "config_fingerprint": config_fingerprint(config),
        "split": SplitManifest.from_split(split, fold).to_dict(),
        "provenance": collect_provenance(),
    }
    if channels_used is not None:
        manifest["channels_used"] = list(channels_used)
    if preprocessing_provenance is not None:
        manifest["preprocessing_provenance"] = preprocessing_provenance
    path = Path(directory) / "manifest.json"
    with atomic_write(path, mode="w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path
