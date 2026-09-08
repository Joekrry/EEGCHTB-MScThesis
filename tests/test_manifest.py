"""Tests for split manifests and run provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eegchtb.evaluation.manifest import (
    SplitManifest,
    collect_provenance,
    write_run_manifest,
)
from eegchtb.evaluation.splits import subject_specific


def _split():
    subjects = np.repeat([1, 2, 3], 8)
    labels = np.tile([0, 1], 12)
    test_mask = np.tile([False] * 4 + [True] * 4, 3)
    return subject_specific(subjects, test_mask, labels, seed=0, val_fraction=0.25)[0]


def test_manifest_reconstructs_split_and_roundtrips(tmp_path):
    split = _split()
    manifest = SplitManifest.from_split(split)
    rebuilt = manifest.to_split()
    for field in ("train", "val", "test"):
        assert np.array_equal(getattr(rebuilt, field), getattr(split, field))

    path = tmp_path / "split.json"
    manifest.save(path)
    assert SplitManifest.load(path) == manifest


def test_verify_disjoint_raises_on_overlap():
    bad = SplitManifest("subject_specific", 0, "1", train=[0, 1], val=[2], test=[1, 3])
    with pytest.raises(ValueError, match="overlap"):
        bad.verify_disjoint()


def test_provenance_has_reproducibility_fields():
    prov = collect_provenance()
    assert prov["python"]
    assert "numpy" in prov["packages"] and "torch" in prov["packages"]
    assert "git_revision" in prov
    assert "cuda_available" in prov["device"]


def test_write_run_manifest_is_reloadable(tmp_path):
    config = {"dataset": "bci_iv_2a", "model": "eegnet", "preprocessing": "standard", "seed": 0}
    path = write_run_manifest(tmp_path, config=config, split=_split())

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["config"] == config
    assert record["config_fingerprint"]
    assert set(record["split"]) == {"protocol", "seed", "fold", "train", "val", "test"}
    assert "provenance" in record
