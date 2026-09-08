"""Tests for the single-experiment runner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import requires_torch

from eegchtb.data import EpochBundle


class _FakeLoader:
    """Two subjects, each with a T and E session of a small learnable 2-class bundle."""

    # a stand-in for the preprocessing provenance a real loader attaches, so the
    # runner's threading of the resolved neighbour graph into the manifest can be
    # tested without MNE
    PREPROCESSING_METADATA = {
        "name": "maximum",
        "applied_steps": ["hjorth_laplacian"],
        "resolved_steps": [
            {
                "name": "hjorth_laplacian",
                "resolved": {
                    "k": 4,
                    "k_effective": 2,
                    "weighting": "inverse_distance",
                    "channels": ["C3", "Cz", "C4"],
                },
            }
        ],
    }

    def __init__(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 1, 128)
        burst = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        self._bundles = {}
        for subject in (1, 2):
            for session in ("T", "E"):
                labels = np.array([0, 1] * 4)
                data = (rng.standard_normal((8, 3, 128)) * 0.1).astype(np.float32)
                data[labels == 1] += burst
                self._bundles[(subject, session)] = EpochBundle(
                    data=data,
                    labels=labels,
                    dataset="fake",
                    subject=subject,
                    session=session,
                    sfreq=128.0,
                    channel_names=["C3", "Cz", "C4"],
                    class_names={0: "a", 1: "b"},
                    trial_ids=np.arange(8),
                    metadata={"preprocessing": self.PREPROCESSING_METADATA},
                )

    def available_subjects(self):
        return [1, 2]

    def available_sessions(self, subject):
        return ["T", "E"]

    def load_session(self, subject, session):
        return self._bundles[(subject, session)]


def _config():
    from eegchtb.experiment import ExperimentConfig

    return ExperimentConfig(
        dataset="fake",
        preprocessing="none",
        protocol="subject_specific",
        seed=0,
        subject=1,
        epochs=3,
        batch_size=4,
    )


@requires_torch()
def test_run_creates_self_contained_directory(tmp_path):
    from eegchtb.experiment import run_experiment

    path = run_experiment(_config(), tmp_path, loader=_FakeLoader(), run_id="smoke")

    assert path == tmp_path / "smoke"
    assert (path / "manifest.json").exists()
    assert (path / ".run_complete").exists()  # run finished cleanly
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    assert set(result["metrics"]) == {"accuracy", "macro_f1", "cohen_kappa", "mcc"}
    assert result["n_trials"]["test"] == 8  # subject 1's E session

    # the run identity records the epoch window and the resolved preprocessing
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    assert cfg["window"]["tmin"] == 0.0 and cfg["window"]["tmax"] == 4.0
    assert "preprocessing_resolved" in cfg
    # and what preprocessing resolved to at run time (the Hjorth neighbour graph
    # and its effective neighbour count) is recorded in the manifest
    provenance = manifest["preprocessing_provenance"]
    resolved = provenance["resolved_steps"][0]["resolved"]
    assert resolved["k_effective"] == 2
    assert resolved["weighting"] == "inverse_distance"


@requires_torch()
def test_save_snapshot_writes_reloadable_model(tmp_path):
    import torch

    from eegchtb.experiment import load_snapshot, run_experiment

    path = run_experiment(
        _config(), tmp_path, loader=_FakeLoader(), run_id="snap", save_snapshot=True
    )
    snapshot_path = path / "model.pt"
    assert snapshot_path.exists()  # only written when save_snapshot=True

    model, snapshot = load_snapshot(snapshot_path)
    assert snapshot["model"] == "eegnet"
    assert (snapshot["n_channels"], snapshot["n_times"], snapshot["n_classes"]) == (3, 128, 2)
    # the rebuilt model carries the trained weights and runs
    out = model(torch.zeros(2, 3, 128))
    assert out.shape == (2, 2)
    # reload is deterministic: two loads give identical predictions
    model2, _ = load_snapshot(snapshot_path)
    x = torch.randn(4, 3, 128)
    assert torch.equal(model(x), model2(x))


@requires_torch()
def test_no_snapshot_by_default(tmp_path):
    from eegchtb.experiment import run_experiment

    path = run_experiment(_config(), tmp_path, loader=_FakeLoader(), run_id="nosnap")
    assert not (path / "model.pt").exists()  # off unless requested


@requires_torch()
def test_forced_failure_removes_run_directory(tmp_path, monkeypatch):
    import eegchtb.experiment as experiment

    def boom(*args, **kwargs):
        raise RuntimeError("training blew up")

    monkeypatch.setattr(experiment, "train_model", boom)
    with pytest.raises(RuntimeError):
        experiment.run_experiment(_config(), tmp_path, loader=_FakeLoader(), run_id="boom")

    assert not (tmp_path / "boom").exists()  # nothing left
