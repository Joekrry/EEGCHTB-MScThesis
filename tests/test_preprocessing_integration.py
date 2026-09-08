"""End-to-end GDF loader integration for preprocessing pipelines."""

from __future__ import annotations

import numpy as np

from conftest import requires_mne

from eegchtb.data import BCICompetitionIV2bLoader, EpochWindow
from eegchtb.data.epochs import EpochBundle

SFREQ = 250.0
CH_NAMES = ["C3", "Cz", "C4"]
CUE_ONSETS = [2.0, 8.0, 14.0]
CUE_CODES = ["769", "770", "769"]


def _synthetic_raw(sfreq=SFREQ, seconds=20):
    import mne

    n_samples = int(seconds * sfreq)
    t = np.arange(n_samples) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10.0 * t),
            2.0 + np.sin(2 * np.pi * 12.0 * t),
            -2.0 + np.sin(2 * np.pi * 14.0 * t),
        ]
    )
    info = mne.create_info(CH_NAMES, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=CUE_ONSETS,
            duration=[0.0] * len(CUE_ONSETS),
            description=CUE_CODES,
        )
    )
    return raw


def _loader(preprocessing=None, cache_dir=None):
    return BCICompetitionIV2bLoader(
        root="unused",
        window=EpochWindow(0.0, 4.0),
        preprocessing=preprocessing,
        cache_dir=cache_dir,
    )


def test_loader_accepts_named_preprocessing_config():
    loader = _loader(preprocessing="minimal")

    assert loader.preprocessing is not None
    assert loader.preprocessing.name == "minimal"
    assert [step.name for step in loader.preprocessing.steps] == ["bandpass"]


@requires_mne()
def test_load_session_applies_pipeline_before_epoching(monkeypatch):
    preprocessing = {
        "name": "standardize_only",
        "steps": [{"name": "standardize"}],
    }
    loader = _loader(preprocessing=preprocessing)
    monkeypatch.setattr(loader, "load_raw", lambda subject, session: _synthetic_raw())

    bundle = loader.load_session(subject=1, session="01T")

    assert bundle.units == "standardized"
    assert bundle.metadata["preprocessing"]["name"] == "standardize_only"
    assert bundle.metadata["preprocessing"]["applied_steps"] == ["standardize"]
    assert np.max(np.abs(bundle.data)) < 5.0


@requires_mne()
def test_load_session_applies_hjorth_on_sparse_montage(monkeypatch):
    preprocessing = {
        "name": "maximum_sparse",
        "steps": [{"name": "hjorth_laplacian"}, {"name": "standardize"}],
    }
    loader = _loader(preprocessing=preprocessing)
    monkeypatch.setattr(loader, "load_raw", lambda subject, session: _synthetic_raw())

    bundle = loader.load_session(subject=1, session="01T")

    metadata = bundle.metadata["preprocessing"]
    assert metadata["applied_steps"] == ["hjorth_laplacian", "standardize"]
    assert "skipped_steps" not in metadata
    assert bundle.channel_names == CH_NAMES  # 3 channels preserved
    resolved = {s["name"]: s["resolved"] for s in metadata["resolved_steps"]}
    graph = resolved["hjorth_laplacian"]
    assert graph["k"] == 4
    assert graph["k_effective"] == 2  # min(k, n_channels - 1) = min(4, 2)
    assert graph["weighting"] == "inverse_distance"
    assert [entry["channel"] for entry in graph["neighbours"]] == CH_NAMES


@requires_mne()
def test_load_session_uses_atomic_cache(monkeypatch, tmp_path):
    preprocessing = {"name": "identity", "steps": []}
    loader = _loader(preprocessing=preprocessing, cache_dir=tmp_path)
    calls = {"load_raw": 0}

    def load_raw(subject, session):
        calls["load_raw"] += 1
        return _synthetic_raw()

    monkeypatch.setattr(loader, "load_raw", load_raw)
    first = loader.load_session(subject=1, session="01T")

    monkeypatch.setattr(
        loader,
        "load_raw",
        lambda subject, session: (_ for _ in ()).throw(RuntimeError("cache miss")),
    )
    second = loader.load_session(subject=1, session="01T")

    assert calls["load_raw"] == 1
    assert list((tmp_path / "bci_iv_2b").glob("*.npz"))
    np.testing.assert_array_equal(second.data, first.data)
    assert second.metadata["preprocessing"]["name"] == "identity"


def test_epoch_bundle_round_trips_metadata(tmp_path):
    bundle = EpochBundle(
        data=np.zeros((1, 1, 2), dtype=np.float32),
        labels=np.array([0]),
        dataset="demo",
        subject=1,
        session="T",
        sfreq=250.0,
        channel_names=["Cz"],
        class_names={0: "left_hand"},
        units="standardized",
        metadata={"preprocessing": {"name": "standard"}},
    )
    path = tmp_path / "bundle.npz"

    bundle.save(path)
    loaded = EpochBundle.load(path)

    assert loaded.units == "standardized"
    assert loaded.metadata == bundle.metadata
