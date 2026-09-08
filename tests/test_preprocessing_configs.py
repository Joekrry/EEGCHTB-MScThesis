"""The four pipeline configs and their determinism (project plan Task 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_mne

from eegchtb.preprocessing import Pipeline

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "preprocessing"

# expected step order per level, so a config edit that silently drops or
# reorders a step is caught here rather than only showing up as a metric change
EXPECTED_STEPS = {
    "minimal": ["bandpass"],
    "standard": ["bandpass", "standardize"],
    "deep": ["bandpass", "rereference", "resample", "standardize"],
    "maximum": [
        "bandpass",
        "rereference",
        "resample",
        "hjorth_laplacian",
        "standardize",
    ],
}
PIPELINE_NAMES = list(EXPECTED_STEPS)


def _config_path(name: str) -> Path:
    return CONFIG_DIR / f"{name}.yaml"


def test_all_four_config_files_exist():
    for name in PIPELINE_NAMES:
        assert _config_path(name).exists(), f"missing config: {name}.yaml"


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_config_loads_with_expected_steps(name):
    # building a pipeline validates every step name against the registry
    pipe = Pipeline.from_yaml(_config_path(name))
    assert pipe.name == name
    assert [s.name for s in pipe.steps] == EXPECTED_STEPS[name]


def _small_sine_raw(sfreq=250.0, seconds=20, freqs=(10.0, 22.0, 60.0)):
    """3-channel synthetic Raw; enough for minimal/standard/deep (no csd)."""
    import mne

    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    data = np.tile(signal, (3, 1))
    info = mne.create_info(["C3", "Cz", "C4"], sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _montage_sine_raw(sfreq=250.0, seconds=20, freqs=(10.0, 22.0, 60.0)):
    """19-channel synthetic Raw with a full 10-20 layout, for maximum (Hjorth Laplacian).

    Each channel gets a distinct per-channel component on top of a shared signal,
    so that the average reference does not null the data and the spatial Laplacian
    has genuine inter-channel structure to act on.
    """
    import mne

    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    ]
    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    shared = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    rng = np.random.default_rng(0)
    data = np.vstack(
        [shared + 0.5 * rng.standard_normal(n) for _ in ch_names]
    )
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _raw_for(name: str):
    return _montage_sine_raw() if name == "maximum" else _small_sine_raw()


@requires_mne()
@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_pipeline_is_deterministic_across_reruns(name):
    raw = _raw_for(name)
    pipe = Pipeline.from_yaml(_config_path(name))
    first = pipe.apply(raw).get_data()
    second = pipe.apply(raw).get_data()
    np.testing.assert_array_equal(first, second)


@requires_mne()
def test_pipelines_produce_different_representations():
    # the modulation gradient must actually modulate: minimal and standard keep
    # the same sampling rate but different bands and standardisation, so their
    # output must differ
    raw = _small_sine_raw()
    minimal = Pipeline.from_yaml(_config_path("minimal")).apply(raw).get_data()
    standard = Pipeline.from_yaml(_config_path("standard")).apply(raw).get_data()
    assert not np.array_equal(minimal, standard)


@requires_mne()
def test_deep_and_maximum_differ_only_by_hjorth_laplacian():
    # maximum is deep's steps plus the Hjorth Laplacian (see
    # configs/preprocessing/maximum.yaml); on the same input the two must diverge
    # because of it, while keeping the same (channels, samples) shape
    raw = _montage_sine_raw()
    deep = np.asarray(Pipeline.from_yaml(_config_path("deep")).apply(raw).get_data())
    maximum = np.asarray(
        Pipeline.from_yaml(_config_path("maximum")).apply(raw).get_data()
    )
    assert deep.shape == maximum.shape
    assert not np.array_equal(deep, maximum)
