"""Full-session preprocessing validation on real BCI IV data (Gate 1)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_mne

from eegchtb.data import (
    BCICompetitionIV2aLoader,
    BCICompetitionIV2bLoader,
    EpochWindow,
)

WINDOW = EpochWindow(tmin=0.0, tmax=4.0)
CONFIGS = ["minimal", "standard", "deep", "maximum"]

# minimal keeps physical units; the other three standardise. deep and maximum
# resample 250 -> 125 Hz, so a 4 s window is 500 samples instead of 1000.
EXPECTED_UNITS = {
    "minimal": "uV",
    "standard": "standardized",
    "deep": "standardized",
    "maximum": "standardized",
}
EXPECTED_SFREQ = {"minimal": 250.0, "standard": 250.0, "deep": 125.0, "maximum": 125.0}


def _expected_n_times(config: str) -> int:
    return int(round(4.0 * EXPECTED_SFREQ[config]))


# BCI IV-2a (22 channels, 4 classes)
@requires_mne()
@pytest.mark.parametrize("config", CONFIGS)
def test_2a_all_configs_units_shape_provenance_and_cues(bci_iv_2a_dir, config):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW, preprocessing=config)
    bundle = loader.load_session(subject=1, session="T")

    assert bundle.units == EXPECTED_UNITS[config]
    assert bundle.sfreq == EXPECTED_SFREQ[config]
    assert bundle.n_times == _expected_n_times(config)
    # restricted by default to the 4 Galea channels of 2a; the Hjorth Laplacian in
    # maximum is a spatial high-pass that keeps every channel, so the montage is
    # unchanged across all four levels
    assert bundle.channel_names == ["C3", "Cz", "C4", "Pz"]
    # cues survive preprocessing (incl. resampling that moves annotations)
    assert bundle.class_distribution() == {0: 72, 1: 72, 2: 72, 3: 72}
    # time axis is aligned to the cue and matches the sample count
    assert bundle.times is not None
    assert len(bundle.times) == bundle.n_times
    assert abs(float(bundle.times[0])) < 1e-6

    prep = bundle.metadata["preprocessing"]
    assert prep["name"] == config
    assert prep["output_units"] == EXPECTED_UNITS[config]
    assert prep["applied_steps"]


@requires_mne()
def test_2a_maximum_applies_hjorth_on_galea_subset(bci_iv_2a_dir):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW, preprocessing="maximum")
    bundle = loader.load_session(1, "T")
    prep = bundle.metadata["preprocessing"]
    # the single montage-adaptive Hjorth Laplacian runs identically on any montage;
    # on the 4 Galea channels of 2a it caps the neighbour count at min(k, n - 1) = 3
    assert "hjorth_laplacian" in prep["applied_steps"]
    assert "skipped_steps" not in prep
    assert bundle.channel_names == ["C3", "Cz", "C4", "Pz"]
    resolved = {s["name"]: s["resolved"] for s in prep["resolved_steps"]}
    graph = resolved["hjorth_laplacian"]
    assert graph["k_effective"] == 3
    assert graph["weighting"] == "inverse_distance"
    assert graph["channels"] == ["C3", "Cz", "C4", "Pz"]


# BCI IV-2b (3 channels, 2 classes)
@requires_mne()
@pytest.mark.parametrize("config", CONFIGS)
def test_2b_all_configs_units_shape_and_provenance(bci_iv_2b_dir, config):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW, preprocessing=config)
    bundle = loader.load_session(subject=1, session="01T")

    assert bundle.units == EXPECTED_UNITS[config]
    assert bundle.sfreq == EXPECTED_SFREQ[config]
    assert bundle.n_times == _expected_n_times(config)
    # the Hjorth Laplacian in maximum keeps every channel, so all four levels run
    # on the full 3-channel montage
    assert bundle.n_channels == 3
    assert set(bundle.filter_known().labels.tolist()).issubset({0, 1})

    prep = bundle.metadata["preprocessing"]
    assert prep["name"] == config
    assert prep["applied_steps"]


@requires_mne()
def test_2b_maximum_applies_hjorth_and_records_the_graph(bci_iv_2b_dir):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW, preprocessing="maximum")
    bundle = loader.load_session(1, "01T")

    prep = bundle.metadata["preprocessing"]
    # the same Hjorth Laplacian as every other montage; on 2b's 3 channels the
    # neighbour count caps at min(k, n - 1) = 2 and all channels are kept
    assert "hjorth_laplacian" in prep["applied_steps"]
    assert "skipped_steps" not in prep
    assert bundle.channel_names == ["C3", "Cz", "C4"]
    resolved = {s["name"]: s["resolved"] for s in prep["resolved_steps"]}
    graph = resolved["hjorth_laplacian"]
    assert graph["k_effective"] == 2
    assert graph["channels"] == ["C3", "Cz", "C4"]


@requires_mne()
def test_2b_deep_is_deterministic_across_reruns(bci_iv_2b_dir):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW, preprocessing="deep")
    first = loader.load_session(1, "01T").data
    second = loader.load_session(1, "01T").data
    np.testing.assert_array_equal(first, second)
