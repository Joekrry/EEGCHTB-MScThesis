"""The montage-adaptive Hjorth surface Laplacian: operator algebra, montage adaptivity, and conditioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_mne

from eegchtb.preprocessing import Pipeline, hjorth_laplacian_operator
from eegchtb.preprocessing.steps import CurrentSourceDensity, HjorthLaplacian

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "preprocessing"

# The three montages the project runs on, all subsets of standard_1020:
# IV-2b (3), the IV-2a Galea subset (4) and the full Galea headset (10).
MONTAGES = {
    "iv2b_3ch": ["C3", "Cz", "C4"],
    "iv2a_galea_4ch": ["C3", "Cz", "C4", "Pz"],
    "galea_10ch": ["F1", "F2", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"],
}


def _line_coords(n: int) -> np.ndarray:
    """n electrodes spaced unevenly along a line; distinct positions, no ties needed."""
    return np.column_stack([np.arange(n, dtype=float) ** 1.3, np.zeros(n), np.zeros(n)])


def _mean_abs_offdiag_corr(data: np.ndarray) -> float:
    corr = np.corrcoef(data)
    upper = np.triu_indices(corr.shape[0], k=1)
    return float(np.mean(np.abs(corr[upper])))


# operator algebra (pure numpy, no MNE)
@pytest.mark.parametrize("n", [3, 4, 10])
@pytest.mark.parametrize("weighting", ["inverse_distance", "uniform"])
def test_rows_sum_to_zero_and_diagonal_is_one(n, weighting):
    operator, _ = hjorth_laplacian_operator(_line_coords(n), k=4, weighting=weighting)
    assert operator.shape == (n, n)
    np.testing.assert_allclose(operator.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(operator), 1.0)


@pytest.mark.parametrize("n,expected", [(3, 2), (4, 3), (10, 4)])
def test_k_eff_is_capped_at_n_minus_one(n, expected):
    # k = 4 throughout; k_eff = min(k, n - 1), so the operator runs on any montage
    # with no channel-count threshold. Each row has exactly k_eff off-diagonal
    # weights.
    operator, graph = hjorth_laplacian_operator(_line_coords(n), k=4)
    assert all(len(entry["neighbours"]) == expected for entry in graph)
    n_nonzero_offdiag = np.count_nonzero(operator - np.diag(np.diag(operator)), axis=1)
    assert list(n_nonzero_offdiag) == [expected] * n


def test_uniform_three_channel_case_is_the_bipolar_derivation():
    # {C3, Cz, C4} with uniform weighting: k_eff = min(4, 2) = 2, so every channel
    # is referenced against the mean of the other two. This is exactly the local
    # bipolar Laplacian; for the central channel it is Cz - (C3 + C4) / 2, and the
    # analogous rows for C3 and C4.
    operator, _ = hjorth_laplacian_operator(_line_coords(3), k=4, weighting="uniform")
    np.testing.assert_allclose(operator[0], [1.0, -0.5, -0.5], atol=1e-12)  # C3
    np.testing.assert_allclose(operator[1], [-0.5, 1.0, -0.5], atol=1e-12)  # Cz
    np.testing.assert_allclose(operator[2], [-0.5, -0.5, 1.0], atol=1e-12)  # C4


@pytest.mark.parametrize("weighting", ["inverse_distance", "uniform"])
def test_signal_common_to_all_channels_maps_to_zero(weighting):
    # rows sum to zero, so a spatial-DC signal (identical across channels) is
    # removed: the operator is a spatial high-pass
    operator, _ = hjorth_laplacian_operator(_line_coords(4), k=4, weighting=weighting)
    rng = np.random.default_rng(0)
    common = rng.standard_normal(200)
    identical = np.tile(common, (4, 1))
    np.testing.assert_allclose(operator @ identical, 0.0, atol=1e-12)


def test_operator_is_deterministic():
    coords = _line_coords(6)
    first, _ = hjorth_laplacian_operator(coords, k=4)
    second, _ = hjorth_laplacian_operator(coords.copy(), k=4)
    np.testing.assert_array_equal(first, second)


def test_inverse_distance_weights_nearer_electrodes_more():
    # neighbours at distances 1 and 2 from channel 0 -> weights proportional to
    # 1 and 1/2, normalised to 2/3 and 1/3
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    operator, _ = hjorth_laplacian_operator(coords, k=2, weighting="inverse_distance")
    np.testing.assert_allclose(operator[0], [1.0, -2.0 / 3.0, -1.0 / 3.0], atol=1e-12)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="weighting"):
        hjorth_laplacian_operator(_line_coords(3), weighting="nope")
    with pytest.raises(ValueError, match="k must be"):
        hjorth_laplacian_operator(_line_coords(3), k=0)
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        hjorth_laplacian_operator(np.zeros((3, 2)))


# montage resolution and the maximum pipeline (MNE-gated)
def _correlated_raw(ch_names, sfreq=250.0, seconds=30):
    import mne

    n = int(sfreq * seconds)
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(0)
    # a shared volume-conducted source plus per-channel noise, so the raw channels
    # are strongly correlated before any spatial transform
    shared = np.sin(2 * np.pi * 11.0 * t) + 0.5 * np.sin(2 * np.pi * 21.0 * t)
    data = np.vstack([shared + 0.3 * rng.standard_normal(n) for _ in ch_names]) * 1e-5
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


@requires_mne()
@pytest.mark.parametrize("name,ch_names", list(MONTAGES.items()))
def test_step_runs_on_every_montage_without_a_threshold(name, ch_names):
    raw = _correlated_raw(ch_names)
    out = HjorthLaplacian().apply(raw)
    # a spatial high-pass keeps the channel and sample counts
    assert out.ch_names == list(ch_names)
    assert out.n_times == raw.n_times


@requires_mne()
@pytest.mark.parametrize("name,ch_names", list(MONTAGES.items()))
def test_resolved_provenance_records_the_neighbour_graph(name, ch_names):
    provenance = HjorthLaplacian().resolved_provenance(_correlated_raw(ch_names))
    assert provenance["k"] == 4
    assert provenance["k_effective"] == min(4, len(ch_names) - 1)
    assert provenance["weighting"] == "inverse_distance"
    assert provenance["channels"] == list(ch_names)
    # one entry per channel, each naming k_eff neighbours by channel name
    assert [e["channel"] for e in provenance["neighbours"]] == list(ch_names)
    for entry in provenance["neighbours"]:
        assert len(entry["neighbours"]) == min(4, len(ch_names) - 1)
        assert entry["channel"] not in entry["neighbours"]
        np.testing.assert_allclose(sum(entry["weights"]), 1.0, atol=1e-12)


@requires_mne()
@pytest.mark.parametrize("name,ch_names", list(MONTAGES.items()))
def test_maximum_pipeline_runs_on_every_montage(name, ch_names):
    raw = _correlated_raw(ch_names)
    out = Pipeline.from_yaml(CONFIG_DIR / "maximum.yaml").apply(raw)
    # deep's resample to 125 Hz, and a spatial high-pass that keeps every channel
    assert out.info["sfreq"] == 125.0
    assert out.ch_names == list(ch_names)
    assert out.n_times == int(round(raw.n_times / 2))


@requires_mne()
def test_step_is_deterministic_on_a_montage():
    first = HjorthLaplacian().apply(_correlated_raw(MONTAGES["galea_10ch"])).get_data()
    second = HjorthLaplacian().apply(_correlated_raw(MONTAGES["galea_10ch"])).get_data()
    np.testing.assert_array_equal(first, second)


@requires_mne()
def test_csd_is_far_worse_conditioned_than_hjorth_on_the_4ch_subset():
    # The rationale for replacing CSD: on the 4-channel IV-2a Galea subset,
    # spherical-spline CSD is underdetermined and collapses the channels toward
    # collinearity, while the Hjorth Laplacian stays well conditioned
    # (methodology Sec. 3.4/3.9).
    raw = _correlated_raw(MONTAGES["iv2a_galea_4ch"])
    hjorth_corr = _mean_abs_offdiag_corr(HjorthLaplacian().apply(raw.copy()).get_data())
    csd_corr = _mean_abs_offdiag_corr(CurrentSourceDensity().apply(raw.copy()).get_data())
    assert csd_corr > 0.8  # CSD is near-collinear on this sparse montage
    assert hjorth_corr < 0.5  # the Hjorth Laplacian stays well conditioned
    assert hjorth_corr < 0.5 * csd_corr  # CSD mean|corr| >> Hjorth mean|corr|
