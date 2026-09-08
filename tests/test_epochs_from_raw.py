"""Tests for the load_raw / epochs_from_raw split, using a synthetic mne.Raw."""

from __future__ import annotations

import numpy as np

from conftest import requires_mne

from eegchtb.data import BCICompetitionIV2bLoader, EpochWindow

SFREQ = 250.0
CH_NAMES = ["C3", "Cz", "C4"]
# cue onsets (seconds) and their GDF codes: 769 = left (label 0), 770 = right (1)
CUE_ONSETS = [2.0, 8.0, 14.0]
CUE_CODES = ["769", "770", "769"]
EXPECTED_LABELS = [0, 1, 0]


def _synthetic_raw(sfreq=SFREQ, seconds=20):
    """A deterministic continuous Raw with cue annotations.

    Channel c, sample n holds c * 10000 + n, so any extracted trial can be
    checked against the source samples exactly.
    """
    import mne

    n_samples = int(seconds * sfreq)
    data = np.zeros((len(CH_NAMES), n_samples), dtype=np.float64)
    for c in range(len(CH_NAMES)):
        data[c] = c * 10000 + np.arange(n_samples)
    info = mne.create_info(CH_NAMES, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=CUE_ONSETS,
            duration=[0.0] * len(CUE_ONSETS),
            description=CUE_CODES,
        )
    )
    return raw, data


def _loader(window):
    # root is unused: epochs_from_raw never touches the filesystem
    return BCICompetitionIV2bLoader(root="unused", window=window)


@requires_mne()
def test_epochs_from_raw_shape_and_labels():
    raw, _ = _synthetic_raw()
    loader = _loader(EpochWindow(0.0, 4.0))
    bundle = loader.epochs_from_raw(raw, subject=1, session="01T")

    assert bundle.n_trials == 3
    assert bundle.n_channels == 3
    assert bundle.n_times == 1000  # 4 s * 250 Hz
    assert bundle.sfreq == SFREQ
    assert bundle.labels.tolist() == EXPECTED_LABELS


@requires_mne()
def test_epochs_from_raw_segments_match_source():
    raw, data = _synthetic_raw()
    # units="V" so no 1e6 scaling; the epoch equals the source samples exactly
    loader = _loader(EpochWindow(0.0, 4.0, units="V"))
    bundle = loader.epochs_from_raw(raw, subject=1, session="01T")

    cue0 = int(round(CUE_ONSETS[0] * SFREQ))  # tmin=0 starts at the cue sample
    expected = data[:, cue0:cue0 + 1000].astype(np.float32)
    np.testing.assert_allclose(bundle.data[0], expected, rtol=1e-4)


@requires_mne()
def test_resample_before_epoching_keeps_trials_correct():
    # The Option 1 property: a pipeline that resamples the continuous Raw must
    # still yield the right number of trials at the right times, because the cue
    # annotations move with the resample.
    raw, _ = _synthetic_raw()
    raw = raw.copy().resample(125.0, verbose="ERROR")

    loader = _loader(EpochWindow(0.0, 4.0, units="V"))
    bundle = loader.epochs_from_raw(raw, subject=1, session="01T")

    assert bundle.sfreq == 125.0
    assert bundle.n_times == 500  # 4 s * 125 Hz
    assert bundle.n_trials == 3
    assert bundle.labels.tolist() == EXPECTED_LABELS


@requires_mne()
def test_load_session_delegates_to_the_split(monkeypatch):
    # load_session should just be load_raw followed by epochs_from_raw
    raw, _ = _synthetic_raw()
    loader = _loader(EpochWindow(0.0, 4.0))
    monkeypatch.setattr(loader, "load_raw", lambda subject, session: raw)

    bundle = loader.load_session(subject=1, session="01T")
    assert bundle.n_trials == 3
    assert bundle.labels.tolist() == EXPECTED_LABELS
