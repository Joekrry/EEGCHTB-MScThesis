"""Integration tests against real BCI IV GDF files; skipped unless BCI_IV_2A_DIR/BCI_IV_2B_DIR are set and MNE is installed."""

from __future__ import annotations

import pytest

from conftest import requires_mne

from eegchtb.data import (
    BCICompetitionIV2aLoader,
    BCICompetitionIV2bLoader,
    EpochWindow,
)

WINDOW = EpochWindow(tmin=0.0, tmax=4.0)  # 4 s -> 1000 samples at 250 Hz


# BCI IV-2a
@requires_mne()
def test_2a_training_session_counts_and_balance(bci_iv_2a_dir):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW)
    bundle = loader.load_session(subject=1, session="T")
    assert bundle.n_trials == 288
    # restricted by default to the Galea intersection of the 22-channel montage
    assert bundle.channel_names == ["C3", "Cz", "C4", "Pz"]
    assert bundle.n_times == 1000
    assert bundle.sfreq == 250.0
    # training session: 72 trials per class
    assert bundle.class_distribution() == {0: 72, 1: 72, 2: 72, 3: 72}


@requires_mne()
def test_2a_channels_none_keeps_all_22(bci_iv_2a_dir):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW, channels=None)
    bundle = loader.load_session(subject=1, session="T")
    assert bundle.n_channels == BCICompetitionIV2aLoader.N_EEG_CHANNELS  # 22


@requires_mne()
def test_2a_evaluation_labels_from_mat(bci_iv_2a_dir):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW)
    if "E" not in loader.available_sessions(1):
        pytest.skip("A01E.gdf not present")
    bundle = loader.load_session(subject=1, session="E")
    assert bundle.n_trials == 288
    # if the A01E.mat true labels are present, they should be balanced 72/class
    if not bundle.has_unknown_labels:
        assert bundle.class_distribution() == {0: 72, 1: 72, 2: 72, 3: 72}


@requires_mne()
def test_2a_all_available_training_sessions_have_288_trials(bci_iv_2a_dir):
    loader = BCICompetitionIV2aLoader(bci_iv_2a_dir, WINDOW)
    subjects = loader.available_subjects()
    assert subjects, "no IV-2a subjects found in data dir"
    for subject in subjects:
        if "T" in loader.available_sessions(subject):
            bundle = loader.load_session(subject, "T")
            assert bundle.n_trials == 288, f"subject {subject}"


# BCI IV-2b
@requires_mne()
def test_2b_channels_and_sampling_rate(bci_iv_2b_dir):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW)
    sessions = loader.available_sessions(1)
    assert sessions, "no IV-2b sessions found for subject 1"
    bundle = loader.load_session(subject=1, session=sessions[0])
    # all three 2b channels are Galea channels, so the default restriction is a no-op
    assert bundle.n_channels == BCICompetitionIV2bLoader.N_EEG_CHANNELS  # 3
    assert bundle.channel_names == ["C3", "Cz", "C4"]
    assert bundle.n_times == 1000
    assert bundle.sfreq == 250.0


@requires_mne()
def test_2b_two_class_problem(bci_iv_2b_dir):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW)
    sessions = loader.available_sessions(1)
    # training session (labels in GDF) should be a two-class problem
    train = [s for s in sessions if s.endswith("T")]
    if not train:
        pytest.skip("no IV-2b training sessions present")
    bundle = loader.load_session(subject=1, session=train[0])
    labels = set(bundle.filter_known().labels.tolist())
    assert labels.issubset({0, 1})


@requires_mne()
def test_2b_session_trial_counts_match_reference(bci_iv_2b_dir):
    loader = BCICompetitionIV2bLoader(bci_iv_2b_dir, WINDOW)
    for session, expected in BCICompetitionIV2bLoader.EXPECTED_TRIALS.items():
        if session in loader.available_sessions(1):
            bundle = loader.load_session(1, session)
            assert bundle.n_trials == expected, f"session {session}"
