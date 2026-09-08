"""BCI Competition IV 2a loader: 9 subjects, 22 EEG (+3 EOG dropped), 4 classes, 250 Hz, 288 trials/session (72/class); T sessions labelled in the GDF, E sessions' labels in the .mat."""

from __future__ import annotations

from . import events as ev
from .gdf import GdfMotorImageryLoader

# Reference:
#   BCI Competition IV - Data set 2a description (Graz BCI Lab, TU Graz)
#   https://www.bbci.de/competition/iv/desc_2a.pdf
#   (Used for: 9 subjects, 22 EEG + 3 EOG channels, 4 classes, 250 Hz,
#    288 trials per session at 72 per class, the T/E session split, the GDF
#    event codes, and the separately released evaluation true-label .mat files.)

# Reference:
#   BCI Competition IV-2a electrode montage, as named in braindecode / MOABB.
#   (Used for: mapping the 22 EEG channels of the GDF, in file order, to their
#    10-20 names. The GDF stores most as numbered placeholders (EEG-0..EEG-16),
#    so a standard_1020 montage and the montage-dependent spatial steps (the
#    Hjorth Laplacian) are only well defined once they carry these names. File
#    order confirmed against the named anchors Fz, C3, Cz, C4, Pz.)
BCICIV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


class BCICompetitionIV2aLoader(GdfMotorImageryLoader):
    """IV-2a: the file layout, reading and epoching all come from GdfMotorImageryLoader."""

    name = "bci_iv_2a"
    expected_sfreq = 250.0
    cue_to_label = ev.BCI_IV_2A_CUE_TO_LABEL
    class_names = ev.BCI_IV_2A_CLASSES
    eeg_channel_names = BCICIV_2A_CHANNELS
    archive_dir_name = "BCICIV_2a_gdf"
    subject_prefix = "A"

    N_EEG_CHANNELS = 22
    SUBJECTS = tuple(range(1, 10))
    SESSIONS = ("T", "E")
    # published trials per session (288 cues; 72/class for training)
    EXPECTED_TRIALS = {"T": 288, "E": 288}
