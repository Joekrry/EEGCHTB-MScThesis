"""BCI Competition IV 2b loader: 9 subjects, 3 EEG (C3, Cz, C4; +3 EOG dropped), 2 classes, 250 Hz, 5 sessions/subject (3 training in the GDF, 2 evaluation labelled in the .mat; 720 trials/subject)."""

from __future__ import annotations

from . import events as ev
from .gdf import GdfMotorImageryLoader

# Reference:
#   BCI Competition IV - Data set 2b description (Graz BCI Lab, TU Graz)
#   https://www.bbci.de/competition/iv/desc_2b.pdf
#   (Used for: 9 subjects, 3 EEG (C3, Cz, C4) + 3 EOG channels, 2 classes,
#    250 Hz, the five-session layout (3 training + 2 evaluation), the per-session
#    trial counts, the GDF event codes, and the evaluation true-label .mat files.)


class BCICompetitionIV2bLoader(GdfMotorImageryLoader):
    """IV-2b: the file layout, reading and epoching all come from GdfMotorImageryLoader."""

    name = "bci_iv_2b"
    expected_sfreq = 250.0
    cue_to_label = ev.BCI_IV_2B_CUE_TO_LABEL
    class_names = ev.BCI_IV_2B_CLASSES
    archive_dir_name = "BCICIV_2b_gdf"
    subject_prefix = "B"

    N_EEG_CHANNELS = 3
    SUBJECTS = tuple(range(1, 10))
    # session suffix appended to B0{subject} -> e.g. B0101T.gdf
    SESSIONS = ("01T", "02T", "03T", "04E", "05E")
    # published trials per session, checked against the downloads by
    # tests/test_bci_loaders.py::test_2b_session_trial_counts_match_reference
    EXPECTED_TRIALS = {"01T": 120, "02T": 120, "03T": 160, "04E": 160, "05E": 160}
