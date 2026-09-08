"""GDF event codes and 0-based class-label maps for BCI Competition IV 2a/2b."""

from __future__ import annotations

# Reference:
#   BCI Competition IV data set 2a / 2b descriptions (Graz BCI Lab, TU Graz)
#   https://www.bbci.de/competition/iv/desc_2a.pdf
#   https://www.bbci.de/competition/iv/desc_2b.pdf
#   (Used for: the GDF event-type codes below, the cue-code to class mapping for
#    each dataset, and the 1-based class labels of the evaluation .mat files.)

# GDF event-type codes (shared across IV-2a and IV-2b unless noted)
EVENT_IDLE_EYES_OPEN = 276
EVENT_IDLE_EYES_CLOSED = 277
EVENT_TRIAL_START = 768
EVENT_CUE_LEFT = 769      # class: left hand
EVENT_CUE_RIGHT = 770     # class: right hand
EVENT_CUE_FOOT = 771      # class: feet (IV-2a only)
EVENT_CUE_TONGUE = 772    # class: tongue (IV-2a only)
EVENT_CUE_UNKNOWN = 783   # cue with hidden label (evaluation sessions)
EVENT_REJECTED_TRIAL = 1023
EVENT_EYE_MOVEMENTS = 1072        # IV-2a
EVENT_FEEDBACK_CONTINUOUS = 781   # IV-2b feedback period
EVENT_NEW_RUN = 32766

# IV-2b EOG calibration markers
EVENT_HORIZONTAL_EYE_MOVEMENT = 1077
EVENT_VERTICAL_EYE_MOVEMENT = 1078
EVENT_EYE_ROTATION = 1079
EVENT_EYE_BLINKS = 1081

# Class definitions
# label -> class name
BCI_IV_2A_CLASSES: dict[int, str] = {
    0: "left_hand",
    1: "right_hand",
    2: "feet",
    3: "tongue",
}

# GDF cue code -> 0-based class label
BCI_IV_2A_CUE_TO_LABEL: dict[int, int] = {
    EVENT_CUE_LEFT: 0,
    EVENT_CUE_RIGHT: 1,
    EVENT_CUE_FOOT: 2,
    EVENT_CUE_TONGUE: 3,
}

BCI_IV_2B_CLASSES: dict[int, str] = {
    0: "left_hand",
    1: "right_hand",
}

BCI_IV_2B_CUE_TO_LABEL: dict[int, int] = {
    EVENT_CUE_LEFT: 0,
    EVENT_CUE_RIGHT: 1,
}

# Galea marker codes, as written by the recording protocol
GALEA_MARKER_TONGUE = 5
GALEA_MARKER_LEFT_HAND = 6
GALEA_MARKER_RIGHT_HAND = 7
GALEA_MARKER_FEET = 8

# Galea markers share IV-2a's label numbering so the two are directly comparable
GALEA_CUE_TO_LABEL: dict[int, int] = {
    GALEA_MARKER_LEFT_HAND: 0,
    GALEA_MARKER_RIGHT_HAND: 1,
    GALEA_MARKER_FEET: 2,
    GALEA_MARKER_TONGUE: 3,
}

# label for trials whose class is unknown; the single definition for the project
UNKNOWN_LABEL = -1


def classlabel_to_label(classlabel: int) -> int:
    """Convert a 1-based competition .mat class label to a 0-based label."""
    return int(classlabel) - 1
