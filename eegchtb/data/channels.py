"""The Galea headset EEG channel set the whole project is restricted to, and name-based selection against it."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Reference:
#   OpenBCI Galea headset 10-20 electrode layout
#   (Used for: the 10 EEG positions every dataset in this project is restricted
#    to, so results extrapolate to the Galea headset.)
GALEA_EEG_CHANNELS: list[str] = [
    "F1", "F2", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
]

# The Galea positions that BCI Competition IV-2a also records; the intersection
# is thin, which is what makes the montage effect measurable.
GALEA_BCI_IV_2A_OVERLAP_CHANNELS: list[str] = ["C3", "Cz", "C4", "Pz"]


def select_channels(
    available: Sequence[str], wanted: Sequence[str] = GALEA_EEG_CHANNELS
) -> list[str]:
    """Names in available that match wanted, case-insensitively, in wanted order."""
    by_lower: dict[str, str] = {}
    for name in available:
        key = name.lower()
        if key not in by_lower:
            by_lower[key] = name

    selected: list[str] = []
    for w in wanted:
        key = w.lower()
        if key in by_lower:
            selected.append(by_lower[key])
    # an empty intersection means the montage does not overlap the requested set at
    # all; failing loudly here stops a run silently training on zero channels
    if not selected:
        raise ValueError(
            f"none of the requested channels {list(wanted)} are present in "
            f"{list(available)}"
        )
    return selected


def channel_positions(
    names: Sequence[str], *, montage: str = "standard_1020"
) -> np.ndarray:
    """2-D projected, normalised (n, 2) electrode coordinates for names, from MNE's 10-20 template (x = right, y = anterior; centroid-centred, unit max radius),
    used to seed the GRF-Net spatial conv; raises if a name is absent."""
    import mne

    template = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    by_lower = {name.lower(): name for name in template}
    missing = [name for name in names if name.lower() not in by_lower]
    if missing:
        raise ValueError(f"channels absent from the {montage} template: {missing}")
    coords = np.array(
        [template[by_lower[name.lower()]][:2] for name in names], dtype=np.float64
    )

    # centre on the montage centroid, then scale so the outermost electrode = 1
    coords -= coords.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(coords, axis=1).max())
    if radius > 0.0:
        coords /= radius
    return coords
