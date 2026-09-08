"""Unit tests for the Galea channel set and name-based selection (no MNE/torch)."""

from __future__ import annotations

import pytest

from eegchtb.data import GALEA_EEG_CHANNELS
from eegchtb.data.channels import select_channels


def test_galea_set_is_the_ten_yellow_channels():
    assert GALEA_EEG_CHANNELS == [
        "F1", "F2", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
    ]


def test_selection_keeps_only_the_intersection_in_wanted_order():
    # BCI IV-2a montage order; the Galea intersection is C3, Cz, C4, Pz
    available = ["Fz", "FC3", "C5", "C3", "Cz", "C4", "CPz", "P1", "Pz", "POz"]
    assert select_channels(available) == ["C3", "Cz", "C4", "Pz"]


def test_selection_orders_by_wanted_not_available():
    available = ["Pz", "C4", "Cz", "C3"]  # scrambled relative to the Galea order
    assert select_channels(available) == ["C3", "Cz", "C4", "Pz"]


def test_selection_is_case_insensitive_and_returns_available_spelling():
    available = ["cz", "C3", "c4", "PZ"]
    assert select_channels(available) == ["C3", "cz", "c4", "PZ"]


def test_selection_honours_a_custom_wanted_list_in_wanted_order():
    available = ["C3", "Cz", "C4", "Pz"]
    assert select_channels(available, ["Cz", "C3"]) == ["Cz", "C3"]


def test_empty_intersection_raises():
    with pytest.raises(ValueError, match="none of the requested channels"):
        select_channels(["Fp1", "Fp2", "Oz"])
