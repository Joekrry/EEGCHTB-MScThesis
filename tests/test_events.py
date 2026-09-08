"""Pure-logic tests for event codes and label mappings."""

from __future__ import annotations

from eegchtb.data import events as ev


def test_2a_has_four_contiguous_classes():
    assert sorted(ev.BCI_IV_2A_CLASSES) == [0, 1, 2, 3]
    assert sorted(ev.BCI_IV_2A_CUE_TO_LABEL.values()) == [0, 1, 2, 3]


def test_2b_has_two_contiguous_classes():
    assert sorted(ev.BCI_IV_2B_CLASSES) == [0, 1]
    assert sorted(ev.BCI_IV_2B_CUE_TO_LABEL.values()) == [0, 1]


def test_cue_codes_map_to_named_classes():
    for code, label in ev.BCI_IV_2A_CUE_TO_LABEL.items():
        assert 769 <= code <= 772
        assert label in ev.BCI_IV_2A_CLASSES
    for code, label in ev.BCI_IV_2B_CUE_TO_LABEL.items():
        assert code in (769, 770)
        assert label in ev.BCI_IV_2B_CLASSES


def test_classlabel_is_converted_to_zero_based():
    assert ev.classlabel_to_label(1) == 0
    assert ev.classlabel_to_label(4) == 3


def test_known_class_order_matches_braindecode_convention():
    # left, right, feet, tongue in cue-code order
    assert ev.BCI_IV_2A_CUE_TO_LABEL[ev.EVENT_CUE_LEFT] == 0
    assert ev.BCI_IV_2A_CUE_TO_LABEL[ev.EVENT_CUE_RIGHT] == 1
    assert ev.BCI_IV_2A_CUE_TO_LABEL[ev.EVENT_CUE_FOOT] == 2
    assert ev.BCI_IV_2A_CUE_TO_LABEL[ev.EVENT_CUE_TONGUE] == 3
