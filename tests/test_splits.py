"""Tests for sequence-safe dataset splitting.

VisDrone is video-derived: consecutive frames within a sequence are near-identical.
A frame-level split places near-duplicates on both sides of the train/test boundary
and inflates mAP by a large, silent margin. Every split in this project must be
grouped by sequence, and `assert_no_sequence_leak` is the gate that proves it.
"""

import pytest

from src.sar.data.splits import (
    sequence_key,
    sequence_split,
    assert_no_sequence_leak,
)


# --- sequence_key -------------------------------------------------------------


def test_sequence_key_is_the_parent_directory_for_visdrone_layout():
    assert sequence_key("uav0000013_00000_v/0000001.jpg") == "uav0000013_00000_v"


def test_sequence_key_uses_the_immediate_parent_for_nested_paths():
    path = "data/raw/VisDrone/images/uav0000077_00720_v/0000123.jpg"
    assert sequence_key(path) == "uav0000077_00720_v"


def test_sequence_key_falls_back_to_the_stem_for_flat_unsequenced_images():
    # HERIDAL-style flat directories have no sequence structure; each image is
    # its own group, which is the conservative choice.
    assert sequence_key("IMG_4821.jpg") == "IMG_4821"


# --- sequence_split -----------------------------------------------------------


def _synthetic_frames(n_sequences=10, frames_per_sequence=50):
    return [
        f"uav0000{s:03d}_00000_v/{f:07d}.jpg"
        for s in range(n_sequences)
        for f in range(frames_per_sequence)
    ]


def test_no_frame_from_same_sequence_appears_in_two_splits():
    files = _synthetic_frames()
    tr, va, te = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    assert_no_sequence_leak(tr, va, te)


def test_split_is_lossless_every_frame_lands_somewhere_exactly_once():
    files = _synthetic_frames()
    tr, va, te = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    assert len(tr) + len(va) + len(te) == len(files)
    assert set(tr) | set(va) | set(te) == set(files)


def test_split_is_deterministic_under_a_fixed_seed():
    files = _synthetic_frames()
    a = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    b = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    assert a == b


def test_different_seeds_produce_different_partitions():
    files = _synthetic_frames(n_sequences=40)
    a = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    b = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=1337)
    assert a != b


def test_input_order_does_not_change_the_partition():
    # Guards against an accidental dependency on filesystem listing order,
    # which would make the split irreproducible on another machine.
    files = _synthetic_frames()
    a = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    b = sequence_split(list(reversed(files)), ratios=(0.7, 0.15, 0.15), seed=42)
    assert [sorted(s) for s in a] == [sorted(s) for s in b]


def test_ratios_are_approximately_honoured_at_sequence_granularity():
    files = _synthetic_frames(n_sequences=100, frames_per_sequence=10)
    tr, va, te = sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)
    # Sequences are equal-length here, so frame ratios track sequence ratios.
    assert 0.60 <= len(tr) / len(files) <= 0.80
    assert 0.08 <= len(va) / len(files) <= 0.22
    assert 0.08 <= len(te) / len(files) <= 0.22


def test_every_split_is_non_empty_when_there_are_enough_sequences():
    files = _synthetic_frames(n_sequences=20)
    for part in sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42):
        assert len(part) > 0


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        sequence_split(_synthetic_frames(), ratios=(0.7, 0.15, 0.30), seed=42)


def test_too_few_sequences_to_fill_every_split_is_an_error_not_a_silent_empty_split():
    # Two sequences cannot populate three splits. Failing loudly here prevents a
    # silently empty test set, which would look like a passing pipeline.
    files = _synthetic_frames(n_sequences=2)
    with pytest.raises(ValueError):
        sequence_split(files, ratios=(0.7, 0.15, 0.15), seed=42)


# --- assert_no_sequence_leak --------------------------------------------------


def test_leak_is_detected_when_present():
    tr = ["uav0000001_00000_v/0000001.jpg"]
    va = ["uav0000001_00000_v/0000002.jpg"]  # same sequence, different frame
    with pytest.raises(AssertionError):
        assert_no_sequence_leak(tr, va, [])


def test_leak_between_train_and_test_is_detected():
    tr = ["uav0000001_00000_v/0000001.jpg"]
    te = ["uav0000001_00000_v/0000009.jpg"]
    with pytest.raises(AssertionError):
        assert_no_sequence_leak(tr, [], te)


def test_leak_between_val_and_test_is_detected():
    va = ["uav0000005_00000_v/0000001.jpg"]
    te = ["uav0000005_00000_v/0000002.jpg"]
    with pytest.raises(AssertionError):
        assert_no_sequence_leak([], va, te)


def test_the_error_message_names_the_offending_sequence():
    tr = ["uav0000042_00000_v/0000001.jpg"]
    va = ["uav0000042_00000_v/0000002.jpg"]
    with pytest.raises(AssertionError, match="uav0000042_00000_v"):
        assert_no_sequence_leak(tr, va, [])


def test_disjoint_sequences_pass_cleanly():
    tr = ["uav0000001_00000_v/0000001.jpg"]
    va = ["uav0000002_00000_v/0000001.jpg"]
    te = ["uav0000003_00000_v/0000001.jpg"]
    assert_no_sequence_leak(tr, va, te)  # must not raise
