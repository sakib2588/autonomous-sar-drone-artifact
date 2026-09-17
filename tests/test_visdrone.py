"""Tests for the VisDrone-specific sequence key.

Ultralytics' VisDrone.yaml conversion flattens the original per-video
directories into one folder per split and renames frames to
"<videoid>_<frameoffset>_d_<globalindex>.jpg". splits.sequence_key's
parent-directory rule returns the split name itself for every file here,
which would silently disable leak detection.

IDs are also re-numbered from a small counter independently per split
(verified 2026-08-04: id "0000072" appears in both train and val, zero
overlapping image bytes) -- so the key must be namespaced by split too, or
two unrelated videos with a coincidentally-shared renumbered id would be
merged into one sequence.
"""

import pytest

from src.sar.data.visdrone import visdrone_sequence_key


def test_extracts_video_id_and_namespaces_by_split():
    key = visdrone_sequence_key("data/raw/VisDrone/images/train/0000002_00005_d_0000014.jpg")
    assert key == "train/0000002"


def test_different_frame_offsets_of_the_same_video_share_the_key():
    a = visdrone_sequence_key("data/raw/VisDrone/images/train/0000008_00889_d_0000039.jpg")
    b = visdrone_sequence_key("data/raw/VisDrone/images/train/0000008_01999_d_0000040.jpg")
    assert a == b == "train/0000008"


def test_same_video_id_in_different_splits_is_not_the_same_key():
    train_key = visdrone_sequence_key("data/raw/VisDrone/images/train/0000072_00100_d_0000001.jpg")
    val_key = visdrone_sequence_key("data/raw/VisDrone/images/val/0000072_00200_d_0000002.jpg")
    assert train_key != val_key


def test_rejects_unexpected_filename_pattern():
    with pytest.raises(ValueError):
        visdrone_sequence_key("data/raw/VisDrone/images/train/not_a_visdrone_name.jpg")
