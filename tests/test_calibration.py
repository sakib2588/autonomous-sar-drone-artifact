import pytest

from src.sar.export.calibration import (
    CALIB_TARGET_IMAGES,
    MIN_RECOMMENDED_CALIB_IMAGES,
    assert_calibration_split_is_train,
    calibration_fraction,
    calibration_image_count,
    calibration_memory_bytes,
)


def test_fraction_targets_five_hundred_images_on_the_tiled_train_split():
    # 40,523 tiled training images; fraction must land the calibration set near 500.
    f = calibration_fraction(40_523)
    assert calibration_image_count(40_523, f) == pytest.approx(CALIB_TARGET_IMAGES, abs=1)


def test_fraction_targets_five_hundred_images_on_the_full_train_split():
    f = calibration_fraction(6_471)
    assert calibration_image_count(6_471, f) == pytest.approx(CALIB_TARGET_IMAGES, abs=1)


def test_fraction_is_one_when_the_split_is_smaller_than_the_target():
    assert calibration_fraction(300) == 1.0


def test_fraction_never_yields_fewer_than_the_recommended_minimum():
    # Ultralytics warns below 300 images; a split that cannot reach it must raise
    # rather than quietly calibrate on too few samples.
    with pytest.raises(ValueError, match="at least"):
        calibration_fraction(120)


def test_val_split_is_rejected_as_leakage():
    with pytest.raises(ValueError, match="leakage"):
        assert_calibration_split_is_train("val")


def test_test_split_is_rejected_as_leakage():
    with pytest.raises(ValueError, match="leakage"):
        assert_calibration_split_is_train("test")


def test_none_is_rejected_because_ultralytics_defaults_it_to_val():
    with pytest.raises(ValueError, match="leakage"):
        assert_calibration_split_is_train(None)


def test_train_split_is_accepted():
    assert assert_calibration_split_is_train("train") is None


def test_memory_estimate_matches_the_float32_concat_size():
    # 500 images at 640px RGB float32 = 500*640*640*3*4 bytes
    assert calibration_memory_bytes(500, 640) == 500 * 640 * 640 * 3 * 4


def test_memory_estimate_flags_the_unbounded_default():
    # The failure mode this module exists to prevent: fraction=1.0 on the tiled
    # train split is 199.1 GB decimal / 185.5 GiB in a single allocation, on a
    # host with 14.5 GiB of RAM.
    assert calibration_memory_bytes(40_523, 640) == 40_523 * 640 * 640 * 3 * 4
    assert calibration_memory_bytes(40_523, 640) > 180 * 1024**3
