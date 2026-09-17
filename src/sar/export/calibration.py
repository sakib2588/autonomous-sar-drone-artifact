"""Guards that keep INT8 calibration honest and bounded.

Two Ultralytics 8.3.253 defaults make the naive call wrong, both read directly
from `ultralytics/engine/exporter.py` on 2026-08-07:

1. `get_int8_calibration_dataloader` selects `data[self.args.split or "val"]`.
   The default calibration split is val. Calibrating on val or test leaks the
   evaluation distribution into the quantisation scales, so the resulting
   accuracy is not an honest estimate of deployed performance.

2. `self.args.fraction` defaults to 1.0 and `export_saved_model` concatenates
   the whole split into one float32 tensor. On the 40,523-image tiled train
   split at 640 px that is a single 199.1 GB (185.5 GiB) allocation, on a host
   with 14.5 GiB of RAM.

Nothing here imports Ultralytics, so it is cheap to unit test.
"""

from __future__ import annotations

# Ultralytics warns below 300 calibration images. 500 gives headroom above that
# warning while keeping the float32 concat near 2.5 GB at 640 px, which fits
# alongside everything else on a 14.5 GiB host.
CALIB_TARGET_IMAGES = 500
MIN_RECOMMENDED_CALIB_IMAGES = 300

_BYTES_PER_FLOAT32 = 4
_CHANNELS = 3


def calibration_fraction(n_train_images: int, target: int = CALIB_TARGET_IMAGES) -> float:
    """Fraction of the train split that yields roughly `target` calibration images.

    Raises when the split cannot supply the minimum Ultralytics recommends,
    rather than silently calibrating on too few samples.
    """
    if n_train_images < MIN_RECOMMENDED_CALIB_IMAGES:
        raise ValueError(
            f"train split has {n_train_images} images; INT8 calibration needs at least "
            f"{MIN_RECOMMENDED_CALIB_IMAGES}"
        )
    if n_train_images <= target:
        return 1.0
    return target / n_train_images


def calibration_image_count(n_train_images: int, fraction: float) -> int:
    """Images Ultralytics will actually load for a given fraction."""
    return int(n_train_images * fraction)


def assert_calibration_split_is_train(split: str | None) -> None:
    """Reject any calibration split other than train.

    `None` is rejected too: Ultralytics resolves a missing split to "val", so
    leaving it unset is the leakage case, not a neutral default.
    """
    if split != "train":
        raise ValueError(
            f"INT8 calibration split is {split!r}; calibrating on anything but 'train' is "
            "leakage (Ultralytics defaults an unset split to 'val')"
        )
    return None


def calibration_memory_bytes(n_images: int, imgsz: int) -> int:
    """Size of the single float32 tensor `export_saved_model` builds."""
    return n_images * imgsz * imgsz * _CHANNELS * _BYTES_PER_FLOAT32
