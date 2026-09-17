import numpy as np
import pytest

from src.sar.bench.decode import (
    CLASS_NAMES,
    decode_yolo_output,
    dequantize,
    nms,
    xywh2xyxy,
)


# --- dequantize -------------------------------------------------------------

def test_dequantize_matches_tflite_affine_scheme():
    # Verified against the real graph's quantization tuple (see decode.py
    # docstring): scale=0.004032, zero_point=-122. A raw value equal to the
    # zero point must dequantize to exactly 0.
    raw = np.array([-122, -22, 5], dtype=np.int8)
    out = dequantize(raw, scale=0.004032, zero_point=-122)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(100 * 0.004032, abs=1e-6)
    assert out[2] == pytest.approx(127 * 0.004032, abs=1e-6)


# --- xywh2xyxy ---------------------------------------------------------------

def test_xywh2xyxy_centered_box():
    boxes = np.array([[100.0, 100.0, 40.0, 20.0]])
    out = xywh2xyxy(boxes)
    assert out[0].tolist() == pytest.approx([80.0, 90.0, 120.0, 110.0])


# --- nms ----------------------------------------------------------------------

def test_nms_drops_heavily_overlapping_lower_score_box():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # near-identical to box 0, lower score
            [100.0, 100.0, 110.0, 110.0],  # far away, must survive
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thres=0.5)
    assert keep == [0, 2]


def test_nms_keeps_non_overlapping_boxes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = np.array([0.6, 0.9])
    keep = nms(boxes, scores, iou_thres=0.5)
    # highest score first, both survive since they never overlap
    assert set(keep) == {0, 1}
    assert keep[0] == 1


def test_nms_empty_input_returns_empty_list():
    assert nms(np.empty((0, 4)), np.empty((0,)), iou_thres=0.5) == []


# --- decode_yolo_output -------------------------------------------------------

def _make_raw(
    detections: list[tuple[float, float, float, float, int, float]], scale: float, zero_point: int
) -> np.ndarray:
    """Build a synthetic (7, 8400) int8 tensor with the given detections planted
    at the front and everything else near-zero confidence, so decode has real
    anchors to reject.

    Box coords here are already normalised [0, 1] (fraction of the 640 input),
    matching what the real exported graph emits -- see the "Box channels are
    normalised" note in decode.py.
    """
    n = 8400
    real = np.zeros((n, 4 + len(CLASS_NAMES)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, conf) in enumerate(detections):
        real[i, 0:4] = [cx, cy, w, h]
        real[i, 4 + cls] = conf
    quantised = np.clip(np.round(real / scale) + zero_point, -128, 127).astype(np.int8)
    return quantised.T  # (8400, 7) -> (7, 8400), matching the real graph's layout


def test_decode_recovers_a_single_planted_detection():
    scale, zero_point = 0.004032, -122
    # cx=0.5, cy=0.5, w=0.15625, h=0.3125 -> in 640px: cx=320 cy=320 w=100 h=200
    raw = _make_raw([(0.5, 0.5, 0.15625, 0.3125, 0, 0.9)], scale, zero_point)
    dets = decode_yolo_output(raw, scale, zero_point, conf_thres=0.25, iou_thres=0.45)
    assert len(dets) == 1
    d = dets[0]
    assert d["class_name"] == "person"
    assert d["conf"] == pytest.approx(0.9, abs=0.05)
    x1, y1, x2, y2 = d["box_xyxy"]
    assert x1 == pytest.approx(270.0, abs=2.0)
    assert y1 == pytest.approx(220.0, abs=2.0)
    assert x2 == pytest.approx(370.0, abs=2.0)
    assert y2 == pytest.approx(420.0, abs=2.0)


def test_decode_below_threshold_conf_is_dropped():
    scale, zero_point = 0.004032, -122
    raw = _make_raw([(0.5, 0.5, 0.15625, 0.3125, 1, 0.1)], scale, zero_point)
    dets = decode_yolo_output(raw, scale, zero_point, conf_thres=0.25, iou_thres=0.45)
    assert dets == []


def test_decode_accepts_batch_dim_and_transposed_layout():
    scale, zero_point = 0.004032, -122
    raw = _make_raw([(0.3125, 0.3125, 0.078125, 0.078125, 2, 0.8)], scale, zero_point)
    batched = raw[None, ...]  # (1, 7, 8400), as the interpreter actually returns it
    dets = decode_yolo_output(batched, scale, zero_point, conf_thres=0.25, iou_thres=0.45)
    assert len(dets) == 1
    assert dets[0]["class_name"] == "two_wheeler"


def test_decode_runs_nms_per_class_independently():
    # Two overlapping boxes but DIFFERENT classes -- both must survive, because
    # NMS here is per-class, matching Ultralytics' agnostic=False default.
    scale, zero_point = 0.004032, -122
    raw = _make_raw(
        [
            (0.5, 0.5, 0.15625, 0.15625, 0, 0.9),  # person
            (0.503, 0.503, 0.15625, 0.15625, 1, 0.85),  # vehicle, near-identical box
        ],
        scale,
        zero_point,
    )
    dets = decode_yolo_output(raw, scale, zero_point, conf_thres=0.25, iou_thres=0.45)
    assert {d["class_name"] for d in dets} == {"person", "vehicle"}


def test_decode_rejects_wrong_channel_count():
    raw = np.zeros((6, 100), dtype=np.int8)  # missing one class channel
    with pytest.raises(ValueError, match="expected 7 channels"):
        decode_yolo_output(raw, 0.004, -120)
