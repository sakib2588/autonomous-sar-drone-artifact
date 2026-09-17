"""YOLO detect-head decode + NMS for the raw TFLite output (field deployment).

Why this exists. `13_pi_bench.py` deliberately stops at the raw model output --
its own docstring says NMS "is measured separately once the decoder is ported."
That is fine for a latency benchmark (the number is a documented lower bound)
but it means nothing in this repo turns the model's raw tensor into an actual
bounding box. A live onboard pipeline needs real boxes, so this module is that
missing decoder.

Verified against the actual exported graph, not assumed:
`checkpoints/y11n_tiled640_s42/weights/best_saved_model/best_full_integer_quant.tflite`
(the fastest artifact per GATE-4, R6/R7 -- this is the deployment choice) --

    input  [1 640 640 3] int8, quant (scale=0.003922, zero_point=-128)
    output [1 7 8400]    int8, quant (scale=0.004032, zero_point=-122)

7 = 4 box channels (cx, cy, w, h) + 3 class scores (already sigmoid-activated --
Ultralytics' Detect.forward applies sigmoid to cls before the export concat).
There is no separate objectness channel; YOLOv8/11 fold it into the per-class
score. Class order is `configs/sar_rgb_tiled.yaml`: 0 person, 1 vehicle,
2 two_wheeler.

**Box channels are normalised to [0, 1] as a fraction of the 640 input, NOT
pixels.** This is not documented anywhere in Ultralytics' export code and was
found empirically: the single int8 tensor covers box AND class channels under
one shared per-tensor scale/zero_point (scale=0.004032, zero_point=-122), and
that pair can only represent real values in roughly [-0.02, 1.0] -- there is no
representable value anywhere near 320 (half of 640). Confirmed by running the
actual graph (`best_full_integer_quant.tflite`) on a real tile: every box
channel's dequantised min/max sat inside [0, 1], never above it. `decode_yolo_output`
multiplies by `img_size` to recover pixels; get this wrong and every box lands
in the top-left 1x1 px corner while confidence scores stay perfectly plausible
-- silent, not a crash.

Boxes returned here are in 640x640 model-input space. The caller must map them
back to the original frame using the SAME letterbox parameters used for
preprocessing -- that mapping is deliberately not done in this module, so it
stays testable without an image.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES: tuple[str, ...] = ("person", "vehicle", "two_wheeler")


def dequantize(raw: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """int8 quantised tensor -> float32, per TFLite's affine scheme.

    real_value = (quantised_value - zero_point) * scale

    A float32 export is passed through unchanged. TFLite reports
    `quantization = (0.0, 0)` for a tensor that was never quantised, and applying
    the affine formula to that multiplies every value by zero -- the model then
    emits no detections at all, with no error and no warning, which is far worse
    than a crash because it looks exactly like a scene containing nobody. Guard
    on scale rather than on dtype so this holds for float16 exports too.
    """
    if scale == 0:
        return raw.astype(np.float32)
    return (raw.astype(np.float32) - zero_point) * scale


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2). Shape (N, 4) in, (N, 4) out."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.empty_like(boxes)
    out[:, 0] = cx - w / 2
    out[:, 1] = cy - h / 2
    out[:, 2] = cx + w / 2
    out[:, 3] = cy + h / 2
    return out


def _box_area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS. Returns indices to keep, highest score first.

    Class-agnostic by construction -- callers that need per-class NMS filter
    `boxes_xyxy`/`scores` to one class before calling this, same as every other
    NMS implementation (torchvision, Ultralytics) so the two are comparable.
    """
    if len(boxes_xyxy) == 0:
        return []
    order = scores.argsort()[::-1]
    areas = _box_area(boxes_xyxy)
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thres]
    return keep


def decode_yolo_output(
    raw: np.ndarray,
    scale: float,
    zero_point: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    class_names: tuple[str, ...] = CLASS_NAMES,
    img_size: int = 640,
) -> list[dict]:
    """Raw (1, 4+nc, 8400) int8 tensor -> a list of detections.

    Each detection: {class_id, class_name, conf, box_xyxy} with box_xyxy in
    `img_size`x`img_size` model-input PIXEL space (float, may be negative /
    past img_size before any clipping -- clip after mapping back to the real
    frame, not here). `img_size` must match the interpreter's input shape
    (640 for every artifact this repo exports); the raw box channels are
    normalised [0, 1] and are multiplied by it here.

    NMS runs per class, matching Ultralytics' default (`agnostic=False`):
    an overlapping person and vehicle box must both be allowed to survive.
    """
    if raw.ndim == 3:
        raw = raw[0]  # drop batch dim if the caller passed the interpreter's raw (1, 7, 8400) output
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T  # (7, 8400) -> (8400, 7); already-transposed input is a no-op here
    nc = len(class_names)
    if raw.shape[1] != 4 + nc:
        raise ValueError(f"expected {4 + nc} channels (4 box + {nc} classes), got {raw.shape[1]}")

    dequant = dequantize(raw, scale, zero_point)
    boxes_cxcywh = dequant[:, :4] * img_size  # normalised [0,1] -> pixels
    class_scores = dequant[:, 4:]

    class_id = class_scores.argmax(axis=1)
    conf = class_scores[np.arange(len(class_scores)), class_id]

    mask = conf >= conf_thres
    if not mask.any():
        return []

    boxes_xyxy = xywh2xyxy(boxes_cxcywh[mask])
    conf = conf[mask]
    class_id = class_id[mask]

    detections: list[dict] = []
    for c in np.unique(class_id):
        c_mask = class_id == c
        keep = nms(boxes_xyxy[c_mask], conf[c_mask], iou_thres)
        c_boxes = boxes_xyxy[c_mask][keep]
        c_conf = conf[c_mask][keep]
        for box, sc in zip(c_boxes, c_conf):
            detections.append(
                {
                    "class_id": int(c),
                    "class_name": class_names[int(c)],
                    "conf": round(float(sc), 5),
                    "box_xyxy": [round(float(v), 2) for v in box],
                }
            )
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections
