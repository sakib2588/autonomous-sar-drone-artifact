#!/usr/bin/env python3
"""Desk-side sanity check: does capture -> inference -> draw -> record run
end to end against a live webcam, without crashing? Not an accuracy test --
see the caveat below.

Run from the repo root:

    .venv/bin/python scripts/webcam_smoke_test.py

IMPORTANT CAVEAT, read before judging the result: the model was trained on
VisDrone -- aerial, top-down, person side length 9-20 px at 640 tile scale
(see notes/20260805-permanent-phase-status-and-blockers.md Section 7.1). A
webcam framing a person a few feet away, straight-on, filling half the frame,
is a completely different domain -- a face-filling-the-frame shot is roughly
30-60x larger than anything labelled "person" in training. A miss here does
NOT mean the model is broken; it means this test is out-of-distribution by
design. What this test actually checks: does the capture -> preprocess ->
inference -> NMS -> draw -> RECORD pipeline run end to end without crashing,
and does the model output *anything* sane (plausible box placement, not NaN,
not a crashed process). For a meaningful accuracy check, the camera needs to
be far enough away / high enough up that the subject shrinks toward the
9-20px range the model was actually trained on -- see
notes/20260820-decision-pi4-field-deployment.md.

Records continuously to an .mp4, same shape as the real drone pipeline
(scripts/15_pi_live.py): record onboard only, no live stream out (per the
2026-08-16 mission-plan decision -- see project_sar_drone_hardware_procurement
memory) -- this script is the desk-side rehearsal of that same behaviour.
Inference runs on 1-of-N frames (SAMPLE_EVERY_S), same reason as the Pi
script: full-rate inference is slower than capture, so running it on every
frame would fall behind. The most recent detection boxes stay drawn on every
recorded frame between inferences, not just the sampled ones -- otherwise the
video would strobe the boxes on and off, which isn't what onboard recording
will actually look like.

One warm-up inference runs BEFORE the recording clock starts. The first
`model.predict()` call pays CUDA/cuDNN one-time setup cost (seconds, not
milliseconds) -- without discarding it, most of DURATION_S gets eaten by
warm-up and the clip ends up far shorter than requested. Same principle as
`src/sar/bench/pi_runner.py`'s warm-up discard, applied here for the same
reason.

Output goes to scripts/webcam_smoke_out/ (gitignored -- recordings are local
scratch, not a tracked artifact).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "webcam_smoke_out"
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ROOT / "checkpoints" / "y11n_tiled640_s42" / "weights" / "best.pt"
DURATION_S = 15
SAMPLE_EVERY_S = 1.0
STILL_EVERY_S = 2.0  # separate, sparser cadence for the inspect-without-a-player JPGs


def find_working_camera(max_index: int = 5) -> tuple[cv2.VideoCapture, int, int]:
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    # Measured 2026-08-23 on the Microdia Vitade AF: the V4L2 default queue is
    # 4 frames, and cap.read() pops the OLDEST. With capture at 24 FPS and the
    # read->infer->draw loop at ~6 FPS the queue sits permanently full, so every
    # frame reaching the model was 166 ms stale before inference even started.
    # Depth 1 measured the lag down to 42 ms at identical throughput (24.06 vs
    # 24.03 FPS). Must be set before the first read to take effect.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"[camera] opened index {idx}, frame shape {frame.shape}")
            return cap, w, h
        cap.release()
    raise RuntimeError(f"no working camera found in indices 0..{max_index - 1}")


def to_detections(result, model) -> list[dict]:
    dets = []
    for box in result.boxes:
        dets.append(
            {
                "class_name": model.names[int(box.cls[0])],
                "conf": round(float(box.conf[0]), 3),
                "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            }
        )
    return dets


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"[FAIL] model checkpoint not found at {MODEL_PATH}")
        print("       this repo's checkpoints/ is gitignored (large binaries) -- ")
        print("       copy checkpoints/y11n_tiled640_s42/weights/best.pt over from")
        print("       the machine that trained it before running this on a new laptop.")
        return 1

    from ultralytics import YOLO

    print(f"[model] loading {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"[model] classes: {model.names}")

    cap, w, h = find_working_camera()

    # Warm-up: pay the CUDA/cuDNN first-call cost now, before the clock starts
    # and before the video writer opens, so it doesn't eat the recording window.
    print("[warmup] running one throwaway inference (discarded)")
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, conf=0.15, verbose=False)

    video_path = OUT_DIR / "recording.mp4"
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), cap_fps, (w, h))

    summary = []
    t_start = time.time()
    next_infer = t_start
    next_still = t_start
    n_frames = 0
    n_infers = 0
    n_stills = 0
    last_dets: list[dict] = []

    print(f"[record] writing {video_path} at {cap_fps:.1f} fps for {DURATION_S}s")

    while time.time() - t_start < DURATION_S:
        ok, frame = cap.read()
        if not ok:
            print("[warn] frame read failed, skipping")
            continue
        now = time.time()
        n_frames += 1

        if now >= next_infer:
            next_infer = now + SAMPLE_EVERY_S
            result = model.predict(frame, imgsz=640, conf=0.15, verbose=False)[0]
            last_dets = to_detections(result, model)
            n_infers += 1
            summary.append({"infer_idx": n_infers, "t": round(now - t_start, 2), "detections": last_dets})
            labels = [f"{d['class_name']} {d['conf']:.2f}" for d in last_dets]
            print(f"[infer {n_infers}] {len(last_dets)} detection(s): {labels}")

        annotated = frame.copy()
        for d in last_dets:
            x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{d['class_name']} {d['conf']:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(annotated)

        if now >= next_still:
            next_still = now + STILL_EVERY_S
            n_stills += 1
            cv2.imwrite(str(OUT_DIR / f"frame_{n_stills:02d}.jpg"), annotated)

    cap.release()
    writer.release()

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    total_dets = sum(len(f["detections"]) for f in summary)
    print(f"\n[done] {n_frames} frames recorded, {n_infers} inference pass(es), {total_dets} total detections")
    print(f"[done] video: {video_path}")
    print(f"[done] {n_stills} still(s) + summary.json in {OUT_DIR}")
    if total_dets == 0:
        print("[note] zero detections is plausible here -- see the domain-mismatch")
        print("       caveat at the top of this file before treating it as a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
