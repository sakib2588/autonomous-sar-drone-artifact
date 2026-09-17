#!/usr/bin/env python3
"""Live webcam viewer with YOLO11n detection and ByteTrack person tracking.

Run from the repo root:

    .venv/bin/python scripts/webcam_live_detect.py

This is a desk rig, not the SAR flight pipeline (contrast with
scripts/15_pi_live.py). It uses stock YOLO11n (COCO-pretrained), which is the
correct model for a webcam pointed at a person up close -- unlike
scripts/webcam_smoke_test.py, which runs this project's aerial VisDrone
checkpoint and is expected to miss on a close-range face. Weights auto-download
on first run via ultralytics (~5-6 MB) to ~/.cache/ultralytics/.

Controls: 'q' or Esc to quit, 't' to toggle tracking on/off live.

WHY THIS TOOL CHANGED (2026-08-23)
----------------------------------
It could not hold a moving person. Three causes were measured, not guessed
(scripts/webcam_diag_out/diag_roomlight.json):

1. Stale frames. CAP_PROP_BUFFERSIZE defaults to 4 on V4L2 and cap.read() pops
   the OLDEST queued frame. Capture ran at 24 FPS against a ~6 FPS inference
   loop, so the queue sat permanently full and every frame reaching the model
   was 166 ms old before inference even began. Depth 1 cut that to 42 ms at
   identical throughput (24.06 vs 24.03 FPS).

2. The confidence threshold. Against their own stationary baseline, a moving
   person lost 28 percentage points of frames at conf 0.35 -- and 15 of those
   points were sitting between 0.15 and 0.35: detected, then thrown away. The
   default is now 0.18. This trades precision for recall deliberately. In a
   search-and-rescue setting a human operator dismisses a false alarm in a
   second, while a missed survivor is final, so the asymmetry runs one way.

3. Motion blur. The remaining 5 points produced no box at all, even at conf
   0.05. No threshold recovers those; only image quality or time does. Hence
   tracking, below. (On this Microdia 0c45:6366 the exposure control is a
   verified no-op, so scene light is the only lever on blur -- see the
   sar-drone-camera-exposure-requirement memory.)

Tracking answers cause 3. ByteTrack's premise is a second association pass:
high-confidence boxes claim their tracks first, then leftover tracks are matched
against the low-confidence boxes a plain detector discards. When even that finds
nothing, a lost track stays alive and Kalman-predicted for track_buffer frames,
and this script draws that prediction as a thin COAST box. A coasted box is
never claimed as a detection -- it is drawn thin and labelled, because the model
did not actually see anything on that frame.

Thresholds live in configs/bytetrack_sar.yaml with the measurement behind each.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKER_CFG = ROOT / "configs" / "bytetrack_sar.yaml"
# Fixed name, not a random one. A random temp file per run leaks on SIGTERM,
# because Python skips `finally` when the default signal handler fires -- caught
# by watching the temp dir grow across killed runs. A fixed name is overwritten
# instead of accumulating, so the worst case is one stale file.
ACTIVE_TRACKER_CFG = Path(tempfile.gettempdir()) / "bytetrack_sar_active.yaml"

CONFIRMED = (0, 255, 0)
COASTED = (0, 200, 255)


def open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    # Depth 1, set before the first read. See cause 1 in the module docstring:
    # the V4L2 default of 4 was costing 124 ms of pure lag for free.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def find_working_camera(max_index: int = 5) -> tuple[cv2.VideoCapture, int, int]:
    for idx in range(max_index):
        cap = open_camera(idx)
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


def tracker_cfg_with(overrides: dict) -> str:
    """Materialise configs/bytetrack_sar.yaml with CLI overrides applied.

    Ultralytics only accepts a tracker as a yaml path, so a changed --conf-thres
    has to be written to a file for track_high_thresh to follow it. Without this
    the displayed threshold and the tracker's first-stage threshold silently
    drift apart, which is the kind of mismatch that takes an hour to find.
    """
    import yaml

    cfg = yaml.safe_load(DEFAULT_TRACKER_CFG.read_text())
    if all(cfg.get(k) == v for k, v in overrides.items()):
        return str(DEFAULT_TRACKER_CFG)  # defaults unchanged: no generated file at all
    cfg.update(overrides)
    ACTIVE_TRACKER_CFG.write_text(yaml.safe_dump(cfg))
    return str(ACTIVE_TRACKER_CFG)


def cleanup_tracker_cfg(path: str) -> None:
    """Delete a GENERATED tracker config, never the tracked one in configs/.

    tracker_cfg_with returns the repo config unchanged when no override is
    needed, so an unguarded unlink here would delete a version-controlled file
    on every default run.
    """
    target = Path(path)
    if target.resolve() == DEFAULT_TRACKER_CFG.resolve():
        return
    target.unlink(missing_ok=True)


def coasted_tracks(model, max_age: int) -> list[tuple[int, list[float], int]]:
    """Kalman-predicted boxes for tracks the detector lost this frame.

    Pulled from the tracker's own lost_stracks rather than by holding the last
    seen box, so a moving subject's coasted box keeps moving with their
    estimated velocity instead of freezing where they were last seen.
    """
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor else None
    if not trackers:
        return []
    tr = trackers[0]
    out = []
    for t in getattr(tr, "lost_stracks", []):
        age = tr.frame_id - t.end_frame
        if 0 < age <= max_age:
            out.append((int(t.track_id), [float(v) for v in t.xyxy], age))
    return out


def draw(frame, boxes: list, coasts: list, names: dict) -> None:
    for xyxy, conf, cls, tid in boxes:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CONFIRMED, 2)
        tag = f"#{tid} " if tid is not None else ""
        cv2.putText(frame, f"{tag}{names.get(cls, cls)} {conf:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CONFIRMED, 2)
    for tid, xyxy, age in coasts:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COASTED, 1)
        cv2.putText(frame, f"#{tid} COAST {age}f", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COASTED, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None, help="v4l2 index; auto-detects if omitted")
    ap.add_argument("--source", default=None,
                    help="MJPEG stream URL instead of a local camera, e.g. "
                         "http://192.168.1.105:8080/stream.mjpg from 20_pi_stream_preview.py. Lets the "
                         "Pi's camera feed a model the Pi cannot run itself (no torch on a Pi 4). "
                         "Inference timing shown is THIS machine's, not the Pi's")
    ap.add_argument("--model", default="yolo11n.pt", help="ultralytics model name or path")
    ap.add_argument("--conf-thres", type=float, default=0.18,
                    help="confirmed-detection threshold. Was 0.35; lowered on measurement -- 15%% of moving "
                         "frames sat between 0.15 and 0.35 and were being discarded")
    ap.add_argument("--low-conf", type=float, default=0.08,
                    help="detector floor fed to the tracker. Boxes between this and --conf-thres never draw "
                         "on their own; they exist so ByteTrack's second pass can continue an existing track")
    ap.add_argument("--imgsz", type=int, default=448,
                    help="448 measured ~6 FPS vs ~2.6 FPS at 640 on this CPU, no accuracy loss for a close subject")
    ap.add_argument("--device", default="cpu", help="torch device; this laptop's Pascal GPU can't run current CUDA wheels")
    ap.add_argument("--no-track", action="store_true", help="plain per-frame detection, no ByteTrack. For A/B only")
    ap.add_argument("--all-classes", action="store_true",
                    help="detect all 80 COCO classes. Default is person-only: this is a SAR rig, and feeding "
                         "low-confidence boxes to the tracker across 80 classes invites phantom tracks")
    ap.add_argument("--coast-frames", type=int, default=None,
                    help="how long to draw a lost track's predicted box. Defaults to the config's track_buffer")
    cli = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    base_cfg = yaml.safe_load(DEFAULT_TRACKER_CFG.read_text())
    coast_frames = cli.coast_frames if cli.coast_frames is not None else int(base_cfg["track_buffer"])
    tracker_path = tracker_cfg_with({"track_high_thresh": cli.conf_thres, "track_low_thresh": cli.low_conf})

    print(f"[model] loading {cli.model} (auto-downloads on first run)")
    model = YOLO(cli.model)
    classes = None if cli.all_classes else [0]
    print(f"[model] {len(model.names)} classes available; detecting "
          f"{'all' if classes is None else 'person only'}")
    print(f"[thresh] confirmed >= {cli.conf_thres}, tracker second-pass floor {cli.low_conf}, "
          f"coast up to {coast_frames} frames")

    if cli.source is not None:
        # A network stream, not V4L2: CAP_PROP_BUFFERSIZE does not apply and the
        # backend is FFMPEG. Latency here is the Pi's capture loop plus network,
        # on top of local inference -- fine for watching, not a latency measurement.
        print(f"[source] opening stream {cli.source}")
        cap = cv2.VideoCapture(cli.source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open stream {cli.source} -- is 20_pi_stream_preview.py running?")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"opened {cli.source} but read no frame")
        h, w = frame.shape[:2]
        print(f"[source] {w}x{h}")
    elif cli.camera is not None:
        cap = open_camera(cli.camera)
        if not cap.isOpened():
            raise RuntimeError(f"could not open /dev/video{cli.camera}")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"opened /dev/video{cli.camera} but couldn't read a frame")
        h, w = frame.shape[:2]
    else:
        cap, w, h = find_working_camera()

    # Warm-up: the first call pays one-time model-graph setup, seconds not
    # milliseconds. Doing it here keeps the opening frames from stalling.
    print("[warmup] running one throwaway inference")
    import numpy as np

    model.predict(np.zeros((h, w, 3), dtype=np.uint8), imgsz=cli.imgsz, conf=cli.low_conf,
                  device=cli.device, verbose=False)

    window = "YOLO11n + ByteTrack (q quit, t toggle tracking)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    tracking = not cli.no_track
    print(f"[run] tracking {'ON' if tracking else 'OFF'}; press 'q' or Esc to quit, 't' to toggle")
    fps_hist: deque[float] = deque(maxlen=15)
    t_prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] frame read failed, skipping")
                continue

            if tracking:
                result = model.track(frame, persist=True, tracker=tracker_path, imgsz=cli.imgsz,
                                     conf=cli.low_conf, classes=classes, device=cli.device, verbose=False)[0]
            else:
                result = model.predict(frame, imgsz=cli.imgsz, conf=cli.low_conf, classes=classes,
                                       device=cli.device, verbose=False)[0]

            boxes = []
            for b in result.boxes:
                conf = float(b.conf[0])
                if conf < cli.conf_thres:
                    continue  # below the confirmed bar; it exists only to feed the tracker
                tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
                boxes.append((b.xyxy[0].tolist(), conf, int(b.cls[0]), tid))
            coasts = coasted_tracks(model, coast_frames) if tracking else []

            annotated = frame.copy()
            draw(annotated, boxes, coasts, model.names)

            now = time.time()
            fps_hist.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            hud = (f"{statistics.median(fps_hist):.1f} FPS | {len(boxes)} confirmed | "
                   f"{len(coasts)} coasting | track {'ON' if tracking else 'OFF'} | conf>={cli.conf_thres}")
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(annotated, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                tracking = not tracking
                print(f"[toggle] tracking {'ON' if tracking else 'OFF'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cleanup_tracker_cfg(tracker_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
