#!/usr/bin/env python3
"""Live on-screen preview: webcam -> tflite inference -> ffplay window on :0.

Ad-hoc demo tool, not part of the field pipeline. opencv-python-headless (the
deployed venv) has no GUI backend, so this pipes raw annotated BGR frames into
ffplay (already present on the Pi) instead of using cv2.imshow. Draws the most
recent detections every frame between inferences so the box doesn't disappear.

Also writes a detections.jsonl next to the field pipeline's own format (see
15_pi_live.py) -- the first version of this script only displayed live and
kept nothing, so a run that was watched by eye left no record to go back and
check confidences/classes against afterward.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode import decode_yolo_output  # noqa: E402


def preprocess(frame, size, scale, zero_point, dtype):
    import cv2
    import numpy as np

    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dtype == np.uint8:
        return img[None, ...]
    if dtype == np.int8:
        q = np.round((img.astype(np.float32) / 255.0) / scale) + zero_point
        return np.clip(q, -128, 127).astype(np.int8)[None, ...]
    return (img.astype(np.float32) / 255.0)[None, ...]


def draw(frame, dets, model_size, fps, infer_ms):
    import cv2

    h, w = frame.shape[:2]
    sx, sy = w / model_size, h / model_size
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        label = f"{d['class_name']} {d['conf']:.2f}"
        cv2.putText(frame, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"{fps:.1f} FPS  infer {infer_ms:.0f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--infer-every-n", type=int, default=5)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--out-dir", default="/home/pi/sar_recordings/live_preview_logs")
    cli = ap.parse_args()

    import cv2
    from tflite_runtime.interpreter import Interpreter

    interp = Interpreter(model_path=cli.model, num_threads=cli.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, model_h, model_w, _ = inp["shape"]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    print(f"model input {inp['shape']} {inp['dtype'].__name__}", flush=True)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = out_dir / f"preview_{stamp}.jsonl"
    log_fh = log_path.open("w", buffering=1)
    print(f"[log] writing detections to {log_path}", flush=True)

    cap = cv2.VideoCapture(cli.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cli.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cli.height)
    cap.set(cv2.CAP_PROP_FPS, cli.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened():
        print("camera failed to open", flush=True)
        return 1
    ok, frame = cap.read()
    if not ok:
        print("camera opened but first read failed", flush=True)
        return 1
    h, w = frame.shape[:2]

    ffplay = subprocess.Popen(
        [
            "ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}", "-framerate", str(cli.fps),
            "-window_title", "SAR live preview (Pi)", "-i", "-",
        ],
        stdin=subprocess.PIPE,
    )

    dets: list[dict] = []
    infer_ms = 0.0
    t_prev = time.time()
    t_end = time.time() + cli.seconds
    frame_idx = 0
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            if frame_idx % cli.infer_every_n == 0:
                x = preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
                t0 = time.perf_counter()
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                y = interp.get_tensor(out["index"])
                infer_ms = (time.perf_counter() - t0) * 1000
                dets = decode_yolo_output(
                    y, out_scale, out_zero, conf_thres=cli.conf_thres, iou_thres=cli.iou_thres, img_size=model_w
                )
                log_fh.write(json.dumps({
                    "frame": frame_idx, "t": time.time(),
                    "infer_ms": round(infer_ms, 2), "detections": dets,
                }) + "\n")
            now = time.time()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            annotated = draw(frame, dets, model_w, fps, infer_ms)
            try:
                ffplay.stdin.write(annotated.tobytes())
            except BrokenPipeError:
                print("[ffplay] window closed", flush=True)
                break
            frame_idx += 1
    finally:
        cap.release()
        try:
            ffplay.stdin.close()
        except Exception:
            pass
        ffplay.terminate()
        log_fh.close()
    print(f"[stop] preview ended, log at {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
