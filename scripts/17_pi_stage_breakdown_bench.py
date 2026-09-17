#!/usr/bin/env python3
"""Instrumented breakdown: separately time capture-read, inference, and
mp4-encode/write per frame, headless. Diagnostic only -- not part of the
field pipeline; contrast with scripts/13_pi_bench.py, which only measures
raw interpreter.invoke() latency against stored tiles and does not touch
the camera or video writer.

Exists because 15_pi_live.py's own wall-clock numbers undercounted
non-inference cost: notes/20260822-handover-pi-numpy-fix-and-416-bench.md
reported "everything outside inference costs only 5 to 13 ms", but this
script found mp4 write alone costs ~47ms/frame at 1280x720 (see
notes/20260822-fps-sweet-spot-and-mp4-encode-cost.md for the corrected
numbers). Measure stage-by-stage rather than subtract two aggregate
timings when reconciling this again.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--infer-every-n", type=int, default=10)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    cli = ap.parse_args()

    import cv2
    from tflite_runtime.interpreter import Interpreter

    t_load0 = time.perf_counter()
    interp = Interpreter(model_path=cli.model, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, model_h, model_w, _ = inp["shape"]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    t_load1 = time.perf_counter()

    t_cam0 = time.perf_counter()
    cap = cv2.VideoCapture(cli.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cli.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cli.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ok, frame = cap.read()
    t_cam1 = time.perf_counter()

    writer = cv2.VideoWriter("/tmp/bench_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))

    read_ms, infer_ms_list, write_ms, loop_ms = [], [], [], []

    for i in range(cli.frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        read_ms.append((t1 - t0) * 1000)

        if i % cli.infer_every_n == 0:
            x = preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
            t2 = time.perf_counter()
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            y = interp.get_tensor(out["index"])
            t3 = time.perf_counter()
            infer_ms_list.append((t3 - t2) * 1000)
            decode_yolo_output(y, out_scale, out_zero, conf_thres=0.25, iou_thres=0.45, img_size=model_w)

        t4 = time.perf_counter()
        writer.write(frame)
        t5 = time.perf_counter()
        write_ms.append((t5 - t4) * 1000)
        loop_ms.append((t5 - t0) * 1000)

    writer.release()
    cap.release()

    def stats(name, arr):
        if not arr:
            print(f"{name}: n=0")
            return
        arr = sorted(arr)
        n = len(arr)
        print(f"{name}: n={n} mean={sum(arr)/n:.1f}ms p50={arr[n//2]:.1f}ms max={arr[-1]:.1f}ms")

    print(f"model load: {(t_load1 - t_load0)*1000:.0f}ms")
    print(f"camera open + first read: {(t_cam1 - t_cam0)*1000:.0f}ms")
    stats("capture read", read_ms)
    stats("inference (invoke)", infer_ms_list)
    stats("mp4 write", write_ms)
    stats("full loop iter", loop_ms)
    total = sum(loop_ms) / 1000
    print(f"sum of loop iters: {total:.2f}s over {cli.frames} frames -> {cli.frames/total:.2f} fps sustained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
