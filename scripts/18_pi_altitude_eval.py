#!/usr/bin/env python3
"""Batch-evaluate a recorded video or image folder against the deployed tflite
model -- built to answer one question the live desk demo cannot: does this
model actually detect a person from real altitude (30-45m), not a close-range
face. Runs ON the Pi, same convention as 13_pi_bench.py / 15_pi_live.py: flat,
no package structure, decode.py must sit next to it.

Workflow this is for: take the flight camera (or any camera) up 30-45m,
record a clip with a person continuously visible for a known window, copy the
clip to the Pi, run this script, read the reported detection rate.

Ground truth is optional but changes what gets reported. Without any
--segment, this only reports "N of M processed frames had >=1 person
detection" -- a meaningless number unless you separately know the person was
actually in frame for all of them. With one or more --segment
LABEL:START:END windows supplied (video input only, repeatable -- e.g. two
segments to compare a standing pose against a walking pass at the same
distance), it reports a per-label recall proxy with a Wilson 95% interval,
since a single field clip is a small, noisy sample and a point estimate alone
invites overconfidence. --absent-start-sec/--absent-end-sec, if given, marks
a window known to have no target in frame at all, for a false-positive rate.

Re-running this script against the SAME recorded clip at different
--sample-every-n values is the intended way to test whether a detection gap
between conditions (e.g. standing vs walking) is really about the target's
appearance, or just about how many independent inference attempts each
condition got before the clip moved on -- the recording only has to happen
once.

This is a field proxy, not a replacement for results/ablation_test.csv or
results/cross_domain_wisard.csv -- those remain the rigorous, bootstrapped
numbers. This script exists because neither of those isolates a specific
altitude band; this does, for whatever altitude you actually flew or climbed
to.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import decode_yolo_output  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def draw(frame, dets, model_size):
    import cv2

    h, w = frame.shape[:2]
    sx, sy = w / model_size, h / model_size
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f"{d['class_name']} {d['conf']:.2f}", (p1[0], max(0, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Closed-form, no
    scipy dependency -- appropriate here since n is a few hundred frames at
    most, not thousands; a normal-approximation interval would undercoverage
    at the edges (p near 0 or 1), which is exactly where a weak detector's
    recall tends to sit.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((center - spread) / denom, (center + spread) / denom)


def iter_frames(input_path: Path, sample_every_n: int):
    import cv2

    if input_path.is_dir():
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for idx, p in enumerate(paths):
            if idx % sample_every_n:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"[warn] unreadable image, skipping: {p}", flush=True)
                continue
            yield idx, idx, frame  # (frame_idx, source_index, frame) -- no real timestamp for stills
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every_n == 0:
            yield idx, idx / fps, frame
        idx += 1
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="video file or directory of images")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sample-every-n", type=int, default=1)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--target-class", default="person")
    ap.add_argument("--segment", action="append", default=[],
                     help="video input only, repeatable: LABEL:START_SEC:END_SEC, a window where the "
                          "target is known to be in frame under some labeled condition (e.g. standing:10:30)")
    ap.add_argument("--absent-start-sec", type=float, default=None,
                     help="video input only: start of a window known to have no target in frame")
    ap.add_argument("--absent-end-sec", type=float, default=None,
                     help="video input only: end of that window")
    cli = ap.parse_args()

    segments = []
    for spec in cli.segment:
        try:
            label, start_s, end_s = spec.rsplit(":", 2)
            segments.append((label, float(start_s), float(end_s)))
        except ValueError:
            print(f"[error] --segment must be LABEL:START:END, got: {spec!r}", flush=True)
            return 1

    import cv2
    from tflite_runtime.interpreter import Interpreter

    input_path = Path(cli.input)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interp = Interpreter(model_path=cli.model, num_threads=cli.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, model_h, model_w, _ = inp["shape"]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    print(f"model input {inp['shape']} {inp['dtype'].__name__}", flush=True)

    has_absent_window = cli.absent_start_sec is not None and cli.absent_end_sec is not None
    if input_path.is_dir():
        if segments:
            print("[warn] --segment ignored for image-folder input (no timeline to place it on)", flush=True)
            segments = []
        if has_absent_window:
            print("[warn] --absent-*-sec ignored for image-folder input", flush=True)
            has_absent_window = False

    log_path = out_dir / "eval.jsonl"
    log_fh = log_path.open("w")

    writer = None
    n_total = 0
    n_hit = 0            # frames with >=1 target-class detection, whole clip
    seg_counts = {label: [0, 0] for label, _, _ in segments}  # label -> [n_frames, n_hit]
    n_absent = 0
    n_hit_absent = 0
    infer_ms_all = []

    import time

    for frame_idx, t_sec, frame in iter_frames(input_path, cli.sample_every_n):
        x = preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_ms_all.append(infer_ms)

        dets = decode_yolo_output(y, out_scale, out_zero, conf_thres=cli.conf_thres,
                                   iou_thres=cli.iou_thres, img_size=model_w)
        target_dets = [d for d in dets if d["class_name"] == cli.target_class]
        hit = len(target_dets) > 0

        n_total += 1
        n_hit += int(hit)

        for label, start_s, end_s in segments:
            if start_s <= t_sec <= end_s:
                seg_counts[label][0] += 1
                seg_counts[label][1] += int(hit)

        if has_absent_window and cli.absent_start_sec <= t_sec <= cli.absent_end_sec:
            n_absent += 1
            n_hit_absent += int(hit)

        log_fh.write(json.dumps({
            "frame": frame_idx, "t_sec": round(t_sec, 3) if isinstance(t_sec, float) else t_sec,
            "infer_ms": round(infer_ms, 2), "detections": dets,
        }) + "\n")

        annotated = draw(frame, dets, model_w)
        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        writer.write(annotated)

    if writer is not None:
        writer.release()
    log_fh.close()

    if n_total == 0:
        print("[error] no frames processed -- check --input path", flush=True)
        return 1

    infer_ms_all.sort()
    mean_ms = sum(infer_ms_all) / len(infer_ms_all)
    p50 = infer_ms_all[len(infer_ms_all) // 2]

    print(f"\n=== {input_path.name} ===")
    print(f"frames processed: {n_total}")
    print(f"inference: mean={mean_ms:.1f}ms p50={p50:.1f}ms")
    print(f"raw detection rate (whole clip, needs external ground truth to interpret): "
          f"{n_hit}/{n_total} = {100*n_hit/n_total:.1f}%")

    if segments:
        print(f"\n{'label':<16} {'n_frames':>9} {'hits':>6} {'recall proxy':>13} {'95% Wilson CI':>18}")
        for label, start_s, end_s in segments:
            n_seg, hit_seg = seg_counts[label]
            if n_seg == 0:
                print(f"{label:<16} [{start_s:.1f}s,{end_s:.1f}s] matched 0 frames -- check window against clip length")
                continue
            recall = hit_seg / n_seg
            lo, hi = wilson_interval(hit_seg, n_seg)
            print(f"{label:<16} {n_seg:>9} {hit_seg:>6} {100*recall:>12.1f}% "
                  f"{100*lo:>7.1f}-{100*hi:<7.1f}%  [{start_s:.1f}s-{end_s:.1f}s]")
        if len(segments) >= 2:
            print("\nCompare recall proxies above across labels/re-runs at different --sample-every-n "
                  "to separate 'appearance changed' from 'got fewer independent looks'.")
    if has_absent_window:
        if n_absent:
            fpr = n_hit_absent / n_absent
            print(f"\nfalse-positive rate in known-absent window "
                  f"[{cli.absent_start_sec:.1f}s, {cli.absent_end_sec:.1f}s]: "
                  f"{n_hit_absent}/{n_absent} = {100*fpr:.1f}%")
        else:
            print("[warn] absent window matched 0 frames -- check the start/end seconds against the clip length")
    if not segments and not has_absent_window:
        print("no --segment given: the raw rate above is uninterpretable on its own. "
              "Re-run with --segment LABEL:START:END, or manually scrub annotated.mp4 and count hits/misses yourself.")

    print(f"\nannotated video: {out_dir / 'annotated.mp4'}")
    print(f"per-frame log:   {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
