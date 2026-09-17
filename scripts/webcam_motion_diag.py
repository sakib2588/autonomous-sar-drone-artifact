#!/usr/bin/env python3
"""Discriminating diagnostic: WHY does a moving person get missed on webcam?

Run from the repo root with the webcam plugged in:

    .venv/bin/python scripts/webcam_motion_diag.py

Four candidate causes produce the same symptom ("detector loses a walking
person, holds a standing one"), and they need different fixes, so guessing is
wasteful. This script measures all four in one run and prints a verdict:

  C1 MOTION BLUR      -- a long shutter smears the subject and YOLO's
                         confidence collapses.
  C2 FRAMERATE DROP   -- the driver delivers well under the advertised FPS,
                         which widens the gap between successive frames.
  C3 STALE FRAMES     -- CAP_PROP_BUFFERSIZE defaults to 4 and cap.read() pops
                         the OLDEST frame, so with capture outrunning inference
                         the queue sits full and every frame reaching the model
                         is already several frames old. (Measured at 166 ms on
                         the Microdia Vitade AF, 2026-08-23; the repo scripts
                         now set depth 1, which cut it to 42 ms at identical
                         throughput. This script still measures it, because the
                         Pi's driver may not honour the setting.)
  C4 CONF THRESHOLD   -- the blurred person IS detected, just below
                         --conf-thres, so it is discarded before display.

Method note: the timed segments run capture and display only -- inference is
run offline afterwards on a stored subset. Putting the model inside the
measurement loop would drag capture down to inference speed and contaminate
the very FPS number being measured.

Known-dead-control warning: on the Microdia 0c45:6366 tested here,
exposure_time_absolute is accepted and read back but IGNORED by the sensor
firmware. Swept 0.1 ms to 200 ms (2000x): mean pixel stayed 118-121 and FPS
stayed 24.04, which is physically impossible for a real 200 ms exposure. So on
that camera the only lever on blur is scene light. Re-run this script under a
bright lamp and compare, using --label to tell the runs apart.

Outputs a JSON summary plus the worst-blur frames as JPGs to
scripts/webcam_diag_out/ (gitignored scratch, same convention as
webcam_smoke_out/).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Set before cv2 loads: OpenCV ships no Qt wayland plugin, so on a wayland
# session it prints a plugin error on every run and silently falls back to
# xcb through XWayland anyway. Pinning xcb skips the noise, same result.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "webcam_diag_out"
KEEP_FRAMES = 40  # per segment, subsampled; caps RAM at a few hundred MB
WINDOW = "SAR motion diagnostic"


def find_working_camera(max_index: int = 5):
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, idx
        cap.release()
    raise RuntimeError(f"no working camera in indices 0..{max_index - 1}")


def fourcc_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def sharpness(frame) -> float:
    """Variance of the Laplacian on the centre crop. Falls as blur rises."""
    h, w = frame.shape[:2]
    crop = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mirror(frame):
    """Self-view is mirrored so the subject can position themselves naturally.
    Display only -- every measurement runs on the unmirrored frame."""
    return cv2.flip(frame, 1)


def banner(disp, text: str, sub: str, color=(0, 255, 0)):
    h, w = disp.shape[:2]
    cv2.rectangle(disp, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.putText(disp, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(disp, sub, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
    return disp


class Preview:
    """imshow that degrades to a no-op if there is no display (Pi over SSH)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            try:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, 960, 720)
            except cv2.error as exc:
                print(f"[preview] disabled, no display available ({exc.__class__.__name__})")
                self.enabled = False

    def show(self, disp) -> int:
        if not self.enabled:
            return 255
        cv2.imshow(WINDOW, disp)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


def probe_props(cap) -> dict:
    return {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "advertised_fps": cap.get(cv2.CAP_PROP_FPS),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "buffersize": cap.get(cv2.CAP_PROP_BUFFERSIZE),
    }


def measure_buffer_depth(cap, idle_s: float = 1.0) -> dict:
    """Stop reading for idle_s, then time each read. Frames already sitting in
    the driver queue come back near-instantly; the first read that has to WAIT
    for the sensor marks the end of the backlog. That count is the lag depth."""
    for _ in range(5):
        cap.read()
    time.sleep(idle_s)
    gaps = []
    for _ in range(12):
        t0 = time.perf_counter()
        cap.read()
        gaps.append((time.perf_counter() - t0) * 1000.0)
    depth = 0
    for g in gaps:
        if g < 3.0:
            depth += 1
        else:
            break
    return {"read_ms": [round(g, 2) for g in gaps], "queued_frames": depth}


def framing_check(cap, prev: Preview, model, imgsz: int, timeout_s: float = 60.0) -> bool:
    """Live self-view with a detection overlay, BEFORE anything is timed.

    Without this the subject stands in front of the camera blind, and a test
    where they were half out of frame looks identical to a real detection
    failure. Inference here is deliberately unmeasured -- it exists so the
    subject can confirm the model sees them at all before the clock starts.
    """
    print("\n>>> FRAMING CHECK")
    print("    Stand where you will do the test. GREEN box = the model sees you.")
    print("    Press SPACE to begin, or q to abort.  (auto-starts in 60 s)")
    if not prev.enabled:
        print("    [no display] skipping framing check, starting in 5 s")
        time.sleep(5.0)
        return True

    t0 = time.time()
    last: list = []
    n = 0
    while time.time() - t0 < timeout_s:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        if n % 3 == 1:
            r = model.predict(frame, imgsz=imgsz, conf=0.15, classes=[0], device="cpu", verbose=False)[0]
            last = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]

        disp = mirror(frame)
        w = disp.shape[1]
        for xyxy, c in last:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(disp, (w - x2, y1), (w - x1, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"person {c:.2f}", (w - x2, max(70, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if last:
            msg, col = f"DETECTED  conf {max(c for _, c in last):.2f}", (0, 255, 0)
        else:
            msg, col = "NOT DETECTED -- step into frame", (0, 165, 255)
        banner(disp, msg, "SPACE = start the test    q = abort", col)

        key = prev.show(disp)
        if key == ord(" "):
            return True
        if key in (ord("q"), 27):
            return False
    return True


def capture_segment(cap, prev: Preview, label: str, sub: str, seconds: float, color) -> tuple[dict, list]:
    print(f"\n>>> {label}  ({seconds:.0f} s)", flush=True)
    t0 = time.time()
    sharp, kept, n = [], [], 0
    while True:
        left = seconds - (time.time() - t0)
        if left <= 0:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        sharp.append(sharpness(frame))
        if len(kept) < KEEP_FRAMES and n % 3 == 0:
            kept.append(frame.copy())
        prev.show(banner(mirror(frame), f"{label}   {left:.0f}", sub, color))
    dur = time.time() - t0
    stats = {
        "frames": n,
        "seconds": round(dur, 2),
        "measured_fps": round(n / dur, 2),
        "sharpness_median": round(float(np.median(sharp)), 1) if sharp else None,
        "sharpness_p10": round(float(np.percentile(sharp, 10)), 1) if sharp else None,
    }
    print(f"    {stats['measured_fps']} FPS, sharpness median {stats['sharpness_median']}", flush=True)
    return stats, kept


def score_frames(model, frames: list, imgsz: int) -> dict:
    """Offline inference at conf=0.05 -- deliberately far BELOW the pipeline's
    0.35 so we can see detections the live tool is silently discarding."""
    confs = []
    for f in frames:
        r = model.predict(f, imgsz=imgsz, conf=0.05, classes=[0], device="cpu", verbose=False)[0]
        c = [float(b.conf[0]) for b in r.boxes]
        confs.append(max(c) if c else 0.0)
    arr = np.array(confs) if confs else np.array([0.0])
    return {
        "n": len(confs),
        "max_person_conf_median": round(float(np.median(arr)), 3),
        "frac_above_0.35": round(float((arr >= 0.35).mean()), 3),
        "frac_above_0.15": round(float((arr >= 0.15).mean()), 3),
        "frac_zero": round(float((arr == 0.0).mean()), 3),
    }


def build_verdict(still: dict, moving: dict, buf: dict, adv: float) -> list[str]:
    """Verdict from the STILL-vs-MOVING contrast, not from absolute cutoffs.

    The first version of this used fixed thresholds on the moving segment alone
    and reported "no single cause dominates" on a run that had lost 27% of its
    moving frames -- because 72.5% retained still looked healthy in isolation.
    It is the paired contrast that carries the signal: same subject, same light,
    same model, differing only by motion.
    """
    keep_still = still["frac_above_0.35"]
    keep_move = moving["frac_above_0.35"]
    drop = keep_still - keep_move
    recoverable = moving["frac_above_0.15"] - keep_move
    hard = moving["frac_zero"]
    sh_still = still["sharpness_median"] or 1.0
    sh_move = moving["sharpness_median"] or 0.0

    v = []
    if drop >= 0.05:
        v.append(f"MOTION-INDUCED LOSS: {keep_still:.0%} of STILL frames clear 0.35, only {keep_move:.0%} of "
                 f"MOVING frames do -- {drop:.0%} points lost to motion alone.")
        if recoverable >= 0.05:
            v.append(f"    C4 RECOVERABLE IN SOFTWARE: {recoverable:.0%} of moving frames land between 0.15 and "
                     f"0.35 -- detected, then discarded by the threshold. Lowering --conf-thres recovers these.")
        if hard > 0:
            v.append(f"    C1 NOT recoverable by threshold: {hard:.0%} of moving frames yield NO box even at "
                     f"conf=0.05. That is image quality, not thresholding.")
    else:
        v.append(f"NO motion-induced loss: still {keep_still:.0%} vs moving {keep_move:.0%} clear 0.35.")

    v.append(f"    sharpness {sh_still:.0f} still -> {sh_move:.0f} moving "
             f"({100 * (1 - sh_move / sh_still):.0f}% drop); absolute sharpness is what predicts detection, "
             f"not the relative drop.")
    if adv and moving["measured_fps"] < 0.7 * adv:
        v.append(f"    C2 FRAMERATE DROP: advertises {adv} FPS, delivers {moving['measured_fps']}.")
    if buf["queued_frames"] >= 3:
        v.append(f"    C3 STALE FRAMES: {buf['queued_frames']} queued -> "
                 f"~{1000 * buf['queued_frames'] / max(adv, 1):.0f} ms lag before inference starts.")
    return v


def compare_runs(label_a: str, label_b: str) -> int:
    """Diff two saved runs. Needs no camera -- reads the JSONs off disk."""
    runs = {}
    for lab in (label_a, label_b):
        path = OUT_DIR / f"diag_{lab}.json"
        if not path.exists():
            print(f"[FAIL] no such run: {path}")
            return 1
        runs[lab] = json.loads(path.read_text())

    print(f"{'':>26} {label_a:>14} {label_b:>14}")
    rows = [
        ("still  sharpness", lambda r: r["still"]["sharpness_median"], "{:.1f}"),
        ("moving sharpness", lambda r: r["moving"]["sharpness_median"], "{:.1f}"),
        ("still  conf median", lambda r: r["still"]["max_person_conf_median"], "{:.3f}"),
        ("moving conf median", lambda r: r["moving"]["max_person_conf_median"], "{:.3f}"),
        ("moving kept @0.35", lambda r: r["moving"]["frac_above_0.35"], "{:.1%}"),
        ("moving kept @0.15", lambda r: r["moving"]["frac_above_0.15"], "{:.1%}"),
        ("moving total miss", lambda r: r["moving"]["frac_zero"], "{:.1%}"),
        ("measured FPS", lambda r: r["moving"]["measured_fps"], "{:.2f}"),
        ("queued frames", lambda r: r["buffer"]["queued_frames"], "{:.0f}"),
    ]
    for name, get, fmt in rows:
        print(f"{name:>26} {fmt.format(get(runs[label_a])):>14} {fmt.format(get(runs[label_b])):>14}")

    for lab in (label_a, label_b):
        r = runs[lab]
        print(f"\n--- {lab} ---")
        for line in build_verdict(r["still"], r["moving"], r["buffer"], r["props_before"]["advertised_fps"]):
            print("  " + line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--seconds", type=float, default=6.0, help="per segment")
    ap.add_argument("--label", default="default", help="tag for this run, e.g. roomlight / lamp. Goes in the JSON filename")
    ap.add_argument("--no-preview", action="store_true", help="headless: no window, no framing check")
    ap.add_argument("--exposure", type=int, default=None,
                    help="force MANUAL exposure, units of 100us (e.g. 20 = 2ms). NOTE: verified a no-op on the "
                         "Microdia 0c45:6366 -- check with v4l2-ctl before trusting it")
    ap.add_argument("--buffersize", type=int, default=1)
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="diff two saved runs and re-print their verdicts. No camera needed")
    cli = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if cli.compare:
        return compare_runs(*cli.compare)

    if cli.camera is not None:
        cap = cv2.VideoCapture(cli.camera, cv2.CAP_V4L2)
        idx = cli.camera
        if not cap.isOpened():
            raise RuntimeError(f"could not open /dev/video{idx}")
    else:
        cap, idx = find_working_camera()
    print(f"[camera] /dev/video{idx}")

    before = probe_props(cap)
    if cli.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # UVC: 1 = manual, 3 = aperture priority
        cap.set(cv2.CAP_PROP_EXPOSURE, cli.exposure)
    if cli.buffersize is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cli.buffersize)
    after = probe_props(cap)

    print("[props]")
    for k in before:
        mark = "" if before[k] == after[k] else f"   -> {after[k]}"
        print(f"    {k:16s} {before[k]}{mark}")

    print("\n[buffer] measuring queue depth (stay out of the way, this is not timed)")
    buf = measure_buffer_depth(cap)
    print(f"    frames served instantly: {buf['queued_frames']}  read_ms={buf['read_ms'][:6]}")

    print(f"\n[model] loading {cli.model}")
    from ultralytics import YOLO

    model = YOLO(cli.model)
    model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=cli.imgsz, device="cpu", verbose=False)  # warm-up

    prev = Preview(enabled=not cli.no_preview)
    try:
        if not framing_check(cap, prev, model, cli.imgsz):
            print("[abort] framing check cancelled")
            return 1

        # Auto-exposure and auto-white-balance keep adapting for seconds after
        # the stream starts, and they move the sharpness metric by 3x on their
        # own (measured 98 -> 286 across back-to-back idle segments). Settle
        # first, or the still-vs-moving comparison is measuring the AE loop
        # rather than the subject.
        print("\n[settle] letting auto-exposure stabilise (3 s)")
        t_settle = time.time()
        while time.time() - t_settle < 3.0:
            ok, frame = cap.read()
            if ok:
                prev.show(banner(mirror(frame), "SETTLING", "hold position, do not move yet", (120, 120, 255)))

        still_stats, still_frames = capture_segment(
            cap, prev, "HOLD STILL", "stand as still as you can", cli.seconds, (0, 255, 0))
        move_stats, move_frames = capture_segment(
            cap, prev, "MOVE NOW", "walk / wave arms fast, keep moving", cli.seconds, (0, 165, 255))
    finally:
        prev.close()
        cap.release()

    print("\n[infer] scoring stored frames offline at conf=0.05")
    still_score = score_frames(model, still_frames, cli.imgsz)
    move_score = score_frames(model, move_frames, cli.imgsz)

    if move_frames:
        order = sorted(range(len(move_frames)), key=lambda i: sharpness(move_frames[i]))
        for rank, i in enumerate(order[:3]):
            cv2.imwrite(str(OUT_DIR / f"{cli.label}_blurriest_moving_{rank}.jpg"), move_frames[i])
    if still_frames:
        cv2.imwrite(str(OUT_DIR / f"{cli.label}_sharpest_still.jpg"), max(still_frames, key=sharpness))

    verdict = build_verdict(
        {**still_stats, **still_score}, {**move_stats, **move_score}, buf, before["advertised_fps"] or 0.0
    )

    summary = {
        "label": cli.label, "camera_index": idx, "props_before": before, "props_after": after,
        "buffer": buf, "still": {**still_stats, **still_score}, "moving": {**move_stats, **move_score},
        "verdict": verdict,
    }
    (OUT_DIR / f"diag_{cli.label}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("STILL  ", json.dumps({**still_stats, **still_score}))
    print("MOVING ", json.dumps({**move_stats, **move_score}))
    print("=" * 72)
    for v in verdict:
        print("  * " + v)
    print(f"\nwrote {OUT_DIR / f'diag_{cli.label}.json'} + JPGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
