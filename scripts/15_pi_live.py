#!/usr/bin/env python3
"""On-Pi live pipeline: capture + record + detect, started headless by systemd.

Runs ON the Raspberry Pi. Copies of src/sar/bench/{decode,sysmon}.py must sit
next to it -- the Pi has no checkout of the repo, same convention as
13_pi_bench.py.

This is NOT the benchmark harness. 13_pi_bench.py measures latency against a
folder of test tiles and stops at the raw model tensor. This script is the
actual field pipeline: it opens the webcam, writes video to the SD card
continuously, runs the exported int8 model on a sampled subset of frames
(inference is far slower than capture -- see 13_pi_bench's own numbers -- so
running it on every frame would fall behind and the buffer would grow
unbounded), decodes real boxes via src/sar/bench/decode.py, and appends one
JSON line per processed frame to a detections log. It does not fly the
drone, upload anywhere, or touch the flight controller -- geotagging via
Pixhawk/MAVLink is listed as pending in specs/sar-drone-design-spec.md
Section 4 and is explicitly out of scope here.

Camera stays [0,1]-normalised the same way training did (see
src/sar/bench/decode.py's dequantisation note) -- the int8 input path below
mirrors 13_pi_bench.py's preprocessing exactly, on purpose. Diverging from
that preprocessing is the single most common way to silently wreck accuracy
while latency looks fine (GATE-4's whole reason for existing).

Two recording defects were fixed on 2026-08-23, both of which made a MOVING
person look uncapturable in the footage while a stationary one looked fine.
First, boxes were drawn only on the 1-in-N frames the model ran on, so the
recording strobed them on for one frame and off for four; the most recent
detection is now held across the gap (thin outline and a "~" prefix, so the
video never implies an inference that did not happen). Second, the mp4 header
was stamped with --fps, but the loop never runs at capture rate -- inference
and 720p encoding both cost wall-clock time -- so 30 FPS footage of a ~9 FPS
reality played back over three times too fast and a walking survivor appeared
to teleport. The true rate is now measured at startup and re-measured every
segment, with a .timing.json sidecar recording what actually happened.

detections.jsonl remains the evidence artifact: one line per REAL inference,
nothing interpolated. Numbers for the paper come from that file, never from
counting boxes in the video.

Designed to survive `systemctl restart` and camera hot-unplug: every
recoverable error is caught and retried with a backoff, not raised, because
this process is meant to run unattended for the length of a flight and a
raised exception here means dead recording with nobody watching a screen.
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode import CLASS_NAMES, decode_yolo_output  # noqa: E402
from tiling import generate_tiles, merge_tile_detections  # noqa: E402
from sysmon import describe, is_clean, parse_throttled  # noqa: E402

_STOP = False


def _handle_stop(signum, frame) -> None:  # noqa: ANN001
    global _STOP
    _STOP = True


def vcgencmd(*args: str) -> str:
    import subprocess

    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout


def open_camera(device: str, width: int, height: int, fps: int, buffersize: int = 2):
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    # Depth 2, NOT 1 -- and this differs from the laptop tools on purpose.
    #
    # Measured on this Pi 2026-08-23, 1280x720 MJPG, ABAB-verified:
    #   buffersize 1 -> 12.00 FPS      (HALF rate)
    #   buffersize 2 -> 23.88 FPS
    #   buffersize 3 -> 23.97 FPS
    #   buffersize 4 -> 23.97 FPS (driver default)
    # A single buffer leaves the driver nothing to fill while the consumer holds
    # the only one, so capture stalls a whole frame interval and the rate halves.
    # Two is the minimum that keeps streaming uninterrupted.
    #
    # The desk webcam behaved differently: depth 1 was free there (24.06 vs
    # 24.03 FPS) and cut 166 ms of stale-frame lag to 42 ms. This Pi measured
    # queue depth 0 at the default, i.e. it had no stale backlog to fix at all.
    # Same OpenCV call, opposite right answer -- which is why this is measured
    # per platform rather than copied across.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # MJPG, not the default YUYV: an uncompressed 1280x720 feed at YUYV
    # saturates USB2 bandwidth on the Pi 4 and caps FPS far below what the
    # camera advertises. This is the standard fix, not a stylistic choice.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def preprocess(frame, size: int, scale: float, zero_point: int, dtype):
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


def draw_detections(frame, dets: list[dict], model_size: int, fresh: bool = True, frame_coords: bool = False):
    import cv2

    h, w = frame.shape[:2]
    # Untiled boxes arrive in model-input space and need mapping back to the
    # frame. Tiled boxes were already shifted to full-frame pixels by
    # merge_tile_detections, so scaling them again would shrink every box.
    sx, sy = (1.0, 1.0) if frame_coords else (w / model_size, h / model_size)
    # A held box -- one carried onto a frame the model did not actually run on --
    # is drawn thin and prefixed "~", so the footage never implies an inference
    # that did not happen. detections.jsonl stays the evidence artifact: one line
    # per REAL inference, nothing interpolated. Any number quoted in the paper
    # comes from that file, never from counting boxes in the video.
    thickness = 2 if fresh else 1
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness)
        label = f"{'' if fresh else '~'}{d['class_name']} {d['conf']:.2f}"
        cv2.putText(frame, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def held_boxes(frame_idx: int, last_infer_frame: int, last_dets: list[dict], hold_frames: int) -> list[dict]:
    """Which boxes to carry onto a frame the model did not actually run on.

    Inference runs on 1-of-N frames, so without this the other N-1 frames drew
    nothing and the recording strobed a box on for one frame and off for four.
    An inference that returns nothing sets last_dets empty, so a genuine loss
    clears the held box instead of freezing a survivor in place forever.
    """
    if not last_dets:
        return []
    return last_dets if (frame_idx - last_infer_frame) < hold_frames else []


def close_segment(writer, seg_path, frames: int, started: float, ended: float, nominal_fps: float) -> None:
    """Release a segment and drop a timing sidecar next to it.

    The mp4 header can only carry one constant frame rate, but the real loop
    rate drifts (thermal throttling, inference jitter, SD-card stalls). The
    sidecar records what actually happened so footage can be re-timed exactly
    afterwards instead of being trusted blind.
    """
    writer.release()
    dur = max(ended - started, 1e-6)
    (seg_path.with_suffix(".timing.json")).write_text(
        json.dumps(
            {
                "segment": seg_path.name,
                "header_fps": round(nominal_fps, 3),
                "frames_written": frames,
                "wall_seconds": round(dur, 3),
                "true_fps": round(frames / dur, 3),
                "note": "header_fps is what the mp4 claims; true_fps is measured. "
                "Re-time with true_fps before reading motion off this video.",
            },
            indent=2,
        )
    )
    print(f"[segment] closed {seg_path.name}: {frames} frames, {dur:.1f}s, true {frames / dur:.2f} FPS", flush=True)


def infer_tiled(interp, inp, out, frame, model_w, in_scale, in_zero, out_scale, out_zero,
                conf_thres: float, iou_thres: float, overlap: float) -> list[dict]:
    """Tile the frame, infer per tile, merge back to full-frame coordinates.

    This is what 11_fullframe_eval.py calls the deployment path -- "tile,
    detect, merge" -- and until 2026-08-23 this script did not do it. It fed
    the whole 1280x720 frame through a single cv2.resize to 640x640, which is
    0.50x horizontally and 0.89x vertically: every person shrinks below the
    trained size AND the aspect ratio is distorted, neither of which the
    tile-trained model ever saw. results/object_size_distribution.csv puts the
    trained person envelope at roughly 8-32 px, and the resize pushes a
    60 m target under it.

    Cost is real and must be budgeted, not discovered in the field: a
    1280x720 frame at 640 tiles with 20% overlap is 6 tiles, so one frame
    costs 6 inferences. Use --infer-every-n to trade detection cadence
    against it.
    """
    tiles = generate_tiles(frame.shape[1], frame.shape[0], tile_size=model_w, overlap=overlap)
    per_tile = []
    for tile in tiles:
        tx0, ty0, tx1, ty1 = tile
        crop = frame[ty0:ty1, tx0:tx1]
        x = preprocess(crop, model_w, in_scale, in_zero, inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        dets = decode_yolo_output(y, out_scale, out_zero, conf_thres=conf_thres,
                                  iou_thres=iou_thres, img_size=model_w)
        per_tile.append((tile, [(d["class_id"], d["conf"], *d["box_xyxy"]) for d in dets]))

    merged = merge_tile_detections(per_tile, iou_thresh=iou_thres)
    return [
        {"class_id": c, "class_name": CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
         "conf": float(cf), "box_xyxy": [float(a), float(b), float(x2), float(y2)]}
        for c, cf, a, b, x2, y2 in merged
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .tflite artifact")
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--buffersize", type=int, default=2,
                    help="V4L2 queue depth. 2 measured optimal on this Pi; 1 HALVES capture rate here "
                         "(12.00 vs 23.88 FPS) even though 1 is free on the dev laptop. Re-measure on new hardware")
    ap.add_argument("--infer-every-n", type=int, default=5, help="run the model on 1 of every N captured frames")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--tile", action="store_true",
                    help="tile the frame and merge, matching how the model was TRAINED and how "
                         "11_fullframe_eval.py measures deployment. Costs one inference per tile "
                         "(6 for 1280x720 at 640/0.2). Without this the frame is resized whole, which "
                         "is what the untiled cross-domain AP50 of 0.0449 looks like")
    ap.add_argument("--tile-overlap", type=float, default=0.2,
                    help="must match the overlap the tiles were BUILT with (results/wisard_tile_manifest.json "
                         "records 0.2) -- a train/deploy tiling mismatch is a silent accuracy leak")
    ap.add_argument("--out-dir", default="/home/pi/sar_recordings")
    ap.add_argument("--segment-seconds", type=int, default=300, help="rotate to a new video file every N seconds")
    ap.add_argument(
        "--writer-fps",
        type=float,
        default=None,
        help="frame rate stamped into the mp4 header. Omit to measure the real loop rate at startup "
        "-- the capture rate is NOT --fps, because inference and encoding both cost wall-clock time",
    )
    ap.add_argument(
        "--calibrate-seconds",
        type=float,
        default=3.0,
        help="length of the startup rate measurement (not recorded). 0 disables it and falls back to --writer-fps/--fps",
    )
    ap.add_argument(
        "--box-hold-frames",
        type=int,
        default=None,
        help="keep drawing the last detection for this many frames after an inference. "
        "Defaults to --infer-every-n, i.e. bridge exactly to the next inference",
    )
    cli = ap.parse_args()

    import cv2
    from tflite_runtime.interpreter import Interpreter

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    throttle_flags = parse_throttled(vcgencmd("get_throttled").strip())
    if not is_clean(throttle_flags):
        print(f"WARNING: Pi is not power-clean at startup -- {describe(throttle_flags)}", flush=True)
        print("Recording anyway (this is a field run, not a benchmark), but fix the 5V supply.", flush=True)

    interp = Interpreter(model_path=cli.model, num_threads=cli.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, model_h, model_w, _ = inp["shape"]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    print(f"model input {inp['shape']} {inp['dtype'].__name__}, output {out['shape']}", flush=True)

    cap = open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
    reconnect_backoff = 1.0

    # The mp4 header used to be stamped with --fps (30), but the loop never runs
    # at capture rate: inference on 1-of-N frames plus 720p mp4v encoding both
    # cost wall-clock time. A 30 FPS header over a ~9 FPS reality plays footage
    # back ~3x too fast, which makes a walking survivor look like they teleport
    # -- the exact "camera cannot capture movement" symptom being chased here.
    # Measure the true rate first, with a throwaway writer in the loop so the
    # encode cost is included rather than optimistically ignored.
    writer_fps = cli.writer_fps
    if writer_fps is None and cli.calibrate_seconds > 0:
        print(f"[calibrate] measuring true loop rate for {cli.calibrate_seconds:.1f}s (not recorded)", flush=True)
        tmp_path = out_dir / ".calibrate_tmp.mp4"
        tmp_writer = None
        t_cal = time.time()
        n_cal = 0
        while time.time() - t_cal < cli.calibrate_seconds and not _STOP:
            ok, frame = cap.read()
            if not ok:
                continue
            if tmp_writer is None:
                fh, fw = frame.shape[:2]
                tmp_writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (fw, fh))
            if n_cal % cli.infer_every_n == 0:
                x = preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                interp.get_tensor(out["index"])
            tmp_writer.write(frame)
            n_cal += 1
        elapsed = time.time() - t_cal
        if tmp_writer is not None:
            tmp_writer.release()
        if tmp_path.exists():
            tmp_path.unlink()  # our own scratch file, created seconds ago in this function
        writer_fps = max(1.0, n_cal / elapsed) if n_cal and elapsed > 0 else float(cli.fps)
        print(f"[calibrate] {n_cal} frames in {elapsed:.2f}s -> header {writer_fps:.2f} FPS (--fps says {cli.fps})", flush=True)
    if writer_fps is None:
        writer_fps = float(cli.fps)

    hold_frames = cli.box_hold_frames if cli.box_hold_frames is not None else cli.infer_every_n

    log_path = out_dir / "detections.jsonl"
    log_fh = log_path.open("a", buffering=1)  # line-buffered: a crash loses at most one line

    writer = None
    seg_path = None
    seg_frames = 0
    segment_start = 0.0
    frame_idx = 0
    last_dets: list[dict] = []
    last_infer_frame = -(10**9)

    print(f"[start] recording to {out_dir}, log at {log_path}", flush=True)

    try:
        while not _STOP:
            if not cap.isOpened():
                print(f"[camera] not open, retrying in {reconnect_backoff:.0f}s", flush=True)
                time.sleep(reconnect_backoff)
                reconnect_backoff = min(reconnect_backoff * 2, 30.0)
                cap = open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
                continue
            reconnect_backoff = 1.0

            ok, frame = cap.read()
            if not ok:
                print("[camera] read failed, reopening", flush=True)
                cap.release()
                cap = open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
                continue

            now = time.time()
            if writer is None or (now - segment_start) >= cli.segment_seconds:
                if writer is not None:
                    close_segment(writer, seg_path, seg_frames, segment_start, now, writer_fps)
                    # Re-time the next segment from what the last one actually did;
                    # the rate drifts as the Pi heats up and starts throttling.
                    measured = seg_frames / max(now - segment_start, 1e-6)
                    if seg_frames > 0:
                        writer_fps = max(1.0, measured)
                stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                seg_path = out_dir / f"flight_{stamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(seg_path), fourcc, writer_fps, (w, h))
                segment_start = now
                seg_frames = 0
                print(f"[segment] {seg_path} at {writer_fps:.2f} FPS", flush=True)

            fresh = frame_idx % cli.infer_every_n == 0
            dets: list[dict] = []
            if fresh:
                t0 = time.perf_counter()
                if cli.tile:
                    dets = infer_tiled(interp, inp, out, frame, model_w, in_scale, in_zero,
                                       out_scale, out_zero, cli.conf_thres, cli.iou_thres, cli.tile_overlap)
                else:
                    x = preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
                    interp.set_tensor(inp["index"], x)
                    interp.invoke()
                    y = interp.get_tensor(out["index"])
                    dets = decode_yolo_output(
                        y, out_scale, out_zero, conf_thres=cli.conf_thres, iou_thres=cli.iou_thres,
                        img_size=model_w
                    )
                infer_ms = (time.perf_counter() - t0) * 1000
                log_fh.write(
                    json.dumps(
                        {
                            "t": now,
                            "frame": frame_idx,
                            "infer_ms": round(infer_ms, 2),
                            "tiled": cli.tile,
                            "detections": dets,
                        }
                    )
                    + "\n"
                )
                # An empty list here is a REAL loss, not a skipped frame, so it
                # must overwrite the held boxes rather than let them linger.
                last_dets = dets
                last_infer_frame = frame_idx

            held = held_boxes(frame_idx, last_infer_frame, last_dets, hold_frames)
            writer.write(draw_detections(frame, dets if fresh else held, model_w, fresh=fresh,
                                         frame_coords=cli.tile))
            seg_frames += 1
            frame_idx += 1
    finally:
        if writer is not None:
            close_segment(writer, seg_path, seg_frames, segment_start, time.time(), writer_fps)
        cap.release()
        log_fh.close()
        print("[stop] clean shutdown", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
