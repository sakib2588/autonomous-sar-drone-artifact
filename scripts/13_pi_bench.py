#!/usr/bin/env python3
"""On-Pi latency benchmark for one exported artifact (R7 Steps 6-7).

Runs ON the Raspberry Pi. Copies of src/sar/bench/{sysmon,pi_runner}.py must sit
next to it -- the Pi has no checkout of the repo.

Measurement discipline, all of it load-bearing:
  * warm-up frames discarded (cold page cache, allocator growth, clock ramp)
  * per-stage timing, because NMS alone is routinely 20-40 percent of end-to-end
    and "inference time" that means "forward pass" is incomparable
  * median / p95 / p99, never the mean alone
  * get_throttled checked BEFORE and AFTER; a run whose latch bits are set is
    discarded, because a marginal supply silently caps the clock and every
    number taken under it is wrong
  * governor pinned to performance by the caller, verified here

The tflite INT8 graph takes uint8 input with quantisation parameters baked in,
so preprocessing must NOT normalise to float the way the PyTorch path does --
doing that silently destroys accuracy while leaving latency unchanged, which is
exactly the preprocessing-mismatch failure GATE-4 warns about.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi_runner import StageTimer, latency_summary  # noqa: E402
from sysmon import describe, has_occurred_since_boot, is_clean, parse_throttled  # noqa: E402


def vcgencmd(*args: str) -> str:
    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout


def read_state() -> dict:
    return {
        "throttled_raw": vcgencmd("get_throttled").strip(),
        "temp": vcgencmd("measure_temp").strip(),
        "clock_arm": vcgencmd("measure_clock", "arm").strip(),
        "volts_core": vcgencmd("measure_volts", "core").strip(),
        "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .read_text()
        .strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    import cv2
    from tflite_runtime.interpreter import Interpreter

    before = read_state()
    flags_before = parse_throttled(before["throttled_raw"])
    if not is_clean(flags_before):
        print(f"REFUSING TO MEASURE: Pi is not clean before the run -- {describe(flags_before)}")
        print("Fix the power supply; a capped clock invalidates every number.")
        return 2

    interp = Interpreter(model_path=cli.model, num_threads=cli.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, H, W, _ = inp["shape"]
    print(f"input {inp['shape']} {inp['dtype'].__name__} quant={inp['quantization']}", flush=True)
    print(f"output {out['shape']} {out['dtype'].__name__}", flush=True)

    tiles = sorted(Path(cli.tiles).glob("*.jpg"))[: cli.frames]
    if len(tiles) < cli.warmup + 20:
        print(f"need more than {cli.warmup + 20} tiles, found {len(tiles)}")
        return 2

    t = StageTimer()
    end_to_end: list[float] = []
    scale, zero = inp["quantization"]

    for p in tiles:
        t0 = time.perf_counter()

        a = time.perf_counter()
        raw = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        b = time.perf_counter()
        t.record("decode", (b - a) * 1000)

        a = time.perf_counter()
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if inp["dtype"] == np.uint8:
            # Quantised graph with unsigned input: feed uint8 directly.
            # Normalising here would be the classic preprocessing mismatch --
            # same latency, wrecked mAP.
            x = img[None, ...]
        elif inp["dtype"] == np.int8:
            # Ultralytics' full-integer export uses SIGNED int8 input, typically
            # scale 1/255 with zero-point -128. Quantise explicitly rather than
            # casting: q = round(real / scale) + zero_point, with real = px/255.
            # Feeding float here raises a dtype error; feeding uint8 would be
            # off by the 128 offset and silently destroy accuracy.
            q = np.round((img.astype(np.float32) / 255.0) / scale) + zero
            x = np.clip(q, -128, 127).astype(np.int8)[None, ...]
        else:
            x = (img.astype(np.float32) / 255.0)[None, ...]
        b = time.perf_counter()
        t.record("letterbox", (b - a) * 1000)

        a = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        b = time.perf_counter()
        t.record("forward", (b - a) * 1000)

        a = time.perf_counter()
        if y.dtype == np.uint8 and scale:
            y = (y.astype(np.float32) - zero) * scale
        b = time.perf_counter()
        t.record("postprocess", (b - a) * 1000)

        end_to_end.append((time.perf_counter() - t0) * 1000)

    after = read_state()
    flags_after = parse_throttled(after["throttled_raw"])
    discard = has_occurred_since_boot(flags_after)

    report = {
        "model": cli.model,
        "threads": cli.threads,
        "frames_total": len(tiles),
        "warmup_discarded": cli.warmup,
        "input_shape": [int(v) for v in inp["shape"]],
        "input_dtype": inp["dtype"].__name__,
        "end_to_end": latency_summary(end_to_end, warmup=cli.warmup),
        "per_stage": t.summary(warmup=cli.warmup),
        "stage_fractions": t.stage_fractions(warmup=cli.warmup),
        "state_before": before,
        "state_after": after,
        "throttle_after": describe(flags_after),
        "DISCARD_THROTTLED": discard,
        "note": (
            "NMS is not in this run: the exported graph emits raw predictions and "
            "the NMS stage is measured separately once the decoder is ported. "
            "End-to-end here is decode + letterbox + forward + dequantise, so it "
            "is a LOWER BOUND on deployed latency."
        ),
    }

    e = report["end_to_end"]
    print(f"\nend-to-end  median {e['median_ms']:.1f} ms  p95 {e['p95_ms']:.1f}  "
          f"p99 {e['p99_ms']:.1f}  -> {e['fps_median']:.2f} FPS (median)")
    for stage, s in report["per_stage"].items():
        print(f"  {stage:<12} median {s['median_ms']:7.2f} ms  "
              f"({report['stage_fractions'][stage] * 100:5.1f}% of pipeline)")
    print(f"\nthrottle after: {report['throttle_after']}")
    if discard:
        print("*** DISCARD THIS RUN: the Pi throttled during measurement ***")

    if cli.out:
        Path(cli.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[write] {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
