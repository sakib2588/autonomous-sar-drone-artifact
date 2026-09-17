#!/usr/bin/env python3
"""One power-measurement trial: throttle-gated timing wrapper around either an
idle window or a live 15_pi_live.py run, printing RESET-NOW / READ-NOW cues
for the human operator reading the USB-C power tester, and writing one
per-trial JSON record. Runs ON the Pi under plain system python3 (no cv2/
tflite import here -- those only run inside the child 15_pi_live.py process).

Protocol source: docs/edge_power_measurement_methodology.md (reset-delta
method, ported from a sibling project). The tester has no host interface, so
wh_reading/time_reading_s can't be captured here -- they stay null in the
written record until filled in from what the operator reports after reading
the meter's display, following the printed RESET-NOW/READ-NOW cues.

Launches 15_pi_live.py detached (nohup + PID file, not a blocked foreground
child) -- per the same protocol doc, a held SSH connection that drops
mid-trial has, in a prior project's incident history, silently killed an
unattended multi-minute run. nohup survives that; a plain child process
would not.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def vcgencmd(*args: str) -> str:
    return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout.strip()


def read_throttle_and_temp() -> tuple[str, str]:
    return vcgencmd("get_throttled"), vcgencmd("measure_temp")


def throttled_clean(raw: str) -> bool:
    try:
        return int(raw.split("=")[1], 16) == 0
    except (IndexError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["idle", "load"], required=True)
    ap.add_argument("--duration-s", type=int, default=300,
                     help="idle: total window length; load: minimum compute time before SIGINT")
    ap.add_argument("--model", default=None, help="required for --mode load")
    ap.add_argument("--infer-every-n", type=int, default=10)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--python-bin", default="/home/pi/sar-pi-live/.venv-tflite/bin/python",
                     help="interpreter for the child 15_pi_live.py (needs cv2/tflite_runtime)")
    ap.add_argument("--script-dir", default="/home/pi/sar-pi-live")
    ap.add_argument("--run-out-dir", default="/home/pi/sar_recordings/power_trials")
    ap.add_argument("--out", required=True, help="path to write this trial's JSON record")
    cli = ap.parse_args()

    if cli.mode == "load" and not cli.model:
        print("[error] --model required for --mode load", flush=True)
        return 1

    throttle_before, temp_start = read_throttle_and_temp()
    if not throttled_clean(throttle_before):
        print(f"[abort] Pi not power-clean before trial: {throttle_before}", flush=True)
        return 2

    t_start = time.time()
    print(f"[RESET-NOW] t={t_start:.0f} -- reset the meter's Wh/mAh/Time counters now", flush=True)

    child_pid = None
    data_dir = None
    if cli.mode == "idle":
        time.sleep(cli.duration_s)
    else:
        out_dir = Path(cli.run_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(t_start)
        log_path = out_dir / f"trial_{stamp}.log"
        pid_path = out_dir / f"trial_{stamp}.pid"
        data_dir = out_dir / f"data_{stamp}"
        cmd = (
            f"nohup {cli.python_bin} {cli.script_dir}/15_pi_live.py "
            f"--model {cli.model} --camera {cli.camera} --threads {cli.threads} "
            f"--infer-every-n {cli.infer_every_n} --out-dir {data_dir} "
            f"< /dev/null > {log_path} 2>&1 & echo $! > {pid_path}"
        )
        subprocess.run(["bash", "-c", cmd], check=True)
        time.sleep(1.0)
        child_pid = int(pid_path.read_text().strip())

        while time.time() - t_start < cli.duration_s:
            time.sleep(2)

        os.kill(child_pid, signal.SIGINT)
        for _ in range(50):  # up to 10s for clean shutdown
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)

    t_end = time.time()
    print(f"[READ-NOW] t={t_end:.0f} -- read Wh and Time off the meter now", flush=True)

    throttle_after, temp_end = read_throttle_and_temp()
    clean = throttled_clean(throttle_before) and throttled_clean(throttle_after)

    median_infer_ms = None
    if cli.mode == "load" and data_dir is not None:
        det_path = data_dir / "detections.jsonl"
        if det_path.exists():
            vals = sorted(json.loads(l)["infer_ms"] for l in det_path.open())
            if vals:
                median_infer_ms = vals[len(vals) // 2]

    record = {
        "mode": cli.mode,
        "t_start": t_start,
        "t_end": t_end,
        "script_wall_time_s": round(t_end - t_start, 1),
        "model": cli.model,
        "infer_every_n": cli.infer_every_n if cli.mode == "load" else None,
        "threads": cli.threads,
        "camera_attached": True,
        "input_source": "live_camera" if cli.mode == "load" else None,
        "runtime": "tflite" if cli.mode == "load" else None,
        "median_infer_ms": median_infer_ms,
        "temp_start_c": temp_start,
        "temp_end_c": temp_end,
        "throttle_before": throttle_before,
        "throttle_after": throttle_after,
        "throttle_clean": clean,
        "wh_reading": None,
        "time_reading_s": None,
        "overhead_fraction": None,
        "discarded": not clean,
        "discard_reason": None if clean else f"throttled: before={throttle_before} after={throttle_after}",
    }

    Path(cli.out).write_text(json.dumps(record, indent=2))
    print(f"[done] wrote {cli.out}", flush=True)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
