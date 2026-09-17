# Pi 4 field deployment -- get it ready before it goes on the frame

Scope: this is the physical build's live pipeline (webcam record + onboard
detection, autostart on power-up). It is separate from the paper's R7 bench
harness (`scripts/13_pi_bench.py`) -- that measures latency against a folder
of test tiles; this runs continuously against a real camera. Do the power
check below BEFORE the Pi is mounted where you can't easily reach a
multimeter.

## 1. Power -- do this before any wiring near the frame

The UBEC bought for this is rated 5A+, dedicated to the Pi (not the ESCs'
built-in BEC) -- confirmed in `project_sar_drone_hardware_procurement`
memory. That is comfortable headroom: a Pi 4 (4GB) under full 4-thread
inference plus a USB webcam plus WiFi draws roughly 1.2-1.5 A sustained, well
under a 5 A rail. Headroom is not the risk here -- **wiring mistakes and a
noisy/out-of-spec rail are.**

1. **Never connect the LiPo (11.1 V nominal, up to 12.6 V full) directly to
   the Pi**, not even for a second. The Pi's 5V rail has no upstream
   regulation of its own on the GPIO header -- 12 V in fries the board
   immediately. The UBEC is the ONLY thing that goes between battery and Pi.
2. **Measure the UBEC's actual output voltage under load before it ever
   touches the Pi.** Target 5.05-5.20 V. Below ~4.75 V at the Pi under load,
   the Pi under-volts and throttles (this is exactly what
   `src/sar/bench/sysmon.py`'s `under_voltage` bit exists to catch) --
   silent, not a crash, and it will quietly cap inference speed in flight.
   Load the UBEC while measuring (e.g. arm the ESCs at idle) since ESC
   switching noise is when a marginal BEC sags or gets ripple on the rail.
3. **Common ground.** UBEC ground, Pi ground, and flight-controller ground
   must all tie to the same reference. A floating or separate ground between
   the companion computer and the FC is a classic intermittent-reset cause
   that's hard to diagnose after it's buried in the frame.
4. **Power the Pi via the GPIO header (pin 4 = 5V, pin 6 = GND), not the
   USB-C port**, for the permanent flight connection -- the USB-C connector
   isn't rated for the vibration and will work loose. But the GPIO path
   bypasses the Pi's onboard USB-C polyfuse, so **put an inline fuse (2.5-3A
   fast-blow, or a small polyfuse) in the UBEC-to-GPIO 5V line.** Without it,
   a wiring short during assembly has nothing standing between it and the
   board.
5. **Verify clean power under real load with the tooling that already
   exists**, before final mount:
   ```bash
   .venv-tflite/bin/python scripts/13_pi_bench.py \
       --model checkpoints/y11n_tiled640_s42/weights/best_saved_model/best_full_integer_quant.tflite \
       --tiles data/processed/sar_rgb_tiled/images/val --threads 4 --frames 300
   ```
   This refuses to report numbers and prints a warning if `get_throttled` is
   dirty before the run, and flags the run for discard if it throttled
   during. Same check, reused rather than reinvented -- if this passes clean
   on the bench, the power path is validated before flight power (battery +
   UBEC) replaces bench power (wall adapter).

## 2. Camera -- confirm it enumerates before writing any code around it

The stripped/bare webcam is still a standard UVC device once it enumerates;
no special driver needed.

```bash
ls /dev/video*                 # expect /dev/video0 (maybe more, UVC often
                                # exposes a metadata node too -- video0 is
                                # usually the actual capture device)
v4l2-ctl --list-formats-ext -d /dev/video0   # confirm MJPG is offered; the
                                              # live script requests MJPG
                                              # because YUYV at 720p
                                              # saturates USB2 on the Pi 4
```
If `v4l2-ctl` is missing: `sudo apt install v4l-utils`.

## 3. Software setup on the Pi (OS is already flashed -- this is what's left)

```bash
# on the Pi, headless over SSH
mkdir -p ~/sar-pi-live/models
python3 -m venv ~/sar-pi-live/.venv-tflite
~/sar-pi-live/.venv-tflite/bin/pip install --upgrade pip
~/sar-pi-live/.venv-tflite/bin/pip install tflite-runtime opencv-python-headless numpy
```
`opencv-python-headless`, not `opencv-python` -- the Pi has no display server
running under systemd, and the full package pulls in GTK/Qt bindings that
will fail to import (or just waste flash space) headless.

From the dev machine, copy the four Python files the live pipeline needs
(mirrors how `13_pi_bench.py` already gets deployed -- flat, no package
structure, because the Pi has no repo checkout). **`tiling.py` is not
optional and does not live under `bench/`**: `15_pi_live.py` does
`from tiling import generate_tiles, merge_tile_detections`, so omitting it
raises `ImportError` at startup, before the camera is ever opened.

```bash
scp scripts/15_pi_live.py \
    src/sar/bench/decode.py \
    src/sar/bench/sysmon.py \
    src/sar/data/tiling.py \
    pi@<pi-ip>:~/sar-pi-live/
scp checkpoints/y11n_tiled416_s42/weights/best_saved_model/best_float32.tflite \
    pi@<pi-ip>:~/sar-pi-live/models/tiled416_float32.tflite
```
The FP32 export, not `best_full_integer_quant.tflite`, is the deployment
choice as of 2026-08-25. Full integer quantisation costs 12.8 pp mAP50 and
39% of small-object AP, and small objects are the entire mission. It buys
nothing measurable: inference goes 645 to 772 ms, but the end-to-end look
rate is indistinguishable, because capture, tiling and encode set the pace
and float gets the XNNPACK delegate that int8 does not. `decode.py` handles
both, but note the trap it guards: a tensor that was never quantised reports
`quantization = (0.0, 0)`, and without the `scale == 0` check the affine
dequantisation multiplies every output by zero, giving no detections, no
exception, and a perfectly normal-looking recording. Reasoning in
`notes/20260825-decision-pi-fp32-switch-and-wifi-powersave.md`. The
**416** checkpoint, not 640, is the deployment model: it measures lower
per-frame AP50 than 640 (0.2136 vs 0.2426) but runs the full tile set 3.4x
more often per unit flight time, and encounter-level detection probability
at real drone speed is dominated by attempt count, not per-attempt AP.
Measured operating point, cost table, and the recall-vs-speed math are in
`notes/20260823-decision-moving-target-capture-fixes.md` (Finding 5). Do
not re-pick the 640 model off raw ms/frame alone -- that number ignores
how few attempts per ground point it buys in the air.

## 4. Manual smoke test before enabling the service

```bash
cd ~/sar-pi-live
.venv-tflite/bin/python 15_pi_live.py \
    --model models/tiled416_float32.tflite \
    --camera /dev/video0 \
    --width 640 --height 480 \
    --tile --tile-overlap 0.2 \
    --conf-thres 0.10 \
    --writer-fps 4.9 --calibrate-seconds 0 \
    --out-dir /home/pi/sar_recordings \
    --segment-seconds 30
```
Flags must match `deploy/sar-pi-live.service` -- this is the smoke test for
what the service will actually run, not a generic connectivity check.

Let it run 60-90 s, Ctrl-C, then check:
- `/home/pi/sar_recordings/flight_*.mp4` plays and shows green detection
  boxes roughly where objects actually are.
- `/home/pi/sar_recordings/detections.jsonl` has one line per processed
  frame with non-empty `infer_ms` and a `detections` list.
- The `.timing.json` sidecar's `true_fps` lands near **4.9**. Do not compare
  it against 1.33: that figure is the *inference* rate (one full 4-tile set
  per `--infer-every-n` frames), while `true_fps` counts every frame written
  to the mp4. At `--infer-every-n 5` the two differ by exactly 5x, so a
  correct build reads about 4.9 written and about 0.98 inferences per second.
  Measured on device 2026-08-28 across three consecutive segments: 4.666,
  4.879, 4.762.
- Expect roughly **one false positive per second** over empty ground. At
  `--conf-thres 0.10` the corpus sweep in `results/threshold_sweep.txt` gives
  0.259 FP per tile, which over 4 tiles at about 1 inference per second is
  about 1 FP/s. That is the chosen operating point, not a fault: `conf` is
  recorded per detection, so the threshold can be raised offline, and the
  same sweep shows 0.10 recovers 0.3865 of annotated people against 0.3009
  at 0.25.

If boxes appear but are all pinned near one corner: that's the box-channel
normalisation bug `decode.py`'s docstring warns about, not a camera problem
-- check `img_size` is being passed as the model's actual input width, and
that `--tile` is set (untiled output is in model-input space and needs
different scaling than tiled full-frame-pixel output).

## 5. Autostart on power-up

```bash
sudo cp deploy/sar-pi-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sar-pi-live.service
sudo systemctl status sar-pi-live.service     # confirm active (running)
journalctl -u sar-pi-live.service -f          # live log while testing
```
Adjust the `--model`/`--camera` paths in `deploy/sar-pi-live.service` first if
your `scp` targets differed from Section 3 above.

**Only after this whole checklist passes clean should the Pi go on the
frame.** Once mounted, re-run Section 1 step 5 in place (battery + UBEC, not
a bench wall adapter) before the first flight -- a supply that was clean on
the bench is not guaranteed clean once ESCs are drawing real current under
the same rail.
