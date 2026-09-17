# Design Spec: Autonomous SAR Drone (multi-class, auto-piloted extension)

Single source of truth for this project. Extends Elashaal et al. 2024 (see
`README.md` for full citation and `notes/20260714-decision-pivot-to-sar-drone.md` for
the pivot decision record). Update this file, not tribal knowledge, when a design
decision changes.

## 1. Mission requirement

Locate a missing person as fast and reliably as possible by autonomously scanning
terrain too vast or dangerous for ground teams, streaming detections back to an
operator in real time, and returning safely before battery exhaustion.

## 2. Hardware (unchanged from the original build)

| Component | Role |
|---|---|
| Pixhawk 2.4.8 | Flight controller — 32-bit ARM Cortex M4, GPS nav, failsafe |
| Raspberry Pi 4 | Companion computer — runs the detection pipeline |
| Camera Module V2 | Live RGB video |
| S500 Frame + 950kv motors, A2217, 40A ESCs | Airframe |
| 3S 3300mAh LiPo | Power |
| M8N GPS Module | Positioning, return-to-home |
| Thermal camera | **Pending — not yet specified.** Needed for the thermal-first half of the dual-model pipeline. |

## 3. Software architecture (revised 2026-08-04, GATE-0b)

**Flutter ground-control app removed from system description.** Confirmed with user:
does not exist in this project, was never inherited (same failure pattern as flight
controller / GPS in Section 2). Do not describe it in the paper.

1. Companion Computer (Pi4) — detection pipeline (RGB branch + thermal
   characterisation branch, separate label spaces, never merged).
2. Flight Controller (Pixhawk) — design reference only; not measured (flight cut
   from scope, see `docs/plans/2026-08-03-sar-pi4-benchmark-plan.md` Section 1).

No control/data-plane software is inherited from the original build; nothing in
this project is bench-validated, not flight-validated.

## 4. Detection pipeline (upgraded — design in progress)

- Model: YOLOv4-tiny -> **YOLO11n**.
- Classes: single-class "Person" -> **multi-class**. Exact class list: **pending**
  (depends on the dataset decision in Section 7).
- Quantization: **INT8** post-training quantization for Pi4 deployment.
- **Dual-model, thermal-first** pipeline: **design pending.** Working assumption —
  thermal model runs first as a cheap candidate-region filter, RGB/YOLO11n model
  confirms and classifies; exact fusion/handoff logic not yet designed.
- **Geotag pinning**: each confirmed detection is tagged with GPS coordinates from
  the Pixhawk/M8N at detection time. Implementation (interpolation vs. nearest-fix,
  storage format) **pending**.

## 5. Autonomous mission planning (new — design in progress)

- Pattern: grid / lawnmower scan over a bounded search area.
- **Pending decisions:** area definition (operator-drawn polygon vs. fixed radius),
  altitude/overlap parameters for camera coverage, replanning behavior on detection
  (continue scan vs. loiter/investigate), integration point with MAVLink waypoint
  upload.

## 6. Failsafe (new)

- **Low-battery auto-return**: trigger threshold **pending** (needs measured
  current draw from Section 8's benchmarking, not assumed).
- On trigger: log the last detection hit (class, confidence, geotag, timestamp)
  before initiating return-to-home.
- Inherits the original design's GPS-based failsafe return via Pixhawk.

## 7. Dataset (open question)

Original: ~4,000 manually-labeled top-view images, Multi-Person Re-ID & Tracking
dataset, single-class. Multi-class detection requires either (a) a multi-class
top-view SAR-relevant dataset, or (b) extending/relabeling the original dataset.
**Source not yet decided** — do not assume or fabricate a dataset choice; resolve
and record here before training begins.

## 8. Evaluation plan

- **Baseline to compare against** (original paper, not ours to claim): 96.54%
  AP@IoU-0.50, 96.44% mean mAP (IoU 0.1-0.9), ~13s total detection time.
- **This project must measure independently, on our own Pi4 hardware:**
  - AP/mAP across IoU thresholds 0.1-0.9 for the multi-class YOLO11n INT8 model.
  - Real-time detection FPS / per-frame latency on the Pi4 (not a proxy device).
  - End-to-end mission metrics: time-to-first-detection during an autonomous scan,
    geotag accuracy, failsafe trigger correctness.
- Statistical protocol: bootstrap 95% CI, no parametric tests at n<=30, seeded and
  deterministic runs, per-class metrics alongside aggregate — per the workspace-wide
  convention (see `README.md`).
- Results are numeric source of truth in `results/`; every number in the paper must
  trace back to a file there.

## 9. Open questions / TODO

- [ ] Thermal camera hardware selection.
- [ ] Multi-class dataset source.
- [ ] Dual-model (thermal-first) fusion logic.
- [ ] Grid/lawnmower mission-planning algorithm and MAVLink integration.
- [ ] Low-battery failsafe trigger threshold (needs measured current draw).
- [ ] Geotag pinning implementation (interpolation strategy, storage format).
- [ ] Power measurement approach for Pi4 benchmarking (external meter, procurement
      status unknown for this project — was pending in the archived FER project too).

## 10. Benchmark framing (unchanged from the original paper)

This system's Wi-Fi + Raspberry Pi companion computer replaces RF telemetry + Nvidia
Jetson used in comparable prior work — a cost/complexity argument inherited from the
original paper, still valid here.
