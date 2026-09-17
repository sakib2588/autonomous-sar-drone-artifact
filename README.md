# Autonomous SAR Drone — Multi-Class Detection Pipeline

Code + experiment results for a university CVPR course project (Semester 10, 2026):
a multi-class person/vehicle detector for an autonomous search-and-rescue drone,
trained on VisDrone and cross-domain evaluated on the WiSARD aerial-SAR dataset,
with quantization and Raspberry Pi 4 deployment benchmarking.

This is an **artifact repository**: it holds the code and the numeric results
(CSV/JSON) needed to understand and reproduce the experiments. It does **not**
include the raw datasets, trained model weights, or run logs — those are
multi-gigabyte and/or third-party-licensed; see [Reproducing the results](#reproducing-the-results)
below for exact instructions to regenerate them.

The project extends Elashaal, Z. A., Lamin, Y. H., Elfandi, M. K., Eluheshi, A. A.,
"Autonomous Search and Rescue Drone," *Libyan Journal of Informatics*, Vol. 01, 2024
(Faculty of Information Technology, University of Tripoli) — a single-class,
manually-piloted YOLOv4-tiny system on a Pixhawk/Raspberry Pi 4 quadcopter. This
project replaces the detector with a multi-class, INT8-quantized YOLO11n/YOLOv8n
pipeline and re-benchmarks it independently rather than reusing the original
paper's numbers.

## What's in this repo

```
src/         pipeline package: dataset tiling, quantized export, cross-domain
             eval, mission/encounter aggregation, Pi benchmarking utilities
scripts/     numbered entry points, run in order (see below)
configs/     Ultralytics data-config YAMLs (VisDrone-derived class taxonomies)
results/     every numeric result (CSV/JSON) this project produced —
             the source of truth for every number reported anywhere
docs/        dataset_provenance.md — exact source, license, and access route
             for every dataset used
specs/       original design spec for the full project (hardware + software)
deploy/      systemd services + health-check script for the Raspberry Pi 4
             deployment target
tests/       pytest suite for the src/ package
requirements.txt   exact pinned dependency freeze captured from the training run
```

Not included (regenerate locally, see below): `data/` (VisDrone + WiSARD, tens
of GB, third-party licensed), `checkpoints/` (trained weights), `logs/` (raw
training logs), `runs/` (Ultralytics scratch output).

## Setup

Python 3.10+, an NVIDIA GPU with CUDA 12.4 for training (CPU works for
inference-only steps). Raspberry Pi 4 deployment uses a **separate** TFLite-only
virtualenv (`scripts/setup_tflite_venv.sh`) — do not mix it with the training env.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` is the exact freeze from the machine the headline results were
trained on (RTX 3060 Ti, 8 GB VRAM, CUDA 12.4). If you're on CPU-only or a
different CUDA version, install `torch`/`torchvision` per the
[official PyTorch instructions](https://pytorch.org/get-started/locally/) first,
then `pip install -r requirements.txt` for the rest.

## Reproducing the results

### 1. Get the data

- **VisDrone (training domain).** Auto-downloaded by Ultralytics the first time
  `scripts/03_build_rgb_datasets.py` runs, via the `VisDrone.yaml` config it
  references — no manual step needed. Official train/val/test-dev split
  (6,471 / 548 / 1,610 images), 10 raw classes remapped to a 3-class taxonomy
  (person / vehicle / two_wheeler) in `configs/`.
- **WiSARD (cross-domain SAR evaluation).** Not auto-downloaded — free, no
  registration, MIT license, direct Google Drive link at
  `sites.google.com/uw.edu/wisard/`. Full provenance, license terms, label
  format, and known dataset quirks (mixed resolution, out-of-range boxes,
  human-free negative sequences) are documented in `docs/dataset_provenance.md`
  — read it before running the cross-domain scripts, it records exact fixes
  the eval code assumes (tile-don't-resize, sequence-level bootstrapping, box
  clipping policy).

Place the extracted data so `data/raw/VisDrone/` and the WiSARD visual subset
are reachable at the paths `scripts/03_build_rgb_datasets.py` and
`scripts/09_build_wisard_tiles.py` expect (see each script's `--help` / header
docstring for the exact flag).

### 2. Build datasets, train, evaluate

Run in this order (numbered scripts are numbered for this reason):

```bash
# Dataset construction (tiled 640x640 + full-frame ablation variant)
python scripts/03_build_rgb_datasets.py

# Training: 4 runs — YOLO11n/YOLOv8n x tiled640/full640/tiled416, seed 42
# Single-GPU, sequential. Drop batch size on OOM (see script header).
bash scripts/04_train_sweep.sh
bash scripts/06_requeue_incomplete.sh   # resumes any run a crash interrupted

# Post-training evaluation on VisDrone test-dev -> results/ablation_test.csv
python scripts/07_posttrain_eval.py

# Export to ONNX/NCNN/TFLite (FP32/FP16/INT8) + parity check vs. the FP32
# checkpoint -> results/export_manifest*.json, results/export_parity.csv
bash scripts/08_export_matrix.sh

# WiSARD cross-domain: build eval tiles, then zero-shot + bootstrap CI eval
python scripts/09_build_wisard_tiles.py
python scripts/10_bootstrap_crossdomain.py
python scripts/11_fullframe_eval.py

# WiSARD fine-tuning data budget curve (how much held-out SAR data helps)
python scripts/22_build_wisard_finetune_split.py
python scripts/39_budget_curve.py

# Figures from the above results
python scripts/23_make_figures.py
```

Everything else in `scripts/` is a specific ablation or diagnostic invoked
individually — `21_scale_sweep.py` / `26_scale_diagnostics.py` (inference
resolution vs. accuracy), `35_build_allneg_dataset.py` / `37_compare_negfrac.py`
(negative-tile ratio ablation), `43_threshold_sweep.py`, `47_external_baseline_eval.py`
(independent third-party detector baseline on the same WiSARD tiles), etc. Each
has a header docstring explaining what it does and what result file it produces.

### 3. Raspberry Pi 4 deployment benchmarking

Requires the separate TFLite venv and physical Pi 4 hardware:

```bash
bash scripts/setup_tflite_venv.sh
python scripts/14_register_tflite_variants.py
python scripts/13_pi_bench.py              # -> results/bench/*.json
python scripts/17_pi_stage_breakdown_bench.py
```

`deploy/` has the systemd services (`sar-pi-live.service`, `sar-pi-health.service`)
used to run the live detection pipeline persistently on the Pi and a health-check
script (`deploy/pi-health-log.sh`); `deploy/README.md` documents the field
deployment setup.

### 4. Tests

```bash
pytest
```

## Results summary

All numbers below are read directly from files in `results/` — see that file
for the full per-run and per-class breakdown, and `results/environment_train.txt`
for the exact training-run dependency freeze.

**VisDrone test-dev, tiled 640px, seed 42** (`results/ablation_test.csv`):

| model | variant | mAP50 (all) | mAP50 (person) | mAP50 (vehicle) |
|---|---|---|---|---|
| YOLO11n | tiled 640 | 0.581 | 0.400 | 0.862 |
| YOLOv8n | tiled 640 | 0.583 | 0.398 | 0.863 |
| YOLO11n | full-frame 640 (no tiling) | 0.415 | 0.239 | 0.728 |
| YOLO11n | tiled 416 | 0.498 | 0.290 | 0.815 |

Tiling recovers +0.166 mAP50 over full-frame at the same 640px input — small
objects are the dominant failure mode without it (`results/object_size_distribution.csv`
shows person boxes averaging 14–21 px on a side across splits).

**Zero-shot cross-domain (VisDrone-trained, evaluated on WiSARD person-only,
no fine-tuning)** (`results/cross_domain_wisard.csv`): AP50 0.244–0.261 across
model/tiling variants, against a reported WiSARD in-domain baseline of 0.892
(`results/protocol_comparison.txt`) — a large domain gap, ~11% of which is
attributable to evaluation protocol (all-tiles vs. positive-tiles-only) rather
than detection quality.

**Export quantization parity** (`results/export_parity.csv`): ONNX/NCNN FP32
export costs -0.37pp mAP50 vs. the native PyTorch checkpoint; FP16 costs a
further -0.14pp. INT8 TFLite export and its Pi 4 latency are in
`results/export_manifest_tflite.json` and `results/bench/`.

**Raspberry Pi 4 on-device latency, INT8 TFLite, 640px** (`results/bench/bench_int8_t4.json`):
median end-to-end 761 ms/frame (1.31 FPS), of which 96% is the forward pass —
measured with no accelerator, NMS decoding excluded (noted as a lower bound in
the file itself).

**WiSARD fine-tuning data budget** (`results/budget_curve.csv`): person AP50
on a held-out WiSARD split rises from ~0.10–0.14 (1 fine-tune sequence) to
~0.45 (31 sequences, the full available budget at seed 42), across 3 seeds.

## License

MIT — see `LICENSE`.

## Course context

Submitted as a CVPR (Computer Vision and Pattern Recognition) course project.
The full research program this artifact is drawn from targets a journal
submission (dataset, deployment, and paper scope beyond what a course
submission needs); this repository is the code-and-results subset relevant to
the course deliverable.
