#!/usr/bin/env bash
# R5 step2: the four training runs that matter. Sequential -- single GPU,
# 8 GB VRAM, no concurrent runs. Each logs/*.log has full Ultralytics output;
# checkpoints/<name>/ has weights + results.csv. batch=32 per plan, drop to 16
# and record it if a run OOMs (BN stats shift with batch size, so it's a
# reportable hyperparameter, not a silent tweak).
#
# Memory settings (added 2026-08-05 after the first run drove the box to
# MemAvailable=265 MB with 5.3 GiB swapped and VRAM to 7.0/8.0 GiB):
#   WORKERS=4  -- 8 DataLoader workers each hold 640px mosaic buffers; on a
#                 14.5 GiB host that is the dominant host-RAM term. 4 keeps the
#                 GPU fed (measured ~4.4 it/s) without the swap thrash.
#   expandable_segments:True -- allocator returns freed blocks to a growable
#                 arena instead of fragmenting fixed pools. Mosaic gives wildly
#                 variable instance counts per batch (447..1565 observed), which
#                 is exactly the fragmentation case this flag fixes. No effect on
#                 numerics, so it does not touch reproducibility of results.
#
# Checkpointing (added 2026-08-05). Ultralytics already writes last.pt and
# best.pt at the end of every epoch, so the exposure from a crash is at most one
# epoch. What it does NOT do by default is (a) keep periodic archival snapshots
# and (b) pick itself back up. Both are added here:
#   SAVE_PERIOD=10 -- also writes weights/epoch{10,20,...}.pt. ~6 MB each for a
#                 yolo11n, so ~72 MB per run. Gives a mid-training weight set if
#                 a late epoch turns out to have overfit, and survives a corrupt
#                 last.pt.
#   auto-resume  -- run() checks for weights/last.pt before starting. If present,
#                 it resumes from it instead of restarting from epoch 0. On
#                 resume Ultralytics reads every hyperparameter back out of the
#                 checkpoint and ignores CLI args, so the resume invocation is
#                 deliberately minimal -- passing batch/imgsz there would be
#                 silently dropped and give a false sense of control.
#   RETRIES=2    -- a run that dies (OOM, driver hiccup) is retried, and because
#                 last.pt now exists the retry is a resume, not a restart.
# To force a genuine from-scratch rerun, delete that run's checkpoints/<name>/
# directory first. This is intentional: no silent overwrite of trained weights.

set -uo pipefail  # not -e: one run OOMing must not block the rest
cd "$(dirname "$0")/.."

PY=.venv/bin/yolo
WORKERS=${SWEEP_WORKERS:-4}
SAVE_PERIOD=10
# RETRIES raised 2->6 on 2026-08-06 (second OOM incident). A resume costs at
# most the in-flight epoch, and the failure mode here is a transient host-RAM
# spike, not a structural error, so a low cap just strands a run mid-sweep --
# which is exactly what happened to y8n_tiled640_s42 at 21/120 epochs.
RETRIES=${SWEEP_RETRIES:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints

run() {
  local name="$1"; shift
  local last="checkpoints/${name}/weights/last.pt"
  local status=0

  # Completion guard (added 2026-08-06, after the OOM kill described below).
  # A finished run's last.pt has had its optimizer stripped and carries
  # epoch=-1, so the resume branch would hand Ultralytics a checkpoint it
  # refuses ("training ... is finished, nothing to resume"), burning both
  # retries on every relaunch. sweep.log's "=== NAME OK ===" line is written
  # only on a clean exit, so it is the authoritative record of what is done and
  # it survives reboots. Skip on it rather than on results.csv row count, which
  # is also written mid-run.
  if grep -q "=== ${name} OK" logs/sweep.log 2>/dev/null; then
    echo "=== skipping ${name}: already completed (sweep.log) ==="
    return 0
  fi

  for attempt in $(seq 1 "$RETRIES"); do
    if [ -f "$last" ]; then
      local done_ep
      done_ep=$(wc -l < "checkpoints/${name}/results.csv" 2>/dev/null || echo "?")
      echo "=== resuming $name from $last (rows in results.csv: $done_ep) attempt $attempt at $(date -Iseconds) ==="
      # resume=True reads all args back out of the checkpoint. Ultralytics
      # 8.3.253 check_resume() (engine/trainer.py:812) does honour a short
      # allow-list of overrides on top of that -- imgsz, batch, device,
      # close_mosaic, augmentations, save_period, workers, cache, patience --
      # so `workers` IS respected here, while anything outside that list is
      # silently dropped. Pass workers only, so host-RAM pressure can be tuned
      # on a resume without checkpoint surgery. Do NOT add batch/imgsz here:
      # those are locked hyperparameters for the sweep.
      $PY detect train resume=True model="$last" workers=$WORKERS \
        >> "logs/${name}.log" 2>&1
    else
      echo "=== starting $name attempt $attempt at $(date -Iseconds) ==="
      $PY detect train "$@" workers=$WORKERS save_period=$SAVE_PERIOD \
        project=checkpoints name="$name" exist_ok=True \
        > "logs/${name}.log" 2>&1
    fi
    status=$?

    if [ "$status" -eq 0 ]; then
      echo "=== $name OK at $(date -Iseconds) ==="
      return 0
    fi
    echo "=== $name exit code $status on attempt $attempt at $(date -Iseconds) ==="
    # No checkpoint to resume from means the failure was at setup, not mid-train.
    # Retrying identically would just fail identically, so stop.
    if [ ! -f "$last" ]; then
      echo "=== $name failed before first checkpoint; not retrying ==="
      return "$status"
    fi
  done

  echo "=== $name still failing after $RETRIES attempts ==="
  return "$status"
}

run y11n_tiled640_s42 \
  model=yolo11n.pt data=configs/sar_rgb_tiled.yaml \
  epochs=120 imgsz=640 batch=32 seed=42 deterministic=True cos_lr=True patience=25

run y8n_tiled640_s42 \
  model=yolov8n.pt data=configs/sar_rgb_tiled.yaml \
  epochs=120 imgsz=640 batch=32 seed=42 deterministic=True cos_lr=True patience=25

run y11n_full640_s42 \
  model=yolo11n.pt data=configs/sar_rgb_full.yaml \
  epochs=120 imgsz=640 batch=32 seed=42 deterministic=True cos_lr=True patience=25

run y11n_tiled416_s42 \
  model=yolo11n.pt data=configs/sar_rgb_tiled.yaml \
  epochs=120 imgsz=416 batch=48 seed=42 deterministic=True cos_lr=True patience=25

echo "=== sweep complete at $(date -Iseconds) ==="
