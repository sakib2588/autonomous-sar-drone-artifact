#!/usr/bin/env bash
# P2-head training run. Adds a stride-4 detection level to YOLO11n and retrains on the
# same tiled VisDrone set as the four baseline runs, so the P2 row in the paper is a
# like-for-like comparison.
#
# WHY
#   At 640 input the stock model's finest level is P3/8, so an 8x8 px person collapses
#   to one feature cell. VisDrone train persons are mean 20.88 px (9.7 percent under
#   8 px); the test split is mean 14.68 px (31.3 percent under 8 px). Chen et al. 2026
#   Table 2 report a P2 level worth mAP50 0.786 -> 0.849 at unchanged cost.
#
# THE COMPARISON IS FAIR, DELIBERATELY
#   pretrained=yolo11n.pt   The four baselines started from COCO weights. Handing P2 a
#                           bare yaml would train it from random init and it would lose
#                           on initialisation alone. default.yaml types `pretrained` as
#                           (bool | str) and engine/trainer.py:677 loads from the path.
#                           The YOLO11 backbone is byte-identical between stock and P2,
#                           so layers 0-10 transfer whole; the new P2 branch and the
#                           shifted downstream head init randomly.
#   batch=16 (not 32)       The P2 level is 160x160 at 640 input, 4x the cells of P3,
#                           roughly +1.6 GB activation under AMP. The plain run measured
#                           4.2 GB at batch 32 on an 8 GB card, so batch 32 here is
#                           marginal. This does NOT change the optimisation: Ultralytics
#                           accumulates to nbs=64, so accumulate = round(64/batch) makes
#                           the effective batch 64 at both 16 and 32. Only BatchNorm
#                           statistics differ -- report that in Methods, it is a real
#                           hyperparameter change, not a silent tweak.
#
# GRAPH VERIFIED BEFORE LAUNCH (a misrouted Concat builds fine and trains quietly worse):
#   stride == [4, 8, 16, 32], Detect.nl == 4, Detect.f == [19, 22, 25, 28],
#   head feature maps at 640 == [(160,160), (80,80), (40,40), (20,20)],
#   params 2,667,084 vs stock 2,624,080 (+1.6 percent).
#
# TIME BUDGET, NOT EPOCH COUNT
#   The 120-epoch target rests on extrapolating one measurement (5.8 it/s at batch 32 on
#   a different dataset). If the P2 branch costs more than assumed, 120 epochs overruns.
#   MAX_HOURS wraps the trainer in `timeout`, so the run stops cleanly with last.pt on
#   disk and is resumable. Report whatever epoch actually completed, and name it.
#
# RESUME
#   Re-running this script picks up from weights/last.pt. On resume Ultralytics reads
#   every hyperparameter back out of the checkpoint; only a short allow-list of CLI
#   overrides is honoured, so the resume invocation passes `workers` and nothing else.
#   A resume needs at least one COMPLETED epoch on disk -- do not kill a fresh run
#   inside the first few minutes.
#   To force a genuine from-scratch rerun, delete checkpoints/<name>/ first. Nothing
#   here overwrites trained weights silently.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/yolo
NAME=${P2_NAME:-y11n_p2_tiled640_s42}
LOG=logs/p2.log
WORKERS=${P2_WORKERS:-2}
EPOCHS=${P2_EPOCHS:-120}
BATCH=${P2_BATCH:-16}
SAVE_PERIOD=${P2_SAVE_PERIOD:-5}
RETRIES=${P2_RETRIES:-6}
MAX_HOURS=${P2_MAX_HOURS:-15}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints

LAST="checkpoints/${NAME}/weights/last.pt"
DEADLINE_SECS=$(python3 -c "print(int(${MAX_HOURS}*3600))")

if grep -q "=== ${NAME} OK" "$LOG" 2>/dev/null; then
  echo "=== skipping ${NAME}: already completed (${LOG}) ==="
  exit 0
fi

echo "=== ${NAME} starting at $(date -Iseconds), budget ${MAX_HOURS}h ===" | tee -a "$LOG"

status=1
for attempt in $(seq 1 "$RETRIES"); do
  if [ -f "$LAST" ]; then
    rows=$(wc -l < "checkpoints/${NAME}/results.csv" 2>/dev/null || echo "?")
    echo "=== resuming ${NAME} from ${LAST} (results.csv rows: ${rows}) attempt ${attempt} at $(date -Iseconds) ===" | tee -a "$LOG"
    timeout --signal=INT "${DEADLINE_SECS}" "$PY" detect train \
      resume=True model="$LAST" workers="$WORKERS" >> "$LOG" 2>&1
    status=$?
  else
    echo "=== fresh start ${NAME} attempt ${attempt} at $(date -Iseconds) ===" | tee -a "$LOG"
    timeout --signal=INT "${DEADLINE_SECS}" "$PY" detect train \
      model=configs/yolo11n-p2.yaml \
      pretrained=yolo11n.pt \
      data=configs/sar_rgb_tiled.yaml \
      epochs="$EPOCHS" imgsz=640 batch="$BATCH" seed=42 \
      deterministic=True cos_lr=True patience=25 \
      workers="$WORKERS" save_period="$SAVE_PERIOD" \
      project=checkpoints name="$NAME" exist_ok=True >> "$LOG" 2>&1
    status=$?
  fi

  if [ "$status" -eq 0 ]; then
    echo "=== ${NAME} OK at $(date -Iseconds) ===" | tee -a "$LOG"
    exit 0
  fi

  # 124 is GNU timeout's "deadline reached". That is a planned stop, not a crash:
  # last.pt is on disk, the run is resumable, and retrying would just burn the
  # budget again immediately.
  if [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
    echo "=== ${NAME} hit the ${MAX_HOURS}h budget at $(date -Iseconds); last.pt kept, resumable ===" | tee -a "$LOG"
    exit 124
  fi

  # No last.pt after a failed attempt means setup broke (bad yaml, missing data),
  # not a transient crash. Retrying that just wastes hours.
  if [ ! -f "$LAST" ]; then
    echo "=== ${NAME} FAILED before completing one epoch (exit ${status}); not retrying ===" | tee -a "$LOG"
    exit "$status"
  fi

  echo "=== ${NAME} attempt ${attempt} exited ${status}; will resume ===" | tee -a "$LOG"
done

echo "=== ${NAME} exhausted ${RETRIES} attempts, last exit ${status} ===" | tee -a "$LOG"
exit "$status"
