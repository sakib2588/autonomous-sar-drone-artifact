#!/usr/bin/env bash
# Negative-fraction control. Identical to the y11n_tiled640_s42 baseline in every argument except
# the dataset, which keeps every object-free tile (9.55% negative) instead of the 17.5%-of-negatives
# sample that produced the baseline's 1.33%.
#
# Resumable by construction, same shape as scripts/24_train_p2.sh: completion guard, minimal-args
# resume (Ultralytics reads hyperparameters back out of the checkpoint and honours only a short
# override allow-list), retry cap, and a wall-clock timeout so an overrun stops with last.pt intact
# rather than being killed mid-write.
set -uo pipefail
cd "$(dirname "$0")/.."

NAME=${AN_NAME:-y11n_tiledallneg640_s42}
LOG=logs/allneg.log
WORKERS=${AN_WORKERS:-4}
EPOCHS=${AN_EPOCHS:-120}
BATCH=${AN_BATCH:-32}
RETRIES=${AN_RETRIES:-6}
MAX_HOURS=${AN_MAX_HOURS:-14}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints

LAST="checkpoints/${NAME}/weights/last.pt"
DEADLINE=$(python3 -c "print(int(${MAX_HOURS}*3600))")

if grep -q "=== ${NAME} OK" "$LOG" 2>/dev/null; then
  echo "=== skipping ${NAME}: already complete ==="; exit 0
fi
echo "=== ${NAME} starting $(date -Iseconds), budget ${MAX_HOURS}h ===" | tee -a "$LOG"

status=1
for attempt in $(seq 1 "$RETRIES"); do
  if [ -f "$LAST" ]; then
    echo "=== resuming from ${LAST}, attempt ${attempt} $(date -Iseconds) ===" | tee -a "$LOG"
    timeout --signal=INT "$DEADLINE" .venv/bin/yolo detect train \
      resume=True model="$LAST" workers="$WORKERS" >> "$LOG" 2>&1
    status=$?
  else
    echo "=== fresh start, attempt ${attempt} $(date -Iseconds) ===" | tee -a "$LOG"
    timeout --signal=INT "$DEADLINE" .venv/bin/yolo detect train \
      model=yolo11n.pt \
      data=configs/sar_rgb_tiled_allneg.yaml \
      epochs="$EPOCHS" imgsz=640 batch="$BATCH" seed=42 \
      deterministic=True cos_lr=True patience=25 \
      workers="$WORKERS" save_period=10 \
      project=checkpoints name="$NAME" exist_ok=True >> "$LOG" 2>&1
    status=$?
  fi

  [ "$status" -eq 0 ] && { echo "=== ${NAME} OK $(date -Iseconds) ===" | tee -a "$LOG"; exit 0; }
  if [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
    echo "=== hit the ${MAX_HOURS}h budget; last.pt kept, resumable ===" | tee -a "$LOG"; exit 124
  fi
  if [ ! -f "$LAST" ]; then
    echo "=== FAILED before one epoch (exit ${status}); not retrying ===" | tee -a "$LOG"; exit "$status"
  fi
  echo "=== attempt ${attempt} exited ${status}; will resume ===" | tee -a "$LOG"
done
echo "=== exhausted ${RETRIES} attempts, last exit ${status} ===" | tee -a "$LOG"
exit "$status"
