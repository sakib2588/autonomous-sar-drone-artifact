#!/usr/bin/env bash
# Seed replication for the headline contrast: tiled640 vs full640, seeds 123 and 456.
#
# Why only these two models. Every number in the paper currently comes from a
# single training run, so the bootstrap CIs quantify EVALUATION uncertainty and
# say nothing about training randomness. The tiled-vs-full-frame contrast is what
# the paper rests on, so it is the one that has to move from n=1 to n=3. y8n and
# tiled416 stay at n=1 and must be labelled as such -- supporting evidence, not
# headline claims.
#
# Why this seed order. Runs are ordered seed-major (both models at 123, then both
# at 456) rather than model-major. If the deadline cuts this short, an
# interrupted run leaves a COMPLETE PAIR at one seed, which is still a usable
# paired contrast. Model-major ordering would leave two seeds of one model and
# none of the other, which is worth nothing for a paired comparison.
#
# Measured costs, from this project's own logs:
#   y11n_tiled640  1267 iters/epoch, 261.5 s/epoch  -> ~8.7 h for 120 epochs
#   y11n_full640    203 iters/epoch                 -> ~3.0 h for 120 epochs
# Four runs is ~23.4 h of compute, realistically ~3 nights with restarts.
#
# Same hyperparameters as the seed-42 runs in 04_train_sweep.sh, seed excepted.
# Do NOT change batch or imgsz here: the contrast is only interpretable if
# everything except seed and tiling is held fixed.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/yolo
WORKERS=${SWEEP_WORKERS:-2}
SAVE_PERIOD=10
RETRIES=${SWEEP_RETRIES:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs checkpoints
LOG=logs/seed_sweep.log

run() {
  local name="$1"; shift
  local last="checkpoints/${name}/weights/last.pt"
  local status=0

  # Same completion guard as 04_train_sweep.sh: a finished run's last.pt has its
  # optimizer stripped and carries epoch=-1, so the resume branch would burn
  # every retry on "training is finished, nothing to resume".
  if grep -q "=== ${name} OK" "$LOG" 2>/dev/null; then
    echo "=== skipping ${name}: already completed ==="
    return 0
  fi

  for attempt in $(seq 1 "$RETRIES"); do
    if [ -f "$last" ]; then
      local done_ep
      done_ep=$(wc -l < "checkpoints/${name}/results.csv" 2>/dev/null || echo "?")
      echo "=== resuming $name (rows: $done_ep) attempt $attempt at $(date -Iseconds) ==="
      $PY detect train resume=True model="$last" workers=$WORKERS >> "logs/${name}.log" 2>&1
    else
      echo "=== starting $name attempt $attempt at $(date -Iseconds) ==="
      $PY detect train "$@" workers=$WORKERS save_period=$SAVE_PERIOD \
        project=checkpoints name="$name" exist_ok=True > "logs/${name}.log" 2>&1
    fi
    status=$?
    if [ "$status" -eq 0 ]; then
      echo "=== $name OK at $(date -Iseconds) ==="
      return 0
    fi
    echo "=== $name exit code $status on attempt $attempt at $(date -Iseconds) ==="
    [ -f "$last" ] || { echo "=== $name failed before first checkpoint; not retrying ==="; return "$status"; }
  done
  echo "=== $name still failing after $RETRIES attempts ==="
  return "$status"
}

{
  echo "=== seed replication started $(date -Iseconds) ==="

  for SEED in 123 456; do
    run "y11n_tiled640_s${SEED}" \
      model=yolo11n.pt data=configs/sar_rgb_tiled.yaml \
      epochs=120 imgsz=640 batch=32 seed=$SEED deterministic=True cos_lr=True patience=25

    run "y11n_full640_s${SEED}" \
      model=yolo11n.pt data=configs/sar_rgb_full.yaml \
      epochs=120 imgsz=640 batch=32 seed=$SEED deterministic=True cos_lr=True patience=25
  done

  echo "=== seed replication complete $(date -Iseconds) ==="
} >> "$LOG" 2>&1
