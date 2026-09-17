#!/usr/bin/env bash
# Keeps the P2 run alive overnight. scripts/24_train_p2.sh already retries internally, but only
# while its own shell lives. This project has two recorded incidents where the Linux OOM killer
# took the trainer AND its supervising shell together (notes/20260806-bug-*), which is exactly the
# case an internal retry cannot cover. This watchdog is a separate detached process whose only job
# is to notice that nothing is running and start it again.
#
# Safe to run alongside the trainer: 24_train_p2.sh resumes from weights/last.pt and skips entirely
# once the completion marker is in logs/p2.log, so a spurious relaunch costs at most one epoch.
#
# Also handles the planned stop: the trainer wraps itself in a 15 h timeout and exits 124 with
# last.pt intact. 120 epochs needs ~20 h at the measured 603 s/epoch, so one such handover is
# EXPECTED, not a failure -- the watchdog just starts the next leg.

set -uo pipefail
cd "$(dirname "$0")/.."

NAME=${P2_NAME:-y11n_p2_tiled640_s42}
LOG=logs/p2.log
WD_LOG=logs/p2_watchdog.log
POLL=${WD_POLL:-120}
MAX_RELAUNCH=${WD_MAX_RELAUNCH:-12}
export P2_BATCH=${P2_BATCH:-8}
export P2_MAX_HOURS=${P2_MAX_HOURS:-15}

mkdir -p logs
relaunches=0
echo "=== watchdog up $(date -Iseconds), poll ${POLL}s, cap ${MAX_RELAUNCH} relaunches ===" >> "$WD_LOG"

while :; do
  if grep -q "=== ${NAME} OK" "$LOG" 2>/dev/null; then
    echo "=== ${NAME} completed; watchdog exiting $(date -Iseconds) ===" >> "$WD_LOG"
    exit 0
  fi

  if ! pgrep -f "24_train_p2.sh" > /dev/null 2>&1; then
    if [ "$relaunches" -ge "$MAX_RELAUNCH" ]; then
      echo "=== relaunch cap ${MAX_RELAUNCH} reached; giving up $(date -Iseconds) ===" >> "$WD_LOG"
      exit 1
    fi
    relaunches=$((relaunches + 1))
    ep=$(( $(wc -l < "checkpoints/${NAME}/results.csv" 2>/dev/null || echo 1) - 1 ))
    echo "=== trainer gone at $(date -Iseconds); relaunch ${relaunches}/${MAX_RELAUNCH} from epoch ${ep} ===" >> "$WD_LOG"
    setsid nohup bash scripts/24_train_p2.sh >> logs/p2_relaunch.out 2>&1 < /dev/null &
    disown 2>/dev/null || true
  fi

  sleep "$POLL"
done
