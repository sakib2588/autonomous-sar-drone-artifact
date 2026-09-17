#!/usr/bin/env bash
# Re-attempt any sweep run that exhausted its retries, once the sweep has
# finished its first pass.
#
# Why (2026-08-06). y8n_tiled640_s42 was OOM-killed three times (02:48, 04:23,
# 05:01), burned RETRIES=2, and the sweep moved on to run 3 with y8n stuck at 21
# of 120 epochs. That is correct behaviour -- one bad run must not block the
# other three -- but it leaves the sweep incomplete, and the y8n-vs-y11n
# comparison is one of the four contrasts the paper needs.
#
# The completion guard added to run() on 2026-08-06 makes 04_train_sweep.sh
# idempotent: any run with an "=== NAME OK ===" line in logs/sweep.log is
# skipped. So simply re-invoking the sweep after it completes re-attempts
# exactly the runs that failed, resuming each from its own last.pt, and touches
# nothing that succeeded. No separate per-run logic is needed here.
#
# Deliberately conservative: MAX_PASSES caps how many times this can loop, so a
# run that fails for a structural reason (rather than a transient memory spike)
# does not re-attempt forever.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ" || exit 1

LOG="$PROJ/logs/requeue.log"
LOCK="$PROJ/logs/.requeue.lock"
MAX_PASSES=2
POLL=120

exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Iseconds) another requeue holds the lock; exiting" >> "$LOG"; exit 0; }

say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# The four run names, in sweep order.
RUNS=(y11n_tiled640_s42 y8n_tiled640_s42 y11n_full640_s42 y11n_tiled416_s42)

incomplete() {
  local missing=()
  for r in "${RUNS[@]}"; do
    grep -q "=== ${r} OK" logs/sweep.log 2>/dev/null || missing+=("$r")
  done
  echo "${missing[@]}"
}

say "requeue watcher started (pid $$, max_passes ${MAX_PASSES})"

pass=0
while [ "$pass" -lt "$MAX_PASSES" ]; do

  # Wait for the current sweep pass to finish.
  while true; do
    if ! pgrep -f "bash scripts/04_train_sweep.sh" > /dev/null 2>&1 \
       && ! pgrep -f "\.venv/bin/yolo" > /dev/null 2>&1; then
      break
    fi
    sleep "$POLL"
  done

  miss="$(incomplete)"
  if [ -z "$miss" ]; then
    say "all four runs complete; nothing to requeue; exiting"
    exit 0
  fi

  pass=$((pass + 1))
  say "pass ${pass}/${MAX_PASSES}: incomplete runs -> ${miss}"
  say "relaunching sweep (completion guard skips finished runs)"
  setsid nohup bash scripts/04_train_sweep.sh >> logs/sweep.log 2>&1 < /dev/null &
  say "sweep relaunched pid $!"
  sleep 300   # let it boot before watching for the next idle window
done

say "MAX_PASSES exhausted; still incomplete -> $(incomplete); needs a human"
