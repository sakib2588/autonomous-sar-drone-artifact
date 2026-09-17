#!/usr/bin/env bash
# Fires 07_posttrain_eval.py the moment the sweep finishes, and not before.
#
# Why a waiter and not "just run it when I notice": two Claude sessions were torn
# down mid-watch overnight on 2026-08-06/07, and the sweep itself has been killed
# twice by the OOM killer. Anything that depends on a human or a chat session
# being alive at 05:00 does not happen. This runs detached, same reasoning as
# 05_sweep_watchdog.sh.
#
# Gate: "=== sweep complete" in logs/sweep.log. That line is printed only after
# all four run() calls return, so it is the authoritative end-of-R5 marker. It is
# NOT the same as "all four succeeded" -- a run that exhausted its retries still
# lets the sweep reach that line. The eval script handles that by skipping any
# run without a best.pt and recording which runs it actually evaluated.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ" || exit 1

LOG="$PROJ/logs/posttrain_eval.log"
LOCK="$PROJ/logs/.posttrain_eval.lock"
POLL=60

# Single instance. Two concurrent vals would put two models on one GPU and race
# on results/test_eval/<name>/.
exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Iseconds) another waiter holds the lock; exiting" >> "$LOG"; exit 0; }

say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

say "post-train eval waiter started (pid $$)"

while ! grep -q "=== sweep complete" logs/sweep.log 2>/dev/null; do
  sleep "$POLL"
done

say "sweep complete detected"

# The sweep prints its completion line before the last trainer's process group
# has fully exited. Starting val while the trainer still holds VRAM risks a CUDA
# OOM on a box where the trainer peaked at 6.8 of 8.0 GiB. Wait it out.
# Match on comm then confirm against /proc cmdline -- a bare `pgrep -f` also
# matches any shell whose own arguments contain the pattern, which is the
# false-positive bug documented in 05_sweep_watchdog.sh.
for _ in $(seq 1 30); do
  alive=0
  for p in $(pgrep -x python3 2>/dev/null); do
    if grep -qa "bin/yolo" "/proc/$p/cmdline" 2>/dev/null; then alive=1; break; fi
  done
  [ "$alive" -eq 0 ] && break
  say "trainer still holding the GPU; waiting"
  sleep 20
done

say "starting test-split eval"
.venv/bin/python scripts/07_posttrain_eval.py >> "$LOG" 2>&1
status=$?
say "eval exited $status"

if [ "$status" -eq 0 ]; then
  say "wrote results/ablation_test.csv and results/train_manifest.json"
else
  say "EVAL FAILED -- results are NOT written; needs a human"
fi
exit "$status"
