#!/usr/bin/env bash
# R6 driver. Waits for the R5 post-training eval to land, then exports and
# validates. Runs detached in its own systemd cgroup for the same reason as
# 05_sweep_watchdog.sh: chat sessions and desktop scopes both die, repeatedly.
#
# Gate: results/train_manifest.json. Written only by 07_posttrain_eval.py after
# the test-split eval completes, so it is the R5-is-done marker. Waiting on the
# manifest rather than on "=== sweep complete" also serialises R6 behind the
# eval, so the two never contend for the GPU.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ" || exit 1

LOG="$PROJ/logs/export_matrix.log"
LOCK="$PROJ/logs/.export_matrix.lock"

exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Iseconds) another export run holds the lock; exiting" >> "$LOG"; exit 0; }

say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

say "R6 waiter started (pid $$)"

while [ ! -f results/train_manifest.json ]; do
  sleep 60
done
say "R5 manifest present; R5 complete"

# Never export while a trainer or the eval still holds VRAM. Match on comm then
# confirm against /proc cmdline -- a bare `pgrep -f` also matches any shell
# whose own arguments contain the pattern, the false-positive bug documented in
# 05_sweep_watchdog.sh.
for _ in $(seq 1 60); do
  alive=0
  for p in $(pgrep -x python3 2>/dev/null); do
    if grep -qaE "bin/yolo|07_posttrain_eval" "/proc/$p/cmdline" 2>/dev/null; then alive=1; break; fi
  done
  [ "$alive" -eq 0 ] && break
  say "GPU still held by a trainer or the eval; waiting"
  sleep 20
done

say "=== leg 1: onnx + ncnn in the training venv ==="
.venv/bin/python -m src.sar.export.matrix --skip-tflite >> "$LOG" 2>&1
say "leg 1 exit $?"

if [ -x .venv-tflite/bin/python ]; then
  say "=== leg 2: tflite int8 in the isolated venv ==="
  .venv-tflite/bin/python -m src.sar.export.matrix --only-tflite >> "$LOG" 2>&1
  say "leg 2 exit $?"
else
  say "WARNING: .venv-tflite missing -- run scripts/setup_tflite_venv.sh; INT8 leg SKIPPED"
fi

say "=== parity re-validation ==="
.venv/bin/python -m src.sar.export.parity >> "$LOG" 2>&1
status=$?
say "parity exit $status"

if [ "$status" -eq 0 ]; then
  say "GATE-4 artifacts written: results/export_parity.csv"
else
  say "PARITY FAILED -- results are NOT written; needs a human"
fi
exit "$status"
