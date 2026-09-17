#!/usr/bin/env bash
# Outer supervisor for scripts/04_train_sweep.sh.
#
# Why this exists (2026-08-06). The sweep script already retries an individual
# run that dies. What it cannot do is restart *itself*. On 2026-08-06 at 02:48
# the OOM killer took the trainer, systemd tore down the surrounding desktop app
# scope, and the sweep script died with it -- so RETRIES=2 never executed and the
# GPU sat idle for 46 minutes unnoticed. See
# notes/20260806-bug-oom-killer-killed-run2.md.
#
# This watchdog runs detached (setsid) outside any desktop scope, polls once a
# minute, and relaunches the sweep if BOTH the sweep script and every trainer are
# gone. Each run resumes from its own last.pt, so a relaunch costs at most the
# in-flight epoch.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ" || exit 1

LOG="$PROJ/logs/watchdog.log"
LOCK="$PROJ/logs/.watchdog.lock"
POLL=60           # seconds between checks
MISS_LIMIT=3      # consecutive dead polls before relaunching (3 min of grace)
MAX_RELAUNCH=20   # backstop against a crash loop hammering the GPU all night

# Single instance only. Two watchdogs could launch two sweeps, which would put
# two trainers on one GPU and have them write the same checkpoint files.
exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Iseconds) another watchdog holds the lock; exiting" >> "$LOG"; exit 0; }

say() { echo "$(date -Iseconds) $*" >> "$LOG"; }

say "watchdog started (pid $$, poll ${POLL}s, miss_limit ${MISS_LIMIT})"

miss=0
relaunches=0

while true; do
  # Terminal state: the sweep prints this only after all four runs return.
  if grep -q "=== sweep complete" logs/sweep.log 2>/dev/null; then
    say "sweep complete detected; watchdog exiting"
    exit 0
  fi

  # A gdown/rsync/unzip elsewhere is exactly what caused the original OOM, so
  # never start a trainer while one is running -- wait it out instead.
  if pgrep -x gdown > /dev/null 2>&1; then
    say "heavy IO (gdown) active; holding off"
    miss=0
    sleep "$POLL"
    continue
  fi

  # Match on cmdline, NOT process name. The sweep invokes the trainer as
  # `.venv/bin/python3 .venv/bin/yolo ...`, so its comm is "python3" and a
  # `pgrep -x yolo` never matches -- the first version of this script had that
  # bug and would have judged a live trainer dead. `ps` also truncates its args
  # column to terminal width, so grepping ps output is unreliable here; pgrep -f
  # reads the full /proc cmdline. This pattern does not match the watchdog's own
  # cmdline (`bash scripts/05_sweep_watchdog.sh`), so it cannot self-trigger.
  #
  # False-positive fix (2026-08-06). `pgrep -f` matches ANY process whose full
  # cmdline CONTAINS the pattern -- including an interactive shell running
  # `pgrep -f "\.venv/bin/yolo"` to check on the sweep, or any `bash -c`
  # wrapper quoting it. Observed live at 14:56: a status check from another
  # terminal reset miss back to 0 and delayed the relaunch. A false "alive" is
  # the dangerous direction; it is exactly the dead-sweep case this watchdog
  # exists to catch. So match on comm first (`pgrep -x python3` / `-x bash`
  # cannot match a `bash -c` wrapper, whose comm is still bash but whose args
  # we then check), then confirm against /proc/PID/cmdline, skipping this
  # watchdog's own PID.
  trainer_alive=0
  sweep_alive=0
  for p in $(pgrep -x python3 2>/dev/null); do
    if grep -qa "bin/yolo" "/proc/$p/cmdline" 2>/dev/null; then trainer_alive=1; break; fi
  done
  for p in $(pgrep -x bash 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    if grep -qa "04_train_sweep.sh" "/proc/$p/cmdline" 2>/dev/null; then sweep_alive=1; break; fi
  done

  if [ "$trainer_alive" -eq 1 ] || [ "$sweep_alive" -eq 1 ]; then
    # Healthy. A gap with sweep_alive=1 and trainer_alive=0 is normal between
    # runs while Ultralytics tears down and the next run boots.
    miss=0
  else
    miss=$((miss + 1))
    say "nothing alive (miss ${miss}/${MISS_LIMIT}); mem: $(free -m | awk '/^Mem:/{print $7" MB avail"}')"

    if [ "$miss" -ge "$MISS_LIMIT" ]; then
      if [ "$relaunches" -ge "$MAX_RELAUNCH" ]; then
        say "hit MAX_RELAUNCH=${MAX_RELAUNCH}; refusing to relaunch again -- needs a human"
        exit 1
      fi
      relaunches=$((relaunches + 1))
      say "RELAUNCHING sweep (relaunch ${relaunches}/${MAX_RELAUNCH})"
      setsid nohup bash scripts/04_train_sweep.sh >> logs/sweep.log 2>&1 < /dev/null &
      say "sweep relaunched pid $!"
      miss=0
      sleep 120   # let it boot before the next verdict
    fi
  fi

  sleep "$POLL"
done
