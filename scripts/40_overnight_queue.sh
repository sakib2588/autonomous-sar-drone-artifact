#!/usr/bin/env bash
# One-owner overnight queue for this PC. Survives the machine dying.
#
# WHY THIS EXISTS
#   On 2026-08-25 this box hard-crashed three times (06:04, 10:01, 18:06) while
#   training. Every restart was manual and each cost hours of wall-clock because
#   nobody was awake. None of the crashes left an OOM, a panic, or a shutdown
#   sequence, so they are not something a training script can prevent -- the only
#   useful response is to make restarting automatic.
#
#   Run this from cron (@reboot AND every 10 minutes). It works out what is
#   unfinished and continues from there. Safe to invoke at any time.
#
# THE ORDER, AND WHY
#   1. control        finish y11n_tiledallneg640_s42 to 120 epochs. The paper's
#                     baseline is epoch 120, so the control must be too or the
#                     comparison measures training length, not negative fraction.
#   2. earlystop s42  cross-domain accuracy against epoch, from checkpoints that
#                     already exist. Inference only, no training. Potentially the
#                     most valuable result here and the cheapest.
#   3. budget curve   how many labelled target sequences are actually needed.
#                     Replaces a Conclusion sentence extrapolated from n=1.
#   4. earlystop s123 second seed, so step 2 is not one trajectory.
#
#   Steps 2-4 could run in either order. Step 1 is first because it is the only
#   one with a deadline attached, and step 2 precedes 3 because it is cheaper and
#   needs no training.
#
# COOPERATIVE, NOT COMPETITIVE
#   Four other projects on this machine run cron guardians that relaunch their own
#   training (see the crontab). This script does NOT kill them and does not disable
#   them -- they belong to other deadlines. It checks whether a foreign trainer holds
#   the GPU and backs off for this tick instead. Two owners racing for one 8 GB card
#   is how this box OOMs; the user's own _resume_watchdog.sh says so.
#
# CONCURRENCY
#   flock keeps one instance. A ten-minute cron against multi-hour jobs would
#   otherwise stack dozens of trainers, which is a far worse failure than a crash.

set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ" || exit 1

# Overridable purely so the stage branching can be tested without spending GPU
# time or disturbing real queue state. Cron sets none of these and gets the
# production values.
PY="${QUEUE_PY:-$PROJ/.venv/bin/python}"
LOG="${QUEUE_LOG:-$PROJ/logs/queue.log}"
MARK="${QUEUE_MARK:-$PROJ/results/queue}"
LOCK="${QUEUE_LOCK:-/tmp/sar_overnight_queue.lock}"
mkdir -p "$MARK" "$PROJ/logs"

exec 9>"$LOCK"
flock -n 9 || exit 0          # another tick is still working; nothing to do

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# --- is a trainer from ANOTHER project holding the GPU? -----------------------
# Match on the process's own command line, not on GPU memory: a desktop compositor
# using 300 MB is not a trainer, and a paused trainer holding memory is not running.
#
# Two rules, and the second one was learned the hard way. First, anything running
# out of THIS project is our own work. Second, a foreign process only counts as a
# trainer if it is actually a Python interpreter: an earlier version matched on
# project-name substrings anywhere in the command line, and `cosmic-files` browsing
# a folder called "NATURAL LANGUAGE PROCESSING" matched it. A file-manager window
# would have blocked this queue for the entire night.
foreign_trainer_running() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 1
  for p in $pids; do
    local cmd
    cmd=$(ps -o cmd= -p "$p" 2>/dev/null) || continue
    case "$cmd" in
      *"$PROJ"*) continue ;;                       # our own work, fine
    esac
    case "$cmd" in
      *python*|*yolo*) return 0 ;;                 # someone else's trainer
    esac
  done
  return 1
}

# --- stage predicates ---------------------------------------------------------
epochs_done() {  # epochs_done <run> -> rows in results.csv minus header
  local f="$PROJ/checkpoints/$1/results.csv"
  [ -f "$f" ] || { echo 0; return; }
  local n; n=$(($(wc -l < "$f") - 1)); [ "$n" -lt 0 ] && n=0; echo "$n"
}

curve_rows() { # curve_rows <csv> -> data rows
  [ -f "$1" ] || { echo 0; return; }
  local n; n=$(($(wc -l < "$1") - 1)); [ "$n" -lt 0 ] && n=0; echo "$n"
}

# ------------------------------------------------------------------------------
if foreign_trainer_running; then
  say "another project's trainer holds the GPU; backing off this tick"
  exit 0
fi

# --- 1. control to 120 --------------------------------------------------------
if [ ! -f "$MARK/control.done" ]; then
  n=$(epochs_done y11n_tiledallneg640_s42)
  if [ "$n" -ge 120 ]; then
    touch "$MARK/control.done"; say "control complete at ${n} epochs"
  else
    if pgrep -f "yolo detect train" >/dev/null 2>&1; then
      say "control still training (${n}/120)"; exit 0
    fi
    say "control at ${n}/120 and not running -- resuming"
    bash "$PROJ/scripts/36_train_allneg.sh" >> "$PROJ/logs/allneg_launch.log" 2>&1
    n=$(epochs_done y11n_tiledallneg640_s42)
    [ "$n" -ge 120 ] && touch "$MARK/control.done" && say "control reached 120"
    exit 0
  fi
fi

# --- 1b. score the finished control -------------------------------------------
# Training the control to 120 without scoring it would leave the actual experiment
# unanswered, which an earlier version of this file did. Both sides use best.pt so
# the comparison matches the paper, whose baseline row is also best.pt. The
# baseline's cache already exists from the published run and is reused, not rebuilt.
if [ ! -f "$MARK/control_scored.done" ]; then
  say "scoring the finished control cross-domain"
  CB="$PROJ/checkpoints/_ctrl_final/weights"
  mkdir -p "$CB"
  # Always refresh the staged copy. Guarding this on "already exists" would score a
  # stale checkpoint forever if the staging ever happened before training finished,
  # and the cache is only reused when it is newer than the checkpoint it came from.
  cp -p "$PROJ/checkpoints/y11n_tiledallneg640_s42/weights/best.pt" "$CB/best.pt"
  CACHE="$PROJ/results/cache/dets__ctrl_final_sz640.npz"
  if [ ! -f "$CACHE" ] || [ "$CB/best.pt" -nt "$CACHE" ]; then
    "$PY" "$PROJ/scripts/29_dump_detections.py" --run _ctrl_final --imgsz 640 --batch 4 \
        >> "$PROJ/logs/ctrl_final_eval.log" 2>&1
  fi
  if [ -f "$PROJ/results/cache/dets__ctrl_final_sz640.npz" ]; then
    "$PY" "$PROJ/scripts/37_compare_negfrac.py" --base y11n_tiled640_s42 --ctrl _ctrl_final \
        > "$PROJ/results/negfrac_control_epoch120.txt" 2>&1
    touch "$MARK/control_scored.done"
    say "control scored; verdict in results/negfrac_control_epoch120.txt"
  else
    say "control scoring incomplete; will retry next tick"
  fi
  exit 0
fi

# --- 2. early-stopping curve, seed 42 ----------------------------------------
if [ ! -f "$MARK/earlystop_s42.done" ]; then
  say "early-stopping curve, seed 42"
  "$PY" "$PROJ/scripts/38_earlystop_curve.py" --run y11n_tiled640_s42 --batch 4 \
      >> "$PROJ/logs/earlystop_s42.log" 2>&1
  # 12 saved epochs plus last.pt
  if [ "$(curve_rows "$PROJ/results/earlystop_curve_y11n_tiled640_s42.csv")" -ge 13 ]; then
    touch "$MARK/earlystop_s42.done"; say "early-stopping curve seed 42 complete"
  else
    say "early-stopping curve seed 42 incomplete; will continue next tick"
  fi
  exit 0
fi

# --- 3. target-domain data budget curve --------------------------------------
if [ ! -f "$MARK/budget.done" ]; then
  say "budget curve, seeds 42,123,456"
  "$PY" "$PROJ/scripts/39_budget_curve.py" --seeds 42,123,456 \
      >> "$PROJ/logs/budget_curve.log" 2>&1
  if [ "$(curve_rows "$PROJ/results/budget_curve.csv")" -ge 18 ]; then   # 6 budgets x 3 seeds
    touch "$MARK/budget.done"; say "budget curve complete"
  else
    say "budget curve incomplete; will continue next tick"
  fi
  exit 0
fi

# --- 4. early-stopping curve, seed 123 ---------------------------------------
if [ ! -f "$MARK/earlystop_s123.done" ]; then
  say "early-stopping curve, seed 123"
  "$PY" "$PROJ/scripts/38_earlystop_curve.py" --run y11n_tiled640_s123 --batch 4 \
      >> "$PROJ/logs/earlystop_s123.log" 2>&1
  if [ "$(curve_rows "$PROJ/results/earlystop_curve_y11n_tiled640_s123.csv")" -ge 13 ]; then
    touch "$MARK/earlystop_s123.done"; say "early-stopping curve seed 123 complete"
  else
    say "early-stopping curve seed 123 incomplete; will continue next tick"
  fi
  exit 0
fi

say "all queued work is complete"
