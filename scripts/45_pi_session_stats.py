"""Per-session inference latency and look rate from the Pi's own detection log.

Backs the Discussion claim that moving the deployed payload from full-integer
INT8 to FP32 raises median inference 645 -> 772 ms (a factor of 1.20) while the
end-to-end look rate is indistinguishable, because capture, tiling and encoding
set the pace rather than the model.

Those numbers previously existed only in notes/20260825-decision-pi-fp32-switch
-and-wifi-powersave.md, with the raw evidence living on the Pi's SD card and
nowhere in the repo. Source pulled to results/pi_sessions/ on 2026-08-27.

    python3 scripts/45_pi_session_stats.py

Two things the log will do to you if you read it naively:

* It is CUMULATIVE and append-only ACROSS BOOTS, so a naive max(t) - min(t) on
  the tail block spans days of powered-off time and reports a look rate near
  zero. Sessions are split on the frame counter resetting to 0.
* A few lines are truncated JSON, from power being pulled mid-write while the
  service was recording. They are counted and skipped, never silently dropped.
"""

import csv
import datetime as dt
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "pi_sessions" / "detections" / "live_640_fp32_cumulative.jsonl"
OUT = ROOT / "results" / "pi_session_stats.csv"

# The paper reports sessions of 6 to 17 minutes; shorter runs are smoke tests
# whose look rate is dominated by start-up rather than by steady state.
MIN_MINUTES = 6.0


def load(path):
    rows, bad = [], 0
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def sessionise(rows):
    out, cur = [], []
    for r in rows:
        if r.get("frame") == 0 and cur:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def main():
    rows, bad = load(SRC)
    print(f"parsed {len(rows)} records, skipped {bad} truncated lines")

    recs = []
    for s in sessionise(rows):
        dur = s[-1]["t"] - s[0]["t"]
        # drop the first record: it carries model-load cost, not steady state
        vals = [r["infer_ms"] for r in s[1:]] or [r["infer_ms"] for r in s]
        recs.append(dict(
            start=dt.datetime.fromtimestamp(s[0]["t"]).strftime("%Y-%m-%d %H:%M"),
            n=len(s),
            median_infer_ms=round(st.median(vals), 1),
            duration_min=round(dur / 60, 2),
            looks_per_s=round(len(s) / dur, 3) if dur > 0 else 0.0,
            steady=(dur / 60) >= MIN_MINUTES and (dur / 60) < 60,
        ))

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    steady = [r for r in recs if r["steady"]]
    int8 = [r for r in steady if r["median_infer_ms"] < 700]
    fp32 = [r for r in steady if r["median_infer_ms"] >= 700]
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(recs)} sessions, "
          f"{len(steady)} steady-state)")
    for name, grp in (("INT8", int8), ("FP32", fp32)):
        if not grp:
            continue
        med = st.median(r["median_infer_ms"] for r in grp)
        lo = min(r["looks_per_s"] for r in grp)
        hi = max(r["looks_per_s"] for r in grp)
        print(f"  {name}: median {med:.1f} ms, look rate {lo:.3f}-{hi:.3f}/s "
              f"over {len(grp)} sessions")
    if int8 and fp32:
        a = st.median(r["median_infer_ms"] for r in int8)
        b = st.median(r["median_infer_ms"] for r in fp32)
        print(f"  ratio: {b / a:.2f}x   (paper reports 645 -> 772 ms, 1.20x)")


if __name__ == "__main__":
    main()
