#!/usr/bin/env python3
"""ROpti, the recall-optimal metric Manzini and Murphy tell this field to report.

Sambolek and Ivasic-Kos define it as

    ROpti = (TP - FP) / (TP + FN)

where TP + FN is simply the number of annotated people. It is recall with false positives
subtracted, on the grounds that in a search-and-rescue context every false alarm consumes an
operator's attention and therefore has a real cost, which plain AP hides.

We cite both papers and then did not report the metric, which is the kind of gap a careful reviewer
notices. This closes it.

TWO NUMBERS, BECAUSE ROPTI IS THRESHOLD-DEPENDENT
ROpti is unbounded below and collapses at a low confidence floor: at our 0.005 collection floor the
detector emits 204,087 false positives against 8,345 people, so ROpti is about -24. Quoting that
would be as misleading as quoting the maximum without saying where it came from. We report

  ROpti @ budget  at the same 0.1 FP/tile operating threshold as the recall we already publish,
                  so the two are directly comparable, and
  ROpti max       the best value over the whole threshold sweep, with the threshold that achieves
                  it, which is what "recall-optimal" is asking for.

Reads only the cached per-sequence histograms, so no inference and no GPU.
"""

from __future__ import annotations

import argparse
import random
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "results" / "cache"

from src.sar.eval.bootstrap import percentile_ci  # noqa: E402


def curves(tp_hist, fp_hist):
    """Cumulative TP and FP as the threshold walks down from the top bin."""
    return np.cumsum(tp_hist[::-1]), np.cumsum(fp_hist[::-1])


def ropti_at_budget(tp_hist, fp_hist, n_gt, n_units, fp_per_unit):
    tp_c, fp_c = curves(tp_hist, fp_hist)
    ok = fp_c <= fp_per_unit * n_units
    if not ok.any():
        return 0.0, 1.0
    i = int(np.max(np.nonzero(ok)[0]))
    thr = (len(tp_hist) - 1 - i) / len(tp_hist)
    return float((tp_c[i] - fp_c[i]) / n_gt), thr


def ropti_max(tp_hist, fp_hist, n_gt):
    tp_c, fp_c = curves(tp_hist, fp_hist)
    r = (tp_c - fp_c) / n_gt
    i = int(np.argmax(r))
    thr = (len(tp_hist) - 1 - i) / len(tp_hist)
    return float(r[i]), thr, float(tp_c[i] / n_gt)


def bootstrap_ci(z, unit_key: str, fp_per_unit: float, n_boot: int, seed: int):
    """Percentile intervals for both ROpti figures, resampling SEQUENCES.

    The point estimates are left untouched: they come from the exact pooled
    histogram above. Only the interval uses resampling, which is the contract the
    rest of this repo follows. Sequences are the resampling unit because WiSARD
    tiles come from consecutive video and are correlated within a flight, so a
    tile-level draw would return an indefensibly narrow interval.

    n_gt and the unit count are resampled with the histograms, not held fixed --
    a draw that happens to pick person-dense sequences must carry its own
    denominator or the ratio is incoherent.
    """
    tp_s, fp_s = z["tp"], z["fp"]
    gt_s, unit_s = z["n_gt"], z[unit_key]
    k = len(tp_s)
    rng = random.Random(seed)
    at_budget, at_max = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        tp, fp = tp_s[idx].sum(0), fp_s[idx].sum(0)
        g, n = int(gt_s[idx].sum()), int(unit_s[idx].sum())
        if g == 0:
            continue
        at_budget.append(ropti_at_budget(tp, fp, g, n, fp_per_unit)[0])
        at_max.append(ropti_max(tp, fp, g)[0])
    return percentile_ci(at_budget), percentile_ci(at_max)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-budget", type=float, default=0.1, help="FP per tile")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    targets = [("flags_y11n_tiled640_s42.npz", "YOLO11n tiled 640", "n_tiles"),
               ("flags_y8n_tiled640_s42.npz",  "YOLOv8n tiled 640", "n_tiles"),
               ("flags_y11n_tiled416_s42.npz", "YOLO11n tiled 416", "n_tiles"),
               ("flags_y11n_full640_s42.npz",  "Untiled 640 control", "n_tiles")]
    targets += [(f"scale_y11n_tiled640_s42_sz{s}.npz", f"YOLO11n tiled, infer @{s}", "n_tiles")
                for s in (512, 448, 384, 320)]

    rows = []
    print(f"{'configuration':30s} {'ROpti@bud':>10s} {'thr':>6s} {'ROpti max':>10s} "
          f"{'thr':>6s} {'recall there':>13s}")
    for fname, label, unit_key in targets:
        f = CACHE / fname
        if not f.exists():
            print(f"{label:30s}  cache missing: {fname}"); continue
        z = np.load(f, allow_pickle=False)
        tp, fp = z["tp"].sum(0), z["fp"].sum(0)
        n_gt, n_units = int(z["n_gt"].sum()), int(z[unit_key].sum())
        rb, tb = ropti_at_budget(tp, fp, n_gt, n_units, cli.fp_budget)
        rm, tm, rec = ropti_max(tp, fp, n_gt)
        (bl, bh), (ml, mh) = bootstrap_ci(z, unit_key, cli.fp_budget, cli.n_boot, cli.seed)
        rows.append({"configuration": label, "n_gt": n_gt, "tiles": n_units,
                     "ropti_at_fp_budget": round(rb, 4), "threshold_at_budget": round(tb, 4),
                     "ropti_at_fp_budget_ci_low": round(bl, 4),
                     "ropti_at_fp_budget_ci_high": round(bh, 4),
                     "ropti_max": round(rm, 4), "threshold_at_max": round(tm, 4),
                     "ropti_max_ci_low": round(ml, 4), "ropti_max_ci_high": round(mh, 4),
                     "recall_at_ropti_max": round(rec, 4),
                     "fp_budget_per_tile": cli.fp_budget})
        print(f"{label:30s} {rb:10.4f} {tb:6.3f} {rm:10.4f} {tm:6.3f} {rec:13.4f}"
              f"   [{ml:+.3f},{mh:+.3f}]")

    out_csv = ROOT / "results" / "ropti.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (ROOT / "results" / "ropti.json").write_text(json.dumps({
        "definition": "ROpti = (TP - FP) / (TP + FN), Sambolek and Ivasic-Kos",
        "why": ("Recommended for search and rescue because a false alarm costs operator time. "
                "Threshold-dependent and unbounded below, so we report it at the same operating "
                "point as our published recall AND at its own optimum."),
        "fp_budget_per_tile": cli.fp_budget,
        "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence",
        "seed": cli.seed,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\n[write] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
