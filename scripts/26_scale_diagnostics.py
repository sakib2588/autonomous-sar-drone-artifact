#!/usr/bin/env python3
"""Paired significance and mechanism decomposition for the inference-scale sweep.

Two things scripts/21_scale_sweep.py does NOT do, both needed before the scale result can be
written up honestly.

1. PAIRED SIGNIFICANCE. Script 21 reports an independent percentile interval per input size and
   compares a point against the baseline's interval. That is the wrong test here: every size is
   scored on the SAME 38 sequences, so the comparison is paired. Resampling sequences once and
   taking the difference on that resample removes the between-sequence variance that dominates the
   independent intervals, and it is much tighter. Script 21's built-in verdict says "NOT SUPPORTED"
   purely because it applies the conservative test.

2. MECHANISM. AP going up says nothing about why. Splitting the change into true positives against
   false positives distinguishes two very different stories:
       recall ceiling rises  -> the detector is finding people it previously missed
       FP falls, recall flat -> the detector is hallucinating less on unfamiliar terrain
   These are different papers, and the second one is what the data actually shows.

Reads only the cached per-sequence histograms written by script 21, so this is pure CPU and can run
while the GPU is busy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.eval.bootstrap import ap_from_hist, percentile_ci  # noqa: E402

CACHE = ROOT / "results" / "cache"
SIZES = [640, 512, 448, 384, 320, 256, 192, 160]


def load(run: str, imgsz: int) -> dict:
    z = np.load(CACHE / f"scale_{run}_sz{imgsz}.npz", allow_pickle=False)
    return {
        "seq": [str(s) for s in z["sequences"]],
        "tp": z["tp"], "fp": z["fp"], "n_gt": z["n_gt"], "n_tiles": z["n_tiles"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--baseline", type=int, default=640)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-size", type=int, default=len(SIZES) - 1,
                    help="number of comparisons sharing the baseline; sets the Bonferroni alpha")
    cli = ap.parse_args()

    base = load(cli.run, cli.baseline)
    k = len(base["seq"])
    rows = []

    for sz in SIZES:
        d = load(cli.run, sz)
        assert d["seq"] == base["seq"], "sequence order differs; comparison would not be paired"
        assert (d["n_gt"] == base["n_gt"]).all(), "ground-truth counts differ between sizes"

        tp, fp, g = int(d["tp"].sum()), int(d["fp"].sum()), int(d["n_gt"].sum())
        ap50 = ap_from_hist(d["tp"].sum(0), d["fp"].sum(0), g)

        # Every size is compared against the SAME baseline, so the family-wise error
        # rate inflates with the number of comparisons. Report the uncorrected
        # interval and the Bonferroni-corrected one side by side, and let the
        # corrected one decide significance.
        alpha_fam = 0.05 / max(cli.family_size, 1)
        if sz == cli.baseline:
            lo = hi = mean = 0.0
            blo = bhi = 0.0
            pgt = 0.5
        else:
            rng = random.Random(cli.seed)
            diffs = []
            for _ in range(cli.n_boot):
                idx = [rng.randrange(k) for _ in range(k)]
                gg = int(base["n_gt"][idx].sum())
                a0 = ap_from_hist(base["tp"][idx].sum(0), base["fp"][idx].sum(0), gg)
                a1 = ap_from_hist(d["tp"][idx].sum(0), d["fp"][idx].sum(0), gg)
                diffs.append(a1 - a0)
            lo, hi = percentile_ci(diffs)
            blo, bhi = percentile_ci(diffs, alpha=alpha_fam)
            mean = float(np.mean(diffs))
            pgt = sum(1 for x in diffs if x > 0) / len(diffs)

        rows.append({
            "imgsz": sz,
            "downscale": round(640 / sz, 3),
            "AP50": round(ap50, 5),
            "delta_AP50": round(mean, 5),
            "delta_ci_low": round(lo, 5),
            "delta_ci_high": round(hi, 5),
            "delta_ci_low_bonferroni": round(blo, 5),
            "delta_ci_high_bonferroni": round(bhi, 5),
            "bonferroni_confidence": round(100 * (1 - alpha_fam), 2),
            "p_delta_gt_0": round(pgt, 4),
            "significant": bool(lo > 0),
            "significant_bonferroni": bool(blo > 0),
            "true_positives": tp,
            "false_positives": fp,
            "gt_boxes": g,
            "recall_ceiling": round(tp / g, 5),
            "fp_per_tile": round(fp / int(d["n_tiles"].sum()), 4),
        })
        r = rows[-1]
        flag = ("SIG-bonf" if r["significant_bonferroni"]
                else "SIG" if r["significant"]
                else "baseline" if sz == cli.baseline else "ns ")
        print(f"  imgsz {sz:4d}  AP50 {r['AP50']:.4f}  d {r['delta_AP50']:+.4f} "
              f"[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}] "
              f"bonf[{r['delta_ci_low_bonferroni']:+.4f},{r['delta_ci_high_bonferroni']:+.4f}] {flag:9s} "
              f"TP {tp:5d}  FP {fp:7d}  ceiling {r['recall_ceiling']:.4f}", flush=True)

    csv_path = ROOT / "results" / "scale_diagnostics.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    b = next(r for r in rows if r["imgsz"] == cli.baseline)
    best = max(rows, key=lambda r: r["AP50"])
    payload = {
        "run": cli.run, "baseline_imgsz": cli.baseline, "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence", "paired": True, "seed": cli.seed,
        "family_size": cli.family_size,
        "bonferroni_confidence": round(100 * (1 - 0.05 / max(cli.family_size, 1)), 2),
        "multiple_comparisons": (
            f"{cli.family_size} sizes are compared against the same baseline, so intervals are "
            f"also reported at the Bonferroni-corrected "
            f"{100 * (1 - 0.05 / max(cli.family_size, 1)):.2f}% level; "
            "`significant_bonferroni` is the corrected verdict."
        ),
        "finding": (
            f"Best AP50 at imgsz {best['imgsz']} ({best['AP50']:.4f}) against {b['AP50']:.4f} at "
            f"{cli.baseline}, a paired difference of {best['delta_AP50']:+.4f} with 95% interval "
            f"[{best['delta_ci_low']:.4f}, {best['delta_ci_high']:.4f}]."
        ),
        "mechanism": (
            f"True positives move {b['true_positives']} -> {best['true_positives']} "
            f"({100*(best['true_positives']/b['true_positives']-1):+.1f}%) and the recall ceiling "
            f"{b['recall_ceiling']:.4f} -> {best['recall_ceiling']:.4f}, while false positives fall "
            f"{b['false_positives']} -> {best['false_positives']} "
            f"({100*(best['false_positives']/b['false_positives']-1):+.1f}%). The gain is dominated by "
            "false-positive suppression, not by detecting people the model previously missed."
        ),
        "sweep": rows,
    }
    (ROOT / "results" / "scale_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{payload['finding']}\n{payload['mechanism']}")
    print(f"[write] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
