#!/usr/bin/env python3
"""Figure 2: the inference-scale result and, in the same figure, why it happens.

Plotting AP alone would be the misleading version of this result. AP rises as the input shrinks,
which invites the reading that matching apparent object size to the training distribution lets the
detector find people it was missing. The decomposition says otherwise, so both panels ship together:

  top    AP@0.5 against input size, with the PAIRED bootstrap interval. Paired because every size is
         scored on the same 38 sequences; the independent per-size intervals in
         results/scale_sweep_wisard.json overlap heavily and understate the effect.
  bottom true positives and false positives on separate axes. TP is close to flat (5081 -> 5202,
         +2.4%) while FP collapses (204087 -> 109493, -46.3%). The gain is precision.

Source: results/scale_diagnostics.csv (scripts/26_scale_diagnostics.py).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ACCENT = "#B3541E"     # the measured curve
MUTED = "#5C6B73"      # secondary series
GRID = "#D9DEDC"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results/scale_diagnostics.csv"))
    ap.add_argument("--out", default=str(ROOT / "paper/figures/fig2_scale.pdf"))
    cli = ap.parse_args()

    rows = list(csv.DictReader(open(cli.src)))
    rows.sort(key=lambda r: -int(r["imgsz"]))
    x = list(range(len(rows)))
    lbl = [r["imgsz"] for r in rows]
    ap50 = [float(r["AP50"]) for r in rows]
    base = next(float(r["AP50"]) for r in rows if r["imgsz"] == "640")
    lo = [base + float(r["delta_ci_low"]) for r in rows]
    hi = [base + float(r["delta_ci_high"]) for r in rows]
    sig = [r["significant"] == "True" for r in rows]
    tp = [int(r["true_positives"]) for r in rows]
    fp = [int(r["false_positives"]) / 1000 for r in rows]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 3.0), dpi=400, sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})

    a1.fill_between(x, lo, hi, color=ACCENT, alpha=0.16, lw=0,
                    label="95% CI, paired")
    a1.plot(x, ap50, "-", color=ACCENT, lw=1.4, zorder=3)
    a1.plot([xi for xi, s in zip(x, sig) if s], [v for v, s in zip(ap50, sig) if s],
            "o", ms=4.2, color=ACCENT, zorder=4, label="differs from 640 ($p<0.05$)")
    a1.plot([xi for xi, s in zip(x, sig) if not s], [v for v, s in zip(ap50, sig) if not s],
            "o", ms=4.2, mfc="white", mec=ACCENT, mew=1.1, zorder=4, label="not significant")
    a1.axhline(base, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    a1.set_ylabel("person AP@0.5", fontsize=7)
    a1.tick_params(labelsize=6.4)
    a1.legend(fontsize=5.4, frameon=False, loc="lower left", handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.25)
    a1.grid(axis="y", color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    a2.plot(x, tp, "s-", color=MUTED, lw=1.3, ms=3.4, label="true positives")
    a2.set_ylabel("true positives", fontsize=7, color=MUTED)
    a2.tick_params(labelsize=6.4, axis="y", colors=MUTED)
    a2.set_ylim(0, max(tp) * 1.35)
    a3 = a2.twinx()
    a3.plot(x, fp, "^-", color=ACCENT, lw=1.3, ms=3.6, label="false positives")
    a3.set_ylabel("false positives (thousands)", fontsize=7, color=ACCENT)
    a3.tick_params(labelsize=6.4, axis="y", colors=ACCENT)
    a3.set_ylim(0, max(fp) * 1.18)
    for s in ("top",):
        a2.spines[s].set_visible(False); a3.spines[s].set_visible(False)
    a2.grid(axis="y", color=GRID, lw=0.5); a2.set_axisbelow(True)

    a2.set_xticks(x); a2.set_xticklabels(lbl, fontsize=6.4)
    a2.set_xlabel("inference input size (px), native tile is 640", fontsize=7)

    h2, l2 = a2.get_legend_handles_labels(); h3, l3 = a3.get_legend_handles_labels()
    a2.legend(h2 + h3, l2 + l3, fontsize=5.4, frameon=False, loc="upper right",
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)

    fig.subplots_adjust(left=0.155, right=0.845, top=0.985, bottom=0.115, hspace=0.09)
    out = Path(cli.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
