#!/usr/bin/env python3
"""Figure 3: encounter-level detection against the independence bound.

Per-frame recall is the number the paper is judged on and it is not the number a search cares
about. A drone gets many looks at the same ground. This plots the measured probability that a
person-bearing frame is detected within a window of K consecutive sampled frames, against the
bound that would hold if the looks were independent.

Both curves matter. The measured one answers "does multiple coverage rescue a weak per-frame
detector" (largely yes: 0.61 to 0.92). The gap answers "how much of that redundancy is real"
(less than it looks: consecutive looks are correlated, and the shortfall peaks near K=4).

Source: results/encounter_aggregation.csv (scripts/31_encounter.py).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MEAS = "#1C5D7A"
BOUND = "#9A9A9A"
FILL = "#C8DCE6"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "results/encounter_aggregation.csv"))
    ap.add_argument("--out", default=str(ROOT / "paper/figures/fig3_encounter.pdf"))
    cli = ap.parse_args()

    rows = list(csv.DictReader(open(cli.src)))
    K = [int(r["K"]) for r in rows]
    p = [float(r["p_detected"]) for r in rows]
    b = [float(r["independence_bound"]) for r in rows]

    fig, ax = plt.subplots(figsize=(3.45, 2.15), dpi=400)
    ax.fill_between(K, p, b, color=FILL, alpha=0.75, lw=0, zorder=1,
                    label="correlation shortfall")
    ax.plot(K, b, "--", color=BOUND, lw=1.1, zorder=2, label="independent looks")
    ax.plot(K, p, "-o", color=MEAS, lw=1.5, ms=3.1, zorder=3, label="measured")

    ax.set_xlabel("consecutive sampled frames $K$", fontsize=7)
    ax.set_ylabel("P(person-bearing frame\ndetected in the window)", fontsize=7)
    ax.set_xlim(0.5, max(K) + 0.5)
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color="#DFE3E5", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=1.5,
              labelspacing=0.28, borderaxespad=0.3)

    # anchor the two endpoints the text quotes
    ax.annotate(f"{p[0]:.2f}", (K[0], p[0]), textcoords="offset points", xytext=(6, -8),
                fontsize=6, color=MEAS)
    ax.annotate(f"{p[-1]:.2f}", (K[-1], p[-1]), textcoords="offset points", xytext=(-16, -10),
                fontsize=6, color=MEAS)

    out = Path(cli.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}  K=1 {p[0]:.4f}  K={K[-1]} {p[-1]:.4f}  max gap {max(bb-pp for pp,bb in zip(p,b)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
