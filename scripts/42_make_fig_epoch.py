#!/usr/bin/env python3
"""Cross-domain accuracy is not monotone in training length.

In-domain validation rises throughout training, so ordinary practice keeps the last checkpoint.
Against the target domain both baseline seeds peak well before the end and then decline, which
means the checkpoint standard practice selects is not the one that transfers. The paper reports
epoch 120, so every cross-domain figure in it is pessimistic by roughly a sixth.

The shape is the point: a rise, a plateau, and a clear fall. A sentence of numbers cannot show
that the decline is sustained rather than a single noisy checkpoint.

Source: results/earlystop_curve_y11n_tiled640_s{42,123}.csv (scripts/38_earlystop_curve.py).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"42": "#1C5D7A", "123": "#B4633A"}
REPORTED = "#9A9A9A"


def series(seed: str) -> tuple[list[int], list[float]]:
    src = ROOT / f"results/earlystop_curve_y11n_tiled640_s{seed}.csv"
    rows = list(csv.DictReader(open(src)))
    return [int(r["epoch"]) for r in rows], [float(r["ap50"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/figures/fig_epoch.pdf"))
    cli = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.45, 1.95), dpi=400)
    summary = []
    for seed, colour in SEEDS.items():
        e, a = series(seed)
        peak = max(range(len(a)), key=lambda i: a[i])
        ax.plot(e, a, "-o", color=colour, lw=1.4, ms=2.8, zorder=3, label=f"seed {seed}")
        ax.plot([e[peak]], [a[peak]], "o", color=colour, ms=6, mfc="none", mew=1.2, zorder=4)
        ax.annotate(f"{a[peak]:.3f} @ {e[peak]}", (e[peak], a[peak]), textcoords="offset points",
                    xytext=(-4, 7), fontsize=5.9, color=colour)
        summary.append((seed, e[peak], a[peak], a[-1], (a[peak] - a[-1]) / a[peak]))

    ax.axvline(120, color=REPORTED, ls=":", lw=1.0, zorder=1)
    ax.annotate("reported", (120, 0.243), textcoords="offset points", xytext=(-33, 0),
                fontsize=5.9, color=REPORTED)

    ax.set_xlabel("training epoch", fontsize=7)
    ax.set_ylabel("WiSARD person AP@0.5", fontsize=7)
    ax.set_xlim(-4, 126)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", color="#DFE3E5", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.9, frameon=False, loc="lower right", handlelength=1.5,
              labelspacing=0.28, borderaxespad=0.3)

    out = Path(cli.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    for seed, pe, pa, last, drop in summary:
        print(f"[seed {seed}] peak {pa:.4f} @ epoch {pe}, epoch120 {last:.4f}, decline {drop:.1%}")
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
