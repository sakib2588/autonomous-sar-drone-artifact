"""The confidence threshold, swept over the whole corpus rather than a sample.

WHY THIS SCRIPT
---------------
The deployed service inherited `--conf-thres 0.25`, the conventional
object-detection default, which nothing in this project ever chose deliberately.
That default was first questioned with an on-device sweep over 80 tiles drawn
40 positive and 40 negative. Two problems with quoting that number in a paper:
the draw is BALANCED, so its recall is not corpus recall on a corpus that is
85.8 percent empty, and it measures the deployed TFLite artifact rather than the
model every other table reports.

This sweeps the same decision over all 37,058 tiles using the cached detections,
so the threshold claim rests on the same evidence and the same model as the rest
of the paper.

Thresholding after matching is exact for recall, not an approximation: matching
is greedy in descending confidence, so discarding the lowest-confidence
detections cannot change which ground-truth box a higher-confidence detection
claimed.

USAGE
  .venv/bin/python scripts/43_threshold_sweep.py [--run RUN] [--imgsz N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
OUT = ROOT / "results" / "threshold_sweep.txt"

GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
DEFAULT_CONVENTION = 0.25
CHOSEN = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    path = CACHE / f"dets_{a.run}_sz{a.imgsz}.npz"
    if not path.is_file():
        raise SystemExit(f"missing cache {path}\nrun: scripts/29_dump_detections.py --run {a.run} --imgsz {a.imgsz}")
    z = np.load(path, allow_pickle=True)
    conf, tp = z["det_conf"], z["det_tp"]
    n_gt, n_tiles = int(z["n_gt"].sum()), len(z["stems"])

    lines = [f"run {a.run} @ imgsz {a.imgsz}",
             f"corpus {n_tiles} tiles, {n_gt} annotated people",
             "",
             f"{'conf':>6} {'TP':>6} {'FP':>8} {'recall':>8} {'precision':>10} {'FP/tile':>8}"]
    stats = {}
    for t in GRID:
        m = conf >= t
        TP = int(tp[m].sum())
        FP = int((~tp[m]).sum())
        stats[t] = (TP, FP, TP / n_gt)
        lines.append(f"{t:6.2f} {TP:6d} {FP:8d} {TP / n_gt:8.4f} "
                     f"{TP / max(TP + FP, 1):10.4f} {FP / n_tiles:8.3f}")

    r_def = stats[DEFAULT_CONVENTION][2]
    r_new = stats[CHOSEN][2]
    lines += ["",
              f"convention {DEFAULT_CONVENTION:.2f} recovers {r_def:.4f} of annotated people",
              f"chosen     {CHOSEN:.2f} recovers {r_new:.4f}",
              f"relative gain from moving the threshold alone   {100 * (r_new / r_def - 1):+.1f}%",
              f"cost in false positives per tile                "
              f"{stats[CHOSEN][1] / n_tiles - stats[DEFAULT_CONVENTION][1] / n_tiles:+.3f}"]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
