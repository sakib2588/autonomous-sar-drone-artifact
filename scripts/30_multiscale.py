#!/usr/bin/env python3
"""Scale-consensus filtering: can two inference scales veto each other's false positives?

MOTIVATION, WHICH IS A PREDICTION FROM OUR OWN MECHANISM
Reducing the inference input from 640 to 320 px removes 46 percent of false positives while the
recall ceiling barely moves. The explanation we give is that those detections are founded on fine
texture -- bushes, rocks, shadow edges -- rather than on people. If that is right, then the false
positives should be scale-fragile and the true positives scale-stable, and requiring a detection to
appear at BOTH scales should suppress false positives harder than either scale alone.

That is a falsifiable prediction, and it is what turns the observation into a method.

STRATEGIES COMPARED
  640 alone / 320 alone     the two baselines
  confirm-640               keep a 640 detection when some 320 detection overlaps it
  confirm-320               keep a 320 detection when some 640 detection overlaps it
  union                     merge both sets and NMS, the naive ensemble

Detections are re-matched to ground truth AFTER filtering, never reusing the flags from the
unfiltered run: removing a detection can promote a former duplicate into a true positive, so reusing
old flags would quietly inflate every filtered variant.

Pure CPU, reads the dumps written by scripts/29_dump_detections.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.eval.bootstrap import average_precision, match_detections  # noqa: E402

TILES = Path("sar-drone-data/processed/sar_wisard_tiled")
CACHE = ROOT / "results" / "cache"
PERSON = 0


def gt_for(stem: str):
    p = TILES / "labels" / "test" / f"{stem}.txt"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) == 5 and int(t[0]) == PERSON:
            cx, cy, bw, bh = (float(v) for v in t[1:])
            out.append(((cx - bw / 2) * 640, (cy - bh / 2) * 640,
                        (cx + bw / 2) * 640, (cy + bh / 2) * 640))
    return out


def load(run: str, sz: int):
    z = np.load(CACHE / f"dets_{run}_sz{sz}.npz", allow_pickle=False)
    stems = [str(s) for s in z["stems"]]
    per = {i: [] for i in range(len(stems))}
    for ti, cf, bx in zip(z["det_tile"], z["det_conf"], z["det_box"]):
        per[int(ti)].append((float(cf), tuple(float(v) for v in bx)))
    for k in per:
        per[k].sort(key=lambda t: -t[0])
    return stems, per, z["n_gt"]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = w * h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, thr=0.5):
    keep = []
    for cf, bx in sorted(dets, key=lambda t: -t[0]):
        if all(iou(bx, k[1]) < thr for k in keep):
            keep.append((cf, bx))
    return keep


def score(per_tile, stems, n_gt):
    """Re-match every kept detection against ground truth, then AP and ROpti."""
    flags, tot_gt = [], 0
    for i, stem in enumerate(stems):
        g = gt_for(stem)
        tot_gt += len(g)
        f, _ = match_detections(per_tile.get(i, []), g, iou_thr=0.5)
        flags.extend(f)
    flags.sort(key=lambda t: -t[0])
    ap = average_precision(flags, tot_gt)
    tp = np.cumsum([1 if f[1] else 0 for f in flags])
    fp = np.cumsum([0 if f[1] else 1 for f in flags])
    ropti = float(np.max((tp - fp) / tot_gt)) if len(flags) else 0.0
    n_tiles = len(stems)
    ok = fp <= 0.1 * n_tiles
    rec_at_budget = float(tp[ok].max() / tot_gt) if ok.any() else 0.0
    return {"AP50": round(ap, 5), "ropti_max": round(ropti, 4),
            "recall_at_0.1fp": round(rec_at_budget, 4),
            "detections": len(flags), "true_positives": int(tp[-1]) if len(flags) else 0,
            "false_positives": int(fp[-1]) if len(flags) else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--hi", type=int, default=640)
    ap.add_argument("--lo", type=int, default=320)
    ap.add_argument("--iou", type=float, default=0.4, help="corroboration IoU")
    cli = ap.parse_args()

    stems, hi, n_gt = load(cli.run, cli.hi)
    _, lo, _ = load(cli.run, cli.lo)

    variants = {}
    variants[f"{cli.hi} alone"] = hi
    variants[f"{cli.lo} alone"] = lo

    conf_hi, conf_lo, uni = {}, {}, {}
    for i in range(len(stems)):
        H, L = hi.get(i, []), lo.get(i, [])
        conf_hi[i] = [d for d in H if any(iou(d[1], e[1]) >= cli.iou for e in L)]
        conf_lo[i] = [d for d in L if any(iou(d[1], e[1]) >= cli.iou for e in H)]
        uni[i] = nms(H + L, 0.5)
    variants[f"confirm-{cli.hi} by {cli.lo}"] = conf_hi
    variants[f"confirm-{cli.lo} by {cli.hi}"] = conf_lo
    variants["union + NMS"] = uni

    rows = []
    print(f"{'variant':26s} {'AP50':>8s} {'ROpti':>8s} {'rec@0.1':>8s} {'dets':>8s} {'TP':>6s} {'FP':>8s}")
    for name, pt in variants.items():
        r = score(pt, stems, n_gt); r["variant"] = name; rows.append(r)
        print(f"{name:26s} {r['AP50']:8.4f} {r['ropti_max']:8.4f} {r['recall_at_0.1fp']:8.4f} "
              f"{r['detections']:8d} {r['true_positives']:6d} {r['false_positives']:8d}")

    base = next(r for r in rows if r["variant"] == f"{cli.lo} alone")
    best = max(rows, key=lambda r: r["AP50"])
    verdict = (f"best variant '{best['variant']}' AP50 {best['AP50']:.4f} against "
               f"{base['AP50']:.4f} for the single-scale {cli.lo} baseline; "
               + ("consensus HELPS" if best["variant"] not in
                  (f"{cli.hi} alone", f"{cli.lo} alone") and best["AP50"] > base["AP50"]
                  else "consensus does NOT beat the single-scale baseline"))
    out = ROOT / "results" / "multiscale_consensus.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "AP50", "ropti_max", "recall_at_0.1fp",
                                           "detections", "true_positives", "false_positives"])
        w.writeheader()
        w.writerows([{k: r[k] for k in w.fieldnames} for r in rows])
    (ROOT / "results" / "multiscale_consensus.json").write_text(json.dumps(
        {"run": cli.run, "scales": [cli.hi, cli.lo], "corroboration_iou": cli.iou,
         "verdict": verdict, "variants": rows}, indent=2) + "\n")
    print(f"\n[verdict] {verdict}\n[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
