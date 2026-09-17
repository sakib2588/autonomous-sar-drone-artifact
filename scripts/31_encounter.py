#!/usr/bin/env python3
"""Encounter-level detection: per-frame recall understates what a search actually achieves.

THE ARGUMENT, AND WHY NOBODY HAS MEASURED IT
A drone does not get one look at a person, it gets many. Fraternali et al. state this in prose --
"to successfully identify a person in a practical application, consistent detection from all
viewpoints is not required... recall is a conservative metric" -- but do not quantify it. Zell et al.
give a mission-level formalism but assume a perfect in-field-of-view detector. What is missing is a
MEASURED per-look probability fed into that framing, which is what this computes.

DEFINITION, CONSTRAINED BY WHAT THE DATA SUPPORTS
WiSARD carries no track identifiers, so "the same person across frames" cannot be resolved. We
therefore define the unit at frame presence within a flight:

  a frame is POSITIVE   if any of its tiles contains an annotated person
  a frame is DETECTED   if any of its tiles yields a true positive above the operating threshold
  a window of K         is K consecutive positive frames from one sequence, and it is detected if
                        ANY frame in it is detected

We report P(detected) against K, alongside the independence bound 1-(1-p)^K. Consecutive frames are
correlated, so the measured curve must sit below the bound; the size of that gap is how much of the
apparent redundancy is real, and it is a result in its own right.

CAVEAT TO CARRY INTO THE PAPER
Tiles were built at frame stride 10, so consecutive kept frames are about a third of a second apart
at 30 fps, not adjacent. K is therefore in units of sampled frames, not raw video frames.

Pure CPU, reads the dump from scripts/29_dump_detections.py.
"""

from __future__ import annotations

import argparse
import random
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "results" / "cache"

from src.sar.eval.bootstrap import percentile_ci  # noqa: E402


def seq_frame(stem: str):
    """(sequence, frame index) from a tile stem `<sequence>_<frame>__t<x>_<y>`."""
    base = stem.split("__t")[0]
    m = re.match(r"^(.*)_(\d+)$", base)
    if not m:
        raise ValueError(f"cannot parse {stem!r}")
    return m.group(1), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--fp-budget", type=float, default=0.1, help="FP per tile, sets the threshold")
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    z = np.load(CACHE / f"dets_{cli.run}_sz{cli.imgsz}.npz", allow_pickle=False)
    stems = [str(s) for s in z["stems"]]
    n_gt = z["n_gt"]
    d_tile, d_conf, d_tp = z["det_tile"], z["det_conf"], z["det_tp"]

    # Operating threshold: walk confidence down until the false-positive budget is spent.
    order = np.argsort(-d_conf)
    fp_run = np.cumsum(~d_tp[order])
    budget = cli.fp_budget * len(stems)
    ok = fp_run <= budget
    thr = float(d_conf[order][ok][-1]) if ok.any() else 1.0
    print(f"operating threshold {thr:.4f} at {cli.fp_budget} FP/tile "
          f"({int(fp_run[ok][-1]) if ok.any() else 0} FP over {len(stems)} tiles)")

    frame_pos, frame_det = defaultdict(bool), defaultdict(bool)
    for i, stem in enumerate(stems):
        key = seq_frame(stem)
        if n_gt[i] > 0:
            frame_pos[key] = True
        frame_det.setdefault(key, False)
    for ti, cf, tp in zip(d_tile, d_conf, d_tp):
        if tp and cf >= thr:
            frame_det[seq_frame(stems[int(ti)])] = True

    by_seq = defaultdict(list)
    for (s, f), pos in frame_pos.items():
        if pos:
            by_seq[s].append(f)
    for s in by_seq:
        by_seq[s].sort()

    n_pos_frames = sum(len(v) for v in by_seq.values())
    print(f"{len(by_seq)} sequences contain people, {n_pos_frames} positive frames\n")

    rows = []
    p1 = None
    print(f"{'K':>3s} {'windows':>8s} {'detected':>9s} {'P(det)':>8s} {'independence':>13s} {'gap':>7s}")
    seq_keys = sorted(by_seq)
    rng = random.Random(cli.seed)
    for K in range(1, cli.kmax + 1):
        # Keep the per-sequence counts, not just their sum: the sequence is the
        # resampling unit, because windows inside one flight share pose,
        # illumination and terrain and are not independent draws.
        per_seq = {}
        for s, frames in by_seq.items():
            t = d = 0
            for i in range(0, len(frames) - K + 1):
                t += 1
                if any(frame_det[(s, f)] for f in frames[i : i + K]):
                    d += 1
            per_seq[s] = (t, d)
        tot = sum(t for t, _ in per_seq.values())
        det = sum(d for _, d in per_seq.values())
        if tot == 0:
            continue
        p = det / tot

        # The operating threshold is held FIXED at the value derived above rather
        # than re-derived inside each resample. Re-deriving it would fold
        # threshold selection into the interval and widen it for the wrong reason.
        k = len(seq_keys)
        samples = []
        for _ in range(cli.n_boot):
            idx = [rng.randrange(k) for _ in range(k)]
            bt = bd = 0
            for j in idx:
                t, d = per_seq[seq_keys[j]]
                bt += t
                bd += d
            if bt:
                samples.append(bd / bt)
        lo, hi = percentile_ci(samples)

        if K == 1:
            p1 = p
        bound = 1 - (1 - p1) ** K
        rows.append({"K": K, "windows": tot, "detected": det, "p_detected": round(p, 4),
                     "p_ci_low": round(lo, 4), "p_ci_high": round(hi, 4),
                     "independence_bound": round(bound, 4), "gap": round(bound - p, 4)})
        print(f"{K:3d} {tot:8d} {det:9d} {p:8.4f} [{lo:.3f},{hi:.3f}] {bound:9.4f} {bound - p:7.4f}")

    out = ROOT / "results" / "encounter_aggregation.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    last = rows[-1]
    (ROOT / "results" / "encounter_aggregation.json").write_text(json.dumps({
        "run": cli.run, "imgsz": cli.imgsz, "operating_threshold": round(thr, 4),
        "fp_budget_per_tile": cli.fp_budget,
        "unit": "frame presence within a sequence; WiSARD has no track ids",
        "frame_stride_caveat": "tiles built at frame stride 10, so K counts sampled frames ~1/3 s apart",
        "single_frame_probability": p1,
        "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence",
        "seed": cli.seed,
        "threshold_policy": ("held fixed at the point-estimate operating threshold inside every "
                             "resample; re-deriving it per draw would fold threshold selection "
                             "into the interval"),
        "finding": (f"A person-bearing frame is detected with probability {p1:.3f} on its own. "
                    f"Over {last['K']} consecutive sampled frames that rises to {last['p_detected']:.3f}, "
                    f"against an independence bound of {last['independence_bound']:.3f}; the "
                    f"{last['gap']:.3f} shortfall is the correlation between consecutive looks."),
        "curve": rows,
    }, indent=2) + "\n")
    print(f"\n[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
