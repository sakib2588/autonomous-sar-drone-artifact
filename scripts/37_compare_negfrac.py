"""Negative-fraction control against its baseline, on the WiSARD tiles.

WHY THIS SCRIPT AND NOT AP
--------------------------
The hypothesis is about false positives, not detection. AP can sit still while
the thing we care about halves, so the headline number here is **false positives
at matched recall**: hold recall fixed at the baseline's operating point, then
ask how many false positives each model needed to get there. AP@0.5 is reported
too, but as context.

Reads the per-detection caches written by `29_dump_detections.py`, which already
carries the greedy IoU-0.5 matching, so both models are scored by identical code
on identical tiles. Nothing here re-runs inference and nothing is written to the
published result files.

USAGE
  .venv/bin/python scripts/37_compare_negfrac.py --base <run> --ctrl <run>

The two runs must be trained identically except for the negative fraction, and
must be compared at the SAME epoch. A control at epoch 120 against a baseline at
epoch 50 measures training length, not negative fraction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"


def load(run: str, imgsz: int) -> dict:
    p = CACHE / f"dets_{run}_sz{imgsz}.npz"
    if not p.is_file():
        raise SystemExit(f"missing cache {p}\nrun: scripts/29_dump_detections.py --run {run} --imgsz {imgsz}")
    z = np.load(p, allow_pickle=True)
    return {
        "n_tiles": len(z["stems"]),
        "n_gt": int(z["n_gt"].sum()),
        "conf": z["det_conf"],
        "tp": z["det_tp"],
    }


def curve(d: dict):
    """Descending-confidence sweep -> (recall, fp, thresholds), one point per detection."""
    order = np.argsort(-d["conf"])
    tp = d["tp"][order].astype(np.int64)
    conf = d["conf"][order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    return ctp / max(d["n_gt"], 1), cfp, conf


def ap50(d: dict) -> float:
    """All-point interpolated AP, matching the paper's metric."""
    rec, cfp, _ = curve(d)
    ctp = rec * d["n_gt"]
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # monotone-decreasing precision envelope
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.concatenate([[0.0], rec])
    p = np.concatenate([[prec[0] if len(prec) else 0.0], prec])
    return float(np.sum(np.diff(r) * p[1:]))


def fp_at_recall(d: dict, target: float):
    """Fewest false positives needed to reach `target` recall, and the threshold."""
    rec, cfp, conf = curve(d)
    idx = np.searchsorted(rec, target, side="left")
    if idx >= len(rec):
        return None, None, rec[-1] if len(rec) else 0.0
    return int(cfp[idx]), float(conf[idx]), float(rec[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline run name")
    ap.add_argument("--ctrl", required=True, help="negative-fraction control run name")
    ap.add_argument("--imgsz", type=int, default=640)
    cli = ap.parse_args()

    b, c = load(cli.base, cli.imgsz), load(cli.ctrl, cli.imgsz)
    if b["n_tiles"] != c["n_tiles"] or b["n_gt"] != c["n_gt"]:
        raise SystemExit(f"evaluation sets differ: {b['n_tiles']}/{b['n_gt']} vs {c['n_tiles']}/{c['n_gt']}")

    print(f"WiSARD zero-shot, {b['n_tiles']:,} tiles, {b['n_gt']:,} annotated people, imgsz {cli.imgsz}")
    print(f"  baseline: {cli.base}")
    print(f"  control : {cli.ctrl}\n")

    print(f"{'metric':<34}{'baseline':>12}{'control':>12}{'change':>12}")
    print("-" * 70)
    ab, ac = ap50(b), ap50(c)
    print(f"{'AP@0.5':<34}{ab:>12.4f}{ac:>12.4f}{ac - ab:>+12.4f}")

    rb, _, _ = curve(b)
    rc, _, _ = curve(c)
    print(f"{'recall ceiling':<34}{rb[-1]:>12.4f}{rc[-1]:>12.4f}{rc[-1] - rb[-1]:>+12.4f}")
    print(f"{'total detections':<34}{len(b['conf']):>12,}{len(c['conf']):>12,}{len(c['conf']) - len(b['conf']):>+12,}")

    print(f"\nfalse positives at matched recall (the headline)")
    print(f"{'target recall':<34}{'baseline FP':>12}{'control FP':>12}{'change':>12}")
    print("-" * 70)
    for t in (0.10, 0.20, 0.30, 0.40, 0.50):
        if t > min(rb[-1], rc[-1]):
            continue
        fb, thb, _ = fp_at_recall(b, t)
        fc, thc, _ = fp_at_recall(c, t)
        if fb is None or fc is None:
            continue
        pct = (fc - fb) / fb * 100 if fb else float("nan")
        print(f"{t:<34.2f}{fb:>12,}{fc:>12,}{pct:>11.1f}%")

    print("\nInterpretation is NOT automatic. A reduction here supports the sampling-policy")
    print("explanation at this dose. No reduction means 'no effect detectable at the largest")
    print("negative fraction this corpus can supply' -- 9.55% against WiSARD's 85.8% -- and")
    print("is NOT a refutation of the mechanism.")


if __name__ == "__main__":
    main()
