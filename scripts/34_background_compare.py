#!/usr/bin/env python3
"""Phase A scoring: false-positive rate on empty ground, in-domain versus out-of-domain.

Both sets contain no people at all, so every detection is a false positive by construction and no
matching is required -- the count IS the error.

  in-domain   3,684 VisDrone negative tiles that select_tiles generated and then discarded. Same
              pipeline, same distribution, never seen in training.
  target      31,793 WiSARD negative tiles.

The WiSARD side is free: results/cache/dets_*.npz already stores per-tile detections and per-tile
ground-truth counts, so the negative-tile subset is a filter, not a new inference pass.

READING THE RESULT
  rates similar        the propensity to fire on empty ground was trained in and is not specific to
                       unfamiliar terrain, which supports the training-composition explanation and
                       predicts the negative-fraction control will work.
  WiSARD much higher   unfamiliar terrain dominates, composition is a smaller factor, and the
                       control will probably disappoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "results" / "cache"
HELDOUT = Path("sar-drone-data/processed/visdrone_heldout_neg")
PERSON = 0


def wisard_negative_stats(run: str, imgsz: int, conf: float):
    """FP counts on WiSARD tiles that contain no annotated person. Pure filter over the cache."""
    z = np.load(CACHE / f"dets_{run}_sz{imgsz}.npz", allow_pickle=False)
    n_gt, tile, cf = z["n_gt"], z["det_tile"], z["det_conf"]
    neg = np.where(n_gt == 0)[0]
    negset = np.zeros(len(n_gt), dtype=bool); negset[neg] = True
    on_neg = negset[tile] & (cf >= conf)
    return int(on_neg.sum()), int(len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.208,
                    help="operating threshold; default is the 0.1 FP/tile point at imgsz 640")
    ap.add_argument("--batch", type=int, default=12)
    cli = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "checkpoints" / cli.run / "weights" / "best.pt"))

    imgs = sorted((HELDOUT / "images" / "test").iterdir())
    fp_in = 0
    for i in range(0, len(imgs), cli.batch):
        chunk = imgs[i : i + cli.batch]
        for r in model.predict([str(p) for p in chunk], imgsz=cli.imgsz, conf=cli.conf,
                               max_det=100, device="0", verbose=False, stream=False):
            b = r.boxes
            if b is not None and len(b):
                fp_in += int(sum(1 for c in b.cls.tolist() if int(c) == PERSON))
        if (i // cli.batch) % 60 == 0:
            print(f"  in-domain {i + len(chunk)}/{len(imgs)}  FP so far {fp_in}", flush=True)

    fp_out, n_out = wisard_negative_stats(cli.run, cli.imgsz, cli.conf)
    n_in = len(imgs)
    r_in, r_out = fp_in / n_in, fp_out / n_out

    out = {
        "run": cli.run, "imgsz": cli.imgsz, "confidence_threshold": cli.conf,
        "in_domain": {"source": "VisDrone train negatives held out of training",
                      "tiles": n_in, "false_positives": fp_in, "fp_per_tile": round(r_in, 4)},
        "target_domain": {"source": "WiSARD negative tiles",
                          "tiles": n_out, "false_positives": fp_out, "fp_per_tile": round(r_out, 4)},
        "ratio_target_over_in_domain": round(r_out / r_in, 3) if r_in > 0 else None,
    }
    v = out["ratio_target_over_in_domain"]
    out["verdict"] = (
        "Cannot compute a ratio: the model produces no false positives on in-domain empty ground."
        if v is None else
        (f"The model fires {v:.2f}x more per tile on unfamiliar empty ground than on familiar empty "
         "ground. Unfamiliar terrain dominates; training composition is a secondary factor and the "
         "negative-fraction control is unlikely to recover much."
         if v >= 3 else
         f"Rates are within {v:.2f}x. The propensity to fire on empty ground is largely trained in "
         "rather than provoked by unfamiliar terrain, which supports the composition explanation "
         "and predicts the negative-fraction control will help.")
    )
    (ROOT / "results" / "background_fp.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'set':44s} {'tiles':>7s} {'FP':>8s} {'FP/tile':>9s}")
    print(f"{'VisDrone negatives, held out of training':44s} {n_in:7d} {fp_in:8d} {r_in:9.4f}")
    print(f"{'WiSARD negatives':44s} {n_out:7d} {fp_out:8d} {r_out:9.4f}")
    print(f"\nratio (target / in-domain): {v}")
    print(f"\n{out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
