#!/usr/bin/env python3
"""Inference-scale sweep on WiSARD: does the cross-domain cliff close when apparent
object size is pushed back into the trained band?

WHY THIS EXISTS
---------------
Measured 2026-08-24, all in identical 640 px tile space, full populations:

    VisDrone train (what the model learned)  mean 20.88 px  median 17.29  74.8% in 8-32 px
    WiSARD         (the zero-shot target)    mean 67.30 px  median 56.78  20.4% in 8-32 px

The cross-domain target is 3.2x LARGER than the training distribution, not smaller.
The project's standing explanation for the cliff -- "people are tiny" -- is correct
for VisDrone's own hard test split and backwards for the cross-domain case.

THE TEST
--------
YOLO is fully convolutional, so shrinking the inference input shrinks apparent object
size identically. Feeding an existing 640 px tile at imgsz=256 downscales content 2.5x,
so a 56 px person is seen by the network at ~22 px -- inside the trained band.
Ultralytics rescales predictions back to orig_shape, so ground truth in 640 space still
matches and no coordinate surgery is needed.

Predicted in-band coverage by downscale factor s = 640/imgsz:

    imgsz   640    512    448    384    320    256    192    160
    s      1.00   1.25   1.43   1.67   2.00   2.50   3.33   4.00
    in-band 21.1%  29.5%  ~34%   ~44%  57.3%  68.6%  ~71%  70.9%
    lost<8px 0.2%   0.5%   ~1%    ~2%   4.0%   7.7%  ~14%  21.3%

WHAT THIS IS NOT
----------------
This is not true ground-sample-distance matching. Re-tiling WiSARD at a larger footprint
would widen the ground area per tile as well as shrinking the objects; downscaling the
input shrinks the objects while holding the footprint fixed. Both change apparent object
size, which is the hypothesis under test, but they are not the same operation. The honest
version is unavailable: the raw WiSARD frames were deleted, only the 640 px tiles survive.

FALSIFICATION
-------------
If no imgsz beats the imgsz=640 baseline (tile AP50 0.2567) by more than one bootstrap CI
step, the scale hypothesis is dead. Report it as a negative result and keep the existing
narrative. Do not go looking for a friendlier subsample.

Caches land in results/cache/scale_<run>_sz<imgsz>[_st<stride>].npz -- a separate namespace
from scripts/10_bootstrap_crossdomain.py's flags_<run>.npz, which the published
cross-domain numbers depend on and which this script must never overwrite.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.eval.bootstrap import (  # noqa: E402
    DEFAULT_BINS,
    ap_from_hist,
    average_precision,
    match_detections,
    percentile_ci,
    recall_at_fp_budget,
    recall_at_fp_budget_hist,
    sequence_histogram,
)

# Reuse script 10's helpers rather than growing a fourth copy of the sequence parser.
_spec = importlib.util.spec_from_file_location(
    "_bootstrap_crossdomain", ROOT / "scripts" / "10_bootstrap_crossdomain.py"
)
_x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_x)
sequence_of = _x.sequence_of
load_gt = _x.load_gt
TILES = _x.TILES
PERSON = _x.PERSON

CACHE = ROOT / "results" / "cache"
DEFAULT_SIZES = [640, 512, 448, 384, 320, 256, 192, 160]


def cache_path(run: str, imgsz: int, stride: int) -> Path:
    suffix = f"_st{stride}" if stride > 1 else ""
    return CACHE / f"scale_{run}_sz{imgsz}{suffix}.npz"


def tile_list(stride: int) -> list[Path]:
    """All tiles, or a sequence-stratified subsample.

    Taking every Nth tile from the globally sorted list keeps each sequence's share of
    the sample proportional to its share of the corpus, because the sort groups tiles by
    sequence. It is deterministic, so a pilot and its full run are directly comparable.
    """
    images = sorted((TILES / "images" / "test").iterdir())
    return images if stride <= 1 else images[::stride]


def collect(run: str, imgsz: int, stride: int, conf_floor: float, batch: int,
            max_det: int, fp_budget: float, n_bins: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "checkpoints" / run / "weights" / "best.pt"))
    lbl_dir = TILES / "labels" / "test"
    images = tile_list(stride)

    per_seq: dict[str, dict] = {}
    all_flags: list[tuple[float, bool]] = []

    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        results = model.predict(
            [str(p) for p in chunk], imgsz=imgsz, conf=conf_floor, max_det=max_det,
            device="0", verbose=False, stream=False,
        )
        for p, r in zip(chunk, results):
            seq = sequence_of(p.stem)
            d = per_seq.setdefault(seq, {"flags": [], "n_gt": 0, "n_tiles": 0})
            h, w = r.orig_shape
            gts = load_gt(lbl_dir / f"{p.stem}.txt", w, h)
            preds = []
            b = r.boxes
            if b is not None and len(b):
                for cls, cf, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()):
                    if int(cls) == PERSON:
                        preds.append((float(cf), tuple(xyxy)))
            flags, n_gt = match_detections(preds, gts, iou_thr=0.5)
            d["flags"].extend(flags)
            d["n_gt"] += n_gt
            d["n_tiles"] += 1
            all_flags.extend(flags)
        if (i // batch) % 60 == 0:
            print(f"  [{run} @{imgsz}] {i + len(chunk)}/{len(images)} tiles, "
                  f"{len(all_flags)} dets", flush=True)

    seqs = sorted(per_seq)
    n_gt_total = sum(per_seq[s]["n_gt"] for s in seqs)
    n_tiles_total = sum(per_seq[s]["n_tiles"] for s in seqs)

    exact_ap = average_precision(all_flags, n_gt_total)
    exact_rec, exact_thr, exact_fpf = recall_at_fp_budget(
        all_flags, n_gt_total, n_tiles_total, fp_budget
    )

    tp = np.stack([sequence_histogram(per_seq[s]["flags"], n_bins)[0] for s in seqs])
    fp = np.stack([sequence_histogram(per_seq[s]["flags"], n_bins)[1] for s in seqs])

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(run, imgsz, stride),
        sequences=np.array(seqs),
        tp=tp, fp=fp,
        n_gt=np.array([per_seq[s]["n_gt"] for s in seqs]),
        n_tiles=np.array([per_seq[s]["n_tiles"] for s in seqs]),
        exact=np.array([exact_ap, exact_rec, exact_thr, exact_fpf]),
        meta=np.array([conf_floor, max_det, n_bins, len(all_flags), imgsz, stride], dtype=float),
    )
    print(f"[cache] {run} @{imgsz}: {len(all_flags)} dets, {len(seqs)} seqs, "
          f"{n_tiles_total} tiles, {n_gt_total} gt", flush=True)
    return load_cache(run, imgsz, stride)


def load_cache(run: str, imgsz: int, stride: int) -> dict:
    z = np.load(cache_path(run, imgsz, stride), allow_pickle=False)
    return {
        "sequences": [str(s) for s in z["sequences"]],
        "tp": z["tp"], "fp": z["fp"],
        "n_gt": z["n_gt"], "n_tiles": z["n_tiles"],
        "exact": z["exact"], "meta": z["meta"],
    }


def bootstrap(c: dict, n_boot: int, fp_budget: float, seed: int) -> dict:
    tp, fp, n_gt, n_tiles = c["tp"], c["fp"], c["n_gt"], c["n_tiles"]
    k = len(c["sequences"])
    rng = random.Random(seed)

    ap_s, rec_s = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        t = tp[idx].sum(axis=0)
        f = fp[idx].sum(axis=0)
        g = int(n_gt[idx].sum())
        n = int(n_tiles[idx].sum())
        ap_s.append(ap_from_hist(t, f, g))
        rec_s.append(recall_at_fp_budget_hist(t, f, g, n, fp_budget))

    ap_lo, ap_hi = percentile_ci(ap_s)
    r_lo, r_hi = percentile_ci(rec_s)
    e_ap, e_rec, e_thr, e_fpf = c["exact"]
    return {
        "imgsz": int(c["meta"][4]),
        "downscale": round(640.0 / float(c["meta"][4]), 3),
        "sequences": k,
        "tiles": int(n_tiles.sum()),
        "gt_boxes": int(n_gt.sum()),
        "detections": int(c["meta"][3]),
        "AP50": round(float(e_ap), 5),
        "AP50_ci_low": round(ap_lo, 5),
        "AP50_ci_high": round(ap_hi, 5),
        "recall_at_fp_budget": round(float(e_rec), 5),
        "recall_ci_low": round(r_lo, 5),
        "recall_ci_high": round(r_hi, 5),
        "operating_threshold": round(float(e_thr), 5),
        "achieved_fp_per_tile": round(float(e_fpf), 5),
        "fp_budget_per_tile": fp_budget,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES,
                    help="inference imgsz values; must be multiples of 32")
    ap.add_argument("--tile-stride", type=int, default=1,
                    help="1 = all 37,058 tiles. >1 takes every Nth for a fast pilot.")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--fp-budget", type=float, default=0.1, help="FP per tile")
    ap.add_argument("--conf-floor", type=float, default=0.005)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out-tag", default="", help="suffix for the output filenames")
    cli = ap.parse_args()

    bad = [s for s in cli.sizes if s % 32]
    if bad:
        print(f"[fatal] imgsz must be a multiple of the model stride 32; got {bad}", file=sys.stderr)
        return 2

    rows = []
    for imgsz in cli.sizes:
        cf = cache_path(cli.run, imgsz, cli.tile_stride)
        if cf.exists() and not cli.recompute:
            print(f"[cached] {cli.run} @{imgsz}", flush=True)
            c = load_cache(cli.run, imgsz, cli.tile_stride)
        else:
            print(f"[collect] {cli.run} @{imgsz} (stride {cli.tile_stride})", flush=True)
            c = collect(cli.run, imgsz, cli.tile_stride, cli.conf_floor, cli.batch,
                        cli.max_det, cli.fp_budget, cli.bins)
        r = bootstrap(c, cli.n_boot, cli.fp_budget, cli.seed)
        rows.append(r)
        print(f"[ok] imgsz {imgsz:4d} (s={r['downscale']:.2f})  "
              f"AP50 {r['AP50']:.4f} [{r['AP50_ci_low']:.4f}, {r['AP50_ci_high']:.4f}]  "
              f"recall@{cli.fp_budget}FP/tile {r['recall_at_fp_budget']:.4f} "
              f"[{r['recall_ci_low']:.4f}, {r['recall_ci_high']:.4f}]  "
              f"@conf {r['operating_threshold']:.3f}", flush=True)

    tag = cli.out_tag or (f"_st{cli.tile_stride}" if cli.tile_stride > 1 else "")
    csv_path = ROOT / "results" / f"scale_sweep_wisard{tag}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = next((r for r in rows if r["imgsz"] == 640), None)
    best = max(rows, key=lambda r: r["AP50"])
    verdict = "NOT TESTED (no imgsz=640 baseline in this sweep)"
    if base is not None:
        gain = best["AP50"] - base["AP50"]
        clears = best["AP50"] > base["AP50_ci_high"]
        verdict = (
            f"best imgsz {best['imgsz']} AP50 {best['AP50']:.4f} vs baseline "
            f"{base['AP50']:.4f} (gain {gain:+.4f}); "
            + ("SUPPORTED: best point clears the baseline CI upper bound"
               if clears else
               "NOT SUPPORTED: best point falls inside the baseline CI")
        )

    payload = {
        "domain": "wisard_vis_tiled",
        "class": "person",
        "run": cli.run,
        "iou_threshold": 0.5,
        "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence",
        "seed": cli.seed,
        "conf_floor": cli.conf_floor,
        "max_det": cli.max_det,
        "tile_stride": cli.tile_stride,
        "hypothesis": (
            "The zero-shot cross-domain drop is driven by a scale-distribution shift. "
            "VisDrone train persons are mean 20.88 px with 74.8 percent inside 8-32 px; "
            "WiSARD persons are mean 67.30 px with 79.5 percent at or above 32 px, all "
            "measured in the same 640 px tile space. Reducing the inference input size "
            "shrinks apparent object size back toward the trained band without retraining."
        ),
        "caveat": (
            "Reducing imgsz shrinks the objects while holding the ground footprint fixed; "
            "true ground-sample-distance matching would also widen the footprint. The two "
            "are not the same operation. Re-tiling from source is impossible here because "
            "the raw WiSARD frames were deleted and only the 640 px tiles survive."
        ),
        "verdict": verdict,
        "sweep": rows,
    }
    json_path = ROOT / "results" / f"scale_sweep_wisard{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[verdict] {verdict}")
    print(f"[write] {csv_path}\n[write] {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
