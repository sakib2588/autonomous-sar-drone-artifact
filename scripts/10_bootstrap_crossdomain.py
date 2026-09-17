#!/usr/bin/env python3
"""Sequence-level bootstrap CIs and the SAR operating point (R8 Steps 4 and 5).

Runs each model once over the tiled WiSARD set, flags every detection TP or FP
per tile, then resamples SEQUENCES with replacement to build percentile CIs.

The resampling unit is the sequence, not the tile and not the box. WiSARD frames
are consecutive video and strongly correlated within a flight, so resampling
tiles would treat near-duplicates as independent and shrink the interval to
something indefensible.

Two things learned the hard way on 2026-08-07, both fixed here:

1. **The naive resample loop does not finish.** Re-sorting every detection on
   each of 10,000 resamples is O(10,000 n log n); at a 0.001 confidence floor
   with 300 detections per tile that is millions of detections per resample. The
   first attempt ran 33 minutes on ONE model with no output and peaked at 4.3 GB
   with 974 MB swapped. Confidence histograms are additive, so a resample is now
   a vector add over 38 arrays plus one cumulative sum.

2. **Inference and statistics must be separable.** Collection costs ~50 min per
   model; tuning the bootstrap should not pay that again. Collection now caches
   per-sequence histograms, counts, and the exact point estimates, so re-running
   the statistics is seconds.

The reported point estimates come from the EXACT functions, computed once during
collection. Only the CI distribution uses the binned approximation, which agrees
with exact AP to within a few thousandths -- far inside the interval width.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

ROOT = Path(__file__).resolve().parents[1]
TILES = Path("sar-drone-data/processed/sar_wisard_tiled")
CACHE = ROOT / "results" / "cache"
PERSON = 0
RUNS = ["y11n_tiled640_s42", "y8n_tiled640_s42", "y11n_full640_s42", "y11n_tiled416_s42"]


def sequence_of(tile_stem: str) -> str:
    """Sequence a tile came from. Tile names are `<frame_stem>__t<x>_<y>`."""
    frame_stem = tile_stem.split("__t")[0]
    seq, _, frame = frame_stem.rpartition("_")
    if not frame.isdigit():
        raise ValueError(f"cannot derive a sequence from {tile_stem!r}")
    return seq


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    out = []
    for line in label_path.read_text().splitlines():
        t = line.split()
        if len(t) != 5 or int(t[0]) != PERSON:
            continue
        cx, cy, bw, bh = (float(v) for v in t[1:])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def collect(run: str, conf_floor: float, batch: int, max_det: int, fp_budget: float, n_bins: int) -> dict:
    """Run inference once and cache per-sequence histograms plus exact points."""
    from ultralytics import YOLO

    args = yaml.safe_load((ROOT / "checkpoints" / run / "args.yaml").read_text())
    imgsz = args["imgsz"]
    model = YOLO(str(ROOT / "checkpoints" / run / "weights" / "best.pt"))

    img_dir, lbl_dir = TILES / "images" / "test", TILES / "labels" / "test"
    images = sorted(img_dir.iterdir())
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
        if (i // batch) % 40 == 0:
            print(f"  [{run}] {i + len(chunk)}/{len(images)} tiles, {len(all_flags)} dets", flush=True)

    seqs = sorted(per_seq)
    n_gt_total = sum(per_seq[s]["n_gt"] for s in seqs)
    n_tiles_total = sum(per_seq[s]["n_tiles"] for s in seqs)

    # Exact point estimates, computed once.
    exact_ap = average_precision(all_flags, n_gt_total)
    exact_rec, exact_thr, exact_fpf = recall_at_fp_budget(
        all_flags, n_gt_total, n_tiles_total, fp_budget
    )

    tp = np.stack([sequence_histogram(per_seq[s]["flags"], n_bins)[0] for s in seqs])
    fp = np.stack([sequence_histogram(per_seq[s]["flags"], n_bins)[1] for s in seqs])

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / f"flags_{run}.npz",
        sequences=np.array(seqs),
        tp=tp, fp=fp,
        n_gt=np.array([per_seq[s]["n_gt"] for s in seqs]),
        n_tiles=np.array([per_seq[s]["n_tiles"] for s in seqs]),
        exact=np.array([exact_ap, exact_rec, exact_thr, exact_fpf]),
        meta=np.array([conf_floor, max_det, n_bins, len(all_flags)], dtype=float),
    )
    print(f"[cache] {run}: {len(all_flags)} detections, {len(seqs)} sequences", flush=True)
    return load_cache(run)


def load_cache(run: str) -> dict:
    z = np.load(CACHE / f"flags_{run}.npz", allow_pickle=False)
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
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--fp-budget", type=float, default=0.1, help="FP per tile")
    ap.add_argument("--conf-floor", type=float, default=0.005)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--recompute", action="store_true", help="ignore cached inference")
    cli = ap.parse_args()

    out: dict[str, dict] = {}
    for run in cli.runs:
        cache_file = CACHE / f"flags_{run}.npz"
        if cache_file.exists() and not cli.recompute:
            print(f"[cached] {run}", flush=True)
            c = load_cache(run)
        else:
            print(f"[collect] {run}", flush=True)
            c = collect(run, cli.conf_floor, cli.batch, cli.max_det, cli.fp_budget, cli.bins)
        print(f"[boot] {run} ({cli.n_boot} resamples over {len(c['sequences'])} sequences)", flush=True)
        out[run] = bootstrap(c, cli.n_boot, cli.fp_budget, cli.seed)
        r = out[run]
        print(f"[ok] {run} AP50 {r['AP50']:.4f} [{r['AP50_ci_low']:.4f}, {r['AP50_ci_high']:.4f}]  "
              f"recall@{cli.fp_budget}FP/tile {r['recall_at_fp_budget']:.4f} "
              f"[{r['recall_ci_low']:.4f}, {r['recall_ci_high']:.4f}] @conf {r['operating_threshold']:.3f}",
              flush=True)

    payload = {
        "domain": "wisard_vis_tiled",
        "class": "person",
        "iou_threshold": 0.5,
        "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence",
        "seed": cli.seed,
        "conf_floor": cli.conf_floor,
        "max_det": cli.max_det,
        "note": (
            "Resampled over sequences, not tiles or boxes: WiSARD frames are "
            "consecutive video and strongly correlated within a flight, so a "
            "tile-level bootstrap would return an indefensibly narrow interval. "
            "Point estimates are exact; the CI distribution uses a "
            f"{cli.bins}-bin confidence histogram, which agrees with exact AP to "
            "within a few thousandths. Per-tile metrics with 20 percent tile "
            "overlap; full-frame numbers require merge_tile_detections."
        ),
        "runs": out,
    }
    dest = ROOT / "results" / "cross_domain_bootstrap.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[write] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
