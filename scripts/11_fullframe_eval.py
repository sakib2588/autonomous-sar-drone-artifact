#!/usr/bin/env python3
"""Full-frame tiled-inference evaluation (R8 Step 3) -- the deployment number.

Why this exists on top of the per-tile numbers. The plan sets the SAR operating
point at 0.1 false positives PER FRAME ("one spurious box every 10 frames of
operator attention"). The per-tile evaluation measures 0.1 FP per TILE, and at
roughly 14 tiles per frame that is ~1.4 FP/frame -- fourteen times the operator
budget. Any recall quoted against the per-tile budget is optimistic against the
spec. Frame-level is also how the drone actually runs: tile, detect, merge.

Merging matters for a second reason. Tiles overlap by 20 percent, so a person
near a seam is detected independently in two tiles. Per-tile metrics count that
twice, in both the ground truth and the predictions. `merge_tile_detections`
shifts every detection into full-frame coordinates and runs per-class NMS across
seams, which is the only way the counts correspond to real objects.

Ground truth here is the ORIGINAL full-frame WiSARD annotation, not the
tile-local labels: same clip policy and same minimum box size used when the
tiles were built, but no per-tile visibility filter, because at frame level
there are no tile boundaries to be truncated by.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sar.data.sar_eval import parse_wisard_label  # noqa: E402
from src.sar.data.tiling import merge_tile_detections  # noqa: E402
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
SRC = Path("sar-drone-data/raw/WiSARD/extracted_vis")
TILES = Path("sar-drone-data/processed/sar_wisard_tiled")
CACHE = ROOT / "results" / "cache"
PERSON = 0
RUNS = ["y11n_tiled640_s42", "y8n_tiled640_s42", "y11n_full640_s42", "y11n_tiled416_s42"]
MIN_BOX_PX = 4.0  # must match scripts/09_build_wisard_tiles.py


def sequence_of(frame_stem: str) -> str:
    seq, _, frame = frame_stem.rpartition("_")
    if not frame.isdigit():
        raise ValueError(f"cannot derive a sequence from {frame_stem!r}")
    return seq


def tile_origin(tile_stem: str) -> tuple[int, int]:
    """(x0, y0) of a tile from its name `<frame_stem>__t<x>_<y>`."""
    _, _, tag = tile_stem.partition("__t")
    x, _, y = tag.partition("_")
    return int(x), int(y)


def group_tiles_by_frame() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted((TILES / "images" / "test").iterdir()):
        groups[p.stem.split("__t")[0]].append(p)
    return dict(groups)


_SOURCE_INDEX: dict[str, Path] = {}


def source_index() -> dict[str, Path]:
    """Map every source frame stem to its path, built once.

    Reconstructing `SRC/<sequence>/<stem>.jpg` does NOT work: 10 of the 38
    sequences name their frames `<sequence>.mp4_00000.jpg` with a five-digit
    counter, so the directory name and the filename prefix differ and the
    extension varies between .jpg and .jpeg. Indexing the filesystem sidesteps
    every naming variant instead of encoding the ones seen so far.
    """
    if not _SOURCE_INDEX:
        for d in SRC.iterdir():
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    _SOURCE_INDEX[p.stem] = p
    return _SOURCE_INDEX


def frame_source(frame_stem: str) -> Path | None:
    return source_index().get(frame_stem)


def frame_gt(frame_stem: str) -> tuple[list[tuple[float, float, float, float]], int, int]:
    """Original full-frame person boxes in pixel xyxy, plus image size."""
    src = frame_source(frame_stem)
    if src is None:
        raise FileNotFoundError(f"no source image for {frame_stem}")
    W, H = Image.open(src).size
    lbl = src.with_suffix(".txt")
    boxes: list[tuple[float, float, float, float]] = []
    if lbl.exists():
        for b in parse_wisard_label(lbl, on_out_of_range="clip"):
            bw, bh = b.w * W, b.h * H
            if (bw * bh) ** 0.5 < MIN_BOX_PX:
                continue
            boxes.append(
                ((b.cx - b.w / 2) * W, (b.cy - b.h / 2) * H,
                 (b.cx + b.w / 2) * W, (b.cy + b.h / 2) * H)
            )
    # A missing label file means zero humans, not unannotated.
    return boxes, W, H


def collect(run: str, conf_floor: float, max_det: int, nms_iou: float) -> dict:
    from ultralytics import YOLO

    args = yaml.safe_load((ROOT / "checkpoints" / run / "args.yaml").read_text())
    imgsz = args["imgsz"]
    model = YOLO(str(ROOT / "checkpoints" / run / "weights" / "best.pt"))

    groups = group_tiles_by_frame()
    per_seq: dict[str, dict] = {}
    all_flags: list[tuple[float, bool]] = []
    n_frames_done = 0

    for frame_stem, tiles in groups.items():
        gts, W, H = frame_gt(frame_stem)
        results = model.predict(
            [str(p) for p in tiles], imgsz=imgsz, conf=conf_floor, max_det=max_det,
            device="0", verbose=False, stream=False,
        )
        dets_per_tile = []
        for p, r in zip(tiles, results):
            tx, ty = tile_origin(p.stem)
            th, tw = r.orig_shape
            dets = []
            b = r.boxes
            if b is not None and len(b):
                for cls, cf, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()):
                    if int(cls) == PERSON:
                        x0, y0, x1, y1 = xyxy
                        dets.append((PERSON, float(cf), x0, y0, x1, y1))
            dets_per_tile.append(((tx, ty, tx + tw, ty + th), dets))

        merged = merge_tile_detections(dets_per_tile, iou_thresh=nms_iou)
        preds = [(conf, (x0, y0, x1, y1)) for _, conf, x0, y0, x1, y1 in merged]
        flags, n_gt = match_detections(preds, gts, iou_thr=0.5)

        seq = sequence_of(frame_stem)
        d = per_seq.setdefault(seq, {"flags": [], "n_gt": 0, "n_frames": 0})
        d["flags"].extend(flags)
        d["n_gt"] += n_gt
        d["n_frames"] += 1
        all_flags.extend(flags)

        n_frames_done += 1
        if n_frames_done % 250 == 0:
            print(f"  [{run}] {n_frames_done}/{len(groups)} frames, {len(all_flags)} dets", flush=True)

    seqs = sorted(per_seq)
    n_gt_total = sum(per_seq[s]["n_gt"] for s in seqs)
    n_frames_total = sum(per_seq[s]["n_frames"] for s in seqs)

    exact_ap = average_precision(all_flags, n_gt_total)
    exact_rec, exact_thr, exact_fpf = recall_at_fp_budget(
        all_flags, n_gt_total, n_frames_total, 0.1
    )

    tp = np.stack([sequence_histogram(per_seq[s]["flags"], DEFAULT_BINS)[0] for s in seqs])
    fp = np.stack([sequence_histogram(per_seq[s]["flags"], DEFAULT_BINS)[1] for s in seqs])

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / f"fullframe_{run}.npz",
        sequences=np.array(seqs), tp=tp, fp=fp,
        n_gt=np.array([per_seq[s]["n_gt"] for s in seqs]),
        n_frames=np.array([per_seq[s]["n_frames"] for s in seqs]),
        exact=np.array([exact_ap, exact_rec, exact_thr, exact_fpf]),
        meta=np.array([conf_floor, max_det, nms_iou, len(all_flags)], dtype=float),
    )
    print(f"[cache] {run}: {n_frames_total} frames, {n_gt_total} GT, {len(all_flags)} merged dets",
          flush=True)
    return load_cache(run)


def load_cache(run: str) -> dict:
    z = np.load(CACHE / f"fullframe_{run}.npz", allow_pickle=False)
    return {
        "sequences": [str(s) for s in z["sequences"]],
        "tp": z["tp"], "fp": z["fp"], "n_gt": z["n_gt"],
        "n_frames": z["n_frames"], "exact": z["exact"], "meta": z["meta"],
    }


def bootstrap(c: dict, n_boot: int, fp_budget: float, seed: int) -> dict:
    tp, fp, n_gt, n_frames = c["tp"], c["fp"], c["n_gt"], c["n_frames"]
    k = len(c["sequences"])
    rng = random.Random(seed)
    ap_s, rec_s = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        t, f = tp[idx].sum(axis=0), fp[idx].sum(axis=0)
        g, n = int(n_gt[idx].sum()), int(n_frames[idx].sum())
        ap_s.append(ap_from_hist(t, f, g))
        rec_s.append(recall_at_fp_budget_hist(t, f, g, n, fp_budget))
    ap_lo, ap_hi = percentile_ci(ap_s)
    r_lo, r_hi = percentile_ci(rec_s)
    e_ap, e_rec, e_thr, e_fpf = c["exact"]
    return {
        "sequences": k,
        "frames": int(n_frames.sum()),
        "gt_boxes": int(n_gt.sum()),
        "merged_detections": int(c["meta"][3]),
        "AP50": round(float(e_ap), 5),
        "AP50_ci_low": round(ap_lo, 5),
        "AP50_ci_high": round(ap_hi, 5),
        "recall_at_fp_budget": round(float(e_rec), 5),
        "recall_ci_low": round(r_lo, 5),
        "recall_ci_high": round(r_hi, 5),
        "operating_threshold": round(float(e_thr), 5),
        "achieved_fp_per_frame": round(float(e_fpf), 5),
        "fp_budget_per_frame": fp_budget,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--fp-budget", type=float, default=0.1, help="FP per FRAME")
    ap.add_argument("--conf-floor", type=float, default=0.005)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--recompute", action="store_true")
    cli = ap.parse_args()

    out: dict[str, dict] = {}
    for run in cli.runs:
        cf = CACHE / f"fullframe_{run}.npz"
        if cf.exists() and not cli.recompute:
            print(f"[cached] {run}", flush=True)
            c = load_cache(run)
        else:
            print(f"[collect] {run}", flush=True)
            c = collect(run, cli.conf_floor, cli.max_det, cli.nms_iou)
        out[run] = bootstrap(c, cli.n_boot, cli.fp_budget, cli.seed)
        r = out[run]
        print(f"[ok] {run} AP50 {r['AP50']:.4f} [{r['AP50_ci_low']:.4f}, {r['AP50_ci_high']:.4f}]  "
              f"recall@{cli.fp_budget}FP/frame {r['recall_at_fp_budget']:.4f} "
              f"[{r['recall_ci_low']:.4f}, {r['recall_ci_high']:.4f}] @conf {r['operating_threshold']:.3f}",
              flush=True)

    payload = {
        "domain": "wisard_vis_fullframe_merged",
        "class": "person",
        "iou_threshold": 0.5,
        "nms_iou": cli.nms_iou,
        "n_bootstrap": cli.n_boot,
        "resampling_unit": "sequence",
        "seed": cli.seed,
        "note": (
            "Frame-level: tiles are detected independently then merged with "
            "merge_tile_detections (per-class NMS across seams) into full-frame "
            "coordinates, matching how the deployed system runs. The false-positive "
            "budget is per FRAME here, which is what the plan specifies; the "
            "per-tile results in cross_domain_bootstrap.json use a budget roughly "
            "14x looser and are therefore optimistic as an operating point. "
            "Ground truth is the original full-frame annotation, so tile-overlap "
            "double counting is removed from both sides."
        ),
        "runs": out,
    }
    dest = ROOT / "results" / "cross_domain_fullframe.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[write] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
