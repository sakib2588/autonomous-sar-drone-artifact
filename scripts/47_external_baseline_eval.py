#!/usr/bin/env python3
"""Score an independent, aerial-domain-pretrained detector on the WiSARD tiles.

WHY THIS SCRIPT
----------------
Round-5 review left one objection between this paper and Accept: every row in
Table I is our own detector. The only non-project weights on disk are COCO
(`yolo11n.pt`, `yolov8n.pt`), and a COCO detector scores near zero on nadir
aerial tiles -- the review protocol explicitly warns against adding that as a
strawman baseline (see `results/protocol_comparison.txt` discussion and
`notes/20260827-handover-submission-ready.md`).

The external checkpoint here is `mshamrai/yolov8s-visdrone` (Hugging Face,
`openrail` license): a YOLOv8s trained by a third party on VisDrone2019-DET,
the same corpus this project trains on, with self-reported mAP@0.5(box)=0.408.
It is genuinely aerial-domain, genuinely independent (never touched WiSARD or
this project's tiling pipeline), and its 10-class output uses VisDrone's
canonical class order -- `pedestrian` (0) and `people` (1) map to "person"
under the exact same `VISDRONE_MAP` this project already uses
(`src/sar/data/taxonomy_rgb.py`), so the comparison is apples to apples on the
label side.

This is NOT a training run. It is one inference pass over the same 37,058
WiSARD tiles the project's own models were scored on, reusing the identical
matching and bootstrap machinery from `scripts/10_bootstrap_crossdomain.py`
(same IoU-0.5 greedy match, same sequence-level 10,000-resample bootstrap) so
the resulting AP50 + CI is directly comparable to every row already in
`results/cross_domain_bootstrap.json`.

Deliberately writes to a SEPARATE file, `results/external_baseline_bootstrap.
json`, rather than overwriting the existing one. `10_bootstrap_crossdomain.py`
has a known footgun: its `main()` writes only the runs passed on that
invocation, so re-running it with a different --runs list silently drops every
other model's numbers from the JSON. This script never touches that file, so
Table I integration is a deliberate manual merge, not an accidental clobber.

USAGE
  .venv/bin/python scripts/47_external_baseline_eval.py
  .venv/bin/python scripts/47_external_baseline_eval.py --recompute
  .venv/bin/python scripts/47_external_baseline_eval.py --imgsz 960   # try native res too
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sar.data.taxonomy_rgb import VISDRONE_MAP  # noqa: E402
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
PERSON = 0  # our own label files: class 0 is always "person" (see 09_build_wisard_tiles.py)

# VisDrone class ids that map to "person" under VISDRONE_MAP, in the exact
# order mshamrai/yolov8s-visdrone (and any other model trained on raw
# VisDrone2019-DET with the canonical class order) emits them.
_VISDRONE_ORDER = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
PERSON_VISDRONE_IDS = frozenset(
    i for i, name in enumerate(_VISDRONE_ORDER) if VISDRONE_MAP.get(name) == "person"
)

EXTERNAL_BASELINES = {
    "ext_y8s_visdrone_hf": {
        "weights": ROOT / "checkpoints" / "ext_yolov8s_visdrone" / "weights" / "best.pt",
        "source": "mshamrai/yolov8s-visdrone (Hugging Face, openrail license)",
        "trained_on": "VisDrone2019-DET, third-party, never touched WiSARD or this project",
        "self_reported_map50": 0.408,
        "person_class_ids": PERSON_VISDRONE_IDS,
    },
}


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


def collect(
    run: str, weights: Path, person_class_ids: frozenset[int],
    imgsz: int, conf_floor: float, batch: int, max_det: int, fp_budget: float, n_bins: int,
    limit: int | None = None,
) -> dict:
    """Run inference once and cache per-sequence histograms plus exact points."""
    from ultralytics import YOLO

    if not weights.is_file():
        raise SystemExit(
            f"missing external weights {weights}\n"
            f"download: curl -sL -o {weights} "
            f"https://huggingface.co/mshamrai/yolov8s-visdrone/resolve/main/best.pt"
        )
    model = YOLO(str(weights))
    got_ids = set(int(k) for k in model.names)
    missing = person_class_ids - got_ids
    if missing:
        raise SystemExit(
            f"{run}: model.names is missing expected class ids {missing} "
            f"(got {sorted(got_ids)}). The checkpoint's class order changed -- "
            f"re-derive PERSON_VISDRONE_IDS before trusting this run."
        )

    img_dir, lbl_dir = TILES / "images" / "test", TILES / "labels" / "test"
    images = sorted(img_dir.iterdir())
    if limit is not None:
        images = images[:limit]
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
                    if int(cls) in person_class_ids:
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
    ap.add_argument("--imgsz", type=int, default=640, help="matches our own tiled640 runs for a fair comparison")
    ap.add_argument("--runs", nargs="*", default=list(EXTERNAL_BASELINES))
    ap.add_argument("--recompute", action="store_true", help="ignore cached inference")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                     help="path check only: score just the first N tiles, tag run name "
                          "'_smoke', never write the real cache or results file")
    cli = ap.parse_args()

    out: dict[str, dict] = {}
    for run in cli.runs:
        spec = EXTERNAL_BASELINES[run]
        run_name = f"{run}_smoke" if cli.smoke else run
        cache_file = CACHE / f"flags_{run_name}.npz"
        if cache_file.exists() and not cli.recompute and not cli.smoke:
            print(f"[cached] {run_name}", flush=True)
            c = load_cache(run_name)
        else:
            print(f"[collect] {run_name} <- {spec['source']}"
                  + (f"  SMOKE n={cli.smoke}" if cli.smoke else ""), flush=True)
            c = collect(
                run_name, spec["weights"], spec["person_class_ids"], cli.imgsz,
                cli.conf_floor, cli.batch, cli.max_det, cli.fp_budget, cli.bins,
                limit=cli.smoke,
            )
        print(f"[boot] {run_name} ({cli.n_boot} resamples over {len(c['sequences'])} sequences)", flush=True)
        result = bootstrap(c, cli.n_boot, cli.fp_budget, cli.seed)
        result["source"] = spec["source"]
        result["trained_on"] = spec["trained_on"]
        result["imgsz"] = cli.imgsz
        result["smoke"] = bool(cli.smoke)
        out[run_name] = result
        r = result
        print(f"[ok] {run_name} AP50 {r['AP50']:.4f} [{r['AP50_ci_low']:.4f}, {r['AP50_ci_high']:.4f}]  "
              f"recall@{cli.fp_budget}FP/tile {r['recall_at_fp_budget']:.4f} "
              f"[{r['recall_ci_low']:.4f}, {r['recall_ci_high']:.4f}] @conf {r['operating_threshold']:.3f}",
              flush=True)

    if cli.smoke:
        print(f"[smoke] path check only ({cli.smoke} tiles) -- results file NOT written, "
              f"accuracy numbers above are meaningless", flush=True)
        return 0

    dest = ROOT / "results" / "external_baseline_bootstrap.json"
    payload = {"domain": "wisard_vis_tiled", "class": "person", "iou_threshold": 0.5,
               "n_bootstrap": cli.n_boot, "resampling_unit": "sequence", "seed": cli.seed,
               "conf_floor": cli.conf_floor, "max_det": cli.max_det, "smoke": False,
               "note": (
                   "Independent, aerial-domain-pretrained detectors, never trained by this "
                   "project, scored on the same 37,058 WiSARD tiles under the same protocol "
                   "as results/cross_domain_bootstrap.json. Written to a SEPARATE file on "
                   "purpose -- merge into Table I by hand, do not overwrite the project-model "
                   "results file with this one."
               ),
               "runs": out}
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[write] {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
