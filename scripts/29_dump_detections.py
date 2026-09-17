#!/usr/bin/env python3
"""Dump per-detection records on the WiSARD tiles, at a chosen inference scale.

The cached histograms in results/cache/ keep only 2000-bin confidence counts per sequence. That is
enough for AP and for bootstrapping, and useless for anything that needs to know WHICH detection
went where. Two analyses need exactly that:

  scripts/30_multiscale.py  needs boxes at two scales in the same coordinate frame, to ask whether
                            a detection found at 640 is corroborated by one at 320.
  scripts/31_encounter.py   needs detections grouped by sequence and frame, to ask whether a person
                            visible across several frames is found in at least one of them.

Ultralytics rescales predictions back to orig_shape, which for these tiles is 640x640 regardless of
the inference input size, so boxes from different scales are directly comparable without any
coordinate surgery.

Written to run alongside a training job: batch is small by default and the script prints VRAM so a
run that starts crowding the GPU can be stopped before it disturbs anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.eval.bootstrap import match_detections  # noqa: E402

TILES = Path("sar-drone-data/processed/sar_wisard_tiled")
PERSON = 0
OUT = ROOT / "results" / "cache"


def load_gt(p: Path, w: int, h: int):
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) == 5 and int(t[0]) == PERSON:
            cx, cy, bw, bh = (float(v) for v in t[1:])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf-floor", type=float, default=0.005)
    ap.add_argument("--max-det", type=int, default=100)
    cli = ap.parse_args()

    from ultralytics import YOLO
    import torch

    model = YOLO(str(ROOT / "checkpoints" / cli.run / "weights" / "best.pt"))
    img_dir, lbl_dir = TILES / "images" / "test", TILES / "labels" / "test"
    images = sorted(img_dir.iterdir())

    stems, n_gt_per_tile = [], []
    d_tile, d_conf, d_box, d_tp = [], [], [], []

    for i in range(0, len(images), cli.batch):
        chunk = images[i : i + cli.batch]
        res = model.predict([str(p) for p in chunk], imgsz=cli.imgsz, conf=cli.conf_floor,
                            max_det=cli.max_det, device="0", verbose=False, stream=False)
        for p, r in zip(chunk, res):
            ti = len(stems)
            stems.append(p.stem)
            h, w = r.orig_shape
            gts = load_gt(lbl_dir / f"{p.stem}.txt", w, h)
            n_gt_per_tile.append(len(gts))
            preds = []
            b = r.boxes
            if b is not None and len(b):
                for cls, cf, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist()):
                    if int(cls) == PERSON:
                        preds.append((float(cf), tuple(xyxy)))
            preds.sort(key=lambda t: -t[0])
            flags, _ = match_detections(preds, gts, iou_thr=0.5)
            for (cf, box), (_, is_tp) in zip(preds, flags):
                d_tile.append(ti); d_conf.append(cf); d_box.append(box); d_tp.append(is_tp)
        if (i // cli.batch) % 150 == 0:
            vram = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"  [{cli.imgsz}] {i + len(chunk)}/{len(images)} tiles, "
                  f"{len(d_conf)} dets, {vram:.2f} GB reserved", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"dets_{cli.run}_sz{cli.imgsz}.npz"
    np.savez_compressed(
        dest,
        stems=np.array(stems),
        n_gt=np.array(n_gt_per_tile, dtype=np.int32),
        det_tile=np.array(d_tile, dtype=np.int32),
        det_conf=np.array(d_conf, dtype=np.float32),
        det_box=np.array(d_box, dtype=np.float32).reshape(-1, 4),
        det_tp=np.array(d_tp, dtype=bool),
        meta=np.array([cli.imgsz, cli.conf_floor, cli.max_det], dtype=float),
    )
    print(f"[write] {dest}  {len(stems)} tiles, {len(d_conf)} detections, "
          f"{int(np.sum(d_tp))} TP, {int(np.sum(n_gt_per_tile))} GT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
