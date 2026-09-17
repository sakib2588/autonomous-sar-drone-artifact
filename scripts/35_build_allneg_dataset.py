#!/usr/bin/env python3
"""Build the negative-fraction control training set: every available VisDrone negative retained.

WHY
`select_tiles`' own docstring says "deliberately retaining ~15-20% negatives is the fix" for a
detector that "has never seen empty ground and hallucinates people in bushes". That fix never
landed. `negative_keep_frac` is a fraction OF AVAILABLE NEGATIVES applied PER IMAGE, and VisDrone is
object-dense: 4,221 negatives against 39,984 positives, with most frames yielding so few that
round(n * 0.175) is zero. The realised training set is 1.33% negative, not 15-20%.

Phase A measured why that matters. On empty ground the detector fires at 0.0597 FP/tile in-domain
and 0.0605 out-of-domain -- a ratio of 1.01. It is not confused by wilderness; it was trained to
expect an object in every tile and says so wherever you point it.

This builds the largest-dose control the corpus can support: 9.55% negative. Not the 85.8% WiSARD
actually presents -- that would need 241,593 negatives and only 4,221 exist, which is itself a
finding about training urban-aerial detectors for wilderness use.

WHAT IT DELIBERATELY DOES NOT DO
  - never calls 03_build_rgb_datasets.write_configs(), which would rewrite configs/sar_rgb_tiled.yaml
    and configs/sar_rgb_full.yaml back to the slow HDD paths and drop their provenance comments
  - never writes into data/processed/ or the published NVMe sar_rgb_tiled/
  - rebuilds ONLY train. val and test are symlinked from the baseline so the control is compared on
    byte-identical evaluation data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.data.taxonomy_rgb import SAR_RGB_CLASSES  # noqa: E402
from src.sar.data.tiling import select_tiles, tile_dataset  # noqa: E402

# module filename starts with a digit, so import by path
_spec = importlib.util.spec_from_file_location("_b3", ROOT / "scripts" / "03_build_rgb_datasets.py")
_b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b3)          # safe: main() is guarded, write_configs is NOT called
read_yolo_labels_px = _b3.read_yolo_labels_px
write_yolo_label = _b3.write_yolo_label

RAW = ROOT / "data" / "raw" / "VisDrone"
BASE = Path("sar-drone-data/processed/sar_rgb_tiled")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sar-drone-data/processed/sar_rgb_tiled_allneg")
    ap.add_argument("--neg-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    out = Path(cli.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # val/test symlinked so the control and the baseline are evaluated on identical bytes
    for kind in ("images", "labels"):
        for split in ("val", "test"):
            link = out / kind / split
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(BASE / kind / split, target_is_directory=True)
    print("val and test symlinked from the baseline dataset (identical eval bytes)")

    img_dir, lbl_dir = RAW / "images" / "train", RAW / "labels" / "train"
    frames = sorted(img_dir.glob("*.jpg"))
    n_pos = n_neg = 0
    for k, ip in enumerate(frames):
        with Image.open(ip) as im:
            w, h = im.size
            boxes = read_yolo_labels_px(lbl_dir / f"{ip.stem}.txt", w, h)
            tiled = tile_dataset(w, h, boxes, tile_size=640, overlap=0.2, min_visible_frac=0.3)
            selected = select_tiles(tiled, negative_keep_frac=cli.neg_frac, seed=cli.seed)
            for idx, (tile, labels) in enumerate(selected):
                tx0, ty0, tx1, ty1 = tile
                name = f"{ip.stem}_t{idx:03d}"
                im.crop(tile).save(out / "images" / "train" / f"{name}.jpg", quality=95)
                write_yolo_label(out / "labels" / "train" / f"{name}.txt", labels, tx1 - tx0, ty1 - ty0)
                if labels: n_pos += 1
                else:      n_neg += 1
        if k % 1000 == 0:
            print(f"  {k}/{len(frames)}  pos {n_pos}  neg {n_neg}", flush=True)

    tot = n_pos + n_neg
    cfg = ROOT / "configs" / "sar_rgb_tiled_allneg.yaml"
    cfg.write_text(
        "# Negative-fraction CONTROL. Identical to sar_rgb_tiled except that the train split keeps\n"
        "# EVERY object-free tile instead of the 17.5-percent-of-negatives sample the baseline used.\n"
        f"# Realised composition: {n_neg}/{tot} = {100*n_neg/tot:.2f} percent negative, against the\n"
        "# baseline's 1.33 percent. WiSARD deployment is 85.79 percent negative; matching that would\n"
        "# need 241,593 negatives and VisDrone contains 4,221, which is a finding in itself.\n"
        "#\n"
        "# val and test are SYMLINKS to the baseline dataset so both models are scored on the same\n"
        "# bytes. Do not regenerate them here.\n"
        f"path: {out}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(SAR_RGB_CLASSES)) + "\n"
    )
    rep = {"out": str(out), "negative_keep_frac": cli.neg_frac, "seed": cli.seed,
           "train_images": len(frames), "tiles": tot, "positive": n_pos, "negative": n_neg,
           "negative_fraction": round(n_neg / tot, 5),
           "baseline_negative_fraction": 0.01333,
           "wisard_negative_fraction": 0.85793,
           "val_test": "symlinked from sar_rgb_tiled"}
    (ROOT / "results" / "dataset_build_report_allneg.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"\n  tiles {tot}  positive {n_pos}  negative {n_neg}  -> {100*n_neg/tot:.2f}% negative")
    print(f"[write] {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
