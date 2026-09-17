#!/usr/bin/env python
"""Build the two RGB training datasets from VisDrone (R4/R5 prerequisite).

1. sar_rgb_tiled/  -- primary. 640x640 tiles, ~20% overlap, boxes clipped and
   dropped below 30% visible area, ~17.5% of empty tiles retained as negatives.
2. sar_rgb_full/    -- ablation. Original full frames, labels remapped to the
   3-class SAR_RGB_CLASSES taxonomy, no tiling (images symlinked, not copied --
   Ultralytics resizes to imgsz on the fly, no need to duplicate 3.7 GB).

Both write YOLO-format image/label pairs plus a `configs/*.yaml` Ultralytics
data config. Run once; re-running overwrites (idempotent).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sar.data.taxonomy_rgb import SAR_RGB_CLASSES
from src.sar.data.tiling import select_tiles, tile_dataset
from src.sar.data.visdrone import visdrone_id_to_rgb_id

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "VisDrone"
TILED_OUT = REPO_ROOT / "data" / "processed" / "sar_rgb_tiled"
FULL_OUT = REPO_ROOT / "data" / "processed" / "sar_rgb_full"
SPLITS = ("train", "val", "test")


def read_yolo_labels_px(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int, int]]:
    """Read YOLO-normalized labels, remap VisDrone id -> RGB id, return pixel xyxy."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        visdrone_id = int(parts[0])
        rgb_id = visdrone_id_to_rgb_id(visdrone_id)
        if rgb_id is None:
            continue
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x0 = (cx - w / 2) * img_w
        y0 = (cy - h / 2) * img_h
        x1 = (cx + w / 2) * img_w
        y1 = (cy + h / 2) * img_h
        boxes.append((rgb_id, round(x0), round(y0), round(x1), round(y1)))
    return boxes


def write_yolo_label(path: Path, boxes: list[tuple[int, int, int, int, int]], w: int, h: int) -> None:
    lines = []
    for cls_id, x0, y0, x1, y1 in boxes:
        cx = (x0 + x1) / 2 / w
        cy = (y0 + y1) / 2 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_tiled(split: str) -> dict:
    img_dir = RAW / "images" / split
    label_dir = RAW / "labels" / split
    out_img_dir = TILED_OUT / "images" / split
    out_label_dir = TILED_OUT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "tiles_written": 0, "positive_tiles": 0, "negative_tiles": 0}

    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            boxes = read_yolo_labels_px(label_dir / f"{stem}.txt", img_w, img_h)
            tiled = tile_dataset(img_w, img_h, boxes, tile_size=640, overlap=0.2, min_visible_frac=0.3)
            selected = select_tiles(tiled, negative_keep_frac=0.175, seed=42)

            for idx, (tile, labels) in enumerate(selected):
                tx0, ty0, tx1, ty1 = tile
                tile_w, tile_h = tx1 - tx0, ty1 - ty0
                crop = im.crop(tile)
                tile_name = f"{stem}_t{idx:03d}"
                crop.save(out_img_dir / f"{tile_name}.jpg", quality=95)
                write_yolo_label(out_label_dir / f"{tile_name}.txt", labels, tile_w, tile_h)
                stats["tiles_written"] += 1
                if labels:
                    stats["positive_tiles"] += 1
                else:
                    stats["negative_tiles"] += 1
        stats["images"] += 1

    return stats


def build_full(split: str) -> dict:
    img_dir = RAW / "images" / split
    label_dir = RAW / "labels" / split
    out_img_dir = FULL_OUT / "images" / split
    out_label_dir = FULL_OUT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0}
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        link_path = out_img_dir / img_path.name
        if not link_path.exists():
            link_path.symlink_to(img_path.resolve())

        with Image.open(img_path) as im:
            img_w, img_h = im.size
        boxes = read_yolo_labels_px(label_dir / f"{stem}.txt", img_w, img_h)
        write_yolo_label(out_label_dir / f"{stem}.txt", boxes, img_w, img_h)
        stats["images"] += 1
    return stats


def write_configs() -> None:
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(SAR_RGB_CLASSES))
    for cfg_name, out_dir in (("sar_rgb_tiled", TILED_OUT), ("sar_rgb_full", FULL_OUT)):
        cfg_path = REPO_ROOT / "configs" / f"{cfg_name}.yaml"
        cfg_path.write_text(
            f"path: {out_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"names:\n{names_yaml}\n"
        )


def main() -> None:
    import json

    (REPO_ROOT / "configs").mkdir(exist_ok=True)
    report = {"tiled": {}, "full": {}}

    for split in SPLITS:
        print(f"[tiled] {split} ...", flush=True)
        report["tiled"][split] = build_tiled(split)
        print(f"  {report['tiled'][split]}", flush=True)

    for split in SPLITS:
        print(f"[full] {split} ...", flush=True)
        report["full"][split] = build_full(split)
        print(f"  {report['full'][split]}", flush=True)

    write_configs()

    out = REPO_ROOT / "results" / "dataset_build_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"done, report at {out}")


if __name__ == "__main__":
    main()
