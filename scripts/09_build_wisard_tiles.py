#!/usr/bin/env python3
"""Build the tiled WiSARD cross-domain evaluation set (R8 input).

Why tile rather than resize. WiSARD VIS is MIXED resolution -- 1920x1080 for
16,000 of 25,735 images, but also 2704x1520, 2720x1530, 3840x2160 and 4096x2160.
A single resize-to-640 policy therefore applies a 3.0x reduction to some
sequences and 6.0x to others, putting the same 72.55 px mean person at ~24 px in
one and ~12 px in another, the latter near the stride-8 detection floor. The
resulting "domain gap" would be substantially a resolution artefact. Tiling
preserves native object scale across every sequence, and it matches how the
models were trained, so there is no train/test tiling mismatch.

Why frames are subsampled. Sequences are consecutive video: in the sample
sequence, 251 of 264 frames carry the same four people. Tiling all 25,735 frames
produces roughly 362,000 tiles -- about 21 GB and 11 hours of inference -- for
almost no additional information, because the frames are near-duplicates. Frame
stride keeps the eval honest about its own effective sample size. The retained
count is written to the manifest so the paper can state it.

Why EVERY tile is kept, unlike training. `select_tiles` deliberately samples
negatives down to ~17.5% for training, because a detector that never sees empty
ground hallucinates people in bushes. At evaluation the opposite is true: the
false-positive rate per frame is the quantity R8 Step 5 needs to set the SAR
operating point, and discarding negative tiles would make it unmeasurable.

Ground-truth boxes are clipped with the same `min_visible_frac` the training
tiler used, so a target counts in a tile only if enough of it is actually there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Run as a script from anywhere, so the repo root must be importable before the
# src.* imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sar.data.sar_eval import is_vis_sequence, parse_wisard_label  # noqa: E402
from src.sar.data.tiling import tile_dataset  # noqa: E402

SRC = Path("sar-drone-data/raw/WiSARD/extracted_vis")
DST = Path("sar-drone-data/processed/sar_wisard_tiled")
ROOT = Path(__file__).resolve().parents[1]

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
PERSON_CLASS = 0


def frame_index(stem: str) -> int:
    """Trailing frame number of a WiSARD frame stem."""
    _, _, frame = stem.rpartition("_")
    if not frame.isdigit():
        raise ValueError(f"cannot read a frame index from {stem!r}")
    return int(frame)


def keep_frame(stem: str, stride: int) -> bool:
    """True if this frame survives temporal subsampling."""
    if stride <= 1:
        return True
    return frame_index(stem) % stride == 0


def yolo_line(cls_id: int, box: tuple[int, int, int, int], tile_w: int, tile_h: int) -> str:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2 / tile_w
    cy = (y0 + y1) / 2 / tile_h
    w = (x1 - x0) / tile_w
    h = (y1 - y0) / tile_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="keep every Nth frame (1 = all)")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visible-frac", type=float, default=0.3)
    ap.add_argument("--min-box-px", type=float, default=4.0,
                    help="drop GT boxes whose native side is below this; WiSARD "
                         "carries boxes down to 0.5 px which no detector can hit")
    ap.add_argument("--limit-sequences", type=int, default=0, help="debug: cap sequences")
    args = ap.parse_args()

    img_dir = DST / "images" / "test"
    lbl_dir = DST / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    seq_stats: dict[str, dict] = {}
    n_frames = n_tiles = n_pos_tiles = n_boxes = n_dropped_small = 0

    seq_dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    if args.limit_sequences:
        seq_dirs = seq_dirs[: args.limit_sequences]

    for d in seq_dirs:
        if not is_vis_sequence(d.name):
            print(f"[skip] {d.name}: not a VIS sequence", flush=True)
            continue

        frames = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
        s_frames = s_tiles = s_pos = s_boxes = 0

        for f in frames:
            if not keep_frame(f.stem, args.stride):
                continue
            W, H = Image.open(f).size

            lbl_path = f.with_suffix(".txt")
            boxes: list[tuple[int, int, int, int, int]] = []
            if lbl_path.exists():
                # clip: WiSARD carries 829 boxes overrunning the frame edge.
                for b in parse_wisard_label(lbl_path, on_out_of_range="clip"):
                    bw, bh = b.w * W, b.h * H
                    if (bw * bh) ** 0.5 < args.min_box_px:
                        n_dropped_small += 1
                        continue
                    x0 = int(round((b.cx - b.w / 2) * W))
                    y0 = int(round((b.cy - b.h / 2) * H))
                    x1 = int(round((b.cx + b.w / 2) * W))
                    y1 = int(round((b.cy + b.h / 2) * H))
                    if x1 > x0 and y1 > y0:
                        boxes.append((PERSON_CLASS, x0, y0, x1, y1))
            # A missing label file means zero humans, not unannotated: the six
            # human-free sequences ship count.txt "number of humans: 0" and no
            # per-frame files. Those frames stay in, as true negatives.

            tiled = tile_dataset(
                W, H, boxes,
                tile_size=args.tile_size,
                overlap=args.overlap,
                min_visible_frac=args.min_visible_frac,
            )

            img = Image.open(f).convert("RGB")
            for (tx0, ty0, tx1, ty1), labels in tiled:
                name = f"{f.stem}__t{tx0}_{ty0}"
                img.crop((tx0, ty0, tx1, ty1)).save(img_dir / f"{name}.jpg", quality=92)
                tw, th = tx1 - tx0, ty1 - ty0
                (lbl_dir / f"{name}.txt").write_text(
                    "\n".join(yolo_line(c, (a, b_, c2, d2), tw, th) for c, a, b_, c2, d2 in labels)
                    + ("\n" if labels else "")
                )
                s_tiles += 1
                if labels:
                    s_pos += 1
                    s_boxes += len(labels)
            s_frames += 1

        seq_stats[d.name] = {
            "frames_kept": s_frames,
            "frames_total": len(frames),
            "tiles": s_tiles,
            "positive_tiles": s_pos,
            "boxes": s_boxes,
        }
        n_frames += s_frames
        n_tiles += s_tiles
        n_pos_tiles += s_pos
        n_boxes += s_boxes
        print(f"[ok] {d.name}: {s_frames}/{len(frames)} frames -> {s_tiles} tiles "
              f"({s_pos} positive, {s_boxes} boxes)", flush=True)

    cfg = ROOT / "configs" / "sar_wisard_tiled.yaml"
    cfg.write_text(
        "# Tiled WiSARD cross-domain EVAL set. Never trained on.\n"
        "# train/val point at the same test dir only because ultralytics requires\n"
        "# the keys; this set has no training split by construction.\n"
        f"path: {DST}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
        "  2: two_wheeler\n"
    )

    manifest = {
        "source": str(SRC),
        "dest": str(DST),
        "frame_stride": args.stride,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_visible_frac": args.min_visible_frac,
        "min_box_px": args.min_box_px,
        "sequences": len(seq_stats),
        "frames_kept": n_frames,
        "tiles": n_tiles,
        "positive_tiles": n_pos_tiles,
        "negative_tiles": n_tiles - n_pos_tiles,
        "boxes": n_boxes,
        "gt_boxes_dropped_below_min_px": n_dropped_small,
        "per_sequence": seq_stats,
        "caveats": [
            "Frames are consecutive video and strongly correlated within a sequence; "
            "bootstrap over sequences, not over tiles or boxes.",
            "Every tile is retained including negatives, so the per-frame false-positive "
            "rate needed for the R8 SAR operating point is measurable.",
            "Tiles from one source frame overlap by the configured fraction, so a person "
            "near a tile seam can appear in two tiles. Per-tile metrics therefore differ "
            "from full-frame metrics; full-frame numbers require merge_tile_detections.",
        ],
    }
    out = ROOT / "results" / "wisard_tile_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[write] {cfg}")
    print(f"[write] {out}")
    print(f"sequences {len(seq_stats)} | frames {n_frames} | tiles {n_tiles} "
          f"({n_pos_tiles} positive) | boxes {n_boxes} | tiny GT dropped {n_dropped_small}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
