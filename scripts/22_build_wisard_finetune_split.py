#!/usr/bin/env python3
"""Sequence-safe train/val/test split of the WiSARD tiles, for fine-tuning.

WHY
---
Every published SAR person-detection number this project can be compared against is
IN-DOMAIN. Sambolek et al. 2024 Table 2 is the clearest case: YOLOv8n scores mAP50 35.9
zero-shot COCO->SARD and 86.8 after fine-tuning on SARD. Our zero-shot VisDrone->WiSARD
result is the deployment reality; a fine-tuned number is the ceiling, and the gap between
them is what the domain shift actually costs.

WiSARD has never been split here. `configs/sar_wisard_tiled.yaml` deliberately points
train/val/test at the same directory and says so: it is an evaluation set with no training
split by construction. That file must not be touched -- the published cross-domain numbers
depend on it. This script writes a new, separate dataset config.

BALANCE ON BOXES, NOT TILES
---------------------------
Balancing tile counts would be a trap. Measured over the 37,058 tiles:

  * 38 sequences, but only 32 contain a single annotated person
  * 6 sequences are entirely human-free (4,616 tiles, 12.5 percent of the corpus)
  * per-sequence box counts run 0 to 1,122; per-sequence tile counts run 160 to 4,430
  * the largest sequence by tiles (200402_Karen_Inspire_VIS, 4,430) holds 527 boxes,
    while 210529_Carnation_Enterprise_VIS_0025 holds 1,122 boxes in 1,056 tiles

So tile count and person count are close to uncorrelated, and a tile-balanced split can put
most of the annotated people on one side while looking perfectly even. `sequence_split` is
therefore called with `weight_fn` counting person boxes.

NO FILE COPYING
---------------
Splits are materialised as newline-delimited path lists, not as copied or symlinked trees.
Ultralytics accepts a .txt path list for train/val/test and resolves labels by substituting
`/images/` with `/labels/` in each path, which works unchanged against the existing tile
directory. That avoids duplicating 5.5 GB and avoids the symlink dereferencing that this
project already had to work around once for `sar_rgb_full`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sar.data.sar_eval import wisard_tile_sequence_key  # noqa: E402
from src.sar.data.splits import assert_no_sequence_leak, sequence_split  # noqa: E402
from src.sar.data.taxonomy_rgb import SAR_RGB_CLASSES  # noqa: E402

TILES = Path("sar-drone-data/processed/sar_wisard_tiled")
PERSON = 0


def box_count(img_path: str) -> int:
    """Person boxes in the label file paired with this tile."""
    lbl = TILES / "labels" / "test" / (Path(img_path).stem + ".txt")
    if not lbl.exists():
        return 0
    n = 0
    for line in lbl.read_text().splitlines():
        t = line.split()
        if len(t) == 5 and int(t[0]) == PERSON:
            n += 1
    return n


def summarise(paths: list[str], counts: dict[str, int]) -> dict:
    seqs = sorted({wisard_tile_sequence_key(p) for p in paths})
    boxes = sum(counts[p] for p in paths)
    pos = sum(1 for p in paths if counts[p] > 0)
    return {
        "sequences": len(seqs),
        "sequence_ids": seqs,
        "tiles": len(paths),
        "positive_tiles": pos,
        "boxes": boxes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(TILES.parent / "sar_wisard_ft"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "sar_wisard_finetune.yaml"))
    cli = ap.parse_args()

    img_dir = TILES / "images" / "test"
    if not img_dir.is_dir():
        print(f"[fatal] tiles not found at {img_dir}", file=sys.stderr)
        return 2
    files = sorted(str(p) for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
    print(f"[load] {len(files)} tiles from {img_dir}", flush=True)

    # Cache box counts once; summarise() and weight_fn would otherwise re-read 37k files.
    counts = {f: box_count(f) for f in files}
    total_boxes = sum(counts.values())
    print(f"[load] {total_boxes} person boxes, "
          f"{sum(1 for v in counts.values() if v)} positive tiles", flush=True)

    train, val, test = sequence_split(
        files,
        ratios=tuple(cli.ratios),
        seed=cli.seed,
        key_fn=wisard_tile_sequence_key,
        weight_fn=counts.__getitem__,
    )
    assert_no_sequence_leak(train, val, test, key_fn=wisard_tile_sequence_key)
    print("[gate] no sequence leakage across the three splits", flush=True)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {"train": train, "val": val, "test": test}
    for name, paths in parts.items():
        (out_dir / f"{name}.txt").write_text("\n".join(paths) + "\n")

    manifest = {
        "source_tiles": str(TILES),
        "split_lists": str(out_dir),
        "ratios": cli.ratios,
        "seed": cli.seed,
        "balanced_on": "person box count (weight_fn), not tile count",
        "key_fn": "sar_eval.wisard_tile_sequence_key",
        "totals": {"tiles": len(files), "boxes": total_boxes},
        "splits": {name: summarise(paths, counts) for name, paths in parts.items()},
        "note": (
            "Whole sequences are allocated, never individual tiles: WiSARD tiles come from "
            "consecutive video frames sampled at stride 10, so a tile-level split would put "
            "near-duplicate frames on both sides of the boundary. Six of the 38 sequences "
            "contain no annotated person and contribute only negatives, which is why the "
            "allocator balances box count rather than tile count."
        ),
    }
    man_path = ROOT / "results" / "wisard_split_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(SAR_RGB_CLASSES))
    Path(cli.config).write_text(
        "# WiSARD fine-tuning split. Generated by scripts/22_build_wisard_finetune_split.py.\n"
        "#\n"
        "# Separate from configs/sar_wisard_tiled.yaml, which is the ZERO-SHOT evaluation set\n"
        "# and must keep pointing every key at images/test. The published cross-domain numbers\n"
        "# depend on that file; do not merge the two.\n"
        "#\n"
        "# Splits are sequence-disjoint and balanced on person-box count, not tile count.\n"
        "# Paths are .txt lists into the existing tile directory, so nothing is duplicated and\n"
        "# labels resolve by the standard /images/ -> /labels/ substitution.\n"
        "#\n"
        "# Declared with the full 3-class RGB label space because the checkpoints carry nc=3.\n"
        "# WiSARD is person-only, so classes 1 and 2 have zero ground truth and any correct\n"
        "# vehicle detection scores as a false positive. Only the PERSON row is meaningful.\n"
        f"path: {out_dir}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n" + names_block + "\n"
    )

    print(f"\n{'split':6s} {'seqs':>5s} {'tiles':>7s} {'pos':>6s} {'boxes':>6s}  {'box share':>9s}")
    for name, paths in parts.items():
        s = manifest["splits"][name]
        share = s["boxes"] / total_boxes if total_boxes else 0.0
        print(f"{name:6s} {s['sequences']:5d} {s['tiles']:7d} {s['positive_tiles']:6d} "
              f"{s['boxes']:6d}  {share:8.1%}")
    print(f"\n[write] {man_path}")
    print(f"[write] {cli.config}")
    print(f"[write] {out_dir}/{{train,val,test}}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
