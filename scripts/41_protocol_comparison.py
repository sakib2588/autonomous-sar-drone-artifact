"""Same detections, two evaluation protocols: ours over every tile, theirs over positives only.

WHY THIS SCRIPT
---------------
The WiSARD visual baseline of 0.892 mAP@0.5 was measured after deleting every
tile that contains no person. We score every tile. Part of the apparent gap is
therefore protocol rather than detection, and this script measures how much by
rescoring one fixed set of detections under both rules. Nothing is re-inferred;
the detections and their greedy IoU-0.5 matches come from the cache written by
`29_dump_detections.py`, so the only thing that changes between the two numbers
is which tiles are allowed to contribute.

The recall denominator is held at the full ground-truth count in both protocols.
Dropping empty tiles removes false positives, not people, so the positives-only
score is a precision effect and must not be read as finding more targets.

USAGE
  .venv/bin/python scripts/41_protocol_comparison.py [--run RUN] [--imgsz N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
OUT = ROOT / "results" / "protocol_comparison.txt"

WISARD_BASELINE = 0.892


def ap50(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP@0.5, identical to scripts/37_compare_negfrac.py."""
    order = np.argsort(-conf)
    t = tp[order].astype(np.int64)
    ctp = np.cumsum(t)
    cfp = np.cumsum(1 - t)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.concatenate([[0.0], rec])
    p = np.concatenate([[prec[0] if len(prec) else 0.0], prec])
    return float(np.sum(np.diff(r) * p[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="y11n_tiled640_s42")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    path = CACHE / f"dets_{a.run}_sz{a.imgsz}.npz"
    if not path.is_file():
        raise SystemExit(f"missing cache {path}\nrun: scripts/29_dump_detections.py --run {a.run} --imgsz {a.imgsz}")
    z = np.load(path, allow_pickle=True)
    n_gt = int(z["n_gt"].sum())
    positive = z["n_gt"] > 0
    on_positive = positive[z["det_tile"]]

    all_tiles = ap50(z["det_conf"], z["det_tp"], n_gt)
    pos_only = ap50(z["det_conf"][on_positive], z["det_tp"][on_positive], n_gt)
    delta = pos_only - all_tiles
    gap = WISARD_BASELINE - all_tiles

    lines = [
        f"run                          {a.run} @ imgsz {a.imgsz}",
        f"tiles                        {len(positive)} total, {int(positive.sum())} with a person, "
        f"{int((~positive).sum())} empty ({(~positive).mean():.1%})",
        f"detections                   {len(z['det_conf'])} total, "
        f"{int((~on_positive).sum())} on empty tiles ({(~on_positive).mean():.1%})",
        f"ground-truth persons         {n_gt}",
        "",
        f"AP@0.5, every tile (ours)    {all_tiles:.4f}",
        f"AP@0.5, positives only       {pos_only:.4f}",
        f"protocol delta               {delta:+.4f}  ({delta / all_tiles:+.1%} relative to ours)",
        "",
        f"WiSARD visual baseline       {WISARD_BASELINE:.3f}",
        f"gap under our protocol       {gap:.4f}",
        f"gap under theirs             {WISARD_BASELINE - pos_only:.4f}",
        f"share of the gap that is protocol, not detection   {delta / gap:.1%}",
    ]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
