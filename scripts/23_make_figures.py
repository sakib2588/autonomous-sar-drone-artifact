#!/usr/bin/env python3
"""Publication figures. Currently Figure 1: the domain shift, shown rather than asserted.

Figure 1 puts one VisDrone training tile beside one WiSARD evaluation tile. Both are 640 px cut at
native resolution, so they are directly comparable, and the bottom row crops an identical 128 px
window around one person from each so the scale difference is visible rather than tabulated.

The contrast this figure has to carry, measured over full populations in the same tile space:
    VisDrone train  n=218,769  mean 20.88 px  74.8% within 8-32 px  median aspect h/w 2.00
    WiSARD          n=  8,345  mean 67.30 px  79.5% at or above 32 px  median aspect h/w 1.46
That is a 3.2x shift toward LARGER targets, plus oblique-to-nadir viewpoint -- the opposite of the
direction the small-object literature would predict, which is why the paper shows it directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WISARD = Path("sar-drone-data/processed/sar_wisard_tiled")
VISDRONE = ROOT / "data/processed/sar_rgb_tiled"
TILE = 640
PERSON = 0

# Representative tiles, chosen for median-typical person size in each corpus.
VD_TILE = "0000214_02750_d_0000255_t005"   # 11 persons, median side 20.8 px
WS_TILE = "200426_SkookumCreek_Mavic_Mini_VIS_0007_00001340__t1024_440"  # nadir, road


def load(img_dir: Path, lbl_dir: Path, stem: str):
    im = Image.open(img_dir / f"{stem}.jpg").convert("RGB")
    boxes = []
    for line in (lbl_dir / f"{stem}.txt").read_text().splitlines():
        t = line.split()
        if len(t) == 5 and int(t[0]) == PERSON:
            cx, cy, bw, bh = (float(v) for v in t[1:])
            boxes.append(((cx - bw / 2) * TILE, (cy - bh / 2) * TILE, bw * TILE, bh * TILE))
    return im, boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=128, help="side of the identical-scale inset")
    ap.add_argument("--out", default=str(ROOT / "paper/figures/fig1_domain_shift.pdf"))
    cli = ap.parse_args()

    vd_im, vd_b = load(VISDRONE / "images/train", VISDRONE / "labels/train", VD_TILE)
    ws_im, ws_b = load(WISARD / "images/test", WISARD / "labels/test", WS_TILE)
    if not vd_b or not ws_b:
        print("[fatal] a chosen tile has no person boxes"); return 2

    # Inset centred on the person closest to that CORPUS's median size, not the largest in the
    # tile. Picking the largest would show 36 px against 56 px and understate the real contrast;
    # the population medians are 17.3 px and 56.8 px respectively.
    VD_MEDIAN, WS_MEDIAN = 17.3, 56.8
    def pick(bs, target): return min(bs, key=lambda b: abs((b[2] * b[3]) ** 0.5 - target))
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.95), dpi=400)
    accent = "#F2C14E"

    for col, (im, bs, title, sub, med) in enumerate([
        (vd_im, vd_b, "VisDrone (training)", "urban, oblique", VD_MEDIAN),
        (ws_im, ws_b, "WiSARD (evaluation)", "wilderness, nadir", WS_MEDIAN),
    ]):
        ax = axes[0][col]
        ax.imshow(im)
        for (x, y, w, h) in bs:
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, lw=0.6, ec=accent))
        ax.set_title(title, fontsize=7, pad=3)
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=5.6,
                color="0.35")
        ax.set_xticks([]); ax.set_yticks([])

        bx, by, bw, bh = pick(bs, med)
        ccx, ccy = bx + bw / 2, by + bh / 2
        half = cli.crop / 2
        x0 = min(max(ccx - half, 0), TILE - cli.crop)
        y0 = min(max(ccy - half, 0), TILE - cli.crop)
        ax.add_patch(mpatches.Rectangle((x0, y0), cli.crop, cli.crop, fill=False, lw=0.7,
                                        ec="white", ls=(0, (2, 1.5))))

        axc = axes[1][col]
        axc.imshow(im.crop((int(x0), int(y0), int(x0 + cli.crop), int(y0 + cli.crop))))
        axc.add_patch(mpatches.Rectangle((bx - x0, by - y0), bw, bh, fill=False, lw=0.8, ec=accent))
        axc.set_xticks([]); axc.set_yticks([])
        axc.text(0.5, -0.06, f"{(bw*bh)**0.5:.0f} px person", transform=axc.transAxes,
                 ha="center", va="top", fontsize=5.6, color="0.35")

    axes[0][0].set_ylabel("640 px tile", fontsize=6)
    axes[1][0].set_ylabel(f"{cli.crop} px inset", fontsize=6)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.06, wspace=0.04, hspace=0.16)
    out = Path(cli.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"[write] {out}")
    v, w = pick(vd_b, VD_MEDIAN), pick(ws_b, WS_MEDIAN)
    print(f"  VisDrone tile {len(vd_b)} persons, inset shows {(v[2]*v[3])**0.5:.1f} px (corpus median {VD_MEDIAN})")
    print(f"  WiSARD   tile {len(ws_b)} persons, inset shows {(w[2]*w[3])**0.5:.1f} px (corpus median {WS_MEDIAN})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
