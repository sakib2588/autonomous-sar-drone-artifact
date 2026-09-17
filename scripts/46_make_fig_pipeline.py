#!/usr/bin/env python3
"""Figure 1: the measurement pipeline, drawn at full text width.

The previous version was six grey boxes carrying prose, which made it a
paragraph with a border rather than a diagram. This one draws the three grids
the study actually ran, one chip per run, so their shape is visible instead of
quoted.

Three things this layout is careful about, because each of them misled in an
earlier draft.

Provenance. Every count is read from the artefact that produced it. The seed
count on the training panel comes from the training checkpoints, not from the
fine-tuning budget checkpoints, which carry their own seeds and would overstate
replication. Only y11n_tiled640 and y11n_full640 ran at three seeds. The other
three configurations ran at one, and the grid shows that gap rather than
averaging it away.

Attribution. The operating threshold 0.1515 belongs to the 320 px encounter
study alone, so it is not printed on the recall-at-budget box, whose threshold
is set per configuration.

Dependency. The on-device branch is fed from the trained checkpoint down the
right margin, not from the accuracy metrics, which it does not consume.

Every count is read from the committed artefacts:
  results/dataset_build_report.json    tile counts per split
  results/dataset_class_counts.csv     classes per split
  results/scale_sweep_wisard.csv       resolution sweep, sequences and tiles
  results/export_manifest_tflite.json  the exported TFLite variants
  checkpoints/                         training, epoch-sweep and budget runs

Run with the repository virtualenv, which is where matplotlib lives:
  .venv/bin/python scripts/46_make_fig_pipeline.py
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline_style as ps

OUT = ROOT / "paper/figures/fig_pipeline.pdf"

WIDTH_IN, HEIGHT_IN = 10.4, 3.10
# Both axes run 0 to 100 over a canvas that is not square, so one x unit is
# several y units on the page. A chip meant to read as a square needs CH set to
# CW times that ratio, otherwise it prints as a wide bar.
ASPECT = WIDTH_IN / HEIGHT_IN

# Chip size is derived from the panel rather than fixed, so shortening the
# canvas cannot push the tallest grid over its own subtitle and footnote.
PANEL_H = 30.0
PANEL_FREE_TOP_PAD, PANEL_FREE_BOT_PAD = 10.8, 6.2
PANEL_FREE_H = PANEL_H - PANEL_FREE_TOP_PAD - PANEL_FREE_BOT_PAD
MAX_ROWS = 3
GX, GY = 0.7, 0.8
CH = (PANEL_FREE_H - (MAX_ROWS - 1) * GY) / MAX_ROWS
CW = CH / ASPECT

TRAIN, ZERO_SHOT, FINE_TUNE = "#4477AA", "#0077BB", "#228833"
ABSENT = "#E4E4E4"
MEAS_EC, MEAS_FC = "#0077BB", "#E1F0F7"
PI_EC, PI_FC = ps.EDGE["c"], ps.FILL["c"]

# Column order for the training grid, with the short labels printed beneath it.
TRAIN_ORDER = [("y11n_tiled640", "tiled 640"),
               ("y11n_full640", "untiled"),
               ("y8n_tiled640", "v8n"),
               ("y11n_tiled416", "416 px"),
               ("y11n_tiledallneg640", "all-neg")]


def counts():
    build = json.load(open(ROOT / "results/dataset_build_report.json"))["tiled"]
    cls = {r["split"]: r for r in csv.DictReader(
        open(ROOT / "results/dataset_class_counts.csv"))}
    sweep = list(csv.DictReader(open(ROOT / "results/scale_sweep_wisard.csv")))
    tfl = json.load(open(ROOT / "results/export_manifest_tflite.json"))["artifacts"]

    known = dict(TRAIN_ORDER)
    train = collections.defaultdict(set)
    budgets, ft_seeds = set(), set()
    epoch_ckpts = 0
    for p in (ROOT / "checkpoints").iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n.startswith("_es_"):
            epoch_ckpts += 1
            continue
        m = re.match(r"y11n_ftbudget_n(\d+)_s(\d+)$", n)
        if m:
            budgets.add(int(m.group(1)))
            ft_seeds.add(int(m.group(2)))
            continue
        m = re.match(r"(.+)_s(\d+)$", n)
        if m and m.group(1) in known:
            train[m.group(1)].add(int(m.group(2)))

    res = sorted(int(r["imgsz"]) for r in sweep)
    return {
        "frames": build["train"]["images"] + build["val"]["images"] + build["test"]["images"],
        "classes": [c for c in cls["train"] if c not in ("split", "total")],
        "tiles": (build["train"]["tiles_written"], build["val"]["tiles_written"],
                  build["test"]["tiles_written"]),
        "wis_seq": int(sweep[0]["sequences"]),
        "wis_tiles": int(sweep[0]["tiles"]),
        "wis_boxes": int(sweep[0]["gt_boxes"]),
        "res": res,
        "train": train,
        "train_seeds": sorted({s for v in train.values() for s in v}),
        "train_runs": sum(len(v) for v in train.values()),
        "epoch_ckpts": epoch_ckpts,
        "budgets": sorted(budgets),
        "ft_seeds": sorted(ft_seeds),
        "external": (ROOT / "checkpoints/ext_yolov8s_visdrone").is_dir(),
        "tflite": sorted({a["artifact_id"].replace("tflite_", "") for a in tfl}),
    }


def chip_block(x, w, y, h, cols, rows):
    """Left edge and top row for a chip grid centred in a panel's free area."""
    left = x + (w - (cols * (CW + GX) - GX)) / 2
    free_top = y + h - PANEL_FREE_TOP_PAD
    free_bot = y + PANEL_FREE_BOT_PAD
    centre = (free_top + free_bot) / 2
    return left, centre - CH / 2 + (rows - 1) * (CH + GY) / 2


def panel(ax, x, w, y, h, title, subtitle, cols, rows, colours, footnote):
    ps.box(ax, x, y, w, h, "", None, "#FCFCFC", "#666666")
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#111111")
    ax.text(x + w / 2, y + h - 6.9, subtitle, ha="center", va="center",
            fontsize=6.9, color="#444444")
    left, top = chip_block(x, w, y, h, cols, rows)
    ps.chips(ax, left, top, colours, cols=cols, cw=CW, ch=CH, gx=GX, gy=GY)
    ax.text(x + w / 2, y + 2.8, footnote, ha="center", va="center",
            fontsize=6.9, color="#444444")


def main():
    c = counts()
    fig, ax = ps.canvas(width=WIDTH_IN, height=HEIGHT_IN)
    grey = "#777777"

    # ------------------------------------------------------------- band 1
    # Both corpora, and what tiling turns the source one into.
    ys, hs = 84, 12
    ps.box(ax, 4, ys, 20, hs, "VisDrone2019-DET",
           f"{c['frames']:,} frames, {len(c['classes'])} classes")
    # matplotlib is not rendering through LaTeX, so the percent sign is written
    # plainly here. Escaping it would print the backslash.
    ps.box(ax, 28, ys, 20, hs, "Tiling", "640 px, 20% overlap")
    ps.box(ax, 52, ys, 20, hs, "VisDrone tiles",
           " / ".join(f"{t:,}" for t in c["tiles"]))
    ps.box(ax, 76, ys, 20, hs, "WiSARD",
           f"{c['wis_seq']} seq, {c['wis_tiles']:,} tiles")
    for x in (24, 48):
        ps.arrow(ax, x, ys + hs / 2, x + 4, ys + hs / 2)
    ps.stage(ax, 5.4, ys + hs - 0.8, "1")

    ax.plot([62, 62], [ys, 80.5], color=grey, lw=1.0, zorder=1)
    ax.plot([86, 86], [ys, 80.5], color=grey, lw=1.0, zorder=1)
    ax.plot([18, 86], [80.5, 80.5], color=grey, lw=1.0, zorder=1)

    # ------------------------------------------------------------- band 2
    # The three grids that actually ran. The training grid is drawn with its
    # gaps, since only two of the five configurations carry three seeds.
    ym, hm = 48, PANEL_H
    grid = [TRAIN if seed in c["train"][key] else ABSENT
            for seed in c["train_seeds"] for key, _ in TRAIN_ORDER]
    panel(ax, 4, 28, ym, hm, "Training",
          ", ".join(lab for _, lab in TRAIN_ORDER),
          len(TRAIN_ORDER), len(c["train_seeds"]), grid,
          f"{c['train_runs']} runs, 120 ep SGD, +{c['epoch_ckpts']} checkpoints")
    panel(ax, 36, 28, ym, hm, "Zero-shot sweep",
          f"{len(c['res'])} resolutions, {c['wis_boxes']:,} boxes",
          len(c["res"]), 1, [ZERO_SHOT] * len(c["res"]),
          f"{min(c['res'])} to {max(c['res'])} px"
          + (", plus external baseline" if c["external"] else ""))
    panel(ax, 68, 28, ym, hm, "Fine-tune budget",
          f"{len(c['budgets'])} budgets x {len(c['ft_seeds'])} seeds, "
          f"{len(c['budgets']) * len(c['ft_seeds'])} runs",
          len(c["budgets"]), len(c["ft_seeds"]),
          [FINE_TUNE] * (len(c["budgets"]) * len(c["ft_seeds"])),
          f"N = {min(c['budgets'])} to {max(c['budgets'])} sequences")
    ps.arrow(ax, 18, 80.5, 18, ym + hm)
    ps.arrow(ax, 50, 80.5, 50, ym + hm)
    ps.arrow(ax, 82, 80.5, 82, ym + hm)
    ps.stage(ax, 5.4, ym + hm - 0.8, "2")
    ps.stage(ax, 37.4, ym + hm - 0.8, "3")

    # ------------------------------------------------------------- band 3
    # Every branch is scored the same way, so they join one bus.
    yd, hd = 26, 16
    for x, w, t, s in ((4, 21, "AP@0.5", "per class, IoU 0.5"),
                       (27, 21, "Recall at 0.1 FP/tile", "threshold per configuration"),
                       (52, 21, "ROpti", "(TP - FP) / (TP + FN)"),
                       (75, 21, "Encounter aggregation", "K = 1 to 20 frames")):
        ps.box(ax, x, yd, w, hd, t, s, MEAS_FC, MEAS_EC, tfs=8.0, sfs=6.9)
    bus = yd + hd + 4.0
    for x in (18, 50, 82):
        ax.plot([x, x], [ym, bus], color=grey, lw=1.0, zorder=1)
    ax.plot([14.5, 85.5], [bus, bus], color=grey, lw=1.0, zorder=1)
    for x in (14.5, 37.5, 62.5, 85.5):
        ps.arrow(ax, x, bus, x, yd + hd)
    ps.stage(ax, 5.4, yd + hd - 0.8, "4")

    # ------------------------------------------------------------- band 4
    # The airframe branch consumes the trained checkpoint, not the metrics
    # above it, so it is fed down the right margin and reads right to left.
    yr, hr = 3, 17
    mid = yr + hr / 2
    ps.box(ax, 66, yr, 30, hr, "TFLite export", "FP32, hybrid, full integer",
           PI_FC, PI_EC, tfs=8.0, sfs=6.9)
    ps.box(ax, 35, yr, 28, hr, "Raspberry Pi 4", "4 threads, no accelerator",
           PI_FC, PI_EC, tfs=8.0, sfs=6.9)
    ps.box(ax, 4, yr, 28, hr, "Latency and parity", "measured on device",
           PI_FC, PI_EC, tfs=8.0, sfs=6.9)
    ax.plot([96, 98], [ys + hs / 2, ys + hs / 2], color=grey, lw=1.0, zorder=1)
    ax.plot([98, 98], [ys + hs / 2, mid], color=grey, lw=1.0, zorder=1)
    ps.arrow(ax, 98, mid, 96, mid)
    ps.arrow(ax, 66, mid, 63, mid)
    ps.arrow(ax, 35, mid, 32, mid)
    ps.stage(ax, 5.4, yr + hr - 0.8, "5")

    fig.savefig(OUT)
    print("wrote", OUT)
    per_cfg = ", ".join(f"{lab} x{len(c['train'][k])}" for k, lab in TRAIN_ORDER)
    print(f"training {c['train_runs']} runs ({per_cfg}), "
          f"{c['epoch_ckpts']} epoch checkpoints, "
          f"{len(c['res'])} resolutions {min(c['res'])}-{max(c['res'])}, "
          f"{len(c['budgets']) * len(c['ft_seeds'])} fine-tune runs, "
          f"external baseline={c['external']}, tiles {c['tiles']}")


if __name__ == "__main__":
    main()
