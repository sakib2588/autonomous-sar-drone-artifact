"""Explainer diagrams: a project-history timeline and a method flowchart.

Renders vector PDFs into notes/figures/ for embedding in
notes/PROJECT_EXPLAINER.md. Documentation figures only -- the manuscript
reads none of them.

    .venv-tflite/bin/python scripts/44_make_explainer_diagrams.py

The timeline auto-splits at phase boundaries so no panel exceeds one A4 text
block, and balances the split rather than filling greedily: a panel taller
than the page gets scaled down until the body text is unreadable, and a
greedy fill leaves a stub final panel.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

INK = "#1F1D1A"
MUTED = "#54504A"
SPINE = "#B4AEA1"
EDGE = "#8C877D"

FIG_W = 7.30
MAX_H = 10.10  # A4 text height at 2.1 cm margins, less a caption

PHASES = {
    "setup":    dict(band="#F1EFE9", accent="#847E70", label="#6A6558"),
    "build":    dict(band="#E5F0E9", accent="#2E7D4F", label="#2E7D4F"),
    "measure":  dict(band="#E3EDF7", accent="#2A5F8F", label="#2A5F8F"),
    "surprise": dict(band="#FBF1DD", accent="#C08018", label="#9C6A12"),
    "free":     dict(band="#ECE7F6", accent="#5E35B1", label="#5E35B1"),
    "cost":     dict(band="#E0F2F0", accent="#00796B", label="#00796B"),
    "hardware": dict(band="#FCE8EC", accent="#B3325A", label="#B3325A"),
    "check":    dict(band="#FAE3E3", accent="#A33232", label="#A33232"),
}

STEPS = [
    ("setup", "THE STARTING POINT",
     "Fix the hardware a rescue team can afford",
     "Elashaal et al. (2024) pair a Raspberry Pi 4 with a Pixhawk and single-class YOLOv4-tiny."
     " Cited for the hardware class, never as an accuracy baseline."),
    ("setup", None,
     "Find the gap nobody measures",
     "The only aerial corpora large enough to train a person detector are urban traffic surveillance;"
     " the missions that matter are flown over wilderness. Published accuracy is almost always"
     " in-domain, so the cost of that mismatch is unquantified."),

    ("build", "BUILD THE PIPELINE",
     "Build both corpora the same way",
     "640 px tiles at native resolution, 20% overlap, 30% area rule. Splits at sequence granularity,"
     " never frame level; MD5 over 8,629 images confirms zero cross-split duplicates."),
    ("build", None,
     "Train the detectors, and a control",
     "YOLO11n and YOLOv8n, 120 epochs, SGD under a cosine schedule, seed 42 plus replicates at 123"
     " and 456, on one 8 GB RTX 3060 Ti. An untiled resize variant is trained identically."),

    ("measure", "MEASURE THE GAP",
     "Fly the urban-trained model over wilderness, on recorded imagery",
     "Zero-shot on 37,058 WiSARD tiles from 38 flight sequences. Person AP@0.5 falls 0.3997 to"
     " 0.2567. Intervals resample SEQUENCES, not tiles: consecutive frames are near-duplicates, so a"
     " tile bootstrap would return a narrow interval that is wrong."),
    ("measure", None,
     "The control turns out to be the finding",
     "The resize-only control collapses to 0.0449, so tiling is worth 5.7x -- more than every other"
     " effect combined. YOLO11n, YOLOv8n and the 416 px variant all overlap, so the architecture"
     " benchmark everyone runs does not resolve here. That null is reported."),

    ("surprise", "THE SURPRISE",
     "The shift runs opposite to the entire small-object literature",
     "In identical tile space, wilderness persons average 67.30 px against VisDrone's 20.88 px,"
     " aspect ratio moving 2.00 to 1.46. Targets are three times LARGER and near-nadir, not smaller."),
    ("surprise", None,
     "Halving the inference input helps, for the wrong reason",
     "Inference resolution peaks at 320 px, worth +0.068 [0.019, 0.117] paired. Decomposed: true"
     " positives move 5,081 to 5,202 while false positives fall 204,087 to 109,493 -- 46% false-alarm"
     " suppression, not better detection. A consensus control run to explain it failed, and is"
     " reported as a negative result."),

    ("free", "WHAT IS FREE TO FIX",
     "Two training-time factors the in-domain view hides",
     "Cross-domain accuracy is not monotone: it peaks at epochs 70 and 60, then decays 18.4% and"
     " 14.6% by epoch 120 while in-domain validation rises. Raising the negative-tile fraction 1.33%"
     " to 9.55% cuts false positives at matched recall by a third, and only late -- one mechanism"
     " links both."),
    ("free", None,
     "The default confidence threshold was never chosen",
     "Swept over all 37,058 tiles: 0.25 recovers 30.1% of annotated people against 38.7% at 0.10. The"
     " inherited library default discards 28% of the recall the same model already supplies, for 0.18"
     " more false positives per tile and no latency cost."),

    ("cost", "WHAT COSTS MONEY",
     "Bound what target data buys, and when to stop paying for it",
     "Fine-tuning is worth 6.1x on held-out flights and saturates near sixteen sequences: seed spread"
     " collapses from 0.347 at N=4 to 0.035 at N=16. Repeated coverage lifts encounter-level"
     " detection from 0.609 to 0.923."),

    ("hardware", "PUT IT ON THE DEVICE",
     "Measure the Pi 4, then ask whether the trade was needed",
     "TFLite at four threads: FP32 495 ms, hybrid 761 ms, full integer 410 ms for 1.21x, costing"
     " 12.80 points. On the live payload that trade is unnecessary: capture, tiling and encoding set"
     " the pace, so both precisions sustain 0.91 to 1.20 detections per second."),

    ("check", "CHECK OUR OWN WORK",
     "Trace every number back to the file that produced it",
     "The audit caught two wrong Methods numbers inside 48 hours and one overstated headline: the"
     " threshold claim rested on 80 balanced tiles and measured the deployed artifact, not the"
     " reported model. Re-run over all 37,058 tiles the gain is +28%, not +45%."),

    ("setup", "WRITE IT UP",
     "Six pages, and say plainly what is not in them",
     "ICCIT 2026: 6 pages, 21 references, an interval on every headline number, 157 repo tests green."
     " Deliberately absent: no independent aerial-built baseline, and nothing flight-validated -- the"
     " payload is built, deployed, and has never left the ground."),
]

SPINE_X = 0.42
CARD_L, CARD_R = 0.92, 7.14
PAD_X = 0.23
WRAP = 92

H_TOP = 0.105
H_PHASE = 0.225
H_TITLE = 0.225
H_TGAP = 0.050
H_BODY = 0.138
H_BOT = 0.140
BAND_PAD = 0.105
BAND_GAP = 0.145
HEADER = 0.88


def build_bands():
    bands, band = [], []
    for phase, label, title, body in STEPS:
        lines = textwrap.wrap(body, WRAP)
        h = H_TOP + H_TITLE + H_TGAP + len(lines) * H_BODY + H_BOT
        if label:
            h += H_PHASE
            if band:
                bands.append(band)
                band = []
        band.append(dict(phase=phase, label=label, title=title, lines=lines, h=h))
    bands.append(band)
    return bands


def split_bands(bands):
    """Fewest panels under MAX_H, then the most balanced split at that count."""
    hs = [sum(s["h"] for s in b) + 2 * BAND_PAD + BAND_GAP for b in bands]
    n = len(bands)

    def pack(k):
        best = [None]

        def rec(i, parts, cur):
            if len(parts) + (1 if cur else 0) > k:
                return
            if i == n:
                if cur and len(parts) + 1 == k:
                    cand = parts + [cur]
                    tallest = max(HEADER + sum(hs[j] for j in p) for p in cand)
                    if tallest <= MAX_H and (best[0] is None or tallest < best[0][0]):
                        best[0] = (tallest, cand)
                return
            rec(i + 1, parts, cur + [i])
            if cur:
                rec(i + 1, parts + [cur], [i])

        rec(0, [], [])
        return best[0]

    for k in range(1, n + 1):
        got = pack(k)
        if got:
            return [[bands[j] for j in p] for p in got[1]]
    return [bands]


def draw_timeline():
    panels = split_bands(build_bands())
    first = 1
    for pi, panel in enumerate(panels, start=1):
        total = HEADER + sum(
            sum(s["h"] for s in b) + 2 * BAND_PAD + BAND_GAP for b in panel)
        fig = plt.figure(figsize=(FIG_W, total))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, total)
        ax.axis("off")

        def Y(c):
            return total - c

        suffix = "" if len(panels) == 1 else f"   ({pi} of {len(panels)})"
        ax.text(0.06, Y(0.34), "How the project got here" + suffix,
                fontsize=18, fontweight="bold", family=SANS, color=INK, va="baseline")
        ax.text(0.06, Y(0.57),
                "Urban-to-wilderness domain gap in aerial person detection"
                " -- one straight line through the whole study",
                fontsize=9.4, family=SANS, color=MUTED, style="italic", va="baseline")

        cursor = HEADER
        spine_top = cursor + 0.08
        n = first

        for band in panel:
            bh = sum(s["h"] for s in band) + 2 * BAND_PAD
            c = PHASES[band[0]["phase"]]
            ax.add_patch(FancyBboxPatch(
                (CARD_L, Y(cursor + bh)), CARD_R - CARD_L, bh,
                boxstyle="round,pad=0,rounding_size=0.085",
                facecolor=c["band"], edgecolor="none", zorder=1))

            y = cursor + BAND_PAD
            for step in band:
                ty = y + H_TOP
                if step["label"]:
                    ax.text(CARD_L + PAD_X, Y(ty + 0.140), step["label"],
                            fontsize=7.2, family=MONO, fontweight="bold",
                            color=c["label"], va="baseline", zorder=3)
                    ty += H_PHASE

                cy = Y(ty + 0.098)
                ax.plot([SPINE_X + 0.125, CARD_L - 0.04], [cy, cy],
                        color=SPINE, lw=1.4, solid_capstyle="round", zorder=2)
                ax.add_patch(Circle((SPINE_X, cy), 0.125, facecolor="white",
                                    edgecolor=c["accent"], lw=1.6, zorder=4))
                ax.text(SPINE_X, cy, str(n), fontsize=7.9, family=SANS,
                        fontweight="bold", color=c["accent"],
                        ha="center", va="center_baseline", zorder=5)
                n += 1

                ax.text(CARD_L + PAD_X, Y(ty + 0.158), step["title"],
                        fontsize=10.9, family=SANS, fontweight="bold",
                        color=INK, va="baseline", zorder=3)
                ty += H_TITLE + H_TGAP

                for line in step["lines"]:
                    ty += H_BODY
                    ax.text(CARD_L + PAD_X, Y(ty - 0.034), line,
                            fontsize=8.4, family=SANS, color=MUTED,
                            va="baseline", zorder=3)
                y += step["h"]
            cursor += bh + BAND_GAP

        ax.plot([SPINE_X, SPINE_X], [Y(spine_top), Y(cursor - BAND_GAP + 0.02)],
                color=SPINE, lw=4.4, solid_capstyle="round", zorder=0)

        name = ("fig_explainer_timeline.pdf" if len(panels) == 1
                else f"fig_explainer_timeline_{pi}.pdf")
        fig.savefig(OUT / name, format="pdf")
        plt.close(fig)
        print(f"  {name}: {FIG_W:.2f} x {total:.2f} in, steps {first}-{n - 1}")
        first = n


# --------------------------------------------------------------------------
# Flowchart
# --------------------------------------------------------------------------

STY = {
    "data":  dict(face="#F1EFE9", edge="#9A9285", title="#3A362F", body="#5A5549"),
    "gate":  dict(face="#FBE4E4", edge="#B33A3A", title="#8E2626", body="#7C3131"),
    "core":  dict(face="#E3EDF7", edge="#2A5F8F", title="#1E4A73", body="#3A5A78"),
    "find":  dict(face="#E5F0E9", edge="#2E7D4F", title="#1F6640", body="#356B4C"),
    "limit": dict(face="#FBF1DD", edge="#C08018", title="#8E6210", body="#7A5A1E"),
}

LEGEND = [("data", "Data and training"), ("gate", "Verification gate"),
          ("core", "Core measurement"), ("find", "Headline finding"),
          ("limit", "Stated limitation")]

F_L, F_R = 0.30, 7.00
GAP_X = 0.20
ARROW = 0.20
B_TOP, B_TGAP, B_BOT = 0.120, 0.045, 0.105
B_TITLE = 0.180
B_LINE = 0.133
TITLE_H = 0.52
LEG_H = 0.50

ROWS = [
    dict(grid=2, cells=[
        dict(slot=0, sty="data", title="VisDrone2019-DET   (training)",
             body="Urban aerial traffic surveillance. 8,629 annotated frames across "
                  "208 / 76 / 61 sequences; ten categories collapsed to three."),
        dict(slot=1, sty="data", title="WiSARD   (evaluation)",
             body="Wilderness search and rescue at 20-120 m. 2,586 visible frames "
                  "across 38 flight sequences, annotated for person only."),
    ]),
    dict(grid=2, cells=[
        dict(slot=0, sty="data", title="Tile, then subsample negatives",
             body="640 px at native resolution, 20% overlap, 30% area rule. "
                  "40,523 training tiles, of which 98.7% contain an object."),
        dict(slot=1, sty="data", title="Tile with identical geometry",
             body="Same cutter, same parameters. 37,058 evaluation tiles, of "
                  "which only 14.2% contain a person."),
    ]),
    dict(grid=2, cells=[
        dict(slot=0, sty="data", title="Train the detectors, and an untiled control",
             body="120 epochs, SGD under a cosine schedule, patience 25, "
                  "deterministic at seed 42 with replicates at 123 and 456."),
    ]),
    dict(grid=2, cells=[
        dict(slot=0, sty="gate", title="GATE  --  splits and in-domain check",
             body="MD5 over 8,629 images: zero cross-split duplicates. In-domain "
                  "person AP@0.5 0.3997 tiled against 0.2390 untiled."),
    ]),
    dict(grid=1, cells=[
        dict(slot=0, sty="core", title="ZERO-SHOT CROSS-DOMAIN EVALUATION",
             body="Strict COCO-style IoU 0.5 over every tile, including the 85.8% holding no person. "
                  "Intervals are 10,000-resample percentile bootstraps over the 38 flight "
                  "SEQUENCES, never over tiles."),
    ]),
    dict(grid=1, cells=[
        dict(slot=0, sty="find", title="Headline:   0.2567 [.141 -- .393]   against a control at 0.0449",
             body="Tiling is worth 5.7x cross-domain, larger than every other effect combined. "
                  "Architecture does not resolve: YOLO11n, YOLOv8n and 416 px all overlap."),
    ]),
    dict(grid=3, cells=[
        dict(slot=0, sty="core", title="Characterise the shift",
             body="67.30 px against 20.88 px, aspect 2.00 to 1.46. Larger and near-nadir, "
                  "not smaller."),
        dict(slot=1, sty="core", title="Knobs that cost nothing",
             body="320 px input: +0.068 via 46% fewer false alarms. Threshold 0.25 to 0.10: "
                  "+28% recall. Stop at epoch 70. Negatives 1.33% to 9.55%."),
        dict(slot=2, sty="find", title="What only target data buys",
             body="Fine-tuning 6.1x, saturating at sixteen sequences. Repeated coverage "
                  "0.609 to 0.923."),
    ]),
    dict(grid=1, cells=[
        dict(slot=0, sty="data", title="Export to TFLite and measure on the Raspberry Pi 4",
             body="Four threads, median of 160 timed frames after 60 warm-up. FP32 495 ms, hybrid "
                  "integer 761 ms, full integer 410 ms for 1.21x at a cost of 12.80 points."),
    ]),
    dict(grid=1, cells=[
        dict(slot=0, sty="limit", title="NOT FLIGHT-VALIDATED",
             body="Every accuracy figure comes from recorded imagery and every latency figure from a "
                  "Pi 4 on a desk. The payload is built, deployed and autostarting, and has never flown."),
    ]),
]

# (from_row, from_cell, to_row, to_cell); bypass carries an explicit target x
EDGES = [
    (0, 0, 1, 0), (0, 1, 1, 1),
    (1, 0, 2, 0),
    (2, 0, 3, 0),
    (3, 0, 4, 0),
    (1, 1, 4, 0, "slot1"),
    (4, 0, 5, 0),
    (5, 0, 6, 0), (5, 0, 6, 1), (5, 0, 6, 2),
    (6, 0, 7, 0), (6, 1, 7, 0), (6, 2, 7, 0),
    (7, 0, 8, 0),
]

WRAP_BY = {1: 112, 2: 58, 3: 38}
TWRAP_BY = {1: 70, 2: 36, 3: 25}   # bold, so narrower than the body
FONT_BY = {1: (10.4, 8.0), 2: (9.4, 7.4), 3: (8.4, 6.8)}


def col_geom(ncols):
    w = (F_R - F_L - GAP_X * (ncols - 1)) / ncols
    return [(F_L + i * (w + GAP_X) + w / 2, w) for i in range(ncols)]


def draw_flowchart():
    laid = []
    for row in ROWS:
        g = row["grid"]
        cols = col_geom(g)
        cells = []
        for cell in row["cells"]:
            cx, w = cols[cell["slot"]]
            lines = textwrap.wrap(cell["body"], WRAP_BY[g])
            tlines = textwrap.wrap(cell["title"], TWRAP_BY[g])
            h = (B_TOP + len(tlines) * B_TITLE + B_TGAP
                 + len(lines) * B_LINE + B_BOT)
            cells.append(dict(cell, cx=cx, w=w, h=h, lines=lines,
                              tlines=tlines, grid=g))
        laid.append(cells)

    total = (TITLE_H + sum(max(c["h"] for c in r) for r in laid)
             + ARROW * (len(laid) - 1) + LEG_H + 0.24)

    fig = plt.figure(figsize=(FIG_W, total))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, total)
    ax.axis("off")

    def Y(c):
        return total - c

    ax.text(FIG_W / 2, Y(0.29), "The measurement pipeline, end to end",
            fontsize=15.5, fontweight="bold", family=SANS, color=INK,
            ha="center", va="baseline")
    ax.text(FIG_W / 2, Y(0.50),
            "Two corpora in, one fixed evaluation protocol, and every branch it feeds",
            fontsize=9.0, family=SANS, color=MUTED, style="italic",
            ha="center", va="baseline")

    geom, cursor = {}, TITLE_H
    for ri, row in enumerate(laid):
        rh = max(c["h"] for c in row)
        for ci, c in enumerate(row):
            s = STY[c["sty"]]
            tfs, bfs = FONT_BY[c["grid"]]
            dashed = c["sty"] == "limit"
            ax.add_patch(FancyBboxPatch(
                (c["cx"] - c["w"] / 2, Y(cursor + rh)), c["w"], rh,
                boxstyle="round,pad=0,rounding_size=0.075",
                facecolor=s["face"], edgecolor=s["edge"], lw=1.25,
                linestyle=(0, (4, 2)) if dashed else "solid", zorder=2))

            ty = cursor + B_TOP
            for tline in c["tlines"]:
                ax.text(c["cx"], Y(ty + 0.142), tline, fontsize=tfs, family=SANS,
                        fontweight="bold", color=s["title"], ha="center",
                        va="baseline", zorder=3)
                ty += B_TITLE
            ty += B_TGAP
            for line in c["lines"]:
                ty += B_LINE
                ax.text(c["cx"], Y(ty - 0.032), line, fontsize=bfs, family=SANS,
                        color=s["body"], ha="center", va="baseline", zorder=3)
            geom[(ri, ci)] = (c["cx"], cursor, rh)
        cursor += rh + ARROW

    bypass_x = col_geom(2)[1][0]
    for e in EDGES:
        r0, c0, r1, c1 = e[:4]
        x0, y0, h0 = geom[(r0, c0)]
        x1, y1, _ = geom[(r1, c1)]
        if len(e) > 4 and e[4] == "slot1":
            x1 = bypass_x
        p0, p1 = (x0, Y(y0 + h0)), (x1, Y(y1))
        # "angle" elbows fail when both approach angles are vertical (parallel
        # lines never intersect), so lateral hops use a gentle arc instead.
        rad = 0.0 if abs(x0 - x1) < 1e-6 else (-0.10 if x1 > x0 else 0.10)
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=10, lw=1.05, color=EDGE,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.0, shrinkB=1.0, zorder=1))

    ly = cursor - ARROW + 0.32
    sw = 0.20
    for i, (key, label) in enumerate(LEGEND):
        col, rown = i % 3, i // 3
        lx = F_L + 0.30 + col * 2.20
        yy = Y(ly + rown * 0.26)
        s = STY[key]
        ax.add_patch(FancyBboxPatch(
            (lx, yy - 0.072), sw, 0.144,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=s["face"], edgecolor=s["edge"], lw=1.1,
            linestyle=(0, (3, 2)) if key == "limit" else "solid", zorder=2))
        ax.text(lx + sw + 0.10, yy, label, fontsize=8.2, family=SANS,
                color=MUTED, va="center_baseline", zorder=3)

    fig.savefig(OUT / "fig_explainer_pipeline.pdf", format="pdf")
    plt.close(fig)
    flag = "" if total <= MAX_H else "   <-- OVER ONE PAGE, WILL BE SCALED DOWN"
    print(f"  fig_explainer_pipeline.pdf: {FIG_W:.2f} x {total:.2f} in, "
          f"{len(laid)} rows{flag}")


if __name__ == "__main__":
    print("timeline:")
    draw_timeline()
    print("flowchart:")
    draw_flowchart()
