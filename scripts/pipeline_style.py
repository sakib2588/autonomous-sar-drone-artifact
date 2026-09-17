r"""Drawing primitives for the full-width study-pipeline figure.

The canvas is wide and short on purpose. At \textwidth it scales DOWN, which
keeps type small and sharp; a tall canvas scales up and blurs it.

Copied rather than imported across the three paper repositories. They are
independent git repositories and each must build standalone, so a shared import
would be a dependency none of them can satisfy on a fresh clone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Paul Tol bright, colourblind-safe.
EDGE = {"a": "#4477AA", "b": "#EE6677", "c": "#AA3377", "d": "#228833"}
FILL = {"a": "#E4EDF6", "b": "#FBE7E9", "c": "#F2E9F5", "d": "#E6F2E8"}
NEUTRAL_FC, NEUTRAL_EC = "#F5F5F5", "#8a8a8a"

RC = {
    "font.size": 7, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
}


def canvas(width=10.4, height=4.35):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, fc=NEUTRAL_FC, ec=NEUTRAL_EC,
        tfs=8.2, sfs=7.0, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#111111")
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center",
                fontsize=sfs, color="#444444")
    elif title:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#111111")


def arrow(ax, x1, y1, x2, y2, color="#777777", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, lw=lw, color=color,
                                 shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}"))


def chips(ax, x, y, colours, cols, cw=1.9, ch=2.6, gx=0.55, gy=0.75, edges=None):
    """One small chip per fitted run. The sweep is drawn, not described.

    `edges` optionally gives a per-chip edge colour, used to mark runs that are
    not comparable to the rest of the grid.
    """
    for i, col in enumerate(colours):
        r, c = divmod(i, cols)
        ec = "none" if edges is None else edges[i]
        ax.add_patch(FancyBboxPatch((x + c * (cw + gx), y - r * (ch + gy)), cw, ch,
                                    boxstyle="round,pad=0.08,rounding_size=0.45",
                                    fc=col, ec=ec,
                                    lw=0.0 if ec == "none" else 0.6))


def stage(ax, x, y, n):
    ax.text(x, y, n, ha="center", va="center", fontsize=7.0, color="#FFFFFF",
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.26", fc="#666666", ec="none"))


def legend(ax, entries, anchor=(0.505, 0.495), ncol=1):
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=4.6,
                      markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in entries]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=anchor, frameon=False,
              fontsize=6.4, handletextpad=0.35, labelspacing=0.3, ncol=ncol)
