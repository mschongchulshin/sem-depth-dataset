import json, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

import emstyle
from emstyle import DEPTH, OTHER, BOX, INK, GREY, PALE

ROOT = "/Volumes/One Touch/em-depth-dataset"
R = json.load(open(f"{ROOT}/cache/validation_results.json"))

emstyle.apply()

FIG_W = 7.087


def box(ax, x, y, w, h, fc, ec=None, lw=0.7, r=0.012, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec or "none", linewidth=lw, zorder=z,
                       alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color=INK, lw=0.9, style="-|>", ms=6, z=5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=z,
                                 shrinkA=2, shrinkB=2))


def heading(ax, x0, w, n, title, sub):
    ax.text(x0, 0.975, n, fontsize=8, fontweight="bold", color=DEPTH, ha="left", va="top")
    ax.text(x0 + 0.016, 0.975, textwrap.fill(title, max(14, int(w * 132))),
            fontsize=7.4, fontweight="semibold", color=INK, ha="left", va="top",
            linespacing=1.35)
    ax.text(x0, 0.862, textwrap.fill(sub, max(20, int(w * 172))), fontsize=5.9,
            color=GREY, ha="left", va="top", linespacing=1.5)


def count(ax, x, y, text):
    ax.text(x, y, text, fontsize=6.2, color=GREY, ha="center")


def main():
    c = R["corpus"]
    px = sorted(int(k) for k in c["px_nm_pct"])

    fig = plt.figure(figsize=(FIG_W, 3.28))
    ax = fig.add_axes([0.0, 0.055, 1.0, 0.945])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    X = [0.012, 0.215, 0.415, 0.640, 0.828]
    Wd = [0.160, 0.160, 0.185, 0.150, 0.166]
    BOT, H = 0.245, 0.455
    MID = BOT + H / 2
    CY = BOT - 0.055

    heading(ax, X[0], Wd[0], "1", "a public reconstruction",
            "OpenOrganelle, CC-BY-4.0, imaging and labels theirs")
    for i, dz in enumerate(np.linspace(0, 0.05, 7)):
        box(ax, X[0] + dz * 0.6, BOT + dz, 0.115, H - 0.06,
            fc=PALE if i < 6 else "#cdd4da", ec="#b6bec4", lw=0.4, z=2 + i)
    count(ax, X[0] + 0.073, CY, f"{c['source_volumes']} source volumes")

    heading(ax, X[1], Wd[1], "2", "extract a box",
            "origin and extent recorded, so the region stays recoverable")
    bx, by, bw, bh = X[1] + 0.012, BOT + 0.035, 0.098, H - 0.105
    box(ax, bx, by, bw, bh, fc="#e6ebef", ec=BOX, lw=0.9, z=4)
    d = 0.030
    for p0, p1 in (((bx, by + bh), (bx + d, by + bh + d)),
                   ((bx + bw, by + bh), (bx + bw + d, by + bh + d)),
                   ((bx + d, by + bh + d), (bx + bw + d, by + bh + d)),
                   ((bx + bw + d, by + bh + d), (bx + bw + d, by + d)),
                   ((bx + bw, by), (bx + bw + d, by + d))):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BOX, lw=0.6, zorder=4)
    count(ax, X[1] + 0.075, CY, f"{c['boxes']:,} boxes")

    heading(ax, X[2], Wd[2], "3", "cut a plane, keep what follows",
            "the instrument mills and images in turn, so what lies under a face is imaged after it")
    xs = np.linspace(X[2] + 0.008, X[2] + 0.165, 6)
    for i, x in enumerate(xs):
        first = i == 0
        box(ax, x, BOT + 0.085, 0.022, H - 0.095,
            fc="#f0d9d6" if first else "#eef1f3",
            ec=DEPTH if first else "#c6ced4", lw=0.9 if first else 0.5, z=4)
    ax.text(xs[0] + 0.011, BOT + H + 0.005, "the face", fontsize=6, color=DEPTH,
            ha="center", fontweight="semibold")
    arrow(ax, (xs[0] + 0.011, BOT + 0.050), (xs[-1] + 0.011, BOT + 0.050),
          color=DEPTH, lw=1.0, ms=7)
    ax.text((xs[0] + xs[-1]) / 2 + 0.011, BOT + 0.008, "milling direction",
            fontsize=6, color=DEPTH, ha="center", va="bottom")
    count(ax, (xs[0] + xs[-1]) / 2 + 0.011, CY, f"{c['planes']:,} cut planes")

    heading(ax, X[3], Wd[3], "4", "trace one ray per pixel",
            "how far the exposed instance continues, and what else lies under it")
    rx = X[3] + 0.004
    box(ax, rx, BOT + 0.050, 0.042, H - 0.095, fc="#f7f7f5", ec="#c6ced4", lw=0.5, z=3)
    cx = rx + 0.026
    arrow(ax, (rx + 0.009, BOT + H - 0.050), (rx + 0.009, BOT + 0.062),
          color=INK, lw=0.7, ms=5, z=6)
    top = BOT + H - 0.055
    L_DEPTH, L_GAP1, L_SECOND, L_GAP2, L_OTHER = 0.090, 0.028, 0.035, 0.020, 0.050
    runs = ((top - L_DEPTH, L_DEPTH, DEPTH),
            (top - L_DEPTH - L_GAP1 - L_SECOND, L_SECOND, DEPTH),
            (top - L_DEPTH - L_GAP1 - L_SECOND - L_GAP2 - L_OTHER, L_OTHER, OTHER))
    for y0, ln, col in runs:
        box(ax, cx - 0.011, y0, 0.022, ln, fc=col, z=5, alpha=0.9, r=0.004)

    own = L_DEPTH + L_SECOND
    bars = ((0.700, ((own, DEPTH), (L_OTHER, OTHER)), "thickness"),
            (0.712, ((own, DEPTH),), "own occupancy"),
            (0.724, ((L_DEPTH, DEPTH),), "depth"))
    ax.plot([cx + 0.013, 0.726], [top, top], color="#c2c9cf", lw=0.5, ls=(0, (1.2, 1.2)),
            zorder=3)
    for mx, parts, lab in bars:
        y = top
        for ln, col in parts:
            ax.plot([mx, mx], [y, y - ln], color=col, lw=2.4, solid_capstyle="butt", zorder=5)
            y -= ln
        ax.text(mx + 0.003, y - 0.007, lab, fontsize=5.6, color=INK, ha="left", va="top")

    heading(ax, X[4], Wd[4], "5", "one record",
            "eight arrays over the plane, ray lengths as integer step counts")
    names = [("em", "#a9afb4"), ("inst_face", "#a9afb4"), ("cls_face", "#a9afb4"),
             ("mask", "#a9afb4"), ("depth_below_steps", DEPTH),
             ("own_occupancy_below_steps", DEPTH), ("thickness_below_steps", OTHER),
             ("clipped", "#a9afb4")]
    yy = BOT + H - 0.030
    for nm, col in names:
        box(ax, X[4], yy - 0.026, Wd[4], 0.021, fc="#f4f5f6", ec="#dde2e6", lw=0.5, z=3)
        ax.add_patch(Rectangle((X[4], yy - 0.026), 0.0045, 0.021, facecolor=col,
                               edgecolor="none", zorder=4))
        ax.text(X[4] + 0.010, yy - 0.0155, nm, fontsize=5.2, color=INK, va="center",
                family="DejaVu Sans Mono")
        yy -= 0.030

    for i in range(4):
        a = X[i] + Wd[i] + 0.006
        b = X[i + 1] - 0.010
        arrow(ax, (a, MID - 0.02), (b, MID - 0.02), color="#aeb5bb", lw=0.9, ms=7)

    ax.plot([0.02, 0.98], [0.105, 0.105], color="#e3e6e9", lw=0.6)
    ax.text(0.5, 0.055,
            f"{c['rows']:,} plane-and-instance records  ·  {c['instances']:,} instances  ·  "
            f"{c['classes']} organelle classes  ·  pixel sizes {px[0]} to {px[-1]} nm",
            fontsize=6.6, color=GREY, ha="center", va="center")

    emstyle.legend_row(fig, [("the exposed instance, its face and the depth below it", DEPTH),
                             ("material that is not the exposed instance", OTHER),
                             ("an extracted region", BOX)], y=0.004)

    png = emstyle.save(fig, "fig1_pipeline")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
