import os
import json
import numpy as np
import matplotlib.pyplot as plt

import emstyle
from emstyle import DEPTH, OTHER, BOX, INK, GREY, PALE

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(f"{ROOT}/cache/validation_results.json"))

emstyle.apply()


def fig_validation():
    fig = plt.figure(figsize=(7.087, 6.3))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.40, height_ratios=[0.82, 1.0],
                          left=0.175, right=0.975, top=0.955, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    sc = R["sphere_cut"]
    TH = sc["_theory_pooled"]
    meas, nn = sc["measured"], sc["_n"]
    cs = sorted(meas, key=lambda c: -meas[c])
    cols = [BOX if abs(meas[c] - TH) / TH <= 0.10 else GREY for c in cs]
    ax.bar(range(len(cs)), [meas[c] for c in cs], color=cols, width=0.66,
           linewidth=0, zorder=3)
    ax.axhline(TH, color=INK, lw=0.8, ls="--", zorder=4)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([f"{emstyle.org_label(c)} ({nn[c]:,})" for c in cs],
                       rotation=45, ha="right")
    ax.set_ylabel("mean depth / diameter")
    ax.set_ylim(0, 0.52)
    ax.set_xlim(-0.7, len(cs) - 0.3)
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    sec = ax.secondary_yaxis("right")
    sec.set_yticks([TH])
    sec.set_yticklabels(["3/8"])
    sec.spines["right"].set_visible(False)
    sec.tick_params(length=2.5, width=0.6, colors=INK, labelsize=6.5)
    emstyle.panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.42])[0])
    cl = R["depth_reliability"]["clipped_pct"]
    cs = sorted([c for c in cl if cl[c] > 0], key=lambda c: -cl[c])
    vals = [cl[c] for c in cs]
    emstyle.hbar(ax, [emstyle.org_label(c) for c in cs], vals,
                 hero=emstyle.org_label(cs[0]), fmt="{:.2f}", pad=0.035)
    ax.set_xlabel("% of pixels where depth is a lower bound")
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_ylim(-0.8, len(cs) - 0.2)
    emstyle.panel(ax, "b", dy=1.06)

    ax = fig.add_subplot(gs[1, 0])
    u = json.load(open(f"{ROOT}/cache/depth_uncertainty.json"))["by_class"]
    cs = sorted(u, key=lambda c: -u[c]["zero"])
    vals = [100.0 * u[c]["zero"] for c in cs]
    emstyle.hbar(ax, [emstyle.org_label(c) for c in cs], vals, fmt="{:.0f}", pad=0.02)
    ax.set_xlabel("% of neighbouring pixel pairs where a one-pixel lateral\n"
                  "error leaves the depth unchanged, high means a stable depth")
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.7, len(cs) - 0.3)
    emstyle.panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    lv = R["label_vs_image"]
    W = lv["window_steps"]
    off = np.arange(-W, W + 1)
    SHOW = [("mito", DEPTH), ("nucleus", OTHER), ("chrom", BOX),
            ("er", GREY), ("pm", INK)]
    drawn = []
    for c, col in SHOW:
        prof = lv["mean_profiles"].get(c)
        if not prof:
            continue
        ax.plot(off, prof, color=col, lw=1.2, label=emstyle.org_label(c), zorder=3)
        drawn.append((emstyle.org_label(c), col))
        if c in ("mito", "nucleus"):
            j = int(np.argmin(prof))
            ax.scatter([off[j]], [prof[j]], s=16, color=col, zorder=4, linewidth=0)
    ax.axvline(0, color=GREY, lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("z steps from the label's last interior step")
    ax.set_ylabel("class-mean normalised intensity")
    ax.set_xticks([-8, -4, 0, 4, 8])
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "d")

    emstyle.legend_row(fig, drawn, y=0.008)
    out = emstyle.save(fig, "fig4_validation")
    print(f"-> {out}")


def fig_reference_bar():
    b = R["depth_baseline"]
    per = b["per_fold"]
    ARMS = [("constant", "constant", "constant", GREY),
            ("edt", "edt_geometry_only", "silhouette geometry", PALE),
            ("photo_only", "photo_only", "photograph", BOX),
            ("photo_plus_silhouette", "photo_plus_silhouette",
             "photograph + silhouette", DEPTH)]
    folds = sorted(per, key=lambda f: -per[f]["photo_plus_silhouette"]["d1"])

    fig = plt.figure(figsize=(7.087, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.62, width_ratios=[1.85, 1.0],
                          left=0.115, right=0.985, top=0.955, bottom=0.145)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(folds))
    for i, f in enumerate(folds):
        lo = per[f]["constant"]["d1"]
        hi = max(per[f][k]["d1"] for k, _, _, _ in ARMS if per[f].get(k))
        ax.hlines(i, lo, hi, color=PALE, lw=1.0, zorder=1)
    for key, _mk, lab, col in ARMS:
        v = [per[f][key]["d1"] if per[f].get(key) else np.nan for f in folds]
        ax.scatter(v, y, s=18, color=col, zorder=3, linewidth=0.35,
                   edgecolors="white", label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("jrc_", "").replace("aic_", "") for f in folds])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\delta_1$, share of pixels within a factor of 1.25")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(len(folds) - 0.3, -0.9)
    emstyle.despine(ax)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    m = b["mean_over_17_folds"]
    keys = [mk for _, mk, _, _ in ARMS]
    labs = [l for _, _, l, _ in ARMS]
    cols = [c for _, _, _, c in ARMS]
    vals = [m[k]["d1"] for k in keys]
    ax.barh(range(len(keys)), vals, color=cols, height=0.62, linewidth=0, zorder=3)
    for i, (k, v) in enumerate(zip(keys, vals)):
        ax.text(v + 0.006, i - 0.16, f"{v:.3f}", va="center", fontsize=7, color=INK)
        ax.text(v + 0.006, i + 0.2, f"rel. error {m[k]['medrel']:.2f}", va="center",
                fontsize=5.4, color=GREY)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labs)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"mean $\delta_1$ over 17 folds")
    ax.set_xlim(0, 0.30)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    emstyle.despine(ax, left=True)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "b")

    emstyle.legend_row(fig, [(lab, col) for _, _, lab, col in ARMS], y=0.015)
    out = emstyle.save(fig, "extra_reference_bar")
    print(f"-> {out}")


if __name__ == "__main__":
    fig_validation()
    fig_reference_bar()
