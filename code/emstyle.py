import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as plt

DEPTH = "#a03530"
OTHER = "#c8952c"
BOX = "#2b5d8a"
INK = "#1a1a1a"
GREY = "#8d9399"
PALE = "#dfe4e8"

CLS_FAMILY = {
    "nucleus": "nuclear", "chrom": "nuclear", "nucleolus": "nuclear", "ne": "nuclear",
    "er": "endomembrane", "golgi": "endomembrane", "endo": "endomembrane",
    "lyso": "endomembrane", "vesicle": "endomembrane", "eres": "endomembrane",
    "ld": "endomembrane",
    "mito": "mitochondrial",
    "pm": "boundary", "ribo": "boundary",
}
FAMILY_COLOR = {"nuclear": "#7b4ea3", "endomembrane": "#2b5d8a",
                "mitochondrial": "#a03530", "boundary": "#8d9399"}
CLS_COLOR = {
    "nucleus": "#5c3580", "chrom": "#7b4ea3", "nucleolus": "#a07cc4", "ne": "#c3aada",
    "er": "#1f4568", "golgi": "#2b5d8a", "endo": "#4b86b4", "lyso": "#74a9d8",
    "vesicle": "#9dc6e8", "eres": "#c0dbf0", "ld": "#5f93bd",
    "mito": "#a03530",
    "pm": "#8d9399", "ribo": "#b9bec3",
}
SPECIMEN = {"cultured": "#2b5d8a", "tissue": "#c8952c"}
SEQ = "Blues"
DIVERGE = "RdBu_r"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "legend.fontsize": 6.6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 300, "svg.fonttype": "none",
        "savefig.bbox": None, "savefig.pad_inches": 0.02,
        "figure.dpi": 120, "axes.grid": False,
        "grid.color": "#e9e9e9", "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False,
    })


def despine(ax, top=True, right=True, left=False, bottom=False):
    for side, off in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        if off:
            ax.spines[side].set_visible(False)


def panel(ax, letter, dx=-0.02, dy=1.04, fs=12):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="bottom", ha="right")


def grid(ax, axis="both"):
    ax.grid(axis=axis, ls="-", lw=0.4, color="#ececec", zorder=0)
    ax.set_axisbelow(True)


def hbar(ax, labels, values, hero=None, fmt="{:,.0f}", pad=0.01, color=None):
    import numpy as np
    y = np.arange(len(labels))[::-1]
    cols = [(DEPTH if l == hero else (color or GREY)) for l in labels]
    ax.barh(y, values, color=cols, height=0.62, zorder=3)
    for yi, v, l in zip(y, values, labels):
        ax.text(v * (1 + pad) if ax.get_xscale() == "log" else v + max(values) * pad,
                yi, fmt.format(v), va="center", ha="left",
                fontsize=6.4, fontweight="bold" if l == hero else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for t, l in zip(ax.get_yticklabels(), labels):
        if l == hero:
            t.set_color(DEPTH); t.set_fontweight("bold")
    despine(ax)
    grid(ax, axis="x")


def strip(ax, labels, series, hero=None, s=9):
    import numpy as np
    y = np.arange(len(labels))[::-1]
    for yi, l in zip(y, labels):
        v = np.asarray(series[l], float)
        c = DEPTH if l == hero else GREY
        ax.scatter(v, np.full(v.size, yi), s=s, c=c, alpha=0.85,
                   linewidths=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for t, l in zip(ax.get_yticklabels(), labels):
        if l == hero:
            t.set_color(DEPTH); t.set_fontweight("bold")
    ax.set_ylim(-0.7, len(labels) - 0.3)
    grid(ax, axis="x")


def legend_row(fig, entries, y=None, fs=6.6, ncol=None, reserve=True):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=t) for t, c in entries]
    n = ncol or min(len(entries), 5)
    rows = (len(entries) + n - 1) // n
    if y is None:
        axes = [a for a in fig.axes if a.get_visible()]
        low = min((a.get_position().y0 for a in axes), default=0.1)
        pad = 0.075 + 0.022 * (rows - 1)
        if reserve and low < pad + 0.04:
            shift = pad + 0.04 - low
            for a in axes:
                b = a.get_position()
                a.set_position([b.x0, b.y0 + shift, b.width, b.height * (1 - shift)])
            low = low + shift
        y = max(low - pad, 0.005)
    fig.legend(handles=handles, loc="lower center", ncol=n,
               frameon=False, fontsize=fs, bbox_to_anchor=(0.5, y),
               handlelength=1.1, handleheight=0.8, columnspacing=1.6)


def save(fig, name, outdir="${SEM_DEPTH_ROOT}/paper/figures"):
    from pathlib import Path
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.svg")
    fig.savefig(p / f"{name}.pdf")
    fig.savefig(p / f"{name}.png", dpi=600)
    plt.close(fig)
    return str(p / f"{name}.svg")


ORG_FULL = {
    "mito": "mitochondrion", "nucleus": "nucleus", "nucleolus": "nucleolus",
    "chrom": "chromatin", "endo": "endosome", "lyso": "lysosome",
    "ld": "lipid droplet", "er": "endoplasmic reticulum",
    "eres": "ER exit site", "golgi": "Golgi apparatus",
    "ne": "nuclear envelope", "pm": "plasma membrane", "vesicle": "vesicle",
    "ribo": "ribosome",
}


def org_label(name):
    return ORG_FULL.get(str(name), str(name))
