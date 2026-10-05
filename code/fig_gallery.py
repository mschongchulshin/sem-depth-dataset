import collections, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/paper/figures"
FOV_NM = 2400.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK = "#1a1a1a"
EMPTY = "#eceff1"
OUTLINE = "#ffd54a"


def index_corpus():
    cache = f"{ROOT}/cache/gallery_index.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    best = collections.defaultdict(list)
    for line in open(f"{ROOT}/cache/inst_bf_exact_full.jsonl"):
        r = json.loads(line)
        if not r.get("unbiased"):
            continue
        a = r["area_nm2"]
        if not (0.02e6 < a < 1.2e6):
            continue
        k = f"{r['organelle']}|{r['dataset']}|{r['level']}"
        if len(best[k]) < 6:
            best[k].append(dict(file=r["file"], inst=r["instance"], px=r["px_nm"], area=a))
    json.dump(best, open(cache, "w"))
    return best


def crop_for(rec):
    try:
        z = np.load(f"{ROOT}/{rec['file']}", allow_pickle=False)
        em, ef, dep = z["em"], z["inst_face"], z["depth_below_nm"]
    except Exception:
        return None
    px = float(rec["px"])
    half = int(round(FOV_NM / px / 2))
    m = ef == rec["inst"]
    if m.sum() < 12:
        return None
    cy, cx = ndimage.center_of_mass(m)
    cy, cx = int(cy), int(cx)
    H, W = em.shape
    y0 = int(np.clip(cy - half, 0, max(H - 2 * half, 0)))
    x0 = int(np.clip(cx - half, 0, max(W - 2 * half, 0)))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    e, mm, dd = em[sl], m[sl], dep[sl]
    if e.shape[0] < 8 or e.shape[1] < 8 or not mm.any():
        return None
    return dict(em=e, mask=mm, depth=dd, px=px)


def stamp(ax, text, color="w"):
    ax.text(0.04, 0.04, text, transform=ax.transAxes, fontsize=4.8, color=color,
            va="bottom", ha="left",
            path_effects=[pe.withStroke(linewidth=1.3, foreground="#111111")])


def draw(ax, c, show_depth=False):
    if c is None:
        ax.set_facecolor(EMPTY)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return False
    if show_depth:
        d = np.where(c["mask"], c["depth"], np.nan)
        cm = plt.get_cmap("magma").copy(); cm.set_bad("#2b2b2b")
        ax.imshow(d, cmap=cm, interpolation="nearest")
    else:
        a = c["em"].astype(float)
        lo, hi = np.percentile(a, [1, 99])
        ax.imshow(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.contour(c["mask"].astype(float), levels=[0.5], colors=[OUTLINE],
                   linewidths=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#b0bec5")
    return True


def fig_organelle_by_volume(idx):
    ORGS = ["mito", "endo", "lyso", "ld", "nucleolus", "chrom", "golgi", "ne", "er"]
    VOLS = ["jrc_mus-liver", "jrc_choroid-plexus-2", "jrc_macrophage-2", "jrc_jurkat-1",
            "jrc_hela-2", "jrc_hela-3", "aic_desmosome-2"]
    fig, axes = plt.subplots(len(ORGS), len(VOLS),
                             figsize=(len(VOLS) * 0.92, len(ORGS) * 0.92))
    filled = 0
    for i, o in enumerate(ORGS):
        for j, v in enumerate(VOLS):
            ax = axes[i, j]
            c, used = None, None
            for lv in ("s2", "s1", "s3", "s0", "s4"):
                for rec in idx.get(f"{o}|{v}|{lv}", []):
                    c = crop_for(rec)
                    if c is not None:
                        used = lv
                        break
                if c is not None:
                    break
            filled += draw(ax, c)
            if c is not None:
                stamp(ax, f"{used}  {c['px']:g} nm")
            if i == 0:
                ax.set_title(v.replace("jrc_", "").replace("aic_", ""), fontsize=6,
                             pad=3, color=INK)
            if j == 0:
                ax.set_ylabel(o, fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=6, color=INK)
    fig.suptitle(f"one organelle name across seven reconstructions. "
                 f"{FOV_NM/1000:g} $\\mu$m field of view in every cell, so a larger object "
                 f"is larger.\nThe stamp gives the pyramid level and pixel size. "
                 f"Grey cells are pairs the source never annotated",
                 fontsize=7, y=1.0, va="bottom")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.965)
    fig.savefig(f"{OUT}/fig4_gallery.pdf"); fig.savefig(f"{OUT}/fig4_gallery.png")
    plt.close(fig)
    print(f"-> fig4_gallery   filled {filled}/{len(ORGS)*len(VOLS)}")


def fig_scale(idx):
    ORGS = ["mito", "endo", "lyso", "chrom", "nucleolus"]
    LEVELS = ["s0", "s1", "s2", "s3", "s4"]
    rows = len(ORGS) * 2
    fig, axes = plt.subplots(rows, len(LEVELS), figsize=(len(LEVELS) * 1.05, rows * 1.05))
    for i, o in enumerate(ORGS):
        for j, lv in enumerate(LEVELS):
            c = None
            for v in ["jrc_mus-liver", "jrc_macrophage-2", "jrc_jurkat-1", "jrc_hela-2",
                      "jrc_hela-3", "jrc_choroid-plexus-2", "aic_desmosome-2",
                      "jrc_hela-1", "jrc_mus-thymus-1"]:
                for rec in idx.get(f"{o}|{v}|{lv}", []):
                    c = crop_for(rec)
                    if c is not None:
                        break
                if c is not None:
                    break
            draw(axes[2 * i, j], c, show_depth=False)
            draw(axes[2 * i + 1, j], c, show_depth=True)
            if c is not None:
                src = [k for k in idx if k.startswith(f"{o}|") and k.endswith(f"|{lv}")]
                stamp(axes[2 * i, j], f"{c['px']:g} nm/px")
            if i == 0:
                axes[0, j].set_title(f"{lv}", fontsize=6, pad=3, color=INK)
        axes[2 * i, 0].set_ylabel(o, fontsize=6.5, rotation=0, ha="right", va="center",
                                  labelpad=6, color=INK)
        axes[2 * i + 1, 0].set_ylabel("depth", fontsize=5.5, rotation=0, ha="right",
                                      va="center", labelpad=6, color="#78909c")
    fig.suptitle(f"one instance of each class at each pyramid level, photograph above and "
                 f"depth below cut beneath it. {FOV_NM/1000:g} $\\mu$m field of view "
                 f"throughout.\nCells are different instances, not one object resampled, and "
                 f"the level name fixes no scale, so each cell is stamped with its own pixel "
                 f"size", fontsize=7, y=1.0, va="bottom")
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.965)
    fig.savefig(f"{OUT}/fig5_scale.pdf"); fig.savefig(f"{OUT}/fig5_scale.png")
    plt.close(fig)
    print("-> fig5_scale")


if __name__ == "__main__":
    print("indexing", flush=True)
    idx = index_corpus()
    print(f"index keys {len(idx):,}", flush=True)
    fig_organelle_by_volume(idx)
    fig_scale(idx)
