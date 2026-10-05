import collections, json, os
import numpy as np
import emstyle
from emstyle import DEPTH, INK, GREY, PALE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors, ticker as mticker
from matplotlib.cm import ScalarMappable
from scipy import ndimage

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOV_NM = 2400.0
CELL_PX = 288

emstyle.apply()

FIG_W = 7.087
TILE_A, TILE_B, GAP, PAIR = 0.22, 0.45, 0.02, 0.05
MARG_L, MID, MARG_R = 1.22, 0.44, 0.92
TOP, BOT = 0.62, 0.46
LETTER_DY = 0.20


def index_by_px():
    cache = f"{ROOT}/cache/gallery_px_index.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    pool_px = collections.defaultdict(list)
    pool_vol = collections.defaultdict(list)
    for line in open(f"{ROOT}/cache/inst_bf_exact_full.jsonl"):
        r = json.loads(line)
        if not r.get("unbiased"):
            continue
        px = float(r["px_nm"])
        npix = r["area_nm2"] / (px * px)
        if npix < 18:
            continue
        rec = (npix, r["file"], r["instance"], px)
        pool_px[f"{r['organelle']}|{px:g}"].append(rec)
        pool_vol[f"{r['organelle']}|{r['dataset']}"].append(rec)

    def middle(pool, n=8):
        out = {}
        for k, v in pool.items():
            v.sort(key=lambda t: t[0])
            mid = len(v) // 2
            take = v[max(0, mid - n // 2): mid + n // 2 + 1]
            out[k] = [dict(file=f, inst=i, px=p, npix=round(a, 1))
                      for a, f, i, p in take]
        return out

    def finest(pool, n=8):
        out = {}
        for k, v in pool.items():
            v.sort(key=lambda t: t[0])
            mid = len(v) // 2
            near = {id(x): abs(i - mid) for i, x in enumerate(v)}
            v2 = sorted(v, key=lambda t: (t[3], near[id(t)]))
            out[k] = [dict(file=f, inst=i, px=p, npix=round(a, 1))
                      for a, f, i, p in v2[:n * 2]]
        return out

    d = dict(by_px=middle(pool_px), by_vol=finest(pool_vol))
    json.dump(d, open(cache, "w"))
    return d


def crop_for(rec, fov_nm=None, fixed_px=None):
    try:
        z = np.load(f"{ROOT}/{rec['file']}", allow_pickle=False)
        em, ef, dep = z["em"], z["inst_face"], z["depth_below_nm"]
    except Exception:
        return None
    px = float(rec["px"])
    half = (fixed_px // 2) if fixed_px else int(round((fov_nm or FOV_NM) / px / 2))
    half = max(4, min(half, min(em.shape) // 2))
    m = ef == rec["inst"]
    if m.sum() < 12:
        return None
    cy, cx = (int(v) for v in ndimage.center_of_mass(m))
    H, W = em.shape
    y0 = int(np.clip(cy - half, 0, max(H - 2 * half, 0)))
    x0 = int(np.clip(cx - half, 0, max(W - 2 * half, 0)))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    e, mm, dd = em[sl], m[sl], dep[sl]
    if e.shape[0] < 8 or e.shape[1] < 8 or not mm.any():
        return None
    side = 2 * half
    if e.shape[0] != side or e.shape[1] != side:
        py, px_ = side - e.shape[0], side - e.shape[1]
        if py < 0 or px_ < 0:
            return None
        pad = ((py // 2, py - py // 2), (px_ // 2, px_ - px_ // 2))
        e = np.pad(e, pad, constant_values=int(np.median(e)))
        mm = np.pad(mm, pad, constant_values=False)
        dd = np.pad(dd, pad, constant_values=np.nan)
    return dict(em=e, mask=mm, depth=dd, px=px)


def first_crop(recs, **kw):
    for rec in recs:
        c = crop_for(rec, **kw)
        if c is not None:
            return c
    return None


def ax_at(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def panel_letter(fig, ax, letter, x, y):
    fw, fh = fig.get_size_inches()
    b = ax.get_position()
    emstyle.panel(ax, letter, dx=(x / fw - b.x0) / b.width,
                  dy=1.0 + (y / fh - b.y1) / b.height)


def show_em(ax, c, alpha=1.0):
    a = c["em"].astype(float)
    lo, hi = np.percentile(a, [1, 99])
    ax.imshow(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray",
              interpolation="nearest", vmin=0, vmax=1, alpha=alpha)


def draw(ax, c, depth=False, stamp=None, norm=None):
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color(GREY)
    ax.set_xticks([]); ax.set_yticks([])
    if c is None:
        ax.set_facecolor(PALE)
        for s in ax.spines.values():
            s.set_visible(False)
        return False
    if depth:
        show_em(ax, c, alpha=0.40)
        cm = plt.get_cmap("magma").copy()
        cm.set_bad(alpha=0.0)
        ax.imshow(c["depth"], cmap=cm, norm=norm, interpolation="nearest")
    else:
        show_em(ax, c)
        ax.contour(c["mask"].astype(float), levels=[0.5], colors=[DEPTH], linewidths=0.6)
    if stamp:
        ax.text(0.04, 0.04, stamp, transform=ax.transAxes, fontsize=4.4, color="w",
                va="bottom", ha="left",
                path_effects=[pe.withStroke(linewidth=1.2, foreground=INK)])
    return True


def col_head(ax, text, fs=5.6, rot=0.0, ha="center"):
    ax.xaxis.set_ticks_position("top")
    lo, hi = sorted(ax.get_xlim())
    ax.set_xticks([(lo + hi) / 2])
    ax.set_xticklabels([text], fontsize=fs, rotation=rot, ha=ha, va="bottom",
                       rotation_mode="anchor", color=INK, linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=2.5)


def row_head(ax, text, fs=6.2, color=INK):
    lo, hi = sorted(ax.get_ylim())
    ax.set_yticks([(lo + hi) / 2])
    ax.set_yticklabels([text], fontsize=fs, color=color)
    ax.tick_params(axis="y", length=0, pad=2.5)


def main():
    idx = index_by_px()
    by_px, by_vol = idx["by_px"], idx["by_vol"]

    ORGS_A = ["nucleus", "er", "mito", "pm", "endo", "vesicle", "chrom", "golgi"]
    VOLS_ALL = ["jrc_hela-2", "jrc_jurkat-1", "jrc_hela-3", "jrc_macrophage-2",
                "jrc_hela-1", "aic_desmosome-2", "jrc_choroid-plexus-2", "jrc_mus-liver",
                "jrc_hela-bfa", "aic_desmosome-3", "jrc_cos7-11"]
    VOLS_NUC = ["jrc_mus-heart-1", "jrc_mus-kidney-3", "jrc_mus-pancreas-4",
                "jrc_mus-thymus-1", "jrc_mus-liver-3", "jrc_mus-skin-1",
                "jrc_mus-hippocampus-1"]
    ORGS_B = ["nucleolus", "mito"]
    PX_ALL = [4, 8, 16, 32, 64, 128]

    crops_a = {(o, v): first_crop(by_vol.get(f"{o}|{v}", []))
               for o in ORGS_A for v in VOLS_ALL}
    crops_n = {v: first_crop(by_vol.get(f"nucleus|{v}", [])) for v in VOLS_NUC}
    crops_b = {(o, p): first_crop(by_px.get(f"{o}|{p:g}", []), fixed_px=CELL_PX)
               for o in ORGS_B for p in PX_ALL}

    orgs_a = [o for o in ORGS_A if any(crops_a[(o, v)] for v in VOLS_ALL)]
    vols = [v for v in VOLS_ALL if any(crops_a[(o, v)] for o in orgs_a)]
    pxs = [p for p in PX_ALL if any(crops_b[(o, p)] for o in ORGS_B)]
    orgs_b = [o for o in ORGS_B if any(crops_b[(o, p)] for p in pxs)]
    dropped = dict(vols=[v for v in VOLS_ALL if v not in vols],
                   orgs_a=[o for o in ORGS_A if o not in orgs_a],
                   px=[p for p in PX_ALL if p not in pxs],
                   orgs_b=[o for o in ORGS_B if o not in orgs_b])

    pool = np.concatenate([c["depth"][np.isfinite(c["depth"])].ravel()
                           for c in crops_b.values() if c is not None])
    vmin, vmax = (float(v) for v in np.percentile(pool, [2, 98]))
    norm = mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)

    na, nb = len(vols), len(pxs)
    BETWEEN = 0.42
    CBAR = 0.42
    H_LIMIT = 8.66

    tile_a = (FIG_W - MARG_L - MARG_R - (na - 1) * GAP) / na
    H_A = len(orgs_a) * tile_a + (len(orgs_a) - 1) * GAP
    nn = len(VOLS_NUC)
    tile_n = tile_a
    W_N = nn * tile_n + (nn - 1) * GAP
    H_N = tile_n
    STRIP = 0.78
    rows_b = 2 * len(orgs_b)
    room = H_LIMIT - TOP - BOT - H_A - STRIP - H_N - BETWEEN - CBAR
    tile_b = (room - (rows_b - 1) * GAP - (len(orgs_b) - 1) * PAIR) / rows_b
    tile_b = max(0.30, min(tile_b, (FIG_W - MARG_L - MARG_R - (nb - 1) * GAP) / nb))
    W_A = na * tile_a + (na - 1) * GAP
    W_B = nb * tile_b + (nb - 1) * GAP
    H_B = rows_b * tile_b + (rows_b - 1) * GAP + (len(orgs_b) - 1) * PAIR
    FIG_H = TOP + H_A + STRIP + H_N + BETWEEN + H_B + CBAR + BOT
    x_b = (FIG_W - W_B) / 2
    y_top = FIG_H - TOP
    y_top_n = y_top - H_A - STRIP
    y_top_b = y_top_n - H_N - BETWEEN

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    filled_a = 0
    ax_a = None
    for i, o in enumerate(orgs_a):
        for j, v in enumerate(vols):
            ax = ax_at(fig, MARG_L + j * (tile_a + GAP),
                       y_top - (i + 1) * tile_a - i * GAP, tile_a, tile_a)
            if ax_a is None:
                ax_a = ax
            c = crops_a[(o, v)]
            filled_a += draw(ax, c, stamp=f"{c['px']:g} nm" if c else None)
            if i == 0:
                col_head(ax, v.replace("jrc_", "").replace("aic_", ""),
                         rot=45, ha="left")
            if j == 0:
                row_head(ax, emstyle.org_label(o))

    filled_n = 0
    ax_n = None
    for j, v in enumerate(VOLS_NUC):
        ax = ax_at(fig, MARG_L + j * (tile_n + GAP), y_top_n - tile_n, tile_n, tile_n)
        if ax_n is None:
            ax_n = ax
        c = crops_n[v]
        filled_n += draw(ax, c, stamp=f"{c['px']:g} nm" if c else None)
        col_head(ax, v.replace("jrc_", "").replace("aic_", ""), rot=45, ha="left")
        if j == 0:
            row_head(ax, "nucleus")

    filled_b = 0
    ax_b = None
    for i, o in enumerate(orgs_b):
        for j, p in enumerate(pxs):
            c = crops_b[(o, p)]
            x = x_b + j * (tile_b + GAP)
            y_pair = y_top_b - i * (2 * tile_b + 2 * GAP + PAIR)
            axp = ax_at(fig, x, y_pair - tile_b, tile_b, tile_b)
            axd = ax_at(fig, x, y_pair - 2 * tile_b - GAP, tile_b, tile_b)
            if ax_b is None:
                ax_b = axp
            filled_b += draw(axp, c)
            draw(axd, c, depth=True, norm=norm)
            if i == 0:
                col_head(axp, f"{p:g} nm\n{CELL_PX * p / 1000:.2f} $\\mu$m")
            if j == 0:
                row_head(axp, emstyle.org_label(o))
                row_head(axd, "depth", fs=5.4, color=GREY)

    cax = ax_at(fig, x_b + W_B * 0.18, y_top_b - H_B - 0.30, W_B * 0.64, 0.055)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), cax=cax,
                      orientation="horizontal")
    cb.set_label("depth below the cut (nm)", fontsize=6.4, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.locator = mticker.LogLocator(base=10.0, subs=(1.0, 3.0))
    cb.formatter = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    cb.update_ticks()

    panel_letter(fig, ax_a, "a", 0.14, FIG_H - LETTER_DY)
    panel_letter(fig, ax_n, "b", 0.14, y_top_n + 0.30)
    panel_letter(fig, ax_b, "c", 0.14, y_top_b + 0.30)

    emstyle.legend_row(fig, [("outline of the exposed instance", DEPTH),
                             ("a class the source never annotated", PALE)],
                       y=0.05 / FIG_H)

    out = emstyle.save(fig, "fig1_gallery")
    print(f"-> {out}")
    print(f"   a  {filled_a}/{len(orgs_a) * na} cells drawn, "
          f"{FOV_NM / 1000:g} um field of view in every cell")
    print(f"   b  {filled_n}/{nn} nucleus-only volumes drawn")
    print(f"   b  {filled_b}/{len(orgs_b) * nb} cells drawn, "
          f"{CELL_PX} px square, depth scale {norm.vmin:.0f} to {norm.vmax:.0f} nm, log")
    print(f"   dropped for carrying nothing: {dropped}")


if __name__ == "__main__":
    main()
