import glob, json, os
import numpy as np
import emstyle
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

emstyle.apply()

FIG_W = 7.087

MINOR_SHARE = 0.01
MINOR_COL = "#ccd2d7"

RAY_INK = emstyle.INK
RAY_MID = "#6e767d"
RAY_PALE = "#aeb6bc"


def score(f):
    try:
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        inst, dep, msk = z["inst_face"], z["depth_below_nm"], z["mask"]
    except Exception:
        return None
    n_inst = len(np.unique(inst)) - 1
    if not (8 <= n_inst <= 60):
        return None
    cov = float(msk.mean())
    if not (0.06 <= cov <= 0.55):
        return None
    d = dep[msk]
    if d.size < 500:
        return None
    classes = {r.get("organelle") for r in m.get("instances", {}).values()}
    classes.discard(None)
    clipped = float(z["clipped"][msk].mean()) if "clipped" in z.files else 1.0
    if clipped > 0.35:
        return None
    spread = float(np.percentile(d, 90) / max(np.percentile(d, 10), 1e-9))
    return dict(file=f, n_inst=n_inst, cov=cov, n_cls=len(classes), clipped=clipped,
                spread=spread, box=os.path.basename(os.path.dirname(os.path.dirname(f))),
                rank=len(classes) * 2 + min(spread, 12) * 0.4 + min(n_inst, 40) * 0.05)


def find_example(n_scan=900, seed=0):
    cache = f"{ROOT}/cache/fig1_example.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(glob.glob(f"{ROOT}/blockface/*/faces/*.npz"))
    files = [f for f in files if "/._" not in f]
    rng = np.random.default_rng(seed); rng.shuffle(files)
    best = []
    for i, f in enumerate(files[:n_scan]):
        if i % 150 == 0 and i:
            print(f"  scanned {i}, kept {len(best)}", flush=True)
        s = score(f)
        if s:
            best.append(s)
    best.sort(key=lambda r: -r["rank"])
    print(f"candidates {len(best)}")
    for r in best[:8]:
        print(f"  rank {r['rank']:.2f}  cls {r['n_cls']}  inst {r['n_inst']}  "
              f"cov {r['cov']:.2f}  clip {r['clipped']:.2f}  {r['box']}")
    json.dump(best[0], open(cache, "w"))
    return best[0]


def hex_rgb(c):
    return tuple(int(c[k:k + 2], 16) / 255 for k in (1, 3, 5))


def ecdf(v):
    u, c = np.unique(np.asarray(v, float), return_counts=True)
    y = np.cumsum(c) / float(v.size)
    return np.concatenate([[0.0], u]), np.concatenate([[0.0], y])


def pick_ortho_row(inst3, zidx, face_ids, org3, step=4):
    below = inst3[zidx:, :, :]
    Zb = below.shape[0]
    best, best_row = None, inst3.shape[1] // 2
    for row in range(0, inst3.shape[1], step):
        sl = below[:, row, :]
        u = np.unique(sl)
        u = u[u > 0]
        shown = [int(i) for i in u if int(i) in face_ids]
        cls = {org3.get(i) for i in shown}
        cls.discard(None)
        exposed = sl[0]
        runs = []
        for c in np.nonzero(exposed)[0]:
            i = int(exposed[c])
            r = 0
            while r < Zb and int(sl[r, c]) == i:
                r += 1
            runs.append(r)
        runs.sort(reverse=True)
        depth_score = sum(runs[:12]) / 12.0 if runs else 0.0
        s = len(shown) + 1.5 * len(cls) + depth_score
        if best is None or s > best:
            best, best_row = s, row
    return best_row


def main():
    ex = find_example()
    f = ex["file"]
    print(f"example {f}")
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    em, inst, dep, msk = z["em"], z["inst_face"], z["depth_below_nm"], z["mask"]
    own = z["own_occupancy_below_nm"] if "own_occupancy_below_nm" in z.files else None
    thick = z["thickness_below_nm"] if "thickness_below_nm" in z.files else None
    px = float(meta.get("pixel_size_nm", 0))
    zstep = float(meta.get("z_step_nm", px))
    box = ex["box"]
    zidx = int(os.path.basename(f).rsplit("_z", 1)[1].split(".")[0])

    org = {int(k): v.get("organelle") for k, v in meta.get("instances", {}).items()}

    raw = f"{ROOT}/raw/{box}.npz"
    org3 = {}
    if os.path.exists(raw):
        with np.load(raw, allow_pickle=False) as d3:
            if "meta" in d3.files:
                m3 = json.loads(str(d3["meta"]))
                org3 = {int(k): v.get("organelle") for k, v in m3.get("instances", {}).items()}

    def name_of(i):
        return org.get(int(i)) or org3.get(int(i))

    ids, npx = np.unique(inst[inst > 0], return_counts=True)
    area = {}
    for i, n in zip(ids, npx):
        o = name_of(i)
        area[o] = area.get(o, 0) + int(n)
    total = float(sum(area.values()))
    major = [o for o, n in sorted(area.items(), key=lambda kv: -kv[1])
             if o is not None and n / total >= MINOR_SHARE]
    pale_names = set()

    def cls_color(o):
        if o in major:
            return emstyle.CLS_COLOR.get(o, MINOR_COL)
        pale_names.add(o)
        return MINOR_COL

    print("face classes " + ", ".join(f"{o}:{100 * n / total:.1f}%" for o, n
                                      in sorted(area.items(), key=lambda kv: -kv[1])))

    n_cls_entries = len(major) + 1
    row_h = 0.027
    n_cls_rows = (n_cls_entries + 3) // 4
    y_cls = 0.006
    y_series = y_cls + n_cls_rows * row_h + 0.004
    gs_bottom = y_series + row_h + 0.085

    fig = plt.figure(figsize=(FIG_W, 5.6))
    outer = fig.add_gridspec(2, 1, hspace=0.11, height_ratios=[1.0, 1.06],
                             bottom=gs_bottom, top=0.965, left=0.085, right=0.915)
    gs_top = outer[0].subgridspec(1, 3, wspace=0.10)
    gs_bot = outer[1].subgridspec(1, 2, wspace=0.20, width_ratios=[1.0, 1.4])

    def scalebar(ax, nm=1000):
        n = nm / px
        H, W = em.shape
        ax.plot([W * 0.05, W * 0.05 + n], [H * 0.94, H * 0.94], color="w", lw=2.2,
                solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=3.4, foreground="k")])
        ax.text(W * 0.05 + n / 2, H * 0.915, f"{nm/1000:g} $\\mu$m", color="w",
                fontsize=5.8, ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="k")])

    ax = fig.add_subplot(gs_top[0, 0])
    ax.imshow(em, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    emstyle.despine(ax)
    scalebar(ax)
    emstyle.panel(ax, "a")

    ax = fig.add_subplot(gs_top[0, 1])
    ax.imshow(em, cmap="gray", interpolation="nearest", alpha=0.55)
    rgb = np.zeros(inst.shape + (4,), float)
    for i in np.unique(inst):
        if i == 0:
            continue
        r, g, b = hex_rgb(cls_color(name_of(i)))
        rgb[inst == i] = (r, g, b, 0.82)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    emstyle.despine(ax)
    emstyle.panel(ax, "b")

    ax = fig.add_subplot(gs_top[0, 2])
    ax.imshow(em, cmap="gray", interpolation="nearest", alpha=0.55)
    dm = np.ma.masked_invalid(np.where(msk, dep, np.nan))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(dm, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    emstyle.despine(ax)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("depth below the cut (nm)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)
    emstyle.panel(ax, "c")

    ax = fig.add_subplot(gs_bot[0, 0])
    em3 = inst3 = None
    if os.path.exists(raw):
        with np.load(raw, allow_pickle=False) as d3:
            if "em" in d3.files and "inst" in d3.files:
                em3, inst3 = d3["em"], d3["inst"]
    drew_ortho = False
    if inst3 is not None:
        W = em.shape[1]
        Z3, H3, W3 = inst3.shape
        x0 = (W3 - W) // 2 if W3 >= W else 0
        face_ids = {int(i) for i in np.unique(inst) if i}
        row = pick_ortho_row(inst3, zidx, face_ids, org3)
        print(f"orthogonal row {row} of {H3}")
        sl_em = em3[:, row, x0:x0 + W]
        sl_in = inst3[:, row, x0:x0 + W]
        ext = (0.0, sl_in.shape[1] * px, (Z3 - zidx) * zstep, -zidx * zstep)
        ax.imshow(sl_em, cmap="gray", interpolation="nearest", extent=ext, aspect="equal")
        ov = np.zeros(sl_in.shape + (4,), float)
        for i in np.unique(sl_in):
            if i == 0:
                continue
            r, g, b = hex_rgb(cls_color(name_of(i)))
            ov[sl_in == i] = (r, g, b, 0.55)
        ax.imshow(ov, interpolation="nearest", extent=ext, aspect="equal")
        ax.axhline(0, color=emstyle.DEPTH, lw=1.1,
                   path_effects=[pe.withStroke(linewidth=2.3, foreground="w")])
        ax.annotate("imaged face, shown in a to c", xy=(0.985, 0.0),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", color=emstyle.DEPTH,
                    fontsize=6, ha="right", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="w")])
        exposed = sl_in[zidx]
        cols = [c for c in range(sl_in.shape[1]) if exposed[c]]
        drawn = []; runs_drawn = []
        for c in cols:
            i = int(exposed[c])
            run = 0
            for z in range(zidx, Z3):
                if int(sl_in[z, c]) != i:
                    break
                run += 1
            if run < 2:
                continue
            if drawn and c - drawn[-1] < max(6, sl_in.shape[1] // 26):
                continue
            drawn.append(c); runs_drawn.append((c, run))
            xc = (c + 0.5) * px
            ax.plot([xc, xc], [0.0, run * zstep], color=emstyle.DEPTH, lw=0.9,
                    solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=2.1, foreground="w")])
            ax.plot([xc], [run * zstep], marker="_", ms=2.6, mew=0.9,
                    color=emstyle.DEPTH,
                    path_effects=[pe.withStroke(linewidth=2.1, foreground="w")])
        print(f"panel d, drew {len(drawn)} measured depths of {len(cols)} exposed columns")
        if drawn:
            lab, lab_run = max(runs_drawn, key=lambda t: t[1])
            ax.annotate("measured depth", xy=((lab + 0.5) * px, lab_run * zstep),
                        xytext=(10, 10), textcoords="offset points",
                        color=emstyle.DEPTH, fontsize=6, ha="left", va="top",
                        arrowprops=dict(arrowstyle="-", color=emstyle.DEPTH, lw=0.7),
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="w")])
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.set_ylabel("depth below the cut (nm)")
        ax.tick_params(axis="y", length=2.5)
        drew_ortho = True
    if not drew_ortho:
        print(f"warning, no orthogonal section for {box}, panel d is empty")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel("depth below the cut (nm)")
    emstyle.despine(ax)
    emstyle.panel(ax, "d")

    ax = fig.add_subplot(gs_bot[0, 1])
    series_entries = []
    if own is not None and thick is not None:
        dd, oo, tt = dep[msk], own[msk], thick[msk]
        hi = float(np.percentile(tt, 99.5))
        for v, c, lw, lab in ((tt, RAY_PALE, 1.8, "total path of any structure"),
                              (oo, RAY_MID, 1.3, "total path of the exposed instance"),
                              (dd, RAY_INK, 1.5, "unbroken run of the exposed instance")):
            x, y = ecdf(v)
            ax.plot(x, y, drawstyle="steps-post", color=c, lw=lw, zorder=3,
                    solid_capstyle="butt")
            series_entries.append((lab, c))
        series_entries.reverse()
        ax.set_xlim(0, hi)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("path below the cut (nm)")
        ax.set_ylabel("fraction of exposed pixels")
        emstyle.grid(ax)
        same = float(np.mean(np.abs(dd - oo) < 1e-3))
        print(f"depth equals own occupancy at {100 * same:.1f}% of exposed pixels")
    emstyle.despine(ax)
    emstyle.panel(ax, "e")

    if series_entries:
        emstyle.legend_row(fig, series_entries, y=y_series)
    cls_entries = [(emstyle.org_label(o), emstyle.CLS_COLOR.get(o, MINOR_COL)) for o in major]
    named_other = sorted(n for n in pale_names if n)
    if named_other:
        tail = " and unnamed" if None in pale_names else ""
        cls_entries.append((f"{len(named_other)} classes under 1%{tail}", MINOR_COL))
        print("pale classes " + ", ".join(named_other) + tail)
    elif pale_names:
        cls_entries.append(("unnamed instances", MINOR_COL))
    if cls_entries:
        emstyle.legend_row(fig, cls_entries, y=y_cls)

    out = emstyle.save(fig, "fig2_record")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
