import collections, json, os
import numpy as np

import emstyle
from emstyle import INK, GREY, PALE, DEPTH, BOX, OTHER, SPECIMEN
import matplotlib.pyplot as plt

emstyle.apply()

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(f"{ROOT}/cache/validation_results.json"))

GOOD = "#2f6f4e"

FULL = 7.087

NUDGE = {"lyso": (-18, -8), "ld": (4, -2), "endo": (-20, 2), "golgi": (3, -8),
         "pm": (-14, 4), "chrom": (4, -6), "er": (3, 3), "mito": (-4, 6),
         "ne": (-16, -8), "nucleolus": (4, -2), "eres": (4, -3)}

TISSUE = {"jrc_mus-liver", "jrc_mus-liver-3", "jrc_mus-thymus-1", "jrc_mus-pancreas-4",
          "jrc_mus-skin-1", "jrc_mus-kidney-3", "jrc_mus-heart-1", "jrc_mus-hippocampus-1",
          "jrc_choroid-plexus-2"}

def load_table():
    cache = f"{ROOT}/cache/fig_corpus_cache.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    per_vol = collections.Counter(); per_vol_inst = collections.defaultdict(set)
    per_cls = collections.Counter(); per_cls_inst = collections.defaultdict(set)
    per_lvl = collections.Counter(); per_tier = collections.Counter()
    planes_per = collections.Counter()
    vols_by_cls = collections.defaultdict(list)
    seen = set()
    ex = os.environ.get("DEPOSIT_EXCLUSIONS")
    skip_box, skip_plane = set(), set()
    if ex:
        e = json.load(open(ex))
        skip_box, skip_plane = set(e["boxes"]), set(e["planes"])
    for line in open(f"{ROOT}/cache/inst_bf_exact_full.jsonl"):
        r = json.loads(line)
        if not r.get("unbiased"):
            continue
        if r["box"] in skip_box or r["file"][:-4] in skip_plane:
            continue
        k = (r["box"], r["instance"])
        per_vol[r["dataset"]] += 1; per_vol_inst[r["dataset"]].add(k)
        per_cls[r["organelle"]] += 1; per_cls_inst[r["organelle"]].add(k)
        per_lvl[float(r["px_nm"])] += 1; per_tier[r["tier"]] += 1
        planes_per[k] += 1
        if k not in seen:
            seen.add(k)
            vols_by_cls[r["organelle"]].append(r["volume_nm3"] / 1e9)
    d = dict(
        per_vol_inst={k: len(v) for k, v in per_vol_inst.items()},
        per_cls_inst={k: len(v) for k, v in per_cls_inst.items()},
        per_lvl=dict(per_lvl), per_tier=dict(per_tier),
        planes_hist=dict(collections.Counter(planes_per.values())),
        vols_by_cls={k: sorted(v) for k, v in vols_by_cls.items()},
    )
    json.dump(d, open(cache, "w"))
    return d

def fig_corpus(d):
    fig = plt.figure(figsize=(FULL, 6.8))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.54, bottom=0.185, left=0.135,
                          height_ratios=[1.35, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, :2])
    items = sorted(d["per_vol_inst"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = [SPECIMEN["tissue"] if n in TISSUE else SPECIMEN["cultured"] for n in names]
    ax.barh(range(len(names)), vals, color=cols, height=0.72, linewidth=0, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("jrc_", "").replace("aic_", "") for n in names])
    ax.set_xscale("log")
    ax.set_xlim(250, max(vals) * 1.6)
    ax.set_xlabel("contained instances")
    ax.axvline(300, color=DEPTH, lw=0.8, ls="--", zorder=4)
    ax.text(305, -0.9, "threshold 300", color=DEPTH, fontsize=5.8,
            va="bottom", ha="left")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "a", dx=-0.185)

    ax = fig.add_subplot(gs[0, 2])
    items = sorted(d["per_cls_inst"].items(), key=lambda kv: kv[1])
    ax.barh(range(len(items)), [v for _, v in items], color=INK, height=0.7,
            linewidth=0, zorder=3)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([emstyle.org_label(k) for k, _ in items])
    ax.set_xscale("log"); ax.set_xlabel("instances")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "b", dx=-0.42)

    ax = fig.add_subplot(gs[1, 0])
    px = sorted(float(k) for k in d["per_lvl"])
    tot = sum(d["per_lvl"].values())
    vals = [100 * d["per_lvl"][str(k) if str(k) in d["per_lvl"] else k] / tot for k in px]
    ax.bar(range(len(px)), vals, color=PALE, edgecolor=INK, linewidth=0.6, width=0.68,
           zorder=3)
    ax.set_xticks(range(len(px)))
    ax.set_xticklabels([f"{k:g}" for k in px])
    ax.set_ylabel("% of observations")
    ax.set_xlabel("pixel size (nm)", labelpad=1)
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "c", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[1, 1])
    ti = sorted(d["per_tier"].items(), key=lambda kv: -kv[1])
    tt = sum(v for _, v in ti)
    ax.bar([k for k, _ in ti], [100 * v / tt for _, v in ti], color=PALE,
           edgecolor=INK, linewidth=0.6, width=0.68, zorder=3)
    ax.set_ylabel("% of observations"); ax.set_xlabel("box tier")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "d", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[1, 2])
    h = {int(k): v for k, v in d["planes_hist"].items()}
    ks = sorted(h)
    ax.bar(ks, [h[k] for k in ks], color=PALE, edgecolor=INK, linewidth=0.6, width=0.72,
           zorder=3)
    ax.set_yscale("log"); ax.set_xlabel("planes showing one instance")
    ax.set_ylabel("instances")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "e", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[2, :])
    order = [c for c in ["vesicle", "eres", "lyso", "ld", "endo", "nucleolus", "golgi",
                         "chrom", "er", "pm", "ne", "mito", "nucleus"]
             if c in d["vols_by_cls"] and len(d["vols_by_cls"][c]) >= 50]
    data = [np.array(d["vols_by_cls"][c]) for c in order]
    bp = ax.boxplot(data, vert=True, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.0),
                    boxprops=dict(facecolor=PALE, edgecolor=INK, lw=0.6),
                    whiskerprops=dict(color=INK, lw=0.6),
                    capprops=dict(color=INK, lw=0.6))
    rel = R["delesse"]
    for i, c in enumerate(order):
        rr = rel.get(c, {}).get("ratio")
        if rr is None:
            continue
        col = GOOD if abs(rr - 1) <= 0.1 else (OTHER if abs(rr - 1) <= 0.5 else DEPTH)
        bp["boxes"][i].set_facecolor(col); bp["boxes"][i].set_alpha(0.42)
    ax.set_yscale("log")
    ax.set_xticklabels([emstyle.org_label(o) for o in order], rotation=45, ha="right")
    ax.set_ylabel(r"instance volume  ($\mu$m$^3$)")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "f", dx=-0.075)

    emstyle.legend_row(fig, [("cultured line", SPECIMEN["cultured"]),
                             ("tissue", SPECIMEN["tissue"])], y=0.004)

    print("->", emstyle.save(fig, "fig3_corpus"))

def fig_validation():
    fig = plt.figure(figsize=(FULL, 6.2))
    gs = fig.add_gridspec(3, 2, hspace=0.66, wspace=0.46, bottom=0.125)

    ax = fig.add_subplot(gs[0, 0])
    dd = R["delesse"]
    for c, v in dd.items():
        rr = v["ratio"]
        col = GOOD if abs(rr - 1) <= 0.1 else (OTHER if abs(rr - 1) <= 0.5 else DEPTH)
        ax.scatter(v["vol"], v["area"], s=16, color=col, zorder=3, linewidth=0)
        dx, dy = NUDGE.get(c, (4, -5))
        ax.annotate(c, (v["vol"], v["area"]), fontsize=5.4, color=INK,
                    xytext=(dx, dy), textcoords="offset points")
    lim = [5e-4, 1.2e2]
    ax.plot(lim, lim, color=GREY, lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("volume fraction (%)"); ax.set_ylabel("profile area fraction (%)")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="both")
    emstyle.panel(ax, "a", dx=-0.22)

    ax = fig.add_subplot(gs[0, 1])
    prev = R["delesse_prev_500box"]
    cs = [c for c in prev if c in dd]
    cs.sort(key=lambda c: dd[c]["ratio"])
    y = np.arange(len(cs))
    ax.hlines(y, [prev[c] for c in cs], [dd[c]["ratio"] for c in cs],
              color=PALE, lw=2.4, zorder=1)
    ax.scatter([prev[c] for c in cs], y, s=13, color=GREY, zorder=3, linewidth=0)
    ax.scatter([dd[c]["ratio"] for c in cs], y, s=13, color=INK, zorder=3, linewidth=0)
    ax.axvline(1.0, color=GOOD, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(cs)
    ax.set_xlabel("area fraction / volume fraction")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "b", dx=-0.22)

    ax = fig.add_subplot(gs[1, 0])
    THEORY = R["sphere_cut"]["_theory_pooled"]
    sc = R["sphere_cut"]["measured"]
    nn_ = R["sphere_cut"]["_n"]
    cs = sorted(sc, key=lambda c: -sc[c])
    cols = [GOOD if abs(sc[c] - THEORY) / THEORY <= 0.10 else GREY for c in cs]
    ax.bar(range(len(cs)), [sc[c] for c in cs], color=cols, width=0.66, linewidth=0,
           zorder=3)
    ax.axhline(THEORY, color=DEPTH, lw=0.9, ls="--", zorder=4)
    ax.text(len(cs) - 0.4, THEORY + 0.012, f"sphere 3/8 = {THEORY}", color=DEPTH,
            fontsize=5.8, ha="right")
    for i, c in enumerate(cs):
        ax.text(i, 0.012, f"{nn_[c]:,}", ha="center", fontsize=5.0, color=INK, rotation=90,
                zorder=5)
    ax.set_xticks(range(len(cs))); ax.set_xticklabels(cs, rotation=45, ha="right")
    ax.set_ylabel("mean depth / diameter")
    ax.set_ylim(0, 0.56)
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "c", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[1, 1])
    g = R["grouping"]
    cs = sorted(g, key=lambda c: g[c])
    cols = [GOOD if abs(g[c] - 1) <= 0.2 else (OTHER if g[c] >= 0.5 else DEPTH) for c in cs]
    ax.barh(range(len(cs)), [g[c] for c in cs], color=cols, height=0.68, linewidth=0,
            zorder=3)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--", zorder=4)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs)
    ax.set_xlabel("identifiers per 3D connected component")
    emstyle.despine(ax)
    emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "d", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[2, 0])
    lv = ["s0", "s1", "s2", "s3", "s4"]
    rc = R["recount_by_level"]
    ax2 = ax.twinx()
    ax2.bar(range(5), [rc[k]["off5"] for k in lv], color=PALE, width=0.66, linewidth=0,
            zorder=1)
    ax2.set_ylabel("% off by more than 5%", color=GREY)
    ax2.tick_params(axis="y", colors=GREY)
    ax.plot(range(5), [rc[k]["ratio"] for k in lv], "o-", color=DEPTH, lw=1.1, ms=4,
            zorder=3)
    ax.axhline(1.0, color=INK, lw=0.7, ls="--", zorder=2)
    ax.set_xticks(range(5)); ax.set_xticklabels(lv)
    ax.set_ylabel("counted / stored volume", color=DEPTH)
    ax.tick_params(axis="y", colors=DEPTH)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    emstyle.despine(ax)
    emstyle.despine(ax2, right=False)
    emstyle.panel(ax, "e", dx=-0.22, dy=1.10)

    ax = fig.add_subplot(gs[2, 1])
    ct = R["containment"]
    xs = [0, 1]
    med = [ct["unbiased_median"], ct["excluded_median"]]
    p99 = [ct["unbiased_p99"], ct["excluded_p99"]]
    mx = [ct["unbiased_max"], ct["excluded_max"]]
    ax.vlines(xs, med, mx, color=PALE, lw=7, zorder=1)
    ax.scatter(xs, mx, s=150, color=GREY, marker="_", linewidth=1.4, zorder=3,
               label="maximum")
    ax.scatter(xs, p99, s=150, color=BOX, marker="_", linewidth=1.4, zorder=4,
               label="99th pct")
    ax.scatter(xs, med, s=30, color=INK, zorder=5, linewidth=0, label="median")
    for x, v in zip(xs, med):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(9, -3), textcoords="offset points",
                    fontsize=5.8, color=INK)
    ax.axhline(1.0, color=GOOD, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(["kept as\ncontained", "excluded"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("source / counted volume")
    ax.legend(loc="upper center", frameon=False, handlelength=1.0, ncol=3,
              columnspacing=0.9, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0.5, 2e7)
    emstyle.despine(ax)
    emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "f", dx=-0.22, dy=1.10)

    emstyle.legend_row(fig, [("500 boxes, 11 volumes", GREY),
                             ("800 boxes, 18 volumes", INK)], y=0.005)

    print("->", emstyle.save(fig, "extra_delesse"))

if __name__ == "__main__":
    print("reading the scoring table", flush=True)
    d = load_table()
    fig_corpus(d)
    fig_validation()
