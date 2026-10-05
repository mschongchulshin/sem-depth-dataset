import collections, json, os, textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import emstyle
from emstyle import DEPTH, OTHER, INK, GREY, PALE, SPECIMEN

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(f"{ROOT}/cache/validation_results.json"))

emstyle.apply()

FIG_W = 7.087

CARD = "#f4f5f6"
NOTE_CARD = "#faf7f2"
NOTE_EDGE = "#ecdcc9"
BODY = "#5a6068"

THRESHOLD = 300

TISSUE = {"jrc_mus-liver", "jrc_mus-liver-3", "jrc_mus-thymus-1", "jrc_mus-pancreas-4",
          "jrc_mus-skin-1", "jrc_mus-kidney-3", "jrc_mus-heart-1", "jrc_mus-hippocampus-1",
          "jrc_choroid-plexus-2", "aic_desmosome-2", "aic_desmosome-3"}


def load():
    cache = f"{ROOT}/cache/fig_corpus_depth.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    per_vol = collections.defaultdict(set); per_cls = collections.defaultdict(set)
    per_px = collections.Counter(); per_tier = collections.Counter()
    planes_per = collections.Counter(); dep_by_cls = collections.defaultdict(list)
    seen = set()
    for line in open(f"{ROOT}/cache/inst_bf_exact_full.jsonl"):
        r = json.loads(line)
        if not r.get("unbiased"):
            continue
        k = (r["box"], r["instance"])
        per_vol[r["dataset"]].add(k); per_cls[r["organelle"]].add(k)
        per_px[float(r["px_nm"])] += 1; per_tier[r["tier"]] += 1
        planes_per[k] += 1
        d = r.get("depth_below_nm_median")
        if k not in seen and d and d > 0:
            seen.add(k); dep_by_cls[r["organelle"]].append(float(d))
    d = dict(per_vol={k: len(v) for k, v in per_vol.items()},
             per_cls={k: len(v) for k, v in per_cls.items()},
             per_px={str(k): v for k, v in per_px.items()}, per_tier=dict(per_tier),
             planes_hist=dict(collections.Counter(planes_per.values())),
             dep_by_cls={k: sorted(v) for k, v in dep_by_cls.items()})
    json.dump(d, open(cache, "w"))
    return d


def fig_corpus(d):
    fig = plt.figure(figsize=(FIG_W, 5.55))
    gs = fig.add_gridspec(3, 6, hspace=0.62, wspace=0.62,
                          height_ratios=[1.45, 0.90, 1.05],
                          left=0.10, right=0.985, top=0.965, bottom=0.115)

    ax = fig.add_subplot(gs[0, :4])
    items = sorted(d["per_vol"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.72, linewidth=0, zorder=3,
            color=[SPECIMEN["tissue"] if n in TISSUE else SPECIMEN["cultured"]
                   for n in names])
    for yi, v in zip(y, vals):
        ax.text(v * 1.14, yi, f"{v:,}", va="center", ha="left", fontsize=5.2, color=BODY)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("jrc_", "").replace("aic_", "") for n in names])
    ax.set_xscale("log"); ax.set_xlim(THRESHOLD, max(vals) * 1.9)
    ax.set_xlabel("contained instances")
    ax.set_ylim(-0.75, len(names) - 0.25)
    emstyle.despine(ax); emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "a")

    ax = fig.add_subplot(gs[0, 4:])
    items = sorted(d["per_cls"].items(), key=lambda kv: kv[1])
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=PALE, edgecolor=INK, linewidth=0.5,
            height=0.7, zorder=3)
    for yi, (_, v) in zip(y, items):
        ax.text(v * 1.22, yi, f"{v:,}", va="center", ha="left", fontsize=5.2, color=BODY)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in items])
    ax.set_xscale("log"); ax.set_xlim(THRESHOLD, max(v for _, v in items) * 3.2)
    ax.set_xlabel("instances")
    ax.set_ylim(-0.75, len(items) - 0.25)
    emstyle.despine(ax); emstyle.grid(ax, axis="x")
    emstyle.panel(ax, "b")

    ax = fig.add_subplot(gs[1, :3])
    px = sorted(float(k) for k in d["per_px"])
    tot = sum(d["per_px"].values())
    pct = [100 * d["per_px"][str(k)] / tot for k in px]
    ax.bar(range(len(px)), pct, color=PALE, edgecolor=INK, linewidth=0.6, width=0.68,
           zorder=3)
    for i, p in enumerate(pct):
        ax.text(i, p + max(pct) * 0.03, f"{p:.1f}", ha="center", va="bottom",
                fontsize=5.4, color=BODY)
    ax.set_ylim(0, max(pct) * 1.22)
    ax.set_xticks(range(len(px))); ax.set_xticklabels([f"{k:g}" for k in px])
    ax.set_ylabel("% of observations"); ax.set_xlabel("pixel size (nm)", labelpad=1)
    emstyle.despine(ax); emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "c")

    ax = fig.add_subplot(gs[1, 3:])
    h = {int(k): v for k, v in d["planes_hist"].items()}
    ks = sorted(h)
    ax.bar(ks, [h[k] for k in ks], color=PALE, edgecolor=INK, linewidth=0.6, width=0.72,
           zorder=3)
    ax.set_yscale("log")
    share = 100 * h[ks[0]] / sum(h.values())
    ax.text(ks[0], h[ks[0]] * 1.35, f"{share:.0f}% cut once", ha="left", va="bottom",
            fontsize=5.6, color=BODY)
    ax.set_ylim(top=h[ks[0]] * 3.6)
    ax.set_xlabel("planes showing one instance"); ax.set_ylabel("instances")
    emstyle.despine(ax); emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "d")

    ax = fig.add_subplot(gs[2, :])
    keep = [c for c, v in d["dep_by_cls"].items() if len(v) >= 50]
    order = sorted(keep, key=lambda c: float(np.median(d["dep_by_cls"][c])))
    data = [np.asarray(d["dep_by_cls"][c], float) for c in order]
    ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
               medianprops=dict(color=DEPTH, lw=1.1),
               boxprops=dict(facecolor=PALE, edgecolor=INK, lw=0.6),
               whiskerprops=dict(color=INK, lw=0.6),
               capprops=dict(color=INK, lw=0.6))
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("per-instance median depth (nm)")
    emstyle.despine(ax); emstyle.grid(ax, axis="y")
    emstyle.panel(ax, "e")

    emstyle.legend_row(fig, [("cultured line", SPECIMEN["cultured"]),
                             ("tissue", SPECIMEN["tissue"])], y=0.0)

    png = emstyle.save(fig, "fig3_corpus")
    print(f"-> {png}")


def fig_deposit():
    fig = plt.figure(figsize=(FIG_W, 3.42))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    C = R["corpus"]

    def rbox(x, y, w, h, fc, ec="none", lw=0.6, r=0.008, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

    def row(x, w, y, stripe, cells):
        rbox(x, y - RH, w, RH, CARD, PALE, 0.5)
        ax.add_patch(Rectangle((x, y - RH), 0.0045, RH, facecolor=stripe,
                               edgecolor="none", zorder=4))
        for tx, ha, fs, col, fam, txt in cells:
            ax.text(tx, y - RH / 2, txt, fontsize=fs, color=col, va="center", ha=ha,
                    family=fam)

    RH, GAPY = 0.044, 0.0515
    MONO = "DejaVu Sans Mono"
    SANS = "sans-serif"

    emstyle.panel(ax, "a", dx=0.028, dy=0.945)
    tree = [("blockface/.../faces/<box>_z<NNNN>.npz",
             f"{C['planes']:,} planes over {C['boxes']:,} boxes", DEPTH),
            ("splits.csv", f"{C['source_volumes']} folds, one per source volume", GREY),
            ("boxes.csv", "origin and extent of every box", GREY),
            ("sources.csv", "specimen, pixel sizes, classes", GREY),
            ("instance_index.parquet", "one row per instance and plane", GREY),
            ("checksums.sha256", "one line per shipped file", GREY),
            ("README.md", "the data dictionary", GREY)]
    y = 0.925
    for name, note, col in tree:
        row(0.012, 0.455, y, col,
            [(0.024, "left", 5.8, INK, MONO, name),
             (0.460, "right", 5.5, GREY, SANS, note)])
        y -= GAPY
    ax.text(0.012, y - 0.006,
            "4.71 GB. The 3D sub-volumes are not shipped. boxes.csv records the\n"
            "region, so every one is recoverable from the public reconstruction.",
            fontsize=5.7, color=GREY, va="top", linespacing=1.6)

    emstyle.panel(ax, "b", dx=0.550, dy=0.945)
    arrs = [("em", "uint8", "the micrograph"),
            ("inst_face", "int32", "which instance"),
            ("cls_face", "uint8", "which class, box-scoped"),
            ("mask", "bool", "where label is cut"),
            ("depth_below_steps", "uint16", "contiguous run"),
            ("own_occupancy_below_steps", "uint16", "its total path below"),
            ("thickness_below_steps", "uint16", "any structure below"),
            ("clipped", "bool", "reached the block bottom")]
    y = 0.925
    for nm, dt, note in arrs:
        row(0.534, 0.462, y, DEPTH if "steps" in nm else GREY,
            [(0.546, "left", 5.6, INK, MONO, nm),
             (0.824, "right", 5.3, BODY, MONO, dt),
             (0.991, "right", 5.4, GREY, SANS, note)])
        y -= GAPY
    ax.text(0.534, y - 0.006,
            "Multiply a step count by z_step_nm for nanometres. A count of 0 means\n"
            "nothing is exposed, not zero depth, and mask is the safer test.",
            fontsize=5.7, color=GREY, va="top", linespacing=1.6)

    emstyle.panel(ax, "c", dx=0.028, dy=0.400)
    joins = [
        ("cls_face is scoped to its own box",
         "Identifier 7 is er across 51% of the pixels it covers corpus-wide and nucleolus "
         "across most of the rest. Join through the organelle name."),
        ("instance ids are unique within a box only",
         "7.5% of numeric ids name different organelles in different boxes. (box, instance) "
         "is the smallest safe key, and no reconstruction-wide id exists."),
        ("the level name fixes no pixel size",
         "s0 is 4 nm in six cultured lines, 16 nm in mus-liver, 128 nm in seven "
         "tissues. Use px_nm, and read z_step_nm separately."),
    ]
    y = 0.375
    CH = 0.090
    for i, (t, b) in enumerate(joins):
        rbox(0.012, y - CH, 0.978, CH, NOTE_CARD, NOTE_EDGE, 0.6)
        ax.add_patch(Rectangle((0.012, y - CH), 0.0045, CH, facecolor=OTHER,
                               edgecolor="none", zorder=4))
        ax.text(0.026, y - 0.030, f"{i+1}.   {t}", fontsize=6.6, color=INK,
                fontweight="semibold", va="center")
        ax.text(0.026, y - 0.067, textwrap.fill(b, 150), fontsize=5.7,
                color=BODY, va="center", linespacing=1.55)
        y -= CH + 0.013

    emstyle.legend_row(fig, [("the depth and the arrays that carry it", DEPTH),
                             ("the tables that index them", GREY),
                             ("a join that needs care", OTHER)], y=0.004)

    png = emstyle.save(fig, "fig5_deposit")
    print(f"-> {png}")


if __name__ == "__main__":
    print("reading the scoring table", flush=True)
    fig_corpus(load())
    fig_deposit()
