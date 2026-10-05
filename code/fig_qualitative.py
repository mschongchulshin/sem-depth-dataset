import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/paper/figures"
R = json.load(open(f"{ROOT}/cache/validation_results.json"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK, GREY = "#1a1a1a", "#8d9399"
NCOL = 5


def pick(d, boxes, n=NCOL):
    v, em = d["valid"], d["em"].astype(np.float32)
    order = list(np.argsort([-int(v[i].sum()) for i in range(v.shape[0])]))
    flat = em.reshape(em.shape[0], -1)
    flat = (flat - flat.mean(1, keepdims=True)) / np.maximum(flat.std(1, keepdims=True), 1e-6)

    out = [int(order[0])]
    rest = [int(i) for i in order[1:]]
    while len(out) < n and rest:
        sim = [max(float(abs(flat[i] @ flat[j]) / flat.shape[1]) for j in out) for i in rest]
        out.append(rest.pop(int(np.argmin(sim))))
    return out
    rest = [int(i) for i in order if i not in out]
    while len(out) < n and rest:
        sim = [max(float(abs(flat[i] @ flat[j]) / flat.shape[1]) for j in out) for i in rest]
        k = int(np.argmin(sim))
        out.append(rest.pop(k))
    return out


def main():
    folds = [("jrc_mus-thymus-1", "best fold"), ("jrc_choroid-plexus-2", "hardest fold")]
    per = R["depth_baseline"]["per_fold"]

    meta = json.load(open(f"{CACHE}/meta.json"))["meta"]
    ds = np.array([m["dataset"] for m in meta])
    bx = np.array([m["box"] for m in meta])

    fig = plt.figure(figsize=(7.2, 7.4))
    outer = fig.add_gridspec(2, 1, hspace=0.20)

    for bi, (fold, tag) in enumerate(folds):
        d = np.load(f"{CACHE}/qual_{fold}.npz")
        te = np.flatnonzero(ds == fold)
        order_in_dump = sorted(te, key=lambda i: -float(np.load(f"{CACHE}/valid.npy",
                                                               mmap_mode="r")[i].mean()))
        boxes = [bx[i] for i in order_in_dump[:d["em"].shape[0]]]
        idx = pick(d, boxes)
        gs = outer[bi].subgridspec(3, NCOL, wspace=0.05, hspace=0.05)
        d1 = per[fold]["photo_plus_silhouette"]["d1"]
        mr = per[fold]["photo_plus_silhouette"]["medrel"]

        for j, i in enumerate(idx):
            em, tr, pr, va = d["em"][i], d["truth"][i], d["pred"][i], d["valid"][i]
            vmax = float(np.percentile(tr[va], 99)) if va.any() else 1.0
            vmin = float(np.percentile(tr[va], 1)) if va.any() else 0.0

            ax = fig.add_subplot(gs[0, j])
            a = em.astype(float)
            lo, hi = np.percentile(a, [1, 99])
            ax.imshow(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#c3cad0")
            if j == 0:
                ax.set_ylabel("micrograph", fontsize=5.8, rotation=0, ha="right",
                              va="center", labelpad=5, color=INK)

            for k, (arr, lab) in enumerate(((tr, "true depth"), (pr, "predicted"))):
                ax = fig.add_subplot(gs[k + 1, j])
                m = np.where(va, arr, np.nan)
                cm = plt.get_cmap("magma").copy(); cm.set_bad("#2b2b2b")
                ax.imshow(m, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4); s.set_color("#c3cad0")
                if j == 0:
                    ax.set_ylabel(lab, fontsize=5.8, rotation=0, ha="right",
                                  va="center", labelpad=5,
                                  color=INK if k == 0 else "#a03530")
            ax = fig.axes[-1]
            ax.text(0.04, 0.05, f"{vmin:.0f} to {vmax:.0f} nm", transform=ax.transAxes,
                    fontsize=4.4, color="w", va="bottom",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="#111")])

        y = 0.965 if bi == 0 else 0.475
        fig.text(0.5, y,
                 f"{'ab'[bi]}    {fold.replace('jrc_', '')}, held out.  {tag}, "
                 f"$\\delta_1$ = {d1:.3f}, median relative error {mr:.2f}",
                 fontsize=7.4, ha="center", va="bottom", color=INK)

    fig.text(0.5, 0.012,
             "Each column is one cut plane. The true and predicted maps of a column share one "
             "colour scale, printed on the prediction.\nGrey is where nothing is exposed and "
             "nothing is scored.",
             fontsize=6.2, ha="center", va="bottom", color=GREY, linespacing=1.5)

    fig.savefig(f"{OUT}/fig8_qualitative.pdf")
    fig.savefig(f"{OUT}/fig8_qualitative.png")
    plt.close(fig)
    print("-> fig8_qualitative")


if __name__ == "__main__":
    main()
