import collections, glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"

def _excluded():
    import json as _j, os as _o
    p = _o.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = _j.load(open(p))
    return set(e["boxes"]), set(e["planes"])

NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 70
NPLANE = 4


def runs_below(match):
    Z = match.shape[0]
    first_false = np.argmax(~match, axis=0)
    all_true = match.all(axis=0)
    return np.where(all_true, Z, first_false)


def main():
    _sb, _sp = _excluded()
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in _sb)
    rng = np.random.default_rng(0); rng.shuffle(raws)

    ratio = collections.defaultdict(list)
    split = collections.Counter()
    total = collections.Counter()
    union_ratio = collections.defaultdict(list)
    n_box = 0

    for p in raws:
        if n_box >= NBOX:
            break
        box = os.path.basename(p)[:-4]
        try:
            with np.load(p, allow_pickle=True) as d:
                if not {"inst", "cls"} <= set(d.files):
                    continue
                meta = json.loads(str(d["meta"]))
                inst, cls = d["inst"], d["cls"]
        except Exception:
            continue
        if inst.ndim != 3 or inst.shape != cls.shape:
            continue
        names = {int(k): v.get("organelle") for k, v in meta.get("instances", {}).items()}
        if not names:
            continue
        Z = inst.shape[0]
        zs = np.linspace(4, int(Z * 0.85) - 1, NPLANE).astype(int)
        zs = sorted(set(int(z) for z in zs if 0 <= z < Z - 2))
        if not zs:
            continue
        n_box += 1
        if n_box % 10 == 0:
            print(f"  {n_box}/{NBOX} boxes", flush=True)

        for z in zs:
            si, sc = inst[z:], cls[z:]
            ei, ec = si[0], sc[0]
            fg = ei > 0
            if fg.sum() < 200:
                continue
            r_inst = runs_below(si == ei[None, :, :])
            r_cls = runs_below(sc == ec[None, :, :])
            r_any = runs_below(si > 0)

            ys, xs = np.nonzero(fg)
            if ys.size > 60000:
                sel = np.random.default_rng(z).choice(ys.size, 60000, replace=False)
                ys, xs = ys[sel], xs[sel]
            for y, x in zip(ys.tolist(), xs.tolist()):
                o = names.get(int(ei[y, x]))
                if not o:
                    continue
                a, b, c = int(r_inst[y, x]), int(r_cls[y, x]), int(r_any[y, x])
                if a <= 0:
                    continue
                total[o] += 1
                if b > a:
                    split[o] += 1
                ratio[o].append(b / a)
                union_ratio[o].append(c / a)

    print(f"\nboxes read {n_box}")
    print(f"\n{'class':<11s} {'pixels':>9s} {'id ended first':>15s} "
          f"{'class/inst med':>15s} {'p90':>7s} {'union/inst med':>15s}")
    out = {}
    for o in sorted(total, key=lambda k: -total[k]):
        if total[o] < 2000:
            continue
        r = np.array(ratio[o]); u = np.array(union_ratio[o])
        frac = 100 * split[o] / total[o]
        out[o] = dict(pixels=total[o], split_pct=round(frac, 2),
                      ratio_median=round(float(np.median(r)), 4),
                      ratio_p90=round(float(np.percentile(r, 90)), 4),
                      union_median=round(float(np.median(u)), 4))
        print(f"{o:<11s} {total[o]:>9,d} {frac:>14.2f}% {np.median(r):>15.3f} "
              f"{np.percentile(r, 90):>7.2f} {np.median(u):>15.3f}")

    json.dump(out, open(f"{ROOT}/cache/depth_by_class.json", "w"), indent=1)
    print(f"\n-> cache/depth_by_class.json")
    print("\nreading")
    print("  a class-run over instance-run ratio of 1.000 means the grouping never cut a "
          "ray short")
    print("  above 1.000 means one organelle carries two identifiers stacked along the "
          "ray, and the stored depth stops at the seam")


if __name__ == "__main__":
    main()
