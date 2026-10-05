import collections, json, os, random, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = "cache/inst_bf_exact_full.jsonl"
NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 800
OLD11 = {"jrc_hela-1", "jrc_hela-2", "jrc_hela-3", "jrc_hela-bfa", "jrc_jurkat-1",
         "jrc_macrophage-2", "jrc_cos7-11", "jrc_mus-liver", "jrc_choroid-plexus-2",
         "aic_desmosome-2", "aic_desmosome-3"}


def pooled(sel, box_area, box_vol, drop=("nucleus",)):
    A, V = collections.Counter(), collections.Counter()
    for b in sel:
        for o, v in box_area[b].items():
            if o not in drop: A[o] += v
        for o, v in box_vol[b].items():
            if o not in drop: V[o] += v
    ta, tv = sum(A.values()), sum(V.values())
    if ta <= 0 or tv <= 0: return {}, float("nan")
    out = {}
    for o in A:
        if A[o] <= 0 or V.get(o, 0) <= 0: continue
        out[o] = (A[o] / ta) / (V[o] / tv)
    if not out: return {}, float("nan")
    return out, float(np.median([abs(v - 1) for v in out.values()])) * 100


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, TABLE))]
    byfile = collections.defaultdict(dict)
    byboxfiles = collections.defaultdict(set)
    meta = {}
    for r in rows:
        byfile[r["file"]][r["instance"]] = r
        byboxfiles[r["box"]].add(r["file"])
        meta[r["box"]] = (r["dataset"], r["tier"], r["level"])

    boxes = [b for b, fl in byboxfiles.items() if len(fl) >= 6]
    random.seed(0); random.shuffle(boxes)
    boxes = boxes[:NBOX]
    print(f"boxes {len(boxes):,}", flush=True)

    box_area = collections.defaultdict(collections.Counter)
    box_vol = collections.defaultdict(collections.Counter)
    for bi, b in enumerate(boxes):
        if bi % 200 == 0 and bi: print(f"  {bi}/{len(boxes)}", flush=True)
        seen = set()
        for f in sorted(byboxfiles[b]):
            p = os.path.join(ROOT, f)
            if not os.path.exists(p): continue
            with np.load(p) as z:
                ids2d = z["inst_face"]
            recs = byfile[f]
            vals, counts = np.unique(ids2d, return_counts=True)
            for i, n in zip(vals.tolist(), counts.tolist()):
                if i == 0 or i not in recs: continue
                r = recs[i]
                box_area[b][r["organelle"]] += r["px_nm"] ** 2 * n
                if i not in seen:
                    seen.add(i); box_vol[b][r["organelle"]] += r["volume_nm3"]

    print("\n=== all boxes, nucleus dropped ===")
    d, m = pooled(boxes, box_area, box_vol)
    for o, v in sorted(d.items(), key=lambda kv: -kv[1]): print(f"    {o:<11s} {v:>7.3f}")
    print(f"    median disagreement {m:.1f}%")

    def stratum(name, groups, key):
        print(f"\n=== by {name} ===")
        print(f"    {'stratum':<24s} {'boxes':>6s} {'median':>8s}   worst classes")
        for g in groups:
            sel = [b for b in boxes if key(meta[b]) == g]
            if len(sel) < 15: continue
            d, m = pooled(sel, box_area, box_vol)
            if not d: continue
            worst = sorted(d.items(), key=lambda kv: -abs(kv[1] - 1))[:3]
            ws = "  ".join(f"{o} {v:.2f}" for o, v in worst)
            print(f"    {str(g):<24s} {len(sel):>6d} {m:>7.1f}%   {ws}")

    stratum("source volume group",
            ["old 11", "new 7"],
            lambda mm: "old 11" if mm[0] in OLD11 else "new 7")
    stratum("tier", sorted({mm[1] for mm in meta.values()}), lambda mm: mm[1])
    stratum("level", sorted({mm[2] for mm in meta.values()}), lambda mm: mm[2])

    nf = sorted(len(byboxfiles[b]) for b in boxes)
    print(f"\nplanes per box  median {nf[len(nf)//2]}  min {nf[0]}  max {nf[-1]}")


if __name__ == "__main__":
    main()
