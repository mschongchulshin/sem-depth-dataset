import collections, json, os, random, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = sys.argv[1] if len(sys.argv) > 1 else "cache/inst_bf_exact_full.jsonl"
NBOX = int(sys.argv[2]) if len(sys.argv) > 2 else 800


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, TABLE))]
    byfile = collections.defaultdict(dict)
    byboxfiles = collections.defaultdict(set)
    for r in rows:
        byfile[r["file"]][r["instance"]] = r
        byboxfiles[r["box"]].add(r["file"])

    boxes = [b for b, fl in byboxfiles.items() if len(fl) >= 6]
    random.seed(0); random.shuffle(boxes)
    boxes = boxes[:NBOX]
    print(f"boxes {len(boxes):,}", flush=True)

    box_area = collections.defaultdict(collections.Counter)
    box_vol = collections.defaultdict(collections.Counter)
    for bi, b in enumerate(boxes):
        if bi % 100 == 0 and bi:
            print(f"  {bi}/{len(boxes)}", flush=True)
        seen = set()
        for f in sorted(byboxfiles[b]):
            p = os.path.join(ROOT, f)
            if not os.path.exists(p):
                continue
            with np.load(p) as z:
                ids2d = z["inst_face"]
            recs = byfile[f]
            vals, counts = np.unique(ids2d, return_counts=True)
            for i, n in zip(vals.tolist(), counts.tolist()):
                if i == 0 or i not in recs:
                    continue
                r = recs[i]
                box_area[b][r["organelle"]] += r["px_nm"] ** 2 * n
                if i not in seen:
                    seen.add(i)
                    box_vol[b][r["organelle"]] += r["volume_nm3"]

    classes = sorted({o for b in boxes for o in box_area[b]})

    def pooled(drop=()):
        A = collections.Counter(); V = collections.Counter()
        for b in boxes:
            for o, v in box_area[b].items():
                if o not in drop: A[o] += v
            for o, v in box_vol[b].items():
                if o not in drop: V[o] += v
        ta, tv = sum(A.values()), sum(V.values())
        out = {}
        for o in classes:
            if o in drop or not A[o] or not V[o]: continue
            af, vf = A[o] / ta, V[o] / tv
            out[o] = (af, vf, af / vf)
        return out

    def within():
        per = collections.defaultdict(list)
        for b in boxes:
            ta, tv = sum(box_area[b].values()), sum(box_vol[b].values())
            if ta <= 0 or tv <= 0: continue
            if len([o for o in box_area[b] if box_area[b][o] > 0]) < 2: continue
            for o in box_area[b]:
                if box_area[b][o] <= 0 or box_vol[b].get(o, 0) <= 0: continue
                per[o].append((box_area[b][o] / ta) / (box_vol[b][o] / tv))
        return {o: (len(v), float(np.median(v))) for o, v in per.items() if len(v) >= 20}

    def show(name, d):
        print(f"\n--- {name} ---")
        print(f"    {'class':<11s} {'area frac':>10s} {'vol frac':>10s} {'ratio':>8s}")
        rs = []
        for o, (af, vf, rr) in sorted(d.items(), key=lambda kv: -kv[1][0]):
            print(f"    {o:<11s} {af*100:>9.2f}% {vf*100:>9.2f}% {rr:>8.3f}")
            rs.append(abs(rr - 1))
        if rs:
            print(f"    median relative disagreement {np.median(rs)*100:.1f}%")

    show("pooled over all boxes", pooled())
    show("pooled, nucleus removed", pooled(drop=("nucleus",)))

    w = within()
    print("\n--- Delesse inside each box, median over boxes ---")
    print(f"    {'class':<11s} {'n boxes':>9s} {'median ratio':>13s}")
    rs = []
    for o, (n, m) in sorted(w.items(), key=lambda kv: -kv[1][0]):
        print(f"    {o:<11s} {n:>9,d} {m:>13.3f}")
        rs.append(abs(m - 1))
    if rs:
        print(f"    median relative disagreement {np.median(rs)*100:.1f}%")

    fr = []
    for b in boxes:
        ta = sum(box_area[b].values())
        if ta > 0:
            fr.append(max(box_area[b].values()) / ta)
    fr.sort()
    print(f"\nhow far one class dominates a box, as its share of that box's labelled area")
    print(f"    median {fr[len(fr)//2]:.3f}   p90 {fr[int(len(fr)*.9)]:.3f}   "
          f"boxes above 0.95: {sum(1 for f in fr if f > 0.95)}/{len(fr)}")


if __name__ == "__main__":
    main()
