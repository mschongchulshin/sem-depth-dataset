import argparse, collections, glob, json, os, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--boxes", type=int, default=120)
    ap.add_argument("--table", default="cache/inst_bf_exact.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    byfile = collections.defaultdict(dict)
    byboxfiles = collections.defaultdict(set)
    for r in rows:
        byfile[r["file"]][r["instance"]] = r
        byboxfiles[r["box"]].add(r["file"])

    boxes = [b for b, fl in byboxfiles.items() if len(fl) >= 6]
    random.seed(0); random.shuffle(boxes)
    boxes = boxes[:a.boxes]
    print(f"boxes {len(boxes):,} with at least 6 faces each")

    classes = sorted({r["organelle"] for r in rows})
    total_area = collections.Counter()
    box_area = collections.defaultdict(lambda: collections.Counter())
    box_vol = collections.defaultdict(lambda: collections.Counter())
    box_nface = collections.Counter()
    n_faces = 0

    for b in boxes:
        seen = set()
        for f in sorted(byboxfiles[b]):
            p = os.path.join(a.root, f)
            if not os.path.exists(p):
                continue
            with np.load(p) as z:
                ids2d = z["inst_face"]
            recs = byfile[f]
            vals, counts = np.unique(ids2d, return_counts=True)
            face = collections.Counter()
            for i, n in zip(vals.tolist(), counts.tolist()):
                if i == 0 or i not in recs:
                    continue
                r = recs[i]
                face[r["organelle"]] += r["px_nm"] ** 2 * n
                if i not in seen:
                    seen.add(i); box_vol[b][r["organelle"]] += r["volume_nm3"]
            tot = sum(face.values())
            if tot <= 0:
                continue
            n_faces += 1; box_nface[b] += 1
            for o in classes:
                total_area[o] += face.get(o, 0.0)
                box_area[b][o] += face.get(o, 0.0)

    vol = collections.Counter()
    for d in box_vol.values():
        for o, v in d.items():
            vol[o] += v
    V = sum(vol.values())
    print(f"faces {n_faces:,}   instances contributing volume "
          f"{sum(len(s) for s in box_vol.values())}")

    A = sum(total_area.values())
    print("\nDelesse, summed profile area against summed volume")
    print(f"    {'class':<11s} {'area frac':>10s} {'volume frac':>12s} {'ratio':>8s}")
    errs = []
    for o in sorted(vol, key=lambda k: -vol[k]):
        vf = vol[o] / V
        if vf < 1e-4:
            continue
        af = total_area[o] / A
        errs.append(abs(af - vf) / vf)
        print(f"    {o:<11s} {af * 100:9.2f}% {vf * 100:11.2f}% {af / vf:8.3f}")
    print(f"  median relative disagreement {np.median(errs) * 100:.1f}%")

    print("\nCavalieri, sum(area)/volume per box. Spread matters, not the level.")
    byo = collections.defaultdict(list)
    for b in box_area:
        if box_nface[b] < 6:
            continue
        for o, ar in box_area[b].items():
            v = box_vol[b].get(o, 0.0)
            if v > 0 and ar > 0:
                byo[o].append(ar / v)
    print(f"    {'class':<11s} {'n boxes':>8s} {'median':>12s} {'IQR/median':>11s}")
    for o in sorted(byo, key=lambda k: -len(byo[k]))[:8]:
        v = np.array(byo[o])
        if len(v) < 5:
            continue
        iqr = (np.percentile(v, 75) - np.percentile(v, 25)) / np.median(v)
        print(f"    {o:<11s} {len(v):>8d} {np.median(v):>12.3e} {iqr:>11.2f}")


if __name__ == "__main__":
    main()
