import collections
import numpy as np

RNG_SIZE = 160


def build(seed, n_big, n_small, size=RNG_SIZE, r_big=22, r_small=7):
    rng = np.random.default_rng(seed)
    vol = np.zeros((size,) * 3, np.int32)
    meta, nid = {}, 1
    zz, yy, xx = np.ogrid[:size, :size, :size]
    for r, n, cls in ((r_big, n_big, "big"), (r_small, n_small, "small")):
        for _ in range(n):
            for _try in range(400):
                c = rng.integers(r + 1, size - r - 1, 3)
                m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
                if (vol[m] != 0).any():
                    continue
                vol[m] = nid; meta[nid] = cls; nid += 1; break
    return vol, meta


def measure(vol, meta, nplanes):
    Z = vol.shape[0]
    zs = np.linspace(4, Z - 5, nplanes).astype(int)
    area = collections.Counter()
    seen = set()
    volu = collections.Counter()
    cnt = np.bincount(vol.ravel())
    for z in zs:
        f = vol[z]
        ids, ns = np.unique(f, return_counts=True)
        for i, n in zip(ids.tolist(), ns.tolist()):
            if i == 0:
                continue
            area[meta[i]] += n
            if i not in seen:
                seen.add(i)
                volu[meta[i]] += int(cnt[i])
    return area, volu


def main():
    specs = [
        dict(seed=1, n_big=6,  n_small=0,   nplanes=12),
        dict(seed=2, n_big=0,  n_small=260, nplanes=6),
        dict(seed=3, n_big=4,  n_small=120, nplanes=9),
        dict(seed=4, n_big=2,  n_small=200, nplanes=7),
        dict(seed=5, n_big=7,  n_small=40,  nplanes=11),
    ]
    boxes = []
    print("per-box truth and per-box Delesse")
    print(f"  {'box':>4s} {'planes':>7s} {'class':>7s} {'area frac':>10s} {'vol frac':>9s} {'ratio':>8s}")
    for bi, sp in enumerate(specs):
        vol, meta = build(sp["seed"], sp["n_big"], sp["n_small"])
        area, volu = measure(vol, meta, sp["nplanes"])
        ta, tv = sum(area.values()), sum(volu.values())
        boxes.append((area, volu, sp["nplanes"]))
        for c in sorted(area):
            af, vf = area[c] / ta, volu[c] / tv
            print(f"  {bi:>4d} {sp['nplanes']:>7d} {c:>7s} {af*100:>9.2f}% {vf*100:>8.2f}% "
                  f"{af/vf:>8.3f}")

    print("\nA. pooled across boxes, what the corpus report does")
    A, V = collections.Counter(), collections.Counter()
    for area, volu, _ in boxes:
        A.update(area); V.update(volu)
    ta, tv = sum(A.values()), sum(V.values())
    pooled = {}
    for c in sorted(A):
        af, vf = A[c] / ta, V[c] / tv
        pooled[c] = af / vf
        print(f"    {c:>7s}  area {af*100:>6.2f}%  vol {vf*100:>6.2f}%  ratio {af/vf:>7.3f}")

    print("\nB. Delesse inside each box, then the median across boxes")
    per = collections.defaultdict(list)
    for area, volu, _ in boxes:
        ta, tv = sum(area.values()), sum(volu.values())
        if len(area) < 2:
            continue
        for c in area:
            per[c].append((area[c] / ta) / (volu[c] / tv))
    within = {}
    for c in sorted(per):
        within[c] = float(np.median(per[c]))
        print(f"    {c:>7s}  n {len(per[c])}  median ratio {within[c]:>7.3f}")

    print("\nC. pooled after equalising planes per box (every box cut 9 times)")
    A2, V2 = collections.Counter(), collections.Counter()
    for sp in specs:
        vol, meta = build(sp["seed"], sp["n_big"], sp["n_small"])
        area, volu = measure(vol, meta, 9)
        A2.update(area); V2.update(volu)
    ta2, tv2 = sum(A2.values()), sum(V2.values())
    for c in sorted(A2):
        af, vf = A2[c] / ta2, V2[c] / tv2
        print(f"    {c:>7s}  area {af*100:>6.2f}%  vol {vf*100:>6.2f}%  ratio {af/vf:>7.3f}")

    print("\nreading")
    pe = np.median([abs(v - 1) for v in pooled.values()]) * 100
    we = np.median([abs(v - 1) for v in within.values()]) * 100
    print(f"  pooled      median error {pe:.1f}%")
    print(f"  within-box  median error {we:.1f}%")
    print(f"  -> {'per-box aggregation is the correct estimator' if we < pe else 'pooling is not the problem'}")


if __name__ == "__main__":
    main()
