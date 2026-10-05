import collections
import numpy as np
from scipy import ndimage


def build(seed=0, size=300, groups=((30, 8, "big"), (8, 120, "small"))):
    rng = np.random.default_rng(seed)
    vol = np.zeros((size,) * 3, np.int32)
    meta, nid = {}, 1
    zz, yy, xx = np.ogrid[:size, :size, :size]
    for r, n, cls in groups:
        for _ in range(n):
            for _try in range(300):
                c = rng.integers(r + 1, size - r - 1, 3)
                m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
                if (vol[m] != 0).any():
                    continue
                vol[m] = nid; meta[nid] = (cls, r); nid += 1; break
    return vol, meta


def truth(vol, meta):
    cnt = np.bincount(vol.ravel())
    v = collections.Counter()
    for i, (cls, _) in meta.items():
        v[cls] += int(cnt[i])
    return v, cnt


def check(name, got, want, tol=0.05):
    ok = abs(got - want) <= tol * max(abs(want), 1e-9)
    print(f"   {name:<44s} got {got:>8.3f}   expect {want:>8.3f}   "
          f"{'pass' if ok else 'FAIL'}")
    return ok


def main():
    vol, meta = build()
    tv, cnt = truth(vol, meta)
    T = sum(tv.values())
    Z = vol.shape[0]
    zs = np.linspace(5, Z - 6, 60).astype(int)
    print(f"synthetic volume {vol.shape}, {len(meta)} spheres, "
          f"true volume fractions " + ", ".join(f"{k} {v / T:.4f}" for k, v in tv.items()))

    print("\n1. Delesse, area fraction against volume fraction")
    area = collections.Counter(); per = collections.defaultdict(list)
    seen, sv = set(), collections.Counter()
    for z in zs:
        sl = vol[z]
        ids, cs = np.unique(sl, return_counts=True)
        f = collections.Counter()
        for i, n in zip(ids.tolist(), cs.tolist()):
            if i == 0:
                continue
            cls = meta[i][0]
            f[cls] += n; area[cls] += n
            if i not in seen:
                seen.add(i); sv[cls] += int(cnt[i])
        t = sum(f.values())
        if t:
            for k in tv:
                per[k].append(f.get(k, 0) / t)
    A, S = sum(area.values()), sum(sv.values())
    for cls in sorted(tv):
        want = tv[cls] / T
        check(f"{cls}: ratio of sums", area[cls] / A, want)
        check(f"{cls}: mean of per-plane ratios", float(np.mean(per[cls])), want)
    print("   the ratio of sums is the theorem; the mean of ratios is not")

    print("\n2. instance grouping, ids over connected components")
    for cls in sorted(tv):
        ids = [i for i, (c, _) in meta.items() if c == cls]
        mask = np.isin(vol, ids)
        _, ncomp = ndimage.label(mask)
        check(f"{cls}: ids / components", len(ids) / max(ncomp, 1), 1.0, tol=0.02)
    print("   each sphere is one id and one component, so the answer is exactly 1")

    print("\n3. below-cut volume as a fraction of the whole object")
    frac = []
    for z in zs:
        face = np.unique(vol[z])
        sub = vol[z:]
        c2 = np.bincount(sub.ravel(), minlength=len(cnt))
        for i in face:
            if i == 0:
                continue
            frac.append(c2[i] / cnt[i])
    check("median below / total", float(np.median(frac)), 0.5, tol=0.10)
    print("   a plane cutting a convex body leaves half of it below, in the median")

    print("\n4. mean depth below, per profile, over the diameter")
    got = collections.defaultdict(list)
    for z in zs:
        sl = vol[z]
        sub = vol[z:]
        for i in np.unique(sl):
            if i == 0:
                continue
            m2 = sl == i
            if m2.sum() < 25:
                continue
            same = sub == i
            run = np.where(same.all(axis=0), same.shape[0], np.argmin(same, axis=0))
            r = meta[i][1]
            got[meta[i][0]].append(float(run[m2].mean()) / (2 * r))
    for cls in sorted(got):
        check(f"{cls}: mean depth / diameter", float(np.median(got[cls])), 0.326, tol=0.15)
    print("   0.326 is the value a uniform random plane through a sphere gives")


if __name__ == "__main__":
    main()
