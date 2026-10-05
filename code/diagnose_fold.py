import collections, json, os, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = "cache/inst_bf_exact_full.jsonl"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "jrc_choroid-plexus-2"
SPHERE_LOG = np.log(4.0 / 3.0 / np.sqrt(np.pi))


def fit_apply(Atr, Vtr, Otr, Ate, Ote):
    out = np.zeros(Ate.size); gm = float(Vtr.mean())
    for o in np.unique(Ote):
        te = Ote == o; tr = Otr == o
        if tr.sum() < 30:
            out[te] = gm; continue
        x, y = Atr[tr], Vtr[tr]
        b = float(((x - x.mean()) * (y - y.mean())).sum() /
                  max(((x - x.mean()) ** 2).sum(), 1e-12))
        out[te] = (y.mean() - b * x.mean()) + b * Ate[te]
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, TABLE)) ]
    rows = [r for r in rows if r.get("unbiased")]
    V = np.log(np.array([r["volume_nm3"] for r in rows]))
    A = np.log(np.maximum(np.array([r["area_nm2"] for r in rows]), 1.0))
    O = np.array([r["organelle"] for r in rows])
    D = np.array([r["dataset"] for r in rows])
    L = np.array([r["level"] for r in rows])
    T = np.array([r["tier"] for r in rows])
    K = np.array([f"{r['box']}|{r['instance']}" for r in rows])

    te = D == TARGET; tr = ~te
    print(f"target {TARGET}")
    print(f"  rows {te.sum():,} of {len(rows):,}")

    def share(mask, arr):
        c = collections.Counter(arr[mask].tolist()); n = sum(c.values())
        return {k: 100 * v / n for k, v in c.items()}, c

    st, ct = share(te, O); sr, cr = share(tr, O)
    print(f"\n{'class':<12s} {'this fold':>10s} {'elsewhere':>10s} {'x':>7s}")
    for k in sorted(set(st) | set(sr), key=lambda k: -st.get(k, 0)):
        a, b = st.get(k, 0.0), sr.get(k, 0.0)
        print(f"  {k:<10s} {a:>9.1f}% {b:>9.1f}% {a / max(b, 1e-9):>7.1f}")

    for name, arr in (("level", L), ("tier", T)):
        s1, _ = share(te, arr); s2, _ = share(tr, arr)
        print(f"\n{name:<12s} {'this fold':>10s} {'elsewhere':>10s}")
        for k in sorted(set(s1) | set(s2), key=lambda k: -s1.get(k, 0)):
            print(f"  {k:<10s} {s1.get(k, 0):>9.1f}% {s2.get(k, 0):>9.1f}%")

    print(f"\n{'class':<10s} {'n':>7s} {'area p50 here':>14s} {'p50 elsewhere':>14s} "
          f"{'vol p50 here':>13s} {'p50 elsewhere':>14s} {'in train range':>15s}")
    for o in sorted(set(O[te].tolist()), key=lambda k: -ct[k]):
        if ct[o] < 200:
            continue
        m1, m2 = te & (O == o), tr & (O == o)
        if m2.sum() < 30:
            print(f"  {o:<8s} {ct[o]:>7,d}   class unseen in the training folds")
            continue
        lo, hi = np.percentile(A[m2], [1, 99])
        inside = 100 * np.mean((A[m1] >= lo) & (A[m1] <= hi))
        print(f"  {o:<8s} {ct[o]:>7,d} {np.exp(np.median(A[m1]))/1e6:>13.3f} "
              f"{np.exp(np.median(A[m2]))/1e6:>13.3f} "
              f"{np.exp(np.median(V[m1]))/1e9:>12.4f} "
              f"{np.exp(np.median(V[m2]))/1e9:>13.4f} {inside:>14.1f}%")
    print("  area in square micrometres, volume in cubic micrometres")

    pred = fit_apply(A[tr], V[tr], O[tr], A[te], O[te])
    p, t = np.exp(pred), np.exp(V[te])
    err = np.abs(p - t) / t
    bias = p / t
    Ote, Kte = O[te], K[te]

    d = collections.defaultdict(lambda: ([], [], None))
    for pp, tt, kk, oo in zip(pred, V[te], Kte, Ote):
        d[kk][0].append(pp); d[kk][1].append(tt)
    inst_o, inst_e, inst_b = [], [], []
    for kk, (ps, ts) in ((k, (v[0], v[1])) for k, v in d.items()):
        P, T2 = np.exp(np.mean(ps)), np.exp(ts[0])
        inst_e.append(abs(P - T2) / T2); inst_b.append(P / T2)
    inst_e = np.array(inst_e); inst_b = np.array(inst_b)
    print(f"\ninstance unit over this fold: median error {np.median(inst_e):.4f}, "
          f"median bias {np.median(inst_b):.3f}, within 2x {np.mean(np.maximum(inst_b, 1/inst_b) <= 2):.1%}")

    print(f"\nerror by class, cut-face unit, and how much of the fold each class is")
    print(f"  {'class':<10s} {'n':>8s} {'share':>7s} {'median err':>11s} "
          f"{'median bias':>12s} {'contribution':>13s}")
    tot = te.sum()
    contrib = []
    for o in sorted(set(Ote.tolist()), key=lambda k: -ct[k]):
        m = Ote == o
        if m.sum() < 100:
            continue
        me, mb = float(np.median(err[m])), float(np.median(bias[m]))
        sh = 100 * m.sum() / tot
        contrib.append((sh * me, o))
        print(f"  {o:<10s} {m.sum():>8,d} {sh:>6.1f}% {me:>11.4f} {mb:>12.3f} "
              f"{sh * me / 100:>12.4f}")
    contrib.sort(reverse=True)
    print(f"  largest contributors: " + ", ".join(f"{o} {c/100:.3f}" for c, o in contrib[:3]))

    print(f"\nrecomputing the fold with one class removed at a time, instance unit")
    for o in [o for _, o in contrib[:5]]:
        keep = Ote != o
        d2 = collections.defaultdict(lambda: ([], []))
        for pp, tt, kk in zip(pred[keep], V[te][keep], Kte[keep]):
            d2[kk][0].append(pp); d2[kk][1].append(tt)
        e2 = np.array([abs(np.exp(np.mean(v[0])) - np.exp(v[1][0])) / np.exp(v[1][0])
                       for v in d2.values()])
        if e2.size:
            print(f"  without {o:<10s} n {e2.size:>7,d}   median {np.median(e2):.4f}")


if __name__ == "__main__":
    main()
