import argparse, collections, json, os
import numpy as np

SPHERE_LOG = np.log(4.0 / 3.0 / np.sqrt(np.pi))


def wasserstein(a, b):
    q = np.linspace(0.005, 0.995, 199)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def saltykov(areas_nm2, nbins=12):
    d = 2.0 * np.sqrt(np.asarray(areas_nm2, dtype=np.float64) / np.pi)
    d = d[d > 0]
    if d.size < 50:
        return None
    dmax = d.max()
    edges = np.linspace(0, dmax, nbins + 1)
    counts = np.histogram(d, bins=edges)[0].astype(np.float64)
    mid = 0.5 * (edges[1:] + edges[:-1])
    P = np.zeros((nbins, nbins))
    for i in range(nbins):
        D = edges[i + 1]
        for j in range(i + 1):
            lo, hi = min(edges[j], D), min(edges[j + 1], D)
            P[i, j] = (np.sqrt(max(D * D - lo * lo, 0.0)) -
                       np.sqrt(max(D * D - hi * hi, 0.0))) / D
    n = np.zeros(nbins)
    left = counts.copy()
    for i in range(nbins - 1, -1, -1):
        if P[i, i] <= 1e-12:
            continue
        n[i] = max(left[i] / P[i, i], 0.0)
        left -= n[i] * P[i]
    n = np.maximum(n, 0.0)
    if n.sum() <= 0:
        return None
    rng = np.random.default_rng(0)
    pick = rng.choice(nbins, size=len(d), p=n / n.sum())
    D = mid[pick]
    return np.log(np.pi / 6.0 * D ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", default="cache/ivol_v2_preds.json")
    ap.add_argument("--table", default="cache/inst_bf_organelle.jsonl")
    ap.add_argument("--min-n", type=int, default=200)
    a = ap.parse_args()

    res = [r for r in json.load(open(os.path.join(a.root, a.run))) if r and r.get("preds")]
    if not res:
        print("no saved predictions in that run"); return
    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    rows = [r for r in rows if r.get("unbiased")]
    area_by = collections.defaultdict(list)
    for r in rows:
        area_by[(r["dataset"], r["organelle"])].append(r["area_nm2"])

    print(f"{len(res)} folds with saved predictions\n")
    print(f"{'fold / class':<34s} {'n':>7s} {'W(model)':>9s} {'W(stereo)':>10s} "
          f"{'W(saltykov)':>12s} {'bias m':>8s} {'bias s':>8s}")
    agg = collections.defaultdict(list)
    for r in res:
        p = r["preds"]
        P, T, B = (np.array(p[k], dtype=np.float64) for k in ("pred", "true", "base"))
        O = np.array(p["org"])
        for org in sorted(set(O)):
            s = O == org
            if s.sum() < a.min_n:
                continue
            ar = area_by.get((r["held"], org))
            sk = saltykov(ar) if ar else None
            wm, ws = wasserstein(P[s], T[s]), wasserstein(B[s], T[s])
            wk = wasserstein(sk, T[s]) if sk is not None else float("nan")
            bm = float(np.exp(P[s]).mean() / np.exp(T[s]).mean())
            bs = float(np.exp(B[s]).mean() / np.exp(T[s]).mean())
            agg["model"].append(wm); agg["stereo"].append(ws)
            if sk is not None:
                agg["saltykov"].append(wk)
            agg["bias_model"].append(bm); agg["bias_stereo"].append(bs)
            print(f"{r['held'][:22] + ' / ' + org:<34s} {int(s.sum()):>7,d} {wm:>9.3f} "
                  f"{ws:>10.3f} {wk:>12.3f} {bm:>8.3f} {bs:>8.3f}")
    print()
    print("Wasserstein distance in log volume, lower is better. bias is mean(pred)/mean(true),")
    print("1.0 is unbiased.")
    for k in ("model", "stereo", "saltykov"):
        if agg[k]:
            v = np.array(agg[k])
            print(f"  W {k:<10s} mean {v.mean():.3f}   median {np.median(v):.3f}   "
                  f"n={len(v)}")
    for k in ("bias_model", "bias_stereo"):
        v = np.array(agg[k])
        print(f"  {k:<12s} median {np.median(v):.3f}   "
              f"quartiles {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}")
    if agg["model"] and agg["stereo"]:
        w = sum(m < s for m, s in zip(agg["model"], agg["stereo"]))
        print(f"\n  model closer than sphere formula on {w}/{len(agg['model'])} groups")
    if agg["model"] and agg["saltykov"]:
        n = min(len(agg["model"]), len(agg["saltykov"]))
        w = sum(m < s for m, s in zip(agg["model"][:n], agg["saltykov"][:n]))
        print(f"  model closer than Saltykov unfolding on {w}/{n} groups")


if __name__ == "__main__":
    main()
