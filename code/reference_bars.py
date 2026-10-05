import collections, json, os, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = sys.argv[1] if len(sys.argv) > 1 else "cache/inst_bf_exact_full.jsonl"
SPHERE_LOG = np.log(4.0 / 3.0 / np.sqrt(np.pi))
MIN_TEST = 300


def rel_stats(pred_log, true_log):
    p, t = np.exp(pred_log), np.exp(true_log)
    e = np.abs(p - t) / t
    fold = np.maximum(p / t, t / p)
    return dict(n=int(e.size),
                median=float(np.median(e)),
                iqr=[float(np.percentile(e, 25)), float(np.percentile(e, 75))],
                within_2x=float(np.mean(fold <= 2.0)))


def group_by_instance(pred_log, true_log, key):
    d = collections.defaultdict(lambda: ([], []))
    for p, t, k in zip(pred_log, true_log, key):
        d[k][0].append(p); d[k][1].append(t)
    P = np.array([np.mean(v[0]) for v in d.values()])
    T = np.array([v[1][0] for v in d.values()])
    return P, T


def fit_apply(Atr, Vtr, Otr, Ate, Ote, kind):
    out = np.zeros(Ate.size)
    gmean = float(Vtr.mean())
    for o in np.unique(Ote):
        te = Ote == o
        tr = Otr == o
        if kind == "sphere":
            out[te] = 1.5 * Ate[te] + SPHERE_LOG
            continue
        if tr.sum() < 30:
            out[te] = gmean
            continue
        if kind == "constant":
            out[te] = float(Vtr[tr].mean())
        elif kind == "calibrated":
            x, y = Atr[tr], Vtr[tr]
            b = float(((x - x.mean()) * (y - y.mean())).sum() /
                      max(((x - x.mean()) ** 2).sum(), 1e-12))
            out[te] = (y.mean() - b * x.mean()) + b * Ate[te]
    return out


def oracle(Ate, Vte, Ote, nbin=10):
    out = np.zeros(Ate.size)
    for o in np.unique(Ote):
        sel = Ote == o
        a, v = Ate[sel], Vte[sel]
        if a.size < nbin * 4:
            out[sel] = v.mean(); continue
        edges = np.quantile(a, np.linspace(0, 1, nbin + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        idx = np.clip(np.digitize(a, edges[1:-1]), 0, nbin - 1)
        pred = np.zeros(a.size)
        for b in range(nbin):
            m = idx == b
            pred[m] = v[m].mean() if m.any() else v.mean()
        out[sel] = pred
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, TABLE))]
    rows = [r for r in rows if r.get("unbiased")]
    print(f"table {TABLE}")
    print(f"unbiased rows {len(rows):,}", flush=True)

    V = np.log(np.array([r["volume_nm3"] for r in rows]))
    A = np.log(np.maximum(np.array([r["area_nm2"] for r in rows]), 1.0))
    O = np.array([r["organelle"] for r in rows])
    D = np.array([r["dataset"] for r in rows])
    K = np.array([f"{r['box']}|{r['instance']}" for r in rows])

    ARMS = ["constant", "sphere", "calibrated"]
    res = {p: {u: collections.defaultdict(list) for u in ("face", "instance")}
           for p in ("held-out", "random")}
    per_fold = collections.defaultdict(dict)

    folds = [d for d in sorted(set(D)) if (D == d).sum() >= MIN_TEST]
    print(f"\nheld-out folds {len(folds)} of {len(set(D))} source volumes "
          f"(threshold {MIN_TEST} rows)")
    for held in folds:
        te = D == held; tr = ~te
        preds = {k: fit_apply(A[tr], V[tr], O[tr], A[te], O[te], k) for k in ARMS}
        preds["oracle"] = oracle(A[te], V[te], O[te])
        for k, p in preds.items():
            s = rel_stats(p, V[te])
            res["held-out"]["face"][k].append(s["median"])
            P, T = group_by_instance(p, V[te], K[te])
            si = rel_stats(P, T)
            res["held-out"]["instance"][k].append(si["median"])
            per_fold[held][k] = (si["median"], si["n"], si["within_2x"])

    rng = np.random.default_rng(0)
    files = np.array([r["file"] for r in rows])
    uf = np.array(sorted(set(files.tolist())))
    for rep in range(len(folds)):
        rng2 = np.random.default_rng(100 + rep)
        pick = set(rng2.choice(uf, size=max(1, len(uf) // len(folds)),
                               replace=False).tolist())
        te = np.array([f in pick for f in files]); tr = ~te
        if te.sum() < MIN_TEST:
            continue
        preds = {k: fit_apply(A[tr], V[tr], O[tr], A[te], O[te], k) for k in ARMS}
        preds["oracle"] = oracle(A[te], V[te], O[te])
        for k, p in preds.items():
            res["random"]["face"][k].append(rel_stats(p, V[te])["median"])
            P, T = group_by_instance(p, V[te], K[te])
            res["random"]["instance"][k].append(rel_stats(P, T)["median"])

    print("\n" + "=" * 74)
    print("median relative error, mean over folds (lower is better)")
    print("=" * 74)
    print(f"  {'arm':<12s} {'held-out face':>14s} {'held-out inst':>14s} "
          f"{'random face':>13s} {'random inst':>13s}")
    summary = {}
    for k in ARMS + ["oracle"]:
        a = np.mean(res['held-out']['face'][k]); b = np.mean(res['held-out']['instance'][k])
        c = np.mean(res['random']['face'][k]);   e = np.mean(res['random']['instance'][k])
        summary[k] = dict(held_face=round(float(a), 4), held_inst=round(float(b), 4),
                          rand_face=round(float(c), 4), rand_inst=round(float(e), 4))
        print(f"  {k:<12s} {a:>14.4f} {b:>14.4f} {c:>13.4f} {e:>13.4f}")

    print("\nleakage, held-out minus random, on the calibrated arm")
    for u in ("face", "instance"):
        h = np.mean(res['held-out'][u]['calibrated'])
        r = np.mean(res['random'][u]['calibrated'])
        print(f"  {u:<9s} held-out {h:.4f}   random {r:.4f}   "
              f"leak {h - r:+.4f} ({100*(h-r)/max(r,1e-9):+.1f}%)")

    print("\nper held-out source volume, instance unit, calibrated arm")
    print(f"  {'source volume':<24s} {'n':>8s} {'median err':>11s} {'within 2x':>10s}")
    for d in folds:
        m, n, w = per_fold[d]["calibrated"]
        print(f"  {d:<24s} {n:>8,d} {m:>11.4f} {w:>9.1%}")
    ms = [per_fold[d]["calibrated"][0] for d in folds]
    print(f"  {'across folds':<24s} {'':>8s} {np.mean(ms):>11.4f} "
          f"  min {min(ms):.3f} max {max(ms):.3f}")

    json.dump(dict(summary=summary,
                   per_fold={d: {k: list(v) for k, v in per_fold[d].items()} for d in folds},
                   folds=folds, table=TABLE, min_test=MIN_TEST),
              open(f"{ROOT}/cache/reference_bars.json", "w"), indent=1)
    print(f"\n-> cache/reference_bars.json")


if __name__ == "__main__":
    main()
