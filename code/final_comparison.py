import argparse, collections, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPHERE_LOG = np.log(4.0 / 3.0 / np.sqrt(np.pi))


def rel(pred, true):
    return float(np.median(np.abs(np.exp(pred) - np.exp(true)) / np.exp(true)))


def by_instance(pred, true, key):
    d = collections.defaultdict(lambda: ([], []))
    for p, t, k in zip(pred, true, key):
        d[k][0].append(p); d[k][1].append(t)
    P = np.array([np.mean(v[0]) for v in d.values()])
    T = np.array([v[1][0] for v in d.values()])
    return rel(P, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--table", default="cache/inst_bf_exact.jsonl")
    ap.add_argument("--run", default="cache/ivol_fixed.json")
    ap.add_argument("--max-train-files", type=int, default=3000)
    ap.add_argument("--max-test-files", type=int, default=800)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    rows = [r for r in rows if r.get("unbiased")]
    V = np.log([r["volume_nm3"] for r in rows])
    A = np.log([max(r["area_nm2"], 1.0) for r in rows])
    O = np.array([r["organelle"] for r in rows])
    D = np.array([r["dataset"] for r in rows])
    F = np.array([r["file"] for r in rows])
    K = np.array([f"{r['box']}|{r['instance']}" for r in rows])

    arms = collections.defaultdict(lambda: collections.defaultdict(dict))
    for held in sorted(set(D)):
        te_all = D == held
        tr_all = ~te_all
        if te_all.sum() < 200:
            continue
        rng = np.random.default_rng(0)

        def cap(mask, n):
            fs = sorted(set(F[mask].tolist()))
            if len(fs) <= n:
                return mask
            keep = set(rng.choice(fs, n, replace=False))
            return mask & np.array([f in keep for f in F])
        tr = cap(tr_all, a.max_train_files)
        te = cap(te_all, a.max_test_files)
        if te.sum() < 100:
            continue

        b = 1.5 * A + SPHERE_LOG
        pred = {"raw sphere": b[te]}
        g = float(np.median((V - b)[tr]))
        pred["global calib"] = b[te] + g
        m = {o: float(np.median((V - b)[tr][O[tr] == o])) for o in set(O[tr])
             if (O[tr] == o).sum() > 20}
        pred["class calib"] = b[te] + np.array([m.get(o, g) for o in O[te]])
        q = np.quantile(A[tr], np.linspace(0, 1, 41))
        bt = np.clip(np.digitize(A[tr], q[1:-1]), 0, 39)
        be = np.clip(np.digitize(A[te], q[1:-1]), 0, 39)
        for name, use_class in (("area lookup", False), ("class+area lookup", True)):
            d = collections.defaultdict(list)
            kt = list(zip(O[tr], bt)) if use_class else [(x,) for x in bt]
            ke = list(zip(O[te], be)) if use_class else [(x,) for x in be]
            for k, v in zip(kt, V[tr]):
                d[k].append(v)
            md = {k: float(np.median(v)) for k, v in d.items() if len(v) >= 5}
            gl = float(np.median(V[tr]))
            pred[name] = np.array([md.get(k, gl) for k in ke])

        for name, p in pred.items():
            arms[name][held] = dict(face=rel(p, V[te]),
                                    inst=by_instance(p, V[te], K[te]), n=int(te.sum()))

    model = {}
    rp = os.path.join(a.root, a.run)
    if os.path.exists(rp):
        for r in json.load(open(rp)):
            if r:
                model[r["held"]] = r["model"]["med"]

    folds = sorted(arms["raw sphere"])
    order = ["raw sphere", "global calib", "area lookup", "class+area lookup", "class calib"]
    print(f"{len(folds)} held-out source volumes\n")
    print(f"{'fold':<24s} {'n':>7s}" + "".join(f"{k[:12]:>14s}" for k in order) +
          f"{'model':>10s}")
    for h in folds:
        line = f"{h:<24s} {arms['raw sphere'][h]['n']:>7,d}"
        line += "".join(f"{arms[k][h]['inst']:>14.3f}" for k in order)
        v = model.get(h)
        line += f"{(f'{v:.3f}' if v is not None else '-'):>10s}"
        print(line)
    print()
    line = f"{'MACRO, per instance':<24s} {'':>7s}"
    line += "".join(f"{np.mean([arms[k][h]['inst'] for h in folds]):>14.3f}" for k in order)
    mv = [model[h] for h in folds if h in model]
    line += f"{(f'{np.mean(mv):.3f}' if mv else '-'):>10s}"
    print(line)
    line = f"{'MACRO, per cut face':<24s} {'':>7s}"
    line += "".join(f"{np.mean([arms[k][h]['face'] for h in folds]):>14.3f}" for k in order)
    print(line + f"{'-':>10s}")
    print(f"\nmodel folds covered {len(mv)} of {len(folds)}")
    if mv:
        best = min(np.mean([arms[k][h]["inst"] for h in folds]) for k in order)
        print(f"best baseline per instance {best:.3f}   model {np.mean(mv):.3f}")


if __name__ == "__main__":
    main()
