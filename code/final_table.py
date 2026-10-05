import collections, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPHERE_LOG = np.log(4.0 / 3.0 / np.sqrt(np.pi))


def load(table):
    return [json.loads(l) for l in open(os.path.join(ROOT, table))]


def same_files(rows, held, max_train=1800, max_test=500):
    tr = [r for r in rows if r["dataset"] != held]
    te = [r for r in rows if r["dataset"] == held]
    rng = np.random.default_rng(0)

    def cap(rs, n):
        fs = sorted({r["file"] for r in rs})
        if len(fs) <= n:
            return rs
        keep = set(rng.choice(fs, n, replace=False))
        return [r for r in rs if r["file"] in keep]
    return cap(tr, max_train), cap(te, max_test)


def rel(pred, true):
    return float(np.median(np.abs(np.exp(pred) - np.exp(true)) / np.exp(true)))


def r2(pred, true):
    tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ((true - pred) ** 2).sum() / tot) if tot > 0 else float("nan")


def baselines(rows, target, max_train=1800, max_test=500):
    out = collections.defaultdict(dict)
    percls = collections.defaultdict(lambda: collections.defaultdict(list))
    for held in sorted({r["dataset"] for r in rows}):
        tr, te = same_files(rows, held, max_train, max_test)
        if len(te) < 200 or not tr:
            continue
        V = lambda rs: np.log(np.array([r[target] for r in rs], dtype=np.float64))
        A = lambda rs: np.log(np.array([max(r["area_nm2"], 1.0) for r in rs], dtype=np.float64))
        vt, ve, at, ae = V(tr), V(te), A(tr), A(te)
        ot = np.array([r["organelle"] for r in tr]); oe = np.array([r["organelle"] for r in te])
        bt, be = 1.5 * at + SPHERE_LOG, 1.5 * ae + SPHERE_LOG
        arms = {"raw sphere": be}
        g = float(np.median(vt - bt))
        arms["global calib"] = be + g
        m = {o: float(np.median((vt - bt)[ot == o])) for o in set(ot)}
        arms["class calib"] = be + np.array([m.get(o, g) for o in oe])
        q = np.quantile(at, np.linspace(0, 1, 41))
        qt = np.clip(np.digitize(at, q[1:-1]), 0, 39)
        qe = np.clip(np.digitize(ae, q[1:-1]), 0, 39)
        for name, use_class in (("area lookup", False), ("class+area lookup", True)):
            d = collections.defaultdict(list)
            kt = list(zip(ot, qt)) if use_class else [(b,) for b in qt]
            ke = list(zip(oe, qe)) if use_class else [(b,) for b in qe]
            for k, v in zip(kt, vt):
                d[k].append(v)
            md = {k: float(np.median(v)) for k, v in d.items() if len(v) >= 5}
            gl = float(np.median(vt))
            arms[name] = np.array([md.get(k, gl) for k in ke])
        for name, p in arms.items():
            out[held][name] = dict(rel=rel(p, ve), r2=r2(p, ve), n=len(ve))
            for o in set(oe):
                s = oe == o
                if s.sum() >= 30:
                    percls[name][o].append(rel(p[s], ve[s]))
    return out, percls


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "volume_nm3"
    table = sys.argv[2] if len(sys.argv) > 2 else "cache/inst_bf_organelle.jsonl"
    runs = {"model (class head)": "cache/ivol_v2.json",
            "model (no class head)": "cache/ivol_v2_nocls.json",
            "model (below-cut)": "cache/ivol_below.json",
            "model (first attempt)": "cache/ivol_bf_organelle.json"}

    rows = [r for r in load(table) if r.get("unbiased") and r.get(target, 0) > 0]
    base, percls = baselines(rows, target)
    folds = sorted(base)
    print(f"target={target}   {len(rows):,} instances   {len(folds)} folds\n")

    got = {}
    for name, path in runs.items():
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        res = [r for r in json.load(open(p)) if r]
        if not res:
            continue
        key = "model" if isinstance(res[0].get("model"), dict) else None
        got[name] = {r["held"]: (r["model"]["med"] if key else r["model_err"]) for r in res}

    arms = ["raw sphere", "global calib", "area lookup", "class+area lookup", "class calib"]
    hdr = f"{'fold':<24s}" + "".join(f"{a[:13]:>15s}" for a in arms) + \
          "".join(f"{n[:15]:>17s}" for n in got)
    print(hdr)
    for h in folds:
        line = f"{h:<24s}" + "".join(f"{base[h][a]['rel']:>15.3f}" for a in arms)
        for n in got:
            v = got[n].get(h)
            line += f"{(f'{v:.3f}' if v is not None else '-'):>17s}"
        print(line)
    print()
    line = f"{'MACRO over folds':<24s}" + \
        "".join(f"{np.mean([base[h][a]['rel'] for h in folds]):>15.3f}" for a in arms)
    for n in got:
        v = [got[n][h] for h in folds if h in got[n]]
        line += f"{(f'{np.mean(v):.3f}' if v else '-'):>17s}"
    print(line)
    line = f"{'  (folds covered)':<24s}" + "".join(f"{len(folds):>15d}" for a in arms)
    for n in got:
        line += f"{len([h for h in folds if h in got[n]]):>17d}"
    print(line)

    print(f"\nR2 in log space, per fold, for the class+area lookup only, to show the spread")
    print(f"{'fold':<24s} {'n':>8s} {'rel':>8s} {'R2':>10s}")
    for h in folds:
        d = base[h]["class+area lookup"]
        print(f"{h:<24s} {d['n']:>8,d} {d['rel']:>8.3f} {d['r2']:>+10.3f}")
    v = [base[h]["class+area lookup"]["r2"] for h in folds]
    print(f"{'mean':<24s} {'':>8s} {'':>8s} {np.mean(v):>+10.3f}")
    print(f"{'median':<24s} {'':>8s} {'':>8s} {np.median(v):>+10.3f}"
          f"   <- the mean is dragged by whichever fold has the least size spread")

    print(f"\nequal-class macro, baselines only (model per-class comes from its own json)")
    print(f"{'class':<12s}" + "".join(f"{a[:13]:>15s}" for a in arms))
    names = sorted({o for a in arms for o in percls[a]},
                   key=lambda o: -len(percls[arms[0]].get(o, [])))
    for o in names:
        print(f"{o:<12s}" + "".join(
            f"{np.mean(percls[a][o]):>15.3f}" if percls[a].get(o) else f"{'-':>15s}"
            for a in arms))
    print(f"{'MACRO':<12s}" + "".join(
        f"{np.mean([np.mean(percls[a][o]) for o in percls[a]]):>15.3f}" for a in arms))


if __name__ == "__main__":
    main()
