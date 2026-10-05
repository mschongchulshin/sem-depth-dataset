import argparse, json, collections
import numpy as np

SPHERE = 4.0 / 3.0 / np.sqrt(np.pi)


def load(path, organelles=None, unbiased_only=True):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if unbiased_only and not r.get("unbiased"):
                continue
            if organelles and r["organelle"] not in organelles:
                continue
            rows.append(r)
    return rows


def med_err(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p) & (t > 0) & (p > 0)
    return float(np.median(np.abs(p[ok] - t[ok]) / t[ok])) if ok.sum() else float("nan")


def fit_scale(rows, keyfn):
    by = collections.defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(np.log(r["volume_nm3"]) - np.log(r["v_stereo"]))
    return {k: float(np.median(v)) for k, v in by.items()}


def fit_prior(rows, keyfn):
    by = collections.defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(np.log(r["volume_nm3"]))
    return {k: float(np.median(v)) for k, v in by.items()}


def fit_linear(rows, feats):
    X, y = [], []
    for r in rows:
        v = [r.get(f) for f in feats]
        if any(x is None or not np.isfinite(x) or x <= 0 for x in v):
            continue
        X.append([1.0] + [np.log(x) for x in v])
        y.append(np.log(r["volume_nm3"]))
    if len(X) < 20:
        return None
    b, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
    return b


def apply_linear(b, r, feats):
    v = [r.get(f) for f in feats]
    if b is None or any(x is None or not np.isfinite(x) or x <= 0 for x in v):
        return np.nan
    return float(np.exp(b[0] + sum(bi * np.log(x) for bi, x in zip(b[1:], v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--min-per-fold", type=int, default=200)
    ap.add_argument("--feats", default="area_nm2,face_feret_nm",
                    help="features for the area+shape rung")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = load(a.table)
    feats = [f for f in a.feats.split(",") if f]
    feats = [f for f in feats if sum(1 for r in rows if r.get(f)) > 0.5 * len(rows)]
    datasets = sorted({r["dataset"] for r in rows})
    print(f"{len(rows):,} unbiased records, {len(datasets)} source volumes")
    print(f"area+shape features in use: {feats}\n")

    RUNGS = ["0 stereology", "1 global calib", "2 class calib", "3 class x tier",
             "4 metadata prior (no area)", "5 area + shape"]
    per_fold = collections.defaultdict(list)
    fold_n = {}

    for held in datasets:
        te = [r for r in rows if r["dataset"] == held]
        tr = [r for r in rows if r["dataset"] != held]
        if len(te) < a.min_per_fold or not tr:
            continue
        fold_n[held] = len(te)
        t = [r["volume_nm3"] for r in te]

        g = fit_scale(tr, lambda r: "all")
        c = fit_scale(tr, lambda r: r["organelle"])
        ct = fit_scale(tr, lambda r: (r["organelle"], r["tier"]))
        pr = fit_prior(tr, lambda r: (r["organelle"], r["tier"]))
        pr_c = fit_prior(tr, lambda r: r["organelle"])
        lin = fit_linear(tr, feats)
        gl_prior = float(np.median([np.log(x["volume_nm3"]) for x in tr]))

        preds = {
            "0 stereology": [r["v_stereo"] for r in te],
            "1 global calib": [r["v_stereo"] * np.exp(g["all"]) for r in te],
            "2 class calib": [r["v_stereo"] * np.exp(c.get(r["organelle"], g["all"])) for r in te],
            "3 class x tier": [r["v_stereo"] * np.exp(
                ct.get((r["organelle"], r["tier"]), c.get(r["organelle"], g["all"]))) for r in te],
            "4 metadata prior (no area)": [np.exp(
                pr.get((r["organelle"], r["tier"]),
                       pr_c.get(r["organelle"], gl_prior))) for r in te],
            "5 area + shape": [apply_linear(lin, r, feats) for r in te],
        }
        for k, p in preds.items():
            per_fold[k].append(med_err(p, t))

    print(f"{'rung':30s} {'macro median err':>17s} {'pooled folds':>13s}")
    print("-" * 63)
    for k in RUNGS:
        v = [x for x in per_fold[k] if np.isfinite(x)]
        if v:
            print(f"{k:30s} {np.mean(v):17.3f} {len(v):13d}")
    print()
    print("held-out volume detail (macro rows above are the mean of these)")
    print(f"{'volume':24s} {'n':>7s} " + " ".join(f"{k.split()[0]:>6s}" for k in RUNGS))
    for i, held in enumerate([d for d in datasets if d in fold_n]):
        line = f"{held:24s} {fold_n[held]:7,d} "
        line += " ".join(f"{per_fold[k][i]:6.3f}" if i < len(per_fold[k]) and
                         np.isfinite(per_fold[k][i]) else "     -" for k in RUNGS)
        print(line)

    if a.out:
        json.dump({k: per_fold[k] for k in RUNGS}, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
