import numpy as np


def _spearman(a, b):
    if a.size < 8:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def per_face(pred, true, rng, n_pairs=20000, tie=0.5):
    out = {}
    n = true.size
    if n < 16:
        return None

    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    dt = true[j] - true[i]
    keep = np.abs(dt) > tie
    if keep.sum() >= 32:
        dp = pred[j][keep] - pred[i][keep]
        sd, st = np.sign(dp), np.sign(dt[keep])
        score = np.where(sd == 0, 0.5, (sd == st).astype(np.float64))
        out["pair_acc"] = float(score.mean())
        out["pair_n"] = int(keep.sum())
    else:
        out["pair_acc"], out["pair_n"] = np.nan, 0

    out["spearman"] = _spearman(pred, true)

    A = np.stack([pred, np.ones_like(pred)], 1).astype(np.float64)
    spread_p = float(np.percentile(pred, 95) - np.percentile(pred, 5))
    coef, aligned = np.array([1.0, 0.0]), pred.astype(np.float64)
    if spread_p > 1e-6 * max(float(np.median(pred)), 1e-9):
        try:
            c, *_ = np.linalg.lstsq(A, true.astype(np.float64), rcond=None)
            if np.isfinite(c).all() and 1e-6 < abs(c[0]) < 1e6:
                coef, aligned = c, A @ c
        except np.linalg.LinAlgError:
            pass
    spread = float(np.percentile(true, 95) - np.percentile(true, 5))
    out["ssi_rel"] = float(np.mean(np.abs(aligned - true)) / max(spread, 1e-9))
    out["fit_scale"] = float(coef[0])

    lr = np.log(np.maximum(true, 1e-6)) - np.log(np.maximum(pred, 1e-6))
    k = float(np.exp(np.median(lr)))
    pk = np.maximum(pred * k, 1e-6)
    r = pk / true
    ratio = np.maximum(r, 1 / r)
    out["scale_k"] = k
    out["d1_aligned"] = float(np.mean(ratio < 1.25))
    out["d2_aligned"] = float(np.mean(ratio < 1.25 ** 2))
    out["absrel_aligned"] = float(np.mean(np.abs(pk - true) / true))
    out["medrel_aligned"] = float(np.median(np.abs(pk - true) / true))
    out["log_ratio_iqr"] = float(np.percentile(lr, 75) - np.percentile(lr, 25))
    return out


def aggregate(faces):
    keys = ("d1_aligned", "d2_aligned", "absrel_aligned", "medrel_aligned",
            "log_ratio_iqr", "scale_k", "pair_acc", "spearman", "ssi_rel", "fit_scale")
    out = {}
    for k in keys:
        v = np.array([f[k] for f in faces if f is not None and k in f], np.float64)
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if v.size else float("nan")
    out["faces"] = int(sum(1 for f in faces if f is not None))
    return out
