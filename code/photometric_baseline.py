import argparse, glob, json, os
import numpy as np

AZIMUTHS = (45, 135, 225, 315)
STEEP = 0.30


def quadrant_gradients(bse, eps=1e-6):
    a = (bse[0] - bse[2]) / (bse[0] + bse[2] + eps)
    b = (bse[1] - bse[3]) / (bse[1] + bse[3] + eps)
    ua, ub = np.deg2rad(AZIMUTHS[0]), np.deg2rad(AZIMUTHS[1])
    gx = a * np.cos(ua) + b * np.cos(ub)
    gy = a * np.sin(ua) + b * np.sin(ub)
    return gx.astype(np.float32), gy.astype(np.float32)


def frankot_chellappa(gx, gy):
    H, W = gx.shape
    wy = np.fft.fftfreq(H).reshape(-1, 1) * 2 * np.pi
    wx = np.fft.fftfreq(W).reshape(1, -1) * 2 * np.pi
    d = wx ** 2 + wy ** 2
    d[0, 0] = 1.0
    Z = (-1j * wx * np.fft.fft2(gx) - 1j * wy * np.fft.fft2(gy)) / d
    Z[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z)).astype(np.float32)


def fit_affine(pred, true, m):
    p, t = pred[m], true[m]
    if p.size < 16 or p.std() < 1e-9:
        return np.zeros_like(pred) + (t.mean() if t.size else 0.0)
    A = np.stack([p, np.ones_like(p)], 1)
    c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return (pred * c[0] + c[1]).astype(np.float32)


def err(pred, true, m):
    if m.sum() < 16:
        return None
    p, t = pred[m], true[m]
    return dict(mae_nm=float(np.abs(p - t).mean()),
                rmse_nm=float(np.sqrt(((p - t) ** 2).mean())),
                absrel=float((np.abs(p - t) / np.maximum(np.abs(t), 1.0)).mean()),
                n=int(m.sum()))


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return {k: sum(r[k] * r["n"] for r in rows) / n for k in rows[0] if k != "n"} | {"n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "corpus", "*", "views", "*.npz")))
    rng = np.random.default_rng(0)
    if len(files) > a.limit:
        files = list(rng.choice(files, a.limit, replace=False))
    print(f"{len(files)} surface views", flush=True)

    all_, flat, steep, occl = [], [], [], []
    frac_steep, frac_occl = [], []
    for f in files:
        z = np.load(f)
        m = z["mask"].astype(bool)
        if m.sum() < 256:
            continue
        depth = np.asarray(z["depth_nm"], dtype=np.float32)
        m &= np.isfinite(depth)
        if m.sum() < 256:
            continue
        h = -depth
        gx, gy = quadrant_gradients(np.asarray(z["bse"], dtype=np.float32))
        rec = fit_affine(frankot_chellappa(gx, gy), h, m)

        nz = np.abs(np.asarray(z["normal"], dtype=np.float32)[2])
        sm = m & (nz < STEEP)
        fm = m & (nz >= STEEP)
        om = m & (np.asarray(z["class_depth_count"]) > 1) if "class_depth_count" in z.files \
            else np.zeros_like(m)
        frac_steep.append(float(sm.sum()) / float(m.sum()))
        frac_occl.append(float(om.sum()) / float(m.sum()))
        all_.append(err(rec, h, m)); flat.append(err(rec, h, fm))
        steep.append(err(rec, h, sm)); occl.append(err(rec, h, om))

    rows = [("all pixels", agg(all_)), ("flat  |nz|>=0.30", agg(flat)),
            ("steep |nz|< 0.30", agg(steep)), ("occluded ray", agg(occl))]
    print(f"\nclassical four-quadrant integration, scale and offset fitted per image")
    print(f"{'region':<20s} {'pixels':>12s} {'MAE nm':>10s} {'RMSE nm':>10s} {'AbsRel':>8s}")
    for name, v in rows:
        if not v:
            print(f"{name:<20s} {'-':>12s}"); continue
        print(f"{name:<20s} {v['n']:>12,d} {v['mae_nm']:>10,.0f} {v['rmse_nm']:>10,.0f} "
              f"{v['absrel']:>8.3f}")
    print(f"\nsteep pixels     {np.mean(frac_steep) * 100:.1f}% of the image")
    print(f"occluded pixels  {np.mean(frac_occl) * 100:.1f}% of the image")
    fa, st = agg(flat), agg(steep)
    if fa and st:
        print(f"steep-to-flat MAE ratio  {st['mae_nm'] / max(fa['mae_nm'], 1e-9):.2f}x")
    if a.out:
        json.dump({n: v for n, v in rows}, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
