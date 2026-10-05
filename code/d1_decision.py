import json, sys
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")


def d1_of(pred, true):
    r = pred / true
    return float(np.mean(np.maximum(r, 1.0 / r) < 1.25))


def best_constant_d1(true, grid=None):
    if grid is None:
        grid = np.unique(np.concatenate([
            np.arange(1, 41, dtype=np.float64),
            np.exp(np.linspace(np.log(0.8), np.log(max(true.max(), 2.0)), 400))]))
    best_a, best_v = grid[0], -1.0
    for a in grid:
        v = float(np.mean(np.maximum(a / true, true / a) < 1.25))
        if v > best_v:
            best_a, best_v = float(a), v
    return best_a, best_v


def main():
    meta = json.load(open(f"{CACHE}/meta.json"))
    N = meta["n"]
    DP = np.load(f"{CACHE}/depth.npy", mmap_mode="r")[:N]
    VM = np.load(f"{CACHE}/valid.npy", mmap_mode="r")[:N]
    ds = np.array([m["dataset"] for m in meta["meta"]])
    PX = np.array([float(m["px_nm"]) for m in meta["meta"]], np.float32)
    ZS = np.array([float(m.get("z_step_nm", 0) or 0) for m in meta["meta"]], np.float32)
    ZS[ZS <= 0] = PX[ZS <= 0]

    vols = sorted(set(ds.tolist()))
    print(f"{N:,} planes, {len(vols)} source volumes\n")

    rng = np.random.default_rng(0)
    steps, vol_of, zs_of = [], [], []
    for i in range(N):
        m = np.asarray(VM[i])
        if not m.any():
            continue
        s = np.asarray(DP[i], np.float32)[m] / ZS[i]
        if s.size > 4000:
            s = s[rng.choice(s.size, 4000, replace=False)]
        steps.append(s.astype(np.float32))
        vol_of.append(np.full(s.size, vols.index(ds[i]), np.int16))
        zs_of.append(np.full(s.size, ZS[i], np.float32))
    S = np.concatenate(steps); V = np.concatenate(vol_of); Z = np.concatenate(zs_of)
    ok = np.isfinite(S) & (S > 0)
    S, V, Z = S[ok], V[ok], Z[ok]
    print(f"scored sample {S.size:,} pixels")
    print("step quantiles", {q: round(float(np.quantile(S, q)), 2)
                             for q in (.1, .25, .5, .75, .9, .99)})
    print(f"exactly 1 step  {float((S == 1).mean()):.3f}")
    print(f"<= 2 steps      {float((S <= 2).mean()):.3f}")
    print(f"<= 4 steps      {float((S <= 4).mean()):.3f}\n")

    rows = []
    for k, v in enumerate(vols):
        te = V == k
        tr = ~te
        if te.sum() < 1000 or tr.sum() < 1000:
            continue
        med = float(np.median(S[tr]))
        a_opt, _ = best_constant_d1(S[tr])
        rows.append((v, d1_of(np.full(te.sum(), med), S[te]),
                     d1_of(np.full(te.sum(), a_opt), S[te]), med, a_opt, int(te.sum())))

    print(f"{'held-out volume':28s}{'median':>9s}{'d1-opt':>9s}{'gain':>8s}"
          f"{'a_med':>8s}{'a_opt':>8s}{'pixels':>12s}")
    for v, dm, do, med, ao, n in rows:
        print(f"{v:28s}{dm:9.3f}{do:9.3f}{do - dm:+8.3f}{med:8.1f}{ao:8.2f}{n:12,}")
    dm = np.mean([r[1] for r in rows]); do = np.mean([r[2] for r in rows])
    print(f"\n{'mean':28s}{dm:9.3f}{do:9.3f}{do - dm:+8.3f}")
    print(f"\nfor reference, the trained arms reach d1  M 0.162  P 0.160  S 0.187  B 0.188")

    json.dump(dict(rows=[dict(volume=r[0], d1_median=r[1], d1_optimal=r[2],
                              a_median=r[3], a_optimal=r[4], pixels=r[5]) for r in rows],
                   mean_d1_median=float(dm), mean_d1_optimal=float(do)),
              open("cache/d1_decision.json", "w"), indent=1)
    print("-> cache/d1_decision.json")


if __name__ == "__main__":
    main()
