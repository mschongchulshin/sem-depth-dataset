import os
import json, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/code")
from relmetrics import per_face, aggregate

CACHES = {
    "depth": ("/private/tmp/claude-501/-Users-hongchulshin/"
              "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache"),
    "height": ("/private/tmp/claude-501/-Users-hongchulshin/"
               "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/heightcache"),
}


def chance_line(task, n_faces=600, seed=0):
    from scipy import ndimage
    C = CACHES[task]
    meta = json.load(open(f"{C}/meta.json"))
    N = meta["n"]
    DP = np.load(f"{C}/depth.npy", mmap_mode="r")[:N]
    VM = np.load(f"{C}/valid.npy", mmap_mode="r")[:N]
    px = np.array([float(m["px_nm"]) for m in meta["meta"]], np.float32)
    zs = np.array([float(m.get("z_step_nm", 0) or 0) for m in meta["meta"]], np.float32)
    zs[zs <= 0] = px[zs <= 0]
    ds = np.array([m["dataset"] for m in meta["meta"]])

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, min(n_faces, N), replace=False)
    out = {k: [] for k in ("constant", "edt", "chord")}
    vols = []
    for i in idx:
        vm = np.asarray(VM[i])
        if vm.sum() < 500:
            continue
        t = np.asarray(DP[i], np.float32)[vm] / zs[i]
        if not np.isfinite(t).all() or (t <= 0).any():
            continue
        e_px = ndimage.distance_transform_edt(vm)
        e = (e_px * px[i] / zs[i])
        lab, n = ndimage.label(vm)
        if n == 0:
            continue
        rmax = ndimage.maximum(e, lab, index=np.arange(1, n + 1))
        R = np.zeros_like(e); R[vm] = rmax[lab[vm] - 1]
        chord = np.sqrt(np.maximum(e * (2 * R - e), 1e-9))[vm]
        for k, p in (("constant", np.ones(t.size)), ("edt", e[vm]), ("chord", chord)):
            r = per_face(p.astype(np.float64), t, rng)
            if r:
                out[k].append(r)
        vols.append(ds[i])
    res = {k: aggregate(v) for k, v in out.items()}
    res["_faces"] = len(vols)
    res["_median_steps"] = float(np.median(np.concatenate(
        [np.asarray(DP[i], np.float32)[np.asarray(VM[i])] / zs[i] for i in idx[:100]])))
    return res


def paired(depth_json, height_json, draws=20000, seed=0):
    d = json.load(open(depth_json))
    h = json.load(open(height_json))
    arms = sorted(a for a in d if d[a].get("ok") and a in h and h[a].get("ok"))
    folds = sorted(set(d[arms[0]]["per_fold"]) & set(h[arms[0]]["per_fold"]))
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ("pair_acc", "d1_aligned", "spearman"):
        diff = np.array([np.mean([h[a]["per_fold"][f][metric] - d[a]["per_fold"][f][metric]
                                  for a in arms]) for f in folds])
        boot = np.array([diff[rng.integers(0, len(folds), len(folds))].mean()
                         for _ in range(draws)])
        out[metric] = dict(mean=float(diff.mean()),
                           ci95=[float(np.percentile(boot, 2.5)),
                                 float(np.percentile(boot, 97.5))],
                           per_volume={f: float(v) for f, v in zip(folds, diff)},
                           volumes=len(folds), arms=len(arms))
    return out


if __name__ == "__main__":
    res = {}
    for task in ("depth", "height"):
        print(f"\nchance and geometry lines for {task}", flush=True)
        r = chance_line(task)
        res[task] = r
        print(f"  faces {r['_faces']}   median target {r['_median_steps']:.1f} steps")
        for k in ("constant", "edt", "chord"):
            print(f"    {k:9s} pair_acc {r[k]['pair_acc']:.4f}   "
                  f"d1_aligned {r[k]['d1_aligned']:.4f}   "
                  f"spearman {r[k]['spearman']:+.3f}")
    print("\npaired over held out volumes, height minus depth")
    p = paired(f"{ROOT}/cache/depth_arms_matched.json",
               f"{ROOT}/cache/height_arms.json")
    res["paired"] = p
    for m, v in p.items():
        sig = "excludes zero" if (v["ci95"][0] > 0 or v["ci95"][1] < 0) else "includes zero"
        print(f"  {m:12s} {v['mean']:+.4f}   95% [{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]   {sig}")
        for f, x in v["per_volume"].items():
            print(f"      {f:26s} {x:+.4f}")
    json.dump(res, open(f"{ROOT}/cache/height_vs_depth.json", "w"), indent=1)
    print(f"\n-> cache/height_vs_depth.json")
