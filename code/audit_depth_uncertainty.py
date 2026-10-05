import os
import collections, json
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    meta = json.load(open(f"{CACHE}/meta.json"))
    N = meta["n"]
    DP = np.load(f"{CACHE}/depth.npy", mmap_mode="r")[:N]
    VM = np.load(f"{CACHE}/valid.npy", mmap_mode="r")[:N]
    recs = meta["meta"]

    grad = collections.defaultdict(list)
    allg = []
    steps = []
    n_used = 0
    rng = np.random.default_rng(0)
    order = rng.permutation(N)[:2500]

    for j, i in enumerate(order):
        if j % 400 == 0 and j:
            print(f"  {j}/{len(order)}", flush=True)
        r = recs[int(i)]
        try:
            z = np.load(f"{ROOT}/{r['file']}", allow_pickle=False)
            m = json.loads(str(z["meta"]))
            ef = z["inst_face"]
        except Exception:
            continue
        zs = float(m.get("z_step_nm", 0) or 0)
        if zs <= 0:
            continue
        names = {int(k): v.get("organelle") for k, v in m.get("instances", {}).items()}
        H, W = ef.shape
        CROP = meta["crop"]
        y0, x0 = (H - CROP) // 2, (W - CROP) // 2
        if H < CROP or W < CROP:
            continue
        ef = ef[y0:y0 + CROP, x0:x0 + CROP]
        d = np.asarray(DP[int(i)], np.float64)
        v = np.asarray(VM[int(i)])
        if v.sum() < 500:
            continue
        n_used += 1
        steps.append(zs)

        for ax in (0, 1):
            a = v & np.roll(v, 1, axis=ax) & (ef == np.roll(ef, 1, axis=ax))
            a[0 if ax == 0 else slice(None), 0 if ax == 1 else slice(None)] = False
            if not a.any():
                continue
            g = np.abs(d - np.roll(d, 1, axis=ax))[a] / zs
            ids = ef[a]
            allg.append(g)
            for u in np.unique(ids):
                o = names.get(int(u))
                if not o:
                    continue
                gg = g[ids == u]
                if gg.size >= 20:
                    grad[o].append(gg)

    print(f"\nplanes used {n_used:,}   z steps present "
          f"{sorted(set(round(s, 2) for s in steps))[:8]} ...")
    G = np.concatenate(allg)
    print(f"neighbouring foreground pixel pairs on one instance  {G.size:,}")
    print("\nlateral sensitivity: depth change between neighbouring pixels, in z steps")
    print("this is what a one-pixel boundary error costs\n")
    print(f"  {'class':<11s}{'pairs':>12s}{'median':>9s}{'p75':>8s}{'p90':>8s}{'p99':>8s}"
          f"{'= 0':>8s}")
    out = {}
    for o in sorted(grad, key=lambda k: -sum(g.size for g in grad[k])):
        g = np.concatenate(grad[o])
        if g.size < 5000:
            continue
        out[o] = dict(pairs=int(g.size), median=float(np.median(g)),
                      p75=float(np.percentile(g, 75)), p90=float(np.percentile(g, 90)),
                      p99=float(np.percentile(g, 99)), zero=float(np.mean(g == 0)))
        print(f"  {o:<11s}{g.size:>12,d}{np.median(g):>9.2f}{np.percentile(g,75):>8.2f}"
              f"{np.percentile(g,90):>8.2f}{np.percentile(g,99):>8.2f}"
              f"{100*np.mean(g==0):>7.1f}%")
    print(f"\n  {'all':<11s}{G.size:>12,d}{np.median(G):>9.2f}{np.percentile(G,75):>8.2f}"
          f"{np.percentile(G,90):>8.2f}{np.percentile(G,99):>8.2f}"
          f"{100*np.mean(G==0):>7.1f}%")
    json.dump(dict(by_class=out,
                   overall=dict(pairs=int(G.size), median=float(np.median(G)),
                                p90=float(np.percentile(G, 90)),
                                p99=float(np.percentile(G, 99)),
                                zero=float(np.mean(G == 0))),
                   planes=n_used),
              open(f"{ROOT}/cache/depth_uncertainty.json", "w"), indent=1)
    print("\nreading")
    print("  a median near zero means the lower surface is perpendicular to the ray, so a "
          "one-pixel lateral error costs no depth")
    print("  a long tail means that class has oblique surfaces, where a boundary error "
          "carries far into the depth")
    print("  the axial direction needs no measurement: one voxel is exactly one z step")
    print("\n-> cache/depth_uncertainty.json")


if __name__ == "__main__":
    main()
