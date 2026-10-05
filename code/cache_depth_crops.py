import json, os, sys
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "/private/tmp/claude-501/-Users-hongchulshin/1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache"
CROP = 256
MAXN = int(sys.argv[1]) if len(sys.argv) > 1 else 9000


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(f"{ROOT}/cache/depth_index.json"))
    byvol = {}
    for r in idx:
        byvol.setdefault(r["dataset"], []).append(r)
    per = max(1, MAXN // max(len(byvol), 1))
    keep = []
    for v, rs in sorted(byvol.items()):
        keep.extend(rs[:per])
    print(f"staging {len(keep):,} planes from {len(byvol)} source volumes", flush=True)

    em = np.lib.format.open_memmap(f"{OUT}/em.npy", mode="w+", dtype=np.uint8,
                                   shape=(len(keep), CROP, CROP))
    dp = np.lib.format.open_memmap(f"{OUT}/depth.npy", mode="w+", dtype=np.uint16,
                                   shape=(len(keep), CROP, CROP))
    vm = np.lib.format.open_memmap(f"{OUT}/valid.npy", mode="w+", dtype=bool,
                                   shape=(len(keep), CROP, CROP))
    meta = []
    n = 0
    for i, r in enumerate(keep):
        if i % 500 == 0 and i:
            print(f"  {i:,}/{len(keep):,}", flush=True)
        try:
            z = np.load(f"{ROOT}/{r['file']}", allow_pickle=False)
            e, d, m = z["em"], z["depth_below_nm"], z["mask"]
            c = z["clipped"] if "clipped" in z.files else np.zeros_like(m)
            mt = json.loads(str(z["meta"]))
            zstep = float(mt.get("z_step_nm", 0) or 0)
            if zstep <= 0:
                continue
        except Exception:
            continue
        H, W = m.shape
        if H < CROP or W < CROP:
            continue
        y0, x0 = (H - CROP) // 2, (W - CROP) // 2
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        dd = d[sl]
        valid = m[sl] & np.isfinite(dd) & ~c[sl]
        dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
        valid &= dd > 0
        if valid.sum() < 1500:
            continue
        em[n] = e[sl]
        dp[n] = np.clip(dd, 0, 65535).astype(np.uint16)
        vm[n] = valid
        meta.append(dict(dataset=r["dataset"], box=r["box"], px_nm=r["px_nm"],
                         z_step_nm=zstep, file=r["file"], frac=float(valid.mean())))
        n += 1
    em.flush(); dp.flush(); vm.flush()
    json.dump(dict(n=n, crop=CROP, meta=meta), open(f"{OUT}/meta.json", "w"))
    byv = {}
    for m2 in meta:
        byv[m2["dataset"]] = byv.get(m2["dataset"], 0) + 1
    print(f"\nstaged {n:,} planes")
    for v, c2 in sorted(byv.items(), key=lambda kv: -kv[1]):
        print(f"  {v:<24s} {c2:,}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
