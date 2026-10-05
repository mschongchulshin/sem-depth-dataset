import glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"
OUT = f"{ROOT}/release/blockface"
RAY = ("depth_below_nm", "own_occupancy_below_nm", "thickness_below_nm")
KEEP = ("em", "inst_face", "cls_face", "mask", "clipped")


def main():
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    files = sorted(f for f in glob.glob(f"{ROOT}/blockface/*/faces/*.npz") if "/._" not in f)
    files = files[shard::nshard]
    print(f"shard {shard}/{nshard}: {len(files):,} planes", flush=True)

    n_ok = n_bad = 0
    worst = 0.0
    bad_files = []
    for i, f in enumerate(files):
        if i % 1000 == 0 and i:
            print(f"  {i:,}/{len(files):,}  ok {n_ok:,}  bad {n_bad}", flush=True)
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except Exception as e:
            n_bad += 1; bad_files.append((f, f"unreadable {e}")); continue
        zs = float(meta.get("z_step_nm", 0) or 0)
        if zs <= 0:
            n_bad += 1; bad_files.append((f, "no z_step_nm")); continue

        out = {k: z[k] for k in KEEP if k in z.files}
        fail = None
        for a in RAY:
            if a not in z.files:
                continue
            v = z[a].astype(np.float64)
            fin = np.isfinite(v)
            cnt = np.zeros(v.shape, np.uint16)
            q = np.where(fin, v / zs, 0.0)
            r = np.round(q)
            if r.max() > 65535 or (fin & (v < 0)).any():
                fail = f"{a} out of range"; break
            err = np.abs(np.where(fin, q - r, 0.0))
            tol = np.maximum(1e-4, np.abs(q) * 1e-5)
            if (err > tol).any():
                fail = f"{a} not a step multiple, max dev {err.max():.2e}"; break
            worst = max(worst, float(err.max()))
            cnt[fin] = r[fin].astype(np.uint16)
            out[a.replace("_nm", "_steps")] = cnt
        if fail:
            n_bad += 1; bad_files.append((f, fail)); continue

        meta["ray_units"] = ("ray quantities are stored as integer z-step counts. "
                             "Multiply by z_step_nm for nanometres. 0 means no exposed "
                             "instance, which mask also marks.")
        meta["z_step_nm"] = zs
        out["meta"] = np.array(json.dumps(meta))

        rel = os.path.relpath(f, f"{ROOT}/blockface")
        dst = f"{OUT}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, **out)

        with np.load(dst, allow_pickle=False) as w:
            for a in RAY:
                k = a.replace("_nm", "_steps")
                if k not in w.files or a not in z.files:
                    continue
                back = w[k].astype(np.float64) * zs
                orig = z[a].astype(np.float64)
                fin = np.isfinite(orig)
                d = np.abs(back[fin] - orig[fin])
                if d.size and d.max() > max(1e-3, zs * 1e-4):
                    fail = f"{a} round trip off by {d.max():.4f} nm"
        if fail:
            n_bad += 1; bad_files.append((f, fail)); os.remove(dst); continue
        n_ok += 1

    print(f"\nshard {shard}: written {n_ok:,}   refused {n_bad}")
    print(f"  worst step-count deviation before rounding {worst:.2e}")
    for f, why in bad_files[:20]:
        print(f"  {os.path.basename(f)}  {why}")
    json.dump([[f, w] for f, w in bad_files],
              open(f"{ROOT}/cache/convert_bad_{shard:02d}.json", "w"))


if __name__ == "__main__":
    main()
