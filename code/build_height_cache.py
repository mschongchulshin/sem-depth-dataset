import argparse, json, os, sys
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ("/private/tmp/claude-501/-Users-hongchulshin/"
       "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/heightcache")
CROP = 256


def run_lengths_up(inst, z):
    face = inst[z]
    out = np.zeros(face.shape, np.uint16)
    alive = face > 0
    for k in range(z - 1, -1, -1):
        alive &= (inst[k] == face)
        if not alive.any():
            break
        out[alive] += 1
    return out, alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-boxes", type=int, default=0, help="0 means every box")
    ap.add_argument("--per-box", type=int, default=6, help="faces sampled per box")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(f"{ROOT}/raw")
                   if f.endswith(".npz") and not f.startswith("._"))
    if a.max_boxes:
        files = files[:a.max_boxes]
    by = {}
    for f in files:
        by.setdefault(f.split("__")[0], []).append(f)
    print(f"{len(files):,} boxes over {len(by)} source volumes", flush=True)

    cap = len(files) * a.per_box
    em = np.lib.format.open_memmap(f"{a.out}/em.npy", mode="w+", dtype=np.uint8,
                                   shape=(cap, CROP, CROP))
    hp = np.lib.format.open_memmap(f"{a.out}/depth.npy", mode="w+", dtype=np.uint16,
                                   shape=(cap, CROP, CROP))
    vm = np.lib.format.open_memmap(f"{a.out}/valid.npy", mode="w+", dtype=bool,
                                   shape=(cap, CROP, CROP))
    meta, n = [], 0
    for bi, f in enumerate(files):
        if bi % 50 == 0:
            print(f"  box {bi:,}/{len(files):,}  staged {n:,}", flush=True)
        try:
            z = np.load(f"{ROOT}/raw/{f}")
            inst, emv = z["inst"], z["em"]
            mt = json.loads(str(z["meta"])) if "meta" in z.files else {}
        except Exception:
            continue
        Z, Y, X = inst.shape
        if min(Y, X) < 64:
            continue
        v = mt.get("voxel_size_nm") or []
        if len(v) != 3:
            continue
        zs, px = float(v[0]), float(v[2])
        if zs <= 0 or px <= 0:
            continue
        y0, x0 = max(0, (Y - CROP) // 2), max(0, (X - CROP) // 2)
        sl = (slice(y0, min(y0 + CROP, Y)), slice(x0, min(x0 + CROP, X)))

        def fit(arr, fill=0):
            a2 = arr[sl]
            py, px2 = CROP - a2.shape[0], CROP - a2.shape[1]
            if py or px2:
                mode = "reflect" if a2.shape[0] > 1 and a2.shape[1] > 1 else "constant"
                kw = {} if mode == "reflect" else dict(constant_values=fill)
                a2 = np.pad(a2, ((0, py), (0, px2)), mode=mode, **kw)
            return a2
        lo, hi = int(0.25 * Z), int(0.95 * Z)
        if hi - lo < a.per_box:
            continue
        for zc in np.linspace(lo, hi, a.per_box).astype(int):
            h, clip = run_lengths_up(inst, int(zc))
            hh, cc = fit(h), fit(clip)
            valid = (fit(inst[zc]) > 0) & ~cc & (hh > 0)
            if (Y < CROP) or (X < CROP):
                keep = np.zeros((CROP, CROP), bool)
                keep[:min(CROP, Y - y0), :min(CROP, X - x0)] = True
                valid &= keep
            if valid.sum() < 1500:
                continue
            em[n] = fit(emv[zc])
            hp[n] = np.clip(hh.astype(np.float64) * zs, 0, 65535).astype(np.uint16)
            vm[n] = valid
            meta.append(dict(dataset=f.split("__")[0], box=f[:-4], px_nm=px,
                             z_step_nm=zs, z=int(zc), frac=float(valid.mean())))
            n += 1
            if n >= cap:
                break
        if n >= cap:
            break
    em.flush(); hp.flush(); vm.flush()
    json.dump(dict(n=n, crop=CROP, meta=meta), open(f"{a.out}/meta.json", "w"))
    byv = {}
    for m2 in meta:
        byv[m2["dataset"]] = byv.get(m2["dataset"], 0) + 1
    print(f"\nstaged {n:,} faces over {len(byv)} source volumes")
    for v, c in sorted(byv.items(), key=lambda kv: -kv[1]):
        print(f"  {v:28s} {c}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
