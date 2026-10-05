import collections, glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"

def _excluded():
    import json as _j, os as _o
    p = _o.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = _j.load(open(p))
    return set(e["boxes"]), set(e["planes"])

NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 90
SPHERE_CONST = 0.375
SPHERE_CONST_PERPLANE = 0.397716


def depth_from_raw(sub_inst, exposed, z_step):
    match = sub_inst == exposed[None, :, :]
    Z = match.shape[0]
    first_false = np.argmax(~match, axis=0)
    all_true = match.all(axis=0)
    return np.where(all_true, Z, first_false) * z_step


def synthetic_check():
    size, R = 160, 34
    zz, yy, xx = np.ogrid[:size, :size, :size]
    c = size // 2
    ball = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= R * R
    vals = []
    rng = np.random.default_rng(0)
    for z in rng.integers(c - R + 2, c + R - 2, 60):
        sub = ball[z:]
        face = sub[0]
        if face.sum() < 50:
            continue
        run = depth_from_raw(sub.astype(np.int32), face.astype(np.int32), 1.0)
        vals.append(float(run[face].mean()) / (2 * R))
    m = float(np.mean(vals))
    ok = abs(m - SPHERE_CONST_PERPLANE) / SPHERE_CONST_PERPLANE < 0.06
    print(f"1. the reference constant, on a synthetic sphere")
    print(f"   measured {m:.4f}   theory {SPHERE_CONST_PERPLANE:.4f} (per-plane)   "
          f"{'pass' if ok else 'FAIL'}   over {len(vals)} planes")
    return ok


def main():
    print("=" * 74)
    synthetic_check()

    _sb, _sp = _excluded()
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in _sb)
    rng = np.random.default_rng(1); rng.shuffle(raws)

    mism = tot_px = 0
    maxdiff = 0.0
    clip = collections.Counter(); clip_tot = collections.Counter()
    ratio = collections.defaultdict(list)
    by_vol = collections.defaultdict(list)
    bounds_bad = 0
    n_box = n_face = 0

    for p in raws:
        if n_box >= NBOX:
            break
        box = os.path.basename(p)[:-4]
        vol = box.split("__")[0]
        faces = sorted(f for f in glob.glob(f"{ROOT}/blockface/{box}/faces/*.npz")
                       if os.path.basename(f)[:-4] not in _sp)
        faces = [f for f in faces if "/._" not in f]
        if not faces:
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                if "inst" not in d.files:
                    continue
                inst3 = d["inst"]
                meta3 = json.loads(str(d["meta"]))
        except Exception:
            continue
        if inst3.ndim != 3:
            continue
        names = {int(k): v.get("organelle") for k, v in meta3.get("instances", {}).items()}
        Z3, H3, W3 = inst3.shape
        n_box += 1
        if n_box % 15 == 0:
            print(f"   {n_box}/{NBOX} boxes", flush=True)

        for fp in faces[:5]:
            try:
                z = np.load(fp, allow_pickle=False)
                stored = z["depth_below_nm"]; msk = z["mask"]
                ef = z["inst_face"]
                clipped = z["clipped"] if "clipped" in z.files else None
                m = json.loads(str(z["meta"]))
            except Exception:
                continue
            zi = int(os.path.basename(fp).rsplit("_z", 1)[1].split(".")[0])
            if zi >= Z3:
                continue
            zs = float(m.get("z_step_nm", 0) or 0)
            if zs <= 0:
                continue
            H, W = stored.shape
            y0 = (H3 - H) // 2 if H3 >= H else 0
            x0 = (W3 - W) // 2 if W3 >= W else 0
            sub = inst3[zi:, y0:y0 + H, x0:x0 + W]
            if sub.shape[1:] != stored.shape:
                continue
            n_face += 1
            recomputed = depth_from_raw(sub, ef, zs)
            sel = msk & (ef > 0)
            if sel.sum() == 0:
                continue
            diff = np.abs(recomputed[sel] - stored[sel])
            mism += int((diff > 0.5 * zs).sum()); tot_px += int(sel.sum())
            maxdiff = max(maxdiff, float(diff.max()))

            avail = (Z3 - zi) * zs
            bounds_bad += int((stored[sel] > avail + 0.5 * zs).sum())

            if clipped is not None:
                for i in np.unique(ef[sel]):
                    o = names.get(int(i))
                    if not o:
                        continue
                    mi = sel & (ef == i)
                    clip_tot[o] += int(mi.sum())
                    clip[o] += int(clipped[mi].sum())

            for i in np.unique(ef[sel]):
                o = names.get(int(i))
                rec = meta3.get("instances", {}).get(str(int(i)))
                if not o or not rec:
                    continue
                mi = sel & (ef == i)
                if clipped is not None:
                    mi = mi & ~clipped
                if mi.sum() < 40:
                    continue
                v = rec.get("volume_nm3")
                if not v or v <= 0 or rec.get("truncated"):
                    continue
                diam = (6.0 * v / np.pi) ** (1.0 / 3.0)
                if diam <= 0:
                    continue
                r = float(stored[mi].mean()) / diam
                if 0 < r < 5:
                    ratio[o].append(r)
                    by_vol[vol].append(r)

    print(f"\n   boxes {n_box}, faces {n_face}")
    print("\n" + "=" * 74)
    print(f"2. independent recomputation from the raw sub-volume")
    print(f"   foreground pixels compared  {tot_px:,}")
    print(f"   disagreeing by more than half a z step  {mism:,} "
          f"({100 * mism / max(tot_px, 1):.4f}%)")
    print(f"   largest absolute disagreement  {maxdiff:.1f} nm")
    print(f"   depth exceeding the block remaining below  {bounds_bad:,}")

    print("\n" + "=" * 74)
    print(f"3. how much of the depth is a lower bound rather than a measurement")
    print(f"   {'class':<11s} {'pixels':>10s} {'clipped':>9s}")
    for o in sorted(clip_tot, key=lambda k: -clip_tot[k]):
        print(f"   {o:<11s} {clip_tot[o]:>10,d} {100*clip[o]/max(clip_tot[o],1):>8.2f}%")

    print("\n" + "=" * 74)
    print(f"4. the sphere-cut law, per class, clipped rays excluded")
    print(f"   theory {SPHERE_CONST} for a sphere, pooled over pixels. Lower is expected for")
    print(f"   elongated bodies, and a")
    print(f"   diameter is not a meaningful reference for a sheet or a network.")
    print(f"   {'class':<11s} {'instances':>10s} {'measured':>9s} {'vs theory':>10s}")
    out = {}
    for o in sorted(ratio, key=lambda k: -len(ratio[k])):
        if len(ratio[o]) < 60:
            continue
        m = float(np.median(ratio[o]))
        out[o] = dict(n=len(ratio[o]), ratio=round(m, 4),
                      dev_pct=round(100 * (m - SPHERE_CONST) / SPHERE_CONST, 1),
                      clipped_pct=round(100 * clip[o] / max(clip_tot[o], 1), 2))
        print(f"   {o:<11s} {len(ratio[o]):>10,d} {m:>9.4f} "
              f"{100*(m-SPHERE_CONST)/SPHERE_CONST:>+9.1f}%")

    print("\n" + "=" * 74)
    print(f"5. per source volume, pooled over classes")
    print(f"   {'source volume':<24s} {'n':>8s} {'measured':>9s}")
    volout = {}
    for v in sorted(by_vol, key=lambda k: -len(by_vol[k])):
        if len(by_vol[v]) < 60:
            continue
        m = float(np.median(by_vol[v]))
        volout[v] = dict(n=len(by_vol[v]), ratio=round(m, 4))
        print(f"   {v:<24s} {len(by_vol[v]):>8,d} {m:>9.4f}")

    json.dump(dict(recompute=dict(pixels=tot_px, mismatch=mism, max_diff_nm=maxdiff,
                                  out_of_bounds=bounds_bad),
                   clipping={o: round(100*clip[o]/max(clip_tot[o],1), 2) for o in clip_tot},
                   sphere_cut=out, by_volume=volout, boxes=n_box, faces=n_face),
              open(f"{ROOT}/cache/depth_reliability.json", "w"), indent=1)
    print(f"\n-> cache/depth_reliability.json")


if __name__ == "__main__":
    main()
