#!/usr/bin/env python3
"""Can the per-pixel depth be trusted, and where can it not?

Four questions, none of which needs a model, asked in the order that matters.

1. Is the shipped array what it claims? Recompute `depth_below_nm` from the raw sub-volume with
   an independent implementation and compare pixel for pixel. A stored array nobody has
   regenerated is an assertion.
2. How much of it is a lower bound rather than a measurement? A ray that reaches the bottom of
   the box is marked `clipped` and its depth is only a floor. The share must be quoted per
   class, because it is the part a user must exclude or censor.
3. Does it obey a law it was never fitted to? For a uniform random plane through a sphere, the
   mean depth below the cut averaged over the profile, divided by the diameter, is 0.326. The
   constant is checked on synthetic spheres first, because a reference that has never met a
   known answer is not a reference.
4. Does any source volume fail on its own? A pooled pass can hide one bad reconstruction.
"""
import collections, glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"

def _excluded():
    """Boxes and planes that the deposit does not ship, so no check counts them."""
    import json as _j, os as _o
    p = _o.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = _j.load(open(p))
    return set(e["boxes"]), set(e["planes"])

NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 90
# For a sphere cut by a uniform random plane, the mean depth below the cut divided by the
# diameter has two values, and they are not equal. Averaging the per-plane mean over planes
# gives (1/3)(1/2 + ln 2) = 0.39772. Averaging over every (plane, pixel) pair, so a plane
# counts by its profile area, gives 3/8 = 0.37500. Both were derived and then checked against
# a voxel sphere of radius 190, which returns 0.39876 and 0.37632.
#
# The specification carried 0.326, which is neither, and the classes it declared "within 5% of
# the law" are in fact 9 to 11 percent below the pooled constant. The corpus measurement pools
# over pixels, so 3/8 is the reference here.
SPHERE_CONST = 0.375
SPHERE_CONST_PERPLANE = 0.397716


def depth_from_raw(sub_inst, exposed, z_step):
    """Contiguous run of the exposed instance below the cut, in nanometres."""
    match = sub_inst == exposed[None, :, :]
    Z = match.shape[0]
    first_false = np.argmax(~match, axis=0)
    all_true = match.all(axis=0)
    return np.where(all_true, Z, first_false) * z_step


def synthetic_check():
    """The 0.326 constant, on spheres whose answer is known."""
    # the radius the Data Descriptor reports. A smaller ball is faster but lands further from
    # the limit, so the published numbers are reproduced only at this size
    R = 190
    size = 2 * R + 20
    zz, yy, xx = np.ogrid[:size, :size, :size]
    c = size // 2
    ball = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= R * R
    vals, tot_d, tot_a = [], 0.0, 0
    for z in range(c - R + 1, c + R):
        sub = ball[z:]
        face = sub[0]
        if face.sum() < 1:
            continue
        run = depth_from_raw(sub.astype(np.int32), face.astype(np.int32), 1.0)
        d = run[face]
        vals.append(float(d.mean()) / (2 * R))
        tot_d += float(d.sum())
        tot_a += int(d.size)
    m = float(np.mean(vals))
    pooled = tot_d / tot_a / (2 * R)
    ok = abs(m - SPHERE_CONST_PERPLANE) / SPHERE_CONST_PERPLANE < 0.06
    print(f"1. the reference constant, on a synthetic sphere")
    print(f"   measured {m:.6f}   theory {SPHERE_CONST_PERPLANE:.6f} (per-plane)   "
          f"{'pass' if ok else 'FAIL'}   over {len(vals)} planes")
    print(f"   measured {pooled:.6f}   theory 0.375000 (pooled over exposed pixels)")
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
    ratio = collections.defaultdict(list)            # depth/diameter per instance
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

            # clipping, per class
            if clipped is not None:
                for i in np.unique(ef[sel]):
                    o = names.get(int(i))
                    if not o:
                        continue
                    mi = sel & (ef == i)
                    clip_tot[o] += int(mi.sum())
                    clip[o] += int(clipped[mi].sum())

            # depth over diameter, per instance, excluding clipped rays
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
