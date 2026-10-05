import collections, glob, json, os, sys
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def main():
    _e = os.environ.get("DEPOSIT_EXCLUSIONS")
    _sb, _sp = (lambda d: (set(d["boxes"]), set(d["planes"])))(json.load(open(_e))) if _e else (set(), set())
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in _sb)
    rng = np.random.default_rng(11); rng.shuffle(raws)

    thick_r = []
    clip_tp = clip_fp = clip_fn = clip_tn = 0
    same_run = tot_fg = 0
    gap_by_class = collections.defaultdict(lambda: [0, 0])
    n_box = n_face = 0

    for p in raws:
        if n_box >= NBOX:
            break
        box = os.path.basename(p)[:-4]
        faces = sorted(f for f in glob.glob(f"{ROOT}/blockface/{box}/faces/*.npz")
                       if "/._" not in f and os.path.basename(f)[:-4] not in _sp)
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
        if n_box % 20 == 0:
            print(f"  {n_box}/{NBOX}", flush=True)

        for fp in faces[:4]:
            try:
                z = np.load(fp, allow_pickle=False)
                own, dep = z["own_occupancy_below_nm"], z["depth_below_nm"]
                thk = z["thickness_below_nm"]
                ef, msk = z["inst_face"], z["mask"]
                clipped = z["clipped"] if "clipped" in z.files else None
                m = json.loads(str(z["meta"]))
            except Exception:
                continue
            zi = int(os.path.basename(fp).rsplit("_z", 1)[1].split(".")[0])
            if zi >= Z3:
                continue
            px = float(m.get("pixel_size_nm", 0) or 0)
            zs = float(m.get("z_step_nm", 0) or 0)
            if px <= 0 or zs <= 0:
                continue
            H, W = own.shape
            y0 = (H3 - H) // 2 if H3 >= H else 0
            x0 = (W3 - W) // 2 if W3 >= W else 0
            sub = inst3[zi:, y0:y0 + H, x0:x0 + W]
            if sub.shape[1:] != own.shape:
                continue
            n_face += 1
            fg = msk & (ef > 0)
            if not fg.any():
                continue

            swept = float(thk[fg].sum()) * px * px
            under = float((sub[:, fg] > 0).sum()) * px * px * zs
            if under > 0:
                thick_r.append(swept / under)

            if clipped is not None:
                match = sub == ef[None, :, :]
                allrun = match.all(axis=0)
                reach = allrun & fg
                c = clipped & fg
                clip_tp += int((c & reach).sum())
                clip_fp += int((c & ~reach).sum())
                clip_fn += int((~c & reach).sum())
                clip_tn += int((~c & ~reach).sum())

            eq = np.isclose(own[fg], dep[fg], rtol=0, atol=zs * 0.25)
            same_run += int(eq.sum()); tot_fg += int(fg.sum())
            for i in np.unique(ef[fg]):
                o = names.get(int(i))
                if not o:
                    continue
                s2 = fg & (ef == i)
                g = gap_by_class[o]
                g[0] += int(np.isclose(own[s2], dep[s2], rtol=0, atol=zs * 0.25).sum())
                g[1] += int(s2.sum())

    print(f"\nboxes {n_box}  faces {n_face}")

    print("\n1. thickness identity, union of all labelled structures")
    print("   integrating thickness_below over the exposed profile must count every labelled")
    print("   voxel below the cut in that column, once each")
    if thick_r:
        a = np.array(thick_r)
        print(f"   planes {a.size:,}   median {np.median(a):.6f}   "
              f"min {a.min():.6f}   max {a.max():.6f}")
        for tol in (1e-9, 1e-6, 1e-4, 1e-2):
            print(f"     within {tol:g} of 1.0   {100*np.mean(np.abs(a-1)<=tol):>7.3f}%")

    print("\n2. the clipped flag against the geometry")
    tot = clip_tp + clip_fp + clip_fn + clip_tn
    if tot:
        print(f"   foreground pixels {tot:,}")
        print(f"     flagged and the run does reach the bottom   {clip_tp:>12,d}")
        print(f"     flagged and it does not                      {clip_fp:>12,d} "
              f"({100*clip_fp/max(tot,1):.4f}%)")
        print(f"     not flagged and it does reach the bottom     {clip_fn:>12,d} "
              f"({100*clip_fn/max(tot,1):.4f}%)")
        print(f"     not flagged and it does not                  {clip_tn:>12,d}")
        agree = 100 * (clip_tp + clip_tn) / tot
        print(f"   agreement {agree:.4f}%")

    print("\n3. how often depth and own occupancy differ, which is where an instance leaves")
    print("   the ray and comes back")
    if tot_fg:
        print(f"   foreground pixels {tot_fg:,}   equal {100*same_run/tot_fg:.2f}%")
        print(f"   {'class':<11s}{'pixels':>12s}{'equal':>9s}")
        out = {}
        for o in sorted(gap_by_class, key=lambda k: -gap_by_class[k][1]):
            e, n = gap_by_class[o]
            if n < 5000:
                continue
            out[o] = round(100 * e / n, 2)
            print(f"   {o:<11s}{n:>12,d}{100*e/n:>8.2f}%")
        json.dump(dict(thickness_identity=dict(
                          planes=len(thick_r),
                          median=float(np.median(thick_r)) if thick_r else None,
                          within_1e6_pct=float(100*np.mean(np.abs(np.array(thick_r)-1)<=1e-6))
                          if thick_r else None),
                       clipped=dict(tp=clip_tp, fp=clip_fp, fn=clip_fn, tn=clip_tn,
                                    agreement_pct=round(agree, 4) if tot else None),
                       depth_equals_own_pct=out, boxes=n_box, faces=n_face),
                  open(f"{ROOT}/cache/remaining_arrays.json", "w"), indent=1)
    print(f"\n-> cache/remaining_arrays.json")


if __name__ == "__main__":
    main()
