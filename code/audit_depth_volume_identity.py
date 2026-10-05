import collections, glob, json, os, sys
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _excluded():
    import json as _j, os as _o
    p = _o.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = _j.load(open(p))
    return set(e["boxes"]), set(e["planes"])

NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main():
    _sb, _sp = _excluded()
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in _sb)
    rng = np.random.default_rng(3); rng.shuffle(raws)

    ratios, ratios_clip = [], []
    shape_frac = []
    shape_by_class = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    n_box = n_face = n_inst = 0
    exact = 0

    for p in raws:
        if n_box >= NBOX:
            break
        box = os.path.basename(p)[:-4]
        faces = sorted(f for f in glob.glob(f"{ROOT}/blockface/{box}/faces/*.npz")
                       if "/._" not in f)
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
        vox = meta3.get("voxel_size_nm")
        Z3, H3, W3 = inst3.shape
        n_box += 1
        if n_box % 20 == 0:
            print(f"  {n_box}/{NBOX} boxes, {n_inst:,} instances", flush=True)

        for fp in faces[:4]:
            try:
                z = np.load(fp, allow_pickle=False)
                own = z["own_occupancy_below_nm"]
                ef = z["inst_face"]
                msk = z["mask"]
                clipped = z["clipped"] if "clipped" in z.files else np.zeros_like(msk)
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

            for i in np.unique(ef[fg]):
                sel = fg & (ef == i)
                if sel.sum() < 40:
                    continue
                swept = float(own[sel].sum()) * px * px
                col = (sub == i) & sel[None, :, :]
                under = float(col.sum()) * px * px * zs
                below = float((sub == i).sum()) * px * px * zs
                if under <= 0 or below <= 0:
                    continue
                r = swept / under
                shape_frac.append(under / below)
                n_inst += 1
                if abs(r - 1.0) < 1e-6:
                    exact += 1
                if clipped[sel].any():
                    ratios_clip.append(r)
                else:
                    ratios.append(r)
                    o = names.get(int(i))
                    if o:
                        by_class[o].append(r)
                        shape_by_class[o].append(under / below)

    print(f"\nboxes {n_box}  faces {n_face}  instances checked {n_inst:,}")

    def rep(name, a):
        if not a:
            print(f"  {name}: none"); return
        a = np.array(a)
        print(f"  {name}")
        print(f"    n {a.size:,}   median {np.median(a):.6f}   mean {a.mean():.6f}")
        print(f"    p1 {np.percentile(a,1):.6f}   p99 {np.percentile(a,99):.6f}   "
              f"min {a.min():.6f}   max {a.max():.6f}")
        for tol in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2):
            print(f"    within {tol:g} of 1.0   {100*np.mean(np.abs(a-1)<=tol):>7.3f}%")

    print("\nIDENTITY: swept ray volume divided by the same voxels counted directly,")
    print("restricted to the column under the exposed profile. Must be 1 exactly.")
    rep("rays that stay inside the box", ratios)
    rep("rays that reach the block bottom (expected to fall short)", ratios_clip)

    if by_class:
        print(f"\n  {'class':<11s}{'n':>9s}{'median':>11s}{'p1':>11s}{'p99':>11s}"
              f"{'exact':>9s}")
        out = {}
        for o in sorted(by_class, key=lambda k: -len(by_class[k])):
            a = np.array(by_class[o])
            if a.size < 30:
                continue
            out[o] = dict(n=int(a.size), median=float(np.median(a)),
                          p1=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)),
                          exact_pct=float(100 * np.mean(np.abs(a - 1) < 1e-9)))
            print(f"  {o:<11s}{a.size:>9,d}{np.median(a):>11.6f}"
                  f"{np.percentile(a,1):>11.6f}{np.percentile(a,99):>11.6f}"
                  f"{100*np.mean(np.abs(a-1)<1e-9):>8.2f}%")
        json.dump(dict(by_class=out,
                       overall=dict(n=len(ratios),
                                    median=float(np.median(ratios)) if ratios else None,
                                    exact_pct=float(100*np.mean(np.abs(np.array(ratios)-1)<1e-9))
                                    if ratios else None),
                       clipped_n=len(ratios_clip), boxes=n_box, faces=n_face),
                  open(f"{ROOT}/cache/depth_volume_identity.json", "w"), indent=1)
    if shape_by_class:
        print("\nSHAPE STATISTIC, not a check: the share of an instance's below-cut volume")
        print("that lies under its own exposed profile. 1 means it does not widen downward.")
        print(f"  {'class':<11s}{'n':>9s}{'median':>10s}{'p25':>10s}{'p75':>10s}")
        sh = {}
        for o in sorted(shape_by_class, key=lambda k: -len(shape_by_class[k])):
            a = np.array(shape_by_class[o])
            if a.size < 30:
                continue
            sh[o] = dict(n=int(a.size), median=float(np.median(a)),
                         p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)))
            print(f"  {o:<11s}{a.size:>9,d}{np.median(a):>10.3f}"
                  f"{np.percentile(a,25):>10.3f}{np.percentile(a,75):>10.3f}")
        d0 = json.load(open(f"{ROOT}/cache/depth_volume_identity.json"))
        d0["shape_under_silhouette"] = sh
        json.dump(d0, open(f"{ROOT}/cache/depth_volume_identity.json", "w"), indent=1)

    print(f"\n-> cache/depth_volume_identity.json")
    print("\nreading")
    print("  a departure from 1.0 means the rays traversed different material than the "
          "labels contain")
    print("  the identity carries no approximation and no fitted constant, so the tolerance "
          "should be floating-point accumulation and nothing more")


if __name__ == "__main__":
    main()
