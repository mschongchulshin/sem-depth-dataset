import collections, glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"
NBOX = int(sys.argv[1]) if len(sys.argv) > 1 else 110
W = 8
MAXRAY = 30000


def main():
    _e = os.environ.get("DEPOSIT_EXCLUSIONS")
    _sb, _sp = (lambda d: (set(d["boxes"]), set(d["planes"])))(json.load(open(_e))) if _e else (set(), set())
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in _sb)
    rng = np.random.default_rng(23); rng.shuffle(raws)

    prof = collections.defaultdict(lambda: np.zeros(2 * W + 1))
    cnt = collections.Counter()
    offs = collections.defaultdict(list)
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
                if not {"inst", "em"} <= set(d.files):
                    continue
                inst3, em3 = d["inst"], d["em"]
                meta3 = json.loads(str(d["meta"]))
        except Exception:
            continue
        if inst3.ndim != 3 or em3.shape != inst3.shape:
            continue
        names = {int(k): v.get("organelle") for k, v in meta3.get("instances", {}).items()}
        Z3, H3, W3 = inst3.shape
        n_box += 1
        if n_box % 15 == 0:
            print(f"  {n_box}/{NBOX}  rays {sum(cnt.values()):,}", flush=True)

        for fp in faces[:3]:
            try:
                z = np.load(fp, allow_pickle=False)
                dep = z["depth_below_nm"]
                ef, msk = z["inst_face"], z["mask"]
                clipped = z["clipped"] if "clipped" in z.files else np.zeros_like(msk)
                m = json.loads(str(z["meta"]))
            except Exception:
                continue
            zi = int(os.path.basename(fp).rsplit("_z", 1)[1].split(".")[0])
            zs = float(m.get("z_step_nm", 0) or 0)
            if zs <= 0 or zi >= Z3:
                continue
            H, Wd = dep.shape
            y0 = (H3 - H) // 2 if H3 >= H else 0
            x0 = (W3 - Wd) // 2 if W3 >= Wd else 0
            if inst3[zi:, y0:y0 + H, x0:x0 + Wd].shape[1:] != dep.shape:
                continue
            n_face += 1

            k = np.rint(np.nan_to_num(dep, nan=0.0) / zs).astype(np.int32)
            good = msk & (ef > 0) & ~clipped & (k > 0)
            good &= (zi + k - W >= 0) & (zi + k + W < Z3)
            ys, xs = np.nonzero(good)
            if ys.size == 0:
                continue
            if ys.size > MAXRAY:
                sel = np.random.default_rng(zi).choice(ys.size, MAXRAY, replace=False)
                ys, xs = ys[sel], xs[sel]
            kk = k[ys, xs]
            ids = ef[ys, xs]

            off = np.arange(-W, W + 1)
            zz = zi + kk[:, None] + off[None, :]
            win = em3[zz, (ys + y0)[:, None], (xs + x0)[:, None]].astype(np.float32)
            mu = win.mean(axis=1, keepdims=True)
            sd = win.std(axis=1, keepdims=True)
            ok = sd[:, 0] > 1e-3
            if not ok.any():
                continue
            wn = (win[ok] - mu[ok]) / sd[ok]
            idok = ids[ok]

            g = np.abs(np.diff(wn, axis=1))
            arg = g.argmax(axis=1) - W

            for o in np.unique([names.get(int(i)) for i in np.unique(idok)]):
                if not o:
                    continue
                sel2 = np.array([names.get(int(i)) == o for i in idok])
                if sel2.sum() < 20:
                    continue
                prof[o] += wn[sel2].sum(axis=0)
                cnt[o] += int(sel2.sum())
                offs[o].append(arg[sel2])

    print(f"\nboxes {n_box}  faces {n_face}  rays {sum(cnt.values()):,}")
    print(f"window +-{W} z steps around the label's declared lower boundary\n")

    out = {}
    print(f"  {'class':<11s}{'rays':>11s}{'peak offset':>12s}{'per-ray median':>15s}"
          f"{'p25':>7s}{'p75':>7s}{'|off|<=1':>10s}")
    for o in sorted(cnt, key=lambda k: -cnt[k]):
        if cnt[o] < 20000:
            continue
        m = prof[o] / cnt[o]
        g = np.abs(np.diff(m))
        peak = int(g.argmax()) - W
        a = np.concatenate(offs[o])
        out[o] = dict(rays=int(cnt[o]), peak_offset=peak,
                      median_offset=float(np.median(a)),
                      p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                      within1=float(np.mean(np.abs(a) <= 1)),
                      mean_profile=[round(float(v), 4) for v in m])
        print(f"  {o:<11s}{cnt[o]:>11,d}{peak:>12d}{np.median(a):>15.1f}"
              f"{np.percentile(a,25):>7.0f}{np.percentile(a,75):>7.0f}"
              f"{100*np.mean(np.abs(a)<=1):>9.1f}%")

    print("\nclass-mean normalised intensity, aligned at the label boundary")
    print("offset 0 is the last step the label calls interior")
    hdr = "  " + "".join(f"{v:>6d}" for v in range(-W, W + 1))
    print(f"  {'class':<11s}" + hdr[2:])
    for o in sorted(out, key=lambda k: -out[k]["rays"])[:8]:
        m = out[o]["mean_profile"]
        print(f"  {o:<11s}" + "".join(f"{v:>6.2f}" for v in m))

    json.dump(dict(window=W, by_class=out, boxes=n_box, faces=n_face,
                   rays=int(sum(cnt.values()))),
              open(f"{ROOT}/cache/label_vs_image.json", "w"), indent=1)
    print(f"\n-> cache/label_vs_image.json")
    print("\nreading")
    print("  a peak offset of 0 puts the label boundary at the same place as the image "
          "transition")
    print("  anything else is a systematic displacement, in steps, that a user can correct "
          "for")
    print("  a per-ray median of 0 with a wide interquartile range means the aggregate is "
          "aligned and single rays are noisy")


if __name__ == "__main__":
    main()
