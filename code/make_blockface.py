import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard as _g
from render_blockface import blockface_stats
from render_sem import norm01, save_png


def stretch(img, mask, lo=0.5, hi=99.5):
    a = img.astype(np.float32)
    ref = a[mask] if mask.any() and mask.sum() > 256 else a[(a > a.min()) & (a < a.max())]
    if ref.size < 64:
        ref = a
    p0, p1 = np.percentile(ref, [lo, hi])
    out = np.clip((a - p0) / max(p1 - p0, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def instance_face_records(inst_sub, cls_sub, st, voxel_zyx, classes, vol_table):
    px_area = float(voxel_zyx[1] * voxel_zyx[2])
    face = st["inst_face"]
    out = {}
    for sid in np.unique(face):
        if sid == 0:
            continue
        sel = face == sid
        n = int(sel.sum())
        if n < 16:
            continue
        ys, xs = np.nonzero(sel)
        db = st["depth_below"][sel]
        tb = st["thickness_below"][sel]
        area = n * px_area
        rec = dict(
            instance=int(sid),
            organelle=classes.get(int(sid)),
            face_pixels=n,
            face_area_nm2=area,
            face_bbox=[int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
            face_feret_nm=float(max((ys.max() - ys.min() + 1) * voxel_zyx[1],
                                    (xs.max() - xs.min() + 1) * voxel_zyx[2])),
            depth_below_nm_median=float(np.nanmedian(db)),
            depth_below_nm_max=float(np.nanmax(db)),
            thickness_below_nm_median=float(np.median(tb)),
            touches_face_border=bool(ys.min() == 0 or xs.min() == 0
                                     or ys.max() == face.shape[0] - 1
                                     or xs.max() == face.shape[1] - 1),
            clipped_at_block_bottom=bool(st["clipped"][sel].any()),
            volume_nm3=vol_table.get(str(int(sid)), {}).get("volume_nm3"),
        )
        rec["volume_from_face_nm3"] = float((area ** 1.5) * (4.0 / 3.0) / np.sqrt(np.pi))
        v = rec["volume_nm3"]
        rec["volume_2d_error"] = (abs(rec["volume_from_face_nm3"] - v) / v) if v else None
        out[str(int(sid))] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="a box from fetch_wholecell.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", type=int, default=12, help="faces to take from this box")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="fraction of the block left below the deepest face, so depth is "
                         "measurable rather than clipped")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--min-face", type=float, default=0.01,
                    help="reject a face with less exposed material than this")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _g.arm()

    d = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    voxel = np.array(meta["voxel_size_nm"], float)
    if "em" not in d.files:
        print("no raw image in this box; only the surface subset can use it", flush=True)
        return
    inst, cls, em = d["inst"], d["cls"], d["em"]
    d.close()
    if em.shape != inst.shape:
        print(f"em {em.shape} does not match labels {inst.shape}; this box cannot be cut",
              flush=True)
        return
    Z = inst.shape[0]
    classes = {int(k): v.get("organelle") for k, v in meta.get("instances", {}).items()}
    vol_table = meta.get("instances", {})

    out = Path(a.out)
    (out / "faces").mkdir(parents=True, exist_ok=True)
    if a.png:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    zmax = int(Z * (1.0 - a.margin))
    rng = np.random.default_rng(a.seed)
    picks = sorted(set(np.linspace(0, max(zmax - 1, 0), a.slices).round().astype(int).tolist()))
    kept, index = 0, []
    for z in picks:
        sub_i = inst[z:]
        sub_c = cls[z:] if cls is not None else None
        st = blockface_stats(sub_i, sub_c, float(voxel[0]))
        frac = float(st["mask"].mean())
        if frac < a.min_face:
            print(f"  reject z={z} exposed={frac:.4f}", flush=True)
            continue
        img = em[z]
        H, W = img.shape
        if a.crop_size and (H > a.crop_size or W > a.crop_size):
            y0, x0 = (H - a.crop_size) // 2, (W - a.crop_size) // 2
            sl = (slice(y0, y0 + a.crop_size), slice(x0, x0 + a.crop_size))
            img = img[sl]
            st = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                  for k, v in st.items()}
            sub_i = sub_i[(slice(None),) + sl]
            sub_c = sub_c[(slice(None),) + sl] if sub_c is not None else None

        img = stretch(img, st["mask"])

        recs = instance_face_records(sub_i, sub_c, st, voxel, classes, vol_table)
        name = f"{Path(a.npz).stem}_z{z:04d}"
        np.savez_compressed(
            out / "faces" / f"{name}.npz",
            em=img.astype(np.uint8),
            depth_below_nm=st["depth_below"],
            thickness_below_nm=st["thickness_below"],
            inst_face=st["inst_face"],
            cls_face=st["cls_face"] if st["cls_face"] is not None else np.zeros(1, np.uint8),
            mask=st["mask"],
            clipped=st["clipped"],
            meta=json.dumps(dict(
                dataset=meta.get("dataset"), source_box=Path(a.npz).stem, z_index=int(z),
                geometry="block-face", real_image=True,
                pixel_size_nm=float(voxel[1]), z_step_nm=float(voxel[0]),
                block_below_nm=float((inst.shape[0] - z) * voxel[0]),
                classes=meta.get("classes", []), exposed_fraction=frac,
                n_instances=len(recs), instances=recs)))
        if a.png:
            save_png(out / "preview" / f"{name}_em.png", norm01(img.astype(np.float32)))
            save_png(out / "preview" / f"{name}_depth.png",
                     norm01(np.nan_to_num(st["depth_below"]), st["mask"]))
            save_png(out / "preview" / f"{name}_thick.png",
                     norm01(st["thickness_below"], st["mask"]))
        index.append(dict(name=name, z=int(z), exposed=frac, n_instances=len(recs)))
        kept += 1
        print(f"  z={z:5d} exposed={frac:.3f} instances={len(recs)}", flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{kept} real block-face images -> {out}", flush=True)


if __name__ == "__main__":
    main()
