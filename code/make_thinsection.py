import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard as _g
from render_sem import add_detector_noise, norm01, save_png


def section_stats(inst, cls, z0, z1, voxel_z_nm):
    slab_i = inst[z0:z1]
    solid_slab = slab_i > 0
    mask = solid_slab.any(axis=0)

    slab_thickness = solid_slab.sum(axis=0) * voxel_z_nm
    total_thickness = (inst > 0).sum(axis=0) * voxel_z_nm

    above = (inst[:z0] > 0).sum(axis=0) * voxel_z_nm if z0 > 0 else np.zeros(mask.shape, np.float32)
    below = (inst[z1:] > 0).sum(axis=0) * voxel_z_nm if z1 < inst.shape[0] else \
        np.zeros(mask.shape, np.float32)

    flat = slab_i.reshape(slab_i.shape[0], -1)
    dom = np.zeros(flat.shape[1], np.int32)
    for col in range(0, flat.shape[1], 65536):
        chunk = flat[:, col:col + 65536]
        nz = chunk.max(axis=0)
        dom[col:col + 65536] = nz
    inst_section = dom.reshape(mask.shape).astype(np.int32)
    cls_section = (cls[z0:z1].max(axis=0).astype(np.uint8) if cls is not None else None)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total_thickness > 0, slab_thickness / total_thickness, 0.0)

    return dict(mask=mask,
                inst_section=inst_section,
                cls_section=cls_section,
                slab_thickness=np.where(mask, slab_thickness, 0).astype(np.float32),
                total_thickness=np.where(mask, total_thickness, 0).astype(np.float32),
                extent_above=np.where(mask, above, 0).astype(np.float32),
                extent_below=np.where(mask, below, 0).astype(np.float32),
                section_fraction=np.where(mask, frac, 0).astype(np.float32),
                clipped=mask & ((inst[0] > 0) | (inst[-1] > 0)))


def transmission_image(em, z0, z1, rng, psf_sigma=0.9, electrons=400.0,
                       line_gain=0.01, line_offset=0.01, charging=0.02):
    slab = em[z0:z1].astype(np.float32) / 255.0
    tau = slab.sum(axis=0) / max(z1 - z0, 1)

    lo, hi = np.percentile(tau, [1.0, 99.0])
    tau_n = np.clip((tau - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = np.exp(-1.8 * tau_n)
    return add_detector_noise(norm01(img), rng, psf_sigma=psf_sigma, electrons=electrons,
                              line_gain=line_gain, line_offset=line_offset, charging=charging)


def instance_records(st, voxel, classes, vol_table):
    px_area = float(voxel[1] * voxel[2])
    sec = st["inst_section"]
    out = {}
    for sid in np.unique(sec):
        if sid == 0:
            continue
        sel = sec == sid
        n = int(sel.sum())
        if n < 16:
            continue
        ys, xs = np.nonzero(sel)
        area = n * px_area
        rec = dict(
            instance=int(sid), organelle=classes.get(int(sid)),
            profile_pixels=n, profile_area_nm2=area,
            profile_feret_nm=float(max((ys.max() - ys.min() + 1) * voxel[1],
                                       (xs.max() - xs.min() + 1) * voxel[2])),
            slab_thickness_nm_median=float(np.median(st["slab_thickness"][sel])),
            total_thickness_nm_median=float(np.median(st["total_thickness"][sel])),
            extent_above_nm_median=float(np.median(st["extent_above"][sel])),
            extent_below_nm_median=float(np.median(st["extent_below"][sel])),
            section_fraction_median=float(np.median(st["section_fraction"][sel])),
            touches_border=bool(ys.min() == 0 or xs.min() == 0
                                or ys.max() == sec.shape[0] - 1 or xs.max() == sec.shape[1] - 1),
            clipped=bool(st["clipped"][sel].any()),
            volume_nm3=vol_table.get(str(int(sid)), {}).get("volume_nm3"))
        rec["volume_from_profile_nm3"] = float((area ** 1.5) * (4.0 / 3.0) / np.sqrt(np.pi))
        v = rec["volume_nm3"]
        rec["volume_2d_error"] = (abs(rec["volume_from_profile_nm3"] - v) / v) if v else None
        out[str(int(sid))] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", type=int, default=12)
    ap.add_argument("--thickness-nm", type=float, default=70.0,
                    help="section thickness; 50-90 nm is the ordinary range for TEM")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--min-section", type=float, default=0.02)
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
    nz = max(1, int(round(a.thickness_nm / voxel[0])))
    if nz >= Z:
        print(f"box is only {Z * voxel[0]:.0f} nm deep, thinner than the section", flush=True)
        return
    classes = {int(k): v.get("organelle") for k, v in meta.get("instances", {}).items()}
    vol_table = meta.get("instances", {})

    out = Path(a.out)
    (out / "sections").mkdir(parents=True, exist_ok=True)
    if a.png:
        (out / "preview").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    starts = sorted(set(np.linspace(0, Z - nz - 1, a.sections).round().astype(int).tolist()))
    index, kept = [], 0
    for z0 in starts:
        z1 = z0 + nz
        st = section_stats(inst, cls, z0, z1, float(voxel[0]))
        frac = float(st["mask"].mean())
        if frac < a.min_section:
            print(f"  reject z={z0} occupied={frac:.4f}", flush=True)
            continue
        img = transmission_image(em, z0, z1, rng)
        H, W = img.shape
        if a.crop_size and (H > a.crop_size or W > a.crop_size):
            y0, x0 = (H - a.crop_size) // 2, (W - a.crop_size) // 2
            sl = (slice(y0, y0 + a.crop_size), slice(x0, x0 + a.crop_size))
            img = img[sl]
            st = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
                  for k, v in st.items()}
        recs = instance_records(st, voxel, classes, vol_table)
        name = f"{Path(a.npz).stem}_s{z0:04d}"
        np.savez_compressed(
            out / "sections" / f"{name}.npz",
            image=img.astype(np.float32),
            slab_thickness_nm=st["slab_thickness"],
            total_thickness_nm=st["total_thickness"],
            extent_above_nm=st["extent_above"], extent_below_nm=st["extent_below"],
            section_fraction=st["section_fraction"],
            inst_section=st["inst_section"],
            cls_section=st["cls_section"] if st["cls_section"] is not None else np.zeros(1, np.uint8),
            mask=st["mask"], clipped=st["clipped"],
            meta=json.dumps(dict(
                dataset=meta.get("dataset"), source_box=Path(a.npz).stem,
                geometry="thin-section", real_image=False,
                z_start=int(z0), z_end=int(z1),
                section_thickness_nm=float(nz * voxel[0]),
                pixel_size_nm=float(voxel[1]),
                classes=meta.get("classes", []), occupied_fraction=frac,
                n_instances=len(recs), instances=recs)))
        if a.png:
            save_png(out / "preview" / f"{name}_tem.png", img)
            save_png(out / "preview" / f"{name}_total.png",
                     norm01(st["total_thickness"], st["mask"]))
            save_png(out / "preview" / f"{name}_frac.png",
                     norm01(st["section_fraction"], st["mask"]))
        index.append(dict(name=name, z0=int(z0), occupied=frac, n_instances=len(recs)))
        kept += 1
        print(f"  z={z0:5d} occupied={frac:.3f} instances={len(recs)}", flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{kept} thin sections at {nz * voxel[0]:.0f} nm -> {out}", flush=True)


if __name__ == "__main__":
    main()
