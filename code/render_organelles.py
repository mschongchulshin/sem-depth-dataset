import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import morphometry as mm
from render_sem import (add_detector_noise, apply_instrument_limits, column_stats,
                        detector_dirs, kanaya_okayama_range_nm, material_term, norm01,
                        normals_from_depth, rotate_view, save_png, screen_space_ao,
                        screen_space_shadow, se_yield, to_isotropic)


def render_instance_view(inst_iso, field_smooth, voxel_nm, em_iso, azimuth, elevation, rng,
                         bse_elevation=35.0, noise=True, cls_iso=None, dr=None,
                         min_valid=None):
    rot_inst = rotate_view(inst_iso, azimuth, elevation, order=0)
    binf = (rot_inst > 0).astype(np.float32)
    field = ndimage.gaussian_filter(binf, field_smooth) if field_smooth > 0 else binf

    st = column_stats(field, voxel_nm)
    depth, mask, hit = st["depth_front"], st["mask"], st["hit"]

    if min_valid is not None and float(mask.mean()) < min_valid:
        return None, None

    rot_cls = rotate_view(cls_iso, azimuth, elevation, order=0) if cls_iso is not None else None
    em_r = rotate_view(em_iso, azimuth, elevation, order=1) if em_iso is not None else None
    normal = normals_from_depth(depth, voxel_nm, mask)
    ao = screen_space_ao(depth, voxel_nm, strength=(dr or {}).get("ao_strength", 1.0))
    mat = material_term(em_r, hit, mask)

    se = se_yield(normal) * ao * mat
    bse = []
    for L in detector_dirs(bse_elevation):
        lam = np.clip(normal[0] * L[0] + normal[1] * L[1] + normal[2] * L[2], 0, None)
        bse.append(lam * screen_space_shadow(depth, voxel_nm, L) * mat * (0.35 + 0.65 * ao))
    bse = np.stack(bse, 0)

    if noise:
        nk = dict(psf_sigma=(dr or {}).get("psf_sigma", 0.8),
                  line_gain=(dr or {}).get("line_gain", 0.02),
                  line_offset=(dr or {}).get("line_offset", 0.015),
                  charging=(dr or {}).get("charging", 0.05))
        se = add_detector_noise(se, rng, electrons=(dr or {}).get("electrons", 180.0), **nk)
        bse = np.stack([add_detector_noise(
            b, rng, electrons=(dr or {}).get("electrons", 180.0) * 0.5, **nk) for b in bse], 0)

    se = norm01(se, mask)
    bse = np.stack([norm01(b, mask) for b in bse], 0)
    se[~mask] = 0.0
    bse[:, ~mask] = 0.0

    H, W = mask.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    solid = rot_inst > 0
    hard_hit = np.argmax(solid, axis=0)
    hard_mask = solid.any(axis=0)
    inst_front = np.where(hard_mask, rot_inst[np.clip(hard_hit, 0, rot_inst.shape[0] - 1),
                                              yy, xx], 0)

    st.update(se=se, bse=bse, normal=normal, inst_front=inst_front.astype(np.uint32))
    if rot_cls is not None:
        cls_front = np.where(hard_mask, rot_cls[np.clip(hard_hit, 0, rot_cls.shape[0] - 1),
                                                yy, xx], 0)
        st["cls_front"] = cls_front.astype(np.uint8)
        st["class_depth_count"] = np.count_nonzero(
            np.stack([(rot_cls == c).any(axis=0) for c in np.unique(rot_cls) if c > 0], 0), 0
        ).astype(np.uint8) if rot_cls.max() > 0 else np.zeros(mask.shape, np.uint8)
    return st, rot_inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="from fetch_wholecell.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--elev-range", type=float, nargs=2, default=[-30.0, 30.0])
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--min-instance-px", type=int, default=64,
                    help="silhouettes smaller than this are not measurable, so are skipped")
    ap.add_argument("--min-valid", type=float, default=0.02)
    ap.add_argument("--min-box-instances", type=int, default=4,
                    help="skip the whole box if fewer instances than this can ever be scored")
    ap.add_argument("--min-usable", type=int, default=1,
                    help="reject a view with fewer untruncated instances than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--max-instances-3d", type=int, default=None)
    ap.add_argument("--kv", type=float, default=None,
                    help="accelerating voltage; sets the interaction-volume depth blur")
    ap.add_argument("--se1-fraction", type=float, default=0.5,
                    help="share of signal from the sharp SE1 component, the rest is blurred")
    ap.add_argument("--stereo-tilt-deg", type=float, default=None,
                    help="total stereo tilt; sets height noise pixel/(2 sin(tilt/2))")
    ap.add_argument("--domain-random", action="store_true",
                    help="resample the imaging parameters per view over a wide range")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    voxel = np.array(meta["voxel_size_nm"], float)
    classes = meta.get("classes", [])

    s = max(1, a.downsample)
    inst = np.ascontiguousarray(d["inst"][::s, ::s, ::s])
    cls = np.ascontiguousarray(d["cls"][::s, ::s, ::s]) if "cls" in d.files else None
    em = np.ascontiguousarray(d["em"][::s, ::s, ::s]) if "em" in d.files else None
    d.close()
    voxel = voxel * s

    print(f"instance volume {inst.shape} at {voxel} nm, "
          f"{len(np.unique(inst)) - 1} instances", flush=True)
    table3d = mm.table_from_volume(inst, voxel, classes=classes, cls=cls,
                                  max_instances=a.max_instances_3d)
    print(f"3D morphometry for {len(table3d)} instances", flush=True)

    scorable = sum(1 for r in table3d.values()
                   if r.get("volume_nm3") and not r.get("touches_border"))
    if scorable < a.min_box_instances:
        print(f"skip: only {scorable} scorable instances in this box "
              f"(need {a.min_box_instances})", flush=True)
        (Path(a.out) / "views").mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "index.json").write_text(json.dumps([]))
        return

    inst_iso, vnm = to_isotropic(inst, voxel, order=0)
    cls_iso = to_isotropic(cls, voxel, order=0)[0] if cls is not None else None
    if cls_iso is not None and cls_iso.shape != inst_iso.shape:
        cls_iso = cls_iso[:inst_iso.shape[0], :inst_iso.shape[1], :inst_iso.shape[2]]
    em_iso = to_isotropic(em.astype(np.float32), voxel, order=1)[0] if em is not None else None
    if em_iso is not None:
        em_iso = em_iso[:inst_iso.shape[0], :inst_iso.shape[1], :inst_iso.shape[2]]
    print(f"isotropic {inst_iso.shape} at {vnm:.2f} nm/voxel", flush=True)

    outdir = Path(a.out)
    (outdir / "views").mkdir(parents=True, exist_ok=True)
    if a.png:
        (outdir / "preview").mkdir(parents=True, exist_ok=True)

    kept, tried, index = 0, 0, []
    while kept < a.views and tried < a.views * 4:
        tried += 1
        az = float(rng.uniform(0, 360))
        el = float(rng.uniform(*a.elev_range))
        dr = None
        if a.domain_random:
            dr = dict(kv=float(rng.uniform(0.7, 3.0)),
                      se1_fraction=float(rng.uniform(0.2, 0.9)),
                      tilt_deg=float(rng.uniform(4.0, 20.0)),
                      bse_elevation=float(rng.uniform(20.0, 55.0)),
                      psf_sigma=float(rng.uniform(0.4, 2.0)),
                      electrons=float(rng.uniform(150.0, 550.0)),
                      line_gain=float(rng.uniform(0.0, 0.06)),
                      line_offset=float(rng.uniform(0.0, 0.05)),
                      charging=float(rng.uniform(0.0, 0.15)),
                      ao_strength=float(rng.uniform(0.4, 1.4)))
        st, rot_inst = render_instance_view(inst_iso, a.smooth, vnm, em_iso, az, el, rng,
                                            noise=not a.no_noise, cls_iso=cls_iso,
                                            bse_elevation=(dr or {}).get("bse_elevation", 35.0),
                                            dr=dr, min_valid=a.min_valid)
        if st is None:
            continue

        H, W = st["mask"].shape
        if a.crop_size and (H > a.crop_size or W > a.crop_size):
            m = st["mask"]
            if m.any():
                ys_i, xs_i = np.nonzero(m)
                cy, cx = int(ys_i.mean()), int(xs_i.mean())
            else:
                cy, cx = H // 2, W // 2
            y0 = int(np.clip(cy - a.crop_size // 2, 0, max(H - a.crop_size, 0)))
            x0 = int(np.clip(cx - a.crop_size // 2, 0, max(W - a.crop_size, 0)))
            ys, xs = slice(y0, y0 + a.crop_size), slice(x0, x0 + a.crop_size)
            for k in ["se", "depth_front", "thickness", "envelope", "mask", "clipped",
                      "inst_front", "hit", "cls_front", "class_depth_count"]:
                if k in st:
                    st[k] = st[k][ys, xs]
            for k in ["bse", "normal"]:
                st[k] = st[k][:, ys, xs]
            rot_inst = rot_inst[:, ys, xs]

        valid = float(st["mask"].mean())

        present = np.unique(rot_inst)
        present = present[present > 0]
        rows = {}
        for i in present:
            sil = (rot_inst == i).any(axis=0)
            npx = int(sil.sum())
            if npx < a.min_instance_px:
                continue
            m3 = table3d.get(int(i))
            if m3 is None:
                continue
            m2 = mm.morphometry_2d(sil, vnm)
            m3v = dict(m3)
            m3v["feret_nm"] = mm.feret_diameter(rot_inst == i, (vnm, vnm, vnm))
            rec = mm.combine(m3v, m2, pixel_nm=vnm)
            rec["feret_native_nm"] = m3.get("feret_nm")
            rec["visible_pixels"] = int((st["inst_front"] == i).sum())
            rec["occluded_fraction"] = float(1.0 - rec["visible_pixels"] / max(npx, 1))
            rows[int(i)] = rec

        inst_limits = None
        if a.kv is not None or a.stereo_tilt_deg is not None or dr:
            d_lim, t_lim, inst_limits = apply_instrument_limits(
                st["depth_front"], st["thickness"], st["mask"], vnm, rng,
                kv=(dr or {}).get("kv", a.kv),
                se1_fraction=(dr or {}).get("se1_fraction", a.se1_fraction),
                tilt_deg=(dr or {}).get("tilt_deg", a.stereo_tilt_deg))
            st["depth_limited"], st["thickness_limited"] = d_lim, t_lim

        usable = [r for r in rows.values() if not r.get("touches_border")]
        vol_nm3 = float(st["thickness"].sum()) * vnm * vnm
        clutter = None
        if "cls_front" in st:
            cf = st["cls_front"]
            vis = np.bincount(cf[cf > 0], minlength=len(classes) + 1)
            clutter = dict(
                visible_class_pixels={classes[c - 1]: int(vis[c])
                                      for c in range(1, min(len(vis), len(classes) + 1))
                                      if vis[c] > 0},
                mean_classes_per_ray=float(st["class_depth_count"][st["mask"]].mean()),
                max_classes_per_ray=int(st["class_depth_count"].max()))
        if len(usable) < a.min_usable:
            print(f"  reject az={az:6.1f} el={el:+6.1f} usable_instances={len(usable)}",
                  flush=True)
            continue

        name = f"{Path(a.npz).stem}_az{az:06.2f}_el{el:+06.2f}"
        vm = dict(dataset=meta.get("dataset"), classes=classes,
                  azimuth_deg=az, elevation_deg=el, pixel_size_nm=float(vnm),
                  depth_units="nm", camera="orthographic", valid_fraction=valid,
                  n_instances=len(rows), n_instances_untruncated=len(usable),
                  instrument_limits=inst_limits, clutter=clutter, domain_random=dr,
                  scene_volume_nm3=vol_nm3,
                  instances={str(k): v for k, v in rows.items()})
        np.savez_compressed(
            outdir / "views" / f"{name}.npz", meta=json.dumps(vm),
            se=st["se"], bse=st["bse"], depth_nm=st["depth_front"],
            thickness_nm=st["thickness"], envelope_nm=st["envelope"], normal=st["normal"],
            mask=st["mask"].astype(np.uint8), inst_front=st["inst_front"],
            **({"cls_front": st["cls_front"],
                "class_depth_count": st["class_depth_count"]} if "cls_front" in st else {}),
            **({"depth_limited_nm": st["depth_limited"],
                "thickness_limited_nm": st["thickness_limited"]} if inst_limits else {}))

        if a.png:
            save_png(outdir / "preview" / f"{name}_se.png", st["se"])
            dn = np.nan_to_num(st["depth_front"], nan=np.nanmax(st["depth_front"]))
            save_png(outdir / "preview" / f"{name}_depth.png", 1.0 - norm01(dn, st["mask"]))
            save_png(outdir / "preview" / f"{name}_thick.png",
                     norm01(st["thickness"], st["mask"]))
            ids = st["inst_front"]
            save_png(outdir / "preview" / f"{name}_inst.png",
                     (ids % 251) / 251.0 * (ids > 0))

        if usable:
            fs = [r["foreshortening"] for r in usable if r.get("foreshortening")]
            v2 = [r["volume_2d_error"] for r in usable if r.get("volume_2d_error")]
            print(f"  keep  az={az:6.1f} el={el:+6.1f} valid={valid:.4f} "
                  f"inst={len(rows)} usable={len(usable)} "
                  f"median foreshortening={np.median(fs) if fs else float('nan'):.3f} "
                  f"median 2D-only volume error={np.median(v2) if v2 else float('nan'):.2f}",
                  flush=True)
        else:
            print(f"  keep  az={az:6.1f} el={el:+6.1f} valid={valid:.4f} inst={len(rows)} "
                  f"(none untruncated)", flush=True)
        index.append({k: v for k, v in vm.items() if k != "instances"})
        kept += 1

    (outdir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {kept} views to {outdir / 'views'}", flush=True)


if __name__ == "__main__":
    main()
