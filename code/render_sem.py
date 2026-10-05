import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage


def to_isotropic(vol, voxel_zyx, order=1):
    voxel_zyx = np.asarray(voxel_zyx, float)
    target = float(voxel_zyx.min())
    zoom = voxel_zyx / target
    if np.allclose(zoom, 1.0):
        return vol, target
    return ndimage.zoom(vol, zoom, order=order, prefilter=False), target


def rotate_view(vol, azimuth_deg, elevation_deg, order=1):
    out = vol
    if abs(azimuth_deg) > 1e-6:
        out = ndimage.rotate(out, azimuth_deg, axes=(1, 2), order=order,
                             reshape=True, prefilter=False, mode="constant", cval=0)
    if abs(elevation_deg) > 1e-6:
        out = ndimage.rotate(out, elevation_deg, axes=(0, 1), order=order,
                             reshape=True, prefilter=False, mode="constant", cval=0)
    return out


def column_stats(field, voxel_nm, level=0.5):
    solid = field >= level
    mask = solid.any(axis=0)
    Z, H, W = field.shape

    front = np.argmax(solid, axis=0)
    back = (Z - 1) - np.argmax(solid[::-1], axis=0)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    i1 = np.clip(front, 0, Z - 1)
    i0 = np.clip(front - 1, 0, Z - 1)
    f1, f0 = field[i1, yy, xx], field[i0, yy, xx]
    denom = f1 - f0
    frac = np.where(np.abs(denom) > 1e-4, (f1 - level) / np.where(denom == 0, 1, denom), 0.0)
    depth_front = (front - np.clip(frac, 0.0, 1.0)) * voxel_nm

    thickness = solid.sum(axis=0) * voxel_nm
    envelope = (back - front + 1) * voxel_nm

    clipped = mask & ((front == 0) | (back == Z - 1))

    depth_front = np.where(mask, depth_front, np.nan).astype(np.float32)
    return dict(depth_front=depth_front,
                thickness=np.where(mask, thickness, 0.0).astype(np.float32),
                envelope=np.where(mask, envelope, 0.0).astype(np.float32),
                mask=mask, hit=front.astype(np.int32), clipped=clipped)


def normals_from_depth(depth_nm, pixel_nm, mask):
    d = np.array(depth_nm, dtype=np.float32)
    if not mask.all():
        d[~mask] = np.nanmedian(d[mask]) if mask.any() else 0.0
    h = -d
    hy = ndimage.sobel(h, axis=0) / (8.0 * pixel_nm)
    hx = ndimage.sobel(h, axis=1) / (8.0 * pixel_nm)
    n = np.stack([-hx, -hy, np.ones_like(h)], 0)
    n /= np.linalg.norm(n, axis=0, keepdims=True) + 1e-12
    return n.astype(np.float32)


def screen_space_ao(depth_nm, pixel_nm, radius_px=10, n_dirs=12, n_steps=5, strength=1.0):
    finite = np.isfinite(depth_nm)
    fill = float(np.nanmax(depth_nm[finite])) if finite.any() else 0.0
    d = np.nan_to_num(depth_nm, nan=fill)
    occl = np.zeros(d.shape, np.float32)
    for ang in np.linspace(0, 2 * np.pi, n_dirs, endpoint=False):
        dy, dx = np.sin(ang), np.cos(ang)
        best = np.zeros(d.shape, np.float32)
        for s in range(1, n_steps + 1):
            r = radius_px * s / n_steps
            sh = ndimage.shift(d, (dy * r, dx * r), order=1, mode="nearest")
            best = np.maximum(best, (d - sh) / (r * pixel_nm))
        best = np.clip(best, 0, None)
        occl += best / (1.0 + best)
    return np.clip(1.0 - strength * occl / n_dirs, 0.05, 1.0).astype(np.float32)


def screen_space_shadow(depth_nm, pixel_nm, light_dir, radius_px=20, n_steps=10):
    lx, ly, lz = light_dir
    lat = float(np.hypot(lx, ly))
    if lat < 1e-6 or lz <= 0:
        return np.ones(depth_nm.shape, np.float32)
    ux, uy = lx / lat, ly / lat
    tan_l = lz / lat
    d = np.nan_to_num(depth_nm, nan=0.0)
    blocked = np.zeros(d.shape, np.float32)
    for s in range(1, n_steps + 1):
        r = radius_px * s / n_steps
        sh = ndimage.shift(d, (-uy * r, -ux * r), order=1, mode="nearest")
        rise = (d - sh) / (r * pixel_nm)
        blocked = np.maximum(blocked, np.clip((rise - tan_l) / max(tan_l, 1e-3), 0, 1))
    return (1.0 - blocked).astype(np.float32)


def detector_dirs(elevation_deg=35.0, azimuths=(45, 135, 225, 315)):
    el = np.deg2rad(elevation_deg)
    return [np.array([np.cos(np.deg2rad(az)) * np.cos(el),
                      np.sin(np.deg2rad(az)) * np.cos(el),
                      np.sin(el)]) for az in azimuths]


def kanaya_okayama_range_nm(kv, Z=6.0, A=12.0, rho=1.2):
    return 1000.0 * 0.0276 * A * (float(kv) ** 1.67) / ((Z ** 0.89) * rho)


def apply_instrument_limits(depth_nm, thickness_nm, mask, pixel_nm, rng,
                            kv=None, interaction_nm=None, se1_fraction=0.5,
                            stereo_sigma_nm=None, tilt_deg=None):
    if interaction_nm is None and kv is not None:
        interaction_nm = kanaya_okayama_range_nm(kv)
    out_d = np.array(depth_nm, dtype=np.float32)
    out_t = np.array(thickness_nm, dtype=np.float32)

    if interaction_nm:
        sigma_px = float(interaction_nm) / (2.355 * pixel_nm)
        if sigma_px > 0.1:
            fill = np.nanmedian(out_d[mask]) if mask.any() else 0.0
            d_filled = np.where(np.isfinite(out_d), out_d, fill)
            blur_d = ndimage.gaussian_filter(d_filled, sigma_px)
            blur_t = ndimage.gaussian_filter(out_t, sigma_px)
            w = float(np.clip(se1_fraction, 0.0, 1.0))
            out_d = np.where(mask, w * d_filled + (1 - w) * blur_d, np.nan).astype(np.float32)
            out_t = (w * out_t + (1 - w) * blur_t).astype(np.float32)

    if stereo_sigma_nm is None and tilt_deg:
        stereo_sigma_nm = pixel_nm / (2.0 * np.sin(np.deg2rad(float(tilt_deg)) / 2.0))
    if stereo_sigma_nm:
        out_d = out_d + rng.normal(0.0, float(stereo_sigma_nm), out_d.shape).astype(np.float32)

    return out_d, out_t, dict(interaction_nm=interaction_nm, kv=kv,
                              se1_fraction=se1_fraction,
                              stereo_sigma_nm=stereo_sigma_nm, tilt_deg=tilt_deg)


def se_yield(normal, cos_floor=0.15, gain=0.45):
    cos_t = np.clip(normal[2], cos_floor, 1.0)
    raw = 1.0 / cos_t
    return (gain * raw / (1.0 + gain * (raw - 1.0))).astype(np.float32)


def material_term(em_vol, hit_idx, mask, lo=1.0, hi=1.6):
    if em_vol is None:
        return np.ones(hit_idx.shape, np.float32)
    H, W = hit_idx.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    g = em_vol[np.clip(hit_idx, 0, em_vol.shape[0] - 1),
               np.clip(yy, 0, em_vol.shape[1] - 1),
               np.clip(xx, 0, em_vol.shape[2] - 1)].astype(np.float32)
    if mask.any():
        p1, p99 = np.percentile(g[mask], [1, 99])
        g = np.clip((g - p1) / max(p99 - p1, 1e-6), 0, 1)
    else:
        g = np.zeros_like(g)
    return (lo + (hi - lo) * g).astype(np.float32)


def add_detector_noise(img, rng, psf_sigma=0.8, electrons=180.0,
                       line_gain=0.02, line_offset=0.015, charging=0.05):
    out = ndimage.gaussian_filter(img.astype(np.float32), psf_sigma)
    if electrons > 0:
        out = rng.poisson(np.clip(out, 0, None) * electrons) / electrons
    H, W = out.shape
    if line_gain > 0:
        out = out * (1.0 + rng.normal(0, line_gain, (H, 1)))
    if line_offset > 0:
        out = out + rng.normal(0, line_offset, (H, 1))
    if charging > 0:
        sig = max(H, W) / 12.0
        out = out + charging * ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), sig) * sig
    return out.astype(np.float32)


def norm01(x, mask=None):
    v = x[mask] if mask is not None and np.any(mask) else x
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(x, np.float32)
    p1, p99 = np.percentile(v, [0.5, 99.5])
    return np.clip((x - p1) / max(p99 - p1, 1e-6), 0, 1).astype(np.float32)


def render_view(field_iso, voxel_nm, em_iso, azimuth, elevation, rng,
                bse_elevation=35.0, noise=True, ao_strength=1.0):
    fr = rotate_view(field_iso, azimuth, elevation, order=1)
    em_r = rotate_view(em_iso, azimuth, elevation, order=1) if em_iso is not None else None

    st = column_stats(fr, voxel_nm)
    depth, mask, hit = st["depth_front"], st["mask"], st["hit"]
    normal = normals_from_depth(depth, voxel_nm, mask)
    ao = screen_space_ao(depth, voxel_nm, strength=ao_strength)
    mat = material_term(em_r, hit, mask)

    se = se_yield(normal) * ao * mat
    bse = []
    for L in detector_dirs(bse_elevation):
        lam = np.clip(normal[0] * L[0] + normal[1] * L[1] + normal[2] * L[2], 0, None)
        bse.append(lam * screen_space_shadow(depth, voxel_nm, L) * mat * (0.35 + 0.65 * ao))
    bse = np.stack(bse, 0)

    if noise:
        se = add_detector_noise(se, rng)
        bse = np.stack([add_detector_noise(b, rng, electrons=90.0) for b in bse], 0)

    se = norm01(se, mask)
    bse = np.stack([norm01(b, mask) for b in bse], 0)
    se[~mask] = 0.0
    bse[:, ~mask] = 0.0
    st.update(se=se, bse=bse, normal=normal)
    return st


def center_crop(st, size):
    keys2d = ["se", "depth_front", "thickness", "envelope", "mask", "clipped"]
    H, W = st["mask"].shape
    if size is None or (H <= size and W <= size):
        return st
    y0, x0 = (H - size) // 2, (W - size) // 2
    out = dict(st)
    for k in keys2d:
        out[k] = st[k][y0:y0 + size, x0:x0 + size]
    for k in ["bse", "normal"]:
        out[k] = st[k][:, y0:y0 + size, x0:x0 + size]
    out["hit"] = st["hit"][y0:y0 + size, x0:x0 + size]
    return out


def save_png(path, img):
    from PIL import Image
    a = (np.clip(np.nan_to_num(img, nan=0.0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def scan_best_views(field_iso, voxel_nm, n=12, elev_range=(-25, 25), seed=0):
    rng = np.random.default_rng(seed)
    cands = []
    for i in range(n):
        az = 360.0 * i / n
        el = float(rng.uniform(*elev_range))
        fr = rotate_view(field_iso, az, el, order=1)
        st = column_stats(fr, voxel_nm)
        score = float(st["mask"].mean() - st["clipped"].mean())
        cands.append((score, az, el, float(st["mask"].mean()), float(st["clipped"].mean())))
    cands.sort(reverse=True)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occupancy", nargs="+", default=["cell"],
                    help="label classes unioned into the solid to be exposed")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--elev-range", type=float, nargs=2, default=[-25.0, 25.0])
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--min-valid", type=float, default=0.25)
    ap.add_argument("--max-clipped", type=float, default=0.45,
                    help="reject views where this fraction of the surface hits a volume face")
    ap.add_argument("--smooth", type=float, default=0.7, help="occupancy pre-smoothing, voxels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--scan-views", action="store_true",
                    help="report the best view directions and exit")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = np.load(a.npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    voxel = np.array(meta["voxel_size_nm"], float)

    s = max(1, a.downsample)
    occ, used = None, []
    for c in a.occupancy:
        key = f"label_{c}"
        if key not in d.files:
            print(f"  skip missing class {c}", flush=True)
            continue
        v = np.ascontiguousarray(d[key][::s, ::s, ::s])
        occ = v if occ is None else np.maximum(occ, v, out=occ)
        used.append(c)
        del v
    if occ is None:
        raise SystemExit(f"none of {a.occupancy} in {a.npz}; has {list(d.files)}")
    em = np.ascontiguousarray(d["em"][::s, ::s, ::s]) if "em" in d.files else None
    d.close()
    voxel = voxel * s

    occ_iso, vnm = to_isotropic(occ.astype(np.float32), voxel, order=1)
    del occ
    field_iso = ndimage.gaussian_filter(occ_iso, a.smooth) if a.smooth > 0 else occ_iso
    del occ_iso
    em_iso = to_isotropic(em.astype(np.float32), voxel, order=1)[0] if em is not None else None
    if em_iso is not None:
        em_iso = em_iso[:field_iso.shape[0], :field_iso.shape[1], :field_iso.shape[2]]
    print(f"occupancy {used} -> {field_iso.shape} at {vnm:.2f} nm/voxel, "
          f"solid fraction={float((field_iso >= 0.5).mean()):.4f}", flush=True)

    if a.scan_views:
        for score, az, el, valid, clip in scan_best_views(field_iso, vnm,
                                                          elev_range=a.elev_range, seed=a.seed):
            print(f"  az={az:6.1f} el={el:+6.1f} score={score:+.3f} "
                  f"valid={valid:.3f} clipped={clip:.3f}", flush=True)
        return

    outdir = Path(a.out)
    (outdir / "views").mkdir(parents=True, exist_ok=True)
    if a.png:
        (outdir / "preview").mkdir(parents=True, exist_ok=True)

    kept, tried, index = 0, 0, []
    while kept < a.views and tried < a.views * 5:
        tried += 1
        az = float(rng.uniform(0, 360))
        el = float(rng.uniform(*a.elev_range))
        t0 = time.perf_counter()
        st = center_crop(render_view(field_iso, vnm, em_iso, az, el, rng,
                                     noise=not a.no_noise), a.crop_size)
        dt = time.perf_counter() - t0

        valid = float(st["mask"].mean())
        clip = float(st["clipped"].mean())
        if valid < a.min_valid or clip > a.max_clipped:
            print(f"  reject az={az:6.1f} el={el:+6.1f} valid={valid:.3f} "
                  f"clipped={clip:.3f} ({dt:.1f}s)", flush=True)
            continue

        vol_nm3 = float(st["thickness"].sum()) * vnm * vnm
        name = f"{Path(a.npz).stem}_az{az:06.2f}_el{el:+06.2f}"
        vm = dict(meta, occupancy=used, azimuth_deg=az, elevation_deg=el,
                  pixel_size_nm=float(vnm), depth_units="nm", camera="orthographic",
                  valid_fraction=valid, clipped_fraction=clip,
                  bse_detector_elevation_deg=35.0,
                  volume_gt_nm3=vol_nm3, volume_gt_um3=vol_nm3 * 1e-9,
                  thickness_mean_nm=float(st["thickness"][st["mask"]].mean()),
                  depth_p05_nm=float(np.nanpercentile(st["depth_front"], 5)),
                  depth_p95_nm=float(np.nanpercentile(st["depth_front"], 95)))
        np.savez_compressed(
            outdir / "views" / f"{name}.npz", meta=json.dumps(vm),
            se=st["se"], bse=st["bse"], depth_nm=st["depth_front"],
            thickness_nm=st["thickness"], envelope_nm=st["envelope"],
            normal=st["normal"], mask=st["mask"].astype(np.uint8),
            clipped=st["clipped"].astype(np.uint8))

        if a.png:
            save_png(outdir / "preview" / f"{name}_se.png", st["se"])
            dn = np.nan_to_num(st["depth_front"], nan=np.nanmax(st["depth_front"]))
            save_png(outdir / "preview" / f"{name}_depth.png", 1.0 - norm01(dn, st["mask"]))
            save_png(outdir / "preview" / f"{name}_thick.png",
                     norm01(st["thickness"], st["mask"]))
            save_png(outdir / "preview" / f"{name}_bse.png",
                     np.concatenate([np.concatenate(list(st["bse"][:2]), 1),
                                     np.concatenate(list(st["bse"][2:]), 1)], 0))
        index.append(vm)
        kept += 1
        print(f"  keep  az={az:6.1f} el={el:+6.1f} valid={valid:.3f} clipped={clip:.3f} "
              f"depth {vm['depth_p05_nm']:.0f}..{vm['depth_p95_nm']:.0f} nm  "
              f"vol={vm['volume_gt_um3']:.3f} um3 ({dt:.1f}s)", flush=True)

    (outdir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {kept} views to {outdir / 'views'}", flush=True)


if __name__ == "__main__":
    main()
