import numpy as np
from scipy import ndimage

try:
    from skimage import measure, morphology
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


def _skeleton_length(skel, spacing):
    spacing = np.asarray(spacing, float)
    idx = np.argwhere(skel)
    if len(idx) < 2:
        return 0.0
    ndim = skel.ndim
    offsets = np.array([o for o in np.ndindex(*([3] * ndim))]) - 1
    keep = []
    for o in offsets:
        if np.all(o == 0):
            continue
        for k in keep:
            if np.all(k == -o):
                break
        else:
            keep.append(o)

    lut = np.zeros(np.array(skel.shape) + 2, bool)
    lut[tuple(slice(1, -1) for _ in range(ndim))] = skel
    total = 0.0
    base = idx + 1
    for o in keep:
        nb = base + o
        hit = lut[tuple(nb[:, d] for d in range(ndim))]
        total += hit.sum() * float(np.linalg.norm(o * spacing))
    return float(total)


def skeleton_length_3d(mask, spacing, isotropic=True, cap_correct=True):
    if not _HAVE_SKIMAGE:
        return None
    spacing = np.asarray(spacing, float)
    m = mask
    sp = spacing
    if isotropic and not np.allclose(spacing, spacing.min()):
        zoom = spacing / spacing.min()
        m = ndimage.zoom(mask.astype(np.uint8), zoom, order=0, prefilter=False) > 0
        sp = np.full(3, spacing.min())
    if m.sum() < 2:
        return 0.0
    skel = morphology.skeletonize(m)
    length = _skeleton_length(skel, sp)
    if not cap_correct or skel.sum() < 2:
        return length
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8),
                          mode="constant") - skel
    ends = skel & (nb == 1)
    if not ends.any():
        return length
    dist = ndimage.distance_transform_edt(m, sampling=sp)
    return float(length + dist[ends].sum())


def skeleton_length_2d(mask, pixel_nm, cap_correct=True):
    if not _HAVE_SKIMAGE or mask.sum() < 2:
        return 0.0
    sp = (pixel_nm, pixel_nm)
    skel = morphology.skeletonize(mask)
    length = _skeleton_length(skel, sp)
    if not cap_correct or skel.sum() < 2:
        return length
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                          mode="constant") - skel
    ends = skel & (nb == 1)
    if not ends.any():
        return length
    dist = ndimage.distance_transform_edt(mask, sampling=sp)
    return float(length + dist[ends].sum())


def feret_diameter(mask, spacing):
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx.astype(np.float64) * np.asarray(spacing, float)
    if len(pts) > 2:
        try:
            from scipy.spatial import ConvexHull
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    d = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def surface_area_3d(mask, spacing, smooth=0.6):
    if not _HAVE_SKIMAGE or mask.sum() < 8:
        return None
    pad = np.pad(mask.astype(np.float32), 2)
    if smooth > 0:
        pad = ndimage.gaussian_filter(pad, smooth)
    try:
        verts, faces, _, _ = measure.marching_cubes(pad, level=0.5, spacing=tuple(spacing))
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        return None


def pca_axes(mask, spacing):
    idx = np.argwhere(mask).astype(np.float64) * np.asarray(spacing, float)
    if len(idx) < 4:
        return None
    idx -= idx.mean(0)
    cov = np.cov(idx.T)
    ev = np.linalg.eigvalsh(cov)[::-1]
    return (4.0 * np.sqrt(np.clip(ev, 0, None))).tolist()


def morphometry_3d(mask, spacing, with_skeleton=True):
    spacing = np.asarray(spacing, float)
    vvol = float(np.prod(spacing))
    n = int(mask.sum())
    out = dict(voxels=n, volume_nm3=n * vvol)
    area = surface_area_3d(mask, spacing)
    out["surface_area_nm2"] = area
    if area and out["volume_nm3"] > 0:
        out["sphericity"] = float((36 * np.pi * out["volume_nm3"] ** 2) ** (1 / 3) / area)
    axes = pca_axes(mask, spacing)
    if axes:
        out["axes_nm"] = axes
        out["elongation"] = float(axes[0] / max(axes[1], 1e-9))
        out["flatness"] = float(axes[1] / max(axes[2], 1e-9))
    out["feret_nm"] = feret_diameter(mask, spacing)
    if with_skeleton:
        out["curve_length_nm"] = skeleton_length_3d(mask, spacing)
    return out


def morphometry_2d(silhouette, pixel_nm):
    n = int(silhouette.sum())
    out = dict(proj_pixels=n, proj_area_nm2=n * pixel_nm * pixel_nm)
    out["proj_length_nm"] = skeleton_length_2d(silhouette, pixel_nm)
    out["proj_feret_nm"] = feret_diameter(silhouette, (pixel_nm, pixel_nm))
    if _HAVE_SKIMAGE and n >= 8:
        props = measure.regionprops(silhouette.astype(np.uint8))
        if props:
            p = props[0]
            out["proj_major_nm"] = float(p.axis_major_length * pixel_nm)
            out["proj_minor_nm"] = float(p.axis_minor_length * pixel_nm)
            out["proj_perimeter_nm"] = float(p.perimeter * pixel_nm)
    return out


def combine(m3, m2, pixel_nm=None, min_major_px=20, min_minor_px=5, min_elongation=1.5):
    out = dict(m3)
    out.update(m2)
    cl, pl = m3.get("curve_length_nm"), m2.get("proj_length_nm")
    if cl and pl and pl > 0:
        valid = True
        if pixel_nm:
            maj, mnr = m2.get("proj_major_nm"), m2.get("proj_minor_nm")
            if maj is None or mnr is None:
                valid = False
            else:
                valid = (maj / pixel_nm >= min_major_px and mnr / pixel_nm >= min_minor_px)
            el = m3.get("elongation")
            if el is not None and el < min_elongation:
                valid = False
        out["foreshortening_valid"] = bool(valid)
        out["foreshortening"] = float(cl / pl)
    f3, f2 = m3.get("feret_nm"), m2.get("proj_feret_nm")
    if f3 and f2 and f2 > 0:
        out["feret_ratio"] = float(f3 / f2)

    a, pa = m3.get("surface_area_nm2"), m2.get("proj_area_nm2")
    if a and pa and pa > 0:
        out["area_ratio"] = float(a / (4.0 * pa))
    v = m3.get("volume_nm3")
    if v and pa and pa > 0:
        out["volume_from_proj_nm3"] = float((pa ** 1.5) * (4.0 / 3.0) / np.sqrt(np.pi))
        out["volume_2d_error"] = float(abs(out["volume_from_proj_nm3"] - v) / v)
    return out


def table_from_volume(inst, spacing, classes=None, cls=None, min_voxels=16,
                      max_instances=None, verbose=False):
    ids = np.unique(inst)
    ids = ids[ids > 0]
    if max_instances:
        ids = ids[:max_instances]
    objs = ndimage.find_objects(inst.astype(np.int32))
    out = {}
    for i in ids:
        sl = objs[int(i) - 1] if int(i) - 1 < len(objs) else None
        if sl is None:
            continue
        patch = inst[sl] == i
        if patch.sum() < min_voxels:
            continue
        rec = morphometry_3d(patch, spacing)
        rec["bbox_zyx"] = [[int(s.start), int(s.stop)] for s in sl]
        rec["touches_border"] = bool(
            sl[0].start == 0 or sl[1].start == 0 or sl[2].start == 0 or
            sl[0].stop == inst.shape[0] or sl[1].stop == inst.shape[1] or
            sl[2].stop == inst.shape[2])
        if cls is not None and classes is not None:
            ci = int(np.bincount(cls[sl][patch]).argmax())
            rec["organelle"] = classes[ci - 1] if 0 < ci <= len(classes) else "unknown"
        out[int(i)] = rec
        if verbose:
            print(f"  id {int(i):6d} {rec.get('organelle','?'):8s} "
                  f"V={rec['volume_nm3']*1e-9:8.4f} um3 "
                  f"L={rec.get('curve_length_nm', 0):8.1f} nm "
                  f"sph={rec.get('sphericity', float('nan')):.3f}", flush=True)
    return out


if __name__ == "__main__":
    sp = np.array([1.0, 1.0, 1.0])
    r = 20
    zz, yy, xx = np.mgrid[-30:31, -30:31, -30:31]
    ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2
    m = morphometry_3d(ball, sp)
    print(f"ball r={r}: V={m['volume_nm3']:.0f} (exact {4/3*np.pi*r**3:.0f})  "
          f"A={m['surface_area_nm2']:.0f} (exact {4*np.pi*r**2:.0f})  "
          f"sphericity={m['sphericity']:.4f} (exact 1)")

    L, rad = 80, 4
    cyl = np.zeros((2 * rad + 7, 2 * rad + 7, L + 10), bool)
    cz = cy = rad + 3
    zz, yy = np.mgrid[:cyl.shape[0], :cyl.shape[1]]
    disc = ((zz - cz) ** 2 + (yy - cy) ** 2) <= rad ** 2
    cyl[:, :, 5:5 + L] = disc[:, :, None]
    m = morphometry_3d(cyl, sp)
    print(f"cylinder L={L} r={rad}: curve_length={m['curve_length_nm']:.1f} (exact ~{L})  "
          f"elongation={m['elongation']:.2f}")

    proj = cyl.any(axis=0)
    m2 = morphometry_2d(proj, 1.0)
    print(f"  projected along the short axis: proj_length={m2['proj_length_nm']:.1f}  "
          f"foreshortening={combine(m, m2)['foreshortening']:.3f} (exact 1)")

    rot = ndimage.rotate(cyl.astype(np.uint8), 60, axes=(0, 2), order=0, reshape=True) > 0
    m3r = morphometry_3d(rot, sp)
    m2r = morphometry_2d(rot.any(axis=0), 1.0)
    c = combine(m3r, m2r)
    t = np.deg2rad(60)
    proj_exact = L * np.cos(t) + (2 * rad) * np.sin(t)
    print(f"  feret ratio untilted={combine(m, m2)['feret_ratio']:.3f} (exact 1)  "
          f"tilted={combine(m3r, m2r)['feret_ratio']:.3f} "
          f"(exact {L / (L * np.cos(np.deg2rad(60)) + 2 * rad * np.sin(np.deg2rad(60))):.3f})")
    print(f"  tilted 60 deg: curve={m3r['curve_length_nm']:.1f} proj={m2r['proj_length_nm']:.1f} "
          f"(exact proj {proj_exact:.1f})  foreshortening={c['foreshortening']:.3f} "
          f"(exact {L / proj_exact:.3f})")
