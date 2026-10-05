import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def depth_to_points(depth_nm, pixel_nm, mask=None):
    H, W = depth_nm.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    good = np.isfinite(depth_nm) if mask is None else (mask & np.isfinite(depth_nm))
    return np.stack([xx[good] * pixel_nm, yy[good] * pixel_nm, depth_nm[good]], 1), good


def height_field_mesh(depth_nm, pixel_nm, mask, max_step_nm=None):
    H, W = depth_nm.shape
    z = np.where(mask & np.isfinite(depth_nm), np.nan_to_num(depth_nm, nan=0.0), np.nan)
    if max_step_nm is None:
        finite = z[np.isfinite(z)]
        span = float(finite.max() - finite.min()) if finite.size else 0.0
        max_step_nm = max(4.0 * pixel_nm, 0.05 * span)

    idx = -np.ones((H, W), np.int64)
    ok = np.isfinite(z)
    idx[ok] = np.arange(int(ok.sum()))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    verts = np.stack([xx[ok] * pixel_nm, yy[ok] * pixel_nm, z[ok]], 1)

    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    za, zb = z[:-1, :-1], z[:-1, 1:]
    zc, zd = z[1:, :-1], z[1:, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    flat = quad & (np.nanmax(np.stack([za, zb, zc, zd]), 0) -
                   np.nanmin(np.stack([za, zb, zc, zd]), 0) < max_step_nm)
    tri = np.concatenate([
        np.stack([a[flat], b[flat], d[flat]], 1),
        np.stack([a[flat], d[flat], c[flat]], 1)], 0)
    return verts, tri


def solid_mesh(depth_nm, thickness_nm, pixel_nm, mask, max_step_nm=None):
    front_v, front_f = height_field_mesh(depth_nm, pixel_nm, mask, max_step_nm)
    back = depth_nm + np.nan_to_num(thickness_nm, nan=0.0)
    back_v, back_f = height_field_mesh(back, pixel_nm, mask, max_step_nm)
    n = len(front_v)
    verts = np.concatenate([front_v, back_v], 0)
    faces = np.concatenate([front_f, back_f[:, ::-1] + n], 0)

    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ok = np.isfinite(np.where(mask, depth_nm, np.nan))
    idx[ok] = np.arange(int(ok.sum()))
    try:
        from skimage import measure as _measure
        loops = _measure.find_contours(mask.astype(float), 0.5)
    except Exception:
        loops = []
    rim = []
    for loop in loops:
        pts = np.round(loop).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)
        ids = idx[pts[:, 0], pts[:, 1]]
        ids = ids[ids >= 0]
        if len(ids) < 3:
            continue
        i0 = ids
        i1 = np.roll(ids, -1)
        keep = i0 != i1
        i0, i1 = i0[keep], i1[keep]
        rim.append(np.stack([i0, i1, i1 + n], 1))
        rim.append(np.stack([i0, i1 + n, i0 + n], 1))
    if rim:
        faces = np.concatenate([faces] + rim, 0)
    return verts, faces


def labels_to_mesh(mask3d, voxel_zyx, smooth=0.6):
    from skimage import measure
    pad = np.pad(mask3d.astype(np.float32), 2)
    if smooth > 0:
        pad = ndimage.gaussian_filter(pad, smooth)
    verts, faces, _, _ = measure.marching_cubes(pad, level=0.5, spacing=tuple(voxel_zyx))
    return verts[:, ::-1], faces


def write_ply(path, verts, faces, colors=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        if colors is None:
            for v in verts:
                f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        else:
            for v, c in zip(verts, colors):
                f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def mesh_area(verts, faces):
    if len(faces) == 0:
        return 0.0
    p = verts[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def render_turntable(verts, faces, out_png, n_views=4, elev=35.0, title=None,
                     cmap="viridis", figsize=3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) == 0:
        raise SystemExit("mesh has no faces")
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, -0.75])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.15, 1.0)
    zc = tri[:, :, 2].mean(1)
    base = plt.get_cmap(cmap)((zc - zc.min()) / max(float(np.ptp(zc)), 1e-9))
    base[:, :3] *= shade[:, None]

    fig = plt.figure(figsize=(figsize * n_views, figsize * 1.15), dpi=150)
    for k in range(n_views):
        ax = fig.add_subplot(1, n_views, k + 1, projection="3d")
        coll = Poly3DCollection(tri, facecolors=base, linewidths=0)
        ax.add_collection3d(coll)
        for setlim, vals in ((ax.set_xlim, verts[:, 0]), (ax.set_ylim, verts[:, 1]),
                             (ax.set_zlim, verts[:, 2])):
            setlim(vals.min(), vals.max())
        try:
            ax.set_box_aspect((float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 1])),
                               max(float(np.ptp(verts[:, 2])), 1e-6)))
        except Exception:
            pass
        ax.view_init(elev=elev, azim=-60 + 90 * k)
        ax.set_axis_off()
        ax.set_title(f"azim {-60 + 90 * k:+.0f}", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def instance_view_geometry(raw_npz, view_meta, instance_id, target_shape=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_sem import rotate_view, to_isotropic

    d = np.load(raw_npz, allow_pickle=False)
    rmeta = json.loads(str(d["meta"]))
    voxel = np.array(rmeta["voxel_size_nm"], float)
    inst = d["inst"]
    sel = (inst == instance_id).astype(np.uint8)
    if sel.sum() == 0:
        raise SystemExit(f"instance {instance_id} not in {raw_npz}")
    iso, vnm = to_isotropic(sel, voxel, order=0)
    rot = rotate_view(iso, view_meta["azimuth_deg"], view_meta["elevation_deg"], order=0) > 0

    mask = rot.any(axis=0)
    Z = rot.shape[0]
    front = np.argmax(rot, axis=0).astype(np.float32) * vnm
    thick = rot.sum(axis=0).astype(np.float32) * vnm
    depth = np.where(mask, front, np.nan).astype(np.float32)
    thick = np.where(mask, thick, 0.0).astype(np.float32)

    if target_shape is not None and mask.shape != tuple(target_shape):
        H, W = mask.shape
        th, tw = target_shape
        y0, x0 = max((H - th) // 2, 0), max((W - tw) // 2, 0)
        sl = (slice(y0, y0 + th), slice(x0, x0 + tw))
        depth, thick, mask = depth[sl], thick[sl], mask[sl]
    return depth, thick, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", help="a rendered view .npz")
    ap.add_argument("--from-labels", help="a fetch_wholecell .npz, for true organelle surfaces")
    ap.add_argument("--instance", type=int, default=None,
                    help="with --from-labels, mesh only this instance id")
    ap.add_argument("--raw", help="the fetch_wholecell npz, needed for --instance with --view")
    ap.add_argument("--pred", help="optional .npy of predicted depth in nm, same shape")
    ap.add_argument("--pred-thickness", help="optional .npy of predicted thickness in nm")
    ap.add_argument("--mode", default="surface", choices=["surface", "solid"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=4)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.from_labels:
        d = np.load(a.from_labels, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        voxel = np.array(meta["voxel_size_nm"], float)
        inst = d["inst"]
        if a.instance is None:
            ids, cnt = np.unique(inst[inst > 0], return_counts=True)
            a.instance = int(ids[np.argmax(cnt)])
            print(f"largest instance is {a.instance} with {cnt.max()} voxels")
        sl = ndimage.find_objects(inst.astype(np.int32))[a.instance - 1]
        patch = inst[sl] == a.instance
        verts, faces = labels_to_mesh(patch, voxel)
        tag = f"labels_inst{a.instance}"
        vol = float(patch.sum()) * float(np.prod(voxel))
        print(f"true mesh: {len(verts)} verts {len(faces)} faces  "
              f"area={mesh_area(verts, faces) * 1e-6:.4f} um2  volume={vol * 1e-9:.4f} um3")
    else:
        if not a.view:
            raise SystemExit("need --view or --from-labels")
        d = np.load(a.view, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        px = float(meta["pixel_size_nm"])
        mask = d["mask"].astype(bool)
        depth = np.load(a.pred) if a.pred else d["depth_nm"]

        if a.instance is not None and a.raw:
            depth, thick, mask = instance_view_geometry(
                a.raw, meta, a.instance, target_shape=mask.shape)
            verts, faces = solid_mesh(depth, thick, px, mask) if a.mode == "solid" else \
                height_field_mesh(depth, px, mask)
            v_int = float(np.nan_to_num(thick)[mask].sum()) * px * px
            tag = f"{Path(a.view).stem}_inst{a.instance}_{a.mode}_gt"
            print(f"instance {a.instance}: integrated volume={v_int * 1e-9:.4f} um3  "
                  f"mesh {len(verts)} verts {len(faces)} faces  "
                  f"area={mesh_area(verts, faces) * 1e-6:.4f} um2")
            write_ply(out / f"{tag}.ply", verts, faces)
            write_obj(out / f"{tag}.obj", verts, faces)
            render_turntable(verts, faces, out / f"{tag}_turntable.png",
                             n_views=a.views, title=tag)
            print(f"-> {out / (tag + '.ply')}\n-> {out / (tag + '_turntable.png')}")
            return

        if a.mode == "solid":
            thick = np.load(a.pred_thickness) if a.pred_thickness else d["thickness_nm"]
            verts, faces = solid_mesh(depth, thick, px, mask)
            vol_int = float(np.nan_to_num(thick)[mask].sum()) * px * px
            print(f"solid: integrated volume={vol_int * 1e-9:.4f} um3")
        else:
            verts, faces = height_field_mesh(depth, px, mask)
        tag = f"{Path(a.view).stem}_{a.mode}" + ("_pred" if a.pred else "_gt")
        print(f"mesh: {len(verts)} verts {len(faces)} faces  "
              f"area={mesh_area(verts, faces) * 1e-6:.4f} um2  pixel={px:.2f} nm")

    write_ply(out / f"{tag}.ply", verts, faces)
    write_obj(out / f"{tag}.obj", verts, faces)
    render_turntable(verts, faces, out / f"{tag}_turntable.png", n_views=a.views, title=tag)
    print(f"-> {out / (tag + '.ply')}\n-> {out / (tag + '_turntable.png')}")


if __name__ == "__main__":
    main()
