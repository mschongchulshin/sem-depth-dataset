import numpy as np

from render_sem import add_detector_noise, kanaya_okayama_range_nm, norm01


def escape_weights(n, voxel_nm, kv, se1_fraction=0.5):
    z = np.arange(n) * voxel_nm
    r_bse = max(kanaya_okayama_range_nm(kv), voxel_nm)
    lam_se = max(0.05 * r_bse, voxel_nm * 0.5)
    w = se1_fraction * np.exp(-z / lam_se) + (1.0 - se1_fraction) * np.exp(-z / (0.4 * r_bse))
    return (w / w.sum()).astype(np.float32)


def blockface_stats(inst_sub, cls_sub, voxel_nm):
    solid = inst_sub > 0
    Z = inst_sub.shape[0]

    inst_face = inst_sub[0].astype(np.int32)
    cls_face = cls_sub[0].astype(np.uint8) if cls_sub is not None else None
    face = inst_face > 0

    thickness_below = solid.sum(axis=0) * voxel_nm

    same = inst_sub == inst_face[None]
    run = np.where(same.all(axis=0), Z, np.argmin(same, axis=0))
    depth_below = run * voxel_nm

    clipped = face & (run >= Z)

    return dict(inst_face=inst_face,
                cls_face=cls_face,
                mask=face,
                thickness_below=np.where(face, thickness_below, 0.0).astype(np.float32),
                depth_below=np.where(face, depth_below, np.nan).astype(np.float32),
                clipped=clipped)


def blockface_image(em_sub, inst_sub, voxel_nm, rng, kv=1.5, se1_fraction=0.5,
                    psf_sigma=0.8, electrons=250.0, line_gain=0.02, line_offset=0.02,
                    charging=0.05, depth_px=None):
    Z = em_sub.shape[0]
    n = int(depth_px or min(Z, max(4, int(round(2.0 * kanaya_okayama_range_nm(kv) / voxel_nm)))))
    n = max(1, min(n, Z))
    w = escape_weights(n, voxel_nm, kv, se1_fraction)
    sig = np.tensordot(w, em_sub[:n].astype(np.float32), axes=(0, 0))
    img = norm01(sig)
    return add_detector_noise(img, rng, psf_sigma=psf_sigma, electrons=electrons,
                              line_gain=line_gain, line_offset=line_offset,
                              charging=charging), n
