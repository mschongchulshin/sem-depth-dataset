import glob, json, os, sys
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 32


def resample(vol, g=G):
    z, y, x = vol.shape
    if min(z, y, x) == 0:
        return None
    iz = np.clip((np.arange(g) + 0.5) * z / g, 0, z - 1).astype(np.int32)
    iy = np.clip((np.arange(g) + 0.5) * y / g, 0, y - 1).astype(np.int32)
    ix = np.clip((np.arange(g) + 0.5) * x / g, 0, x - 1).astype(np.int32)
    return vol[np.ix_(iz, iy, ix)]


def box_shapes(raw_path, zs):
    d = np.load(raw_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    vox = meta.get("voxel_size_nm")
    if not vox:
        return None, None
    inst = d["inst"]
    Z = inst.shape[0]
    out = {}
    for z in zs:
        if not (0 <= z < Z):
            continue
        sub = inst[z:]
        face = np.unique(inst[z])
        slices = ndimage.find_objects(sub)
        for iid in face:
            if iid == 0 or iid > len(slices):
                continue
            sl = slices[int(iid) - 1]
            if sl is None:
                continue
            reg = (slice(0, sl[0].stop), sl[1], sl[2])
            crop = sub[reg] == iid
            if crop.sum() < 8:
                continue
            r = resample(crop)
            if r is None:
                continue
            out[(int(z), int(iid))] = np.packbits(r.reshape(-1))
    return out, meta


def main():
    boxes = sorted(glob.glob(os.path.join(ROOT, "blockface", "*", "index.json")))
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    boxes = boxes[shard::nshard]
    keys, arrs = [], []
    n_box = 0
    for idx_path in boxes:
        box = os.path.basename(os.path.dirname(idx_path))
        raw = os.path.join(ROOT, "raw", box + ".npz")
        if not os.path.exists(raw):
            continue
        try:
            faces = json.load(open(idx_path))
        except Exception:
            continue
        byz = {int(f["z"]): f["name"] for f in faces if "z" in f}
        if not byz:
            continue
        try:
            sh, meta = box_shapes(raw, sorted(byz))
        except Exception as e:
            print(f"  {box}: {type(e).__name__} {e}", flush=True); continue
        if not sh:
            continue
        for (z, iid), packed in sh.items():
            keys.append(f"blockface/{box}/faces/{byz[z]}.npz|{iid}")
            arrs.append(packed)
        n_box += 1
        if n_box % 25 == 0:
            print(f"  shard {shard}: {n_box} boxes, {len(keys):,} shapes", flush=True)
    if keys:
        out = os.path.join(ROOT, "cache", f"shapes_{shard:02d}.npz")
        np.savez_compressed(out, keys=np.array(keys), shapes=np.stack(arrs))
        print(f"done. {n_box} boxes, {len(keys):,} shapes -> {out}")


if __name__ == "__main__":
    main()
