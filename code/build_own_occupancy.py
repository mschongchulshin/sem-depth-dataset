import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def box_occupancy(raw_path, zs):
    d = np.load(raw_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    vox = meta.get("voxel_size_nm")
    if not vox:
        return None
    vz = float(vox[0])
    inst = d["inst"]
    Z = inst.shape[0]
    out = {}
    for z in zs:
        if not (0 <= z < Z):
            continue
        face = inst[z]
        sub = inst[z:]
        occ = (sub == face[None]).sum(axis=0).astype(np.float32) * vz
        out[z] = np.where(face > 0, occ, 0.0).astype(np.float32)
    return out


def main():
    boxes = sorted(glob.glob(os.path.join(ROOT, "blockface", "*", "index.json")))
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    boxes = boxes[shard::nshard]
    n_box = n_face = 0
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
            occ = box_occupancy(raw, sorted(byz))
        except Exception as e:
            print(f"  {box}: {type(e).__name__} {e}", flush=True); continue
        if not occ:
            continue
        for z, arr in occ.items():
            p = os.path.join(ROOT, "blockface", box, "faces", byz[z] + ".npz")
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
            if data["mask"].shape != arr.shape:
                continue
            data["own_occupancy_below_nm"] = arr
            np.savez_compressed(p, **data)
            n_face += 1
        n_box += 1
        if n_box % 25 == 0:
            print(f"  shard {shard}: {n_box} boxes, {n_face:,} faces", flush=True)
    print(f"done. shard {shard}: {n_box} boxes, {n_face:,} faces updated")


if __name__ == "__main__":
    main()
