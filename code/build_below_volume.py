import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def voxel_nm3(meta):
    v = meta.get("voxel_size_nm")
    if not v:
        return None
    return float(v[0]) * float(v[1]) * float(v[2])


def box_below(raw_path, zs):
    d = np.load(raw_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    vox = voxel_nm3(meta)
    if vox is None:
        return None, None
    inst = d["inst"]
    Z, mx = inst.shape[0], int(inst.max())
    if mx == 0:
        return {}, meta
    per = np.zeros((Z, mx + 1), dtype=np.int64)
    for z in range(Z):
        s = inst[z].ravel()
        per[z, :] = np.bincount(s, minlength=mx + 1)[:mx + 1]
    cum = np.cumsum(per[::-1], axis=0)[::-1]
    out = {}
    for z in zs:
        if 0 <= z < Z:
            c = cum[z]
            face = np.unique(inst[z])
            out[z] = {int(i): float(c[i]) * vox for i in face if i != 0 and c[i] > 0}
    return out, meta


def main():
    boxes = sorted(glob.glob(os.path.join(ROOT, "blockface", "*", "index.json")))
    shard, nshard = 0, 1
    if len(sys.argv) > 2:
        shard, nshard = int(sys.argv[1]), int(sys.argv[2])
        boxes = boxes[shard::nshard]
    elif len(sys.argv) > 1:
        boxes = [b for b in boxes if sys.argv[1] in b]
    outp = os.path.join(ROOT, "cache", f"below_volume_{shard:02d}.jsonl")
    n_box = n_rec = 0
    with open(outp, "w") as fh:
        for idx_path in boxes:
            box = os.path.basename(os.path.dirname(idx_path))
            raw = os.path.join(ROOT, "raw", box + ".npz")
            if not os.path.exists(raw):
                continue
            try:
                faces = json.load(open(idx_path))
            except Exception:
                continue
            zs = sorted({int(f["z"]) for f in faces if "z" in f})
            if not zs:
                continue
            try:
                below, meta = box_below(raw, zs)
            except Exception as e:
                print(f"  {box}: {type(e).__name__} {e}", flush=True)
                continue
            if not below:
                continue
            byz = {int(f["z"]): f["name"] for f in faces if "z" in f}
            for z, d in below.items():
                name = byz.get(z)
                if not name:
                    continue
                rel = os.path.join("blockface", box, "faces", name + ".npz")
                for iid, v in d.items():
                    fh.write(json.dumps(dict(file=rel, box=box, instance=iid,
                                             z=z, v_below_nm3=v)) + "\n")
                    n_rec += 1
            n_box += 1
            if n_box % 25 == 0:
                print(f"  shard {shard}: {n_box} boxes, {n_rec:,} records", flush=True)
    print(f"done. {n_box} boxes, {n_rec:,} records -> {outp}")


if __name__ == "__main__":
    main()
