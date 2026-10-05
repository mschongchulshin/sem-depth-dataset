import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    raws = sorted(glob.glob(os.path.join(ROOT, "raw", "*.npz")))[shard::nshard]
    out = os.path.join(ROOT, "cache", f"exact_volume_{shard:02d}.jsonl")
    n_box = n_rec = 0
    with open(out, "w") as fh:
        for p in raws:
            box = os.path.basename(p)[:-4]
            try:
                with np.load(p, allow_pickle=True) as d:
                    meta = json.loads(str(d["meta"]))
                    inst = d["inst"]
                    counts = np.bincount(inst.ravel())
            except Exception as e:
                print(f"  {box}: {type(e).__name__} {e}", flush=True); continue
            vs = meta.get("voxel_size_nm")
            if not vs:
                continue
            vox = float(np.prod([float(x) for x in vs]))
            for sid, rec in meta.get("instances", {}).items():
                i = int(sid)
                if i <= 0 or i >= len(counts) or counts[i] == 0:
                    continue
                fh.write(json.dumps(dict(
                    box=box, instance=i, level=meta.get("level"),
                    v_exact_nm3=float(counts[i]) * vox,
                    v_stored_nm3=rec.get("volume_nm3"),
                    voxels=int(counts[i]), truncated=bool(rec.get("truncated")))) + "\n")
                n_rec += 1
            n_box += 1
            if n_box % 50 == 0:
                print(f"  shard {shard}: {n_box} boxes, {n_rec:,} records", flush=True)
    print(f"done. shard {shard}: {n_box} boxes, {n_rec:,} records -> {out}")


if __name__ == "__main__":
    main()
