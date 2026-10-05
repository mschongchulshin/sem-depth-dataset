import argparse, collections, glob, json, os, sys
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--min-ids", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raws = sorted(glob.glob(os.path.join(ROOT, "raw", "*.npz")))[a.shard::a.nshard]
    out = a.out or os.path.join(ROOT, "cache", f"inst_semantics_{a.shard:02d}.jsonl")
    n_box = n_rec = 0
    with open(out, "w") as fh:
        for p in raws:
            box = os.path.basename(p)[:-4]
            try:
                with np.load(p, allow_pickle=True) as d:
                    meta = json.loads(str(d["meta"]))
                    inst = d["inst"]
            except Exception as e:
                print(f"  {box}: {type(e).__name__} {e}", flush=True); continue
            byo = collections.defaultdict(list)
            for sid, rec in meta.get("instances", {}).items():
                o = rec.get("organelle")
                if o:
                    byo[o].append(int(sid))
            if not byo:
                continue
            present = np.unique(inst)
            for o, ids in byo.items():
                ids = [i for i in ids if i in present]
                if len(ids) < a.min_ids:
                    continue
                mask = np.isin(inst, ids)
                nvox = int(mask.sum())
                if nvox < 50:
                    continue
                _, ncomp = ndimage.label(mask)
                scattered = 0
                for i in ids[:200]:
                    _, k = ndimage.label(inst == i)
                    if k > 1:
                        scattered += 1
                fh.write(json.dumps(dict(
                    box=box, dataset=meta.get("dataset"), level=meta.get("level"),
                    organelle=o, n_ids=len(ids), n_components=int(ncomp),
                    ratio=len(ids) / max(ncomp, 1), voxels=nvox,
                    ids_scattered=scattered, ids_checked=min(len(ids), 200))) + "\n")
                n_rec += 1
            n_box += 1
            if n_box % 25 == 0:
                print(f"  shard {a.shard}: {n_box} boxes, {n_rec:,} records", flush=True)
    print(f"done. shard {a.shard}: {n_box} boxes, {n_rec:,} records -> {out}")


if __name__ == "__main__":
    main()
