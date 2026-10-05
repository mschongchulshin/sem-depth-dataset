import argparse
import json
import sys

import numpy as np

from fetch_openorganelle import BUCKET_URL, list_children, open_array, read_zattrs


def probe(dataset, crop, classes, level):
    gt = f"{dataset}/{dataset}.zarr/recon-1/labels/groundtruth/{crop}"
    have = set(list_children(f"{gt}/"))
    out = {}
    for c in classes:
        if c not in have:
            continue
        try:
            att = read_zattrs(f"{gt}/{c}")
            paths = [d["path"] for d in att["multiscales"][0]["datasets"]]
            lvl = level if level in paths else paths[-1]
            v = np.asarray(open_array(f"{gt}/{c}/{lvl}")[:])
            occ = (v > 0) & (v != 255)
            out[c] = float(occ.mean())
            out.setdefault("_shape", list(v.shape))
            out.setdefault("_level", lvl)
        except Exception as e:
            out[c] = f"err:{type(e).__name__}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jrc_hela-2")
    ap.add_argument("--crops", nargs="*", default=None)
    ap.add_argument("--classes", nargs="+", default=["ecs", "pm", "cell", "mito", "er", "nuc"])
    ap.add_argument("--level", default="s3")
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()

    crops = a.crops or sorted(list_children(
        f"{a.dataset}/{a.dataset}.zarr/recon-1/labels/groundtruth/"))[:a.limit]
    rows = []
    for crop in crops:
        try:
            r = probe(a.dataset, crop, a.classes, a.level)
        except Exception as e:
            print(f"{crop:10s} FAILED {type(e).__name__}: {e}", file=sys.stderr)
            continue
        cell = r.get("cell", 0.0)
        ecs = r.get("ecs", 0.0)
        cell = cell if isinstance(cell, float) else 0.0
        ecs = ecs if isinstance(ecs, float) else 0.0
        surf = min(cell, 1.0 - cell) if cell > 0 else 0.0
        rows.append((surf + ecs, crop, r))
        print(f"{crop:10s} shape={r.get('_shape')} lvl={r.get('_level')} "
              + " ".join(f"{c}={r[c]:.3f}" if isinstance(r.get(c), float) else f"{c}=-"
                         for c in a.classes)
              + f"  surface_score={surf + ecs:.3f}", flush=True)

    rows.sort(reverse=True)
    print("\nbest candidates for a real exposed surface")
    for score, crop, _ in rows[:8]:
        print(f"  {crop:10s} score={score:.3f}")


if __name__ == "__main__":
    main()
