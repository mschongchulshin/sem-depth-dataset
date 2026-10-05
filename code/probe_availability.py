import json, sys, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n5 import N5Array
from fetch_wholecell import seg_url

DATASETS = ["jrc_hela-2", "jrc_hela-3", "jrc_hela-1", "jrc_hela-bfa", "jrc_jurkat-1",
            "jrc_macrophage-2", "jrc_cos7-11", "jrc_choroid-plexus-2",
            "aic_desmosome-2", "aic_desmosome-3", "jrc_mus-liver", "jrc_mus-kidney",
            "jrc_dauer-larva", "jrc_mus-heart-1", "jrc_mus-hippocampus-1", "jrc_mus-kidney-3",
            "jrc_mus-liver-3", "jrc_mus-pancreas-4", "jrc_mus-skin-1", "jrc_mus-thymus-1",
            "jrc_mus-epididymis-1", "jrc_mus-epididymis-2", "jrc_mus-liver-zon-2"]
ORGANELLES = ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "eres",
              "nucleus", "nucleolus", "perox", "mt", "ribo", "pm", "ne", "chrom"]


def probe(args):
    ds, org = args
    try:
        a = N5Array(seg_url(ds, org), "s0")
    except FileNotFoundError:
        return ds, org, dict(ok=False, why="absent")
    except Exception as e:
        return ds, org, dict(ok=False, why=f"{type(e).__name__}: {e}"[:120])
    try:
        vox = [float(v) for v in a.voxel_size_nm()]
        shape = [int(v) for v in a.shape]
        return ds, org, dict(ok=True, voxel_nm=vox, shape=shape, comp=a.compression,
                             dtype=str(a.dtype), anisotropy=round(max(vox) / min(vox), 2),
                             extent_um=[round(s * v / 1000.0, 1) for s, v in zip(shape, vox)])
    except Exception as e:
        return ds, org, dict(ok=False, why=f"{type(e).__name__}: {e}"[:120])


if __name__ == "__main__":
    jobs = [(d, o) for d in DATASETS for o in ORGANELLES]
    out = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for ds, org, rec in ex.map(probe, jobs):
            out.setdefault(ds, {})[org] = rec
            if rec["ok"]:
                print(f"{ds:24s} {org:9s} vox={rec['voxel_nm']} "
                      f"extent={rec['extent_um']} um aniso={rec['anisotropy']} "
                      f"{rec['comp']}", flush=True)
    Path("cache/availability.json").write_text(json.dumps(out, indent=1))
    print("\n=== per dataset, readable organelles at s0 ===")
    for ds in DATASETS:
        good = sorted(o for o, r in out.get(ds, {}).items() if r["ok"])
        print(f"{ds:24s} {len(good):2d}  {good}")
