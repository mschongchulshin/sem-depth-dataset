import json, os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n5 import N5Array
from fetch_wholecell import seg_url

ROOT = Path(os.environ.get("EM_DEPTH_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "cache" / "cells.json"

MIN_NUCLEUS_UM3 = 5.0


def cells_in(dataset, level="s4", min_um3=MIN_NUCLEUS_UM3):
    a = N5Array(seg_url(dataset, "nucleus"), level)
    vox = np.array(a.voxel_size_nm(), float)
    vol = a.read([0, 0, 0], a.shape)
    ids = np.unique(vol[vol > 0])
    if ids.size == 0:
        return [], vox.tolist()
    objs = ndimage.find_objects(vol.astype(np.int32))
    vvol_um3 = float(np.prod(vox)) * 1e-9
    out = []
    for i in ids:
        sl = objs[int(i) - 1] if int(i) - 1 < len(objs) else None
        if sl is None:
            continue
        sub = vol[sl] == i
        n = int(sub.sum())
        if n * vvol_um3 < min_um3:
            continue
        com = np.array([sl[d].start for d in range(3)]) + np.array(ndimage.center_of_mass(sub))
        lo = np.array([sl[d].start for d in range(3)], float)
        hi = np.array([sl[d].stop for d in range(3)], float)
        out.append(dict(id=int(i),
                        centroid_nm=(com * vox).tolist(),
                        bbox_lo_nm=(lo * vox).tolist(),
                        bbox_hi_nm=(hi * vox).tolist(),
                        nucleus_um3=round(n * vvol_um3, 3)))
    out.sort(key=lambda r: -r["nucleus_um3"])
    return out, vox.tolist()


if __name__ == "__main__":
    avail = json.loads((ROOT / "cache" / "availability.json").read_text())
    want = [d for d, m in avail.items() if m.get("nucleus", {}).get("ok")]
    res = {}
    for ds in want:
        try:
            cells, vox = cells_in(ds)
            res[ds] = dict(level="s4", voxel_nm=vox, n_cells=len(cells), cells=cells)
            print(f"{ds:24s} {len(cells):4d} cells  "
                  f"nucleus volumes um3 "
                  f"{[c['nucleus_um3'] for c in cells[:5]]}", flush=True)
        except Exception as e:
            res[ds] = dict(error=f"{type(e).__name__}: {e}"[:150], n_cells=0, cells=[])
            print(f"{ds:24s} FAILED {type(e).__name__}: {e}"[:120], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    tot = sum(r.get("n_cells", 0) for r in res.values())
    print(f"\n{tot} cells located across {sum(1 for r in res.values() if r.get('n_cells'))} datasets")
    print(f"-> {OUT}")
