import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fetch_openorganelle import list_children
from n5 import N5Array

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
ROOT = Path(os.environ.get("EM_DEPTH_ROOT", Path(__file__).resolve().parent.parent))
TIER_NM = {"fine": 4.0, "mid": 8.0, "coarse": 16.0}
CELL_VOL_FLOOR_UM3 = 10.0


def seg_url(ds, name):
    return f"{BUCKET}/{ds}/{ds}.n5/labels/{name}_seg"


def probe_pyramid(ds, organelle):
    out, comp, shapes = {}, None, {}
    for lvl in ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]:
        try:
            a = N5Array(seg_url(ds, organelle), lvl)
        except NotImplementedError as e:
            return {}, f"unsupported:{e}".split()[0], {}
        except Exception:
            break
        v = a.voxel_size_nm()
        out[lvl] = float(v[2]) if v is not None else None
        shapes[lvl] = a.shape
        comp = a.compression
    return out, comp, shapes


def count_instances(ds, organelle, level):
    a = N5Array(seg_url(ds, organelle), level)
    vv = float(np.prod(a.voxel_size_nm())) * 1e-9
    v = a.read([0, 0, 0], a.shape)
    ids, cnt = np.unique(v[v > 0], return_counts=True)
    if len(ids) == 0:
        return dict(n=0, fill=0.0, median_um3=None, n_above_floor=0)
    vols = cnt * vv
    return dict(n=int(len(ids)), fill=float((v > 0).mean()),
                median_um3=float(np.median(vols)),
                n_above_floor=int((vols > CELL_VOL_FLOOR_UM3).sum()))


def census_one(ds):
    rec = dict(dataset=ds, segs=[], pyramid={}, compression=None, shapes={},
               cells=None, classes={}, tiers=[], error=None)
    try:
        kids = list_children(f"{ds}/{ds}.n5/labels/")
        segs = sorted(k for k in kids if k.endswith("_seg"))
        rec["segs"] = [s.replace("_seg", "") for s in segs]
        if not segs:
            return rec

        anchor = rec["segs"][0]
        for cand in ("mito", "er", "nucleus", anchor):
            if cand in rec["segs"]:
                anchor = cand
                break
        pyr, comp, shapes = probe_pyramid(ds, anchor)
        rec["pyramid"], rec["compression"] = pyr, comp
        rec["shapes"] = {k: list(v) for k, v in shapes.items()}
        if not pyr:
            rec["error"] = f"pyramid unreadable ({comp})"
            return rec

        finest = min(v for v in pyr.values() if v)
        rec["tiers"] = [t for t, nm in TIER_NM.items()
                        if any(abs(np.log2(v / nm)) <= np.log2(1.6) for v in pyr.values() if v)]
        rec["finest_nm"] = finest

        coarse = sorted(pyr.items(), key=lambda kv: -(kv[1] or 0))[0][0]
        if "nucleus" in rec["segs"]:
            try:
                rec["cells"] = count_instances(ds, "nucleus", coarse)["n_above_floor"]
            except Exception as e:
                rec["cells"] = f"err:{type(e).__name__}"

        for c in rec["segs"]:
            if c.endswith("-mem") or c in ("ecs", "pm"):
                continue
            try:
                rec["classes"][c] = count_instances(ds, c, coarse)
            except Exception as e:
                rec["classes"][c] = dict(error=f"{type(e).__name__}")
        rec["counted_at_level"] = coarse
        rec["counted_at_nm"] = pyr.get(coarse)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "cache" / "census.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    survey = ROOT / "cache" / "survey.json"
    if survey.exists():
        cand = [r["dataset"] for r in json.loads(survey.read_text()) if r["n_seg"]]
    else:
        cand = sorted(d for d in list_children("") if d)
    if a.only:
        cand = [c for c in cand if c in a.only]
    print(f"census over {len(cand)} datasets with instance segmentations", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(census_one, cand))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

    print(f"\n{'dataset':22s} {'finest':>7s} {'levels':>7s} {'comp':>6s} {'cells':>6s} "
          f"{'tiers':>16s}  classes")
    print("-" * 120)
    for r in sorted(rows, key=lambda x: -(len(x["segs"]))):
        if r.get("error") and not r["pyramid"]:
            print(f"{r['dataset']:22s} {'-':>7s} {'-':>7s} {'-':>6s} {'-':>6s} {'-':>16s}  "
                  f"ERROR {r['error']}")
            continue
        print(f"{r['dataset']:22s} {r.get('finest_nm', 0):7.1f} {len(r['pyramid']):7d} "
              f"{str(r['compression'])[:6]:>6s} {str(r.get('cells')):>6s} "
              f"{','.join(sorted(r['tiers'])):>16s}  {len(r['segs'])} classes")

    print(f"\ninstance counts (at each dataset's coarsest level)")
    print(f"{'dataset':22s} {'nm':>5s}  " + "  ".join(f"{c:>9s}" for c in
          ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "nucleus"]))
    print("-" * 120)
    tot = {}
    for r in sorted(rows, key=lambda x: -(len(x["segs"]))):
        if not r["classes"]:
            continue
        cells = []
        for c in ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "nucleus"]:
            d = r["classes"].get(c)
            if not d or d.get("error"):
                cells.append(f"{'-':>9s}")
            else:
                cells.append(f"{d['n']:>9d}")
                tot[c] = tot.get(c, 0) + d["n"]
        print(f"{r['dataset']:22s} {r.get('counted_at_nm', 0):5.0f}  " + "  ".join(cells))
    print(f"{'TOTAL':22s} {'':5s}  " + "  ".join(
        f"{tot.get(c, 0):>9d}" for c in
        ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "nucleus"]))

    cells_known = [r["cells"] for r in rows if isinstance(r.get("cells"), int)]
    print(f"\ncells counted: {sum(cells_known)} across {len(cells_known)} datasets "
          f"({len(rows) - len(cells_known)} without a nucleus label)")
    for t in ("fine", "mid", "coarse"):
        n = [r["dataset"] for r in rows if t in r["tiers"]]
        print(f"  tier {t:6s} ({TIER_NM[t]:.0f} nm): {len(n)} datasets  {sorted(n)}")
    bad = [r["dataset"] for r in rows if r.get("error")]
    if bad:
        print(f"\nunreadable or partial: {bad}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
