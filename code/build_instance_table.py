import argparse, json, os, glob, sys
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPHERE = 4.0 / 3.0 / np.sqrt(np.pi)

GEO = {
    "blockface":   dict(sub="blockface",   sd="faces",    area="face_area_nm2",
                        bad=("touches_face_border", "clipped_at_block_bottom")),
    "thinsection": dict(sub="thinsection", sd="sections", area="profile_area_nm2",
                        bad=("touches_border", "clipped")),
    "surface":     dict(sub="corpus",      sd="views",    area="proj_area_nm2",
                        bad=("touches_border",)),
}
EXTRA = ("depth_below_nm_median", "thickness_below_nm_median", "face_feret_nm",
         "slab_thickness_nm_median", "total_thickness_nm_median", "section_fraction_median",
         "profile_feret_nm", "proj_feret_nm", "proj_major_nm", "proj_minor_nm",
         "proj_perimeter_nm", "occluded_fraction", "sphericity", "elongation", "flatness",
         "surface_area_nm2", "curve_length_nm", "proj_length_nm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="blockface", choices=sorted(GEO))
    ap.add_argument("--stride", type=int, default=1, help="take every Nth file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = GEO[a.geometry]
    plan = {b["name"]: b for b in json.load(open(f"{ROOT}/corpus/plan.json"))}

    truncated = set()
    for rp in glob.glob(f"{ROOT}/raw/*.npz"):
        try:
            with np.load(rp, allow_pickle=True) as d:
                rm = json.loads(str(d["meta"]))
        except Exception:
            continue
        bx = os.path.basename(rp)[:-4]
        for sid, rec in rm.get("instances", {}).items():
            if rec.get("truncated"):
                truncated.add((bx, int(sid)))
    print(f"truncated instances in raw metadata: {len(truncated):,}", flush=True)
    files = sorted(f for f in glob.glob(f"{ROOT}/{g['sub']}/*/{g['sd']}/*.npz")
                   if "/._" not in f)[::a.stride]
    print(f"{a.geometry}: {len(files):,} files", flush=True)

    rows, seen_bad, n_files = [], 0, 0
    for i, f in enumerate(files):
        if i % 2000 == 0 and i:
            print(f"  {i:,}/{len(files):,}  rows={len(rows):,}", flush=True)
        try:
            m = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        except Exception:
            continue
        n_files += 1
        box = os.path.basename(os.path.dirname(os.path.dirname(f)))
        p = plan.get(box, {})
        px = float(m.get("pixel_size_nm", 0) or 0)
        for sid, rec in m.get("instances", {}).items():
            v = rec.get("volume_nm3")
            area = rec.get(g["area"])
            if not v or not area or area <= 0:
                continue
            bad = any(rec.get(fl) for fl in g["bad"]) or (box, int(sid)) in truncated
            if bad:
                seen_bad += 1
            r = dict(geometry=a.geometry, file=os.path.relpath(f, ROOT), box=box,
                     dataset=p.get("dataset", "?"), tier=p.get("tier", "?"),
                     level=p.get("level", "?"), cell_id=p.get("cell_id"),
                     sampling=p.get("sampling", "?"),
                     instance=int(sid), organelle=rec.get("organelle", "?"),
                     px_nm=px, area_nm2=float(area), volume_nm3=float(v),
                     unbiased=(not bad))
            r["v_stereo"] = float(area ** 1.5 * SPHERE)
            r["err_stereo"] = abs(r["v_stereo"] - v) / v
            for k in EXTRA:
                if rec.get(k) is not None:
                    r[k] = rec[k]
            rows.append(r)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ub = sum(1 for r in rows if r["unbiased"])
    print(f"\nfiles read {n_files:,}")
    print(f"records    {len(rows):,}   unbiased {ub:,} ({100*ub/max(len(rows),1):.1f}%)")
    print(f"datasets   {len(set(r['dataset'] for r in rows))}")
    print(f"organelles {len(set(r['organelle'] for r in rows))}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
