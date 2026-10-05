import collections, csv, glob, json, os
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"
REL = f"{ROOT}/release"

WHAT = {
    "jrc_hela-1": ("HeLa", "cultured line"), "jrc_hela-2": ("HeLa", "cultured line"),
    "jrc_hela-3": ("HeLa", "cultured line"),
    "jrc_hela-bfa": ("HeLa, brefeldin A treated", "cultured line"),
    "jrc_jurkat-1": ("Jurkat T", "cultured line"),
    "jrc_macrophage-2": ("macrophage", "cultured line"),
    "jrc_cos7-11": ("COS-7", "cultured line"),
    "jrc_mus-liver": ("mouse liver", "tissue"),
    "jrc_mus-liver-3": ("mouse liver", "tissue"),
    "jrc_mus-thymus-1": ("mouse thymus", "tissue"),
    "jrc_mus-pancreas-4": ("mouse pancreas", "tissue"),
    "jrc_mus-skin-1": ("mouse skin", "tissue"),
    "jrc_mus-kidney-3": ("mouse kidney", "tissue"),
    "jrc_mus-heart-1": ("mouse heart", "tissue"),
    "jrc_mus-hippocampus-1": ("mouse hippocampus", "tissue"),
    "jrc_choroid-plexus-2": ("mouse choroid plexus", "tissue"),
    "aic_desmosome-2": ("desmosome-forming epithelium", "tissue"),
    "aic_desmosome-3": ("desmosome-forming epithelium", "tissue"),
}


def main():
    planes = collections.Counter()
    for f in glob.glob(f"{REL}/blockface/*/faces/*.npz"):
        if "/._" in f:
            continue
        planes[os.path.basename(os.path.dirname(os.path.dirname(f)))] += 1
    print(f"deposited planes over {len(planes):,} boxes", flush=True)

    rows = []
    for i, p in enumerate(sorted(glob.glob(f"{ROOT}/raw/*.npz"))):
        if i % 400 == 0 and i:
            print(f"  {i:,}", flush=True)
        box = os.path.basename(p)[:-4]
        if box not in planes:
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                m = json.loads(str(d["meta"]))
                shape = d["inst"].shape
        except Exception:
            continue
        st = m.get("box_start_zyx") or [None] * 3
        sp = m.get("box_stop_zyx") or [None] * 3
        vx = m.get("voxel_size_nm") or [None] * 3
        inst = m.get("instances", {})
        cls = sorted({r.get("organelle") for r in inst.values()} - {None})
        rows.append(dict(
            box=box, source_volume=box.split("__")[0],
            tier=box.split("__")[1].rsplit("_b", 1)[0] if "__" in box else "?",
            level=m.get("level"),
            voxel_z_nm=vx[0], voxel_y_nm=vx[1], voxel_x_nm=vx[2],
            start_z=st[0], start_y=st[1], start_x=st[2],
            stop_z=sp[0], stop_y=sp[1], stop_x=sp[2],
            shape_z=shape[0], shape_y=shape[1], shape_x=shape[2],
            n_planes=planes[box], n_instances=len(inst),
            n_truncated=sum(1 for r in inst.values() if r.get("truncated")),
            classes="|".join(cls)))

    cols = list(rows[0].keys())
    with open(f"{REL}/boxes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r[c] is None else r[c]) for c in cols})
    print(f"\nboxes.csv  {len(rows):,} boxes, {os.path.getsize(f'{REL}/boxes.csv')/1e3:.0f} KB")

    byv = collections.defaultdict(lambda: dict(boxes=0, planes=0, inst=0, trunc=0,
                                               px=set(), zs=set(), lv=set(), tier=set(),
                                               cls=set()))
    for r in rows:
        d = byv[r["source_volume"]]
        d["boxes"] += 1; d["planes"] += r["n_planes"]
        d["inst"] += r["n_instances"]; d["trunc"] += r["n_truncated"]
        if r["voxel_x_nm"]: d["px"].add(float(r["voxel_x_nm"]))
        if r["voxel_z_nm"]: d["zs"].add(round(float(r["voxel_z_nm"]), 2))
        d["lv"].add(r["level"]); d["tier"].add(r["tier"])
        d["cls"].update(c for c in r["classes"].split("|") if c)

    scols = ["source_volume", "specimen", "kind", "boxes", "planes", "instances",
             "truncated_instances", "pixel_sizes_nm", "z_steps_nm", "levels", "tiers",
             "n_classes", "classes"]
    with open(f"{REL}/sources.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(scols)
        for v in sorted(byv, key=lambda k: -byv[k]["planes"]):
            d = byv[v]
            spec, kind = WHAT.get(v, ("", ""))
            w.writerow([v, spec, kind, d["boxes"], d["planes"], d["inst"], d["trunc"],
                        "|".join(f"{x:g}" for x in sorted(d["px"])),
                        "|".join(f"{x:g}" for x in sorted(d["zs"])),
                        "|".join(sorted(x for x in d["lv"] if x)),
                        "|".join(sorted(d["tier"])),
                        len(d["cls"]), "|".join(sorted(d["cls"]))])
    print(f"sources.csv  {len(byv)} source volumes")

    print(f"\n  {'source volume':<24s}{'specimen':<30s}{'boxes':>7s}{'planes':>8s}"
          f"{'px nm':>12s}{'cls':>5s}")
    for v in sorted(byv, key=lambda k: -byv[k]["planes"]):
        d = byv[v]
        spec = WHAT.get(v, ("", ""))[0]
        print(f"  {v:<24s}{spec:<30s}{d['boxes']:>7,d}{d['planes']:>8,d}"
              f"{'|'.join(f'{x:g}' for x in sorted(d['px'])):>12s}{len(d['cls']):>5d}")


if __name__ == "__main__":
    main()
