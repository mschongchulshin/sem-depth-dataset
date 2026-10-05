import glob, json, os, sys
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"
SPHERE = 4.0 / 3.0 / np.sqrt(np.pi)
BAD = ("touches_face_border", "clipped_at_block_bottom")
EXTRA = ("depth_below_nm_median", "thickness_below_nm_median", "face_feret_nm",
         "own_occupancy_below_nm_median")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/cache/inst_bf_exact_full.jsonl"

    plan = {b["name"]: b for b in json.load(open(f"{ROOT}/corpus/plan.json"))}
    print(f"plan: {len(plan):,} boxes", flush=True)

    exact, truncated = {}, set()
    for line in open(f"{ROOT}/cache/exact_volume.jsonl"):
        r = json.loads(line)
        k = (r["box"], int(r["instance"]))
        exact[k] = r["v_exact_nm3"]
        if r.get("truncated"):
            truncated.add(k)
    print(f"recount: {len(exact):,} instances over "
          f"{len(set(b for b, _ in exact)):,} boxes, {len(truncated):,} truncated", flush=True)

    files = sorted(f for f in glob.glob(f"{ROOT}/blockface/*/faces/*.npz") if "/._" not in f)
    print(f"faces: {len(files):,}", flush=True)

    n_rows = n_ub = n_files = n_norecount = 0
    boxes = set()
    with open(out, "w") as fh:
        for i, f in enumerate(files):
            if i % 2000 == 0 and i:
                print(f"  {i:,}/{len(files):,}  rows={n_rows:,}", flush=True)
            try:
                m = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
            except Exception:
                continue
            n_files += 1
            box = os.path.basename(os.path.dirname(os.path.dirname(f)))
            boxes.add(box)
            p = plan.get(box, {})
            px = float(m.get("pixel_size_nm", 0) or 0)
            for sid, rec in m.get("instances", {}).items():
                area = rec.get("face_area_nm2")
                stored = rec.get("volume_nm3")
                if not area or area <= 0 or not stored:
                    continue
                k = (box, int(sid))
                v = exact.get(k)
                if v is None or v <= 0:
                    n_norecount += 1
                    continue
                bad = any(rec.get(fl) for fl in BAD) or k in truncated
                r = dict(geometry="blockface", file=os.path.relpath(f, ROOT), box=box,
                         dataset=p.get("dataset", "?"), tier=p.get("tier", "?"),
                         level=p.get("level", "?"), cell_id=p.get("cell_id"),
                         sampling=p.get("sampling", "?"),
                         instance=int(sid), organelle=rec.get("organelle", "?"),
                         px_nm=px, area_nm2=float(area),
                         volume_nm3=float(v), volume_stored_nm3=float(stored),
                         unbiased=(not bad))
                r["v_stereo"] = float(area ** 1.5 * SPHERE)
                r["err_stereo"] = abs(r["v_stereo"] - v) / v
                for key in EXTRA:
                    if rec.get(key) is not None:
                        r[key] = rec[key]
                fh.write(json.dumps(r) + "\n")
                n_rows += 1
                n_ub += (not bad)

    print(f"\nfiles read    {n_files:,}")
    print(f"boxes         {len(boxes):,}")
    print(f"rows          {n_rows:,}   unbiased {n_ub:,} ({100*n_ub/max(n_rows,1):.1f}%)")
    print(f"dropped, no recount available  {n_norecount:,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
