import collections, glob, json, os, sys
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{ROOT}/release"
RAY = ("depth_below_steps", "own_occupancy_below_steps", "thickness_below_steps")
DTYPE = {"em": "uint8", "inst_face": "int32", "cls_face": "uint8", "mask": "bool",
         "clipped": "bool", "depth_below_steps": "uint16",
         "own_occupancy_below_steps": "uint16", "thickness_below_steps": "uint16"}


def main():
    shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)

    depth_of = {}
    bp = f"{REL}/boxes.csv"
    if os.path.exists(bp):
        import csv
        for r in csv.DictReader(open(bp)):
            try:
                depth_of[r["box"]] = int(r["shape_z"])
            except Exception:
                pass

    files = sorted(f for f in glob.glob(f"{REL}/blockface/*/faces/*.npz") if "/._" not in f)
    files = files[shard::nshard]
    print(f"shard {shard}/{nshard}: {len(files):,} planes, "
          f"{len(depth_of):,} box depths known", flush=True)

    fail = collections.Counter()
    ex = collections.defaultdict(list)
    px_fg = px_tot = 0
    maxstep = 0
    n = 0

    for i, f in enumerate(files):
        if i % 2000 == 0 and i:
            print(f"  {i:,}/{len(files):,}  failures {sum(fail.values())}", flush=True)
        rel = os.path.relpath(f, REL)
        try:
            z = np.load(f, allow_pickle=False)
        except Exception as e:
            fail["unreadable"] += 1; ex["unreadable"].append(rel); continue
        n += 1
        names = set(z.files)

        miss = set(DTYPE) - names
        if miss:
            fail["missing_array"] += 1; ex["missing_array"].append(f"{rel} {sorted(miss)}")
            continue
        for k, want in DTYPE.items():
            if str(z[k].dtype) != want:
                fail["dtype"] += 1; ex["dtype"].append(f"{rel} {k} {z[k].dtype}"); break

        shapes = {z[k].shape for k in DTYPE}
        if len(shapes) != 1:
            fail["shape"] += 1; ex["shape"].append(f"{rel} {shapes}"); continue

        try:
            m = json.loads(str(z["meta"]))
            px = float(m["pixel_size_nm"]); zs = float(m["z_step_nm"]); m["classes"]
            if px <= 0 or zs <= 0:
                raise ValueError("non-positive spacing")
        except Exception as e:
            fail["meta"] += 1; ex["meta"].append(f"{rel} {e}"); continue

        d, o, t = (z[k].astype(np.int64) for k in RAY)
        msk, ef, clp = z["mask"], z["inst_face"], z["clipped"]
        fg = ef > 0
        px_tot += int(msk.size); px_fg += int(fg.sum())
        maxstep = max(maxstep, int(t.max()) if t.size else 0)

        if (d[fg] > o[fg]).any():
            fail["order_depth_gt_own"] += 1; ex["order_depth_gt_own"].append(rel)
        if (o[fg] > t[fg]).any():
            fail["order_own_gt_thickness"] += 1; ex["order_own_gt_thickness"].append(rel)
        if not np.array_equal(msk, fg):
            bad = int((msk ^ fg).sum())
            fail["mask_ne_exposed"] += 1
            ex["mask_ne_exposed"].append(f"{rel} {bad} px")
        if (d[~fg] != 0).any():
            fail["nonzero_off_mask"] += 1; ex["nonzero_off_mask"].append(rel)
        if fg.any() and (d[fg] == 0).any():
            fail["zero_on_mask"] += 1; ex["zero_on_mask"].append(rel)

        Z = depth_of.get(os.path.basename(os.path.dirname(os.path.dirname(f))))
        if Z:
            try:
                zi = int(os.path.basename(f).rsplit("_z", 1)[1].split(".")[0])
            except Exception:
                zi = None
            if zi is not None:
                avail = Z - zi
                if (t > avail).any():
                    fail["exceeds_block"] += 1
                    ex["exceeds_block"].append(f"{rel} max {int(t.max())} avail {avail}")
                reach = fg & (d == avail)
                if not np.array_equal(clp & fg, reach):
                    dis = int(((clp & fg) ^ reach).sum())
                    fail["clipped_mismatch"] += 1
                    ex["clipped_mismatch"].append(f"{rel} {dis} px")

    print(f"\nshard {shard}: planes read {n:,}")
    print(f"  pixels {px_tot:,}   foreground {px_fg:,}")
    print(f"  largest step count {maxstep:,} against a uint16 ceiling of 65,535")
    if not fail:
        print("  no failures")
    for k, v in fail.most_common():
        print(f"  {k:<26s} {v:>8,d} planes")
        for e in ex[k][:3]:
            print(f"      {e}")
    json.dump(dict(shard=shard, planes=n, pixels=px_tot, foreground=px_fg,
                   max_step=maxstep, failures=dict(fail),
                   examples={k: v[:20] for k, v in ex.items()}),
              open(f"{ROOT}/cache/conformance_{shard:02d}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
