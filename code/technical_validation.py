import argparse, collections, glob, json, os, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LITERATURE_NM = {
    "mito": (600, 1400),
    "lyso": (200, 1000),
    "endo": (250, 400),
}


def hdr(n, title, invariant):
    print(f"\n{'=' * 78}")
    print(f"{n}. {title}")
    print(f"   invariant: {invariant}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--table", default="cache/inst_bf_exact.jsonl")
    ap.add_argument("--boxes", type=int, default=150)
    ap.add_argument("--faces", type=int, default=400)
    a = ap.parse_args()
    random.seed(0)

    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    ub = [r for r in rows if r.get("unbiased")]
    print(f"table {os.path.basename(a.table)}")
    print(f"  rows {len(rows):,}   unbiased {len(ub):,}   "
          f"unique instances {len({(r['box'], r['instance']) for r in ub}):,}")
    print(f"  source volumes {len({r['dataset'] for r in ub})}   "
          f"faces {len({r['file'] for r in ub}):,}   "
          f"classes {len({r['organelle'] for r in ub})}")

    skip_box, skip_plane = set(), set()
    if os.environ.get("DEPOSIT_EXCLUSIONS"):
        e = json.load(open(os.environ["DEPOSIT_EXCLUSIONS"]))
        skip_box, skip_plane = set(e["boxes"]), set(e["planes"])
        print(f"excluding {len(skip_box)} boxes and {len(skip_plane)} planes")

    raws = sorted(p for p in glob.glob(os.path.join(a.root, "raw", "*.npz"))
                  if os.path.basename(p)[:-4] not in skip_box)
    random.shuffle(raws)
    raws = raws[:a.boxes]

    hdr(1, "instance ids", "array ids and metadata keys are the same set, and fit the dtype")
    only_a = only_m = mx = 0
    for p in raws:
        try:
            with np.load(p, allow_pickle=True) as d:
                arr = set(np.unique(d["inst"]).tolist()) - {0}
                meta = json.loads(str(d["meta"]))
        except Exception:
            continue
        keys = {int(k) for k in meta.get("instances", {})}
        only_a += len(arr - keys); only_m += len(keys - arr)
        mx = max(mx, max(arr) if arr else 0)
    print(f"   ids only in the array      {only_a}")
    print(f"   ids only in the metadata   {only_m}")
    print(f"   largest id {mx:,} against a uint16 ceiling of 65,535")
    print(f"   background id 0 in table   {sum(1 for r in rows if r['instance'] == 0)}")

    hdr(2, "stored volume against a direct count",
        "volume_nm3 equals voxels in this box's array times its own voxel volume")
    ex = {}
    p2 = os.path.join(a.root, "cache", "exact_volume.jsonl")
    if os.path.exists(p2):
        bylv = collections.defaultdict(list)
        for l in open(p2):
            r = json.loads(l)
            if r.get("v_stored_nm3") and not r.get("truncated"):
                bylv[r["level"]].append(r["v_exact_nm3"] / r["v_stored_nm3"])
            ex[(r["box"], r["instance"])] = r["v_exact_nm3"]
        print(f"   {'level':<7s} {'n':>9s} {'exact/stored':>13s} {'off by >5%':>12s}")
        for lv in sorted(bylv):
            v = np.array(bylv[lv])
            print(f"   {lv:<7s} {len(v):>9,d} {np.median(v):>13.3f} "
                  f"{np.mean(np.abs(v - 1) > 0.05) * 100:>11.1f}%")
        print("   the corpus mixes pyramid levels while volumes were measured at one of them,")
        print("   so the table ships a recount in each box's own array as volume_nm3")

    hdr(3, "truncation", "an instance kept as unbiased touches no wall of its box")
    idx = collections.defaultdict(set)
    for r in ub:
        idx[r["box"]].add(r["instance"])
    checked = touching = 0
    for p in raws:
        want = idx.get(os.path.basename(p)[:-4])
        if not want:
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                inst = d["inst"]
        except Exception:
            continue
        wall = set()
        for sl in (inst[0], inst[-1], inst[:, 0], inst[:, -1], inst[:, :, 0], inst[:, :, -1]):
            wall |= set(np.unique(sl).tolist())
        wall.discard(0)
        checked += len(want); touching += len(want & wall)
    print(f"   unbiased instances checked  {checked:,}")
    print(f"   touching a wall in 3D       {touching:,} "
          f"({touching / max(checked, 1) * 100:.2f}%)")

    hdr(4, "organelle name", "the name in the metadata matches the class array for that id")
    agree = disagree = 0
    pur = []
    for p in raws[:60]:
        try:
            with np.load(p, allow_pickle=True) as d:
                inst, cls = d["inst"], d["cls"]
                meta = json.loads(str(d["meta"]))
        except Exception:
            continue
        classes = meta.get("classes") or []
        if not classes:
            continue
        counts = np.bincount(inst.ravel())
        flat = inst.ravel().astype(np.int64) * 256 + cls.ravel().astype(np.int64)
        pair = np.bincount(flat)
        byi = collections.defaultdict(list)
        for code in np.nonzero(pair)[0]:
            i, c = divmod(int(code), 256)
            if i > 0:
                byi[i].append((int(pair[code]), c))
        for sid, rec in meta.get("instances", {}).items():
            i = int(sid)
            if i <= 0 or i >= len(counts) or counts[i] == 0 or rec.get("truncated"):
                continue
            ps = byi.get(i)
            if not ps:
                continue
            npx, dom = max(ps)
            pur.append(npx / sum(x[0] for x in ps))
            name = classes[dom - 1] if 0 < dom <= len(classes) else None
            if name == rec.get("organelle"):
                agree += 1
            else:
                disagree += 1
    tot = agree + disagree
    print(f"   instances checked {tot:,}   agree {agree / max(tot, 1) * 100:.1f}%   "
          f"disagree {disagree / max(tot, 1) * 100:.1f}%")
    if pur:
        print(f"   class purity within one instance, median {np.median(pur):.3f}")

    hdr(5, "ray quantities", "depth <= own occupancy <= union thickness, per pixel")
    faces = sorted(f for f in glob.glob(os.path.join(a.root, "blockface", "*", "faces", "*.npz"))
                   if os.path.basename(os.path.dirname(os.path.dirname(f))) not in skip_box
                   and os.path.basename(f)[:-4] not in skip_plane)
    random.shuffle(faces)
    px = bad1 = bad2 = 0
    for f in faces[:a.faces]:
        try:
            with np.load(f) as z:
                if "own_occupancy_below_nm" not in z.files:
                    continue
                d_, o_, t_ = (z["depth_below_nm"], z["own_occupancy_below_nm"],
                              z["thickness_below_nm"])
                m = z["mask"].astype(bool) & np.isfinite(d_)
        except Exception:
            continue
        px += int(m.sum())
        bad1 += int((d_[m] > o_[m] + 1e-3).sum())
        bad2 += int((o_[m] > t_[m] + 1e-3).sum())
    print(f"   foreground pixels {px:,}")
    print(f"   depth > own occupancy      {bad1:,} ({bad1 / max(px, 1) * 100:.3f}%)")
    print(f"   own occupancy > thickness  {bad2:,} ({bad2 / max(px, 1) * 100:.3f}%)")

    hdr(6, "Delesse", "per-plane area fraction of a class equals its volume fraction")
    print("   run code/validate_stereology.py, which needs every face of a box rather than")
    print("   a scatter of faces across many boxes")

    hdr(7, "size against the literature", "equivalent-sphere diameter falls in the published range")
    byo = collections.defaultdict(list)
    for r in ub:
        byo[r["organelle"]].append(r["volume_nm3"])
    print(f"   {'class':<11s} {'n':>9s} {'median diam':>12s} {'published':>16s} {'verdict':>10s}")
    for o in sorted(byo, key=lambda k: -len(byo[k])):
        v = np.array(byo[o])
        if len(v) < 100:
            continue
        dia = (6 * np.median(v) / np.pi) ** (1 / 3)
        lit = LITERATURE_NM.get(o)
        if lit:
            ok = "within" if lit[0] <= dia <= lit[1] else ("below" if dia < lit[0] else "above")
            print(f"   {o:<11s} {len(v):>9,d} {dia:>11.0f}nm {f'{lit[0]}-{lit[1]}nm':>16s} "
                  f"{ok:>10s}")
        else:
            print(f"   {o:<11s} {len(v):>9,d} {dia:>11.0f}nm {'-':>16s} {'-':>10s}")


if __name__ == "__main__":
    main()
