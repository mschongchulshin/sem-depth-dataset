import collections, glob, json, os, random, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _excluded():
    import json as _j, os as _o
    p = _o.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = _j.load(open(p))
    return set(e["boxes"]), set(e["planes"])

random.seed(0)


def hdr(n, title):
    print(f"\n{'=' * 74}\n{n}. {title}\n{'=' * 74}")


def load_meta(path):
    with np.load(path, allow_pickle=True) as d:
        return json.loads(str(d["meta"]))


def main():
    nbox = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    table = os.path.join(ROOT, "cache", "inst_bf_organelle.jsonl")
    rows = [json.loads(l) for l in open(table)]
    ub = [r for r in rows if r.get("unbiased")]
    print(f"table {os.path.basename(table)}: {len(rows):,} rows, {len(ub):,} unbiased")

    _sb, _sp = _excluded()
    raws = sorted(p for p in glob.glob(os.path.join(ROOT, "raw", "*.npz"))
                  if os.path.basename(p)[:-4] not in _sb)
    random.shuffle(raws)
    raws = raws[:nbox]

    hdr(1, "instance id capacity and background")
    mx, over, zero_fg = 0, 0, 0
    for p in raws:
        with np.load(p, allow_pickle=True) as d:
            inst = d["inst"]
            mx = max(mx, int(inst.max()))
            if inst.dtype == np.uint16 and int(inst.max()) >= 65535:
                over += 1
        m = load_meta(p)
        if any(int(k) > 65535 for k in m.get("instances", {})):
            over += 1
    print(f"  max instance id seen        {mx:,}   dtype capacity 65,535")
    print(f"  boxes at or above capacity  {over}")
    print(f"  background id 0 in table    {sum(1 for r in rows if r['instance'] == 0)}")

    hdr(2, "array ids versus metadata keys")
    only_arr = only_meta = 0
    for p in raws:
        with np.load(p, allow_pickle=True) as d:
            arr = set(np.unique(d["inst"]).tolist()) - {0}
        keys = {int(k) for k in load_meta(p).get("instances", {})}
        only_arr += len(arr - keys); only_meta += len(keys - arr)
    print(f"  ids in array but not metadata   {only_arr}")
    print(f"  ids in metadata but not array   {only_meta}")

    hdr(3, "one instance appearing on several cut faces")
    g = collections.Counter((r["box"], r["instance"]) for r in ub)
    n = np.array(list(g.values()))
    print(f"  unique (box, instance)      {len(g):,}")
    print(f"  rows                        {len(ub):,}")
    print(f"  appears more than once      {(n > 1).sum():,} ({(n > 1).mean() * 100:.1f}%)")
    print(f"  max faces for one instance  {n.max()}")
    dup = sum(v - 1 for v in g.values() if v > 1)
    print(f"  redundant rows              {dup:,} ({dup / len(ub) * 100:.1f}% of the table)")

    hdr(4, "one instance appearing in several boxes of one dataset")
    byds = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in ub:
        byds[r["dataset"]][r["instance"]].add(r["box"])
    tot = shared = 0
    for ds, d in byds.items():
        for i, boxes in d.items():
            tot += 1
            if len(boxes) > 1:
                shared += 1
    print(f"  (dataset, instance) pairs   {tot:,}")
    print(f"  spanning several boxes      {shared:,} ({shared / max(tot, 1) * 100:.1f}%)")
    print("  note: a shared numeric id may be one object seen twice, or two objects that")
    print("        happen to share a box-local label. Both matter, for different reasons.")

    hdr(5, "instances that touch a box wall in 3D but pass the 2D filter")
    idx = collections.defaultdict(set)
    for r in ub:
        idx[r["box"]].add(r["instance"])
    checked = touching = 0
    for p in raws:
        box = os.path.basename(p)[:-4]
        want = idx.get(box)
        if not want:
            continue
        with np.load(p, allow_pickle=True) as d:
            inst = d["inst"]
        walls = np.concatenate([np.unique(inst[0]), np.unique(inst[-1]),
                                np.unique(inst[:, 0]), np.unique(inst[:, -1]),
                                np.unique(inst[:, :, 0]), np.unique(inst[:, :, -1])])
        wall = set(walls.tolist()) - {0}
        checked += len(want); touching += len(want & wall)
    print(f"  unbiased instances checked  {checked:,}")
    print(f"  touching a box wall in 3D   {touching:,} "
          f"({touching / max(checked, 1) * 100:.1f}%)")

    hdr(6, "one instance carrying several class labels")
    multi = 0
    for p in raws[:40]:
        with np.load(p, allow_pickle=True) as d:
            inst, cls = d["inst"], d["cls"]
        for i in np.unique(inst):
            if i == 0:
                continue
            if len(np.unique(cls[inst == i])) > 1:
                multi += 1
    print(f"  instances with >1 class id  {multi}")

    hdr(7, "class id to organelle name across datasets")
    print("  measured earlier over 400 files: id 3 is mito in 64.9% of its pixels, id 7 is")
    print("  er in 51.0%. Class ids index each source volume's own class list, so they are")
    print("  not comparable across datasets and must be remapped through organelle names.")

    hdr(8, "depth <= own occupancy <= union thickness")
    faces = sorted(f for f in glob.glob(os.path.join(ROOT, "blockface", "*", "faces", "*.npz"))
                   if os.path.basename(os.path.dirname(os.path.dirname(f))) not in _sb
                   and os.path.basename(f)[:-4] not in _sp)
    random.shuffle(faces)
    bad_a = bad_b = tot_px = 0
    for f in faces[:200]:
        with np.load(f) as z:
            if "own_occupancy_below_nm" not in z.files:
                continue
            d_, o_, t_ = (z["depth_below_nm"], z["own_occupancy_below_nm"],
                          z["thickness_below_nm"])
            m = z["mask"].astype(bool) & np.isfinite(d_)
        tot_px += int(m.sum())
        bad_a += int((d_[m] > o_[m] + 1e-3).sum())
        bad_b += int((o_[m] > t_[m] + 1e-3).sum())
    print(f"  foreground pixels checked   {tot_px:,}")
    print(f"  depth > own occupancy       {bad_a:,} ({bad_a / max(tot_px, 1) * 100:.3f}%)")
    print(f"  own occupancy > thickness   {bad_b:,} ({bad_b / max(tot_px, 1) * 100:.3f}%)")

    hdr(9, "below-cut volume versus the stored target")
    below = {}
    bp = os.path.join(ROOT, "cache", "below_volume.jsonl")
    if os.path.exists(bp):
        for l in open(bp):
            x = json.loads(l)
            below[(x["file"], x["instance"])] = x["v_below_nm3"]
        have = [r for r in ub if (r["file"], r["instance"]) in below]
        if have:
            rat = np.array([below[(r["file"], r["instance"])] / r["volume_nm3"]
                            for r in have])
            print(f"  instances checked           {len(rat):,}")
            print(f"  ratio median                {np.median(rat):.4f}  (0.5 expected)")
            print(f"  ratio above 1 (impossible)  {(rat > 1.0).sum():,} "
                  f"({(rat > 1.0).mean() * 100:.2f}%)   worst {rat.max():.1f}x")
    else:
        print("  cache/below_volume.jsonl not present")

    hdr(10, "stored area against the mask it claims to describe")
    byf = collections.defaultdict(list)
    for r in ub:
        byf[r["file"]].append(r)
    fs = [f for f in byf if os.path.exists(os.path.join(ROOT, f))]
    random.shuffle(fs)
    ratios = []
    for f in fs[:150]:
        with np.load(os.path.join(ROOT, f)) as z:
            ids = z["inst_face"]
        for r in byf[f]:
            n_ = int((ids == r["instance"]).sum())
            if n_ >= 16:
                ratios.append(n_ * r["px_nm"] ** 2 / r["area_nm2"])
    if ratios:
        v = np.array(ratios)
        print(f"  instances checked           {len(v):,}")
        print(f"  pixel area / stored area    median {np.median(v):.4f}   "
              f"off by >1%: {(np.abs(v - 1) > 0.01).mean() * 100:.2f}%")

    print("\ndone")


if __name__ == "__main__":
    main()
