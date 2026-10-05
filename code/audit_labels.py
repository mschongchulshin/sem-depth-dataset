import collections, glob, json, os, random, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    fs = sorted(glob.glob(os.path.join(ROOT, "raw", "*.npz")))
    random.seed(0); random.shuffle(fs)

    agree = collections.Counter()
    conf = collections.Counter()
    purity = []
    id_names = collections.defaultdict(set)
    for f in fs[:n]:
        try:
            with np.load(f, allow_pickle=True) as d:
                meta = json.loads(str(d["meta"]))
                inst, cls = d["inst"], d["cls"]
        except Exception:
            continue
        classes = meta.get("classes") or []
        recs = meta.get("instances", {})
        if not classes or not recs:
            continue
        mx = int(inst.max())
        flat = inst.ravel().astype(np.int64) * 256 + cls.ravel().astype(np.int64)
        pair = np.bincount(flat)
        nz = np.nonzero(pair)[0]
        byinst = collections.defaultdict(list)
        for code in nz:
            i, c = divmod(int(code), 256)
            if i > 0:
                byinst[i].append((int(pair[code]), c))
        counts = np.bincount(inst.ravel())
        for sid, rec in recs.items():
            i = int(sid)
            if i <= 0 or i >= len(counts) or counts[i] == 0 or rec.get("truncated"):
                continue
            pairs = byinst.get(i)
            if not pairs:
                continue
            tot = sum(p[0] for p in pairs)
            npx, dom = max(pairs)
            name = classes[dom - 1] if 0 < dom <= len(classes) else None
            purity.append(npx / tot)
            meta_name = rec.get("organelle")
            same_vol = counts[i] == rec.get("voxels_in_box")
            agree[(bool(same_vol), name == meta_name)] += 1
            if name != meta_name:
                conf[(meta_name, name)] += 1
            id_names[i].add(name)

    tot = sum(agree.values())
    print(f"instances checked {tot:,}")
    print("\nlabel from the class array versus the label in the metadata")
    for sv in (True, False):
        s = agree[(sv, True)] + agree[(sv, False)]
        if not s:
            continue
        lbl = "volume matches" if sv else "volume disagrees"
        print(f"  {lbl:<18s} n={s:>7,d}   label agrees {agree[(sv, True)] / s * 100:5.1f}%"
              f"   disagrees {agree[(sv, False)] / s * 100:5.1f}%")
    p = np.array(purity)
    print(f"\nclass purity within an instance   median {np.median(p):.3f}   "
          f"below 0.9: {np.mean(p < 0.9) * 100:.1f}%")
    if conf:
        print("\nmost common disagreements, metadata name -> array name")
        for (a, b), c in conf.most_common(8):
            print(f"  {str(a):<12s} -> {str(b):<12s} {c:>6,d}")
    multi = sum(1 for v in id_names.values() if len(v) > 1)
    print(f"\nnumeric ids mapping to more than one array-derived name across boxes: "
          f"{multi:,} of {len(id_names):,}")


if __name__ == "__main__":
    main()
