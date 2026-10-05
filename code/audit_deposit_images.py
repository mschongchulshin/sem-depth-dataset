import hashlib, json, os, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEP = f"{ROOT}/release/blockface"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = []
    skip, skip_plane = set(), set()
    if os.environ.get("AUDIT_EXCLUDE"):
        e = json.load(open(os.environ["AUDIT_EXCLUDE"]))
        if isinstance(e, dict):
            skip, skip_plane = set(e.get("boxes", [])), set(e.get("planes", []))
        else:
            skip = set(e)
        print(f"excluding {len(skip)} boxes and {len(skip_plane)} planes", flush=True)
    for d in sorted(os.listdir(DEP)):
        if d.startswith(".") or d in skip:
            continue
        fd = f"{DEP}/{d}/faces"
        if not os.path.isdir(fd):
            continue
        for f in sorted(os.listdir(fd)):
            if f.endswith(".npz") and not f.startswith(".") and f[:-4] not in skip_plane:
                files.append((d, f"{fd}/{f}"))
    if limit:
        files = files[:limit]
    print(f"{len(files):,} archives over "
          f"{len({d for d, _ in files})} box directories", flush=True)

    em_hash = defaultdict(list)
    dep_hash = defaultdict(list)
    shapes = Counter()
    bad = defaultdict(list)
    stats = []

    for i, (box, path) in enumerate(files):
        if i % 2000 == 0 and i:
            print(f"  {i:,}/{len(files):,}", flush=True)
        name = os.path.basename(path)[:-4]
        try:
            z = np.load(path, allow_pickle=False)
            em = z["em"]; m = z["mask"]; dp = z["depth_below_steps"]
            cl = z["clipped"]; inst = z["inst_face"]
        except Exception as e:
            bad["unreadable"].append((name, str(e)[:60])); continue

        if not (em.shape == m.shape == dp.shape == cl.shape == inst.shape):
            bad["shape mismatch"].append((name, f"{em.shape} {m.shape} {dp.shape}"))
            continue
        shapes[em.shape] += 1

        em_hash[hashlib.md5(em.tobytes()).hexdigest()].append(name)
        dep_hash[hashlib.md5(dp.tobytes()).hexdigest()].append(name)

        sd = float(em.std())
        lo = float((em == 0).mean()); hi = float((em == 255).mean())
        frac = float(m.mean())
        if sd < 1.0:
            bad["blank image"].append((name, f"sd {sd:.3f}"))
        if lo + hi > 0.5:
            bad["saturated"].append((name, f"black {lo:.2f} white {hi:.2f}"))
        if frac < 0.005:
            bad["almost no label"].append((name, f"mask {frac:.4f}"))
        if m.any():
            dv = dp[m]
            if dv.size and dv.min() == dv.max():
                bad["constant depth"].append((name, f"all {int(dv.min())}"))
            if (dv == 0).any():
                bad["zero depth inside mask"].append(
                    (name, f"{int((dv == 0).sum())} px"))
        if (dp[~m] != 0).any():
            bad["depth outside mask"].append((name, f"{int((dp[~m] != 0).sum())} px"))
        stats.append((sd, frac, float(dp[m].mean()) if m.any() else 0.0))

    S = np.array(stats) if stats else np.zeros((0, 3))
    dup_em = {k: v for k, v in em_hash.items() if len(v) > 1}
    dup_dp = {k: v for k, v in dep_hash.items() if len(v) > 1}
    cross = {k: v for k, v in dup_em.items()
             if len({n.rsplit("_z", 1)[0] for n in v}) > 1}

    print(f"\n{'=' * 70}\nREPORT")
    print(f"  archives read        {len(stats):,}")
    print(f"  distinct shapes      {len(shapes)}   commonest "
          f"{', '.join(f'{s[0]}x{s[1]} n={c}' for s, c in shapes.most_common(3))}")
    print(f"\n  identical micrographs {sum(len(v) for v in dup_em.values()):,} planes "
          f"in {len(dup_em):,} groups")
    print(f"    of which across different boxes  {sum(len(v) for v in cross.values()):,} "
          f"planes in {len(cross):,} groups")
    print(f"  identical depth maps  {sum(len(v) for v in dup_dp.values()):,} planes "
          f"in {len(dup_dp):,} groups")
    for k in sorted(bad):
        print(f"  {k:24s} {len(bad[k]):,}")
    if S.size:
        print(f"\n  image sd        median {np.median(S[:, 0]):.1f}  "
              f"min {S[:, 0].min():.2f}  max {S[:, 0].max():.1f}")
        print(f"  labelled share  median {np.median(S[:, 1]):.3f}  "
              f"min {S[:, 1].min():.4f}  max {S[:, 1].max():.3f}")
        print(f"  mean depth      median {np.median(S[:, 2]):.1f} steps")
    for k in sorted(bad):
        if bad[k]:
            print(f"\n  {k}, first five:")
            for n, why in bad[k][:5]:
                print(f"    {n}   {why}")
    if cross:
        print("\n  cross-box identical micrographs, first three groups:")
        for v in list(cross.values())[:3]:
            print(f"    {len(v)} planes: {', '.join(v[:3])}")

    out = dict(archives=len(stats), shapes={str(k): v for k, v in shapes.items()},
               dup_em_groups=len(dup_em), dup_em_planes=sum(len(v) for v in dup_em.values()),
               cross_box_groups=len(cross),
               cross_box_planes=sum(len(v) for v in cross.values()),
               dup_depth_groups=len(dup_dp),
               problems={k: len(v) for k, v in bad.items()},
               problem_names={k: [n for n, _ in v] for k, v in bad.items()},
               examples={k: v[:20] for k, v in bad.items()},
               cross_box_examples=[v[:5] for v in list(cross.values())[:20]])
    dst = os.environ.get("AUDIT_OUT", f"{ROOT}/cache/deposit_image_audit.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
