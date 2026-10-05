import argparse
import collections
import json
from pathlib import Path

import numpy as np


def list_views(root):
    return sorted(p for p in Path(root).glob("views/*.npz") if not p.name.startswith("._"))

def add_axis_ratio(r):
    axes, pm = r.get("axes_nm"), r.get("proj_major_nm")
    if axes and pm and pm > 0:
        r["major_axis_ratio"] = float(axes[0] / pm)
    return r


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = list_views(a.root)
    if not files:
        raise SystemExit(f"no views under {a.root}")

    by = collections.defaultdict(list)
    seen = collections.defaultdict(set)
    n_views = 0
    for f in files:
        meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
        n_views += 1
        for sid, r in meta["instances"].items():
            if r.get("touches_border"):
                continue
            org = r.get("organelle", "?")
            by[org].append(add_axis_ratio(r))
            seen[org].add(sid)

    print(f"{n_views} views, {sum(len(v) for v in by.values())} instance observations, "
          f"{sum(len(v) for v in seen.values())} unique instances\n")

    hdr = (f"{'organelle':10s} {'obs':>5s} {'uniq':>5s} {'vol um3':>9s} {'area um2':>9s} "
           f"{'curve nm':>9s} {'sphericity':>10s} {'elong':>6s} "
           f"{'foresh':>7s} {'n_val':>6s} {'feretR':>7s} {'axRatio':>8s} {'areaRatio':>10s} "
           f"{'vol2Derr':>9s} {'occl':>6s}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for org, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        valid = [r for r in rows if r.get("foreshortening_valid")]
        rec = dict(
            observations=len(rows), unique=len(seen[org]),
            volume_um3=med(rows, "volume_nm3") * 1e-9,
            area_um2=med(rows, "surface_area_nm2") * 1e-6,
            curve_length_nm=med(rows, "curve_length_nm"),
            sphericity=med(rows, "sphericity"), elongation=med(rows, "elongation"),
            foreshortening=med(valid, "foreshortening") if valid else float("nan"),
            n_foreshortening_valid=len(valid),
            feret_ratio=med(rows, "feret_ratio"),
            major_axis_ratio=med(rows, "major_axis_ratio"),
            area_ratio=med(rows, "area_ratio"),
            volume_2d_error=med(rows, "volume_2d_error"),
            occluded_fraction=med(rows, "occluded_fraction"))
        summary[org] = rec
        print(f"{org:10s} {rec['observations']:5d} {rec['unique']:5d} "
              f"{rec['volume_um3']:9.4f} {rec['area_um2']:9.4f} {rec['curve_length_nm']:9.0f} "
              f"{rec['sphericity']:10.3f} {rec['elongation']:6.2f} "
              f"{rec['foreshortening']:7.3f} {rec['n_foreshortening_valid']:6d} "
              f"{rec['feret_ratio']:7.3f} {rec['major_axis_ratio']:8.3f} "
              f"{rec['area_ratio']:10.3f} {rec['volume_2d_error']:9.2f} "
              f"{rec['occluded_fraction']:6.2f}")

    allrows = [r for rows in by.values() for r in rows]
    allvalid = [r for r in allrows if r.get("foreshortening_valid")]
    print(f"\nall instances: median 2D-only volume error = {med(allrows, 'volume_2d_error'):.2f}")
    if allvalid:
        fs = [r["foreshortening"] for r in allvalid]
        print(f"gated foreshortening: n={len(fs)} median={np.median(fs):.3f} "
              f"p90={np.percentile(fs, 90):.3f} max={np.max(fs):.3f} min={np.min(fs):.3f}")
        bad = [x for x in fs if x < 1.0]
        print(f"  below the geometric floor of 1.0: {len(bad)} of {len(fs)} "
              f"({100 * len(bad) / len(fs):.1f} %)")
    for name in ("feret_ratio", "major_axis_ratio"):
        vv = np.array([r[name] for r in allrows if r.get(name) is not None])
        if vv.size:
            print(f"{name}: n={len(vv)} median={np.median(vv):.3f} "
                  f"p90={np.percentile(vv, 90):.3f} max={vv.max():.3f} min={vv.min():.3f}")
            print(f"  below the geometric floor of 1.0: {int((vv < 1.0).sum())} of {len(vv)} "
                  f"({100 * (vv < 1.0).mean():.1f} %)")
    ar = [r["major_axis_ratio"] for r in allrows if r.get("major_axis_ratio") is not None]
    if ar:
        ar = np.array(ar)
        print(f"major-axis ratio: n={len(ar)} median={np.median(ar):.3f} "
              f"p90={np.percentile(ar, 90):.3f} max={ar.max():.3f} min={ar.min():.3f}")
        print(f"  below the geometric floor of 1.0: {int((ar < 1.0).sum())} of {len(ar)} "
              f"({100 * (ar < 1.0).mean():.1f} %)  "
              f"[worst deficit {max(0.0, 1.0 - ar.min()):.4f}]")

    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
