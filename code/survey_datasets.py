import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_openorganelle import list_children


def probe(ds):
    rec = dict(dataset=ds, n_seg=0, segs=[], n_crops=0, error=None)
    try:
        for stem in (f"{ds}/{ds}.n5/labels/", f"{ds}/{ds}.zarr/recon-1/labels/"):
            try:
                kids = list_children(stem)
            except Exception:
                continue
            segs = [k for k in kids if k.endswith("_seg")]
            if segs and not rec["segs"]:
                rec["segs"] = sorted(segs)
                rec["n_seg"] = len(segs)
            if "groundtruth" in kids:
                try:
                    crops = list_children(stem + "groundtruth/")
                    rec["n_crops"] = len([c for c in crops if c.startswith("crop")])
                except Exception:
                    pass
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Volumes/One Touch/em-depth-dataset/cache/survey.json")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    datasets = sorted(d for d in list_children("") if d)
    print(f"{len(datasets)} datasets in the bucket, probing...", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(probe, datasets))

    usable = [r for r in rows if r["n_seg"] or r["n_crops"]]
    print(f"\n{len(usable)} of {len(datasets)} carry organelle ground truth\n")
    print(f"{'dataset':32s} {'_seg':>5s} {'crops':>6s}  organelles")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: (-x["n_seg"], -x["n_crops"], x["dataset"])):
        if not (r["n_seg"] or r["n_crops"]):
            continue
        names = ", ".join(s.replace("_seg", "") for s in r["segs"][:10])
        if r["n_seg"] > 10:
            names += f", +{r['n_seg'] - 10} more"
        print(f"{r['dataset']:32s} {r['n_seg']:5d} {r['n_crops']:6d}  {names}")

    bare = [r["dataset"] for r in rows if not (r["n_seg"] or r["n_crops"])]
    print(f"\nraw EM only, no organelle labels ({len(bare)}):")
    for i in range(0, len(bare), 4):
        print("  " + "  ".join(f"{b:28s}" for b in bare[i:i + 4]))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
