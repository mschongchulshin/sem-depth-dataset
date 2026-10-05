import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/release"

PLAN = {
    "ship": {
        "cache/inst_bf_exact_full.jsonl":
            ("instances.parquet", "one row per cut plane and instance, the scoring table"),
        "cache/exact_volume.jsonl":
            ("volumes.parquet", "per box and instance, the volume recounted in that box's "
                                "own array. Provenance for volume_nm3."),
        "cache/below_volume.jsonl":
            ("below.parquet", "per plane and instance, the part of the instance lying below "
                              "that plane"),
    },
    "ship_small": ["cache/inst_semantics_*.jsonl"],
    "drop": {
        "cache/inst_bf_exact.jsonl": "superseded, built over 1,622 boxes of 1,900",
        "cache/inst_bf_organelle.jsonl": "intermediate, before the volume recount was joined",
        "cache/inst_bf_organelle_OLD.jsonl": "intermediate, two revisions behind",
        "cache/inst_blockface.jsonl": "intermediate, carries the source volume figure only",
        "cache/inst_bf_below.jsonl": "intermediate",
        "cache/inst_surface.jsonl": "the surface subset is withdrawn, its area field does not "
                                    "mean what its name says",
        "cache/inst_thinsection.jsonl": "the thin-section subset is a synthesis, not an "
                                        "acquisition",
        "cache/exact_volume_0*.jsonl": "shards, concatenated into exact_volume.jsonl",
        "cache/below_volume_0*.jsonl": "shards",
    },
}
ROUND = {"v_stereo": 6, "err_stereo": 6, "volume_nm3": 6, "volume_stored_nm3": 6,
         "area_nm2": 6, "v_exact_nm3": 6, "v_stored_nm3": 6, "v_below_nm3": 6}
CATEGORICAL = ("geometry", "box", "dataset", "tier", "level", "sampling", "organelle")


def sig(x, n):
    x = np.asarray(x, float)
    out = x.astype(float).copy()
    m = np.isfinite(x) & (x != 0)
    if not m.any():
        return out
    e = np.floor(np.log10(np.abs(x[m]))) - (n - 1)
    scale = np.power(10.0, e)
    out[m] = np.round(x[m] / scale) * scale
    return out


def convert(src, dst, note):
    path = f"{ROOT}/{src}"
    if not os.path.exists(path):
        print(f"  missing {src}"); return None
    before = os.path.getsize(path)
    rows = (json.loads(l) for l in open(path))
    df = pd.DataFrame(rows)
    for c, n in ROUND.items():
        if c in df.columns:
            df[c] = sig(df[c].to_numpy(), n).astype(np.float64)
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in df.columns:
        if df[c].dtype == object and c not in CATEGORICAL:
            u = df[c].nunique(dropna=True)
            if u < len(df) / 20:
                df[c] = df[c].astype("category")
    pq = f"{OUT}/{dst}"
    df.to_parquet(pq, compression="zstd", index=False)
    csv = pq.replace(".parquet", ".csv.gz")
    df.to_csv(csv, index=False, compression="gzip")
    a, b = os.path.getsize(pq), os.path.getsize(csv)
    print(f"  {src}")
    print(f"    {note}")
    print(f"    {len(df):,} rows x {len(df.columns)} cols")
    print(f"    jsonl {before/1e6:>8.1f} MB  ->  parquet {a/1e6:>7.1f} MB "
          f"({before/max(a,1):.0f}x)   csv.gz {b/1e6:>7.1f} MB")
    return dict(src=src, rows=len(df), cols=list(df.columns),
                jsonl_mb=round(before/1e6, 1), parquet_mb=round(a/1e6, 2),
                csvgz_mb=round(b/1e6, 2), note=note)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("SHIPPING\n")
    manifest = {}
    for src, (dst, note) in PLAN["ship"].items():
        r = convert(src, dst, note)
        if r:
            manifest[dst] = r
        print()

    sem = sorted(glob.glob(f"{ROOT}/cache/inst_semantics_*.jsonl"))
    if sem:
        rows = [json.loads(l) for p in sem for l in open(p)]
        df = pd.DataFrame(rows)
        df.to_parquet(f"{OUT}/grouping.parquet", compression="zstd", index=False)
        df.to_csv(f"{OUT}/grouping.csv.gz", index=False, compression="gzip")
        sz = os.path.getsize(f"{OUT}/grouping.parquet")
        print(f"  cache/inst_semantics_*.jsonl  ({len(sem)} shards)")
        print(f"    identifiers per 3D connected component, per class and box")
        print(f"    {len(df):,} rows -> grouping.parquet {sz/1e6:.2f} MB\n")
        manifest["grouping.parquet"] = dict(src="cache/inst_semantics_*.jsonl", rows=len(df),
                                            cols=list(df.columns),
                                            parquet_mb=round(sz/1e6, 2))

    print("NOT SHIPPING\n")
    freed = 0
    for pat, why in PLAN["drop"].items():
        for p in sorted(glob.glob(f"{ROOT}/{pat}")):
            s = os.path.getsize(p); freed += s
            print(f"  {os.path.relpath(p, ROOT):<36s} {s/1e6:>8.1f} MB   {why}")
    print(f"\n  total not shipped {freed/1e9:.2f} GB")

    tot = sum(m.get("parquet_mb", 0) for m in manifest.values())
    print(f"\nrelease tables total {tot:.1f} MB, in {OUT}")
    json.dump(manifest, open(f"{OUT}/tables_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
