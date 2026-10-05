import hashlib, json, os, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = f"{ROOT}/release/blockface"
OUT = os.environ.get("PACK_OUT", f"{ROOT}/dist")
META = ["splits.csv", "boxes.csv", "sources.csv", "instance_index.parquet",
        "instance_index.csv.gz", "checksums.sha256", "README.md"]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    skip_box, skip_plane = set(), set()
    if os.environ.get("DEPOSIT_EXCLUSIONS"):
        e = json.load(open(os.environ["DEPOSIT_EXCLUSIONS"]))
        skip_box, skip_plane = set(e["boxes"]), set(e["planes"])

    by_vol = {}
    for d in sorted(os.listdir(DEP)):
        if d.startswith(".") or d in skip_box:
            continue
        if os.path.isdir(f"{DEP}/{d}/faces"):
            by_vol.setdefault(d.split("__")[0], []).append(d)
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(by_vol)} source volumes -> {OUT}", flush=True)

    man, t0 = [], time.time()
    for v, boxes in sorted(by_vol.items()):
        dst = f"{OUT}/{v}.tar"
        n = 0
        with tarfile.open(dst, "w") as tf:
            for b in boxes:
                fd = f"{DEP}/{b}/faces"
                for fn in sorted(os.listdir(fd)):
                    if not fn.endswith(".npz") or fn.startswith(".") or fn[:-4] in skip_plane:
                        continue
                    tf.add(f"{fd}/{fn}", arcname=f"blockface/{b}/faces/{fn}")
                    n += 1
        sz = os.path.getsize(dst)
        man.append(dict(file=f"{v}.tar", planes=n, bytes=sz, sha256=sha256(dst)))
        print(f"  {v:24s} {n:5d} planes  {sz / 1e9:5.2f} GB", flush=True)

    for m in META:
        p = f"{ROOT}/release/{m}"
        if os.path.exists(p):
            man.append(dict(file=m, planes=None, bytes=os.path.getsize(p), sha256=sha256(p)))

    json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
    tot = sum(x["bytes"] for x in man)
    print(f"\n{len(man)} files, {tot / 1e9:.2f} GB, "
          f"{sum(x['planes'] or 0 for x in man):,} planes, {(time.time() - t0) / 60:.1f} min")
    print(f"-> {OUT}/MANIFEST.json")


if __name__ == "__main__":
    main()
