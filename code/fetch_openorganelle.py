import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import zarr

BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com"


def http_get(url, retries=4):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as e:
            last = e
    raise last


def list_children(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "delimiter": "/", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = http_get(f"{BUCKET_URL}/?{urllib.parse.urlencode(q)}").decode()
        out += [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", xml) if p != prefix]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return [p[len(prefix):].strip("/") for p in out]


def read_zattrs(path):
    return json.loads(http_get(f"{BUCKET_URL}/{path}/.zattrs"))


def scale_translation(zattrs, level="s0"):
    ds = next(d for d in zattrs["multiscales"][0]["datasets"] if d["path"] == level)
    scale = translation = None
    for t in ds["coordinateTransformations"]:
        if t["type"] == "scale":
            scale = np.array(t["scale"], float)
        elif t["type"] == "translation":
            translation = np.array(t["translation"], float)
    if translation is None:
        translation = np.zeros(3)
    return scale, translation


def open_array(path):
    return zarr.open(zarr.storage.FsspecStore.from_url(f"{BUCKET_URL}/{path}"), mode="r")


def fetch_crop(dataset, crop, classes, outdir, em_group="em/fibsem-uint8"):
    root = f"{dataset}/{dataset}.zarr/recon-1"
    gt = f"{root}/labels/groundtruth/{crop}"
    available = set(list_children(f"{gt}/"))
    want = [c for c in classes if c in available]
    if not want:
        print(f"  [{crop}] none of {classes} present (has {sorted(available)[:12]}...)")
        return None

    labels, ref = {}, None
    for c in want:
        att = read_zattrs(f"{gt}/{c}")
        scale, trans = scale_translation(att)
        arr = open_array(f"{gt}/{c}/s0")
        vol = np.asarray(arr[:])
        binv = ((vol > 0) & (vol != 255)).astype(np.uint8)
        labels[c] = binv
        frac = float(binv.mean())
        print(f"  [{crop}] {c:10s} {vol.shape} fill={frac:.4f}")
        if ref is None:
            ref = dict(shape=vol.shape, scale=scale, trans=trans)

    em = None
    try:
        em_att = read_zattrs(f"{root}/{em_group}")
        es, et = scale_translation(em_att)
        lo = np.round((ref["trans"] - et) / es).astype(int)
        hi = np.round((ref["trans"] + ref["scale"] * ref["shape"] - et) / es).astype(int)
        emz = open_array(f"{root}/{em_group}/s0")
        lo = np.clip(lo, 0, np.array(emz.shape))
        hi = np.clip(hi, 0, np.array(emz.shape))
        em = np.asarray(emz[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        print(f"  [{crop}] em         {em.shape} from index {lo.tolist()}..{hi.tolist()}")
        if em.shape != ref["shape"]:
            idx = [np.clip(np.round(np.linspace(0, em.shape[d] - 1, ref["shape"][d])).astype(int),
                           0, em.shape[d] - 1) for d in range(3)]
            em = em[np.ix_(*idx)]
            print(f"  [{crop}] em resampled -> {em.shape}")
    except Exception as e:
        print(f"  [{crop}] em fetch failed ({e}); continuing label-only")

    meta = dict(dataset=dataset, crop=crop, classes=want,
                voxel_size_nm=ref["scale"].tolist(),
                world_translation_nm=ref["trans"].tolist(),
                axes="zyx", source=f"{BUCKET_URL}/{gt}")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset}__{crop}.npz"
    payload = {f"label_{k}": v for k, v in labels.items()}
    if em is not None:
        payload["em"] = em.astype(np.uint8)
    np.savez_compressed(path, meta=json.dumps(meta), **payload)
    print(f"  [{crop}] -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jrc_hela-2")
    ap.add_argument("--crops", nargs="+", default=["crop1"])
    ap.add_argument("--classes", nargs="+",
                    default=["cell", "mito", "er", "nuc", "golgi", "ves", "lyso", "ld", "endo"])
    ap.add_argument("--out", default="${EM_DEPTH_ROOT}/raw")
    ap.add_argument("--list-crops", action="store_true")
    a = ap.parse_args()

    if a.list_crops:
        p = f"{a.dataset}/{a.dataset}.zarr/recon-1/labels/groundtruth/"
        print("\n".join(sorted(list_children(p))))
        return

    out = Path(a.out)
    for crop in a.crops:
        print(f"fetching {a.dataset}/{crop}")
        try:
            fetch_crop(a.dataset, crop, a.classes, out)
        except Exception as e:
            print(f"  [{crop}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
