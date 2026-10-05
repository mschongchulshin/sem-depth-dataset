import json
import urllib.request
import numpy as np
from scipy import ndimage

from n5 import N5Array

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"

CANDIDATES = [
    ("n5", "em/fibsem-uint16"),
    ("n5", "em/fibsem-uint8"),
    ("n5", "em/fibsem-int16"),
    ("zarr", "recon-1/em/fibsem-uint8"),
    ("zarr", "recon-1/em/fibsem-uint16"),
]


def _head(url, name):
    try:
        return json.loads(urllib.request.urlopen(f"{url}/{name}", timeout=20).read())
    except Exception:
        return None


def find_em(dataset):
    for kind, path in CANDIDATES:
        ext = ".zarr" if kind == "zarr" else ".n5"
        url = f"{BUCKET}/{dataset}/{dataset}{ext}/{path}"
        attrs = _head(url, ".zattrs" if kind == "zarr" else "attributes.json")
        if attrs is None:
            continue
        levels = []
        ms = attrs.get("multiscales")
        if ms:
            for dsr in ms[0]["datasets"]:
                sc = next((t["scale"] for t in dsr.get("coordinateTransformations", [])
                           if t.get("type") == "scale"), None)
                if sc:
                    levels.append((dsr["path"], [float(v) for v in sc]))
        if not levels and kind == "n5":
            for i in range(8):
                a = _head(f"{url}/s{i}", "attributes.json")
                if a is None:
                    break
                px = a.get("pixelResolution", {}).get("dimensions") or a.get("resolution")
                if px:
                    levels.append((f"s{i}", [float(v) for v in reversed(px)]))
        if levels:
            return kind, url, levels
    return None, None, None


def _read(kind, url, level, lo, hi):
    if kind == "n5":
        return N5Array(f"{url}", level).read(lo, hi) if False else \
            N5Array(url, level).read(lo, hi)
    import zarr
    a = zarr.open_array(f"{url}/{level}", mode="r")
    lo = np.clip(lo, 0, np.array(a.shape))
    hi = np.clip(hi, 0, np.array(a.shape))
    return a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def read_em_nm(dataset, start_nm, stop_nm, out_shape, verbose=True):
    kind, url, levels = find_em(dataset)
    if kind is None:
        if verbose:
            print(f"  em: no array found for {dataset} under any known name", flush=True)
        return None
    target_nm = (np.asarray(stop_nm) - np.asarray(start_nm)) / np.asarray(out_shape)
    level, vox = min(levels, key=lambda lv: abs(np.log(np.mean(lv[1]) / np.mean(target_nm))))
    vox = np.asarray(vox, float)
    lo = np.floor(np.asarray(start_nm) / vox).astype(int)
    hi = np.ceil(np.asarray(stop_nm) / vox).astype(int)
    try:
        raw = _read(kind, url, level, lo, hi)
    except Exception as e:
        if verbose:
            print(f"  em read failed at {level}: {type(e).__name__} {e}", flush=True)
        return None
    if raw.size == 0:
        return None
    if tuple(raw.shape) != tuple(out_shape):
        zoom = np.array(out_shape, float) / np.array(raw.shape, float)
        raw = ndimage.zoom(raw.astype(np.float32), zoom, order=1)
        raw = raw[:out_shape[0], :out_shape[1], :out_shape[2]]
        if tuple(raw.shape) != tuple(out_shape):
            pad = [(0, max(0, t - c)) for t, c in zip(out_shape, raw.shape)]
            raw = np.pad(raw, pad, mode="edge")
    r = raw.astype(np.float32)
    interior = r[(r > r.min()) & (r < r.max())]
    ref = interior if interior.size > 1000 else r
    if tuple(raw.shape) != tuple(out_shape):
        if verbose:
            print(f"  em shape {raw.shape} != labels {tuple(out_shape)}, dropping", flush=True)
        return None
    lo_p, hi_p = np.percentile(ref, [0.2, 99.8])
    img = np.clip((r - lo_p) / max(hi_p - lo_p, 1e-6), 0, 1)
    if verbose:
        print(f"  em       {img.shape} from {kind}:{level} at {vox.tolist()} nm", flush=True)
    return (img * 255).astype(np.uint8)
