import bz2
import gzip
import json
import lzma
import zlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DTYPES = {"uint8": ">u1", "uint16": ">u2", "uint32": ">u4", "uint64": ">u8",
          "int8": ">i1", "int16": ">i2", "int32": ">i4", "int64": ">i8",
          "float32": ">f4", "float64": ">f8"}


def _decompress(payload, kind):
    if kind == "raw":
        return payload
    if kind == "gzip":
        return gzip.decompress(payload)
    if kind == "zlib":
        return zlib.decompress(payload)
    if kind == "xz":
        return lzma.decompress(payload)
    if kind == "bzip2":
        return bz2.decompress(payload)
    if kind == "blosc":
        import numcodecs.blosc as nb
        return nb.decompress(payload)
    if kind == "zstd":
        try:
            import numcodecs.zstd as nz
            return nz.decompress(payload)
        except Exception:
            import zstandard
            return zstandard.ZstdDecompressor().decompress(payload)
    if kind == "lz4":
        import numcodecs.lz4 as nl
        return nl.decompress(payload)
    raise NotImplementedError(f"N5 compression {kind}")


def _get(url, retries=4):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
    raise last


class N5Array:
    def __init__(self, base_url, level="s0"):
        self.url = f"{base_url.rstrip('/')}/{level}"
        raw_attrs = _get(f"{self.url}/attributes.json")
        if raw_attrs is None:
            raise FileNotFoundError(f"no N5 array at {self.url}")
        attrs = json.loads(raw_attrs)
        self.dims_f = list(attrs["dimensions"])
        self.block_f = list(attrs["blockSize"])
        self.dtype = DTYPES[attrs["dataType"]]
        comp = attrs.get("compression", {"type": "raw"})
        self.compression = comp.get("type", "raw")
        if self.compression not in ("gzip", "raw", "blosc", "zlib", "lz4", "xz", "zstd"):
            raise NotImplementedError(f"N5 compression {self.compression}")
        self.shape = tuple(self.dims_f[::-1])
        self.block = tuple(self.block_f[::-1])

        grp = _get(f"{base_url.rstrip('/')}/attributes.json")
        self.group_attrs = json.loads(grp) if grp else {}
        self.level_attrs = attrs

    def _decode(self, raw):
        if raw is None:
            return None
        head = np.frombuffer(raw[:4], dtype=">u2")
        ndim = int(head[1])
        off = 4 + 4 * ndim
        dims = np.frombuffer(raw[4:off], dtype=">u4").astype(int)
        payload = raw[off:]
        payload = _decompress(payload, self.compression)
        body = np.frombuffer(payload, dtype=self.dtype)
        n = int(np.prod(dims))
        if body.size < n:
            raise ValueError(f"short N5 block: {body.size} < {n}")
        blk = body[:n].reshape(tuple(dims), order="F")
        return blk.transpose(2, 1, 0)

    def read(self, start_zyx, stop_zyx, workers=12):
        start = np.clip(np.asarray(start_zyx, int), 0, self.shape)
        stop = np.clip(np.asarray(stop_zyx, int), 0, self.shape)
        out = np.zeros(tuple(stop - start), dtype=np.dtype(self.dtype).newbyteorder("="))
        if out.size == 0:
            return out

        bz, by, bx = self.block
        g0 = start // np.array([bz, by, bx])
        g1 = (stop - 1) // np.array([bz, by, bx])

        jobs = []
        for gz in range(g0[0], g1[0] + 1):
            for gy in range(g0[1], g1[1] + 1):
                for gx in range(g0[2], g1[2] + 1):
                    jobs.append((gz, gy, gx))

        def work(g):
            gz, gy, gx = g
            return g, self._decode(_get(f"{self.url}/{gx}/{gy}/{gz}"))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for (gz, gy, gx), blk in ex.map(work, jobs):
                if blk is None:
                    continue
                o = np.array([gz * bz, gy * by, gx * bx])
                lo = np.maximum(o, start)
                hi = np.minimum(o + np.array(blk.shape), stop)
                if np.any(hi <= lo):
                    continue
                sl_b = tuple(slice(lo[d] - o[d], hi[d] - o[d]) for d in range(3))
                sl_o = tuple(slice(lo[d] - start[d], hi[d] - start[d]) for d in range(3))
                out[sl_o] = blk[sl_b]
        return out

    def voxel_size_nm(self):
        for src in (self.level_attrs, self.group_attrs):
            tr = src.get("transform")
            if tr and "scale" in tr:
                axes = tr.get("axes", ["z", "y", "x"])
                scale = dict(zip(axes, tr["scale"]))
                return np.array([scale.get("z"), scale.get("y"), scale.get("x")], float)
            if "pixelResolution" in src:
                d = src["pixelResolution"]["dimensions"]
                return np.array(d[::-1], float)
        return None

    def translation_nm(self):
        for src in (self.level_attrs, self.group_attrs):
            tr = src.get("transform")
            if tr and "translate" in tr:
                axes = tr.get("axes", ["z", "y", "x"])
                t = dict(zip(axes, tr["translate"]))
                return np.array([t.get("z", 0.0), t.get("y", 0.0), t.get("x", 0.0)], float)
        return np.zeros(3)


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else (
        "https://janelia-cosem-datasets.s3.amazonaws.com/"
        "jrc_hela-2/jrc_hela-2.n5/labels/mito_seg")
    lvl = sys.argv[2] if len(sys.argv) > 2 else "s2"
    a = N5Array(base, lvl)
    print("shape zyx", a.shape, "block", a.block, "dtype", a.dtype,
          "voxel_nm", a.voxel_size_nm(), "translate", a.translation_nm())
    mid = np.array(a.shape) // 2
    box = np.array([64, 64, 64])
    v = a.read(mid - box, mid + box)
    ids = np.unique(v)
    print("read", v.shape, "nonzero frac", float((v > 0).mean()),
          "n instances", len(ids) - (1 if ids[0] == 0 else 0))
