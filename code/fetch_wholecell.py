import argparse
import os
import guard as _g
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from n5 import N5Array
from em_reader import read_em_nm

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"


def seg_url(dataset, name):
    return f"{BUCKET}/{dataset}/{dataset}.n5/labels/{name}_seg"


def em_url(dataset):
    return f"{BUCKET}/{dataset}/{dataset}.n5/em/fibsem-uint16"


EM_CANDIDATES = [
    ("n5", "em/fibsem-uint16"),
    ("n5", "em/fibsem-uint8"),
    ("zarr", "recon-1/em/fibsem-uint8"),
    ("zarr", "recon-1/em/fibsem-uint16"),
    ("n5", "em/fibsem-int16"),
]


def em_urls(dataset):
    for kind, path in EM_CANDIDATES:
        ext = ".zarr" if kind == "zarr" else ".n5"
        yield f"{BUCKET}/{dataset}/{dataset}{ext}/{path}"


CACHE = Path("/Volumes/One Touch/em-depth-dataset/cache")


def instance_volume_table(dataset, organelle, level="s4"):
    a = N5Array(seg_url(dataset, organelle), level)
    vox = a.voxel_size_nm()
    vvol = float(np.prod(vox))
    v = a.read([0, 0, 0], a.shape)
    ids, cnt = np.unique(v[v > 0], return_counts=True)
    return {int(i): float(c) * vvol for i, c in zip(ids, cnt)}, a.shape, vox


def instance_volume_table_refined(dataset, organelle, coarse_level="s4", fine_level="s0",
                                  ids=None, pad=2, use_cache=True, verbose=True):
    key = f"{dataset}__{organelle}__{coarse_level}_to_{fine_level}.json"
    cache = CACHE / key
    cached = {}
    if use_cache and cache.exists():
        cached = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        if ids is not None and set(int(i) for i in ids) <= set(cached):
            if verbose:
                print(f"    volume table cache covers all {len(ids)} requested ids ({key})",
                      flush=True)
            return cached
        if verbose:
            miss = "?" if ids is None else len(set(int(i) for i in ids) - set(cached))
            print(f"    volume table cache has {len(cached)} instances, {miss} missing ({key})",
                  flush=True)

    coarse = N5Array(seg_url(dataset, organelle), coarse_level)

    if fine_level == coarse_level:
        cv0 = coarse.read([0, 0, 0], coarse.shape)
        vol0 = float(np.prod(coarse.voxel_size_nm()))
        want = None if ids is None else set(int(i) for i in ids)
        vals, counts = np.unique(cv0[cv0 > 0], return_counts=True)
        table = {int(v): float(c) * vol0 for v, c in zip(vals, counts)
                 if want is None or int(v) in want}
        table.update(cached)
        if verbose:
            print(f"    {len(table)} instance volumes counted directly at {coarse_level}, "
                  f"no per-instance reads needed", flush=True)
        if use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(table))
        return table

    fine = N5Array(seg_url(dataset, organelle), fine_level)
    cvox, fvox = coarse.voxel_size_nm(), fine.voxel_size_nm()
    scale = cvox / fvox
    fvol = float(np.prod(fvox))

    cv = coarse.read([0, 0, 0], coarse.shape)
    present = np.unique(cv[cv > 0])
    if ids is not None:
        want = set(int(i) for i in ids)
        present = np.array([i for i in present if int(i) in want])
    objs = ndimage.find_objects(cv.astype(np.int32))
    if verbose:
        print(f"    refining {len([i for i in present if int(i) not in cached])} instance volumes "
              f"{coarse_level} -> {fine_level}", flush=True)

    order = ["s0", "s1", "s2", "s3", "s4"]
    levels = order[order.index(fine_level):order.index(coarse_level) + 1]
    _levels = {fine_level: fine}
    budget = float(os.environ.get("EMD_PATCH_GB", 4)) * 1024 ** 3

    table = dict(cached)
    measured_at = {}
    n_over = 0
    present = np.array([i for i in present if int(i) not in table])

    min_span = float(os.environ.get("EMD_REFINE_MIN_SPAN_VOX", 25))
    big_floor = (np.pi / 6.0) * min_span ** 3
    cvol = float(np.prod(cvox))
    counts = np.bincount(cv.ravel())
    keep = []
    for i in present:
        n_coarse = int(counts[int(i)]) if int(i) < counts.size else 0
        if n_coarse >= big_floor:
            table[int(i)] = float(n_coarse) * cvol
            measured_at[int(i)] = coarse_level
        else:
            keep.append(i)
    if verbose and len(keep) < len(present):
        print(f"    {len(present) - len(keep)} instances already well resolved at "
              f"{coarse_level}, refining {len(keep)}", flush=True)
    present = np.array(keep)

    for n, i in enumerate(present):
        sl = objs[int(i) - 1] if int(i) - 1 < len(objs) else None
        if sl is None:
            continue
        lo = np.array([sl[d].start for d in range(3)]) * scale - pad
        hi = np.array([sl[d].stop for d in range(3)]) * scale + pad
        lo = np.clip(np.floor(lo).astype(int), 0, np.array(fine.shape))
        hi = np.clip(np.ceil(hi).astype(int), 0, np.array(fine.shape))
        if np.any(hi <= lo):
            continue
        chosen, patch = None, None
        for lv in levels:
            arr = _levels.get(lv)
            if arr is None:
                try:
                    arr = _levels[lv] = N5Array(seg_url(dataset, organelle), lv)
                except Exception:
                    _levels[lv] = False
                    continue
            if arr is False:
                continue
            sc = cvox / arr.voxel_size_nm()
            l2 = np.clip(np.floor(np.array([sl[d].start for d in range(3)]) * sc - pad).astype(int),
                         0, np.array(arr.shape))
            h2 = np.clip(np.ceil(np.array([sl[d].stop for d in range(3)]) * sc + pad).astype(int),
                         0, np.array(arr.shape))
            if np.any(h2 <= l2):
                continue
            nbytes = float(np.prod(h2 - l2)) * 8.0
            if nbytes > budget:
                continue
            try:
                patch = arr.read(l2, h2)
            except Exception as e:
                if verbose:
                    print(f"      id {int(i)} read at {lv} failed: {type(e).__name__}", flush=True)
                continue
            chosen = lv
            vvol = float(np.prod(arr.voxel_size_nm()))
            break
        if chosen is None:
            if verbose:
                print(f"      id {int(i)} skipped: bbox exceeds the "
                      f"{budget / 1024 ** 3:.0f} GB patch budget at every level", flush=True)
            continue
        n_over += (chosen != fine_level)
        table[int(i)] = float((patch == i).sum()) * vvol
        measured_at[int(i)] = chosen
        del patch
        if verbose and (n + 1) % 25 == 0:
            print(f"      {n + 1}/{len(present)} done", flush=True)

    if verbose and n_over:
        print(f"    {n_over}/{len(table)} instances too large for {fine_level}, "
              f"measured at a coarser level instead", flush=True)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(table))
        (CACHE / key.replace(".json", "__level.json")).write_text(json.dumps(measured_at))
    return table


def find_dense_box(dataset, organelles, box_zyx, level="s4", probe_level="s3"):
    if isinstance(organelles, str):
        organelles = [organelles]
    dens = None
    probe_vox = shape = None
    for org in organelles:
        try:
            a = N5Array(seg_url(dataset, org), probe_level)
        except Exception:
            continue
        v = (a.read([0, 0, 0], a.shape) > 0).astype(np.float32)
        if dens is None:
            probe_vox, shape = a.voxel_size_nm(), v.shape
            dens = np.zeros(shape, np.float32)
        if v.shape == shape:
            dens += v
    if dens is None:
        raise SystemExit(f"could not probe any of {organelles}")
    k = np.maximum(np.array(shape) // 12, 1)
    dens = ndimage.uniform_filter(dens, size=tuple(int(x) for x in k))
    cz, cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    a = N5Array(seg_url(dataset, organelles[0]), probe_level)
    tgt = N5Array(seg_url(dataset, organelles[0]), level)
    tgt_vox = tgt.voxel_size_nm()
    scale = probe_vox / tgt_vox
    centre = (np.array([cz, cy, cx], float) * scale).astype(int)
    half = np.array(box_zyx, int) // 2
    start = np.clip(centre - half, 0, np.array(tgt.shape) - np.array(box_zyx))
    start = np.maximum(start, 0)
    return start, np.minimum(start + np.array(box_zyx), np.array(tgt.shape)), float(dens.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jrc_hela-2")
    ap.add_argument("--organelles", nargs="+", default=["mito", "lyso", "ld", "endo"])
    ap.add_argument("--level", default="s1", help="pyramid level to read the box at")
    ap.add_argument("--table-level", default="s4",
                    help="coarse level used to locate each instance's bounding box")
    ap.add_argument("--fine-level", default="s0",
                    help="level the instance volume is actually counted at; '' to disable")
    ap.add_argument("--box", type=int, nargs=3, default=[256, 256, 256], help="Z Y X voxels")
    ap.add_argument("--start", type=int, nargs=3, default=None,
                    help="box origin in Z Y X at --level; omit to auto-locate")
    ap.add_argument("--probe-level", default="s3",
                    help="level the density search runs at; s4 is too coarse and misplaces the box")
    ap.add_argument("--anchor", default=None,
                    help="organelle whose density picks the box; default the first one")
    ap.add_argument("--out", default="/Volumes/One Touch/em-depth-dataset/raw")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    if a.start is None:
        probe = a.organelles if a.anchor is None else [a.anchor]
        start, stop, dens = find_dense_box(a.dataset, probe, a.box, a.level, a.probe_level)
        print(f"auto box from {probe} density {dens:.4f} probed at {a.probe_level}: "
              f"{start.tolist()} .. {stop.tolist()}",
              flush=True)
    else:
        probe = N5Array(seg_url(a.dataset, a.organelles[0]), a.level)
        shape = np.array(probe.shape)
        box = np.array(a.box, int)
        start = np.clip(np.array(a.start, int), 0, np.maximum(shape - box, 0))
        stop = np.minimum(start + box, shape)
        print(f"box {start.tolist()} .. {stop.tolist()} in a volume of {shape.tolist()}",
              flush=True)

    inst = None
    cls = None
    classes = []
    table = {}
    voxel = None
    id_offset = 0

    for ci, org in enumerate(a.organelles, start=1):
        try:
            arr = N5Array(seg_url(a.dataset, org), a.level)
        except Exception as e:
            print(f"  skip {org}: {type(e).__name__} {e}", flush=True)
            continue
        if voxel is None:
            voxel = arr.voxel_size_nm()
            n = int(np.prod(stop - start))
            _g.preflight(n * 5 + n * 8, f"box {tuple(stop - start)} at {a.level}")
            inst = np.zeros(tuple(stop - start), np.uint32)
            cls = np.zeros(tuple(stop - start), np.uint8)
        v = arr.read(start, stop)
        present = np.unique(v[v > 0])
        if present.size == 0:
            print(f"  {org:8s} empty in this box", flush=True)
            classes.append(org)
            continue

        faces = [v[0], v[-1], v[:, 0], v[:, -1], v[:, :, 0], v[:, :, -1]]
        touching = set(int(x) for f in faces for x in np.unique(f) if x > 0)

        if a.fine_level:
            want = np.array([i for i in present if int(i) not in touching])
            vt = instance_volume_table_refined(a.dataset, org, a.table_level, a.fine_level,
                                               ids=want) if want.size else {}
        else:
            vt, _, _ = instance_volume_table(a.dataset, org, a.table_level)

        sel = v > 0
        inst[sel] = v[sel].astype(np.uint32) + id_offset
        cls[sel] = ci
        box_counts = np.bincount(v.ravel())
        for i in present:
            gid = int(i) + id_offset
            table[str(gid)] = dict(
                organelle=org, local_id=int(i),
                volume_nm3=vt.get(int(i)),
                volume_um3=(vt.get(int(i)) * 1e-9) if vt.get(int(i)) is not None else None,
                voxels_in_box=int(box_counts[int(i)]) if int(i) < box_counts.size else 0,
                truncated=bool(int(i) in touching))
        n_ok = sum(1 for i in present if int(i) not in touching)
        print(f"  {org:8s} {present.size:5d} instances in box, {n_ok} untruncated, "
              f"fill={float(sel.mean()):.4f}", flush=True)
        classes.append(org)
        id_offset += int(max(present.max(), 1)) + 1

    if inst is None:
        raise SystemExit("no organelle could be read")

    start_nm = (np.asarray(start, float) * voxel).tolist()
    stop_nm = (np.asarray(stop, float) * voxel).tolist()
    em = read_em_nm(a.dataset, start_nm, stop_nm, tuple(int(v) for v in (stop - start)))
    if em is None:
        print("  em unavailable: this box cannot serve the block-face or thin-section subsets",
              flush=True)

    meta = dict(em_norm="interior-p0.2-p99.8",
                dataset=a.dataset, source="OpenOrganelle n5 whole-cell instance segmentation",
                level=a.level, table_level=a.table_level,
                voxel_size_nm=[float(x) for x in voxel], axes="zyx",
                volume_measured_at_level=a.fine_level or a.table_level,
                box_start_zyx=[int(x) for x in start], box_stop_zyx=[int(x) for x in stop],
                classes=classes, instances=table)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nm = a.name or f"{a.dataset}__{'_'.join(a.organelles)}__{a.level}"
    path = out / f"{nm}.npz"
    inst_out = inst.astype(np.uint16) if int(inst.max()) <= np.iinfo(np.uint16).max else inst
    payload = dict(meta=json.dumps(meta), inst=inst_out, cls=cls)
    if em is not None:
        payload["em"] = em
    np.savez_compressed(path, **payload)
    print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB), "
          f"{len(table)} instances, occupancy={float((inst > 0).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
