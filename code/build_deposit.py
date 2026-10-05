import collections, glob, hashlib, json, os
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{ROOT}/release"
V = json.load(open(f"{ROOT}/cache/validation_results.json"))


def build_splits():
    files = sorted(f for f in glob.glob(f"{REL}/blockface/*/faces/*.npz") if "/._" not in f)
    recs = []
    for i, f in enumerate(files):
        if i % 5000 == 0 and i:
            print(f"    {i:,}/{len(files):,}", flush=True)
        rel = f"blockface/{os.path.relpath(f, f'{REL}/blockface')}"
        box = os.path.basename(os.path.dirname(os.path.dirname(f)))
        vol = box.split("__")[0]
        tier = box.split("__")[1].rsplit("_b", 1)[0] if "__" in box else "?"
        try:
            z = np.load(f, allow_pickle=False)
            m = json.loads(str(z["meta"]))
            msk = z["mask"]; ef = z["inst_face"]
            ids = np.unique(ef)
            n_inst = int((ids > 0).sum())
            cls = z["cls_face"]
            n_cls = int(len(np.unique(cls[msk])) if msk.any() else 0)
            px = float(m.get("pixel_size_nm", 0) or 0)
            zs = float(m.get("z_step_nm", 0) or 0)
            frac = float(msk.mean())
        except Exception:
            continue
        recs.append((rel, box, vol, tier, px, zs, n_inst, n_cls, frac))
    vols = sorted({r[2] for r in recs})
    fold = {v: i for i, v in enumerate(vols)}
    out = ["file,box,source_volume,tier,px_nm,z_step_nm,n_instances,n_classes,"
           "labelled_fraction,fold"]
    for rel, box, vol, tier, px, zs, ni, nc, fr in recs:
        out.append(f"{rel},{box},{vol},{tier},{px:g},{zs:g},{ni},{nc},{fr:.4f},{fold[vol]}")
    with open(f"{REL}/splits.csv", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"  splits.csv  {len(recs):,} planes of {len(files):,} deposited, "
          f"{len(vols)} folds")
    return vols, fold, len(recs)


def build_checksums():
    files = []
    for pat in ("blockface/*/faces/*.npz", "*.parquet", "*.csv.gz", "*.csv", "*.md", "*.json"):
        files.extend(sorted(f for f in glob.glob(f"{REL}/{pat}") if "/._" not in f))
    print(f"  hashing {len(files):,} files", flush=True)
    lines = []
    total = 0
    for i, f in enumerate(files):
        if i % 4000 == 0 and i:
            print(f"    {i:,}/{len(files):,}", flush=True)
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        total += os.path.getsize(f)
        lines.append(f"{h.hexdigest()}  {os.path.relpath(f, REL)}")
    with open(f"{REL}/checksums.sha256", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  checksums.sha256  {len(lines):,} files, {total/1e9:.2f} GB")
    return len(lines), total


def build_readme(vols, n_planes, n_files, total_bytes):
    c = V["corpus"]; px = V["level_is_not_a_scale"]
    txt = f"""# Block-face micrographs paired with per-pixel depth

{n_planes:,} block-face electron micrographs, each paired with the distance every exposed
organelle continues below the imaged face, measured rather than estimated.

Focused ion beam scanning electron microscopy mills a slice and images the freshly exposed
face, then repeats. The material behind any one face is therefore imaged by the faces that
follow it, so a depth map for that face exists exactly and is normally discarded when the
stack is assembled. This corpus recovers it.

## Provenance

Images and segmentations derive from OpenOrganelle and Janelia COSEM whole-cell FIB-SEM
reconstructions under CC-BY-4.0. Nothing here is a new acquisition. {len(vols)} source volumes
contribute. Changes made to the source are stated in the accompanying Data Descriptor.

This deposit is released under CC-BY-4.0, inherited from the source.

## Layout

```
blockface/<volume>__<tier>_bNNN/faces/<box>_z<NNNN>.npz   one cut plane
splits.csv                                               the evaluation protocol
checksums.sha256                                         SHA-256 per file
```

## One cut plane

All arrays share the plane's shape, height by width.

| array | dtype | meaning |
|---|---|---|
| `em` | uint8 | the micrograph, as acquired |
| `inst_face` | int32 | which instance is exposed at this pixel, 0 for none |
| `cls_face` | uint8 | which class is exposed, indexing this box's own `classes` list |
| `mask` | bool | pixels where the plane cuts labelled material |
| `depth_below_steps` | uint16 | contiguous run of the exposed instance below this pixel |
| `own_occupancy_below_steps` | uint16 | total path below belonging to that same instance |
| `thickness_below_steps` | uint16 | occupied path below belonging to any structure |
| `clipped` | bool | the ray reached the block bottom, so the value is a lower bound |
| `meta` | json | `pixel_size_nm`, `z_step_nm`, `classes`, per-instance records |

**Ray quantities are integer step counts.** Multiply by `z_step_nm` from `meta` for nanometres.
They are stored this way because every value is an exact multiple of the step and float32 was
returning 20.639999 for three steps of 6.88 nm. Converting back is exact.

```python
import json, numpy as np
z = np.load("blockface/jrc_hela-2__cell_b004/faces/jrc_hela-2__cell_b004_z0120.npz")
m = json.loads(str(z["meta"]))
depth_nm = z["depth_below_steps"].astype(np.float32) * m["z_step_nm"]
```

The ordering `depth <= own occupancy <= thickness` holds at every foreground pixel tested,
0 violations over 18,803,994 pixels.

## Three things that will be got wrong

**1. A step count of 0 means nothing is exposed, not zero depth.** `mask` says the same thing
and is the safer test. In the original float32 arrays this was NaN, and because NaN times zero
is NaN, multiplying by the mask did not remove it from a mean or a loss. That trap is gone in
this deposit, but code written against the original arrays still carries it.

**2. `cls_face` is scoped to its own box.** The identifier indexes that source volume's class
list, carried in the plane's `classes` field. The same number names different organelles in
different volumes: identifier 7 is `er` across 51% of the pixels it covers corpus-wide and
`nucleolus` across most of the rest. **Join through the organelle name.**

**3. The pyramid level name does not fix a pixel size.** The reconstructions do not share a
base resolution. `s0` is 4 nm in the ten cultured-line volumes, 16 nm in `jrc_mus-liver`, and
128 nm in all seven mouse-tissue volumes. Use `pixel_size_nm` from `meta`, or the `px_nm`
column of `splits.csv`.

| pixel size | planes | share of planes | instances | share of instances |
|---|---|---|---|---|
| 4 nm | 880 | 3.3% | {px['px_nm_counts']['4']:,} | 0.2% |
| 8 nm | 4,356 | 16.3% | {px['px_nm_counts']['8']:,} | 4.2% |
| 16 nm | 5,526 | 20.6% | {px['px_nm_counts']['16']:,} | 26.5% |
| 32 nm | 5,822 | 21.7% | {px['px_nm_counts']['32']:,} | 18.0% |
| 64 nm | 5,850 | 21.8% | {px['px_nm_counts']['64']:,} | 37.4% |
| 128 nm | 4,349 | 16.2% | {px['px_nm_counts']['128']:,} | 13.7% |

The two columns disagree because a coarse box holds more labelled objects per plane. Counted
by plane the corpus is nearly balanced across scales. Counted by contained instance it is not,
and 4 nm falls to 0.2%. Which denominator applies depends on whether the unit of work is an
image or an object.

Instance identifiers are unique within a box only, so `(box, instance)` is the smallest safe
key. The source carries no reconstruction-wide identifier, so an organelle appearing in two
overlapping boxes cannot be merged.

## Resolution, specimen and class change together

Any stratified comparison in this corpus confounds three things at once, and the confounding is
close to total. The class mix follows the pixel size:

| pixel size | contained instances | what is in it |
|---|---|---|
| 4 nm | 772 | `endo` 38%, `nucleolus` 30%, `ld` 20%, `lyso` 10% |
| 8 nm | 15,940 | `endo` 55%, `lyso` 13%, `ld` 9%, `mito` 9% |
| 16 nm | 100,065 | `endo` 46%, `mito` 19%, `lyso` 12%, `chrom` 11% |
| 32 nm | 68,181 | `mito` 77%, `chrom` 12% |
| 64 nm | 141,252 | `mito` 93% |
| **128 nm** | 51,730 | **`nucleus` 100.0%** |

The seven mouse-tissue volumes carry exactly one annotated class between them, `nucleus`, and
they are the only volumes cut at 128 nm. Whether a number is lower at 128 nm because the boxes
are coarse, because the specimen is tissue, or because the objects are nuclei cannot be told
apart from this corpus. Restrict to one class before comparing scales, and to one scale before
comparing classes.

`sources.csv` carries the annotated class list for every volume so this can be checked rather
than assumed.

## Evaluation

`splits.csv` assigns every plane to a fold, one per source volume. Two cells from one
reconstruction share tissue, staining, instrument and the network that labelled them, and
adjacent planes of one box can show the same organelle, so a random split over planes leaks.

Reading it:

```python
import pandas as pd
s = pd.read_csv("splits.csv")
test  = s[s.fold == 3].file.tolist()
train = s[s.fold != 3].file.tolist()
```

An instance visible on several planes of one box appears once per plane. An evaluation whose
unit is the organelle rather than the cut face must group by `(box, instance)` first.

## What the depth is and is not

It is the run of the **exposed instance**, not of the class and not of any material. Where an
instance leaves the ray and re-enters it, `depth_below_steps` stops at the first gap while
`own_occupancy_below_steps` counts the whole path.

`thickness_below_steps` is a union over every structure and is a property of the neighbourhood
rather than of one object. It is context, not object mass.

Depth is a lower bound wherever `clipped` is true. That is 1.52% of nucleus pixels, 0.11% of
chromatin, 0.06% of mitochondria and 0.00% of every other class.

A one-voxel error in the source segmentation moves the depth by one step axially, and laterally
by the local surface obliquity, whose median across all classes is also one step. Nuclear
envelope and Golgi are the exceptions, with 90th percentiles of 14 and 9 steps, because they
are thin oblique membranes.

## Annotations are the source's

No proofreading was performed. Instance-level defects are inherited, not introduced. The Data
Descriptor reports which classes are reliable for which purpose, measured rather than assumed.

## Deposit

{n_files:,} files, {total_bytes/1e9:.2f} GB.
"""
    with open(f"{REL}/README.md", "w") as fh:
        fh.write(txt)
    print(f"  README.md  {len(txt.splitlines())} lines")


if __name__ == "__main__":
    os.makedirs(REL, exist_ok=True)
    print("building deposit files", flush=True)
    vols, fold, n_planes = build_splits()
    build_readme(vols, n_planes, 0, 0)
    n_files, total = build_checksums()
    build_readme(vols, n_planes, n_files, total)
    print(f"\n-> {REL}")
