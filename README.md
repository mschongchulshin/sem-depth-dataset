# Depth from single block-face SEM micrographs

Code that builds the dataset described in *A dataset for depth estimation from single
block-face electron micrographs of cells and tissues*, and that runs every check reported in
its Technical Validation.

The dataset is archived separately at figshare under
[10.6084/m9.figshare.34070952](https://doi.org/10.6084/m9.figshare.34070952). This repository
holds only the code. Comments and docstrings were removed for release, and the manuscript
carries the description of what each step does.

## Entry points

| stage | script | what it does |
|---|---|---|
| 1 | `code/fetch_wholecell.py` | downloads each source box and stores it contrast stretched per box over the interior 0.2 and 99.8 percentiles, stamped as `em_norm` |
| 2 | `code/build_corpus.py` | places the cubic sub-volumes, called boxes, and drives stage 3 with `--slices 16 --margin 0.25` |
| 3 | `code/make_blockface.py` | cuts the planes and traces the axial column below every labelled pixel. Contrast is set again per face here, so grey levels are not comparable between planes of one box |
| 4 | `code/build_own_occupancy.py` | adds the own-occupancy ray to each plane |
| 5 | `code/build_deposit.py` | converts the rays to integer step counts, writes the data dictionary and the checksums |
| 6 | `code/build_instance_table.py` | summarises each exposed instance into the per-instance index |
| 7 | `code/pack_deposit.py` | writes `instance_index.parquet`, one tar per source volume and `MANIFEST.json` |
| 8 | `code/upload_figshare.py` | uploads the deposit, each file in parallel parts |

`code/build_release.py` belongs to an earlier layout and writes `instances.parquet`,
`volumes.parquet` and `below.parquet`. It is kept for provenance and is not part of the
pipeline that produced the deposit.

Checks and figures.

| script | what it does |
|---|---|
| `code/census_validation.py` | the censuses over all 23,968 planes |
| `code/audit_depth_reliability.py` | the sphere-cut and label-alignment checks, at the radius 190 sphere the Data Descriptor reports |
| `code/fig_gallery_merged.py` | Figure 1 |
| `code/fig_record.py` | Figure 2 |
| `code/fig_corpus.py` | Figure 3 |
| `code/fig_depth_validation.py` | Figure 4 |

## Source data

Images and voxel-wise organelle labels come from eighteen whole-cell reconstructions released
under CC-BY-4.0 by OpenOrganelle and the Janelia COSEM project. The scripts read them from the
public multiscale arrays, and nothing in this repository redistributes them. Work that uses
this code should cite the two papers that describe those reconstructions.

> Xu, C. S. et al. An open-access volume electron microscopy atlas of whole cells and tissues.
> *Nature* **599**, 147-151 (2021). https://doi.org/10.1038/s41586-021-03992-4

> Heinrich, L. et al. Whole-cell organelle segmentation in volume electron microscopy.
> *Nature* **599**, 141-146 (2021). https://doi.org/10.1038/s41586-021-03977-3

## Data root

Every script reads the deposit through the `SEM_DEPTH_ROOT` environment variable, which points at the directory holding `blockface/`, `release/` and `MANIFEST.json`. Without it the scripts fall back to the parent of this `code/` directory.

```
export SEM_DEPTH_ROOT=/path/to/sem-depth-dataset
```

## Requirements

Python 3.10 or later with `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `zarr` and
`s3fs`.

## Licence

MIT, see `LICENSE`.
