# Depth from single block-face SEM micrographs

Code that builds the dataset described in *A dataset for depth estimation from single
block-face electron micrographs of cells and tissues*, and that runs every check reported in
its Technical Validation.

The dataset is archived separately at figshare under
[10.6084/m9.figshare.34070952](https://doi.org/10.6084/m9.figshare.34070952). This repository
holds only the code. Comments and docstrings were removed for release, and the manuscript
carries the description of what each step does.

## Entry points

| script | what it does |
|---|---|
| `code/build_corpus.py` | places the cubic sub-volumes, called boxes, in each source reconstruction |
| `code/make_blockface.py` | cuts the planes and traces the axial column below every labelled pixel |
| `code/build_instance_table.py` | summarises each exposed instance into the per-instance index |
| `code/build_release.py` | assembles the deposit |
| `code/pack_deposit.py` | writes one tar per source volume and the checksums |
| `code/census_validation.py` | the censuses over all 23,968 planes |
| `code/audit_depth_reliability.py` | the sphere-cut and label-alignment checks |
| `code/fig_gallery_merged.py` | Figure 1 |
| `code/fig_record.py` | Figure 2 |
| `code/fig_corpus.py` | Figure 3 |
| `code/fig_depth_validation.py` | Figure 4 |
| `code/upload_zenodo.py` | uploads the deposit |

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
