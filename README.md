# Spatially Uniform Rainfall Dataset Preparation with SFINCS

This repository contains the supplied notebooks and Python modules used to prepare training and testing datasets for rainfall-driven flood modelling. Spatially uniform rainfall is called **Type A** in the original filenames. The workflow selects terrain, adds sloped terrain variants, runs SFINCS, exports simulation results to NPZ, builds multiscale graphs, and checks temporal samples for graph neural network training.

Model training and post-training analysis are outside this repository's scope.

## Repository contents

| Location | Purpose |
| --- | --- |
| `notebooks/type_a_temporal_uniform/` | Four original notebooks; run in the order below |
| `scripts/sfincs_dataset_creation/` | `dem.py`, `hyetograph.py`, `sfincs.py` |
| `scripts/sfincs_dataset_conversion/` | `pytorch_geometric.py`, `temporal_learning.py`, `type_a_data.py` |
| `data/dem/Raw_DEMs/DEM/` | Local input DEM text files; placeholder only |
| `data/rainfall/rainfall_events/` | Local input rainfall files; placeholder only |
| `data/Synthetic_DEMs/` | Generated terrain variants and selection CSVs |
| `exe/SFINCS_v2.3.0_mt_Faber_release_exe/` | Local SFINCS executable and its runtime files |
| `models/` | Generated SFINCS case folders |
| `outputs/` | Generated logs, NPZ datasets, split manifest and graphs |
| `environment.yml` | Conda setup template inferred from imports |
| `environment_rebuild.yml` | Adapted environment for rebuilding the SFINCS test-case models (HydroMT-SFINCS 1.x) |
| `scripts/sfincs_rebuild/`, `notebooks/sfincs_rebuild/` | Build regular 100 m and subgrid 200/400 m models for the test splits |
| `config/dataset_reference.yaml` | Human-readable reference to settings in the original code |
| `.github/workflows/source-checks.yml` | Python/notebook syntax checks; no simulations |


## Environment setup

It includes HydroMT-SFINCS, NumPy, pandas, Matplotlib, xarray, NetCDF readers, PyTorch, PyTorch Geometric, tqdm and Jupyter.

 The SFINCS executable is separate from the Python package `hydromt-sfincs`; installing this environment does not install `sfincs.exe`. SFINCS v2.3.0. is used for running SFINCS simulations.


## Workflow

Run each notebook top to bottom, in this order.

| Order | Notebook | Main outputs |
| --- | --- | --- |
| 1 | `synthetic_dem_creation.ipynb` | Terrain selection CSVs, sloped DEM text files and `sloped_dem_summary.csv` |
| 2 | `type_A_sfincs_to_npz.ipynb` | SFINCS cases/logs, one NPZ per successful DEM–rainfall case, `run_summary.csv`, `dataset_split_type_A.json` |
| 3 | `type_A_prepare_training_data.ipynb` | NPZ/split checks and saved complete multiscale graphs per split |
| 4 | `type_A_test_temporal_samples.ipynb` | Shape, temporal-indexing and PyG batching checks for an example training event |

### Terrain selection

The synthetic-terrain notebook selects 20 base 64×64 DEMs using four relief groups and five samples per group with seed 42. It keeps all available 128×128 base DEMs. It adds a 5 m ramp to the first five selected 64×64 DEMs and a 10 m ramp to the next five; for 128×128 DEMs, the corresponding selections contain three DEMs each. Ramp directions use a random seed of 42.

The implemented ramp amplitudes are **5 m and 10 m**.

### Simulation and export

`sfincs.py` prepares case folders and ASCII elevation/mask files, writes the SFINCS configuration through HydroMT-SFINCS, copies rainfall forcing, runs the external executable, records runtime/logs, and saves NPZ output. Configured duration is 45 h with output every 1,800 s. Settings include Manning roughness 0.04, zero infiltration, wet/dry threshold 0.01 m, and EPSG 32643. Grid spacing is obtained from each DEM's coordinates.

Each NPZ stores `dem`, `water_depth`, `model_time_seconds`, `rainfall_time_seconds`, `rain_mmhr` and JSON `metadata`.

### Dataset splits

The actual split function checks `RP100` first, then `RP010`, then the DEM's domain role. It does not use a configurable list of return periods.

| JSON key | Terrain/domain role | Rainfall filename |
| --- | --- | --- |
| `train` | Seen (`train`, 64×64 selection) | Contains neither `RP010` nor `RP100` |
| `validation` | Seen | Contains `RP010` |
| `test_unseen_rainfall` | Seen | Contains `RP100` |
| `test_unseen_domain` | Unseen (`test`, 128×128 selection) | Any non-`RP100`, including `RP010` |
| `test_unseen_domain_and_rainfall` | Unseen | Contains `RP100` |


### Rebuilding the test-case models (regular + subgrid)

`scripts/sfincs_rebuild/` and `notebooks/sfincs_rebuild/build_test_models.ipynb` build, but do not run, SFINCS models for the three test splits (`test_unseen_rainfall`, `test_unseen_domain`, `test_unseen_domain_and_rainfall`; 126 cases) in three variants:

| Variant | Compute grid | Subgrid | Folder |
| --- | --- | --- | --- |
| `regular_100m` | 100 m | none; identical inputs to `sfincs.py` | `models/rebuild/regular_100m/` |
| `subgrid_200m` | 200 m | 25 m pixels (bilinear refinement of the 100 m DEM) | `models/rebuild/subgrid_200m/` |
| `subgrid_400m` | 400 m | 25 m pixels | `models/rebuild/subgrid_400m/` |

All variants share the time, physics and rainfall settings of the original setup and a closed domain. The case table and build summary are written to `outputs/rebuild/`. Use the environment `environment_rebuild.yml` (`sfincs-uniform-dataset-rebuild`), an adapted copy of `environment.yml` with HydroMT-SFINCS pinned to 1.x and without PyTorch. To run the models locally on Windows, `python scripts/sfincs_rebuild/write_batch.py [--sfincs-exe PATH] [--variants ...] [--splits ...] [--threads N]` writes `models/rebuild/run_sfincs.bat`, which runs each case in its own folder (log in `sfincs_log.txt`), skips cases that already have `sfincs_map.nc`, and records results in `models/rebuild/run_status.txt`. After the runs, `notebooks/sfincs_rebuild/compare_results.ipynb` (functions in `scripts/sfincs_rebuild/compare.py`) compares the subgrid variants with `regular_100m` on maximum water depth and maximum water level: `zsmax` of each run is downscaled onto the 100 m DEM, and per-case RMSE/MAE/bias, maximum-depth error, flood-extent CSI/hit rate/false alarm ratio, zsmax errors (in cells flooded in the reference) and runtimes are written to `outputs/rebuild/comparison.csv`. On Windows, start Jupyter from the activated environment (`conda activate sfincs-uniform-dataset-rebuild`); calling the environment's `python.exe` directly can load mismatched DLLs and crash matplotlib.

### Graphs and temporal samples

`pytorch_geometric.py` creates bidirectional four-neighbour grid graphs at fine, medium and coarse resolutions with edge distances and pooling maps. Terrain is pooled twice using 2×2 mean pooling, so each spatial dimension must be divisible by four. Rainfall is averaged over each model interval and repeated over fine-grid nodes. Complete graphs retain the full water-depth and rainfall sequences.

Default temporal inputs are `[h(t-2), h(t-1), h(t), P(t)]`, and the default six-step targets are `[h(t+1), ..., h(t+6)]`. Each sample additionally stores rainfall for the six rollout intervals and their durations. For 91 frames, these defaults produce 83 temporal samples per event. The test notebook checks temporal alignment and index offsets when batching two samples.


