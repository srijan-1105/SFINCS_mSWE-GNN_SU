"""
Build (but do not run) SFINCS models for the test-split cases in three variants:

- regular_100m : the original 100 m regular model, written exactly as in
                 scripts/sfincs_dataset_creation/sfincs.py (ASCII dep/msk, same sfincs.inp)
- subgrid_200m : 200 m compute grid, subgrid tables from a 25 m DEM (8x8 pixels per cell)
- subgrid_400m : 400 m compute grid, subgrid tables from a 25 m DEM (16x16 pixels per cell)

The 25 m DEM is a bilinear refinement of the 100 m DEM. All variants use the same physics,
time settings and rainfall file, and a closed domain (all cells active, no boundaries).
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from scripts.sfincs_dataset_creation.dem import read_dem_file
from scripts.sfincs_dataset_creation.sfincs import (
    DTOUT_SECONDS,
    case_folder_type_A,
    write_dep_and_mask,
    write_type_a_sfincs_inp,
)

EPSG = 32643
SUBGRID_PIXEL_SIZE = 25.0  # Subgrid pixel size [m] for both subgrid variants.

VARIANTS = {
    "regular_100m": {"res": 100.0, "subgrid": False},
    "subgrid_200m": {"res": 200.0, "subgrid": True},
    "subgrid_400m": {"res": 400.0, "subgrid": True},
}

# Time and physics settings, identical to write_type_a_sfincs_inp() in sfincs.py.
BASE_CONFIG = dict(
    tref="20200101 000000",
    tstart="20200101 000000",
    tstop="20200102 210000",
    dtout=DTOUT_SECONDS,
    dthisout=DTOUT_SECONDS,
    dtmaxout=10800,
    alpha=0.5,
    theta=1.0,
    huthresh=0.01,
    manning=0.04,
    qinf=0.0,
    zsini=-999.0,
    outputformat="net",
    storecumprcp=1,
)


# ============================================================
# DEM conversion: xyz text file -> 100 m and 25 m GeoTIFFs
# ============================================================

def _grid_to_dataarray(grid, x_coords, y_coords):
    """Wrap a south-up DEM grid (rows = increasing y) into a north-up raster DataArray."""

    da = xr.DataArray(
        np.flipud(grid).astype(np.float32),
        coords={"y": y_coords[::-1].astype(np.float64), "x": x_coords.astype(np.float64)},
        dims=("y", "x"),
        name="elevtn",
    )
    da.raster.set_crs(EPSG)
    da.raster.set_nodata(-9999.0)
    return da


def dem_to_geotiff(dem_path, tif_folder):
    """Write the 100 m DEM as a GeoTIFF (cached)."""

    dem_path = Path(dem_path)
    tif_folder = Path(tif_folder)
    tif_folder.mkdir(parents=True, exist_ok=True)
    tif_100m = tif_folder / f"{dem_path.stem}_100m.tif"

    if not tif_100m.exists():
        dem_grid, x_coords, y_coords = read_dem_file(dem_path)
        _grid_to_dataarray(dem_grid, x_coords, y_coords).raster.to_raster(tif_100m)

    return tif_100m


def dem_to_25m(dem_path, tif_folder, pixel_size=SUBGRID_PIXEL_SIZE):
    """
    Write a bilinear refinement of the 100 m DEM at pixel_size (cached).

    The 100 m values sit at the cell centres. The 25 m pixel centres in the outer half cell lie outside
    those centres; the grid is padded by edge replication, so those pixels take the nearest edge value
    along the boundary-normal direction instead of being extrapolated.
    """
    dem_path = Path(dem_path)
    tif_folder = Path(tif_folder)
    tif_folder.mkdir(parents=True, exist_ok=True)
    tif_fine = tif_folder / f"{dem_path.stem}_{pixel_size:g}m.tif"

    if not tif_fine.exists():
        dem_grid, x_coords, y_coords = read_dem_file(dem_path)
        dx = float(x_coords[1] - x_coords[0])
        dy = float(y_coords[1] - y_coords[0])

        # Pad by one cell on every side (edge values), so interpolation covers the full domain extent.
        padded = np.pad(dem_grid.astype(np.float64), 1, mode="edge")
        x_pad = np.concatenate([[x_coords[0] - dx], x_coords, [x_coords[-1] + dx]]).astype(np.float64)
        y_pad = np.concatenate([[y_coords[0] - dy], y_coords, [y_coords[-1] + dy]]).astype(np.float64)
        interpolator = RegularGridInterpolator((y_pad, x_pad), padded, method="linear")

        x0 = float(x_coords.min() - dx / 2.0)
        y0 = float(y_coords.min() - dy / 2.0)
        nx_fine = int(round(len(x_coords) * dx / pixel_size))
        ny_fine = int(round(len(y_coords) * dy / pixel_size))
        x_fine = x0 + pixel_size * (np.arange(nx_fine) + 0.5)
        y_fine = y0 + pixel_size * (np.arange(ny_fine) + 0.5)

        yy, xx = np.meshgrid(y_fine, x_fine, indexing="ij")
        fine_grid = interpolator(np.column_stack([yy.ravel(), xx.ravel()])).reshape(ny_fine, nx_fine)

        _grid_to_dataarray(fine_grid, x_fine, y_fine).raster.to_raster(tif_fine)

    return tif_fine


# ============================================================
# Model builders
# ============================================================

def build_regular_100m(case_row, project_root, variant_root):
    """
    Build the original regular 100 m model: run_type_A_case() from sfincs.py without running SFINCS.
    """
    project_root = Path(project_root)
    dem_path = project_root / case_row["dem_path"]
    rainfall_path = project_root / case_row["rainfall_path"]

    _, case_dir = case_folder_type_A(dem_path, rainfall_path, Path(variant_root))

    dem_grid, x_coords, y_coords = read_dem_file(dem_path)
    write_dep_and_mask(case_dir, dem_grid)
    shutil.copyfile(rainfall_path, case_dir / "sfincs.prcp")
    write_type_a_sfincs_inp(case_dir, dem_grid, x_coords, y_coords)

    return case_dir, dem_grid.shape[1], dem_grid.shape[0]


def build_subgrid(case_row, project_root, variant_root, res, tif_folder):
    """
    Build a subgrid model with compute resolution res [m] and SUBGRID_PIXEL_SIZE subgrid pixels.
    """
    project_root = Path(project_root)
    dem_path = project_root / case_row["dem_path"]
    rainfall_path = project_root / case_row["rainfall_path"]

    _, case_dir = case_folder_type_A(dem_path, rainfall_path, Path(variant_root))

    _, x_coords, y_coords = read_dem_file(dem_path)
    dx_dem = float(x_coords[1] - x_coords[0])
    dy_dem = float(y_coords[1] - y_coords[0])
    x0 = float(x_coords.min() - dx_dem / 2.0)
    y0 = float(y_coords.min() - dy_dem / 2.0)
    mmax = int(round(len(x_coords) * dx_dem / res))
    nmax = int(round(len(y_coords) * dy_dem / res))
    nr_subgrid_pixels = int(round(res / SUBGRID_PIXEL_SIZE))

    tif_fine = dem_to_25m(dem_path, tif_folder)

    sf = SfincsModel(root=str(case_dir), mode="w+", write_gis=False)
    sf.setup_grid(x0=x0, y0=y0, dx=res, dy=res, nmax=nmax, mmax=mmax, rotation=0.0, epsg=EPSG)
    sf.setup_dep(datasets_dep=[{"elevtn": str(tif_fine), "reproj_method": "average"}])  # Coarse dep (mask, plots).
    sf.setup_mask_active(zmin=-1.0e4)  # All cells active, no boundary cells: closed domain as in the original.
    sf.setup_subgrid(
        datasets_dep=[{"elevtn": str(tif_fine)}],
        nr_subgrid_pixels=nr_subgrid_pixels,
        manning_land=BASE_CONFIG["manning"],
        manning_sea=BASE_CONFIG["manning"],
        huthresh=BASE_CONFIG["huthresh"],
        write_dep_tif=True,
    )
    sf.setup_config(**BASE_CONFIG, prcfile="sfincs.prcp")
    sf.write()

    # write_grid() adds depfile for the coarse dep but skips writing sfincs.dep when a subgrid is present;
    # remove the dangling key so SFINCS takes elevation from sbgfile only.
    sf.config.pop("depfile", None)
    sf.write_config()

    shutil.copyfile(rainfall_path, case_dir / "sfincs.prcp")  # Same rainfall file as the regular model.

    return case_dir, mmax, nmax


def build_case(case_row, variant, project_root, model_root, tif_folder):
    """Build one case for one variant; returns (case_dir, mmax, nmax)."""

    settings = VARIANTS[variant]
    variant_root = Path(model_root) / variant

    if settings["subgrid"]:
        return build_subgrid(case_row, project_root, variant_root, settings["res"], tif_folder)
    return build_regular_100m(case_row, project_root, variant_root)


def build_all(case_table, project_root, model_root, tif_folder, variants=tuple(VARIANTS),
              overwrite=False, summary_csv=None):
    """
    Build every case in case_table for every variant. Existing case folders are skipped unless
    overwrite=True. One failing case does not stop the batch.
    """
    model_root = Path(model_root)
    rows = []

    jobs = [(variant, case_row) for variant in variants for _, case_row in case_table.iterrows()]
    for variant, case_row in tqdm(jobs, desc="Building models"):
        case_dir = model_root / variant / case_row["case_name"]

        if case_dir.joinpath("sfincs.inp").exists() and not overwrite:
            rows.append({"case_name": case_row["case_name"], "variant": variant, "split": case_row["split"],
                         "status": "exists", "folder": str(case_dir)})
            continue

        try:
            case_dir, mmax, nmax = build_case(case_row, variant, project_root, model_root, tif_folder)
            status = "ok"
        except Exception as exc:
            mmax = nmax = None
            status = f"failed: {exc}"  # Catch-all - one bad case shouldn't stop the whole batch.

        rows.append({"case_name": case_row["case_name"], "variant": variant, "split": case_row["split"],
                     "status": status, "mmax": mmax, "nmax": nmax, "folder": str(case_dir)})

    summary = pd.DataFrame(rows)
    if summary_csv is not None:
        Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_csv, index=False)

    return summary
