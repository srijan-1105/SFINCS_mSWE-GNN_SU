"""
Build the table of DEM-rainfall cases for the three generalisation test splits.

The DEM selection and split rule are the same as in
notebooks/type_a_temporal_uniform/type_A_sfincs_to_npz.ipynb (cells 6 and 11).
"""

from pathlib import Path

import pandas as pd

from scripts.sfincs_dataset_creation.dem import read_dem_file
from scripts.sfincs_dataset_creation.hyetograph import list_rainfall_files

TEST_SPLITS = (
    "test_unseen_rainfall",
    "test_unseen_domain",
    "test_unseen_domain_and_rainfall",
)


def assign_split(domain_role, rainfall_name):
    """Combine domain seen/unseen status with rainfall seen/unseen status (copied from the notebook)."""

    if "RP100" in rainfall_name:
        return "test_unseen_domain_and_rainfall" if domain_role == "test" else "test_unseen_rainfall"
    if "RP010" in rainfall_name:
        return "validation" if domain_role == "train" else "test_unseen_domain"
    return "train" if domain_role == "train" else "test_unseen_domain"


def load_dem_list(project_root):
    """List the flat and sloped DEMs with their domain role, as in the notebook."""

    project_root = Path(project_root)
    dem_folder = project_root / "data" / "dem" / "Raw_DEMs" / "DEM"
    synthetic_folder = project_root / "data" / "Synthetic_DEMs"

    flat_64 = pd.read_csv(synthetic_folder / "selected_flat_64x64.csv")
    flat_128 = pd.read_csv(synthetic_folder / "selected_flat_128x128.csv")
    sloped = pd.read_csv(synthetic_folder / "sloped_dem_summary.csv")

    flat_64_list = pd.DataFrame({
        "dem_name": flat_64["dem_name"],
        "domain_role": "train",
        "topography": "flat",
        "dem_path": [dem_folder / name for name in flat_64["dem_name"]],
    })

    flat_128_list = pd.DataFrame({
        "dem_name": flat_128["dem_name"],
        "domain_role": "test",
        "topography": "flat",
        "dem_path": [dem_folder / name for name in flat_128["dem_name"]],
    })

    sloped_list = pd.DataFrame({
        "dem_name": sloped["output_dem"],
        "domain_role": sloped["dataset_role"],
        "topography": "sloped",
        "dem_path": [synthetic_folder / name for name in sloped["output_dem"]],
    })

    return pd.concat([flat_64_list, flat_128_list, sloped_list], ignore_index=True)


def build_case_table(project_root, out_csv=None):
    """
    Cross the DEMs with the rainfall events and keep only the test-split cases.

    case_name follows case_folder_type_A(): <dem stem>_<rainfall stem>_type_A
    """
    project_root = Path(project_root)
    rainfall_folder = project_root / "data" / "rainfall" / "rainfall_events"

    dem_list = load_dem_list(project_root)
    rainfall_files = list_rainfall_files(rainfall_folder)

    rows = []
    for _, dem_row in dem_list.iterrows():
        dem_grid, _, _ = read_dem_file(dem_row["dem_path"])  # Read once per DEM to record the grid size.
        for rainfall_name in rainfall_files:
            split = assign_split(dem_row["domain_role"], rainfall_name)
            if split not in TEST_SPLITS:
                continue
            rows.append({
                "case_name": f"{Path(dem_row['dem_name']).stem}_{Path(rainfall_name).stem}_type_A",
                "dem_name": dem_row["dem_name"],
                "dem_path": str(Path(dem_row["dem_path"]).relative_to(project_root)),
                "rainfall": rainfall_name,
                "rainfall_path": str((rainfall_folder / rainfall_name).relative_to(project_root)),
                "split": split,
                "domain_role": dem_row["domain_role"],
                "topography": dem_row["topography"],
                "nx": int(dem_grid.shape[1]),
                "ny": int(dem_grid.shape[0]),
            })

    case_table = pd.DataFrame(rows)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        case_table.to_csv(out_csv, index=False)

    return case_table
