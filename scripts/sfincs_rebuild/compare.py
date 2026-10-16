"""
Compare the maximum water depth and maximum water level of the rebuilt SFINCS runs against the
regular 100 m reference.

Only zsmax is downscaled: for every variant, the maximum water level over the whole simulation
(zsmax, maximum over all dtmaxout windows) is mapped onto the 100 m DEM,

    hmax_100m = max(zsmax_cell - dem_100m, 0)

where each 200 m / 400 m cell covers an exact 2x2 / 4x4 block of 100 m DEM cells (grids are
aligned, both with origin (0, 0) and rows running south to north). For regular_100m the factor is 1.
Cells that are never wet have zsmax = NaN in the regular output and get depth 0.

Per case and variant the metrics are computed against regular_100m: RMSE / MAE / bias of hmax,
error in the domain-maximum depth, flood extent statistics (CSI, hit rate, false alarm ratio) at
a depth threshold, stored volume of the hmax map, rainfall volume and SFINCS runtime.

zsmax itself is compared on the same 100 m grid (each coarse cell's zsmax applied to its block), but
only in cells flooded in the reference (hmax_ref > threshold): elsewhere the regular zsmax is NaN
(never wet) and a water level has no meaning.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scripts.sfincs_dataset_creation.dem import read_dem_file

REFERENCE = "regular_100m"
VARIANTS = ("regular_100m", "subgrid_200m", "subgrid_400m")
FLOOD_THRESHOLD = 0.01  # Depth [m] above which a cell counts as flooded for the extent statistics.


# ============================================================
# Reading and downscaling
# ============================================================

def read_run(case_dir):
    """Read zsmax (maximum over all windows), rainfall depth and runtime from one sfincs_map.nc."""

    with xr.open_dataset(Path(case_dir) / "sfincs_map.nc") as dataset:
        zsmax = dataset["zsmax"].max(dim="timemax", skipna=True).values  # All-NaN cells stay NaN (never wet).
        rain_depth = float(np.nanmean(dataset["cumprcp"].isel(timemax=-1).values))  # Cumulative rainfall [m].
        runtime = float(dataset["total_runtime"].values[0]) if "total_runtime" in dataset else np.nan

    return zsmax, rain_depth, runtime


def zsmax_to_dem_grid(zsmax, dem_grid):
    """Map a (coarse) zsmax grid onto the 100 m DEM grid: each cell's value fills its block of DEM cells."""

    factor_y = dem_grid.shape[0] // zsmax.shape[0]
    factor_x = dem_grid.shape[1] // zsmax.shape[1]
    if (zsmax.shape[0] * factor_y, zsmax.shape[1] * factor_x) != dem_grid.shape:
        raise ValueError(f"zsmax {zsmax.shape} does not tile the DEM {dem_grid.shape}")

    return np.repeat(np.repeat(zsmax, factor_y, axis=0), factor_x, axis=1)


def downscale_zsmax(zsmax, dem_grid):
    """Map a (coarse) zsmax grid onto the 100 m DEM and return the maximum water depth there."""

    hmax = zsmax_to_dem_grid(zsmax, dem_grid) - dem_grid
    return np.where(np.isfinite(hmax), np.maximum(hmax, 0.0), 0.0)


def load_run(case_row, variant, project_root, model_root):
    """
    Return a dict with the 100 m maximum-depth map (hmax), zsmax on the 100 m grid (NaN where never wet),
    rainfall depth, runtime and cell size for one run.
    """
    dem_grid, x_coords, _ = read_dem_file(Path(project_root) / case_row["dem_path"])
    zsmax, rain_depth, runtime = read_run(Path(model_root) / variant / case_row["case_name"])

    return {
        "hmax": downscale_zsmax(zsmax, dem_grid),
        "zsmax": zsmax_to_dem_grid(zsmax, dem_grid),
        "rain_depth": rain_depth,
        "runtime": runtime,
        "cell_size": float(x_coords[1] - x_coords[0]),
    }


# ============================================================
# Metrics
# ============================================================

def hmax_metrics(hmax_ref, hmax, cell_size, threshold=FLOOD_THRESHOLD):
    """Error and flood-extent statistics of hmax against hmax_ref (both on the same 100 m grid)."""

    difference = hmax - hmax_ref
    flooded_ref = hmax_ref > threshold
    flooded = hmax > threshold

    hits = int(np.sum(flooded & flooded_ref))
    misses = int(np.sum(~flooded & flooded_ref))
    false_alarms = int(np.sum(flooded & ~flooded_ref))
    cell_area = cell_size ** 2

    return {
        "rmse_hmax_m": float(np.sqrt(np.mean(difference ** 2))),
        "mae_hmax_m": float(np.mean(np.abs(difference))),
        "bias_hmax_m": float(np.mean(difference)),
        "max_hmax_ref_m": float(hmax_ref.max()),
        "max_hmax_m": float(hmax.max()),
        "max_hmax_error_m": float(hmax.max() - hmax_ref.max()),
        "flooded_area_ref_km2": float(flooded_ref.sum() * cell_area / 1e6),
        "flooded_area_km2": float(flooded.sum() * cell_area / 1e6),
        "csi": hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan,
        "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else np.nan,
        "false_alarm_ratio": false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan,
        "volume_hmax_ref_m3": float(hmax_ref.sum() * cell_area),
        "volume_hmax_m3": float(hmax.sum() * cell_area),
    }


def zsmax_metrics(zsmax_ref, zsmax, hmax_ref, threshold=FLOOD_THRESHOLD):
    """Error statistics of zsmax against zsmax_ref in the cells flooded in the reference."""

    flooded_ref = hmax_ref > threshold
    difference = (zsmax - zsmax_ref)[flooded_ref & np.isfinite(zsmax) & np.isfinite(zsmax_ref)]

    if difference.size == 0:
        return {"rmse_zsmax_m": np.nan, "mae_zsmax_m": np.nan, "bias_zsmax_m": np.nan,
                "p95_abs_zsmax_m": np.nan, "max_abs_zsmax_m": np.nan, "n_cells_zsmax": 0}

    return {
        "rmse_zsmax_m": float(np.sqrt(np.mean(difference ** 2))),
        "mae_zsmax_m": float(np.mean(np.abs(difference))),
        "bias_zsmax_m": float(np.mean(difference)),
        "p95_abs_zsmax_m": float(np.percentile(np.abs(difference), 95)),
        "max_abs_zsmax_m": float(np.max(np.abs(difference))),
        "n_cells_zsmax": int(difference.size),
    }


def compare_case(case_row, project_root, model_root, variants=VARIANTS, threshold=FLOOD_THRESHOLD):
    """Compare every variant of one case against the reference; returns one row per variant."""

    ref = load_run(case_row, REFERENCE, project_root, model_root)
    hmax_ref, cell_size = ref["hmax"], ref["cell_size"]
    domain_area = hmax_ref.size * cell_size ** 2

    rows = []
    for variant in variants:
        run = ref if variant == REFERENCE else load_run(case_row, variant, project_root, model_root)
        row = {
            "case_name": case_row["case_name"],
            "split": case_row["split"],
            "topography": case_row["topography"],
            "rainfall": case_row["rainfall"],
            "grid": f"{case_row['nx']}x{case_row['ny']}",
            "variant": variant,
        }
        row.update(hmax_metrics(hmax_ref, run["hmax"], cell_size, threshold))
        row.update(zsmax_metrics(ref["zsmax"], run["zsmax"], hmax_ref, threshold))
        row.update({
            # Rainfall volume; volume_hmax / rain_volume >= ~1 because hmax is the maximum over time.
            "rain_volume_m3": run["rain_depth"] * domain_area,
            "runtime_s": run["runtime"],
            "runtime_ref_s": ref["runtime"],
        })
        rows.append(row)

    return rows


def compare_all(case_table, project_root, model_root, variants=VARIANTS, threshold=FLOOD_THRESHOLD,
                out_csv=None):
    """Compare all cases; runs without sfincs_map.nc are reported in 'status' and skipped."""

    rows = []
    for _, case_row in case_table.iterrows():
        try:
            rows.extend(compare_case(case_row, project_root, model_root, variants, threshold))
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            rows.append({"case_name": case_row["case_name"], "split": case_row["split"],
                         "status": f"failed: {exc}"})

    comparison = pd.DataFrame(rows)
    if "status" not in comparison:
        comparison["status"] = "ok"
    comparison["status"] = comparison["status"].fillna("ok")

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_csv, index=False)

    return comparison


def summarize(comparison, by=("split", "variant")):
    """Mean metrics per group (default: split x variant)."""

    columns = ["rmse_hmax_m", "mae_hmax_m", "bias_hmax_m", "max_hmax_error_m", "csi", "hit_rate",
               "false_alarm_ratio", "rmse_zsmax_m", "mae_zsmax_m", "bias_zsmax_m", "p95_abs_zsmax_m", "runtime_s"]
    ok = comparison[comparison["status"] == "ok"]
    summary = ok.groupby(list(by))[columns].mean()
    summary["n_cases"] = ok.groupby(list(by)).size()
    return summary


# ============================================================
# Plots
# ============================================================

def plot_case(case_row, project_root, model_root, variants=VARIANTS, threshold=FLOOD_THRESHOLD,
              quantity="hmax"):
    """
    Maps per variant (top) and their difference with the reference (bottom).

    quantity: "hmax" (maximum depth) or "zsmax" (maximum water level; shown only in cells flooded in the
    reference, as in zsmax_metrics).
    """
    runs = {variant: load_run(case_row, variant, project_root, model_root) for variant in variants}
    if REFERENCE not in runs:
        runs[REFERENCE] = load_run(case_row, REFERENCE, project_root, model_root)
    flooded_ref = runs[REFERENCE]["hmax"] > threshold

    if quantity == "hmax":
        maps = {variant: np.ma.masked_where(run["hmax"] <= threshold, run["hmax"]) for variant, run in runs.items()}
        diffs = {variant: run["hmax"] - runs[REFERENCE]["hmax"] for variant, run in runs.items()}
        cmap, label = "Blues", "Maximum water depth [m]"
    elif quantity == "zsmax":
        maps = {variant: np.ma.masked_where(~flooded_ref, run["zsmax"]) for variant, run in runs.items()}
        diffs = {variant: np.ma.masked_where(~flooded_ref, run["zsmax"] - runs[REFERENCE]["zsmax"])
                 for variant, run in runs.items()}
        cmap, label = "viridis", "Maximum water level [m]"
    else:
        raise ValueError(f"quantity must be 'hmax' or 'zsmax', not {quantity!r}")

    vmin = min(float(maps[variant].min()) for variant in variants)
    vmax = max(float(maps[variant].max()) for variant in variants)
    # Colour limit from the 99th percentile, so a few edge cells (a coarse cell spanning two ponds) don't hide the rest.
    vmax_diff = max(float(np.percentile(np.abs(np.ma.asarray(diffs[variant]).compressed()), 99))
                    for variant in variants) or 1e-3

    fig, axes = plt.subplots(2, len(variants), figsize=(3.6 * len(variants), 7), constrained_layout=True,
                             squeeze=False)

    for column, variant in enumerate(variants):
        image = axes[0, column].imshow(maps[variant], origin="lower", cmap=cmap,
                                       vmin=0 if quantity == "hmax" else vmin, vmax=vmax)
        axes[0, column].set_title(f"{variant}: {quantity}", fontsize=9)

        diff = axes[1, column].imshow(diffs[variant], origin="lower", cmap="RdBu_r",
                                      vmin=-vmax_diff, vmax=vmax_diff)
        axes[1, column].set_title(f"{variant} - {REFERENCE}", fontsize=9)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.colorbar(image, ax=axes[0, :].tolist(), label=label)
    fig.colorbar(diff, ax=axes[1, :].tolist(), label="Difference [m]")
    fig.suptitle(f"{case_row['case_name']} ({case_row['split']})")
    plt.show()

    return fig


# Summary figure: fixed colour per variant (reference grey, then categorical slots 1 and 2).
VARIANT_COLORS = {"regular_100m": "#737370", "subgrid_200m": "#2a78d6", "subgrid_400m": "#eb6834"}
VARIANT_LABELS = {"regular_100m": "regular 100 m", "subgrid_200m": "subgrid 200 m", "subgrid_400m": "subgrid 400 m"}
SPLIT_LABELS = {
    "test_unseen_rainfall": "unseen\nrainfall",
    "test_unseen_domain": "unseen\ndomain",
    "test_unseen_domain_and_rainfall": "unseen domain\n+ rainfall",
}


def plot_summary(comparison, out_png=None, seed=0):
    """
    One summary figure: per split, the per-case distribution (dots) and median (bar) of
    (a) hmax RMSE, (b) flood-extent CSI, (c) zsmax RMSE for the subgrid variants, and (d) SFINCS runtime
    for all variants.
    """
    ok = comparison[comparison["status"] == "ok"]
    rng = np.random.default_rng(seed)
    subgrid_variants = [v for v in VARIANTS if v != REFERENCE]

    panels = [
        ("rmse_hmax_m", 100.0, "RMSE of max. depth [cm]", "a  Maximum water depth", subgrid_variants),
        ("csi", 1.0, "CSI of flood extent (h > 5 cm) [-]", "b  Flood extent", subgrid_variants),
        ("rmse_zsmax_m", 100.0, "RMSE of max. water level [cm]", "c  Maximum water level", subgrid_variants),
        ("runtime_s", 1.0, "SFINCS runtime [s]", "d  Runtime", list(VARIANTS)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)

    for axis, (column, scale, ylabel, title, variants) in zip(axes, panels):
        width = 0.8 / len(variants)
        for i_split, split in enumerate(SPLIT_LABELS):
            for i_var, variant in enumerate(variants):
                values = ok.loc[(ok["split"] == split) & (ok["variant"] == variant), column].to_numpy() * scale
                if values.size == 0:
                    continue
                x_center = i_split + (i_var - (len(variants) - 1) / 2) * width
                jitter = rng.uniform(-0.3, 0.3, values.size) * width
                axis.scatter(x_center + jitter, values, s=12, color=VARIANT_COLORS[variant], alpha=0.55,
                             linewidths=0, zorder=2)
                axis.hlines(np.median(values), x_center - 0.4 * width, x_center + 0.4 * width,
                            color="#1f1f1e", linewidth=2, zorder=3)

        axis.set_xticks(range(len(SPLIT_LABELS)))
        axis.set_xticklabels(SPLIT_LABELS.values(), fontsize=9)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if column == "csi":
            axis.set_ylim(0.6, 1.01)  # Dots, not bars: a non-zero baseline keeps the 0.7-1.0 spread readable.
        else:
            axis.set_ylim(bottom=0)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=VARIANT_COLORS[v],
                          label=VARIANT_LABELS[v]) for v in VARIANTS]
    handles.append(plt.Line2D([], [], color="#1f1f1e", linewidth=2, label="median"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle("Subgrid SFINCS (25 m subgrid pixels) vs regular 100 m, per test case", fontsize=12)

    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()

    return fig
