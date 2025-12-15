from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import logging


def _plot_uncertainty_map_png(
    arr: np.ndarray,
    out_png: Path,
    y_top_m: float,
    y_bottom_m: float,
    xmin: float = 0.0,
    xmax: float | None = None,
    coords: pd.DataFrame | None = None,
    show_cpt_locations: bool = True,
    font_size: int = 8,
    title: str = "Leave-out uncertainty (Std Dev)",
):
    import matplotlib.pyplot as plt

    n_rows_total, n_cols = arr.shape
    if xmax is None:
        xmax = float(n_cols - 1)

    # Assume uniform pixel width in "x units" (meters if you provide xmin/xmax in meters)
    global_dx = (xmax - xmin) / max(n_cols - 1, 1)

    base_width = 20
    height = base_width / 8
    fig, ax = plt.subplots(figsize=(base_width, height))

    im = ax.imshow(
        arr,
        cmap="hot",
        vmin=None,
        vmax=None,
        aspect="auto",
        extent=[xmin - global_dx / 2, xmax + global_dx / 2, n_rows_total - 0.5, -0.5],
    )

    cbar = plt.colorbar(im, label="Uncertainty (Std Dev)", extend="neither")
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("Uncertainty (Std Dev)", fontsize=font_size)

    # CPT vertical lines (if coords contain cum_along_m and you provided xmin/xmax in meters)
    if show_cpt_locations and coords is not None and "cum_along_m" in coords.columns:
        for cpt_x in coords["cum_along_m"].values:
            if xmin <= cpt_x <= xmax:
                ax.axvline(x=cpt_x, color="black", linewidth=1, alpha=0.5, zorder=10)

    ax.set_xlabel(
        "Distance along line (m)" if coords is not None else "X (pixel index)",
        fontsize=font_size,
    )
    ax.set_ylabel("Depth Index (global)", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)

    # Right y-axis: depth in meters
    def idx_to_m(y_idx):
        return y_top_m + (y_idx / (n_rows_total - 1)) * (y_bottom_m - y_top_m)

    def m_to_idx(y_m):
        denom = y_bottom_m - y_top_m
        return 0 if abs(denom) < 1e-12 else (y_m - y_top_m) * (n_rows_total - 1) / denom

    right = ax.secondary_yaxis("right", functions=(idx_to_m, m_to_idx))
    right.set_ylabel("Depth (m)", fontsize=font_size)
    right.tick_params(labelsize=font_size)

    plt.title(title, fontsize=font_size)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=800, bbox_inches="tight")
    plt.close()


def _load_mosaic_csv(path: Path, expected_rows: int = 64) -> np.ndarray:
    """
    Loads a mosaic csv into (rows, cols) float array.

    Handles your formats:
    - original_mosaic.csv: first row is a coordinate/index row (0..N-1) -> dropped
    - validation_mosaic.csv: directly data
    """
    df = pd.read_csv(path, header=None)
    arr = df.to_numpy(dtype=float)

    # Drop coordinate/header row if present (original often has 65 rows: 1 coord + 64 data)
    if arr.shape[0] == expected_rows + 1:
        row0 = arr[0, :]
        diffs = np.diff(row0)

        # Heuristic: monotonic and roughly constant step (index-like)
        if np.all(diffs >= 0) and np.nanmedian(np.abs(diffs)) > 0:
            # common in your original: 0..(ncols-1) or similar
            arr = arr[1:, :]

    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {arr.shape[0]} rows; expected {expected_rows} data rows."
        )

    return arr


def _downsample_columns_area_weighted(data: np.ndarray, target_cols: int) -> np.ndarray:
    """
    Conservative downsampling along columns using area-weighted (box) averaging
    over a normalized physical domain [0, 1] (same extent assumption).
    """
    rows, src_cols = data.shape
    if target_cols == src_cols:
        return data.copy()
    if target_cols > src_cols:
        raise ValueError(
            "downsample_columns_area_weighted only supports target_cols <= src_cols."
        )

    src_edges = np.linspace(0.0, 1.0, src_cols + 1)
    tgt_edges = np.linspace(0.0, 1.0, target_cols + 1)

    W = np.zeros((src_cols, target_cols), dtype=float)

    for j in range(target_cols):
        a, b = tgt_edges[j], tgt_edges[j + 1]

        # Candidate src bins that could overlap this target bin
        i0 = np.searchsorted(src_edges, a, side="right") - 1
        i1 = np.searchsorted(src_edges, b, side="left")
        i0 = max(i0, 0)
        i1 = min(i1, src_cols)

        denom = b - a
        for i in range(i0, i1):
            left = max(src_edges[i], a)
            right = min(src_edges[i + 1], b)
            overlap = max(0.0, right - left)
            if overlap > 0:
                W[i, j] = overlap / denom

    # Each target column should sum to 1 (within tolerance)
    if not np.allclose(W.sum(axis=0), 1.0, atol=1e-6):
        raise RuntimeError("Downsampling weights do not sum to 1. Check edge logic.")

    return data @ W  # (rows, src_cols) @ (src_cols, target_cols) -> (rows, target_cols)


def run_leaveout_uncertainty(
    original_mosaic_csv: Path,
    validation_root: Path,
    output_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
    expected_rows: int = 64,
    validation_filename: str = "validation_mosaic.csv",
    logger: logging.Logger | None = None,
) -> dict:
    """
    Computes leave-out uncertainty map:
    sigma(y,x) = std across validation mosaics at each pixel (after downsampling all to min cols).

    Saves:
    - sigma_map.csv
    - original_downsampled.csv
    - target_cols.txt
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Collect all validation mosaic paths: run_XX/validation_mosaic.csv
    val_paths = sorted(validation_root.glob(f"run_*/{validation_filename}"))
    if not val_paths:
        raise FileNotFoundError(
            f"No validation mosaics found under: {validation_root}\\run_*\\{validation_filename}"
        )

    # Find smallest column count among validations
    min_cols = None
    min_path = None
    val_arrays = []

    log(
        f"Scanning {len(val_paths)} validation mosaics to find smallest column count..."
    )
    for p in val_paths:
        arr = _load_mosaic_csv(p, expected_rows=expected_rows)
        val_arrays.append(arr)
        if min_cols is None or arr.shape[1] < min_cols:
            min_cols = arr.shape[1]
            min_path = p

    target_cols = int(min_cols)
    log(f"Target columns = {target_cols} (smallest found in: {min_path})")

    # Load original and downsample
    original = _load_mosaic_csv(original_mosaic_csv, expected_rows=expected_rows)
    original_ds = _downsample_columns_area_weighted(original, target_cols)

    # Downsample validations and stack
    vals_ds = np.stack(
        [_downsample_columns_area_weighted(a, target_cols) for a in val_arrays], axis=0
    )  # (n_runs, rows, target_cols)

    # Std map across runs
    sigma_map = np.std(vals_ds, axis=0, ddof=1)  # (rows, target_cols)

    # Save outputs
    sigma_csv = output_folder / "leaveout_sigma_map.csv"
    orig_csv = output_folder / "original_downsampled_to_target.csv"
    meta_txt = output_folder / "leaveout_target_cols.txt"

    pd.DataFrame(sigma_map).to_csv(sigma_csv, header=False, index=False)
    pd.DataFrame(original_ds).to_csv(orig_csv, header=False, index=False)
    meta_txt.write_text(
        f"target_cols={target_cols}\n"
        f"n_runs={vals_ds.shape[0]}\n"
        f"rows={vals_ds.shape[1]}\n"
        f"example_min_cols_file={min_path}\n",
        encoding="utf-8",
    )

    # Optional: plot PNG for quick inspection
    sigma_png = output_folder / "leaveout_sigma_map.png"

    coords_df = None
    try:
        # Same coords file you already generate in sections
        coords_csv = (validation_root.parent / "3_sections" / "cpt_coords_with_distances.csv")
        if coords_csv.exists():
            coords_df = pd.read_csv(coords_csv)
    except Exception:
        coords_df = None

    _plot_uncertainty_map_png(
        sigma_map,
        sigma_png,
        y_top_m=y_top_m,
        y_bottom_m=y_bottom_m,
        xmin=0.0,
        xmax=float(sigma_map.shape[1] - 1),
        coords=coords_df,
        show_cpt_locations=True,
        font_size=10,
        )
    log(f"Saved sigma PNG: {sigma_png}")

    log(f"Saved sigma map: {sigma_csv}")
    log(f"Saved downsampled original: {orig_csv}")
    log(f"Saved metadata: {meta_txt}")

    return {
        "target_cols": target_cols,
        "n_runs": int(vals_ds.shape[0]),
        "sigma_map_csv": str(sigma_csv),
        "original_downsampled_csv": str(orig_csv),
    }
