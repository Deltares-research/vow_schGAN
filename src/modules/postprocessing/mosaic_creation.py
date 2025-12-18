"""
Mosaic creation module.

This module handles combining schema sections into seamless mosaics.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def run_mosaic_creation(
    sections_folder: Path,
    gan_images_folder: Path,
    mosaic_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
    n_cols: int,
    n_rows: int,
    show_cpt_locations: bool = True,
    create_colormap_func: Optional[Callable] = None,
    ic_boundaries: Optional[Tuple] = None,
    mosaic_prefix: str = "schemaGAN",
    file_suffix: str = "gan",
):
    """Combine all generated schema sections into a seamless mosaic.

    Args:
        sections_folder: Path to folder containing section manifest and coordinates.
        gan_images_folder: Path to folder with generated schema CSV files.
        mosaic_folder: Output folder for mosaic files.
        y_top_m: Top depth in meters for visualization axis.
        y_bottom_m: Bottom depth in meters for visualization axis.
        n_cols: Number of columns in sections.
        n_rows: Number of rows in sections.
        show_cpt_locations: Whether to show CPT position markers.
        create_colormap_func: Function to create custom colormap.
        ic_boundaries: Tuple of IC boundary values for colormap.
        mosaic_prefix: Prefix for output files.
        file_suffix: Suffix for input files.
    """
    from core.create_mosaic import (
        load_inputs,
        find_latest_gan_csv_for_row,
        build_mosaic,
        plot_mosaic,
    )

    # Ensure output directory exists
    mosaic_folder.mkdir(parents=True, exist_ok=True)

    # Check if we have generated schemas
    gan_files = list(gan_images_folder.glob(f"*_{file_suffix}.csv"))
    if not gan_files:
        logger.warning(f"No {file_suffix} files found. Skipping mosaic creation.")
        return

    logger.info(f"Creating mosaic from {len(gan_files)} {file_suffix} files...")

    # Load manifest and coordinates
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    logger.info(f"Loading manifest from: {manifest_csv}")
    logger.info(f"Loading coordinates from: {coords_csv}")

    try:
        manifest, coords = load_inputs(manifest_csv, coords_csv)
    except Exception as e:
        logger.error(f"Failed to load manifest or coordinates: {e}")
        raise

    # Update create_mosaic module constants temporarily
    from core import create_mosaic

    original_values = _save_original_constants(create_mosaic)

    create_mosaic.GAN_DIR = gan_images_folder
    create_mosaic.N_COLS = n_cols
    create_mosaic.N_ROWS_WINDOW = n_rows
    create_mosaic.Y_TOP_M = y_top_m
    create_mosaic.Y_BOTTOM_M = y_bottom_m
    create_mosaic.FILE_SUFFIX = file_suffix

    try:
        # Match files to sections
        logger.info(f"Matching {file_suffix} CSV files to sections...")
        manifest["gan_csv"] = manifest.apply(
            lambda row: find_latest_gan_csv_for_row(row), axis=1
        )

        # Check for missing files
        missing = manifest[manifest["gan_csv"].isna()]
        if not missing.empty:
            logger.warning(
                f"Missing GAN CSV for sections: {missing['section_index'].tolist()}"
            )

        manifest = manifest.dropna(subset=["gan_csv"]).reset_index(drop=True)
        if manifest.empty:
            raise RuntimeError("No sections with GAN CSVs found.")

        # Build mosaic
        logger.info("Building mosaic with vertical and horizontal blending...")
        mosaic, xmin, xmax, global_dx, n_rows_total = build_mosaic(manifest, coords)
        logger.info(f"Mosaic built: shape={mosaic.shape}, n_rows_total={n_rows_total}")

        # Save mosaic CSV
        mosaic_csv = mosaic_folder / f"{mosaic_prefix}_mosaic.csv"
        pd.DataFrame(mosaic).to_csv(mosaic_csv, index=False)
        logger.info(f"Mosaic CSV saved: {mosaic_csv}")

        # Create visualization
        mosaic_png = mosaic_folder / f"{mosaic_prefix}_mosaic.svg"
        logger.info(f"Creating mosaic visualization: {mosaic_png}")

        # Set visualization parameters
        if file_suffix == "uncertainty":
            vmin_val, vmax_val = None, None
            cmap_val = "hot"
            colorbar_label = "Uncertainty (Std Dev)"
            ic_boundaries_val = None
        else:
            if create_colormap_func:
                #cmap_val, vmin_val, vmax_val = create_colormap_func()
                cmap_val, vmin_val, vmax_val = "viridis", 1.3, 4.2
            else:
                cmap_val, vmin_val, vmax_val = "viridis", 1.3, 4.2
            colorbar_label = "IC Value"
            ic_boundaries_val = ic_boundaries

        plot_mosaic(
            mosaic,
            xmin,
            xmax,
            global_dx,
            n_rows_total,
            mosaic_png,
            coords=coords,
            show_cpt_locations=show_cpt_locations,
            vmin=vmin_val,
            vmax=vmax_val,
            cmap=cmap_val,
            colorbar_label=colorbar_label,
            ic_boundaries=ic_boundaries_val,
        )

        # Create interactive HTML
        try:
            from core.utils import create_interactive_html

            mosaic_html = mosaic_folder / f"{mosaic_prefix}_mosaic.html"
            extent = (xmin, xmax, n_rows_total - 1, 0)
            create_interactive_html(
                mosaic_png,
                mosaic_html,
                title=f"{mosaic_prefix.replace('_', ' ').title()} Mosaic",
                extent=extent,
                xlabel="Distance along line (m)",
                ylabel="Depth Index",
            )
            logger.info(f"Interactive HTML created: {mosaic_html.name}")
        except Exception as html_err:
            logger.warning(f"Could not create interactive HTML: {html_err}")

        logger.info(f"Mosaic complete: {mosaic_csv.name} & {mosaic_png.name}")
        logger.info("Step 6 complete.")

    finally:
        # Restore original values
        _restore_original_constants(create_mosaic, original_values)


def _save_original_constants(module):
    """Save original module constants."""
    return {
        "GAN_DIR": getattr(module, "GAN_DIR", None),
        "N_COLS": getattr(module, "N_COLS", None),
        "N_ROWS_WINDOW": getattr(module, "N_ROWS_WINDOW", None),
        "Y_TOP_M": getattr(module, "Y_TOP_M", None),
        "Y_BOTTOM_M": getattr(module, "Y_BOTTOM_M", None),
        "FILE_SUFFIX": getattr(module, "FILE_SUFFIX", "gan"),
    }


def _restore_original_constants(module, original_values):
    """Restore original module constants."""
    for key, value in original_values.items():
        if value is not None:
            setattr(module, key, value)
