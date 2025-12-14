"""
Schema generation module using SchemaGAN model.

This module handles schema generation from section files using the trained GAN model.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import re
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from core.utils import IC_normalization, reverse_IC_normalization
from core.uncertainty_quantification import (
    compute_mc_dropout_uncertainty,
    visualize_uncertainty,
    save_uncertainty_csv,
)

logger = logging.getLogger(__name__)


def run_schema_generation(
    sections_folder: Path,
    gan_images_folder: Path,
    model_path: Path,
    y_top_m: float,
    y_bottom_m: float,
    n_rows: int,
    n_cols: int,
    ic_boundaries: Tuple[float, float, float, float, float, float],
    show_cpt_locations: bool = True,
    create_colormap_func=None,
    trace_func=None,
    uncertainty_folder: Optional[Path] = None,
    compute_uncertainty: bool = False,
    n_mc_samples: int = 50,
) -> Tuple[int, int]:
    """Generate detailed subsurface schemas using trained SchemaGAN model.

    Args:
        sections_folder: Path to folder containing section CSV files and manifest.
        gan_images_folder: Output folder for generated schema images and data.
        model_path: Path to trained SchemaGAN model file (.h5).
        y_top_m: Top depth in meters.
        y_bottom_m: Bottom depth in meters.
        n_rows: Number of rows in section (height).
        n_cols: Number of columns in section (width).
        ic_boundaries: Tuple of 6 IC boundary values for colormap.
        show_cpt_locations: Whether to show CPT position markers.
        create_colormap_func: Function to create custom IC colormap.
        trace_func: Function for verbose logging.
        uncertainty_folder: Output folder for uncertainty maps.
        compute_uncertainty: Whether to compute MC Dropout uncertainty.
        n_mc_samples: Number of MC samples for uncertainty estimation.

    Returns:
        Tuple of (success_count, fail_count)
    """
    # Ensure output directory exists
    gan_images_folder.mkdir(parents=True, exist_ok=True)

    # Get section files
    section_files = sorted(sections_folder.glob("section_*_cpts_*.csv"))
    if not section_files:
        raise FileNotFoundError(f"No section CSVs found in {sections_folder}")

    logger.info(f"Generating schemas for {len(section_files)} sections...")
    if trace_func:
        trace_func(
            f"Found {len(section_files)} section CSV files for schema generation"
        )

    # Load model
    logger.info("Loading SchemaGAN model...")
    model = load_model(model_path, compile=False)

    # Set random seed
    seed = np.random.randint(20220412, 20230412)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    # Load manifest and coordinates
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    manifest_df = pd.read_csv(manifest_csv)
    coords_df = pd.read_csv(coords_csv)

    success_count = 0
    fail_count = 0

    # Unpack IC boundaries
    (
        ic_min,
        ic_sand_sandmix,
        ic_sandmix_siltmix,
        ic_siltmix_clay,
        ic_clay_organic,
        ic_max,
    ) = ic_boundaries

    for i, section_file in enumerate(section_files, 1):
        try:
            # Load and prepare section data
            df = pd.read_csv(section_file)

            # Remove index column if it exists
            if df.shape[1] == n_cols + 1:
                df_vals = df.iloc[:, 1:]
            else:
                df_vals = df

            # Convert to GAN input format
            cs = df_vals.to_numpy(dtype=float).reshape(1, n_rows, n_cols, 1)

            # Normalize and predict
            cs_norm = IC_normalization([cs, cs])[0]
            gan_result = model.predict(cs_norm, verbose=0)
            gan_result = reverse_IC_normalization(gan_result)
            gan_result = np.squeeze(gan_result)

            # Save CSV
            output_csv = gan_images_folder / f"{section_file.stem}_seed{seed}_gan.csv"
            pd.DataFrame(gan_result).to_csv(output_csv, index=False)
            logger.info(f"[INFO] Created GAN CSV: {output_csv.name}")

            # Parse section index
            m = re.search(r"section_(\d+)", section_file.stem)
            sec_index = int(m.group(1)) if m else i

            # Get section placement for proper coordinates
            r = manifest_df.loc[manifest_df["section_index"] == sec_index]
            if not r.empty:
                r = r.iloc[0]
                total_span = float(r["span_m"] + r["left_pad_m"] + r["right_pad_m"])
                start_idx = int(r["start_idx"])
                end_idx = int(r["end_idx"])
                m0 = float(coords_df.loc[start_idx, "cum_along_m"])
                x0 = m0 - float(r["left_pad_m"])
                dx = 1.0 if total_span <= 0 else total_span / (n_cols - 1)
                x1 = x0 + (n_cols - 1) * dx
                cpt_positions = coords_df.loc[start_idx:end_idx, "cum_along_m"].values
            else:
                x0, x1, dx = 0, n_cols - 1, 1
                cpt_positions = []

            # Create and save PNG
            output_png = gan_images_folder / f"{section_file.stem}_seed{seed}_gan.png"

            # Get colormap
            if create_colormap_func:
                cmap, vmin, vmax = create_colormap_func()
            else:
                cmap, vmin, vmax = "viridis", 0, 4.5

            plt.figure(figsize=(10, 2.4))
            plt.imshow(
                gan_result,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                extent=[x0, x1, n_rows - 1, 0],
            )
            cbar = plt.colorbar(label="IC Value", extend="both", pad=0.07)
            cbar.ax.tick_params(labelsize=config.PLOT_FONT_SIZE)
            cbar.set_label("IC Value", fontsize=config.PLOT_FONT_SIZE)

            # Set custom ticks if using IC colormap
            if create_colormap_func:
                cbar.set_ticks(
                    [
                        ic_min,
                        ic_sand_sandmix,
                        ic_sandmix_siltmix,
                        ic_siltmix_clay,
                        ic_clay_organic,
                        ic_max,
                    ]
                )
                cbar.set_ticklabels(
                    [f"{v:g}" for v in ic_boundaries], fontsize=config.PLOT_FONT_SIZE
                )

            ax = plt.gca()
            ax.set_xlabel("Distance along line (m)", fontsize=config.PLOT_FONT_SIZE)
            ax.set_ylabel("Depth Index", fontsize=config.PLOT_FONT_SIZE)
            ax.tick_params(axis="both", labelsize=config.PLOT_FONT_SIZE)

            # Top x-axis: pixel index
            def m_to_px(x):
                return (x - x0) / dx if dx != 0 else 0

            def px_to_m(p):
                return x0 + p * dx

            top = ax.secondary_xaxis("top", functions=(m_to_px, px_to_m))
            top.set_xlabel(
                f"Pixel index (0…{n_cols-1})", fontsize=config.PLOT_FONT_SIZE
            )
            top.tick_params(labelsize=config.PLOT_FONT_SIZE)

            # Right y-axis: real depth
            def idx_to_meters(y_idx):
                return y_top_m + (y_idx / (n_rows - 1)) * (y_bottom_m - y_top_m)

            def meters_to_idx(y_m):
                denom = y_bottom_m - y_top_m
                return (
                    0.0
                    if abs(denom) < 1e-12
                    else (y_m - y_top_m) * (n_rows - 1) / denom
                )

            right = ax.secondary_yaxis(
                "right", functions=(idx_to_meters, meters_to_idx)
            )
            right.set_ylabel("Depth (m)", fontsize=config.PLOT_FONT_SIZE)
            right.tick_params(labelsize=config.PLOT_FONT_SIZE)

            # Add CPT position markers
            if show_cpt_locations:
                for cpt_x in cpt_positions:
                    ax.axvline(
                        x=cpt_x,
                        color="black",
                        linewidth=1,
                        linestyle="-",
                        alpha=0.5,
                        zorder=10,
                    )

            ax.set_xlim(x0, x1)

            plt.title(
                f"SchemaGAN Generated Image (Section {sec_index:03d}, Seed: {seed})",
                fontsize=config.PLOT_FONT_SIZE,
            )
            plt.tight_layout()
            plt.savefig(output_png, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"[INFO] Created GAN PNG: {output_png.name}")

            # Create interactive HTML
            try:
                from core.utils import create_interactive_html

                html_path = output_png.with_suffix(".html")
                create_interactive_html(
                    output_png, html_path, title=f"Section {sec_index:03d}"
                )
                if trace_func:
                    trace_func(f"Interactive HTML: {html_path.name}")
            except Exception:
                pass

            success_count += 1
            logger.info(
                f"[{i:03d}/{len(section_files)}] Generated: {output_csv.name} & {output_png.name}"
            )

            # Compute uncertainty if enabled
            if compute_uncertainty and uncertainty_folder is not None:
                _process_uncertainty(
                    cs_norm,
                    model,
                    section_file,
                    uncertainty_folder,
                    seed,
                    n_mc_samples,
                    n_rows,
                    n_cols,
                    x0,
                    x1,
                    y_top_m,
                    y_bottom_m,
                    cpt_positions,
                    show_cpt_locations,
                    i,
                    len(section_files),
                    trace_func,
                )

        except Exception as e:
            fail_count += 1
            logger.error(
                f"[{i:03d}/{len(section_files)}] Failed on {section_file.name}: {e}"
            )

    logger.info(
        f"Schema generation complete. Success: {success_count}, Failed: {fail_count}"
    )
    logger.info("Step 5 complete.")
    return success_count, fail_count


def _process_uncertainty(
    cs_norm,
    model,
    section_file,
    uncertainty_folder,
    seed,
    n_mc_samples,
    n_rows,
    n_cols,
    x0,
    x1,
    y_top_m,
    y_bottom_m,
    cpt_positions,
    show_cpt_locations,
    i,
    total,
    trace_func,
):
    """Process uncertainty quantification for a section."""
    try:
        logger.info(
            f"[{i:03d}/{total}] Computing MC Dropout uncertainty ({n_mc_samples} samples)..."
        )
        if trace_func:
            trace_func(f"MC Dropout: Starting {n_mc_samples} forward passes")

        # Compute uncertainty
        pred_mean, pred_std = compute_mc_dropout_uncertainty(
            model, cs_norm, n_samples=n_mc_samples
        )

        # Denormalize mean prediction
        pred_mean_denorm = reverse_IC_normalization(pred_mean)

        # Ensure correct shapes
        if pred_std.shape != (n_rows, n_cols):
            logger.warning(
                f"Uncertainty shape mismatch: got {pred_std.shape}, expected ({n_rows}, {n_cols})"
            )
            if pred_std.ndim == 3:
                pred_std = pred_std[:, :, 0]
            if pred_mean_denorm.ndim == 3:
                pred_mean_denorm = pred_mean_denorm[:, :, 0]

        # Save CSVs
        uncertainty_csv = (
            uncertainty_folder / f"{section_file.stem}_seed{seed}_uncertainty.csv"
        )
        save_uncertainty_csv(pred_std, uncertainty_csv)
        logger.info(f"[INFO] Saved uncertainty CSV: {uncertainty_csv.name}")

        mean_csv = uncertainty_folder / f"{section_file.stem}_seed{seed}_mean.csv"
        save_uncertainty_csv(pred_mean_denorm, mean_csv)
        logger.info(f"[INFO] Saved mean prediction CSV: {mean_csv.name}")

        # Create visualization
        uncertainty_png = (
            uncertainty_folder / f"{section_file.stem}_seed{seed}_uncertainty.png"
        )
        visualize_uncertainty(
            uncertainty_map=pred_std,
            output_png=uncertainty_png,
            x0=x0,
            x1=x1,
            y_top_m=y_top_m,
            y_bottom_m=y_bottom_m,
            cpt_positions=cpt_positions,
            show_cpt_locations=show_cpt_locations,
            mean_prediction=pred_mean_denorm,
        )
        logger.info(f"[INFO] Created uncertainty PNG: {uncertainty_png.name}")

    except Exception as ue:
        logger.warning(f"[{i:03d}/{total}] Failed to compute uncertainty: {ue}")
        if trace_func:
            trace_func(f"Uncertainty computation failed: {ue}")
