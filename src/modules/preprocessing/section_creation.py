"""
Section creation module.

This module handles creation of spatial sections with overlapping CPTs for SchemaGAN input.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_section_creation(
    coords_csv: Path,
    cpt_csv: Path,
    output_folder: Path,
    n_cols: int,
    n_rows: int,
    cpts_per_section: int,
    overlap: int,
    left_pad: float,
    right_pad: float,
    dir_from: str,
    dir_to: str,
    vertical_overlap: float = 0.0,
) -> List[Dict[str, Any]]:
    """Create spatial sections with overlapping CPTs for SchemaGAN input.

    This function sorts CPTs spatially, divides them into overlapping sections both
    horizontally (by CPT count) and vertically (by depth windows), and generates input
    matrices (n_rows x n_cols) where CPT data is positioned at correct spatial locations
    with padding. Areas without data are filled with zeros.

    Args:
        coords_csv: Path to CSV file with CPT coordinates (columns: name, x, y).
        cpt_csv: Path to compressed CPT data CSV (columns: Depth_Index, CPT names...).
        output_folder: Directory where section files will be saved.
        n_cols: Number of columns (width) in each section matrix.
        n_rows: Number of rows (depth levels) in each section matrix.
        cpts_per_section: Number of CPTs to include in each section.
        overlap: Number of CPTs that overlap between consecutive sections.
        left_pad: Left padding as fraction of section span.
        right_pad: Right padding as fraction of section span.
        dir_from: Starting direction ("west", "east", "north", or "south").
        dir_to: Ending direction ("west", "east", "north", or "south").
        vertical_overlap: Vertical overlap percentage between depth windows.

    Returns:
        Combined manifest list of dictionaries for all depth windows.
    """
    from core.create_schGAN_input_file import (
        process_sections,
        write_manifest,
        validate_input_files,
        split_cpt_into_windows,
    )

    # Load data
    coords_df = pd.read_csv(coords_csv)
    cpt_df_full = pd.read_csv(cpt_csv)

    # Adjust padding strategy based on horizontal overlap
    actual_left_pad = left_pad
    actual_right_pad = right_pad

    if overlap == 0:
        logger.info(
            "No horizontal overlap (overlap=0) detected. "
            "Using extended padding (50%) to fill gaps between CPT groups."
        )
        actual_left_pad = 0.50
        actual_right_pad = 0.50
    else:
        actual_left_pad = left_pad
        actual_right_pad = right_pad
        logger.info(
            f"Horizontal overlap detected (overlap={overlap}). "
            f"Using padding: left={actual_left_pad:.2%}, right={actual_right_pad:.2%}"
        )

    # Determine if we need vertical windowing
    total_cpt_rows = len(cpt_df_full)

    if total_cpt_rows == n_rows:
        # No vertical windowing needed
        logger.info(
            f"CPT data has {total_cpt_rows} rows matching n_rows={n_rows}. "
            "Processing as single depth level."
        )

        validate_input_files(coords_df, cpt_df_full, n_rows)

        manifest = process_sections(
            coords_df=coords_df,
            cpt_df=cpt_df_full,
            out_dir=output_folder,
            n_cols=n_cols,
            n_rows=n_rows,
            per=cpts_per_section,
            overlap=overlap,
            left_pad_frac=actual_left_pad,
            right_pad_frac=actual_right_pad,
            from_where=dir_from,
            to_where=dir_to,
            depth_window=None,
            depth_start_row=None,
            depth_end_row=None,
            write_distances=True,
        )

    elif total_cpt_rows > n_rows:
        # Vertical windowing required
        logger.info(
            f"CPT data has {total_cpt_rows} rows > n_rows={n_rows}. "
            f"Splitting into vertical windows with {vertical_overlap:.1f}% overlap."
        )

        # Split CPT data into vertical windows
        depth_windows = split_cpt_into_windows(
            cpt_df=cpt_df_full,
            window_rows=n_rows,
            vertical_overlap_pct=vertical_overlap,
        )

        logger.info(f"Created {len(depth_windows)} depth windows")

        all_manifests = []

        for w_idx, start_row, end_row, cpt_df_win in depth_windows:
            logger.info(
                f"Processing depth window z_{w_idx:02d} "
                f"(rows {start_row}..{end_row-1} of original CPT data)..."
            )

            validate_input_files(coords_df, cpt_df_win, n_rows)

            # Only write distances file for first window (identical for all)
            write_dists = w_idx == 0

            manifest = process_sections(
                coords_df=coords_df,
                cpt_df=cpt_df_win,
                out_dir=output_folder,
                n_cols=n_cols,
                n_rows=n_rows,
                per=cpts_per_section,
                overlap=overlap,
                left_pad_frac=actual_left_pad,
                right_pad_frac=actual_right_pad,
                from_where=dir_from,
                to_where=dir_to,
                depth_window=w_idx,
                depth_start_row=start_row,
                depth_end_row=end_row,
                write_distances=write_dists,
            )

            all_manifests.extend(manifest)

        manifest = all_manifests

    else:
        raise ValueError(
            f"CPT data has {total_cpt_rows} rows but n_rows={n_rows}. "
            f"CPT data must have at least n_rows rows."
        )

    write_manifest(manifest, output_folder)
    logger.info(f"Created {len(manifest)} total sections in: {output_folder}")
    logger.info("Step 4 complete.")

    return manifest
