"""
Coordinate extraction module for CPT data.

This module handles extraction and validation of CPT coordinates from GEF files.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_coordinate_extraction(cpt_folder: Path, coords_csv: Path):
    """Extract and validate CPT coordinates from GEF files.

    This function processes all .gef files in the specified folder, validates their
    coordinates, and exports them to a CSV file. Invalid files are moved to a 'no_coords'
    subfolder.

    Args:
        cpt_folder: Path to folder containing .gef CPT files.
        coords_csv: Output path for the coordinates CSV file.

    Raises:
        FileNotFoundError: If the CPT folder does not exist.

    Note:
        Coordinates are validated for Netherlands RD system (6-digit format).
        Auto-corrects common scaling errors (e.g., 123.456 → 123456.000).
    """
    from core.extract_coords import process_cpt_coords

    if not cpt_folder.exists():
        raise FileNotFoundError(f"CPT folder does not exist: {cpt_folder}")

    process_cpt_coords(cpt_folder, coords_csv)
    logger.info(f"Coordinates extracted to: {coords_csv}")
    logger.info("Step 2 complete.")
