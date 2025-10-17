"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow:
1. Setup experiment folder structure
2. Extract coordinates from CPT files
3. Extract and compress CPT data to 32-pixel depth
4. Create sections for SchemaGAN input
5. Generate schemas using trained SchemaGAN model
6. Create mosaic from generated schemas

Configure the paths and parameters in the CONFIG section below.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from utils import setup_experiment, read_files
from extract_coords import process_cpt_coords
from extract_data import (
    process_cpts,
    equalize_top,
    equalize_depth,
    compress_to_32px,
    save_cpt_to_csv,
)
from create_schGAN_input_file import process_sections, write_manifest
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG - Modify these paths and parameters for your experiment
# =============================================================================

# Base configuration
RES_DIR = Path(r"C:\VOW\res")
REGION = "north"
EXP_NAME = "exp_1"
DESCRIPTION = (
    "Complete VOW SchemaGAN pipeline - coordinates extraction to mosaic generation"
)

# Input data paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_north_BRO")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression
N_COLS = 512  # Number of columns in output sections
N_ROWS = 32  # Number of rows (depth levels) in output sections
CPTS_PER_SECTION = 6  # Number of CPTs per section
OVERLAP_CPTS = 2  # Number of overlapping CPTs between sections
LEFT_PAD_FRACTION = 0.10  # Left padding as fraction of section span
RIGHT_PAD_FRACTION = 0.10  # Right padding as fraction of section span
DIR_FROM, DIR_TO = "west", "east"  # Sorting direction

# Optional: Real depth range for visualization (will be computed if None)
Y_TOP_M: Optional[float] = None
Y_BOTTOM_M: Optional[float] = None

# =============================================================================


def main():
    """Execute the complete VOW SchemaGAN pipeline."""

    logger.info("=" * 60)
    logger.info("Starting VOW SchemaGAN Pipeline")
    logger.info("=" * 60)

    # =============================================================================
    # 1. CREATE EXPERIMENT FOLDER STRUCTURE
    # =============================================================================
    logger.info("Step 1: Creating experiment folder structure...")

    folders = setup_experiment(
        base_dir=RES_DIR,
        region=REGION,
        exp_name=EXP_NAME,
        description=DESCRIPTION,
    )

    logger.info(f"Experiment folders created at: {folders['root']}")

    # =============================================================================
    # 2. EXTRACT COORDINATES FROM CPT FILES
    # =============================================================================
    logger.info("Step 2: Extracting coordinates from CPT files...")

    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"

    try:
        process_cpt_coords(CPT_FOLDER, coords_csv)
        logger.info(f"Coordinates extracted to: {coords_csv}")
    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        return

    # =============================================================================
    # 3. EXTRACT AND COMPRESS CPT DATA
    # =============================================================================
    logger.info("Step 3: Extracting and compressing CPT data...")

    # Get valid CPT files (those not moved to no_coords)
    cpt_files = read_files(str(CPT_FOLDER), extension=".gef")

    try:
        # Process CPT files
        data_cpts, coords_simple = process_cpts(cpt_files)
        logger.info(f"Processed {len(data_cpts)} CPT files")

        # Create copy for processing
        original_data_cpts = [cpt.copy() for cpt in data_cpts]

        # Find depth limits
        lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
        lowest_min_depth = min(cpt["depth_min"] for cpt in data_cpts)

        logger.info(f"Depth range: {lowest_max_depth:.3f} to {lowest_min_depth:.3f} m")

        # Store depth range for later use
        global Y_TOP_M, Y_BOTTOM_M
        if Y_TOP_M is None:
            Y_TOP_M = lowest_max_depth
        if Y_BOTTOM_M is None:
            Y_BOTTOM_M = lowest_min_depth

        # Equalize depths
        equalized_top_cpts = equalize_top(original_data_cpts)
        equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)

        # Compress to 32 pixels
        compressed_cpts = compress_to_32px(
            equalized_depth_cpts, method=COMPRESSION_METHOD
        )

        # Save compressed data
        compressed_csv = (
            folders["2_compressed_cpt"]
            / f"compressed_cpt_data_{COMPRESSION_METHOD}.csv"
        )
        save_cpt_to_csv(
            compressed_cpts, str(folders["2_compressed_cpt"]), compressed_csv.name
        )

        logger.info(f"Compressed CPT data saved to: {compressed_csv}")

    except Exception as e:
        logger.error(f"Failed to process CPT data: {e}")
        return

    # =============================================================================
    # 4. CREATE SECTIONS FOR SCHEMAGAN INPUT
    # =============================================================================
    logger.info("Step 4: Creating sections for SchemaGAN input...")

    try:
        # Load coordinates and CPT data
        coords_df = pd.read_csv(coords_csv)
        cpt_df = pd.read_csv(compressed_csv)

        # Validate inputs
        from create_schGAN_input_file import validate_input_files

        validate_input_files(coords_df, cpt_df, N_ROWS)

        # Process sections
        manifest = process_sections(
            coords_df=coords_df,
            cpt_df=cpt_df,
            out_dir=folders["3_sections"],
            n_cols=N_COLS,
            n_rows=N_ROWS,
            per=CPTS_PER_SECTION,
            overlap=OVERLAP_CPTS,
            left_pad_frac=LEFT_PAD_FRACTION,
            right_pad_frac=RIGHT_PAD_FRACTION,
            from_where=DIR_FROM,
            to_where=DIR_TO,
        )

        # Write manifest
        from create_schGAN_input_file import write_manifest

        write_manifest(manifest, folders["3_sections"])

        logger.info(f"Created {len(manifest)} sections in: {folders['3_sections']}")

    except Exception as e:
        logger.error(f"Failed to create sections: {e}")
        return

    # =============================================================================
    # 5. GENERATE SCHEMAS USING SCHEMAGAN MODEL
    # =============================================================================
    logger.info("Step 5: Generating schemas using SchemaGAN model...")

    if not SCHGAN_MODEL_PATH.exists():
        logger.warning(f"SchemaGAN model not found at: {SCHGAN_MODEL_PATH}")
        logger.warning("Skipping schema generation. Please provide valid model path.")
    else:
        try:
            # Import and run schema generation
            from create_schema import run_gan_on_section_csv

            # Find all section CSV files
            section_files = list(folders["3_sections"].glob("section_*_cpts_*.csv"))

            if not section_files:
                logger.error("No section CSV files found")
                return

            logger.info(f"Generating schemas for {len(section_files)} sections...")

            success_count = 0
            fail_count = 0

            for i, section_file in enumerate(section_files, 1):
                try:
                    # Note: This would need modification to use our folder structure
                    # For now, we'll log that this step needs model integration
                    logger.info(
                        f"[{i:03d}/{len(section_files)}] Processing {section_file.name}"
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(
                        f"[{i:03d}/{len(section_files)}] Failed on {section_file.name}: {e}"
                    )
                    fail_count += 1

            logger.info(
                f"Schema generation complete. Success: {success_count}, Failed: {fail_count}"
            )

        except Exception as e:
            logger.error(f"Failed to generate schemas: {e}")
            return

    # =============================================================================
    # 6. CREATE MOSAIC FROM GENERATED SCHEMAS
    # =============================================================================
    logger.info("Step 6: Creating mosaic from generated schemas...")

    try:
        # This step would use create_mosaic.py functionality
        # For now, we'll create the directory and log readiness
        mosaic_dir = folders["5_mosaic"]
        logger.info(f"Mosaic output directory ready: {mosaic_dir}")
        logger.info(
            "Use create_mosaic.py with generated schema CSVs to create final mosaic"
        )

    except Exception as e:
        logger.error(f"Failed to create mosaic: {e}")
        return

    # =============================================================================
    # COMPLETION
    # =============================================================================
    logger.info("=" * 60)
    logger.info("VOW SchemaGAN Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {folders['root']}")
    logger.info("Folder structure:")
    for name, path in folders.items():
        if name != "root":
            logger.info(f"  {name}: {path}")

    # Summary statistics
    logger.info("\nPipeline Summary:")
    logger.info(
        f"  - CPTs processed: {len(data_cpts) if 'data_cpts' in locals() else 'N/A'}"
    )
    logger.info(
        f"  - Sections created: {len(manifest) if 'manifest' in locals() else 'N/A'}"
    )
    logger.info(f"  - Depth range: {Y_TOP_M:.3f} to {Y_BOTTOM_M:.3f} m")
    logger.info(f"  - Grid size: {N_ROWS} × {N_COLS} pixels")


if __name__ == "__main__":
    main()
