"""
Main script for the VOW SchemaGAN pipeline.

This script orchestrates the complete workflow by calling individual script functions:
1. Setup experiment folder structure
2. Extract coordinates from CPT files (calls extract_coords.py)
3. Extract and compress CPT data (calls extract_data.py)
4. Create sections for SchemaGAN input (calls create_schGAN_input_file.py)
5. Generate schemas using trained SchemaGAN model (calls create_schema.py)
6. Create mosaic from generated schemas (calls create_mosaic.py)

Configure the paths and parameters in the CONFIG section below.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Add GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from utils import setup_experiment

# =============================================================================
# CONFIG - Modify these paths and parameters for your experiment
# =============================================================================

# Base configuration
RES_DIR = Path(r"C:\VOW\res")
REGION = "north"
EXP_NAME = "exp_3"
DESCRIPTION = "Lets try using less overlapped CPTs and see what happens. In this case just 1 overlapping"

# Input data paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_north_BRO")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"  # "mean" or "max" for IC value compression
N_COLS = 512  # Number of columns in output sections
N_ROWS = 32  # Number of rows (depth levels) in output sections
CPTS_PER_SECTION = 6  # Number of CPTs per section
OVERLAP_CPTS = 1  # Number of overlapping CPTs between sections
LEFT_PAD_FRACTION = 0.10  # Left padding as fraction of section span
RIGHT_PAD_FRACTION = 0.10  # Right padding as fraction of section span
DIR_FROM, DIR_TO = "west", "east"  # Sorting direction

# Optional: Real depth range for visualization (will be computed if None)
Y_TOP_M: Optional[float] = None
Y_BOTTOM_M: Optional[float] = None

# =============================================================================


def run_coordinate_extraction(cpt_folder: Path, coords_csv: Path):
    """Wrapper for extract_coords.py functionality."""
    from extract_coords import process_cpt_coords

    if not cpt_folder.exists():
        raise FileNotFoundError(f"CPT folder does not exist: {cpt_folder}")

    process_cpt_coords(cpt_folder, coords_csv)
    logger.info(f"Coordinates extracted to: {coords_csv}")


def run_cpt_data_processing(
    cpt_folder: Path, output_folder: Path, compression_method: str = "mean"
):
    """Wrapper for extract_data.py functionality."""
    from extract_data import (
        process_cpts,
        equalize_top,
        equalize_depth,
        compress_to_32px,
        save_cpt_to_csv,
    )
    from utils import read_files

    # Get CPT files
    cpt_files = read_files(str(cpt_folder), extension=".gef")

    # Process CPTs
    data_cpts, coords_simple = process_cpts(cpt_files)
    logger.info(f"Processed {len(data_cpts)} CPT files")

    # Find depth limits and equalize
    original_data_cpts = [cpt.copy() for cpt in data_cpts]
    lowest_max_depth = min(cpt["depth_max"] for cpt in data_cpts)
    lowest_min_depth = min(cpt["depth_min"] for cpt in data_cpts)

    logger.info(f"Depth range: {lowest_max_depth:.3f} to {lowest_min_depth:.3f} m")

    # Process data
    equalized_top_cpts = equalize_top(original_data_cpts)
    equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)
    compressed_cpts = compress_to_32px(equalized_depth_cpts, method=compression_method)

    # Save results
    output_filename = f"compressed_cpt_data_{compression_method}.csv"
    save_cpt_to_csv(compressed_cpts, str(output_folder), output_filename)

    output_path = output_folder / output_filename
    logger.info(f"Compressed CPT data saved to: {output_path}")

    return output_path, lowest_max_depth, lowest_min_depth


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
):
    """Wrapper for create_schGAN_input_file.py functionality."""
    from create_schGAN_input_file import (
        process_sections,
        write_manifest,
        validate_input_files,
    )
    import pandas as pd

    # Load data
    coords_df = pd.read_csv(coords_csv)
    cpt_df = pd.read_csv(cpt_csv)

    # Validate and process
    validate_input_files(coords_df, cpt_df, n_rows)

    manifest = process_sections(
        coords_df=coords_df,
        cpt_df=cpt_df,
        out_dir=output_folder,
        n_cols=n_cols,
        n_rows=n_rows,
        per=cpts_per_section,
        overlap=overlap,
        left_pad_frac=left_pad,
        right_pad_frac=right_pad,
        from_where=dir_from,
        to_where=dir_to,
    )

    write_manifest(manifest, output_folder)
    logger.info(f"Created {len(manifest)} sections in: {output_folder}")

    return manifest


def run_schema_generation(
    sections_folder: Path,
    gan_images_folder: Path,
    model_path: Path,
    y_top_m: float,
    y_bottom_m: float,
):
    """Wrapper for create_schema.py functionality with dynamic configuration."""

    # Configure create_schema module variables
    import create_schema

    create_schema.SECTIONS_DIR = sections_folder
    create_schema.OUT_DIR = gan_images_folder
    create_schema.PATH_TO_MODEL = model_path
    create_schema.MANIFEST_CSV = sections_folder / "manifest_sections.csv"
    create_schema.COORDS_WITH_DIST_CSV = (
        sections_folder / "cpt_coords_with_distances.csv"
    )
    create_schema.Y_TOP_M = y_top_m
    create_schema.Y_BOTTOM_M = y_bottom_m

    # Ensure output directory exists
    gan_images_folder.mkdir(parents=True, exist_ok=True)

    # Get section files
    section_files = sorted(sections_folder.glob("section_*_cpts_*.csv"))
    if not section_files:
        raise FileNotFoundError(f"No section CSVs found in {sections_folder}")

    logger.info(f"Generating schemas for {len(section_files)} sections...")

    # Load model and run generation
    from create_schema import model, run_gan_on_section_csv

    success_count = 0
    fail_count = 0

    for i, section_file in enumerate(section_files, 1):
        try:
            csv_out, png_out = run_gan_on_section_csv(section_file)
            success_count += 1
            logger.info(
                f"[{i:03d}/{len(section_files)}] Generated: {csv_out.name} & {png_out.name}"
            )
        except Exception as e:
            fail_count += 1
            logger.error(
                f"[{i:03d}/{len(section_files)}] Failed on {section_file.name}: {e}"
            )

    logger.info(
        f"Schema generation complete. Success: {success_count}, Failed: {fail_count}"
    )
    return success_count, fail_count


def run_mosaic_creation(
    sections_folder: Path,
    gan_images_folder: Path,
    mosaic_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
):
    """Wrapper for create_mosaic.py functionality with dynamic configuration."""

    # Configure create_mosaic module variables
    import create_mosaic

    create_mosaic.MANIFEST_CSV = sections_folder / "manifest_sections.csv"
    create_mosaic.COORDS_WITH_DIST_CSV = (
        sections_folder / "cpt_coords_with_distances.csv"
    )
    create_mosaic.GAN_DIR = gan_images_folder
    create_mosaic.OUT_DIR = mosaic_folder
    create_mosaic.Y_TOP_M = y_top_m
    create_mosaic.Y_BOTTOM_M = y_bottom_m

    # Ensure output directory exists
    mosaic_folder.mkdir(parents=True, exist_ok=True)

    # Check if we have generated schemas
    gan_files = list(gan_images_folder.glob("*_gan.csv"))
    if not gan_files:
        logger.warning("No generated schema files found. Skipping mosaic creation.")
        return

    logger.info(f"Creating mosaic from {len(gan_files)} generated schemas...")

    # Run mosaic creation
    from create_mosaic import main as create_mosaic_main

    create_mosaic_main()

    logger.info(f"Mosaic created in: {mosaic_folder}")


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
        base_dir=RES_DIR, region=REGION, exp_name=EXP_NAME, description=DESCRIPTION
    )
    logger.info(f"Experiment folders created at: {folders['root']}")

    # =============================================================================
    # 2. EXTRACT COORDINATES FROM CPT FILES
    # =============================================================================
    logger.info("Step 2: Extracting coordinates from CPT files...")

    coords_csv = folders["1_coords"] / "cpt_coordinates.csv"

    try:
        run_coordinate_extraction(CPT_FOLDER, coords_csv)
    except Exception as e:
        logger.error(f"Failed to extract coordinates: {e}")
        return

    # =============================================================================
    # 3. EXTRACT AND COMPRESS CPT DATA
    # =============================================================================
    logger.info("Step 3: Extracting and compressing CPT data...")

    try:
        compressed_csv, y_top, y_bottom = run_cpt_data_processing(
            CPT_FOLDER, folders["2_compressed_cpt"], COMPRESSION_METHOD
        )

        # Store depth range for later steps
        global Y_TOP_M, Y_BOTTOM_M
        if Y_TOP_M is None:
            Y_TOP_M = y_top
        if Y_BOTTOM_M is None:
            Y_BOTTOM_M = y_bottom

    except Exception as e:
        logger.error(f"Failed to process CPT data: {e}")
        return

    # =============================================================================
    # 4. CREATE SECTIONS FOR SCHEMAGAN INPUT
    # =============================================================================
    logger.info("Step 4: Creating sections for SchemaGAN input...")

    try:
        manifest = run_section_creation(
            coords_csv,
            compressed_csv,
            folders["3_sections"],
            N_COLS,
            N_ROWS,
            CPTS_PER_SECTION,
            OVERLAP_CPTS,
            LEFT_PAD_FRACTION,
            RIGHT_PAD_FRACTION,
            DIR_FROM,
            DIR_TO,
        )
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
            run_schema_generation(
                folders["3_sections"],
                folders["4_gan_images"],
                SCHGAN_MODEL_PATH,
                Y_TOP_M,
                Y_BOTTOM_M,
            )
        except Exception as e:
            logger.error(f"Failed to generate schemas: {e}")
            return

    # =============================================================================
    # 6. CREATE MOSAIC FROM GENERATED SCHEMAS
    # =============================================================================
    logger.info("Step 6: Creating mosaic from generated schemas...")

    try:
        run_mosaic_creation(
            folders["3_sections"],
            folders["4_gan_images"],
            folders["5_mosaic"],
            Y_TOP_M,
            Y_BOTTOM_M,
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
        f"  - Sections created: {len(manifest) if 'manifest' in locals() else 'N/A'}"
    )
    logger.info(f"  - Depth range: {Y_TOP_M:.3f} to {Y_BOTTOM_M:.3f} m")
    logger.info(f"  - Grid size: {N_ROWS} × {N_COLS} pixels")


if __name__ == "__main__":
    main()
