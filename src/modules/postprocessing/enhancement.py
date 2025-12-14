"""
Enhancement module for boundary sharpening.

This module handles enhancement of generated schemas using various methods.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_enhancement(
    gan_images_folder: Path,
    output_folder: Path,
    sections_folder: Path,
    y_top_m: float,
    y_bottom_m: float,
    show_cpt_locations: bool = True,
    method: str = "guided_filter",
):
    """Apply boundary enhancement to generated schemas.

    Args:
        gan_images_folder: Folder containing original GAN output CSVs
        output_folder: Folder where enhanced CSVs and PNGs will be saved
        sections_folder: Folder containing manifest and coordinates
        y_top_m: Top depth in meters for visualization
        y_bottom_m: Bottom depth in meters for visualization
        show_cpt_locations: Whether to show CPT position markers
        method: Enhancement method to use
    """
    from core.boundary_enhancement import enhance_schema_from_file, create_enhanced_png

    logger.info(f"Enhancement method: {method}")
    logger.info(f"Input folder: {gan_images_folder}")
    logger.info(f"Output folder: {output_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Load manifest and coords for PNG generation
    manifest_csv = sections_folder / "manifest_sections.csv"
    coords_csv = sections_folder / "cpt_coords_with_distances.csv"

    create_pngs = manifest_csv.exists() and coords_csv.exists()
    if not create_pngs:
        logger.warning("Manifest or coords CSV not found. PNGs will not be created.")

    # Find all GAN-generated CSV files
    gan_csv_files = sorted(gan_images_folder.glob("*_gan.csv"))

    if not gan_csv_files:
        logger.warning(f"No GAN CSV files found in {gan_images_folder}")
        return

    logger.info(f"Found {len(gan_csv_files)} GAN images to enhance")

    ok, fail = 0, 0

    for i, csv_file in enumerate(gan_csv_files, 1):
        try:
            output_csv = output_folder / csv_file.name
            output_png = output_folder / csv_file.name.replace(".csv", ".png")

            # Apply enhancement
            enhanced_csv, method_used = enhance_schema_from_file(
                csv_file, output_csv, method=method
            )

            # Create PNG visualization
            if create_pngs:
                try:
                    create_enhanced_png(
                        enhanced_csv,
                        output_png,
                        manifest_csv,
                        coords_csv,
                        y_top_m,
                        y_bottom_m,
                        show_cpt_locations=show_cpt_locations,
                    )
                except Exception as png_error:
                    logger.warning(
                        f"Failed to create PNG for {csv_file.name}: {png_error}"
                    )

            ok += 1
            if (i % 10 == 0) or (i == len(gan_csv_files)):
                logger.info(
                    f"[{i:03d}/{len(gan_csv_files)}] Enhanced: {csv_file.name} using {method_used}"
                )

        except Exception as e:
            fail += 1
            logger.error(f"[{i:03d}/{len(gan_csv_files)}] FAIL on {csv_file.name}: {e}")

    logger.info(f"Enhancement complete: {ok} succeeded, {fail} failed")
    logger.info(f"Enhanced schemas saved in: {output_folder}")
    logger.info("Step 6 complete.")
