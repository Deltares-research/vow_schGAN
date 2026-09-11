# Based on extract_data.py, but WITHOUT compressing the CPTs: only the equalize_top
# step is applied, so each CPT keeps its real (uncompressed) depth resolution and size.
# equalize_depth and compress_cpt_data are intentionally NOT used here.
# CPTs are aligned by sample position (assuming a constant ~0.02 m step, verified via
# check_uniform_step) rather than interpolated, so only real measured IC values are kept.

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

try:
    from .utils import read_files
except ImportError:
    # Allow running this file directly (e.g. `python extract_data_full.py`)
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import read_files

# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod

# Get logger for this module
logger = logging.getLogger(__name__)

# Disable noisy GeoLib logging only for GeoLib, not for everything
geolib_logger = logging.getLogger("geolib_plus")
geolib_logger.setLevel(logging.ERROR)


def process_cpts(gef_list: list[Path]):
    """
    Process CPT files using geolib_plus library.

    Params:
        gef_list (list[Path]): List of paths to GEF files.

    Returns:
        data (list): List of dicts with CPT data.
        coords (list): List of dicts with CPT coordinates.
    """
    data = []
    coords = []
    for cpt in gef_list:
        cpt_gef = GefCpt()
        try:
            cpt_gef.read(cpt)
        except Exception as e:
            logger.warning(f"Error reading CPT file {cpt}: {e}")
            continue

        cpt_gef.pre_process_data()

        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.ROBERTSON
        interpreter.user_defined_water_level = True
        cpt_gef.pwp = 0

        cpt_gef.interpret_cpt(interpreter)

        # Extract relevant part of the file name
        file_name = cpt.stem  # Get the file name without extension
        cpt_id = file_name.split("_")[0]  # Extract up to the first underscore

        data.append(
            {
                "Name": cpt_id,
                "depth": cpt_gef.depth_to_reference,
                "depth_max": max(cpt_gef.depth_to_reference, default=0),
                "depth_min": min(cpt_gef.depth_to_reference, default=0),
                "IC": cpt_gef.IC,
                "coordinates": cpt_gef.coordinates,
            }
        )

        coords.append(
            {"name": cpt_id, "x": cpt_gef.coordinates[0], "y": cpt_gef.coordinates[1]}
        )
        # Log which file is processed
        logger.info(f"Processed CPT file: {cpt.name}")

    return data, coords


def equalize_top(data_cpts: list[dict]) -> list[dict]:
    """
    Equalize the starting depth of all CPTs by removing IC and depth data above the lowest maximum depth.

    Params:
        data_cpts (list): List of dictionaries containing CPT data.

    Returns:
        list: List of dictionaries with equalized CPT data.
    """
    equalized_cpts = []

    # Find the lowest maximum depth across all CPTs
    lowest_top_depth = min(cpt["depth_max"] for cpt in data_cpts)
    logger.info(f"Equalizing to the lowest maximum depth of {lowest_top_depth} m")

    for cpt in data_cpts:
        equalized_cpt = cpt.copy()

        depth = equalized_cpt["depth"]
        IC = equalized_cpt["IC"]

        # Filter depth and IC values that are strictly below the lowest maximum depth
        filtered_data = [(d, ic) for d, ic in zip(depth, IC) if d < lowest_top_depth]

        equalized_cpt["depth"], equalized_cpt["IC"] = (
            zip(*filtered_data) if filtered_data else ([], [])
        )

        equalized_cpt["depth_min"] = min(equalized_cpt["depth"], default=0)
        equalized_cpt["depth_max"] = max(equalized_cpt["depth"], default=0)

        equalized_cpts.append(equalized_cpt)

    return equalized_cpts


def check_uniform_step(data_cpts: list[dict], expected_step: float = 0.02, tolerance: float = 0.1) -> float:
    """
    Verify that every CPT samples depth at roughly the same constant interval.

    Params:
        data_cpts (list): List of dictionaries containing CPT data.
        expected_step (float): Expected depth increment in meters (default 0.02 m, standard for CPTs).
        tolerance (float): Fractional tolerance allowed around expected_step (default 10%).

    Returns:
        float: The median step size actually observed across all CPTs.
    """
    steps = []
    for cpt in data_cpts:
        depth = np.asarray(cpt["depth"], dtype=float)
        if len(depth) < 2:
            continue
        cpt_step = np.median(np.abs(np.diff(depth)))
        steps.append(cpt_step)
        if abs(cpt_step - expected_step) > expected_step * tolerance:
            logger.warning(
                f"CPT {cpt['Name']}: median depth step is {cpt_step:.4f} m, "
                f"expected ~{expected_step} m"
            )

    observed_step = float(np.median(steps))
    logger.info(f"Observed median depth step across all CPTs: {observed_step:.4f} m")
    return observed_step


def save_full_cpt_to_csv(
    data_cpts: list, output_folder: str, output_name: str, depth_step: float = 0.02
):
    """
    Save the uncompressed, equalized-top CPT data to a single CSV file.

    Only real measured IC values are kept (no interpolation). Each CPT has its own
    absolute reference depth (surface elevation minus measured depth), so exact
    depth values almost never match across CPTs. Since equalize_top already aligns
    all CPTs to the same starting reference and CPTs sample at a constant interval
    (verified by check_uniform_step), CPTs are instead aligned by sample position:
    row i is the i-th real measurement after the equalized top for every CPT,
    labeled with a relative "Depth_from_top" (0, depth_step, 2*depth_step, ...).
    Shorter CPTs are padded with NaN once their real data runs out.

    Args:
        data_cpts (list): List of dictionaries containing equalized (uncompressed) CPT data.
        output_folder (str): Directory where the CSV file will be saved.
        output_name (str): Name of the output CSV file.
        depth_step (float): Nominal depth increment in meters used to label rows (default 0.02 m).

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    max_len = max(len(cpt["IC"]) for cpt in data_cpts)
    depth_from_top = np.arange(max_len) * depth_step

    data = {"Depth_from_top": depth_from_top}
    for cpt in data_cpts:
        IC = np.asarray(cpt["IC"], dtype=float)
        padded = np.full(max_len, np.nan)
        padded[: len(IC)] = IC  # real measured values only, no interpolation
        data[cpt["Name"]] = padded

    df = pd.DataFrame(data)

    output_file = os.path.join(output_folder, output_name)
    df.to_csv(output_file, index=False)
    logger.info(f"Full (uncompressed) CPT data saved to: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    #### USER INPUT ####
    CPT_FOLDER = Path(r"C:\VOW\dike_south_BRO")
    OUT_FOLDER = Path(r"C:\VOW\res\input")
    OUT_NAME = "full_cpt_data_equalized_top_south.csv"
    ####################

    # Directory containing the CPT files
    cpts_path = read_files(path=CPT_FOLDER, extension=".gef")

    # Process CPT files (interpret IC, no equalization yet)
    data_cpts, coords = process_cpts(cpts_path)

    # Equalize only the top (shallow) depth across all CPTs
    equalized_top_cpts = equalize_top(data_cpts)

    # Verify all CPTs sample at the standard ~0.02 m interval before position-based alignment
    depth_step = check_uniform_step(equalized_top_cpts, expected_step=0.02)

    # Save the uncompressed, equalized-top CPT data to CSV
    save_full_cpt_to_csv(equalized_top_cpts, OUT_FOLDER, OUT_NAME, depth_step=depth_step)
