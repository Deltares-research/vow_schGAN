# In this code, we give a folder with .gef files and we generate a csv with coordinates from it.
# The output csv has columns: name, x, y.
# It does several steps in between.
# 1st: checks that all .gef files have coordinates. If not, move the file to a subfolder called "no_coords".
# 2nd: checks that the coordinates are in the right format (not 123.456 instead of 123456.000) and in the Netherlands RD coordinate system.
# 3rd: saves the coordinates in a csv file tagging those that were fixed.

import sys
from pathlib import Path

# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

import csv
import numpy as np
from pathlib import Path
import logging

# sys.path.append("C:\schemaGAN")
# import schemaGAN

from utils import read_files
from geolib_plus.gef_cpt import GefCpt

# to avoid warnings coming from GeoLib
initial_logging_level = logging.getLogger().getEffectiveLevel()
logging.disable(logging.ERROR)
# remove warning messages
logging.getLogger().setLevel(logging.ERROR)


def check_file_for_coords(
    cpt_object: GefCpt, file: Path, cpt_folder: Path, sucess_count: int, fail_count: int
):
    """
    Check if a GEF file has valid coordinates. If not, move the file to a subfolder "no_coords".
    Args:
        cpt_object (GefCpt): The CPT object containing the coordinates.
        file (Path): The path to the GEF file.
        cpt_folder (Path): The folder containing the GEF files.
        sucess_count (int): The count of files with valid coordinates.
        fail_count (int): The count of files moved to "no_coords".
    Returns:
        Tuple[int, int]: Updated counts of successful and failed files.
    """

    try:
        x, y = cpt_object.coordinates  # tuple (x, y)

        if x is None or y is None or np.isnan(x) or np.isnan(y):
            no_coords_folder = cpt_folder / "no_coords"
            no_coords_folder.mkdir(exist_ok=True)
            file.rename(no_coords_folder / file.name)
            fail_count += 1
        else:
            sucess_count += 1

    except Exception as e:
        no_coords_folder = cpt_folder / "no_coords"
        no_coords_folder.mkdir(exist_ok=True)
        file.rename(no_coords_folder / file.name)
        fail_count += 1

    return sucess_count, fail_count


def fix_broken_coords(cpt_object: GefCpt):
    """
    Check and fix coordinates that are clearly scaled or formatted incorrectly.
    Coordinates in the Netherlands (RD system) should have a 6-digit integer part,
    e.g., 123456.000 (not 123.456 or 123456789).

    Rules:
      - If already 6 digits -> keep as is.
      - If <1000 -> multiply by 1000 (common scaling issue).
      - Otherwise -> keep unchanged.

    Returns:
        tuple: (x_fixed, y_fixed, was_fixed)
    """
    # Get original coordinates
    x, y = cpt_object.coordinates

    def int_part_is_six_digits(value):
        """Return True if the integer part has exactly 6 digits."""
        try:
            ival = int(abs(float(value)))
            return 100000 <= ival <= 999999
        except Exception:
            return False

    def fix_value(value):
        """Try to fix a single coordinate according to the rules above."""
        try:
            v = float(value)
        except Exception:
            return value, False

        # Already correct
        if int_part_is_six_digits(v):
            return v, False

        # Looks too small → probably missing scale factor
        if v < 1000:
            v_fixed = v * 1000.0
            if int_part_is_six_digits(v_fixed):
                return v_fixed, True

        # Otherwise leave unchanged
        return v, False

    x_fixed, x_fixed_flag = fix_value(x)
    y_fixed, y_fixed_flag = fix_value(y)
    was_fixed = x_fixed_flag or y_fixed_flag

    return x_fixed, y_fixed, was_fixed


def save_coordinates_to_csv(rows, output_csv: Path):
    """
    Save extracted and (optionally) fixed coordinates to a CSV file.

    Args:
        rows (list of tuples): Each tuple should contain (name, x, y, fixed)
        output_csv (Path): Path to the output CSV file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "fixed"])  # header
        writer.writerows(rows)

    print(f"Coordinates saved to: {output_csv}")


def process_cpt_coords(cpt_folder: Path, output_csv: Path) -> None:
    """
    Process .GEF files in a specified folder to extract and validate coordinates,
    then save the results to a CSV file.
    """
    # 1. Read the .GEF files from the specified folder
    all_files = read_files(str(cpt_folder), extension=".gef")
    print(f"Processing {len(all_files)} files for coordinates...")

    sucess_count = 0
    fail_count = 0
    rows = []

    # 2. Loop through each file
    for file in all_files:
        # 3. Create the dGeoLib CPT object and read the file
        cpt = GefCpt()
        cpt.read(str(file))
        # 3. Check file by file for coordinates and move those without to "no_coords"
        sucess_count, fail_count = check_file_for_coords(
            cpt_object=cpt,
            file=file,
            cpt_folder=cpt_folder,
            sucess_count=sucess_count,
            fail_count=fail_count,
        )
        # 4. Fix broken coordinates
        x_fixed, y_fixed, was_fixed = fix_broken_coords(cpt)

        # 6. Append to rows for the CSV
        rows.append([file.stem, x_fixed, y_fixed, "TRUE" if was_fixed else "FALSE"])

    # 7. Save to CSV
    save_coordinates_to_csv(rows, output_csv)

    # 5. Print summary
    print("Coordinate check completed.")
    print(f"Total files with valid coordinates: {sucess_count}")
    print(f"Total files moved to 'no_coords' folder: {fail_count}")
    print(f"Total files written to CSV: {len(rows)}")


if __name__ == "__main__":

    ### PATHS AND SETTINGS ########################################################
    GEF_FOLDER = Path(r"C:\VOW\data\test_cpts")
    OUTPUT_CSV = Path(r"C:\VOW\data\test_outputs")
    CSV_NAME = "coordinates_cpts_test_result.csv"
    OUTPUT_CSV = OUTPUT_CSV / CSV_NAME
    ###########################################################################

    # Run the extraction
    process_cpt_coords(GEF_FOLDER, OUTPUT_CSV)
