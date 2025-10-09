# Simple code to extract the coordinates from .GEF files in a folder and save them
# to a CSV file.

import sys
from pathlib import Path

# Add your local GEOLib-Plus path
sys.path.append(r"D:\GEOLib-Plus")

import csv
import numpy as np
from pathlib import Path
import logging

import sys

sys.path.append("C:\schemaGAN")
import schemaGAN


from utils import read_files
from geolib_plus.gef_cpt import GefCpt


# to avoid warnings coming from GeoLib
initial_logging_level = logging.getLogger().getEffectiveLevel()
logging.disable(logging.ERROR)
# remove warning messages
logging.getLogger().setLevel(logging.ERROR)


def extract_coords(gef_folder: Path, output_csv: Path) -> None:
    """
    Extract coordinates from .GEF files in a specified folder and save them to a CSV file.
    """
    # Read .GEF files from the specified folder
    gef_files = read_files(str(gef_folder), extension=".gef")

    # Extract coordinates
    coords = []
    for file in gef_files:
        cpt = GefCpt()
        cpt.read(str(file))
        coords.append(
            {"name": file.stem, "x": cpt.coordinates[0], "y": cpt.coordinates[1]}
        )

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "x", "y"])
        for r in coords:
            w.writerow([r["name"], r["x"], r["y"]])
    print(f"Coordinates saved to {output_csv}")


if __name__ == "__main__":

    ### PATHS AND SETTINGS ###
    GEF_FOLDER = Path(r"C:\VOW\data\cpts\betuwepand\dike_south_BRO")
    OUTPUT_CSV = Path(r"C:\VOW\gis\coords")
    CSV_NAME = "TESTEST.csv"
    ##########################

    OUTPUT_CSV = OUTPUT_CSV / CSV_NAME

    # Run the extraction
    extract_coords(GEF_FOLDER, OUTPUT_CSV)
