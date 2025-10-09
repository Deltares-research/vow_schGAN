import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import logging
import csv
import pickle
from shapely.geometry import Point

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import (
    UnitWeightMethod,
    ShearWaveVelocityMethod,
    OCRMethod,
)


# to avoid warnings coming from GeoLib
initial_logging_level = logging.getLogger().getEffectiveLevel()
logging.disable(logging.ERROR)


import shutil


def read_cpt_files(path: str, extension: str = ".gef"):
    """
    Try reading all CPT files in a directory.
    Returns two lists: successfully read CPT objects and failed filenames.
    """
    extension = extension.lower()
    cpt_files = [
        Path(path, c)
        for c in os.listdir(path)
        if Path(path, c).is_file() and c.lower().endswith(extension)
    ]

    successes = []
    failures = []

    for cpt_file in cpt_files:
        try:
            cpt = GefCpt()
            cpt.read(cpt_file)
            successes.append((cpt_file, cpt))
        except:
            modified_filename = (
                str(cpt_file)
                .replace("_IMBRO", "")
                .replace("_A", "")
                .replace(".gef", ".xml")
            )
            try:
                cpt = GefCpt()
                cpt.read(modified_filename)
                successes.append((cpt_file, cpt))
            except:
                failures.append(cpt_file)

    return successes, failures


def sort_cpt_files(base_folder, successes, failures):
    """
    Create 'success' and 'failure' subfolders and move files accordingly.
    """
    success_folder = Path(base_folder, "success")
    failure_folder = Path(base_folder, "failure")

    success_folder.mkdir(exist_ok=True)
    failure_folder.mkdir(exist_ok=True)

    # Move success files
    for file, _ in successes:
        shutil.move(str(file), success_folder / file.name)

    # Move failure files
    for file in failures:
        shutil.move(str(file), failure_folder / file.name)

    print(f"Moved {len(successes)} files to {success_folder}")
    print(f"Moved {len(failures)} files to {failure_folder}")


# Define the Path to the folder containing the ALL CPT files
cpt_folder = Path(r"C:\ark\data\sonderingen\SON\failure\failure")
# cpt_folder = Path(r"C:\ark\data\sonderingen\SON")


successes, failures = read_cpt_files(cpt_folder)
sort_cpt_files(cpt_folder, successes, failures)
