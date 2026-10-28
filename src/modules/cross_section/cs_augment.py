"""Augment a cross-section with copies of existing tests at fixed positions near the line ends.

SchemaGAN works best with about 6 occupied columns. The real data often
leaves the ends of the line empty, so copies are placed at fixed distances
(the same for every cross-section):

    left:  5 m from the start (A)
    right: 10 m and 50 m from the end (B)

Rules (design decisions):
- Each copy duplicates the test NEAREST to its position. A real CPT is
  preferred only as a tie-break: among tests within one SchemaGAN column width
  of the nearest one, a real CPT wins over a borehole.
- A copy keeps the original's NAP elevations (no shift to the local ground).
- Only real tests are copied, never other copies.
- Copies are kept in the tables with `augmented=True` and `copy_of=<original>`.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

AUGMENT_LEFT_FROM_START_M = (5.0,)
AUGMENT_RIGHT_FROM_END_M = (10.0, 50.0)


def augmentation_positions(line_length_m: float) -> List[Tuple[str, float]]:
    positions = [(f"left +{d:g} m", d) for d in AUGMENT_LEFT_FROM_START_M]
    positions += [(f"right -{d:g} m", line_length_m - d) for d in AUGMENT_RIGHT_FROM_END_M]
    return [(label, x) for label, x in positions if 0.0 <= x <= line_length_m]


def choose_source(tests: pd.DataFrame, position_m: float, tie_tolerance_m: float) -> pd.Series:
    """Nearest real test to `position_m`; a CPT wins over a borehole within `tie_tolerance_m` of the nearest."""
    distance = (tests["distance_along_m"] - position_m).abs()
    near = tests[distance <= distance.min() + tie_tolerance_m].assign(_dist=distance)
    near = near.assign(_cpt_first=(near["kind"] != "cpt").astype(int)).sort_values(["_cpt_first", "_dist"])
    return near.iloc[0]


def augment_cross_section(
    tests: pd.DataFrame, ic_profiles: pd.DataFrame, line, column_width_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add copies at the augmentation positions. Returns (tests, ic_profiles) including the copies, renumbered."""
    tests = tests.copy()
    tests["augmented"] = False
    tests["copy_of"] = None
    tests["copy_label"] = None
    real = tests[tests["status"] == "ok"]
    if not len(real):
        return tests, ic_profiles

    new_rows, new_profiles = [], []
    for label, position in augmentation_positions(line.length_m):
        src = choose_source(real, position, column_width_m)
        copy_name = f"{src['name']}@{position:.1f}m"
        row = src.drop(labels=[c for c in ("_dist", "_cpt_first") if c in src]).copy()
        overrides = {
            "name": copy_name, "distance_along_m": position, "offset_m": np.nan, "beyond_line_ends": False,
            "profile_ground_m_nap": line.ground_at(position), "x": np.nan, "y": np.nan,
            "augmented": True, "copy_of": src["name"], "copy_label": label,
            "note": f"augmented copy of {src['name']} (from {src['distance_along_m']:.1f} m), NAP elevations kept",
        }
        for key, value in overrides.items():  # Series.update skips NaN, so assign explicitly
            row[key] = value
        new_rows.append(row)
        prof = ic_profiles[ic_profiles["name"] == src["name"]].copy()
        prof["name"] = copy_name
        prof["distance_along_m"] = position
        new_profiles.append(prof)

    tests = pd.concat([tests, pd.DataFrame(new_rows)], ignore_index=True)
    tests = tests.sort_values("distance_along_m", na_position="last").reset_index(drop=True)
    tests["number"] = range(1, len(tests) + 1)
    ic_profiles = pd.concat([ic_profiles, *new_profiles], ignore_index=True)
    if "augmented" not in ic_profiles:
        ic_profiles["augmented"] = False
    ic_profiles["augmented"] = ic_profiles["name"].isin(tests.loc[tests["augmented"], "name"])
    return tests, ic_profiles


def occupied_columns(tests: pd.DataFrame, frame) -> int:
    """Number of distinct SchemaGAN columns holding at least one drawn test (copies included)."""
    drawn = tests[tests["status"] == "ok"]
    cols = np.clip(((drawn["distance_along_m"] - frame.left_m) / frame.col_width_m).astype(int), 0, frame.n_cols - 1)
    return int(cols.nunique())
