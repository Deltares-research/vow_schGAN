"""Mask SchemaGAN output above the ground surface.

A cell is kept when its CENTRE lies at or below the ground surface at the
column centre (design decision; easy to change in `surface_mask`). Masked
cells become NaN.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_frame(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ground_at_columns(profile_csv: Path, frame: dict) -> np.ndarray:
    """Ground elevation (m NAP) at each column centre, interpolated linearly along the line."""
    p = pd.read_csv(profile_csv, sep=None, engine="python")
    p.columns = [c.strip().lower() for c in p.columns]
    p = p.dropna(subset=["dist", "elev_nap"])
    centres = frame["left_m"] + (np.arange(frame["n_cols"]) + 0.5) * frame["col_width_m"]
    return np.interp(centres, p["dist"], p["elev_nap"])


def surface_mask(frame: dict, ground: np.ndarray) -> np.ndarray:
    """Boolean (n_rows x n_cols): True where the cell centre is at or below the ground (cell kept)."""
    row_centres = frame["top_m_nap"] - (np.arange(frame["n_rows"]) + 0.5) * frame["row_height_m"]
    return row_centres[:, None] <= ground[None, :]


def apply_mask(matrix: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = matrix.astype(float).copy()
    out[~keep] = np.nan
    return out


def write_matrix(matrix: np.ndarray, path: Path):
    """Same plain format as the input matrix (no header, no index); NaN (masked) written as empty fields."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in matrix:
            f.write(",".join("" if np.isnan(v) else f"{v:.4f}" for v in row) + "\n")
