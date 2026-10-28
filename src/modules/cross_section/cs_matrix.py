"""Convert a cross-section into the 512 x 32 SchemaGAN input matrix.

The matrix covers the SchemaGAN frame (`cs_data.SchemaganFrame`): columns run
from the start to the end of the profile line, rows from the frame top (row 0)
down to the design bottom (row 31). Cells hold Ic where a test provides data
and 0 everywhere else (0 = unknown, to be filled by SchemaGAN).

Rules (design decisions):
- Column: each test goes into column floor((distance_along - left) / column_width),
  clipped to 0..n_cols-1. Distances are not stretched or shifted.
- Same column: one test per column, priority real CPT > real borehole > copy
  (copy of a CPT before copy of a borehole), then the test nearest the column
  centre. The others are reported as dropped.
- Row value: median of the test's valid Ic samples whose elevation falls in the
  row (upper bound inclusive, lower bound exclusive), but only if the test's
  valid data covers at least half of the row height; otherwise 0. Each sample
  stands for the interval halfway to its neighbours; missing Ic values
  (e.g. unsupported borehole layers) count as not covered.
- Data above the frame top or below the frame bottom is not used.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_ROW_COVERAGE = 0.5
ROW_STATISTIC = "median"


def column_index(distance_along_m: float, frame) -> int:
    return int(np.clip(np.floor((distance_along_m - frame.left_m) / frame.col_width_m), 0, frame.n_cols - 1))


def row_edges(frame) -> np.ndarray:
    """Elevations of the n_rows + 1 row boundaries, from the frame top down to the frame bottom."""
    return frame.top_m_nap - np.arange(frame.n_rows + 1) * frame.row_height_m


def _sample_cells(elevations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(top, bottom) of the interval each sample represents: halfway to its neighbours (samples sorted top-down)."""
    if len(elevations) == 1:
        return elevations + 0.01, elevations - 0.01
    mid = (elevations[:-1] + elevations[1:]) / 2.0
    tops = np.r_[elevations[0] + (elevations[0] - mid[0]), mid]
    bottoms = np.r_[mid, elevations[-1] - (mid[-1] - elevations[-1])]
    return tops, bottoms


def column_values(profile: pd.DataFrame, frame) -> Tuple[np.ndarray, np.ndarray]:
    """Row values (0 where not filled) and row coverage fractions for one test profile."""
    prof = profile.dropna(subset=["elevation_m_nap"]).sort_values("elevation_m_nap", ascending=False)
    prof = prof.drop_duplicates("elevation_m_nap")
    values = np.zeros(frame.n_rows)
    coverage = np.zeros(frame.n_rows)
    if not len(prof):
        return values, coverage

    elev = prof["elevation_m_nap"].to_numpy(dtype=float)
    ic = prof["ic"].to_numpy(dtype=float)
    valid = np.isfinite(ic)
    cell_top, cell_bottom = _sample_cells(elev)
    edges = row_edges(frame)
    row_of_sample = np.floor((frame.top_m_nap - elev) / frame.row_height_m).astype(int)

    for i in range(frame.n_rows):
        upper, lower = edges[i], edges[i + 1]
        overlap = np.clip(np.minimum(cell_top, upper) - np.maximum(cell_bottom, lower), 0.0, None)
        coverage[i] = float(overlap[valid].sum()) / frame.row_height_m
        in_row = valid & (row_of_sample == i)
        if coverage[i] >= MIN_ROW_COVERAGE and in_row.any():
            values[i] = float(np.median(ic[in_row]))
    return values, np.minimum(coverage, 1.0)


def _priority(test) -> Tuple[int, int]:
    if not test.augmented:
        return (0 if test.kind == "cpt" else 1, 0)
    return (2, 0 if test.kind == "cpt" else 1)


def build_matrix(tests: pd.DataFrame, ic_profiles: pd.DataFrame, frame) -> Tuple[np.ndarray, pd.DataFrame]:
    """The (n_rows x n_cols) input matrix and a table describing which test fills which column."""
    matrix = np.zeros((frame.n_rows, frame.n_cols))
    drawn = tests[tests["status"] == "ok"].copy()
    if "augmented" not in drawn:
        drawn["augmented"] = False
    if "copy_of" not in drawn:
        drawn["copy_of"] = None
    drawn["column"] = [column_index(d, frame) for d in drawn["distance_along_m"]]
    drawn["column_centre_m"] = frame.left_m + (drawn["column"] + 0.5) * frame.col_width_m

    rows: List[Dict[str, object]] = []
    for col, group in drawn.groupby("column"):
        ranked = sorted(group.itertuples(), key=lambda t: (*_priority(t), abs(t.distance_along_m - t.column_centre_m)))
        winner = ranked[0]
        values, coverage = column_values(ic_profiles[ic_profiles["name"] == winner.name], frame)
        matrix[:, col] = values
        filled = np.flatnonzero(values)
        for t in ranked:
            rows.append({
                "number": t.number, "name": t.name, "kind": t.kind, "augmented": bool(t.augmented),
                "copy_of": t.copy_of if t.augmented else None,
                "distance_along_m": round(float(t.distance_along_m), 3), "column": int(col),
                "used": t is winner,
                "dropped_reason": None if t is winner else f"same column as {winner.name} (lower priority)",
                "first_row": int(filled[0]) if t is winner and len(filled) else None,
                "last_row": int(filled[-1]) if t is winner and len(filled) else None,
                "n_rows_filled": int(len(filled)) if t is winner else 0,
                "first_row_coverage": round(float(coverage[filled[0]]), 2) if t is winner and len(filled) else None,
                "last_row_coverage": round(float(coverage[filled[-1]]), 2) if t is winner and len(filled) else None,
            })
    assignments = pd.DataFrame(rows).sort_values(["column", "used"], ascending=[True, False]).reset_index(drop=True)
    return matrix, assignments


def write_matrix_csv(matrix: np.ndarray, path: Path):
    """Plain matrix: n_rows lines of n_cols comma-separated values, no header, no index; zeros written as 0."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in matrix:
            f.write(",".join("0" if v == 0 else f"{v:.4f}" for v in row) + "\n")


def write_metadata(path: Path, cs_label: str, frame, assignments: pd.DataFrame, matrix: np.ndarray, matrix_file: str):
    edges = row_edges(frame)
    used = assignments[assignments["used"]]
    lines = [
        f"SchemaGAN input matrix for cross-section {cs_label}",
        f"Matrix file: {matrix_file}  ({frame.n_rows} rows x {frame.n_cols} columns, no header, no index)",
        "Cell values: Ic where a test provides data, 0 elsewhere (0 = unknown).",
        "",
        "Frame",
        f"  columns: distance along the line {frame.left_m:.3f} - {frame.right_m:.3f} m, "
        f"column width {frame.col_width_m:.4f} m; column j covers [left + j*width, left + (j+1)*width)",
        f"  rows:    elevation {frame.top_m_nap:.3f} m NAP (top of row 0, highest test ground: {frame.top_source}) "
        f"to {frame.bottom_m_nap:.3f} m NAP (bottom of row {frame.n_rows - 1}), row height {frame.row_height_m:.4f} m; "
        "row i covers (top - (i+1)*height, top - i*height]",
        "",
        "Rules",
        "  column = floor((distance along line - left) / column width)",
        f"  row value = {ROW_STATISTIC} of the test's Ic samples in the row, only if the test's valid data covers "
        f">= {MIN_ROW_COVERAGE:.0%} of the row height; otherwise 0",
        "  one test per column: real CPT > real borehole > augmented copy",
        "",
        f"Columns with data: {len(used)}; non-zero cells: {int(np.count_nonzero(matrix))}; "
        f"Ic range in matrix: {matrix[matrix > 0].min():.3f} - {matrix.max():.3f}" if np.count_nonzero(matrix) else
        "Columns with data: 0",
        "",
        "Column assignments",
    ]
    table = assignments.copy()
    table["copy_of"] = table["copy_of"].fillna("")
    table["dropped_reason"] = table["dropped_reason"].fillna("")
    lines.append(table.to_string(index=False))
    lines += ["", "Row elevations (m NAP)"]
    lines += [f"  row {i:2d}: {edges[i]:7.3f} to {edges[i + 1]:7.3f}" for i in range(frame.n_rows)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
