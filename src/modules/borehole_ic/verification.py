"""Lightweight (non-pytest) verification checks for the synthetic Ic profile.

Each check returns (passed: bool, detail: str). This is a prototype: plain
assertions/comparisons in a loop, not a test framework.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.triangles import TRIANGLES, all_composition_columns, class_polygon  # noqa: E402

Check = Tuple[str, bool, str]


def check_layer_intervals(layers_df: pd.DataFrame) -> Check:
    bad = layers_df[layers_df["thickness_m"] <= 0]
    if len(bad):
        return "valid_layer_intervals", False, f"{len(bad)} layer(s) with non-positive thickness: {bad['layer_index'].tolist()}"
    gaps = []
    for i in range(len(layers_df) - 1):
        lower = layers_df.iloc[i]["lower_boundary_m"]
        upper_next = layers_df.iloc[i + 1]["upper_boundary_m"]
        if abs(lower - upper_next) > 1e-6:
            gaps.append((i, i + 1, lower, upper_next))
    if gaps:
        return "valid_layer_intervals", False, f"gap/overlap between consecutive layers: {gaps}"
    return "valid_layer_intervals", True, f"{len(layers_df)} layers, all positive thickness, contiguous boundaries"


def check_vertical_alignment(samples_df: pd.DataFrame, surface_elevation_m_nap: float) -> Check:
    expected = surface_elevation_m_nap - samples_df["depth_m"]
    diff = (expected - samples_df["elevation_m_nap"]).abs().max()
    ok = diff < 1e-6
    return "vertical_alignment", ok, f"max |expected - stored| elevation = {diff:.2e} m NAP"


def check_composition_sums(samples_df: pd.DataFrame) -> Check:
    """Each triangle's composition columns sum to 100 wherever that triangle was sampled."""
    n_checked, max_dev = 0, 0.0
    for spec in TRIANGLES.values():
        cols = [c for c in spec.columns if c in samples_df]
        if len(cols) != 3:
            continue
        valid = samples_df.dropna(subset=cols)
        if len(valid):
            max_dev = max(max_dev, float((valid[cols].sum(axis=1) - 100.0).abs().max()))
            n_checked += len(valid)
    ok = max_dev < 1e-6
    return "composition_sums_to_100", ok, f"max |sum - 100| over {n_checked} samples = {max_dev:.2e}"


def check_polygon_containment(samples_df: pd.DataFrame, layers_df: pd.DataFrame) -> Check:
    from shapely.geometry import Point

    non_transition = samples_df[(~samples_df["is_transition"]) & samples_df["x"].notna()]
    violations = 0
    for row in non_transition.itertuples():
        class_code = layers_df.iloc[row.source_layer_index]["class_code"]
        poly = class_polygon(class_code)
        if not poly.contains(Point(row.x, row.y)) and poly.distance(Point(row.x, row.y)) > 1e-6:
            violations += 1
    ok = violations == 0
    return "polygon_containment_outside_transitions", ok, f"{violations} / {len(non_transition)} samples outside their own source polygon"


def check_ic_agreement(samples_df: pd.DataFrame) -> Check:
    """Stored Ic equals a direct evaluation of the sample's own triangle surface (rows with a composition)."""
    valid = samples_df.dropna(subset=["x", "y", "synthetic_ic"])
    max_dev = 0.0
    for row in valid.itertuples():
        recomputed = TRIANGLES[row.triangle].surface.value_at_cartesian(row.x, row.y)
        dev = abs(recomputed - row.synthetic_ic) if recomputed is not None else np.inf
        max_dev = max(max_dev, dev)
    ok = max_dev < 1e-9
    return "ic_agrees_with_direct_evaluation", ok, f"max |recomputed - stored| Ic over {len(valid)} samples = {max_dev:.2e}"


def _max_abs_diff_nan_aware(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a - b|; matching NaNs count as equal, a NaN on only one side counts as a mismatch."""
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    if np.any(a_nan != b_nan):
        return np.inf
    both = ~a_nan
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def check_reproducibility(build_fn: Callable[[], pd.DataFrame]) -> Check:
    a = build_fn()
    b = build_fn()
    numeric_cols = ["x", "y", *all_composition_columns(), "synthetic_ic"]
    max_dev = max(
        _max_abs_diff_nan_aware(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)) for c in numeric_cols
    )
    ok = max_dev < 1e-12
    return "reproducibility_same_seed", ok, f"max abs diff across two runs with the same seed = {max_dev:.2e}"


def check_finite_in_supported(samples_df: pd.DataFrame, layers_df: pd.DataFrame) -> Check:
    supported_rows = samples_df[~samples_df["source_layer_index"].isna()]
    non_transition_supported = supported_rows[
        (~supported_rows["is_transition"])
        & supported_rows["source_layer_index"].apply(lambda i: bool(layers_df.iloc[int(i)]["supported"]))
    ]
    n_nan = non_transition_supported["synthetic_ic"].isna().sum()
    n_nan += samples_df[samples_df["is_transition"]]["synthetic_ic"].isna().sum()
    ok = n_nan == 0
    return "finite_ic_in_supported_intervals", ok, f"{n_nan} NaN Ic value(s) found where a class was supported"


def check_unsupported_gaps_preserved(samples_df: pd.DataFrame, layers_df: pd.DataFrame) -> Check:
    unsupported_layers = layers_df[~layers_df["supported"]]
    missing = []
    for layer in unsupported_layers.itertuples():
        rows = samples_df[
            (~samples_df["is_transition"])
            & (samples_df["source_layer_index"] == layer.layer_index)
        ]
        if not len(rows) or rows["synthetic_ic"].notna().any():
            missing.append(layer.layer_index)
    ok = len(missing) == 0
    return (
        "unsupported_layers_preserved_as_gaps", ok,
        f"{len(unsupported_layers)} unsupported layer(s); {len(missing)} not cleanly preserved as NaN gaps",
    )


def run_all_checks(
    layers_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    surface_elevation_m_nap: float,
    build_fn: Callable[[], pd.DataFrame],
) -> List[Check]:
    return [
        check_layer_intervals(layers_df),
        check_vertical_alignment(samples_df, surface_elevation_m_nap),
        check_composition_sums(samples_df),
        check_polygon_containment(samples_df, layers_df),
        check_ic_agreement(samples_df),
        check_reproducibility(build_fn),
        check_finite_in_supported(samples_df, layers_df),
        check_unsupported_gaps_preserved(samples_df, layers_df),
    ]


def format_report(checks: List[Check]) -> str:
    lines = ["Verification report", "=" * 60]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    lines.append("=" * 60)
    lines.append(f"{n_pass}/{len(checks)} checks passed")
    return "\n".join(lines)
