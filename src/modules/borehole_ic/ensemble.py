"""Ensemble utilities: repeat `build_synthetic_profile` across independent random streams.

Only the random stream varies across realizations - borehole layers, depth
grid, mapping, sampling/transition parameters, and the unsupported-interval
mask are identical every time (they are deterministic functions of the
borehole and the fixed parameters, never of the RNG).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.composition_sampling import CORRELATION_LENGTH_M, RANDOM_SEED  # noqa: E402
from modules.borehole_ic.profile_builder import DEPTH_STEP_M, TRANSITION_THICKNESS_M, build_synthetic_profile  # noqa: E402


def derive_realization_seeds(base_seed: int, n_realizations: int) -> List[int]:
    """Derive `n_realizations` independent, reproducible integer seeds from `base_seed`.

    Uses `numpy.random.SeedSequence.spawn`, the standard way to produce
    statistically independent child streams from one base seed (rather than
    e.g. `base_seed + i`, which does not guarantee independence).
    """
    children = np.random.SeedSequence(base_seed).spawn(n_realizations)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def build_ensemble(
    borehole: dict,
    n_realizations: int = 20,
    base_seed: int = RANDOM_SEED,
    depth_step_m: float = DEPTH_STEP_M,
    transition_thickness_m: float = TRANSITION_THICKNESS_M,
    correlation_length_m: float = CORRELATION_LENGTH_M,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], dict]:
    """Build `n_realizations` independent synthetic Ic realizations for one borehole.

    Returns:
        (layers_df, ensemble_df, seeds, run_info)
        `layers_df`: the (seed-independent) resolved layer table.
        `ensemble_df`: one row per depth grid point, columns
            depth_m, elevation_m_nap, is_transition, source_layer_index,
            source_class_code, ic_realization_0 .. ic_realization_{n-1},
            ic_median, ic_p05, ic_p95.
        `seeds`: the exact per-realization seeds used (for the report).
        `run_info`: the (seed-independent) `run_info` from the first build,
            for documenting segments/groups/transition thicknesses once.
    """
    seeds = derive_realization_seeds(base_seed, n_realizations)

    layers_df = None
    run_info = None
    ic_columns = {}
    base_columns = None
    for r, seed in enumerate(seeds):
        l_df, s_df, r_info = build_synthetic_profile(
            borehole, depth_step_m=depth_step_m, transition_thickness_m=transition_thickness_m,
            correlation_length_m=correlation_length_m, random_seed=seed,
        )
        if layers_df is None:
            layers_df = l_df
            run_info = r_info
            base_columns = s_df[["depth_m", "elevation_m_nap", "is_transition", "source_layer_index",
                                 "source_class_code", "is_interpolated"]]
        ic_columns[f"ic_realization_{r}"] = s_df["synthetic_ic"].to_numpy()

    ensemble_df = base_columns.copy()
    for name, values in ic_columns.items():
        ensemble_df[name] = values

    ic_matrix = ensemble_df[[f"ic_realization_{r}" for r in range(n_realizations)]].to_numpy()
    ensemble_df["ic_median"] = np.nanmedian(ic_matrix, axis=1)
    ensemble_df["ic_p05"] = np.nanpercentile(ic_matrix, 5, axis=1)
    ensemble_df["ic_p95"] = np.nanpercentile(ic_matrix, 95, axis=1)

    run_info = dict(run_info)
    run_info["n_realizations"] = n_realizations
    run_info["base_seed"] = base_seed
    run_info["realization_seeds"] = seeds
    run_info["seed_derivation"] = "numpy.random.SeedSequence(base_seed).spawn(n_realizations), each -> generate_state(1, uint32)"
    return layers_df, ensemble_df, seeds, run_info


def align_reference_to_grid(
    reference_df: pd.DataFrame, depth_grid: np.ndarray, max_gap_factor: float = 3.0,
) -> np.ndarray:
    """Linearly interpolate the reference CPT's Ic onto `depth_grid`, never extrapolating or bridging gaps.

    Args:
        reference_df: must have "depth_m" and "reference_ic_geolib_plus".
        depth_grid: target depths (the borehole/ensemble grid).
        max_gap_factor: a target depth is left as NaN (not bridged) if its
            two bracketing CPT samples are farther apart than
            `max_gap_factor` times the CPT's own median sample spacing.

    Returns:
        np.ndarray, same length as `depth_grid`, NaN outside the CPT's
        measured range or across any unexpectedly large internal gap.
    """
    depths = reference_df["depth_m"].to_numpy()
    ic = reference_df["reference_ic_geolib_plus"].to_numpy()
    order = np.argsort(depths)
    depths, ic = depths[order], ic[order]
    median_step = np.median(np.diff(depths))
    max_gap = max_gap_factor * median_step

    aligned = np.full(len(depth_grid), np.nan)
    idx = np.searchsorted(depths, depth_grid)
    for i, z in enumerate(depth_grid):
        j = idx[i]
        if j <= 0 or j >= len(depths):
            continue  # would require extrapolation - leave as gap
        d0, d1 = depths[j - 1], depths[j]
        if (d1 - d0) > max_gap:
            continue  # larger gap than the CPT's own sampling - do not bridge
        t = (z - d0) / (d1 - d0) if d1 > d0 else 0.0
        aligned[i] = ic[j - 1] * (1 - t) + ic[j] * t
    return aligned


def summarize_comparison(ensemble_df: pd.DataFrame, aligned_reference_ic: np.ndarray) -> pd.DataFrame:
    """Descriptive (not validation) comparison between the ensemble band and the aligned reference CPT.

    Returns one row per group: "overall", "transition", and one row per
    distinct non-transition `source_class_code`.
    """
    df = ensemble_df.copy()
    df["reference_ic"] = aligned_reference_ic
    comparable = df.dropna(subset=["ic_median", "reference_ic"])

    def _stats(subset: pd.DataFrame, label: str) -> Optional[dict]:
        if not len(subset):
            return None
        band_width = subset["ic_p95"] - subset["ic_p05"]
        signed_diff = subset["ic_median"] - subset["reference_ic"]
        inside = (subset["reference_ic"] >= subset["ic_p05"]) & (subset["reference_ic"] <= subset["ic_p95"])
        return {
            "group": label,
            "n_comparable_samples": int(len(subset)),
            "median_band_width": float(band_width.median()),
            "median_signed_diff_synthetic_minus_reference": float(signed_diff.median()),
            "fraction_reference_inside_band": float(inside.mean()),
        }

    rows = []
    overall = _stats(comparable, "overall")
    if overall:
        rows.append(overall)
    transition_rows = _stats(comparable[comparable["is_transition"]], "transition (all classes)")
    if transition_rows:
        rows.append(transition_rows)
    non_transition = comparable[~comparable["is_transition"]]
    for class_code in sorted(non_transition["source_class_code"].dropna().unique()):
        class_rows = _stats(non_transition[non_transition["source_class_code"] == class_code], class_code)
        if class_rows:
            rows.append(class_rows)
    return pd.DataFrame(rows)
