"""Evaluate synthetic (borehole-derived) Ic against a nearby measured CPT.

For one borehole-CPT pair this builds the synthetic Ic ensemble from the
borehole, interprets the CPT with geolib_plus, aligns the measured Ic onto
the borehole's depth grid by ELEVATION (m NAP, so different ground levels are
handled), and computes descriptive accuracy metrics. Helpers then pool the
results of many pairs.

Conventions:
- error = synthetic ensemble median Ic - measured CPT Ic (positive = predicted too high / too fine).
- Only depths where both the synthetic median and the aligned measured Ic are
  finite are compared; the measured Ic is never extrapolated or bridged across
  CPT gaps (`ensemble.align_reference_to_grid`).
- Each compared sample is tagged "exact" (exact NEN class), "interpolated"
  (unsupported layer, Ic interpolated from its neighbours), "guessed"
  (generic class such as "Zand*") or "transition" (blend zone between layers).
- Robertson zones use Ic thresholds 1.31 / 2.05 / 2.60 / 2.95 / 3.60; a value
  exactly on a threshold belongs to the finer-grained (higher-Ic) zone.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.ensemble import align_reference_to_grid, build_ensemble  # noqa: E402

ROBERTSON_THRESHOLDS = (1.31, 2.05, 2.60, 2.95, 3.60)
SAMPLE_CATEGORIES = ("exact", "guessed", "interpolated", "transition")
ROBERTSON_ZONE_NAMES = {
    7: "7 gravelly/dense sand",
    6: "6 sand",
    5: "5 sand mixtures",
    4: "4 silt mixtures",
    3: "3 clay",
    2: "2 organic/peat",
}


def robertson_zone(ic) -> np.ndarray:
    """Robertson soil-behaviour zone (7 = coarsest ... 2 = organic) per Ic value; 0 where Ic is NaN."""
    ic = np.asarray(ic, dtype=float)
    zone = 7 - np.digitize(ic, ROBERTSON_THRESHOLDS, right=False)
    return np.where(np.isnan(ic), 0, zone)


def align_reference_by_elevation(reference_df: pd.DataFrame, grid_elevation_m_nap) -> np.ndarray:
    """Measured Ic interpolated onto grid elevations (m NAP); NaN outside the CPT or across its gaps."""
    as_depth = pd.DataFrame({
        "depth_m": -reference_df["elevation_m_nap"].to_numpy(dtype=float),
        "reference_ic_geolib_plus": reference_df["reference_ic_geolib_plus"].to_numpy(dtype=float),
    }).dropna(subset=["depth_m"])
    return align_reference_to_grid(as_depth, -np.asarray(grid_elevation_m_nap, dtype=float))


def compute_metrics(samples: pd.DataFrame) -> Dict[str, float]:
    """Accuracy metrics for compared samples (columns ic_median, ic_p05, ic_p95, reference_ic)."""
    n = len(samples)
    if n == 0:
        return {"n_samples": 0}
    err = samples["ic_median"] - samples["reference_ic"]
    inside = (samples["reference_ic"] >= samples["ic_p05"]) & (samples["reference_ic"] <= samples["ic_p95"])
    same_zone = robertson_zone(samples["ic_median"]) == robertson_zone(samples["reference_ic"])
    r = samples["ic_median"].corr(samples["reference_ic"]) if n > 2 else np.nan
    return {
        "n_samples": int(n),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_abs_error": float(err.abs().median()),
        "pearson_r": float(r) if pd.notna(r) else np.nan,
        "fraction_inside_band": float(inside.mean()),
        "robertson_zone_agreement": float(same_zone.mean()),
        "mean_band_width": float((samples["ic_p95"] - samples["ic_p05"]).mean()),
    }


def evaluate_pair(
    borehole: dict, reference_df: pd.DataFrame, n_realizations: int = 20, base_seed: int = 42,
) -> Dict[str, object]:
    """Build the ensemble for one borehole and compare it with the measured CPT profile.

    Returns a dict with:
        layers_df, ensemble_df (adds reference_ic, error, sample_category),
        samples (compared rows only), layer_table (per borehole layer means),
        metrics (overall + per category), info (overlap interval, elevations).
    """
    layers_df, ensemble_df, seeds, run_info = build_ensemble(
        borehole, n_realizations=n_realizations, base_seed=base_seed,
    )
    ensemble_df["reference_ic"] = align_reference_by_elevation(reference_df, ensemble_df["elevation_m_nap"])
    ensemble_df["error"] = ensemble_df["ic_median"] - ensemble_df["reference_ic"]

    is_guess = layers_df["is_guess"].to_numpy(dtype=bool)
    layer_idx = ensemble_df["source_layer_index"].astype(int).to_numpy()
    ensemble_df["sample_category"] = np.where(
        ensemble_df["is_transition"], "transition",
        np.where(ensemble_df["is_interpolated"], "interpolated", np.where(is_guess[layer_idx], "guessed", "exact")),
    )
    ensemble_df["zone_synthetic"] = robertson_zone(ensemble_df["ic_median"])
    ensemble_df["zone_measured"] = robertson_zone(ensemble_df["reference_ic"])

    samples = ensemble_df.dropna(subset=["ic_median", "reference_ic"]).copy()

    metrics = {"overall": compute_metrics(samples)}
    for category in SAMPLE_CATEGORIES:
        metrics[category] = compute_metrics(samples[samples["sample_category"] == category])

    in_layer = samples[~samples["is_transition"]]
    layer_rows = []
    for idx, grp in in_layer.groupby("source_layer_index"):
        layer = layers_df.iloc[int(idx)]
        layer_rows.append({
            "layer_index": int(idx),
            "soil_name_nen5104": layer["soil_name_nen5104"],
            "class_code": layer["class_code"],
            "is_guess": bool(layer["is_guess"]),
            "top_elevation_m_nap": float(grp["elevation_m_nap"].max()),
            "bottom_elevation_m_nap": float(grp["elevation_m_nap"].min()),
            "n_samples": int(len(grp)),
            "measured_ic_mean": float(grp["reference_ic"].mean()),
            "measured_ic_std": float(grp["reference_ic"].std(ddof=0)),
            "synthetic_ic_mean": float(grp["ic_median"].mean()),
            "error_mean": float(grp["error"].mean()),
        })

    cpt_elev = reference_df["elevation_m_nap"].dropna()
    info = {
        "borehole_surface_elevation_m_nap": borehole.get("surface_elevation_m_nap"),
        "cpt_top_elevation_m_nap": float(cpt_elev.max()) if len(cpt_elev) else np.nan,
        "cpt_bottom_elevation_m_nap": float(cpt_elev.min()) if len(cpt_elev) else np.nan,
        "borehole_bottom_elevation_m_nap": float(ensemble_df["elevation_m_nap"].min()),
        "overlap_top_m_nap": float(samples["elevation_m_nap"].max()) if len(samples) else np.nan,
        "overlap_bottom_m_nap": float(samples["elevation_m_nap"].min()) if len(samples) else np.nan,
        "realization_seeds": seeds,
    }
    return {
        "layers_df": layers_df, "ensemble_df": ensemble_df, "samples": samples,
        "layer_table": pd.DataFrame(layer_rows), "metrics": metrics, "info": info,
    }


def flatten_pair_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """One flat row: overall metrics unprefixed, per-category metrics prefixed (e.g. exact_mae)."""
    row = dict(metrics["overall"])
    for category in SAMPLE_CATEGORIES:
        for key in ("n_samples", "bias", "mae", "fraction_inside_band"):
            row[f"{category}_{key}"] = metrics[category].get(key, np.nan)
    return row


def class_metrics(pooled: pd.DataFrame) -> pd.DataFrame:
    """Metrics per source class over all pairs (transition samples excluded)."""
    rows = []
    in_layer = pooled[~pooled["is_transition"]]
    for code, grp in in_layer.groupby("source_class_code"):
        rows.append({"class_code": code, "n_pairs": int(grp["pair_id"].nunique()), **compute_metrics(grp)})
    return pd.DataFrame(rows).sort_values("class_code").reset_index(drop=True) if rows else pd.DataFrame()


def zone_confusion(pooled: pd.DataFrame) -> pd.DataFrame:
    """Counts of measured zone (rows) vs synthetic zone (columns), zones 7..2."""
    zones = list(ROBERTSON_ZONE_NAMES)
    table = pd.crosstab(pooled["zone_measured"], pooled["zone_synthetic"]).reindex(index=zones, columns=zones, fill_value=0)
    table.index = [ROBERTSON_ZONE_NAMES[z] for z in zones]
    table.columns = [ROBERTSON_ZONE_NAMES[z] for z in zones]
    return table


def overall_summary(pooled: pd.DataFrame, layer_table: Optional[pd.DataFrame]) -> Dict[str, object]:
    summary = {"all_samples": compute_metrics(pooled)}
    for category in SAMPLE_CATEGORIES:
        summary[f"{category}_samples"] = compute_metrics(pooled[pooled["sample_category"] == category])
    if layer_table is not None and len(layer_table):
        err = layer_table["synthetic_ic_mean"] - layer_table["measured_ic_mean"]
        summary["layer_means"] = {
            "n_layers": int(len(layer_table)),
            "bias": float(err.mean()),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "pearson_r": float(layer_table["synthetic_ic_mean"].corr(layer_table["measured_ic_mean"]))
            if len(layer_table) > 2 else np.nan,
        }
    return summary
