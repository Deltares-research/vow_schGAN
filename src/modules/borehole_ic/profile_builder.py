"""Orchestrate: resolved borehole layers -> sampled compositions -> synthetic Ic profile.

This is the core of the workflow: it does NOT touch the triangle geometry or
the Ic surfaces, it only *calls* them at sampled composition coordinates.

Each layer is routed to the triangle of its main soil noun (mineral, peat or
gravel; see `triangles.py`) and sampled inside its class polygon there; Ic
comes from that triangle's own surface.

Transitions between two supported groups:
- same triangle: the sampled (x, y) points are blended (smoothstep) and the
  blended point is evaluated - unchanged from the mineral-only workflow;
- different triangles: (x, y) positions belong to different composition
  spaces and cannot be blended, so each side's Ic is evaluated in its own
  triangle and the two Ic values are blended with the same smoothstep weight.
  These rows carry no composition (NaN x/y/percentages).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.composition_sampling import (  # noqa: E402
    CORRELATION_LENGTH_M, RANDOM_SEED,
    LayerSampler, bounded_transition_half_thickness, smoothstep,
)
from modules.borehole_ic.soil_mapping import map_soil_name_to_class  # noqa: E402
from modules.borehole_ic.triangles import TRIANGLES, all_composition_columns, class_polygon, spec_for_class  # noqa: E402
from modules.NEN_graph.classification import classify_composition_generic  # noqa: E402
from modules.NEN_graph.coordinates import cartesian_to_composition  # noqa: E402

DEPTH_STEP_M = 0.02  # kept as its own named constant, deliberately not reused for anything else
TRANSITION_THICKNESS_M = 0.20  # requested transition thickness; may be shortened per-boundary (see composition_sampling)


def resolve_layers(raw_layers: List[dict]) -> List[dict]:
    """Attach NEN class resolution to each raw borehole layer, in original order."""
    resolved = []
    for i, layer in enumerate(raw_layers):
        mapping = map_soil_name_to_class(layer.get("soil_name_nen5104"))
        merged = dict(layer)
        merged["layer_index"] = i
        merged["thickness_m"] = layer["lower_boundary_m"] - layer["upper_boundary_m"]
        merged["supported"] = mapping["supported"]
        merged["class_code"] = mapping["class_code"]
        merged["triangle"] = mapping["triangle"]
        merged["resolved_dutch_name"] = mapping["dutch_name"]
        merged["unsupported_reason"] = mapping["reason"]
        merged["is_guess"] = mapping["is_guess"]
        merged["match_note"] = mapping["match_note"]
        resolved.append(merged)
    return resolved


def _group_supported_runs(layers: List[dict]) -> List[dict]:
    """Group consecutive layers sharing the same (supported, class_code).

    A run shares ONE sampler across its full depth range, so the stochastic
    trajectory continues uninterrupted at an internal same-class boundary
    (e.g. Ks1 -> Ks1) instead of restarting / getting an artificial
    transition. Only boundaries BETWEEN runs (a genuine class change, or a
    supported/unsupported change) get their own sampler and, where both
    sides are supported, a transition zone.
    """
    groups: List[dict] = []
    for i, layer in enumerate(layers):
        key = (layer["supported"], layer["class_code"])
        if groups and groups[-1]["key"] == key:
            groups[-1]["layer_indices"].append(i)
        else:
            groups.append({"key": key, "layer_indices": [i]})
    for g in groups:
        g["supported"], g["class_code"] = g["key"]
        g["start"] = layers[g["layer_indices"][0]]["upper_boundary_m"]
        g["end"] = layers[g["layer_indices"][-1]]["lower_boundary_m"]
        g["thickness_m"] = g["end"] - g["start"]
    return groups


def _layer_index_at_depth(layers: List[dict], z: float) -> int:
    """Original layer index whose [upper, lower) interval contains `z` (for labeling only)."""
    for idx, layer in enumerate(layers):
        if layer["upper_boundary_m"] - 1e-9 <= z <= layer["lower_boundary_m"] + 1e-9:
            return idx
    return 0 if z <= layers[0]["upper_boundary_m"] else len(layers) - 1


def _boundary_half_thicknesses(groups: List[dict], transition_thickness_m: float) -> Dict[int, float]:
    """Half-thickness of the transition zone AFTER group i (i.e. at boundary groups[i]/groups[i+1]).

    0.0 when either neighboring group is unsupported (no blending across an
    unsupported interval) or when there is no next group. Same-class runs
    are already merged into a single group, so every remaining group
    boundary is a genuine class (or support-status) change.
    """
    half = {}
    for i in range(len(groups) - 1):
        a, b = groups[i], groups[i + 1]
        if a["supported"] and b["supported"]:
            half[i] = bounded_transition_half_thickness(transition_thickness_m / 2.0, a["thickness_m"], b["thickness_m"])
        else:
            half[i] = 0.0
    return half


def _build_segments(groups: List[dict], half_th: Dict[int, float]) -> List[dict]:
    """Ordered, contiguous, non-overlapping segments tiling [0, final_depth]."""
    segments = []
    for i, group in enumerate(groups):
        ext_upper = half_th.get(i - 1, 0.0)
        ext_lower = half_th.get(i, 0.0)
        segments.append({
            "kind": "core", "group_index": i,
            "start": group["start"] + ext_upper,
            "end": group["end"] - ext_lower,
        })
        if half_th.get(i, 0.0) > 0.0:
            b = group["end"]
            segments.append({
                "kind": "transition", "from_group": i, "to_group": i + 1,
                "start": b - half_th[i], "end": b + half_th[i],
            })
    return segments


def _segment_for_depth(segments: List[dict], z: float) -> dict:
    for seg in segments:
        if seg["start"] - 1e-9 <= z <= seg["end"] + 1e-9:
            return seg
    return segments[-1] if z > segments[-1]["end"] else segments[0]


def _build_depth_grid(final_depth_m: float, depth_step_m: float) -> np.ndarray:
    n_steps = int(round(final_depth_m / depth_step_m))
    return np.round(np.linspace(0.0, n_steps * depth_step_m, n_steps + 1), 6)


def build_synthetic_profile(
    borehole: dict,
    depth_step_m: float = DEPTH_STEP_M,
    transition_thickness_m: float = TRANSITION_THICKNESS_M,
    correlation_length_m: float = CORRELATION_LENGTH_M,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the resolved layer table and the sampled synthetic Ic profile for one borehole.

    Returns:
        (layers_df, samples_df, run_info) where `run_info` documents the
        segment tiling and per-layer sampler fallback counts for the report.
    """
    layers = resolve_layers(borehole["layers"])
    layers_df = pd.DataFrame(layers)
    groups = _group_supported_runs(layers)

    half_th = _boundary_half_thicknesses(groups, transition_thickness_m)
    segments = _build_segments(groups, half_th)

    samplers: Dict[int, LayerSampler] = {}
    for i, group in enumerate(groups):
        if group["supported"]:
            group["spec"] = spec_for_class(group["class_code"])
            samplers[i] = LayerSampler(
                class_polygon(group["class_code"]),
                seed=random_seed + 1000 * i,
                correlation_length_m=correlation_length_m,
            )

    surface_elevation = borehole.get("surface_elevation_m_nap")
    depth_grid = _build_depth_grid(borehole["final_depth_m"], depth_step_m)
    composition_columns = all_composition_columns()

    rows = []
    for z in depth_grid:
        seg = _segment_for_depth(segments, z)
        row = {"depth_m": float(z), "elevation_m_nap": (surface_elevation - z) if surface_elevation is not None else None}

        if seg["kind"] == "core":
            g = seg["group_index"]
            group = groups[g]
            i = _layer_index_at_depth(layers, z)
            row.update({"is_transition": False, "source_layer_index": i, "source_class_code": group["class_code"]})
            if not group["supported"]:
                row.update(_empty_sample(composition_columns))
                row["note"] = layers[i]["unsupported_reason"]
            else:
                x, y, is_fallback = samplers[g].sample_at(z)
                row.update(_evaluate_point(group["spec"], x, y, composition_columns))
                row.update({"is_fallback": is_fallback, "note": layers[i]["match_note"]})
        else:
            g_from, g_to = seg["from_group"], seg["to_group"]
            spec_from, spec_to = groups[g_from]["spec"], groups[g_to]["spec"]
            x_from, y_from, fb_from = samplers[g_from].sample_at(z)
            x_to, y_to, fb_to = samplers[g_to].sample_at(z)
            t = (z - seg["start"]) / (seg["end"] - seg["start"])
            weight = float(smoothstep(np.array([t]))[0])
            row.update({
                "is_transition": True, "source_layer_index": groups[g_from]["layer_indices"][-1],
                "source_class_code": f"{groups[g_from]['class_code']}->{groups[g_to]['class_code']}",
            })
            if spec_from.key == spec_to.key:
                x = (1 - weight) * x_from + weight * x_to
                y = (1 - weight) * y_from + weight * y_to
                row.update(_evaluate_point(spec_from, x, y, composition_columns))
                row["note"] = f"transition weight={weight:.3f}"
            else:
                ic_from = _ic_or_nan(spec_from.surface.value_at_cartesian(x_from, y_from))
                ic_to = _ic_or_nan(spec_to.surface.value_at_cartesian(x_to, y_to))
                row.update(_empty_sample(composition_columns))
                row.update({
                    "triangle": f"{spec_from.key}->{spec_to.key}",
                    "synthetic_ic": (1 - weight) * ic_from + weight * ic_to,
                    "note": f"cross-triangle transition: Ic blended, weight={weight:.3f}",
                })
            row["is_fallback"] = fb_from or fb_to
        rows.append(row)

    samples_df = pd.DataFrame(rows)
    run_info = {
        "groups": [{"class_code": g["class_code"], "supported": g["supported"], "layer_indices": g["layer_indices"],
                    "triangle": g["spec"].key if g.get("spec") else None,
                    "start": g["start"], "end": g["end"]} for g in groups],
        "segments": segments,
        "transition_half_thicknesses": half_th,
        "fallback_counts": {i: s.fallback_count for i, s in samplers.items()},
        "depth_step_m": depth_step_m,
        "transition_thickness_m_requested": transition_thickness_m,
        "correlation_length_m": correlation_length_m,
        "random_seed": random_seed,
    }
    return layers_df, samples_df, run_info


def _ic_or_nan(value: Optional[float]) -> float:
    return float(value) if value is not None else np.nan


def _empty_sample(composition_columns) -> dict:
    """No-composition values for unsupported layers and cross-triangle transitions."""
    sample = {"triangle": None, "x": np.nan, "y": np.nan}
    sample.update({col: np.nan for col in composition_columns})
    sample.update({"evaluated_class_code": None, "synthetic_ic": np.nan, "is_fallback": False})
    return sample


def _evaluate_point(spec, x: float, y: float, composition_columns) -> dict:
    """Composition, class and Ic of a Cartesian point in one triangle; other triangles' columns are NaN."""
    triangle = spec.triangle
    comp = cartesian_to_composition(x, y, triangle)
    result = classify_composition_generic(comp, triangle)
    sample = {"triangle": spec.key, "x": x, "y": y}
    sample.update({col: np.nan for col in composition_columns})
    sample.update({spec.column(k): comp[k] for k in spec.component_keys})
    sample["evaluated_class_code"] = result.get("class_code") or (result.get("candidate_classes") or [None])[0]
    sample["synthetic_ic"] = _ic_or_nan(spec.surface.value_at_cartesian(x, y))
    return sample
