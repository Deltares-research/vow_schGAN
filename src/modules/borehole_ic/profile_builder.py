"""Orchestrate: resolved borehole layers -> sampled compositions -> synthetic Ic profile.

This is the core of the workflow: it does NOT touch the triangle geometry or
the canonical Ic surface, it only *calls* them (`classification.py`,
`ic_mapping.py`) at sampled composition coordinates.
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
from modules.NEN_graph.classification import classify_composition_generic, get_nen_triangle, get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import cartesian_to_composition  # noqa: E402
from modules.NEN_graph.ic_mapping import get_ic_surface  # noqa: E402

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
        merged["resolved_dutch_name"] = mapping["dutch_name"]
        merged["unsupported_reason"] = mapping["reason"]
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
    triangle = get_nen_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_ic_surface()

    layers = resolve_layers(borehole["layers"])
    layers_df = pd.DataFrame(layers)
    groups = _group_supported_runs(layers)

    half_th = _boundary_half_thicknesses(groups, transition_thickness_m)
    segments = _build_segments(groups, half_th)

    samplers: Dict[int, LayerSampler] = {}
    for i, group in enumerate(groups):
        if group["supported"]:
            samplers[i] = LayerSampler(
                polygons[group["class_code"]].shapely_polygon,
                seed=random_seed + 1000 * i,
                correlation_length_m=correlation_length_m,
            )

    surface_elevation = borehole.get("surface_elevation_m_nap")
    depth_grid = _build_depth_grid(borehole["final_depth_m"], depth_step_m)

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
                row.update({
                    "x": np.nan, "y": np.nan,
                    "sand_pct": np.nan, "silt_pct": np.nan, "lutum_pct": np.nan,
                    "evaluated_class_code": None, "synthetic_ic": np.nan, "is_fallback": False,
                    "note": layers[i]["unsupported_reason"],
                })
            else:
                x, y, is_fallback = samplers[g].sample_at(z)
                comp = cartesian_to_composition(x, y, triangle)
                result = classify_composition_generic(comp, triangle)
                ic = surface.value_at_cartesian(x, y)
                row.update({
                    "x": x, "y": y,
                    "sand_pct": comp["sand"], "silt_pct": comp["silt"], "lutum_pct": comp["lutum"],
                    "evaluated_class_code": result.get("class_code") or (result.get("candidate_classes") or [None])[0],
                    "synthetic_ic": ic, "is_fallback": is_fallback, "note": None,
                })
        else:
            g_from, g_to = seg["from_group"], seg["to_group"]
            x_from, y_from, fb_from = samplers[g_from].sample_at(z)
            x_to, y_to, fb_to = samplers[g_to].sample_at(z)
            t = (z - seg["start"]) / (seg["end"] - seg["start"])
            weight = float(smoothstep(np.array([t]))[0])
            x = (1 - weight) * x_from + weight * x_to
            y = (1 - weight) * y_from + weight * y_to
            comp = cartesian_to_composition(x, y, triangle)
            result = classify_composition_generic(comp, triangle)
            ic = surface.value_at_cartesian(x, y)
            row.update({
                "is_transition": True, "source_layer_index": groups[g_from]["layer_indices"][-1],
                "source_class_code": f"{groups[g_from]['class_code']}->{groups[g_to]['class_code']}",
                "x": x, "y": y,
                "sand_pct": comp["sand"], "silt_pct": comp["silt"], "lutum_pct": comp["lutum"],
                "evaluated_class_code": result.get("class_code") or (result.get("candidate_classes") or [None])[0],
                "synthetic_ic": ic, "is_fallback": (fb_from or fb_to),
                "note": f"transition weight={weight:.3f}",
            })
        rows.append(row)

    samples_df = pd.DataFrame(rows)
    run_info = {
        "groups": [{"class_code": g["class_code"], "supported": g["supported"], "layer_indices": g["layer_indices"],
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
