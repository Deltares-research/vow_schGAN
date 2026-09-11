"""Figures for the borehole-to-Ic workflow: 3-panel elevation profile + triangle scatter.

Reuses `NEN_graph.visualization` for the triangle rendering itself (colors,
polygon boundaries, Ic background, hover script) rather than re-implementing
any triangle drawing here.
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import get_nen_triangle  # noqa: E402
from modules.NEN_graph.ic_mapping import get_ic_surface  # noqa: E402
from modules.NEN_graph.visualization import _COLORS, _build_hover_script, plot_nen_triangle_with_ic  # noqa: E402

# Lengkeek lithology indices (1-10) have NO established color palette anywhere
# in geolib_plus - these colors are invented purely for this figure's
# legibility and must not be read as an authoritative Lengkeek palette.
_LENGKEEK_COLORS = {i: c for i, c in enumerate(plt.get_cmap("tab10").colors, start=1)}


def _draw_class_bands(ax, bands, color_lookup, fallback_color="#dddddd", min_label_thickness=0.05):
    """Draw horizontal (top, bottom, key) bands on `ax` (elevation on y-axis).

    Bands thinner than `min_label_thickness` are still drawn (so the true,
    possibly noisy, banding is visible) but are not labeled, to keep text
    legible; raise `min_label_thickness` for panels with many thin bands
    (e.g. the raw, unsmoothed reference CPT classification).
    """
    for top, bottom, key, label_text in bands:
        color = color_lookup.get(key, fallback_color) if key is not None else fallback_color
        hatch = None if key is not None else "//"
        ax.axhspan(bottom, top, facecolor=color, edgecolor="black", linewidth=0.4, hatch=hatch, alpha=0.85)
        if label_text and (top - bottom) > min_label_thickness:
            ax.text(0.5, (top + bottom) / 2.0, label_text, ha="center", va="center", fontsize=7,
                    transform=ax.get_yaxis_transform())


def _layer_bands(layers_df: pd.DataFrame, surface_elevation_m_nap: float):
    return [
        (surface_elevation_m_nap - row.upper_boundary_m, surface_elevation_m_nap - row.lower_boundary_m,
         row.class_code, row.class_code or "unsupported")
        for row in layers_df.itertuples()
    ]


def _reference_lengkeek_bands(reference_df: pd.DataFrame):
    """Contiguous (top, bottom, lithology_index, label) bands from the raw reference CPT classification."""
    bands = []
    if not len(reference_df):
        return bands
    lith = reference_df["lithology_index"].to_numpy()
    elev = reference_df["elevation_m_nap"].to_numpy()
    start = 0
    for i in range(1, len(lith) + 1):
        if i == len(lith) or lith[i] != lith[start]:
            idx = int(lith[start])
            label = f"{reference_df['main_group'].iloc[start]}\n{reference_df['qualifier'].iloc[start]}"
            bands.append((elev[start], elev[i - 1], idx, label))
            start = i
    return bands


def _shared_ic_xlim(*ic_series, margin_fraction: float = 0.05):
    values = pd.concat(ic_series).dropna()
    if not len(values):
        return None
    margin = margin_fraction * (values.max() - values.min() or 1.0)
    return values.min() - margin, values.max() + margin


def plot_three_panel(
    layers_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    surface_elevation_m_nap: float,
    title: str,
    output_path: Optional[Path] = None,
):
    """Borehole NEN classes | synthetic Ic realization | reference Lengkeek CPT | synthetic vs reference overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 9), sharey=True, gridspec_kw={"width_ratios": [1, 2, 2, 2]})

    # Panel 1: borehole NEN classes.
    layer_bands = _layer_bands(layers_df, surface_elevation_m_nap)
    _draw_class_bands(axes[0], layer_bands, _COLORS)
    axes[0].set_title("Borehole NEN 5104 classes")
    axes[0].set_ylabel("Elevation (m NAP)")
    axes[0].set_xticks([])

    # Panel 2: synthetic Ic realization, with the same NEN-class background shading.
    _draw_class_bands(axes[1], layer_bands, _COLORS)
    axes[1].plot(samples_df["synthetic_ic"], samples_df["elevation_m_nap"], color="black", linewidth=1.0)
    axes[1].set_title("Synthetic Ic realization")
    axes[1].set_xlabel("Ic (-)")

    # Panel 3: reference CPT Lengkeek classification (lithology-index bands, invented colors).
    ref_bands = _reference_lengkeek_bands(reference_df)
    _draw_class_bands(axes[2], ref_bands, _LENGKEEK_COLORS, min_label_thickness=0.35)
    if len(reference_df):
        axes[2].plot(reference_df["reference_ic_geolib_plus"], reference_df["elevation_m_nap"],
                     color="black", linewidth=0.8)
    axes[2].set_title("Reference CPT - Lengkeek (geolib_plus)")
    axes[2].set_xlabel("Ic (-), geolib_plus")

    # Panel 4: synthetic and reference Ic overlaid on one axis for direct comparison.
    axes[3].plot(samples_df["synthetic_ic"], samples_df["elevation_m_nap"], color="tab:blue", linewidth=1.2,
                 label="synthetic Ic (borehole)")
    if len(reference_df):
        axes[3].plot(reference_df["reference_ic_geolib_plus"], reference_df["elevation_m_nap"],
                     color="black", linestyle="--", linewidth=1.0, label="reference CPT Ic (measured)")
    axes[3].set_title("Synthetic vs reference CPT")
    axes[3].set_xlabel("Ic (-)")
    axes[3].legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Shared Ic x-axis scale across all Ic panels so the profiles are directly comparable.
    ic_xlim = _shared_ic_xlim(samples_df["synthetic_ic"], reference_df.get("reference_ic_geolib_plus", pd.Series(dtype=float)))
    if ic_xlim is not None:
        axes[1].set_xlim(ic_xlim)
        axes[2].set_xlim(ic_xlim)
        axes[3].set_xlim(ic_xlim)

    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_ensemble_three_panel(
    layers_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    surface_elevation_m_nap: float,
    n_realizations: int,
    title: str,
    output_path: Optional[Path] = None,
):
    """Borehole NEN classes | synthetic Ic ensemble (median + 5-95% band) | reference Lengkeek CPT."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharey=True, gridspec_kw={"width_ratios": [1, 2, 2]})

    # Panel 1: borehole NEN classes.
    layer_bands = _layer_bands(layers_df, surface_elevation_m_nap)
    _draw_class_bands(axes[0], layer_bands, _COLORS)
    axes[0].set_title("Borehole NEN 5104 classes")
    axes[0].set_ylabel("Elevation (m NAP)")
    axes[0].set_xticks([])

    # Panel 2: synthetic Ic ensemble - faint individual realizations, median, and 5-95% band.
    elevation = ensemble_df["elevation_m_nap"]
    for r in range(n_realizations):
        axes[1].plot(ensemble_df[f"ic_realization_{r}"], elevation, color="gray", alpha=0.20, linewidth=0.6,
                      label="individual realizations" if r == 0 else None)
    axes[1].fill_betweenx(elevation, ensemble_df["ic_p05"], ensemble_df["ic_p95"], color="tab:blue", alpha=0.25,
                           label="central 90% interval across 20 synthetic realizations")
    axes[1].plot(ensemble_df["ic_median"], elevation, color="tab:blue", linewidth=1.8, label="ensemble median")
    if len(reference_df):
        axes[1].plot(reference_df["reference_ic_geolib_plus"], reference_df["elevation_m_nap"],
                      color="black", linestyle="--", linewidth=1.0, label="reference CPT Ic (measured)")
    axes[1].set_title(f"Synthetic Ic ensemble (n={n_realizations})")
    axes[1].set_xlabel("Ic (-)")
    axes[1].legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    # Panel 3: reference CPT Lengkeek classification (unchanged from the single-realization figure).
    ref_bands = _reference_lengkeek_bands(reference_df)
    _draw_class_bands(axes[2], ref_bands, _LENGKEEK_COLORS, min_label_thickness=0.35)
    if len(reference_df):
        axes[2].plot(reference_df["reference_ic_geolib_plus"], reference_df["elevation_m_nap"],
                     color="black", linewidth=0.8)
    axes[2].set_title("Reference CPT - Lengkeek (geolib_plus)")
    axes[2].set_xlabel("Ic (-), geolib_plus")

    # Shared Ic x-axis scale across panels 2 and 3 (existing behaviour, preserved).
    ic_cols = [f"ic_realization_{r}" for r in range(n_realizations)] + ["ic_p05", "ic_p95"]
    ic_xlim = _shared_ic_xlim(
        ensemble_df[ic_cols].stack(), reference_df.get("reference_ic_geolib_plus", pd.Series(dtype=float)),
    )
    if ic_xlim is not None:
        axes[1].set_xlim(ic_xlim)
        axes[2].set_xlim(ic_xlim)

    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_triangle_samples(
    samples_df: pd.DataFrame,
    output_path: Path,
    title: str = "Sampled compositions overlaid on the NEN Ic field",
):
    """Companion interactive triangle scatter: every actual sampled point, over the canonical Ic background.

    Reuses `plot_nen_triangle_with_ic` verbatim for the background (Ic field,
    polygon boundaries, continuous-hover script) and adds one extra discrete
    scatter trace of the real sampled (x, y) points with a rich per-point
    hovertemplate (depth, elevation, source/evaluated class, composition, Ic).
    """
    import plotly.graph_objects as go

    triangle = get_nen_triangle()
    surface = get_ic_surface()
    fig = plot_nen_triangle_with_ic(output_path=output_path)  # writes once; we overwrite below with samples added

    valid = samples_df.dropna(subset=["x", "y"])
    hover_text = [
        (
            f"depth: {r.depth_m:.2f} m<br>elevation: {r.elevation_m_nap:.2f} m NAP<br>"
            f"source class: {r.source_class_code}<br>evaluated class: {r.evaluated_class_code}<br>"
            f"sand/silt/lutum: {r.sand_pct:.1f}/{r.silt_pct:.1f}/{r.lutum_pct:.1f} %<br>"
            f"synthetic Ic: {r.synthetic_ic:.3f}<br>transition: {r.is_transition}"
        )
        for r in valid.itertuples()
    ]
    fig.add_trace(go.Scatter(
        x=valid["x"], y=valid["y"], mode="markers",
        marker=dict(size=4, color=valid["depth_m"], colorscale="Cividis", showscale=True,
                    colorbar=dict(title="Depth (m)", x=1.15)),
        text=hover_text, hoverinfo="text", name="sampled points",
    ))
    fig.update_layout(title=title)

    div_id = triangle.triangle_id.replace("_", "-") + "-ic-plot"
    from modules.NEN_graph.classification import get_triangle_polygons  # noqa: E402
    polygons = get_triangle_polygons(triangle)
    post_script = _build_hover_script(
        div_id, polygons, "english_name", triangle,
        ic_field_js=surface.to_js_field(), classification_label_override="NEN class",
    )
    fig.write_html(str(output_path), div_id=div_id, post_script=post_script)
    return fig
