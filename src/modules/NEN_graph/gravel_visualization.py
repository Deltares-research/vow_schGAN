"""Visualization for the lutum+silt-zand-grind (gravel main-name) triangle.

Mirrors `visualization.plot_nen_triangle_with_ic` for the original
lutum-silt-zand chart, but restricted to the grind>=30% mapped region
(Gs, Gz1-4): the same colour palette, "Representative Ic" colourbar label,
Robertson contours, and unified hover/colour/contour evaluator
(`RestrictedIcSurface`), with g1-g3 (gravel modifiers, no Ic here) drawn as
neutral/hatched shading and labelled "Ic from main-name triangle" on hover.

Does not modify the original `visualization.py` chart functions (only adds
optional, backward-compatible parameters to the shared `_build_hover_script`
helper - see that module).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from modules.NEN_graph.classification import get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import SQRT3_OVER_2  # noqa: E402
from modules.NEN_graph.gravel_ic_mapping import (  # noqa: E402
    MAPPED_CLASSES,
    UNMAPPED_CLASSES,
    UNMAPPED_IC_TEXT,
    get_gravel_ic_surface,
    get_gravel_triangle,
)
from modules.NEN_graph.visualization import (  # noqa: E402
    _add_plotly_axis_ticks,
    _build_hover_script,
)

logger = logging.getLogger(__name__)

ROBERTSON_IC_CONTOURS = (2.05, 2.60, 2.95)
EXTRA_IC_CONTOURS = (1.31,)
COLORSCALE_IC_MIN = 1.00
COLORSCALE_IC_MAX = 1.60

_NEUTRAL_COLOR = "#d9d9d9"


def _sample_grid(surface, grid_resolution: int):
    xs = np.linspace(0.0, 1.0, grid_resolution)
    ys = np.linspace(0.0, SQRT3_OVER_2, grid_resolution)
    z = np.full((grid_resolution, grid_resolution), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            w_top = y / SQRT3_OVER_2
            w_right = x - 0.5 * w_top
            w_left = 1.0 - w_right - w_top
            if w_left >= -1e-9 and w_right >= -1e-9 and w_top >= -1e-9:
                value = surface.value_at_cartesian(x, y)
                if value is not None:
                    z[j, i] = value
    return xs, ys, z


def plot_gravel_triangle_with_ic(
    output_path: Optional[Path] = None,
    grid_resolution: int = 150,
    colorscale: str = "Viridis",
    opacity: float = 0.75,
    show_robertson_contours: bool = True,
):
    """Interactive Plotly gravel-main-name triangle with a representative-Ic gradient.

    Restricted to the grind>=30% mapped region (Gs, Gz1-4). g1-g3 (gravel
    modifiers) are drawn as neutral shading with no Ic value; hovering
    there reports "Ic from main-name triangle" instead of a number. Colour,
    contour lines, and hover all evaluate the same `RestrictedIcSurface`
    (via `to_js_field()`), so they cannot disagree.

    Args:
        output_path (Path, optional): where to write the HTML file. Defaults
            to `nen5104_triangle_grind_ic_interactive.html` next to this module.
        grid_resolution (int): number of samples per axis for the gradient mesh.
        colorscale (str): Plotly colorscale name (same default as the original chart).
        opacity (float): 0-1 transparency of the Ic gradient background.
        show_robertson_contours (bool): overlay contour lines at Ic = 2.05, 2.60, 2.95.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    import plotly.graph_objects as go

    triangle = get_gravel_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_gravel_ic_surface()

    xs, ys, z = _sample_grid(surface, grid_resolution)

    vmin, vmax = COLORSCALE_IC_MIN, COLORSCALE_IC_MAX

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=xs, y=ys, z=z,
        colorscale=colorscale,
        opacity=opacity,
        zmin=vmin, zmax=vmax, zauto=False,
        contours=dict(coloring="heatmap"),
        colorbar=dict(title="Representative Ic"),
        connectgaps=False,
        hoverinfo="skip",
        showlegend=False,
    ))

    if show_robertson_contours:
        for level in ROBERTSON_IC_CONTOURS:
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=z,
                showscale=False,
                connectgaps=False,
                contours=dict(coloring="lines", start=level, end=level, size=1, showlabels=True),
                line=dict(color="black", width=1.2),
                hoverinfo="skip",
                showlegend=False,
            ))

    for level in EXTRA_IC_CONTOURS:
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z,
            showscale=False,
            connectgaps=False,
            contours=dict(coloring="lines", start=level, end=level, size=1, showlabels=True),
            line=dict(color="royalblue", width=1.2, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Neutral shading for g1-g3 (no Ic here - drawn as flat fill, on top of
    # the (empty, NaN) heatmap in that region).
    for code in UNMAPPED_CLASSES:
        poly = polygons[code]
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts], y=[v[1] for v in verts],
            mode="lines", fill="toself",
            fillcolor="rgba(217,217,217,0.55)",
            line=dict(color="black", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2), (0.0, 0.0)]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in outer], y=[p[1] for p in outer], mode="lines",
        line=dict(color="black", width=2), hoverinfo="skip", showlegend=False,
    ))

    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts], y=[v[1] for v in verts],
            mode="lines", line=dict(color="black", width=1),
            hoverinfo="skip", showlegend=False,
        ))
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        fig.add_annotation(x=cx, y=cy, text=f"<b>{code}</b>", showarrow=False, font=dict(size=11, color="black"))

    _add_plotly_axis_ticks(fig, triangle)

    fig.update_layout(
        title="NEN 5104 - lutum+silt-zand-grind: gravel main-name representative Ic gradient",
        xaxis=dict(visible=False, range=[-0.30, 1.35]),
        yaxis=dict(visible=False, range=[-0.20, 1.0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1000,
        height=850,
        hovermode="closest",
    )

    div_id = triangle.triangle_id.replace("_", "-") + "-ic-plot"
    if output_path is None:
        output_path = Path(__file__).parent / "nen5104_triangle_grind_ic_interactive.html"

    post_script = _build_hover_script(
        div_id, polygons, "english_name", triangle,
        ic_field_js=surface.to_js_field(), classification_label_override="NEN class",
        unmapped_classes=list(UNMAPPED_CLASSES), unmapped_ic_text=UNMAPPED_IC_TEXT,
    )

    fig.write_html(str(output_path), div_id=div_id, post_script=post_script)
    return fig


def plot_gravel_triangle_with_ic_static(
    output_path: Optional[Path] = None,
    grid_resolution: int = 300,
    cmap: str = "viridis",
    show_robertson_contours: bool = True,
):
    """Static matplotlib PNG rendering of the gravel-main-name Ic gradient.

    Same evaluator (`RestrictedIcSurface`) as the interactive HTML;
    numerically identical colour field, restricted to grind>=30%, with
    g1-g3 shaded neutral gray.

    Args:
        output_path (Path, optional): where to save the PNG. Defaults to
            `nen5104_triangle_grind_render.png` next to this module.
        grid_resolution (int): number of samples per axis for the gradient mesh.
        cmap (str): matplotlib colormap name.
        show_robertson_contours (bool): overlay contour lines at Ic = 2.05, 2.60, 2.95.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    triangle = get_gravel_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_gravel_ic_surface()

    xs, ys, z = _sample_grid(surface, grid_resolution)
    zm = np.ma.masked_invalid(z)

    vmin, vmax = COLORSCALE_IC_MIN, COLORSCALE_IC_MAX

    fig, ax = plt.subplots(figsize=(9, 8))

    mesh = ax.pcolormesh(xs, ys, zm, cmap=cmap, shading="auto", alpha=0.85, vmin=vmin, vmax=vmax, zorder=1)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Representative Ic")

    if show_robertson_contours:
        cs = ax.contour(xs, ys, zm, levels=list(ROBERTSON_IC_CONTOURS), colors="black", linewidths=1.0, zorder=2)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    cs_extra = ax.contour(
        xs, ys, zm, levels=list(EXTRA_IC_CONTOURS), colors="royalblue", linewidths=1.2,
        linestyles="dashed", zorder=2,
    )
    ax.clabel(cs_extra, inline=True, fontsize=7, fmt="%.2f")

    for code in UNMAPPED_CLASSES:
        poly = polygons[code]
        patch = MplPolygon(
            poly.cartesian_vertices, closed=True, facecolor=_NEUTRAL_COLOR,
            edgecolor="black", linewidth=0.8, alpha=0.75, hatch="//", zorder=3,
        )
        ax.add_patch(patch)

    for code, poly in polygons.items():
        patch = MplPolygon(
            poly.cartesian_vertices, closed=True, fill=False,
            edgecolor="black", linewidth=0.9, zorder=4,
        )
        ax.add_patch(patch)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2)]
    ax.add_patch(MplPolygon(outer, closed=True, fill=False, edgecolor="black", linewidth=1.5, zorder=6))

    from modules.NEN_graph.visualization import _add_axis_labels, _add_axis_ticks

    _add_axis_ticks(ax, triangle)
    _add_axis_labels(ax, triangle)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.20, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("NEN 5104 - lutum+silt-zand-grind: gravel main-name representative Ic")

    if output_path is None:
        output_path = Path(__file__).parent / "nen5104_triangle_grind_render.png"
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    png_path = Path(__file__).parent / "nen5104_triangle_grind_render.png"
    static_fig = plot_gravel_triangle_with_ic_static(output_path=png_path)
    print(f"Saved static render to {png_path}")

    html_path = Path(__file__).parent / "nen5104_triangle_grind_ic_interactive.html"
    plot_gravel_triangle_with_ic(output_path=html_path)
    print(f"Saved interactive render to {html_path}")
