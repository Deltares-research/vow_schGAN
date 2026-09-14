"""Visualization for the organic stof-lutum-silt+zand (peat/veen) triangle.

Geometry-only chart (classification and visualization for review; no Ic
field, colour bar, or Robertson contours - Ic values are a follow-up task).
Mirrors the styling established for the gravel main-name chart: muted
categorical colours for the five veen main-name regions (Vm, Vk1, Vk3, Vz1,
Vz3), and neutral/hatched shading (with visible boundaries and labels) for
the h1-h3 humus modifiers.

Does not modify the original `visualization.py` chart functions beyond the
already-additive, backward-compatible `extra_lines_by_code` parameter of
`_build_hover_script` (see that module).
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
from modules.NEN_graph.peat_classification import (  # noqa: E402
    MAIN_NAME_CLASSES,
    MODIFIER_CLASSES,
    get_peat_triangle,
)
from modules.NEN_graph.peat_ic_mapping import (  # noqa: E402
    UNMAPPED_CLASSES,
    UNMAPPED_IC_TEXT,
    get_peat_ic_surface,
)
from modules.NEN_graph.visualization import (  # noqa: E402
    DEFAULT_NEN_GRAPH_OUTPUT_DIR,
    _add_axis_labels,
    _add_axis_ticks,
    _add_plotly_axis_ticks,
    _build_hover_script,
    _hex_to_rgba,
    _sample_ic_grid,
)

logger = logging.getLogger(__name__)

# Muted, qualitative colours for the five veen main-name regions only.
_MAIN_COLORS = {
    "Vm": "#8c96c6",
    "Vk1": "#88419d",
    "Vk3": "#6e016b",
    "Vz1": "#a1d99b",
    "Vz3": "#238b45",
}
_NEUTRAL_COLOR = "#d9d9d9"

# Robertson threshold (per user specification) vs. additional reference
# levels used only to inspect the field's shape - NOT Robertson boundaries.
ROBERTSON_IC_CONTOURS = (3.60,)
REFERENCE_IC_CONTOURS = (3.80, 4.00, 4.20)
COLORSCALE_IC_MIN = 3.55
COLORSCALE_IC_MAX = 4.40


def _extra_hover_lines(triangle) -> dict:
    """Class code -> hover line stating whether it supplies a main name."""
    lines = {}
    for poly in triangle.polygons:
        if poly.code in MAIN_NAME_CLASSES:
            lines[poly.code] = "Main name: veen (this triangle)"
        else:
            lines[poly.code] = "Main name: from lutum-silt-zand(-grind) triangle (modifier only)"
    return lines


def plot_peat_triangle_static(output_path: Optional[Path] = None):
    """Static matplotlib PNG of the peat/veen triangle geometry.

    Args:
        output_path (Path, optional): where to save the PNG. Defaults to
            `nen5104_triangle_peat_render.png` next to this module.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)

    fig, ax = plt.subplots(figsize=(9, 8))

    for code, poly in polygons.items():
        if code in MODIFIER_CLASSES:
            patch = MplPolygon(
                poly.cartesian_vertices, closed=True, facecolor=_NEUTRAL_COLOR,
                edgecolor="black", linewidth=0.8, alpha=0.75, hatch="//", zorder=2,
            )
        else:
            patch = MplPolygon(
                poly.cartesian_vertices, closed=True, facecolor=_MAIN_COLORS.get(code, "#dddddd"),
                edgecolor="black", linewidth=0.8, alpha=0.8, zorder=2,
            )
        ax.add_patch(patch)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2)]
    ax.add_patch(MplPolygon(outer, closed=True, fill=False, edgecolor="black", linewidth=1.5, zorder=6))

    _add_axis_ticks(ax, triangle)
    _add_axis_labels(ax, triangle)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.20, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(triangle.metadata.get("title", triangle.triangle_id))

    if output_path is None:
        output_path = DEFAULT_NEN_GRAPH_OUTPUT_DIR / "peat" / "nen5104_triangle_peat_render.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


def plot_peat_triangle_plotly(output_path: Optional[Path] = None):
    """Interactive Plotly HTML of the peat/veen triangle geometry.

    Hover shows organic matter/lutum/silt+zand percentages, the region
    code + Dutch description, and whether the region supplies a main name
    or only a humus modifier.

    Args:
        output_path (Path, optional): where to write the HTML file. Defaults
            to `nen5104_triangle_peat_interactive.html` next to this module.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    import plotly.graph_objects as go

    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)

    fig = go.Figure()

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2), (0.0, 0.0)]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in outer], y=[p[1] for p in outer], mode="lines",
        line=dict(color="black", width=2), hoverinfo="skip", showlegend=False,
    ))

    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        if code in MODIFIER_CLASSES:
            fillcolor = "rgba(217,217,217,0.55)"
        else:
            fillcolor = _hex_to_rgba(_MAIN_COLORS.get(code, "#dddddd"), 0.7)
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts], y=[v[1] for v in verts],
            mode="lines", fill="toself",
            fillcolor=fillcolor,
            line=dict(color="black", width=1),
            hoverinfo="skip", showlegend=False,
        ))
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        fig.add_annotation(x=cx, y=cy, text=f"<b>{code}</b>", showarrow=False, font=dict(size=11, color="black"))

    _add_plotly_axis_ticks(fig, triangle)

    fig.update_layout(
        title=triangle.metadata.get("title", triangle.triangle_id),
        xaxis=dict(visible=False, range=[-0.30, 1.35]),
        yaxis=dict(visible=False, range=[-0.20, 1.0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1000,
        height=850,
        hovermode="closest",
    )

    div_id = triangle.triangle_id.replace("_", "-") + "-plot"
    if output_path is None:
        output_path = DEFAULT_NEN_GRAPH_OUTPUT_DIR / "peat" / "nen5104_triangle_peat_interactive.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    post_script = _build_hover_script(
        div_id, polygons, "dutch_name", triangle,
        classification_label_override="NEN class",
        extra_lines_by_code=_extra_hover_lines(triangle),
    )

    fig.write_html(str(output_path), div_id=div_id, post_script=post_script)
    return fig


def plot_peat_triangle_with_ic(
    output_path: Optional[Path] = None,
    grid_resolution: int = 150,
    colorscale: str = "Viridis",
    opacity: float = 0.75,
    show_robertson_contour: bool = True,
):
    """Interactive Plotly peat/veen triangle with a representative-Ic gradient.

    Restricted to the 5 veen main-name polygons (Vm, Vk1, Vk3, Vz1, Vz3).
    h1-h3 (humus modifiers) are drawn as neutral/hatched shading with no
    Ic value; hovering there reports "Ic from main-name triangle" instead
    of a number. Colour, contour lines, and hover all evaluate the same
    `RestrictedIcSurface` (via `to_js_field()`), so they cannot disagree.

    Args:
        output_path (Path, optional): where to write the HTML file. Defaults
            to `nen5104_triangle_peat_ic_interactive.html` next to this module.
        grid_resolution (int): number of samples per axis for the gradient mesh.
        colorscale (str): Plotly colorscale name.
        opacity (float): 0-1 transparency of the Ic gradient background.
        show_robertson_contour (bool): overlay the Robertson threshold contour at Ic = 3.60.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    import plotly.graph_objects as go

    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_peat_ic_surface()

    xs, ys, z = _sample_ic_grid(surface, grid_resolution)

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

    if show_robertson_contour:
        for level in ROBERTSON_IC_CONTOURS:
            fig.add_trace(go.Contour(
                x=xs, y=ys, z=z,
                showscale=False,
                connectgaps=False,
                contours=dict(coloring="lines", start=level, end=level, size=1, showlabels=True),
                line=dict(color="black", width=1.2),
                name="Robertson threshold (3.60)",
                hoverinfo="skip",
                showlegend=False,
            ))

    for level in REFERENCE_IC_CONTOURS:
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z,
            showscale=False,
            connectgaps=False,
            contours=dict(coloring="lines", start=level, end=level, size=1, showlabels=True),
            line=dict(color="royalblue", width=1.0, dash="dash"),
            name="reference level (not a Robertson boundary)",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Neutral shading for h1-h3 (no Ic here - drawn as flat fill, on top of
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
        title="NEN 5104 - organic stof-lutum-silt+zand: peat/veen representative Ic gradient",
        xaxis=dict(visible=False, range=[-0.30, 1.35]),
        yaxis=dict(visible=False, range=[-0.20, 1.0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1000,
        height=850,
        hovermode="closest",
    )

    div_id = triangle.triangle_id.replace("_", "-") + "-ic-plot"
    if output_path is None:
        output_path = DEFAULT_NEN_GRAPH_OUTPUT_DIR / "peat" / "nen5104_triangle_peat_ic_interactive.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    post_script = _build_hover_script(
        div_id, polygons, "dutch_name", triangle,
        ic_field_js=surface.to_js_field(), classification_label_override="NEN class",
        unmapped_classes=list(UNMAPPED_CLASSES), unmapped_ic_text=UNMAPPED_IC_TEXT,
    )

    fig.write_html(str(output_path), div_id=div_id, post_script=post_script)
    return fig


def plot_peat_triangle_with_ic_static(
    output_path: Optional[Path] = None,
    grid_resolution: int = 300,
    cmap: str = "viridis",
    show_robertson_contour: bool = True,
):
    """Static matplotlib PNG rendering of the peat/veen Ic gradient.

    Same evaluator (`RestrictedIcSurface`) as the interactive HTML;
    numerically identical colour field, restricted to Vm/Vk1/Vk3/Vz1/Vz3,
    with h1-h3 shaded neutral gray.

    Args:
        output_path (Path, optional): where to save the PNG. Defaults to
            `nen5104_triangle_peat_ic_render.png` next to this module.
        grid_resolution (int): number of samples per axis for the gradient mesh.
        cmap (str): matplotlib colormap name.
        show_robertson_contour (bool): overlay the Robertson threshold contour at Ic = 3.60.

    Returns:
        matplotlib.figure.Figure: the figure.
    """
    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_peat_ic_surface()

    xs, ys, z = _sample_ic_grid(surface, grid_resolution)
    zm = np.ma.masked_invalid(z)

    vmin, vmax = COLORSCALE_IC_MIN, COLORSCALE_IC_MAX

    fig, ax = plt.subplots(figsize=(9, 8))

    mesh = ax.pcolormesh(xs, ys, zm, cmap=cmap, shading="auto", alpha=0.85, vmin=vmin, vmax=vmax, zorder=1)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Representative Ic")

    if show_robertson_contour:
        cs = ax.contour(xs, ys, zm, levels=list(ROBERTSON_IC_CONTOURS), colors="black", linewidths=1.0, zorder=2)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    cs_ref = ax.contour(
        xs, ys, zm, levels=list(REFERENCE_IC_CONTOURS), colors="royalblue", linewidths=0.9,
        linestyles="dashed", zorder=2,
    )
    ax.clabel(cs_ref, inline=True, fontsize=7, fmt="%.2f")

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

    _add_axis_ticks(ax, triangle)
    _add_axis_labels(ax, triangle)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.20, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "NEN 5104 - organic stof-lutum-silt+zand: peat/veen representative Ic\n"
        "(dashed = reference levels, not Robertson boundaries)"
    )

    if output_path is None:
        output_path = DEFAULT_NEN_GRAPH_OUTPUT_DIR / "peat" / "nen5104_triangle_peat_ic_render.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the peat/veen triangle geometry + Ic plots.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_NEN_GRAPH_OUTPUT_DIR / "peat",
        help="Folder to write the 4 output files to (default: res/NEN_graph/peat).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    png_path = args.output_dir / "nen5104_triangle_peat_render.png"
    static_fig = plot_peat_triangle_static(output_path=png_path)
    print(f"Saved static render to {png_path}")

    html_path = args.output_dir / "nen5104_triangle_peat_interactive.html"
    plot_peat_triangle_plotly(output_path=html_path)
    print(f"Saved interactive render to {html_path}")

    ic_png_path = args.output_dir / "nen5104_triangle_peat_ic_render.png"
    plot_peat_triangle_with_ic_static(output_path=ic_png_path)
    print(f"Saved Ic static render to {ic_png_path}")

    ic_html_path = args.output_dir / "nen5104_triangle_peat_ic_interactive.html"
    plot_peat_triangle_with_ic(output_path=ic_html_path)
    print(f"Saved Ic interactive render to {ic_html_path}")
