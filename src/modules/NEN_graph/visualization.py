"""Generic ternary-triangle visualization (matplotlib + Plotly).

None of the geometry/layout logic in this module references specific
component names (e.g. sand/silt/lutum): corner labels, axis captions,
legend text, and hover wording are all derived from a `TriangleDefinition`.
Only `_COLORS` (fill colors keyed by NEN class code) and the
`plot_nen_triangle*` wrapper functions are NEN-specific; they simply call
the generic `plot_triangle*` functions with the default NEN triangle.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import (  # noqa: E402
    classify_composition_generic,
    get_nen_triangle,
    get_triangle_polygons,
)
from modules.NEN_graph.coordinates import cartesian_to_composition, composition_to_cartesian  # noqa: E402
from modules.NEN_graph.triangle_definition import TriangleDefinition  # noqa: E402

logger = logging.getLogger(__name__)

# Light, qualitative colors grouped by main soil group (klei/leem/zand).
# NEN-specific: these distinguish classes only; they do not encode any numeric scale.
_COLORS = {
    "Ks1": "#c6dbef", "Ks2": "#9ecae1", "Ks3": "#6baed6", "Ks4": "#4292c6",
    "Kz1": "#c7e9c0", "Kz2": "#a1d99b", "Kz3": "#74c476",
    "Lz1": "#fff2ae", "Lz3": "#fdd49e",
    "Zk": "#fcbba1", "Zs1": "#fdd0a2", "Zs2": "#fdae6b",
    "Zs3": "#fd8d3c", "Zs4": "#e6550d",
}

# Deterministic qualitative fallback palette (ColorBrewer Set3 + Paired),
# used when a generic triangle is plotted without an explicit color map.
_DEFAULT_PALETTE = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
]

# Structural (opposite_corner, label_corner, other_corner) triples for the
# triangle's three edges. Purely geometric - independent of component names.
_EDGE_CORNERS = [
    ("top", "left", "right"),    # edge opposite the top corner (the "bottom" edge)
    ("right", "top", "left"),    # edge opposite the right corner (the "left" edge)
    ("left", "right", "top"),    # edge opposite the left corner (the "right" edge)
]

_MPL_TICK_STYLE = {
    "top":   {"dx": 0.0,    "dy": -1.0, "ha": "center", "va": "top",    "rotation": 0},
    "right": {"dx": -0.866, "dy": 0.5,  "ha": "right",  "va": "center", "rotation": 60},
    "left":  {"dx": 0.866,  "dy": 0.5,  "ha": "left",   "va": "center", "rotation": -60},
}
_MPL_TITLE_STYLE = {
    "top":   {"xy": (0.5, -0.14),            "rotation": 0,   "va": "top"},
    "right": {"xy": (-0.20, 0.5 * 0.866 / 2), "rotation": 60,  "va": "center"},
    "left":  {"xy": (1.20, 0.5 * 0.866 / 2),  "rotation": -60, "va": "center"},
}

_PLOTLY_TICK_STYLE = {
    "top":   {"dx": 0.0,    "dy": -0.035, "yanchor": "top"},
    "right": {"dx": -0.035, "dy": 0.02,   "xanchor": "right", "textangle": 60},
    "left":  {"dx": 0.035,  "dy": 0.02,   "xanchor": "left",  "textangle": -60},
}
_PLOTLY_TITLE_STYLE = {
    "top":   {"xy": (0.5, -0.16),  "textangle": 0},
    "right": {"xy": (-0.24, 0.22), "textangle": 60},
    "left":  {"xy": (1.24, 0.22),  "textangle": -60},
}


def _resolve_colors(triangle: TriangleDefinition, colors: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return a code -> hex-color mapping; a deterministic fallback when `colors` is None."""
    if colors is not None:
        return colors
    return {poly.code: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, poly in enumerate(triangle.polygons)}


def _classification_label(triangle: TriangleDefinition) -> str:
    return triangle.metadata.get("classification_label", "Classification")


def _format_readout(composition: Dict[str, float], triangle: TriangleDefinition, result: dict) -> str:
    """Build a multi-line "<label>: <value>%" readout, ending with the classification."""
    lines = [f"{triangle.label_for(k)}: {composition[k]:.1f}%" for k in triangle.component_keys]
    if result["status"] == "inside":
        label = f"{result['class_code']} ({result['class_name']})"
    elif result["status"] == "on_boundary":
        label = "boundary: " + ", ".join(result["candidate_classes"])
    else:
        label = result["status"]
    lines.append(f"{_classification_label(triangle)}: {label}")
    return "\n".join(lines)


def _gridlines(triangle: TriangleDefinition, step: int = 10):
    """Cartesian line segments for the full ternary background grid.

    For each component, yields the interior lines of constant value
    (excluding 0/total, which coincide with the outer edges). Each line is
    parallel to the edge opposite that component's corner.
    """
    keys = triangle.component_keys
    total = triangle.total
    n = int(round(total / step))
    lines = []
    for v in keys:
        others = [k for k in keys if k != v]
        for i in range(1, n):
            s = i * step
            comp_a = {v: s, others[0]: total - s, others[1]: 0}
            comp_b = {v: s, others[0]: 0, others[1]: total - s}
            lines.append((composition_to_cartesian(comp_a, triangle), composition_to_cartesian(comp_b, triangle)))
    return lines


def _add_ternary_grid(ax, triangle: TriangleDefinition, step: int = 10):
    """Draw the full ternary grid lightly behind the classification polygons."""
    for (x0, y0), (x1, y1) in _gridlines(triangle, step):
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5, alpha=0.5, zorder=0)


def _add_axis_ticks(ax, triangle: TriangleDefinition, step: int = 10, tick_len: float = 0.02):
    """Draw percentage tick marks (0-total) and labels along each edge."""
    total = triangle.total
    n = int(round(total / step))
    for opposite, label_corner, other_corner in _EDGE_CORNERS:
        style = _MPL_TICK_STYLE[opposite]
        opposite_key = triangle.component_by_corner(opposite).key
        label_key = triangle.component_by_corner(label_corner).key
        other_key = triangle.component_by_corner(other_corner).key
        for i in range(0, n + 1):
            t = i * step
            x, y = composition_to_cartesian({opposite_key: 0, label_key: t, other_key: total - t}, triangle)
            dx, dy = style["dx"] * tick_len, style["dy"] * tick_len
            ax.plot([x, x + dx], [y, y + dy], color="black", linewidth=0.8)
            ax.text(
                x + dx * 2.2, y + dy * 2.2, str(int(t)),
                ha=style["ha"], va=style["va"], fontsize=7, rotation=style["rotation"],
            )


def _add_axis_labels(ax, triangle: TriangleDefinition):
    for opposite, label_corner, _other in _EDGE_CORNERS:
        style = _MPL_TITLE_STYLE[opposite]
        label = triangle.label_for(triangle.component_by_corner(label_corner).key)
        x, y = style["xy"]
        ax.text(x, y, f"{label} (%)", ha="center", va=style["va"], fontsize=11, rotation=style["rotation"])


def plot_triangle(
    triangle: TriangleDefinition,
    ax: Optional[plt.Axes] = None,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    name_field: str = "english_name",
    interactive: bool = True,
    grid_step: int = 10,
):
    """Draw an arbitrary ternary classification triangle.

    Args:
        triangle (TriangleDefinition): the triangle to draw.
        ax (matplotlib.axes.Axes, optional): axes to draw on; a new figure
            is created if not given.
        colors (dict, optional): class code -> hex color; a deterministic
            fallback palette is used when omitted.
        show_legend (bool): whether to draw the code -> name legend.
        name_field (str): polygon attribute to use for legend text
            ("english_name" or "dutch_name" for the NEN dataset).
        interactive (bool): attach a hover readout (composition + class + status).
        grid_step (int): spacing (in `triangle.total` units) of the background grid.

    Returns:
        matplotlib.figure.Figure: the figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * (3 ** 0.5))]
    ax.add_patch(MplPolygon(outer, closed=True, fill=False, edgecolor="black", linewidth=1.5, zorder=5))

    _add_ternary_grid(ax, triangle, grid_step)

    polygons = get_triangle_polygons(triangle)
    palette = _resolve_colors(triangle, colors)
    for code, poly in polygons.items():
        color = palette.get(code, "#dddddd")
        patch = MplPolygon(
            poly.cartesian_vertices, closed=True, facecolor=color,
            edgecolor="black", linewidth=0.8, alpha=0.6, zorder=1,
        )
        ax.add_patch(patch)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    _add_axis_ticks(ax, triangle, grid_step)
    _add_axis_labels(ax, triangle)

    if show_legend:
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=palette.get(code, "#dddddd"), edgecolor="black")
            for code in polygons
        ]
        labels = [f"{code} - {getattr(poly, name_field)}" for code, poly in polygons.items()]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8, frameon=True)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.20, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(triangle.metadata.get("title", triangle.triangle_id))

    if interactive:
        _attach_hover_readout(fig, ax, triangle)

    return fig


def _attach_hover_readout(fig, ax, triangle: TriangleDefinition):
    """Attach a motion-notify handler showing composition + classification."""
    annotation = ax.annotate(
        "", xy=(0, 0), xytext=(0.5, 1.02), textcoords="data",
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
    )
    annotation.set_visible(False)

    def _on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        composition = cartesian_to_composition(event.xdata, event.ydata, triangle)
        result = classify_composition_generic(composition, triangle)
        annotation.set_text(_format_readout(composition, triangle, result))
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a "#rrggbb" color to a "rgba(r,g,b,a)" string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _build_hover_script(div_id: str, polygons, name_field: str, triangle: TriangleDefinition) -> str:
    """JS snippet giving a pixel-accurate (non-snapping) coordinate/class readout.

    Rather than relying on Plotly's hover system (which always snaps to the
    nearest point of a discrete trace), this converts the raw mouse pixel
    position to data coordinates using Plotly's own axis pixel<->data
    mapping, then classifies the point client-side against the same polygon
    geometry used by the Python backend. The readout is a plain floating
    HTML overlay div updated on every mousemove, so it is fully continuous.

    The embedded JS contains no assumptions about specific component names:
    component keys/labels, the corner assignment, the total, and the
    classification label are all embedded as data read from `triangle`.
    """
    import json

    polys_js = [
        {
            "code": code,
            "name": f"{code} - {getattr(poly, name_field)}",
            "verts": [[round(x, 6), round(y, 6)] for x, y in poly.cartesian_vertices],
        }
        for code, poly in polygons.items()
    ]
    components_js = [{"key": c.key, "label": c.label} for c in triangle.components]
    corner_keys_js = {c.corner: c.key for c in triangle.components}

    polys_json = json.dumps(polys_js)
    components_json = json.dumps(components_js)
    corner_keys_json = json.dumps(corner_keys_js)
    classification_label_json = json.dumps(_classification_label(triangle))
    total_json = json.dumps(triangle.total)

    return f"""
    (function() {{
        var gd = document.getElementById('{div_id}');
        var polys = {polys_json};
        var components = {components_json};
        var cornerKeys = {corner_keys_json};
        var classificationLabel = {classification_label_json};
        var TOTAL = {total_json};
        var TOL = 0.003;

        var overlay = document.createElement('div');
        overlay.style.position = 'absolute';
        overlay.style.pointerEvents = 'none';
        overlay.style.background = 'rgba(255,255,255,0.9)';
        overlay.style.border = '1px solid #888';
        overlay.style.borderRadius = '4px';
        overlay.style.padding = '4px 8px';
        overlay.style.font = '12px sans-serif';
        overlay.style.zIndex = 1000;
        overlay.style.display = 'none';
        gd.style.position = 'relative';
        gd.appendChild(overlay);

        function pointInPolygon(x, y, verts) {{
            var inside = false;
            for (var i = 0, j = verts.length - 1; i < verts.length; j = i++) {{
                var xi = verts[i][0], yi = verts[i][1];
                var xj = verts[j][0], yj = verts[j][1];
                var hit = ((yi > y) !== (yj > y)) &&
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (hit) inside = !inside;
            }}
            return inside;
        }}

        function distToSegment(px, py, x1, y1, x2, y2) {{
            var dx = x2 - x1, dy = y2 - y1;
            var lenSq = dx * dx + dy * dy;
            var t = lenSq === 0 ? 0 : ((px - x1) * dx + (py - y1) * dy) / lenSq;
            t = Math.max(0, Math.min(1, t));
            return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
        }}

        function distToPolygon(x, y, verts) {{
            var minD = Infinity;
            for (var i = 0; i < verts.length; i++) {{
                var j = (i + 1) % verts.length;
                var d = distToSegment(x, y, verts[i][0], verts[i][1], verts[j][0], verts[j][1]);
                if (d < minD) minD = d;
            }}
            return minD;
        }}

        function classify(x, y) {{
            var candidates = [];
            for (var i = 0; i < polys.length; i++) {{
                var p = polys[i];
                var inside = pointInPolygon(x, y, p.verts);
                var d = inside ? 0 : distToPolygon(x, y, p.verts);
                if (inside || d <= TOL) candidates.push(p);
            }}
            if (candidates.length === 0) return 'unclassified';
            if (candidates.length === 1) return candidates[0].name;
            return 'boundary: ' + candidates.map(function(p) {{ return p.code; }}).join(', ');
        }}

        function cartesianToComposition(x, y) {{
            var wTop = y / (Math.sqrt(3) / 2);
            var wRight = x - 0.5 * wTop;
            var wLeft = 1 - wRight - wTop;
            var comp = {{}};
            comp[cornerKeys.left] = wLeft * TOTAL;
            comp[cornerKeys.right] = wRight * TOTAL;
            comp[cornerKeys.top] = wTop * TOTAL;
            return comp;
        }}

        gd.addEventListener('mousemove', function(evt) {{
            var fl = gd._fullLayout;
            if (!fl) return;
            var rect = gd.getBoundingClientRect();
            var xpx = evt.clientX - rect.left - fl._size.l;
            var ypx = evt.clientY - rect.top - fl._size.t;
            var x = fl.xaxis.p2d(xpx);
            var y = fl.yaxis.p2d(ypx);

            var outside = x < -0.02 || x > 1.02 || y < -0.02 ||
                (y > Math.sqrt(3) * Math.min(x, 1 - x) + 0.02);
            if (outside) {{
                overlay.style.display = 'none';
                return;
            }}

            var comp = cartesianToComposition(x, y);
            var lines = [];
            for (var k = 0; k < components.length; k++) {{
                var c = components[k];
                lines.push(c.label + ': ' + comp[c.key].toFixed(1) + '%');
            }}
            lines.push(classificationLabel + ': ' + classify(x, y));
            overlay.innerHTML = lines.join('<br>');
            overlay.style.left = (evt.clientX - rect.left + 14) + 'px';
            overlay.style.top = (evt.clientY - rect.top + 14) + 'px';
            overlay.style.display = 'block';
        }});

        gd.addEventListener('mouseleave', function() {{
            overlay.style.display = 'none';
        }});
    }})();
    """


def _add_plotly_axis_ticks(fig, triangle: TriangleDefinition, step: int = 10):
    """Add percentage tick labels and axis titles along each edge."""
    total = triangle.total
    n = int(round(total / step))
    for opposite, label_corner, other_corner in _EDGE_CORNERS:
        style = _PLOTLY_TICK_STYLE[opposite]
        opposite_key = triangle.component_by_corner(opposite).key
        label_key = triangle.component_by_corner(label_corner).key
        other_key = triangle.component_by_corner(other_corner).key
        for i in range(0, n + 1):
            t = i * step
            x, y = composition_to_cartesian({opposite_key: 0, label_key: t, other_key: total - t}, triangle)
            kwargs = dict(x=x + style["dx"], y=y + style["dy"], text=str(int(t)), showarrow=False, font=dict(size=10))
            if "yanchor" in style:
                kwargs["yanchor"] = style["yanchor"]
            if "xanchor" in style:
                kwargs["xanchor"] = style["xanchor"]
            if "textangle" in style:
                kwargs["textangle"] = style["textangle"]
            fig.add_annotation(**kwargs)

    for opposite, label_corner, _other in _EDGE_CORNERS:
        style = _PLOTLY_TITLE_STYLE[opposite]
        label = triangle.label_for(triangle.component_by_corner(label_corner).key)
        x, y = style["xy"]
        fig.add_annotation(
            x=x, y=y, text=f"{label} (%)", showarrow=False,
            font=dict(size=13), textangle=style["textangle"],
        )


def plot_triangle_plotly(
    triangle: TriangleDefinition,
    output_path: Optional[Path] = None,
    colors: Optional[Dict[str, str]] = None,
    grid_step: int = 10,
    name_field: str = "english_name",
):
    """Build an interactive Plotly version of an arbitrary ternary triangle and save it as HTML.

    The full ternary grid is drawn lightly behind the (semi-transparent)
    classification polygons. A small JS snippet embedded in the saved HTML
    tracks the raw mouse pixel position (via Plotly's own axis pixel<->data
    mapping) and classifies it client-side against the same polygon
    geometry, giving a fully continuous composition + classification
    readout with no snapping to a fixed sample grid.

    Args:
        triangle (TriangleDefinition): the triangle to draw.
        output_path (Path, optional): where to write the HTML file. Defaults
            to `<triangle_id>_interactive.html` next to this module.
        colors (dict, optional): class code -> hex color; a deterministic
            fallback palette is used when omitted.
        grid_step (int): spacing (in `triangle.total` units) of the background grid.
        name_field (str): polygon attribute to use for legend text.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    import plotly.graph_objects as go

    polygons = get_triangle_polygons(triangle)
    palette = _resolve_colors(triangle, colors)
    fig = go.Figure()
    div_id = triangle.triangle_id.replace("_", "-") + "-plot"

    for (x0, y0), (x1, y1) in _gridlines(triangle, grid_step):
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="lightgray", width=0.6),
            hoverinfo="skip", showlegend=False,
        ))

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * (3 ** 0.5)), (0.0, 0.0)]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in outer], y=[p[1] for p in outer], mode="lines",
        line=dict(color="black", width=2), hoverinfo="skip", showlegend=False,
    ))

    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts], y=[v[1] for v in verts],
            mode="lines", fill="toself",
            fillcolor=_hex_to_rgba(palette.get(code, "#dddddd"), 0.65),
            line=dict(color="black", width=1),
            name=f"{code} - {getattr(poly, name_field)}",
            hoverinfo="skip", showlegend=True,
        ))
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        fig.add_annotation(x=cx, y=cy, text=f"<b>{code}</b>", showarrow=False, font=dict(size=11, color="black"))

    _add_plotly_axis_ticks(fig, triangle, grid_step)

    fig.update_layout(
        title=triangle.metadata.get("title", triangle.triangle_id),
        xaxis=dict(visible=False, range=[-0.30, 1.35]),
        yaxis=dict(visible=False, range=[-0.20, 1.0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1000,
        height=850,
        hovermode="closest",
        legend=dict(x=1.02, y=1.0),
    )

    if output_path is None:
        output_path = Path(__file__).parent / f"{triangle.triangle_id}_interactive.html"
    fig.write_html(
        str(output_path),
        div_id=div_id,
        post_script=_build_hover_script(div_id, polygons, name_field, triangle),
    )
    return fig


# --- Backward-compatible NEN 5104 sand/silt/lutum API -----------------------


def plot_nen_triangle(
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    label_lang: str = "english",
    interactive: bool = True,
):
    """Backward-compatible: draw the NEN 5104 classification triangle.

    Args:
        ax (matplotlib.axes.Axes, optional): axes to draw on; a new figure
            is created if not given.
        show_legend (bool): whether to draw the code -> name legend.
        label_lang (str): "english" or "dutch", which name to use in the legend.
        interactive (bool): attach a hover readout (composition/class/status).

    Returns:
        matplotlib.figure.Figure: the figure containing the plot.
    """
    name_field = "english_name" if label_lang == "english" else "dutch_name"
    return plot_triangle(
        get_nen_triangle(), ax=ax, colors=_COLORS, show_legend=show_legend,
        name_field=name_field, interactive=interactive,
    )


def plot_nen_triangle_plotly(
    output_path: Optional[Path] = None,
    grid_step: int = 10,
    label_lang: str = "english",
):
    """Backward-compatible: build the interactive Plotly NEN 5104 triangle and save it as HTML.

    Args:
        output_path (Path, optional): where to write the HTML file. Defaults
            to `nen5104_triangle_interactive.html` next to this module.
        grid_step (int): spacing (in percent) of the background ternary grid.
        label_lang (str): "english" or "dutch", which name to use in the legend.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    name_field = "english_name" if label_lang == "english" else "dutch_name"
    if output_path is None:
        output_path = Path(__file__).parent / "nen5104_triangle_interactive.html"
    return plot_triangle_plotly(
        get_nen_triangle(), output_path=output_path, colors=_COLORS,
        grid_step=grid_step, name_field=name_field,
    )


if __name__ == "__main__":
    out_path = Path(__file__).parent / "nen5104_triangle_render.png"
    figure = plot_nen_triangle(interactive=False)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved static render to {out_path}")

    html_path = Path(__file__).parent / "nen5104_triangle_interactive.html"
    plot_nen_triangle_plotly(output_path=html_path)
    print(f"Saved interactive render to {html_path}")

