"""Visualization for the NEN 5104 lutum-silt-zand classification triangle.

Colors distinguish classification areas only (no gradient, no Ic scale).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from .classification import get_polygons, classify_composition
from .coordinates import ternary_to_cartesian, cartesian_to_ternary

logger = logging.getLogger(__name__)

# Light, qualitative colors grouped by main soil group (klei/leem/zand).
# These distinguish classes only; they do not encode any numeric scale.
_COLORS = {
    "Ks1": "#c6dbef", "Ks2": "#9ecae1", "Ks3": "#6baed6", "Ks4": "#4292c6",
    "Kz1": "#c7e9c0", "Kz2": "#a1d99b", "Kz3": "#74c476",
    "Lz1": "#fff2ae", "Lz3": "#fdd49e",
    "Zk": "#fcbba1", "Zs1": "#fdd0a2", "Zs2": "#fdae6b",
    "Zs3": "#fd8d3c", "Zs4": "#e6550d",
}


def _ternary_gridlines(step: int = 10):
    """Cartesian line segments for the full ternary background grid.

    Yields, for each of the three components, the interior lines of
    constant value (excluding 0/100 which coincide with the outer edges).
    Each line is parallel to the edge opposite that component's vertex.
    """
    lines = []
    for s in range(step, 100, step):
        lines.append((ternary_to_cartesian(s, 100 - s, 0), ternary_to_cartesian(s, 0, 100 - s)))
    for si in range(step, 100, step):
        lines.append((ternary_to_cartesian(100 - si, si, 0), ternary_to_cartesian(0, si, 100 - si)))
    for l in range(step, 100, step):
        lines.append((ternary_to_cartesian(100 - l, 0, l), ternary_to_cartesian(0, 100 - l, l)))
    return lines


def _add_ternary_grid(ax, step: int = 10):
    """Draw the full ternary grid lightly behind the classification polygons."""
    for (x0, y0), (x1, y1) in _ternary_gridlines(step):
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5, alpha=0.5, zorder=0)


def _add_axis_ticks(ax, step: int = 10):
    """Draw percentage tick marks (0-100) and labels along each edge."""
    tick_len = 0.02
    for s in range(0, 101, step):
        x, y = ternary_to_cartesian(s, 100 - s, 0)  # bottom edge (lutum=0)
        ax.plot([x, x], [y, y - tick_len], color="black", linewidth=0.8)
        ax.text(x, y - tick_len * 2.2, str(s), ha="center", va="top", fontsize=7)
    for l in range(0, 101, step):
        x, y = ternary_to_cartesian(100 - l, 0, l)  # left edge (silt=0)
        dx, dy = -tick_len * 0.866, tick_len * 0.5
        ax.plot([x, x + dx], [y, y + dy], color="black", linewidth=0.8)
        ax.text(x + dx * 2.2, y + dy * 2.2, str(l), ha="right", va="center", fontsize=7, rotation=60)
    for si in range(0, 101, step):
        x, y = ternary_to_cartesian(0, si, 100 - si)  # right edge (sand=0)
        dx, dy = tick_len * 0.866, tick_len * 0.5
        ax.plot([x, x + dx], [y, y + dy], color="black", linewidth=0.8)
        ax.text(x + dx * 2.2, y + dy * 2.2, str(si), ha="left", va="center", fontsize=7, rotation=-60)


def _add_axis_labels(ax):
    ax.text(0.5, -0.14, "sand (%)", ha="center", va="top", fontsize=11)
    ax.text(-0.20, 0.5 * 0.866 / 2, "lutum (%)", ha="center", va="center", fontsize=11, rotation=60)
    ax.text(1.20, 0.5 * 0.866 / 2, "silt (%)", ha="center", va="center", fontsize=11, rotation=-60)


def plot_nen_triangle(
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    label_lang: str = "english",
    interactive: bool = True,
):
    """Draw the NEN 5104 classification triangle.

    Args:
        ax (matplotlib.axes.Axes, optional): axes to draw on; a new figure
            is created if not given.
        show_legend (bool): whether to draw the code -> name legend.
        label_lang (str): "english" or "dutch", which name to use in the legend.
        interactive (bool): attach a hover readout (sand/silt/lutum/class/status).

    Returns:
        matplotlib.figure.Figure: the figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure

    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * (3 ** 0.5))]
    ax.add_patch(MplPolygon(outer, closed=True, fill=False, edgecolor="black", linewidth=1.5, zorder=5))

    _add_ternary_grid(ax)

    polygons = get_polygons()
    for code, poly in polygons.items():
        color = _COLORS.get(code, "#dddddd")
        patch = MplPolygon(
            poly.cartesian_vertices, closed=True, facecolor=color,
            edgecolor="black", linewidth=0.8, alpha=0.6, zorder=1,
        )
        ax.add_patch(patch)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    _add_axis_ticks(ax)
    _add_axis_labels(ax)

    if show_legend:
        name_attr = "english_name" if label_lang == "english" else "dutch_name"
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=_COLORS.get(code, "#dddddd"), edgecolor="black")
            for code in polygons
        ]
        labels = [f"{code} - {getattr(poly, name_attr)}" for code, poly in polygons.items()]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8, frameon=True)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.20, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("NEN 5104 - lutum-silt-zand classification triangle")

    if interactive:
        _attach_hover_readout(fig, ax)

    return fig


def _attach_hover_readout(fig, ax):
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
        sand, silt, lutum = cartesian_to_ternary(event.xdata, event.ydata)
        result = classify_composition(sand, silt, lutum)
        if result["status"] == "inside":
            text = f"sand={sand:.1f} silt={silt:.1f} lutum={lutum:.1f}\n{result['class_code']} ({result['class_name']})"
        elif result["status"] == "on_boundary":
            text = f"sand={sand:.1f} silt={silt:.1f} lutum={lutum:.1f}\non boundary: {', '.join(result['candidate_classes'])}"
        else:
            text = f"sand={sand:.1f} silt={silt:.1f} lutum={lutum:.1f}\n{result['status']}"
        annotation.set_text(text)
        annotation.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a "#rrggbb" color to a "rgba(r,g,b,a)" string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _build_hover_script(div_id: str, polygons, name_attr: str) -> str:
    """JS snippet giving a pixel-accurate (non-snapping) coordinate/class readout.

    Rather than relying on Plotly's hover system (which always snaps to the
    nearest point of a discrete trace), this converts the raw mouse pixel
    position to data coordinates using Plotly's own axis pixel<->data
    mapping, then classifies the point client-side against the same polygon
    geometry used by the Python backend. The readout is a plain floating
    HTML overlay div updated on every mousemove, so it is fully continuous.
    """
    import json

    polys_js = [
        {
            "code": code,
            "name": f"{code} - {getattr(poly, name_attr)}",
            "verts": [[round(x, 6), round(y, 6)] for x, y in poly.cartesian_vertices],
        }
        for code, poly in polygons.items()
    ]
    polys_json = json.dumps(polys_js)

    return f"""
    (function() {{
        var gd = document.getElementById('{div_id}');
        var polys = {polys_json};
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

        function cartesianToTernary(x, y) {{
            var lutum = (y / (Math.sqrt(3) / 2)) * 100;
            var silt = x * 100 - 0.5 * lutum;
            var sand = 100 - silt - lutum;
            return [sand, silt, lutum];
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

            var comp = cartesianToTernary(x, y);
            var sand = comp[0].toFixed(1), silt = comp[1].toFixed(1), lutum = comp[2].toFixed(1);
            var label = classify(x, y);
            overlay.innerHTML = 'sand: ' + sand + '%<br>silt: ' + silt + '%<br>lutum: ' + lutum + '%<br>' + label;
            overlay.style.left = (evt.clientX - rect.left + 14) + 'px';
            overlay.style.top = (evt.clientY - rect.top + 14) + 'px';
            overlay.style.display = 'block';
        }});

        gd.addEventListener('mouseleave', function() {{
            overlay.style.display = 'none';
        }});
    }})();
    """


def _add_plotly_axis_ticks(fig, step: int = 10):
    """Add percentage tick labels and axis titles along each edge."""
    for s in range(0, 101, step):
        x, y = ternary_to_cartesian(s, 100 - s, 0)  # bottom edge (lutum=0)
        fig.add_annotation(x=x, y=y - 0.035, text=str(s), showarrow=False, font=dict(size=10), yanchor="top")
    for l in range(0, 101, step):
        x, y = ternary_to_cartesian(100 - l, 0, l)  # left edge (silt=0)
        fig.add_annotation(
            x=x - 0.035, y=y + 0.02, text=str(l), showarrow=False,
            font=dict(size=10), textangle=60, xanchor="right",
        )
    for si in range(0, 101, step):
        x, y = ternary_to_cartesian(0, si, 100 - si)  # right edge (sand=0)
        fig.add_annotation(
            x=x + 0.035, y=y + 0.02, text=str(si), showarrow=False,
            font=dict(size=10), textangle=-60, xanchor="left",
        )
    fig.add_annotation(x=0.5, y=-0.16, text="sand (%)", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=-0.24, y=0.22, text="lutum (%)", showarrow=False, font=dict(size=13), textangle=60)
    fig.add_annotation(x=1.24, y=0.22, text="silt (%)", showarrow=False, font=dict(size=13), textangle=-60)


def plot_nen_triangle_plotly(
    output_path: Optional[Path] = None,
    grid_step: int = 10,
    label_lang: str = "english",
):
    """Build an interactive Plotly version of the NEN 5104 triangle and save it as HTML.

    The full ternary grid is drawn lightly behind the (semi-transparent)
    classification polygons. A small JS snippet embedded in the saved HTML
    tracks the raw mouse pixel position (via Plotly's own axis pixel<->data
    mapping) and classifies it client-side against the same polygon
    geometry, giving a fully continuous sand/silt/lutum + classification
    readout with no snapping to a fixed sample grid.

    Args:
        output_path (Path, optional): where to write the HTML file. Defaults
            to `nen5104_triangle_interactive.html` next to this module.
        grid_step (int): spacing (in percent) of the background ternary grid.
        label_lang (str): "english" or "dutch", which name to use in the legend.

    Returns:
        plotly.graph_objects.Figure: the interactive figure.
    """
    import plotly.graph_objects as go

    polygons = get_polygons()
    fig = go.Figure()
    div_id = "nen5104-triangle-plot"

    for (x0, y0), (x1, y1) in _ternary_gridlines(grid_step):
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

    name_attr = "english_name" if label_lang == "english" else "dutch_name"
    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        fig.add_trace(go.Scatter(
            x=[v[0] for v in verts], y=[v[1] for v in verts],
            mode="lines", fill="toself",
            fillcolor=_hex_to_rgba(_COLORS.get(code, "#dddddd"), 0.65),
            line=dict(color="black", width=1),
            name=f"{code} - {getattr(poly, name_attr)}",
            hoverinfo="skip", showlegend=True,
        ))
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        fig.add_annotation(x=cx, y=cy, text=f"<b>{code}</b>", showarrow=False, font=dict(size=11, color="black"))

    _add_plotly_axis_ticks(fig)

    fig.update_layout(
        title="NEN 5104 - lutum-silt-zand classification triangle",
        xaxis=dict(visible=False, range=[-0.30, 1.35]),
        yaxis=dict(visible=False, range=[-0.20, 1.0], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        width=1000,
        height=850,
        hovermode="closest",
        legend=dict(x=1.02, y=1.0),
    )

    if output_path is None:
        output_path = Path(__file__).parent / "nen5104_triangle_interactive.html"
    fig.write_html(
        str(output_path),
        div_id=div_id,
        post_script=_build_hover_script(div_id, polygons, name_attr),
    )
    return fig


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    out_path = Path(__file__).parent / "nen5104_triangle_render.png"
    figure = plot_nen_triangle(interactive=False)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved static render to {out_path}")

    html_path = Path(__file__).parent / "nen5104_triangle_interactive.html"
    plot_nen_triangle_plotly(output_path=html_path)
    print(f"Saved interactive render to {html_path}")
