"""2D cross-section figure: tests as Ic-coloured columns at their distance along the line, elevation in m NAP."""

import io
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from modules.cross_section.cs_data import DESIGN_BOTTOM_M_NAP, SchemaganFrame

IC_VMIN, IC_VMAX = 1.0, 4.4  # covers mineral, gravel and peat Ic; values outside are shown at the end colours
ROBERTSON_THRESHOLDS = (1.31, 2.05, 2.60, 2.95, 3.60)
ROBERTSON_ZONES = [  # (lower, upper, label)
    (IC_VMIN, 1.31, "gravelly sand"), (1.31, 2.05, "sand"), (2.05, 2.60, "sand mixtures"),
    (2.60, 2.95, "silt mixtures"), (2.95, 3.60, "clay"), (3.60, IC_VMAX, "organic / peat"),
]


def _cell_edges(elevations: np.ndarray) -> np.ndarray:
    """Cell edges around sample elevations sorted from top to bottom."""
    if len(elevations) == 1:
        return np.array([elevations[0] + 0.01, elevations[0] - 0.01])
    mid = (elevations[:-1] + elevations[1:]) / 2.0
    return np.r_[elevations[0] + (elevations[0] - mid[0]), mid, elevations[-1] - (mid[-1] - elevations[-1])]


def _draw_column(ax, x_center, width, profile: pd.DataFrame, cmap, norm, dashed: bool, augmented: bool = False):
    prof = profile.dropna(subset=["elevation_m_nap"]).sort_values("elevation_m_nap", ascending=False)
    prof = prof.drop_duplicates("elevation_m_nap")
    if not len(prof):
        return None, None
    elev = prof["elevation_m_nap"].to_numpy(dtype=float)
    edges = _cell_edges(elev)
    values = np.ma.masked_invalid(prof["ic"].to_numpy(dtype=float)).reshape(-1, 1)
    mesh = ax.pcolormesh([x_center - width / 2, x_center + width / 2], edges, values,
                         cmap=cmap, norm=norm, shading="flat", zorder=2, alpha=0.75 if augmented else 1.0)
    ax.add_patch(Rectangle((x_center - width / 2, edges[-1]), width, edges[0] - edges[-1], fill=False,
                           edgecolor="dimgray" if augmented else "black", hatch="////" if augmented else None,
                           linewidth=0.8, linestyle="--" if dashed else "-", zorder=3))
    return mesh, edges[0]


def _label_offsets(positions: np.ndarray, min_spacing: float) -> np.ndarray:
    """Horizontal offsets (in points) that spread the number labels of tests closer than `min_spacing`."""
    offsets = np.zeros(len(positions))
    order = np.argsort(positions)
    group = [order[0]] if len(order) else []
    groups = []
    for i in order[1:]:
        if positions[i] - positions[group[-1]] < min_spacing:
            group.append(i)
        else:
            groups.append(group)
            group = [i]
    if group:
        groups.append(group)
    for g in groups:
        for j, i in enumerate(g):
            offsets[i] = (j - (len(g) - 1) / 2.0) * 16.0
    return offsets


def plot_cross_section(line, tests: pd.DataFrame, ic_profiles: pd.DataFrame, title: str,
                       output_path: Optional[Path] = None, frame: Optional[SchemaganFrame] = None,
                       occupied: Optional[int] = None):
    """Cross-section along the profile line: ground surface, tests as Ic columns, numbered, with a key.

    If `frame` is given, the SchemaGAN rectangle is drawn on top. Rows with
    `augmented=True` (see `cs_augment`) are drawn hatched and labelled "n=original".
    """
    drawn = tests[tests["status"] == "ok"].copy()
    if "augmented" not in drawn:
        drawn["augmented"] = False
    number_of = dict(zip(tests["name"], tests["number"]))
    key_lines = []
    for r in tests.itertuples():
        kind = "CPT" if r.kind == "cpt" else "borehole"
        if getattr(r, "augmented", False) is True:
            key_lines.append(f"{r.number:>2}  copy of #{number_of.get(r.copy_of, '?')} {r.copy_of} ({kind}) placed at "
                             f"{r.distance_along_m:6.1f} m [{r.copy_label}], NAP elevations kept")
        elif r.status == "ok":
            ground = f"ground {r.surface_elevation_m_nap:6.2f} m NAP" + (" (from profile)" if r.surface_from_profile else "")
            place = f"at {r.distance_along_m:6.1f} m, offset {r.offset_m:+5.1f} m" + (" (beyond A/B)" if r.beyond_line_ends else "")
            key_lines.append(f"{r.number:>2}  {r.name:<26} {kind:<8} {place}   {ground}   bottom {r.bottom_elevation_m_nap:6.2f} m NAP")
        else:
            key_lines.append(f"{r.number:>2}  {r.name:<26} {kind:<8} NOT DRAWN: {r.status} {r.note}".rstrip())
    gap_text = ", ".join(f"{s:.0f}-{e:.0f} m" for s, e in line.elevation_gaps) or "none"
    key_lines.append(f"Line A {tuple(np.round(line.a, 2))} -> B {tuple(np.round(line.b, 2))} (RD), length {line.length_m:.1f} m; "
                     f"offset > 0 = left of A->B; ground elevation interpolated linearly over gaps: {gap_text}")
    if frame is not None:
        key_lines.append(
            f"SchemaGAN frame {frame.n_cols} x {frame.n_rows}: x {frame.left_m:.1f}-{frame.right_m:.1f} m, "
            f"top {frame.top_m_nap:.2f} m NAP (highest test ground: {frame.top_source}), bottom {frame.bottom_m_nap:g} m NAP; "
            f"cell {frame.col_width_m:.3f} m wide x {frame.row_height_m:.3f} m high")
        if occupied is not None:
            key_lines[-1] += f"; occupied columns: {occupied}"

    fig, (ax, ax_key) = plt.subplots(2, 1, figsize=(16, 8 + 0.2 * len(key_lines)),
                                     gridspec_kw={"height_ratios": [8, 0.2 * len(key_lines) + 0.3], "hspace": 0.25})
    cmap = plt.get_cmap("viridis")
    norm = Normalize(IC_VMIN, IC_VMAX)

    width = max(1.0, 0.012 * max(line.length_m, 1.0))
    p = line.profile
    if "elev_nap" in p:
        bottom_guess = np.nanmin(drawn["bottom_elevation_m_nap"]) if len(drawn) else p["elev_nap"].min() - 10
        ax.fill_between(p["dist"], p["elev_nap"], bottom_guess - 5, color="#d8cfc4", alpha=0.25, zorder=0, linewidth=0)
        if p.get("elev_filled", pd.Series(False)).any():
            # Filled line first (dashed), measured line on top: the dash only shows across the gaps.
            ax.plot(p["dist"], p["elev_nap"], color="tab:red", linewidth=1.2, linestyle="--", zorder=4)
        ax.plot(p["dist"], p.get("elev_nap_raw", p["elev_nap"]), color="black", linewidth=1.2, zorder=5)

    # Copies and boreholes first, real CPTs last, so a real CPT is drawn on top where they overlap.
    tops = {}
    mesh = None
    draw_order = drawn.assign(_order=np.where(drawn["augmented"], 0, np.where(drawn["kind"] == "cpt", 2, 1))).sort_values("_order")
    for r in draw_order.itertuples():
        m, top = _draw_column(ax, r.distance_along_m, width, ic_profiles[ic_profiles["name"] == r.name],
                              cmap, norm, dashed=(r.kind == "borehole"), augmented=bool(r.augmented))
        if not r.augmented:
            mesh = m or mesh
        if top is not None:
            tops[r.number] = max(top, r.surface_elevation_m_nap)

    if len(drawn):
        offsets = _label_offsets(drawn["distance_along_m"].to_numpy(), 1.5 * width)
        for (r, dx) in zip(drawn.itertuples(), offsets):
            label = f"{r.number}={number_of.get(r.copy_of, '?')}" if r.augmented else str(r.number)
            ax.annotate(label, xy=(r.distance_along_m, tops.get(r.number, r.surface_elevation_m_nap)),
                        xytext=(dx, 12), textcoords="offset points", ha="center", va="bottom", fontsize=8, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25" if r.augmented else "circle,pad=0.25",
                                  facecolor="#eeeeee" if r.augmented else "white", edgecolor="black", linewidth=0.6),
                        arrowprops=dict(arrowstyle="-", linewidth=0.5, color="black"))

    xs = [0.0, line.length_m] + list(drawn["distance_along_m"])
    ax.set_xlim(min(xs) - 2 * width, max(xs) + 2 * width)
    y_top = max([*tops.values(), *(p["elev_nap"] if "elev_nap" in p else [])] or [0.0])
    y_bottom = min(np.nanmin(drawn["bottom_elevation_m_nap"]) if len(drawn) else y_top - 10, DESIGN_BOTTOM_M_NAP)
    ax.set_ylim(y_bottom - 1.0, y_top + 0.12 * (y_top - y_bottom) + 1.0)
    ax.axhline(DESIGN_BOTTOM_M_NAP, color="tab:blue", linestyle="--", linewidth=1.2, zorder=5)
    ax.annotate(f"design bottom {DESIGN_BOTTOM_M_NAP:g} m NAP", xy=(1.0, DESIGN_BOTTOM_M_NAP),
                xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color="tab:blue")
    if frame is not None:
        ax.add_patch(Rectangle((frame.left_m, frame.bottom_m_nap), frame.right_m - frame.left_m,
                               frame.top_m_nap - frame.bottom_m_nap, fill=False,
                               edgecolor="dimgray", linewidth=1.4, zorder=7))
        ax.annotate(f"SchemaGAN frame {frame.n_cols} x {frame.n_rows}", xy=(frame.right_m, frame.top_m_nap),
                    xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8, color="dimgray")
    for x_end, label in ((0.0, "A"), (line.length_m, "B")):
        ax.annotate(label, xy=(x_end, ax.get_ylim()[1]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Distance along line from A (m)")
    ax.set_ylabel("Elevation (m NAP)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.075, fraction=0.02, extend="both", ticks=ROBERTSON_THRESHOLDS)
        cbar.ax.set_title("Ic (-)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        for lower, upper, label in ROBERTSON_ZONES:
            cbar.ax.text(-0.3, (lower + upper) / 2.0, label, transform=cbar.ax.get_yaxis_transform(),
                         va="center", ha="right", fontsize=7)

    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", linestyle="-", label="CPT (measured Ic)"),
        Patch(facecolor="white", edgecolor="black", linestyle="--", label="borehole (synthetic Ic, seed 42)"),
        *([Patch(facecolor="white", edgecolor="dimgray", hatch="////", label="augmented copy (label n=original)")]
          if drawn["augmented"].any() else []),
        Line2D([], [], color="black", linewidth=1.2, label="ground surface (profile)"),
        Line2D([], [], color="tab:red", linewidth=1.2, linestyle="--", label="ground surface, interpolated over gaps"),
        Line2D([], [], color="tab:blue", linewidth=1.2, linestyle="--", label=f"design bottom ({DESIGN_BOTTOM_M_NAP:g} m NAP)"),
        *([Patch(fill=False, edgecolor="dimgray", linewidth=1.4, label=f"SchemaGAN frame ({frame.n_cols} x {frame.n_rows})")]
          if frame is not None else []),
    ], loc="lower right", fontsize=8, framealpha=0.9)

    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    exaggeration = ((x1 - x0) / bbox.width) / ((y1 - y0) / bbox.height)
    ax.set_title(f"{title}\nvertical exaggeration ~{exaggeration:.1f}x; column width {width:.1f} m (display only)")

    ax_key.axis("off")
    ax_key.text(0.0, 1.0, "\n".join(key_lines), family="monospace", fontsize=8, va="top", ha="left",
                transform=ax_key.transAxes)
    if output_path is not None:
        save_png(fig, output_path)
    return fig


def plot_matrix_preview(matrix: np.ndarray, frame: SchemaganFrame, assignments: pd.DataFrame, title: str,
                        output_path: Optional[Path] = None):
    """The SchemaGAN input matrix as an image: zeros white, Ic in the cross-section colours, used columns labelled."""
    fig, ax = plt.subplots(figsize=(16, 4.2))
    shown = np.ma.masked_equal(matrix, 0.0)
    mesh = ax.imshow(shown, cmap="viridis", norm=Normalize(IC_VMIN, IC_VMAX), aspect="auto",
                     interpolation="nearest", extent=(-0.5, frame.n_cols - 0.5, frame.n_rows - 0.5, -0.5))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("dimgray")
    used = assignments[assignments["used"]]
    for r in used.itertuples():
        label = f"{r.number}" + (f"\n(copy)" if r.augmented else "")
        ax.annotate(label, xy=(r.column, -0.5), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.5, frame.n_cols - 0.5)
    ax.set_xticks(np.arange(0, frame.n_cols + 1, 64))
    ax.set_yticks([0, 8, 16, 24, frame.n_rows - 1])
    ax.set_xlabel(f"column (0-{frame.n_cols - 1}); {frame.col_width_m:.3f} m per column, "
                  f"{frame.left_m:.1f}-{frame.right_m:.1f} m along the line")
    ax.set_ylabel(f"row\n(top of 0 = {frame.top_m_nap:.2f},\nbottom of {frame.n_rows - 1} = {frame.bottom_m_nap:.2f} m NAP)",
                  fontsize=9)
    fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.02, extend="both", ticks=ROBERTSON_THRESHOLDS).set_label("Ic (-)")
    ax.set_title(f"{title}\nwhite = 0 (no data); {len(used)} columns with data; labels = test numbers in the cross-section plot",
                 fontsize=10, pad=30)
    fig.tight_layout()
    if output_path is not None:
        save_png(fig, output_path)
    return fig


def save_png(fig, path: Path, attempts: int = 10, wait_s: float = 1.0):
    """Render to memory, then write; retries when the file is briefly blocked (e.g. open in an image viewer)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    data = buffer.getvalue()
    for attempt in range(1, attempts + 1):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return
        except OSError as exc:
            if attempt == attempts:
                raise OSError(f"could not write {path} after {attempts} attempts (is it open in another program?): {exc}")
            time.sleep(wait_s)
