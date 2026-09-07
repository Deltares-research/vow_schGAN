"""Diagnostic: compare the legacy 3-anchor Mean Ic field against the new
class-constrained (piecewise-linear) Mean Ic field, per NEN 5104 polygon.

Read-only diagnostics only: this does not change `ic_mapping.py` or the
plotting model in `visualization.py`.

`calculate_ic_rw` (Robertson-Wride) is kept as a separate reference
calculation only - it is NOT used in the comparison below.
"""

import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import classify_composition, get_polygons  # noqa: E402
from modules.NEN_graph.coordinates import SQRT3_OVER_2, ternary_to_cartesian  # noqa: E402
from modules.NEN_graph.ic_mapping import (  # noqa: E402
    IC_CLASS_MAPPING,
    calculate_ic_rw,  # noqa: F401  (kept available as a separate reference calc, unused below)
    calculate_mean_ic_class_constrained,
    calculate_mean_ic_legacy_three_anchor,
    get_class_constrained_ic_surface,
    summarize_vertex_trend,
)

TOLERANCE = 1e-6  # allowed numerical slack for the pct_outside_target_range check


def sample_grid(step: int = 1):
    """Yield every (sand, silt, lutum) integer-percent point on the ternary grid."""
    for sand in range(0, 101, step):
        for silt in range(0, 101 - sand, step):
            yield sand, silt, 100 - sand - silt


def _build_summary_rows(per_polygon):
    """Aggregate per-polygon legacy/new Ic pairs into summary rows against IC_CLASS_MAPPING targets."""
    rows = []
    for code in sorted(per_polygon):
        pairs = per_polygon[code]
        legacy_values = [legacy for legacy, _new in pairs]
        new_values = [new for _legacy, new in pairs]
        n = len(pairs)

        target = IC_CLASS_MAPPING[code]
        ic_min, ic_max = target["ic_min"], target["ic_max"]
        n_outside = sum(1 for v in new_values if v < ic_min - TOLERANCE or v > ic_max + TOLERANCE)

        rows.append({
            "class_code": code,
            "n_points": n,
            "target_ic_min": ic_min,
            "target_ic_center": target["ic_center"],
            "target_ic_max": ic_max,
            "legacy_ic_min": min(legacy_values),
            "legacy_ic_median": median(legacy_values),
            "legacy_ic_max": max(legacy_values),
            "generated_ic_min": min(new_values),
            "generated_ic_median": median(new_values),
            "generated_ic_max": max(new_values),
            "pct_outside_target_range": 100.0 * n_outside / n,
        })
    return rows


def _write_csv(rows, path: Path):
    fieldnames = [
        "class_code", "n_points",
        "target_ic_min", "target_ic_center", "target_ic_max",
        "legacy_ic_min", "legacy_ic_median", "legacy_ic_max",
        "generated_ic_min", "generated_ic_median", "generated_ic_max",
        "pct_outside_target_range",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _draw_outer_triangle(ax):
    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2), (0.0, 0.0)]
    ax.plot([p[0] for p in outer], [p[1] for p in outer], color="black", linewidth=1.2)


def _draw_polygon_boundaries(ax, polygons):
    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        ax.plot([v[0] for v in verts], [v[1] for v in verts], color="black", linewidth=0.8)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=7, fontweight="bold")


def plot_ic_diagnostic_figure(xs, ys, legacy_ic_values, new_ic_values, output_path: Path = None) -> Path:
    """Three side-by-side triangles: legacy 3-anchor Ic, new class-constrained Ic, and their difference."""
    polygons = get_polygons()
    triang = mtri.Triangulation(xs, ys)

    ic_vmin = min(min(legacy_ic_values), min(new_ic_values))
    ic_vmax = max(max(legacy_ic_values), max(new_ic_values))
    diff_values = [new - legacy for legacy, new in zip(legacy_ic_values, new_ic_values)]
    diff_abs_max = max(abs(min(diff_values)), abs(max(diff_values)))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [
        (axes[0], legacy_ic_values, "viridis", ic_vmin, ic_vmax, "Legacy 3-anchor Ic"),
        (axes[1], new_ic_values, "viridis", ic_vmin, ic_vmax, "New class-constrained Ic"),
        (axes[2], diff_values, "RdBu_r", -diff_abs_max, diff_abs_max, "Difference (new \u2212 legacy)"),
    ]
    for ax, values, cmap, vmin, vmax, title in panels:
        contour = ax.tricontourf(triang, values, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(contour, ax=ax, shrink=0.8)
        _draw_outer_triangle(ax)
        _draw_polygon_boundaries(ax, polygons)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, SQRT3_OVER_2 + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig.suptitle("Ic field diagnostic - legacy vs. class-constrained - NEN 5104 lutum-silt-zand triangle", fontsize=13)
    fig.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent / "ic_diagnostic_figure.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_diagnostic(csv_path: Path = None, figure_path: Path = None):
    """Sample the full 1% ternary grid, build the per-polygon CSV summary, and save the figure.

    Points that don't classify to exactly one NEN polygon (on a shared
    boundary, or - not expected on this grid - unclassified) are still
    included in the continuous field figure, but excluded from the
    per-polygon summary rows.

    Returns:
        Tuple[list, Path, Path]: (summary_rows, csv_path, figure_path).
    """
    surface = get_class_constrained_ic_surface()

    xs, ys, legacy_ic_values, new_ic_values = [], [], [], []
    per_polygon = {}

    for sand, silt, lutum in sample_grid():
        composition = {"sand": sand, "silt": silt, "lutum": lutum}
        legacy_ic = calculate_mean_ic_legacy_three_anchor(composition)
        new_ic = calculate_mean_ic_class_constrained(composition, surface=surface)
        x, y = ternary_to_cartesian(sand, silt, lutum)

        xs.append(x)
        ys.append(y)
        legacy_ic_values.append(legacy_ic)
        new_ic_values.append(new_ic)

        result = classify_composition(sand, silt, lutum)
        if result["status"] == "inside" and new_ic is not None:
            per_polygon.setdefault(result["class_code"], []).append((legacy_ic, new_ic))

    summary_rows = _build_summary_rows(per_polygon)

    if csv_path is None:
        csv_path = Path(__file__).parent / "ic_diagnostic_summary.csv"
    _write_csv(summary_rows, csv_path)

    figure_path = plot_ic_diagnostic_figure(xs, ys, legacy_ic_values, new_ic_values, output_path=figure_path)

    return summary_rows, csv_path, figure_path


if __name__ == "__main__":
    summary, out_csv, out_fig = run_diagnostic()
    print(f"Saved summary CSV to {out_csv}")
    print(f"Saved diagnostic figure to {out_fig}")
    print()

    surface = get_class_constrained_ic_surface()
    print(f"Vertex range validation warnings: {len(surface.validation_warnings)}")
    for w in surface.validation_warnings:
        print(f"  - {w}")
    print(f"Vertex trend correlation (fines fraction vs. Ic): {summarize_vertex_trend():.3f}")
    print()

    print(
        f"{'class':<6}{'n':>6}{'tgt_min':>9}{'tgt_ctr':>9}{'tgt_max':>9}"
        f"{'gen_min':>9}{'gen_med':>9}{'gen_max':>9}{'pct_out':>9}"
    )
    for row in summary:
        print(
            f"{row['class_code']:<6}{row['n_points']:>6}"
            f"{row['target_ic_min']:>9.2f}{row['target_ic_center']:>9.2f}{row['target_ic_max']:>9.2f}"
            f"{row['generated_ic_min']:>9.2f}{row['generated_ic_median']:>9.2f}{row['generated_ic_max']:>9.2f}"
            f"{row['pct_outside_target_range']:>9.2f}"
        )

