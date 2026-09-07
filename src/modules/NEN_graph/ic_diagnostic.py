"""Diagnostics for the canonical class-anchored, piecewise-linear
representative-Ic surface (`modules.NEN_graph.ic_mapping.IcSurface`).

Read-only diagnostics only: this does not change `ic_mapping.py` or the
plotting model in `visualization.py`. Nothing here enforces any property
of the surface (envelopes, monotonic trends, etc.) - it only samples the
surface and reports what it finds, honestly, including violations.
"""

import csv
import math
import sys
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import classify_composition, get_polygons  # noqa: E402
from modules.NEN_graph.coordinates import SQRT3_OVER_2, ternary_to_cartesian  # noqa: E402
from modules.NEN_graph.ic_mapping import (  # noqa: E402
    IC_CLASS_ENVELOPES,
    IC_CLASS_TARGETS,
    calculate_mean_ic,
    get_ic_surface,
)

TOLERANCE = 1e-6  # allowed numerical slack for the pct_outside_envelope check
ROBERTSON_LEVELS = (2.05, 2.60, 2.95)


def sample_grid(step: int = 1):
    """Yield every (sand, silt, lutum) integer-percent point on the ternary grid."""
    for sand in range(0, 101, step):
        for silt in range(0, 101 - sand, step):
            yield sand, silt, 100 - sand - silt


def _robertson_zone_index(value: float) -> int:
    """Bucket `value` against `ROBERTSON_LEVELS`: 0 = below all, 3 = above all."""
    lo, mid, hi = ROBERTSON_LEVELS
    if value < lo:
        return 0
    if value < mid:
        return 1
    if value < hi:
        return 2
    return 3


# --- per-class summary --------------------------------------------------


def _build_summary_rows(per_polygon: dict, surface) -> list:
    """Per-class sampled Ic stats against `IC_CLASS_TARGETS`/`IC_CLASS_ENVELOPES`."""
    rep_xy_by_class = {
        c["classes"][0]: c["xy"] for c in surface.controls if c["control_type"] == "class_representative"
    }

    rows = []
    for code in sorted(per_polygon):
        values = per_polygon[code]
        n = len(values)

        envelope = IC_CLASS_ENVELOPES[code]
        ic_min, ic_max = envelope["ic_min"], envelope["ic_max"]
        n_outside = sum(1 for v in values if v < ic_min - TOLERANCE or v > ic_max + TOLERANCE)
        worst_departure = max((max(ic_min - v, v - ic_max, 0.0) for v in values), default=0.0)

        zone_counts = [0, 0, 0, 0]
        for v in values:
            zone_counts[_robertson_zone_index(v)] += 1

        evaluated_at_representative_point = surface.value_at_cartesian(*rep_xy_by_class[code])

        rows.append({
            "class_code": code,
            "n_points": n,
            "target_ic": IC_CLASS_TARGETS[code],
            "evaluated_at_representative_point": evaluated_at_representative_point,
            "envelope_ic_min": ic_min,
            "envelope_ic_max": ic_max,
            "sampled_ic_min": min(values),
            "sampled_ic_median": median(values),
            "sampled_ic_max": max(values),
            "pct_outside_envelope": 100.0 * n_outside / n,
            "worst_envelope_departure": worst_departure,
            "frac_zone_lt_2.05": zone_counts[0] / n,
            "frac_zone_2.05_2.60": zone_counts[1] / n,
            "frac_zone_2.60_2.95": zone_counts[2] / n,
            "frac_zone_gt_2.95": zone_counts[3] / n,
        })
    return rows


def _write_summary_csv(rows: list, path: Path):
    fieldnames = [
        "class_code", "n_points", "target_ic", "evaluated_at_representative_point",
        "envelope_ic_min", "envelope_ic_max",
        "sampled_ic_min", "sampled_ic_median", "sampled_ic_max",
        "pct_outside_envelope", "worst_envelope_departure",
        "frac_zone_lt_2.05", "frac_zone_2.05_2.60", "frac_zone_2.60_2.95", "frac_zone_gt_2.95",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_control_point_csv(surface, path: Path):
    """One row per control point: location, assigned Ic, type, and contributing classes."""
    fieldnames = ["x", "y", "ic", "control_type", "classes", "composition", "is_fallback"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in surface.controls:
            writer.writerow({
                "x": c["xy"][0],
                "y": c["xy"][1],
                "ic": c["ic"],
                "control_type": c["control_type"],
                "classes": ";".join(c["classes"]) if c["classes"] else "",
                "composition": c["composition"] if c["composition"] is not None else "",
                "is_fallback": c["is_fallback"],
            })


# --- validation checks ---------------------------------------------------


def check_domain_coverage(surface) -> dict:
    """Confirm the control points' convex hull covers the full outer triangle."""
    from shapely.geometry import MultiPoint

    hull = MultiPoint(surface.triangulation.points).convex_hull
    expected_area = 0.5 * SQRT3_OVER_2  # base = 1, height = sqrt(3)/2
    return {
        "hull_area": hull.area,
        "expected_area": expected_area,
        "covers_full_triangle": abs(hull.area - expected_area) < 1e-9,
    }


def check_control_point_reproduction(surface) -> dict:
    """Re-evaluate the surface at every control point; it should reproduce the assigned value exactly."""
    max_error = 0.0
    for c in surface.controls:
        value = surface.value_at_cartesian(*c["xy"])
        if value is None:
            max_error = float("inf")
            continue
        max_error = max(max_error, abs(value - c["ic"]))
    return {"max_reproduction_error": max_error}


def check_edge_continuity(surface) -> dict:
    """Confirm every internal simplex edge gives the same value from either adjacent simplex."""
    tri = surface.triangulation
    edges: dict = {}
    for simplex_id, simplex in enumerate(tri.simplices):
        for i in range(3):
            edge = tuple(sorted((int(simplex[i]), int(simplex[(i + 1) % 3]))))
            edges.setdefault(edge, []).append(simplex_id)

    def _value_via_simplex(simplex_id, xy):
        transform = tri.transform[simplex_id]
        delta = np.array(xy) - transform[2]
        bary_ab = transform[:2].dot(delta)
        bary = np.array([bary_ab[0], bary_ab[1], 1.0 - bary_ab.sum()])
        vertex_ids = tri.simplices[simplex_id]
        return float(bary.dot(surface.values[vertex_ids]))

    max_error = 0.0
    n_checked = 0
    for edge, simplex_ids in edges.items():
        if len(simplex_ids) != 2:
            continue  # boundary edge, only one adjacent simplex - nothing to compare
        p0, p1 = tri.points[edge[0]], tri.points[edge[1]]
        midpoint = (p0 + p1) / 2.0
        value_a = _value_via_simplex(simplex_ids[0], midpoint)
        value_b = _value_via_simplex(simplex_ids[1], midpoint)
        max_error = max(max_error, abs(value_a - value_b))
        n_checked += 1
    return {"n_edges_checked": n_checked, "max_continuity_error": max_error}


def check_finite_values(surface, step: int = 1) -> dict:
    """Sample the full grid and confirm every in-domain point returns a finite Ic value."""
    n_checked = 0
    n_nonfinite = 0
    for sand, silt, lutum in sample_grid(step):
        value = calculate_mean_ic({"sand": sand, "silt": silt, "lutum": lutum}, surface=surface)
        if value is None:
            continue
        n_checked += 1
        if not math.isfinite(value):
            n_nonfinite += 1
    return {"n_checked": n_checked, "n_nonfinite": n_nonfinite}


def _reversals_along_line(values: list, threshold: float) -> list:
    """Diffs that go against the line's overall (start-to-end) direction, i.e. local reversals."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return []
    direction = 1.0 if valid[-1] - valid[0] >= 0 else -1.0
    reversals = []
    prev = None
    for v in values:
        if v is None:
            continue
        if prev is not None:
            delta = v - prev
            if direction * delta < -threshold:
                reversals.append(delta)
        prev = v
    return reversals


def check_composition_trends(surface, step: int = 1, reversal_threshold: float = 1e-9) -> dict:
    """Report (do not enforce) local reversals along two proposed composition trends.

    - "sand_to_silt_at_fixed_lutum": walking sand -> silt at each fixed lutum row.
    - "sand_to_lutum_at_fixed_silt": walking sand -> lutum at each fixed silt row.
    """
    reversals_a = []
    for lutum in range(0, 101, step):
        line = [
            calculate_mean_ic({"sand": 100 - silt - lutum, "silt": silt, "lutum": lutum}, surface=surface)
            for silt in range(0, 101 - lutum, step)
        ]
        reversals_a.extend(_reversals_along_line(line, reversal_threshold))

    reversals_b = []
    for silt in range(0, 101, step):
        line = [
            calculate_mean_ic({"sand": 100 - silt - lutum, "silt": silt, "lutum": lutum}, surface=surface)
            for lutum in range(0, 101 - silt, step)
        ]
        reversals_b.extend(_reversals_along_line(line, reversal_threshold))

    return {
        "sand_to_silt_at_fixed_lutum": {
            "n_reversals": len(reversals_a),
            "max_reversal_magnitude": max((abs(d) for d in reversals_a), default=0.0),
        },
        "sand_to_lutum_at_fixed_silt": {
            "n_reversals": len(reversals_b),
            "max_reversal_magnitude": max((abs(d) for d in reversals_b), default=0.0),
        },
    }


# --- static figures --------------------------------------------------------


def _draw_outer_triangle(ax):
    outer = [(0.0, 0.0), (1.0, 0.0), (0.5, SQRT3_OVER_2), (0.0, 0.0)]
    ax.plot([p[0] for p in outer], [p[1] for p in outer], color="black", linewidth=1.2)


def _draw_polygon_boundaries(ax, polygons):
    for code, poly in polygons.items():
        verts = poly.cartesian_vertices + [poly.cartesian_vertices[0]]
        ax.plot([v[0] for v in verts], [v[1] for v in verts], color="black", linewidth=0.8)
        cx, cy = poly.shapely_polygon.representative_point().coords[0]
        ax.text(cx, cy, code, ha="center", va="center", fontsize=7, fontweight="bold", clip_on=True)


def plot_ic_diagnostic_figure(xs, ys, ic_values, output_path: Path = None) -> Path:
    """Single triangle: the canonical representative-Ic field, Robertson-zone contours, and NEN boundaries."""
    polygons = get_polygons()
    triang = mtri.Triangulation(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.tricontourf(triang, ic_values, levels=100, cmap="turbo")
    fig.colorbar(contour, ax=ax, shrink=0.8, label="Representative Ic")
    ax.tricontour(
        triang, ic_values, levels=list(ROBERTSON_LEVELS),
        colors="black", linewidths=0.8, linestyles="dashed",
    )
    _draw_outer_triangle(ax)
    _draw_polygon_boundaries(ax, polygons)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3_OVER_2 + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("NEN 5104 - class-anchored representative Ic field", fontsize=12)
    fig.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent / "ic_diagnostic_figure.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_control_point_diagnostic_figure(surface, output_path: Path = None) -> Path:
    """Optional inspection figure: every control point (by type) plus the Delaunay triangulation edges."""
    polygons = get_polygons()
    tri = surface.triangulation

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices, color="lightgray", linewidth=0.6, zorder=1)

    colors = {"class_representative": "tab:blue", "perimeter_anchor": "tab:orange", "corner": "tab:red"}
    markers = {"class_representative": "o", "perimeter_anchor": "s", "corner": "^"}
    for control_type in ("perimeter_anchor", "class_representative", "corner"):
        points = [c["xy"] for c in surface.controls if c["control_type"] == control_type]
        if not points:
            continue
        ax.scatter(
            [p[0] for p in points], [p[1] for p in points],
            c=colors[control_type], marker=markers[control_type], s=30, zorder=3,
            label=f"{control_type} (n={len(points)})", edgecolors="black", linewidths=0.4,
        )

    _draw_outer_triangle(ax)
    _draw_polygon_boundaries(ax, polygons)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3_OVER_2 + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Ic surface construction - control points and Delaunay triangulation", fontsize=11)
    fig.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent / "ic_control_point_diagnostic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- entry point -----------------------------------------------------------


def run_diagnostic(
    csv_path: Path = None,
    control_csv_path: Path = None,
    figure_path: Path = None,
    control_figure_path: Path = None,
) -> dict:
    """Sample the full 1% ternary grid, run all validation checks, and save the CSVs/figures.

    Points that don't classify to exactly one NEN polygon (on a shared
    boundary) are still included in the continuous field figure, but
    excluded from the per-class summary rows.

    Returns:
        dict: summary rows, output paths, validation check results, and
            any representative-point fallbacks.
    """
    surface = get_ic_surface()

    xs, ys, ic_values = [], [], []
    per_polygon: dict = {}

    for sand, silt, lutum in sample_grid():
        composition = {"sand": sand, "silt": silt, "lutum": lutum}
        ic = calculate_mean_ic(composition, surface=surface)
        x, y = ternary_to_cartesian(sand, silt, lutum)

        xs.append(x)
        ys.append(y)
        ic_values.append(ic if ic is not None else float("nan"))

        result = classify_composition(sand, silt, lutum)
        if result["status"] == "inside" and ic is not None:
            per_polygon.setdefault(result["class_code"], []).append(ic)

    summary_rows = _build_summary_rows(per_polygon, surface)
    if csv_path is None:
        csv_path = Path(__file__).parent / "ic_diagnostic_summary.csv"
    _write_summary_csv(summary_rows, csv_path)

    if control_csv_path is None:
        control_csv_path = Path(__file__).parent / "ic_control_points.csv"
    _write_control_point_csv(surface, control_csv_path)

    figure_path = plot_ic_diagnostic_figure(xs, ys, ic_values, output_path=figure_path)
    control_figure_path = plot_control_point_diagnostic_figure(surface, output_path=control_figure_path)

    checks = {
        "domain_coverage": check_domain_coverage(surface),
        "control_point_reproduction": check_control_point_reproduction(surface),
        "edge_continuity": check_edge_continuity(surface),
        "finite_values": check_finite_values(surface),
        "composition_trends": check_composition_trends(surface),
    }

    return {
        "summary_rows": summary_rows,
        "csv_path": csv_path,
        "control_csv_path": control_csv_path,
        "figure_path": figure_path,
        "control_figure_path": control_figure_path,
        "checks": checks,
        "representative_point_fallbacks": surface.representative_point_fallbacks,
    }


if __name__ == "__main__":
    result = run_diagnostic()

    print(f"Saved per-class summary CSV to {result['csv_path']}")
    print(f"Saved control-point CSV to {result['control_csv_path']}")
    print(f"Saved diagnostic figure to {result['figure_path']}")
    print(f"Saved control-point construction figure to {result['control_figure_path']}")
    print()

    if result["representative_point_fallbacks"]:
        print(f"Representative-point fallbacks (centroid outside polygon): {result['representative_point_fallbacks']}")
    else:
        print("Representative-point fallbacks: none (every class centroid was strictly inside its polygon)")
    print()

    checks = result["checks"]
    print("Validation checks:")
    dc = checks["domain_coverage"]
    print(
        f"  - domain coverage: hull area {dc['hull_area']:.6f} vs expected {dc['expected_area']:.6f} "
        f"(covers full triangle: {dc['covers_full_triangle']})"
    )
    print(f"  - control-point reproduction: max error {checks['control_point_reproduction']['max_reproduction_error']:.3e}")
    ec = checks["edge_continuity"]
    print(f"  - edge continuity: {ec['n_edges_checked']} shared edges checked, max discontinuity {ec['max_continuity_error']:.3e}")
    fv = checks["finite_values"]
    print(f"  - finite values: {fv['n_checked']} in-domain points sampled, {fv['n_nonfinite']} non-finite")
    for name, report in checks["composition_trends"].items():
        print(f"  - trend '{name}': {report['n_reversals']} local reversals, max magnitude {report['max_reversal_magnitude']:.4f}")
    print()

    print(
        f"{'class':<6}{'n':>6}{'target':>9}{'at_rep':>9}{'env_min':>9}{'env_max':>9}"
        f"{'smp_min':>9}{'smp_med':>9}{'smp_max':>9}{'pct_out':>9}"
    )
    for row in result["summary_rows"]:
        print(
            f"{row['class_code']:<6}{row['n_points']:>6}"
            f"{row['target_ic']:>9.2f}{row['evaluated_at_representative_point']:>9.2f}"
            f"{row['envelope_ic_min']:>9.2f}{row['envelope_ic_max']:>9.2f}"
            f"{row['sampled_ic_min']:>9.2f}{row['sampled_ic_median']:>9.2f}{row['sampled_ic_max']:>9.2f}"
            f"{row['pct_outside_envelope']:>9.2f}"
        )
