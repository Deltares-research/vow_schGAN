"""Lightweight verification of the peat/veen representative-Ic surface (`peat_ic_mapping`).

Hand-rolled [PASS]/[FAIL] checks (no pytest), mirroring the convention used
by `verify_peat_geometry.py`. Any failing check is reported with the
specific conflicting values/points - targets, geometry, and tolerances are
never silently adjusted to force a pass.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

import numpy as np  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from modules.NEN_graph.classification import get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import SQRT3_OVER_2, composition_to_cartesian  # noqa: E402
from modules.NEN_graph.peat_classification import MODIFIER_CLASSES, get_peat_triangle  # noqa: E402
from modules.NEN_graph.peat_ic_mapping import (  # noqa: E402
    IC_PEAT_VERTEX_VALUES,
    MAPPED_CLASSES,
    PEAT_CLASS_ENVELOPES,
    PEAT_CLASS_TARGETS,
    get_peat_ic_surface,
)

COORD_TOL = 1e-9
CONTINUITY_TOL = 1e-6
GRADIENT_TOL = 1e-9
JS_AGREEMENT_TOL = 1e-9

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and status == 'FAIL' else ""))
    if not condition:
        failures.append(label)
    return condition


def js_style_value_at(x, y, field):
    """Python port of the exact `meanIcFromField()` JS function (see
    `visualization._build_hover_script`) - an independent barycentric
    evaluator over the same `to_js_field()` data, for cross-checking
    against `RestrictedIcSurface.value_at_cartesian` (which uses scipy's
    Delaunay `transform` machinery instead).
    """
    points, simplices, values = field["points"], field["simplices"], field["values"]
    best_val, best_score = None, -np.inf
    for s in simplices:
        a, b, c = points[s[0]], points[s[1]], points[s[2]]
        ax, ay, bx, by, cx, cy = a[0], a[1], b[0], b[1], c[0], c[1]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-15:
            continue
        wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom
        wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denom
        wc = 1 - wa - wb
        score = min(wa, wb, wc)
        if score > best_score:
            best_score = score
            best_val = wa * values[s[0]] + wb * values[s[1]] + wc * values[s[2]]
    if best_score < -1e-6:
        return None
    return best_val


def main():
    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)
    surface = get_peat_ic_surface()
    field = surface.to_js_field()

    print("=" * 70)
    print("1. Reproduction of the 10 boundary controls")
    print("=" * 70)
    for coords, expected in IC_PEAT_VERTEX_VALUES.items():
        vertex = dict(zip(triangle.component_keys, coords))
        x, y = composition_to_cartesian(vertex, triangle)
        actual = surface.value_at_cartesian(x, y)
        ok = actual is not None and abs(actual - expected) <= COORD_TOL
        check(
            f"boundary control {coords} -> {expected}",
            ok,
            f"got {actual}",
        )

    print()
    print("=" * 70)
    print("2. Reproduction of the 5 class-centroid targets")
    print("=" * 70)
    centroid_controls = {c["classes"][0]: c for c in surface.controls if c["control_type"] == "class_representative"}
    for code, target in PEAT_CLASS_TARGETS.items():
        ctrl = centroid_controls.get(code)
        ok = ctrl is not None
        if ok:
            x, y = ctrl["xy"]
            actual = surface.value_at_cartesian(x, y)
            ok = actual is not None and abs(actual - target) <= COORD_TOL
            check(f"class-centroid target {code} -> {target}", ok, f"got {actual} at {ctrl['xy']}")
        else:
            check(f"class-centroid target {code} -> {target}", False, "no class_representative control found")

    print()
    print("=" * 70)
    print("3. No NaNs/None inside the mapped peat domain")
    print("=" * 70)
    none_count = 0
    sampled = 0
    for code in MAPPED_CLASSES:
        poly = polygons[code].shapely_polygon
        minx, miny, maxx, maxy = poly.bounds
        n = 25
        for i in range(n):
            for j in range(n):
                x = minx + (maxx - minx) * (i + 0.5) / n
                y = miny + (maxy - miny) * (j + 0.5) / n
                if not poly.contains(Point(x, y)):
                    continue
                sampled += 1
                value = surface.value_at_cartesian(x, y)
                if value is None or (isinstance(value, float) and value != value):
                    none_count += 1
    check(
        f"no NaN/None over {sampled} interior samples across {len(MAPPED_CLASSES)} mapped classes",
        none_count == 0,
        f"{none_count} None/NaN samples found",
    )

    print()
    print("=" * 70)
    print("4. Continuity across every shared peat-class edge")
    print("=" * 70)
    shared_edges = [
        ("Vm", "Vz1", (35.0, 0.0, 65.0)),
        ("Vm", "Vk1", (70.0, 30.0, 0.0)),
        ("Vz1", "Vk1", (40.5488, 4.7561, 54.6951)),
        ("Vz1", "Vz3", (22.5, 0.0, 77.5)),
        ("Vk1", "Vk3", (45.0, 55.0, 0.0)),
        ("Vz3", "Vk3", (24.956, 6.0035, 69.0405)),
    ]
    for code_a, code_b, vertex_coords in shared_edges:
        vertex = dict(zip(("organic_matter", "lutum", "silt_zand"), vertex_coords))
        x, y = composition_to_cartesian(vertex, triangle)
        val_at_vertex = surface.value_at_cartesian(x, y)
        # sample slightly to either side of the shared vertex, still within each polygon
        poly_a = polygons[code_a].shapely_polygon
        poly_b = polygons[code_b].shapely_polygon
        cx_a, cy_a = poly_a.centroid.x, poly_a.centroid.y
        cx_b, cy_b = poly_b.centroid.x, poly_b.centroid.y
        eps = 1e-4
        xa, ya = x + eps * (cx_a - x), y + eps * (cy_a - y)
        xb, yb = x + eps * (cx_b - x), y + eps * (cy_b - y)
        val_a = surface.value_at_cartesian(xa, ya)
        val_b = surface.value_at_cartesian(xb, yb)
        ok = val_at_vertex is not None and val_a is not None and val_b is not None
        if ok:
            ok = abs(val_a - val_at_vertex) <= 0.01 and abs(val_b - val_at_vertex) <= 0.01
        check(
            f"continuity at {code_a}/{code_b} shared vertex {vertex_coords}",
            ok,
            f"vertex={val_at_vertex}, {code_a}-side={val_a}, {code_b}-side={val_b}",
        )

    print()
    print("=" * 70)
    print("5. Per-class target [min, centre, max] vs generated [min, median, max]")
    print("=" * 70)
    print(f"{'class':<6}{'target_min':>11}{'target_ctr':>11}{'target_max':>11}"
          f"{'gen_min':>11}{'gen_median':>12}{'gen_max':>11}")
    class_range_violations = []
    for code in MAPPED_CLASSES:
        env = PEAT_CLASS_ENVELOPES[code]
        poly = polygons[code].shapely_polygon
        minx, miny, maxx, maxy = poly.bounds
        n = 60
        values = []
        for i in range(n):
            for j in range(n):
                x = minx + (maxx - minx) * (i + 0.5) / n
                y = miny + (maxy - miny) * (j + 0.5) / n
                if not poly.contains(Point(x, y)):
                    continue
                v = surface.value_at_cartesian(x, y)
                if v is not None:
                    values.append(v)
        gen_min, gen_max = min(values), max(values)
        gen_median = statistics.median(values)
        print(f"{code:<6}{env['ic_min']:>11.2f}{env['ic_centre']:>11.2f}{env['ic_max']:>11.2f}"
              f"{gen_min:>11.3f}{gen_median:>12.3f}{gen_max:>11.3f}")
        if gen_min < env["ic_min"] - 1e-6 or gen_max > env["ic_max"] + 1e-6:
            class_range_violations.append((code, gen_min, gen_max, env["ic_min"], env["ic_max"]))

    print()
    print("=" * 70)
    print("6. Class-range violations")
    print("=" * 70)
    check(
        "no class generates Ic outside its own [ic_min, ic_max] envelope",
        len(class_range_violations) == 0,
        "; ".join(
            f"{c}: generated [{gmin:.3f},{gmax:.3f}] vs target [{tmin},{tmax}]"
            for c, gmin, gmax, tmin, tmax in class_range_violations
        ),
    )
    check(
        "no boundary-control envelope conflicts flagged during surface construction",
        len(surface.boundary_conflicts) == 0,
        str(surface.boundary_conflicts),
    )

    print()
    print("=" * 70)
    print("7. Agreement between Python (scipy Delaunay) and JS-logic (ported) evaluation")
    print("=" * 70)
    rep_points = [c["xy"] for c in surface.controls]
    rng = np.random.default_rng(42)
    for code in MAPPED_CLASSES:
        poly = polygons[code].shapely_polygon
        minx, miny, maxx, maxy = poly.bounds
        tries = 0
        added = 0
        while added < 10 and tries < 500:
            tries += 1
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                rep_points.append((x, y))
                added += 1

    max_diff = 0.0
    mismatches = []
    for x, y in rep_points:
        py_val = surface.value_at_cartesian(x, y)
        js_val = js_style_value_at(x, y, field)
        if py_val is None and js_val is None:
            continue
        if py_val is None or js_val is None:
            mismatches.append((x, y, py_val, js_val))
            continue
        diff = abs(py_val - js_val)
        max_diff = max(max_diff, diff)
        if diff > JS_AGREEMENT_TOL:
            mismatches.append((x, y, py_val, js_val))
    check(
        f"Python vs JS-logic agreement at {len(rep_points)} representative points (max diff {max_diff:.2e})",
        len(mismatches) == 0,
        str(mismatches[:5]),
    )

    print()
    print("=" * 70)
    print("8. Non-numerical hover for h1-h3 (Ic comes from the main-name triangle)")
    print("=" * 70)
    for code in MODIFIER_CLASSES:
        poly = polygons[code].shapely_polygon
        cx, cy = poly.representative_point().x, poly.representative_point().y
        value = surface.value_at_cartesian(cx, cy)
        check(f"{code} representative point has no numerical Ic (outside mapped surface)", value is None, f"got {value}")

    html_path = Path(__file__).parent / "nen5104_triangle_peat_ic_interactive.html"
    if html_path.exists():
        html_text = html_path.read_text(encoding="utf-8")
        check(
            "'Ic from main-name triangle' text embedded in the interactive HTML",
            "Ic from main-name triangle" in html_text,
        )
    else:
        check("interactive HTML exists to check hover text", False, f"{html_path} not found")

    print()
    print("=" * 70)
    print("9. Gradient check: replacing lutum OR silt+zand with organic matter must not decrease Ic")
    print("=" * 70)
    points = surface.triangulation.points
    simplices = surface.triangulation.simplices
    values = surface.values
    grad_violations = []
    for s in simplices:
        (x1, y1), (x2, y2), (x3, y3) = points[s[0]], points[s[1]], points[s[2]]
        v1, v2, v3 = values[s[0]], values[s[1]], values[s[2]]
        m = np.array([[x1, y1, 1.0], [x2, y2, 1.0], [x3, y3, 1.0]])
        rhs = np.array([v1, v2, v3])
        try:
            a_coef, b_coef, _c_coef = np.linalg.solve(m, rhs)
        except np.linalg.LinAlgError:
            continue  # degenerate (zero-area) simplex; no gradient to check
        grad_sz_to_om = 0.5 * a_coef + (SQRT3_OVER_2) * b_coef
        grad_lu_to_om = -0.5 * a_coef + (SQRT3_OVER_2) * b_coef
        if grad_sz_to_om < -GRADIENT_TOL:
            grad_violations.append(("silt_zand->organic_matter", tuple(s.tolist()), grad_sz_to_om))
        if grad_lu_to_om < -GRADIENT_TOL:
            grad_violations.append(("lutum->organic_matter", tuple(s.tolist()), grad_lu_to_om))
    check(
        f"no decreasing gradient in {len(simplices)} interpolation triangles",
        len(grad_violations) == 0,
        str(grad_violations[:10]),
    )

    print()
    print("=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
