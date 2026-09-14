"""Manual verification script for the organic stof-lutum-silt+zand (peat/veen)
triangle geometry (geometry/classification review stage - no Ic yet).

Run directly (no pytest required):
    python src/modules/NEN_graph/verify_peat_geometry.py

Checks vertex validity/coverage/non-overlap, representative-point and
boundary classification, the organic_matter=0 "no modifier" edge case, and
prints a compact vertex table (composition percentages) for review against
Figuur 3 / Tabel 3, followed by a PASS/FAIL summary. Also re-runs the
original triangle's verification to confirm it is unaffected.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from shapely.geometry import Point  # noqa: E402

from modules.NEN_graph.classification import get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import cartesian_to_composition, composition_to_cartesian  # noqa: E402
from modules.NEN_graph.peat_classification import (  # noqa: E402
    MAIN_NAME_CLASSES,
    MODIFIER_CLASSES,
    classify_peat_composition,
    get_peat_triangle,
)

FAILURES = []


def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(message)
    print(f"[{status}] {message}")


def main():
    triangle = get_peat_triangle()
    polygons = get_triangle_polygons(triangle)
    keys = triangle.component_keys

    print("=== 1. Original lutum-silt-zand chart is unaffected ===")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_geometry.py")],
        capture_output=True, text=True,
    )
    check(result.returncode == 0, "original verify_geometry.py still passes unchanged")
    print()

    print("=== 2. Vertex validity (non-negative, sums to 100) ===")
    for poly_def in triangle.polygons:
        for v in poly_def.vertices:
            values = [v[k] for k in keys]
            check(all(x >= 0 for x in values), f"{poly_def.code} vertex {v} non-negative")
            total = sum(values)
            check(abs(total - 100) <= 1e-6, f"{poly_def.code} vertex {v} sums to 100 (got {total})")
    print()

    print("=== 3. Polygon validity, non-overlap, coverage ===")
    check(len(polygons) == 8, f"8 polygons loaded (got {len(polygons)})")
    for code, poly in polygons.items():
        check(poly.shapely_polygon.is_valid, f"{code} polygon is valid (no self-intersection)")

    codes = list(polygons.keys())
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = polygons[codes[i]].shapely_polygon, polygons[codes[j]].shapely_polygon
            overlap_area = a.intersection(b).area
            # Vertices are rounded to 4 decimal places (percentage units), so shared
            # edges between adjacent polygons can show a negligible rounding-induced
            # sliver; 1e-6 matches the sum-to-100 tolerance used elsewhere and is far
            # below any geometrically meaningful overlap (outer triangle area ~0.433).
            check(overlap_area <= 1e-6, f"{codes[i]} / {codes[j]} do not overlap (area={overlap_area:.2e})")

    outer_area = (3 ** 0.5) / 4.0
    total_area = sum(p.shapely_polygon.area for p in polygons.values())
    check(
        abs(total_area - outer_area) <= 1e-6,
        f"8 polygons exactly tile the outer triangle (total={total_area:.9f}, expected={outer_area:.9f})",
    )
    print()

    print("=== 4. Shared boundaries align ===")
    shared_edges = [
        ("Vm", "Vz1", {"organic_matter": 35, "lutum": 0, "silt_zand": 65}),
        ("Vm", "Vk1", {"organic_matter": 70, "lutum": 30, "silt_zand": 0}),
        ("Vz1", "Vk1", {"organic_matter": 40.5488, "lutum": 4.7561, "silt_zand": 54.6951}),
        ("Vz1", "Vz3", {"organic_matter": 22.5, "lutum": 0, "silt_zand": 77.5}),
        ("Vk1", "Vk3", {"organic_matter": 45, "lutum": 55, "silt_zand": 0}),
        ("Vz3", "Vk3", {"organic_matter": 24.956, "lutum": 6.0035, "silt_zand": 69.0405}),
        ("Vz3", "h3", {"organic_matter": 15, "lutum": 0, "silt_zand": 85}),
        ("Vk3", "h3", {"organic_matter": 30, "lutum": 70, "silt_zand": 0}),
        ("h3", "h2", {"organic_matter": 8, "lutum": 0, "silt_zand": 92}),
        ("h2", "h1", {"organic_matter": 2.5, "lutum": 0, "silt_zand": 97.5}),
    ]
    for code_a, code_b, comp in shared_edges:
        x, y = composition_to_cartesian(comp, triangle)
        point = Point(x, y)
        da = polygons[code_a].shapely_polygon.distance(point)
        db = polygons[code_b].shapely_polygon.distance(point)
        check(da <= 1e-9 and db <= 1e-9, f"{code_a}/{code_b} shared vertex {comp} touches both polygons ({da:.2e}, {db:.2e})")
    print()

    print("=== 5. Representative point of every region classifies correctly ===")
    for code, poly in polygons.items():
        rx, ry = poly.shapely_polygon.representative_point().coords[0]
        comp = cartesian_to_composition(rx, ry, triangle)
        result = classify_peat_composition(comp["organic_matter"], comp["lutum"], comp["silt_zand"])
        check(
            result["status"] == "inside" and result["class_code"] == code,
            f"representative point of {code} classifies as {code} (got {result['status']}/{result.get('class_code')})",
        )
        expected_main_name = code in MAIN_NAME_CLASSES
        check(
            result["supplies_main_name"] == expected_main_name,
            f"{code} supplies_main_name == {expected_main_name} (got {result['supplies_main_name']})",
        )
    print()

    print("=== 6. Known boundary points return on_boundary with expected candidates ===")
    boundary_cases = [
        ({"organic_matter": 40.5488, "lutum": 4.7561, "silt_zand": 54.6951}, {"Vz1", "Vk1"}),
        ({"organic_matter": 24.956, "lutum": 6.0035, "silt_zand": 69.0405}, {"Vz3", "Vk3"}),
        ({"organic_matter": 8, "lutum": 0, "silt_zand": 92}, {"h3", "h2"}),
    ]
    for comp, expected in boundary_cases:
        result = classify_peat_composition(comp["organic_matter"], comp["lutum"], comp["silt_zand"])
        got = set(result.get("candidate_classes", []))
        check(
            result["status"] == "on_boundary" and expected.issubset(got),
            f"{comp} on_boundary with candidates including {expected} (got {result})",
        )
    print()

    print("=== 7. organic_matter=0 edge case reports 'no_modifier', not h1 ===")
    for comp in [
        {"organic_matter": 0, "lutum": 50, "silt_zand": 50},
        {"organic_matter": 0, "lutum": 10, "silt_zand": 90},
        {"organic_matter": 0, "lutum": 0, "silt_zand": 100},
        {"organic_matter": 0, "lutum": 100, "silt_zand": 0},
    ]:
        result = classify_peat_composition(comp["organic_matter"], comp["lutum"], comp["silt_zand"])
        check(
            result["status"] == "no_modifier" and result["humus_modifier"] is None,
            f"{comp} classifies as no_modifier (got {result['status']}/{result.get('humus_modifier')})",
        )
    print()

    print("=== 8. h1/h2/h3 (organic_matter>0) report humus_modifier, not a main name ===")
    for code in MODIFIER_CLASSES:
        poly = polygons[code]
        rx, ry = poly.shapely_polygon.representative_point().coords[0]
        comp = cartesian_to_composition(rx, ry, triangle)
        result = classify_peat_composition(comp["organic_matter"], comp["lutum"], comp["silt_zand"])
        check(
            result["status"] == "inside" and result["humus_modifier"] == code and result["supplies_main_name"] is False,
            f"{code} representative point reports humus_modifier={code}, supplies_main_name=False (got {result})",
        )
    print()

    print("=== 9. Invalid compositions are rejected ===")
    for organic_matter, lutum, silt_zand in [(-5, 50, 55), (40, 40, 30), (float("nan"), 50, 50)]:
        result = classify_peat_composition(organic_matter, lutum, silt_zand)
        check(result["status"] == "invalid" and result.get("errors"), f"({organic_matter},{lutum},{silt_zand}) rejected as invalid (got {result})")
    print()

    print("=== 10. Region vertex table (composition percentages, for review against Figuur 3) ===")
    header = f"{'code':<5} {'main_group':<10} " + " ".join(f"{k:>14}" for k in keys)
    print(header)
    for poly_def in triangle.polygons:
        for v in poly_def.vertices:
            row = f"{poly_def.code:<5} {poly_def.main_group:<10} " + " ".join(f"{v[k]:>14.4f}" for k in keys)
            print(row)
    print()

    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
