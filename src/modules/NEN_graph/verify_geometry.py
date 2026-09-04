"""Manual verification script for the NEN 5104 triangle geometry.

Run directly (no pytest required):
    python src/modules/NEN_graph/verify_geometry.py

Checks vertex validity, polygon validity/overlap/coverage, coordinate
round-trips, and representative classification results, printing a
PASS/FAIL summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import (  # noqa: E402
    DEFAULT_TOLERANCE,
    classify_composition,
    get_polygons,
    load_triangle_data,
)
from modules.NEN_graph.coordinates import cartesian_to_ternary, ternary_to_cartesian  # noqa: E402

FAILURES = []


def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAILURES.append(message)
    print(f"[{status}] {message}")


def main():
    data = load_triangle_data()
    polygons = get_polygons()

    # 1. Every vertex has three valid, non-negative percentages summing to 100.
    for poly_def in data["polygons"]:
        for v in poly_def["vertices"]:
            values = (v["sand"], v["silt"], v["lutum"])
            check(all(x >= 0 for x in values), f"{poly_def['code']} vertex {v} non-negative")
            total = sum(values)
            check(
                abs(total - 100) <= 1e-6,
                f"{poly_def['code']} vertex {v} sums to 100 (got {total})",
            )

    # 2. Every polygon is geometrically valid (no self-intersection).
    for code, poly in polygons.items():
        check(poly.shapely_polygon.is_valid, f"{code} polygon is valid (no self-intersection)")

    # 3 & 4. No two polygon interiors overlap (only touch on shared boundaries).
    codes = list(polygons.keys())
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = polygons[codes[i]].shapely_polygon, polygons[codes[j]].shapely_polygon
            overlap_area = a.intersection(b).area
            check(
                overlap_area <= 1e-9,
                f"{codes[i]} / {codes[j]} do not overlap (overlap area={overlap_area:.2e})",
            )

    # 6. Coverage: total polygon area should equal the outer triangle area.
    outer_area = (3 ** 0.5) / 4.0  # equilateral triangle, side length 1
    total_area = sum(p.shapely_polygon.area for p in polygons.values())
    check(
        abs(total_area - outer_area) <= 1e-6,
        f"polygons cover the full triangle (total={total_area:.6f}, expected={outer_area:.6f})",
    )

    # 7. ternary -> cartesian -> ternary round trip.
    samples = [(100, 0, 0), (0, 100, 0), (0, 0, 100), (33.3, 33.3, 33.4), (17.5, 42.5, 40)]
    for sand, silt, lutum in samples:
        x, y = ternary_to_cartesian(sand, silt, lutum)
        rs, ri, rl = cartesian_to_ternary(x, y)
        ok = all(abs(a - b) < 1e-9 for a, b in zip((sand, silt, lutum), (rs, ri, rl)))
        check(ok, f"round-trip for ({sand}, {silt}, {lutum}) -> ({rs:.6f}, {ri:.6f}, {rl:.6f})")

    # 8. Representative point inside every polygon returns the expected class.
    for code, poly in polygons.items():
        rx, ry = poly.shapely_polygon.representative_point().coords[0]
        sand, silt, lutum = cartesian_to_ternary(rx, ry)
        result = classify_composition(sand, silt, lutum)
        check(
            result["status"] == "inside" and result["class_code"] == code,
            f"representative point of {code} classifies as {code} (got {result})",
        )

    # 9. Points on known shared edges return on_boundary with both candidates.
    boundary_cases = [
        ((25, 25, 50), {"Ks1", "Ks2"}),  # midpoint of the Ks1/Ks2 edge
        ((50, 42, 8), {"Ks4", "Kz3", "Lz3", "Zs4"}),  # 4-way junction (post Zk/Zs3/Zs4 rework)
        ((50, 25, 25), {"Ks3", "Ks4", "Kz1"}),  # 3-way junction (post Ks4/Kz1 rework)
    ]
    for (sand, silt, lutum), expected in boundary_cases:
        result = classify_composition(sand, silt, lutum)
        got = set(result.get("candidate_classes", []))
        check(
            result["status"] == "on_boundary" and expected.issubset(got),
            f"({sand},{silt},{lutum}) on_boundary with candidates including {expected} (got {result})",
        )

    # 10. Invalid compositions are rejected with clear errors.
    invalid_cases = [
        (-5, 50, 55),
        (40, 40, 30),
        (float("nan"), 50, 50),
    ]
    for sand, silt, lutum in invalid_cases:
        result = classify_composition(sand, silt, lutum)
        check(
            result["status"] == "invalid" and result.get("errors"),
            f"({sand},{silt},{lutum}) rejected as invalid (got {result})",
        )

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
