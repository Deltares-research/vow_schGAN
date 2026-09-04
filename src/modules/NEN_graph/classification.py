"""Generic polygon geometry and composition classification for ternary triangles.

Numerical tolerance:
    All percentage comparisons (sum-to-total check, negative check, and
    boundary/vertex detection) use the same tolerance (default 1e-6, in
    the triangle's total units - percentage-points for the NEN triangle).
    This is tight enough to distinguish real class boundaries (which
    differ by at least 2 percentage points in the NEN dataset) while
    absorbing ordinary floating-point rounding.

Backward compatibility:
    `classify_composition(sand, silt, lutum, tolerance=...)`, `get_polygons()`,
    `load_polygons()`, `load_triangle_data()`, `NenPolygon` and `DATA_PATH` are
    kept working exactly as before - they are thin wrappers around the
    generic engine (`classify_composition_generic`, `ClassPolygon`,
    `get_triangle_polygons`, `triangle_definition.load_triangle_definition`)
    bound to the default NEN 5104 lutum-silt-zand triangle.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

from shapely.geometry import Point, Polygon

from .coordinates import composition_to_cartesian
from .triangle_definition import PolygonDefinition, TriangleDefinition, load_triangle_definition

logger = logging.getLogger(__name__)

NEN_DATA_PATH = Path(__file__).parent / "data" / "nen5104_lutum_silt_zand.json"
DATA_PATH = NEN_DATA_PATH  # backward-compatible alias for the old constant name
DEFAULT_TOLERANCE = 1e-6  # percentage points (NEN triangle's total is 100)


class ClassPolygon:
    """A single classification polygon for a generic ternary triangle."""

    def __init__(self, definition: PolygonDefinition, triangle: TriangleDefinition):
        self.code = definition.code
        self.dutch_name = definition.dutch_name
        self.english_name = definition.english_name
        self.main_group = definition.main_group
        self.vertices = definition.vertices
        self.source_note = definition.source_note
        self.cartesian_vertices = [composition_to_cartesian(v, triangle) for v in definition.vertices]
        self.shapely_polygon = Polygon(self.cartesian_vertices)


# Backward-compatible name for code that imports `NenPolygon` directly.
NenPolygon = ClassPolygon


def get_triangle_polygons(triangle: TriangleDefinition) -> Dict[str, ClassPolygon]:
    """Build (code -> ClassPolygon) for any loaded `TriangleDefinition`."""
    return {p.code: ClassPolygon(p, triangle) for p in triangle.polygons}


def classify_composition_generic(
    composition: Dict[str, float],
    triangle: TriangleDefinition,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Classify an arbitrary composition against any `TriangleDefinition`.

    Args:
        composition (dict): component key -> value.
        triangle (TriangleDefinition): the triangle's components/polygons.
        tolerance (float): tolerance (in the triangle's total units) used
            for the sum-to-total check, the negative-value check, and
            boundary detection.

    Returns:
        dict: one of
            {"status": "invalid", "errors": [...], "composition": {...}}
            {"status": "inside", "class_code": str, "class_name": str,
             "dutch_name": str, "composition": {...}}
            {"status": "on_boundary", "candidate_classes": [...], "composition": {...}}
            {"status": "unclassified", "candidate_classes": [], "composition": {...}}
    """
    keys = triangle.component_keys
    result_composition = dict(composition)

    try:
        values = {k: float(composition[k]) for k in keys}
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "invalid",
            "errors": [f"composition must provide numeric values for {keys} ({exc})"],
            "composition": result_composition,
        }

    errors = []
    if any(math.isnan(v) for v in values.values()):
        errors.append(f"{', '.join(keys)} must not be NaN")
    else:
        if any(v < -tolerance for v in values.values()):
            errors.append(f"{', '.join(keys)} must not be negative")
        total = sum(values.values())
        if abs(total - triangle.total) > tolerance:
            errors.append(f"{' + '.join(keys)} must equal {triangle.total} (got {total})")

    if errors:
        return {"status": "invalid", "errors": errors, "composition": result_composition}

    polygons = get_triangle_polygons(triangle)
    x, y = composition_to_cartesian(values, triangle)
    point = Point(x, y)
    cartesian_tolerance = tolerance / triangle.total

    candidates = [
        code
        for code, poly in polygons.items()
        if poly.shapely_polygon.distance(point) <= cartesian_tolerance
    ]

    if not candidates:
        return {
            "status": "unclassified",
            "candidate_classes": [],
            "composition": result_composition,
        }

    if len(candidates) == 1:
        poly = polygons[candidates[0]]
        return {
            "status": "inside",
            "class_code": poly.code,
            "class_name": poly.english_name,
            "dutch_name": poly.dutch_name,
            "composition": result_composition,
        }

    return {
        "status": "on_boundary",
        "candidate_classes": sorted(candidates),
        "composition": result_composition,
    }


# --- Backward-compatible NEN 5104 sand/silt/lutum API -----------------------


def load_triangle_data(path: Path = NEN_DATA_PATH) -> dict:
    """Backward-compatible: load the raw declarative triangle JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_polygons(path: Path = NEN_DATA_PATH) -> Dict[str, NenPolygon]:
    """Backward-compatible: load all NEN polygons keyed by class code."""
    triangle = load_triangle_definition(path)
    return get_triangle_polygons(triangle)


_POLYGONS_CACHE: Optional[Dict[str, NenPolygon]] = None
_NEN_TRIANGLE_CACHE: Optional[TriangleDefinition] = None


def get_nen_triangle() -> TriangleDefinition:
    """Return the cached `TriangleDefinition` for the NEN 5104 triangle."""
    global _NEN_TRIANGLE_CACHE
    if _NEN_TRIANGLE_CACHE is None:
        _NEN_TRIANGLE_CACHE = load_triangle_definition(NEN_DATA_PATH)
    return _NEN_TRIANGLE_CACHE


def get_polygons() -> Dict[str, NenPolygon]:
    """Backward-compatible: return the cached dict of NEN polygons, loading it on first use."""
    global _POLYGONS_CACHE
    if _POLYGONS_CACHE is None:
        _POLYGONS_CACHE = get_triangle_polygons(get_nen_triangle())
    return _POLYGONS_CACHE


def classify_composition(
    sand: float,
    silt: float,
    lutum: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Backward-compatible: classify a sand/silt/lutum composition against the NEN 5104 triangle.

    Args:
        sand (float): sand percentage.
        silt (float): silt percentage.
        lutum (float): lutum percentage.
        tolerance (float): tolerance in percentage points used for the
            sum-to-100 check, the negative-value check, and boundary
            detection.

    Returns:
        dict: see `classify_composition_generic`.
    """
    composition = {"sand": sand, "silt": silt, "lutum": lutum}
    return classify_composition_generic(composition, get_nen_triangle(), tolerance)

