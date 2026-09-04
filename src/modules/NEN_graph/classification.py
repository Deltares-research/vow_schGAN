"""Polygon geometry loading and composition classification for NEN 5104.

Numerical tolerance:
    All percentage comparisons (sum-to-100 check, negative check, and
    boundary/vertex detection) use the same tolerance (default 1e-6, in
    percentage-points). This is tight enough to distinguish real class
    boundaries (which differ by at least 2 percentage points in this
    dataset) while absorbing ordinary floating-point rounding.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from shapely.geometry import Point, Polygon

from .coordinates import ternary_to_cartesian

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "nen5104_triangle.json"
DEFAULT_TOLERANCE = 1e-6  # percentage points


class NenPolygon:
    """A single NEN 5104 classification polygon."""

    def __init__(
        self,
        code: str,
        dutch_name: str,
        english_name: str,
        main_group: str,
        vertices: List[dict],
        source_note: str,
    ):
        self.code = code
        self.dutch_name = dutch_name
        self.english_name = english_name
        self.main_group = main_group
        self.vertices = vertices
        self.source_note = source_note
        self.cartesian_vertices = [
            ternary_to_cartesian(v["sand"], v["silt"], v["lutum"]) for v in vertices
        ]
        self.shapely_polygon = Polygon(self.cartesian_vertices)


def load_triangle_data(path: Path = DATA_PATH) -> dict:
    """Load the raw declarative triangle JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_polygons(path: Path = DATA_PATH) -> Dict[str, NenPolygon]:
    """Load all NEN polygons keyed by class code."""
    data = load_triangle_data(path)
    return {p["code"]: NenPolygon(**p) for p in data["polygons"]}


_POLYGONS_CACHE: Optional[Dict[str, NenPolygon]] = None


def get_polygons() -> Dict[str, NenPolygon]:
    """Return the cached dict of NEN polygons, loading it on first use."""
    global _POLYGONS_CACHE
    if _POLYGONS_CACHE is None:
        _POLYGONS_CACHE = load_polygons()
    return _POLYGONS_CACHE


def classify_composition(
    sand: float,
    silt: float,
    lutum: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Classify a sand/silt/lutum composition against the NEN 5104 triangle.

    Args:
        sand (float): sand percentage.
        silt (float): silt percentage.
        lutum (float): lutum percentage.
        tolerance (float): tolerance in percentage points used for the
            sum-to-100 check, the negative-value check, and boundary
            detection.

    Returns:
        dict: one of
            {"status": "invalid", "errors": [...], "composition": {...}}
            {"status": "inside", "class_code": str, "class_name": str,
             "dutch_name": str, "composition": {...}}
            {"status": "on_boundary", "candidate_classes": [...], "composition": {...}}
            {"status": "unclassified", "candidate_classes": [], "composition": {...}}
    """
    composition = {"sand": sand, "silt": silt, "lutum": lutum}

    try:
        sand_f, silt_f, lutum_f = float(sand), float(silt), float(lutum)
    except (TypeError, ValueError):
        return {
            "status": "invalid",
            "errors": ["sand, silt and lutum must be numeric"],
            "composition": composition,
        }

    errors = []
    if any(math.isnan(v) for v in (sand_f, silt_f, lutum_f)):
        errors.append("sand, silt and lutum must not be NaN")
    else:
        if sand_f < -tolerance or silt_f < -tolerance or lutum_f < -tolerance:
            errors.append("sand, silt and lutum must not be negative")
        total = sand_f + silt_f + lutum_f
        if abs(total - 100.0) > tolerance:
            errors.append(f"sand + silt + lutum must equal 100 (got {total})")

    if errors:
        return {"status": "invalid", "errors": errors, "composition": composition}

    polygons = get_polygons()
    x, y = ternary_to_cartesian(sand_f, silt_f, lutum_f)
    point = Point(x, y)
    cartesian_tolerance = tolerance / 100.0

    candidates = [
        code
        for code, poly in polygons.items()
        if poly.shapely_polygon.distance(point) <= cartesian_tolerance
    ]

    if not candidates:
        return {
            "status": "unclassified",
            "candidate_classes": [],
            "composition": composition,
        }

    if len(candidates) == 1:
        poly = polygons[candidates[0]]
        return {
            "status": "inside",
            "class_code": poly.code,
            "class_name": poly.english_name,
            "dutch_name": poly.dutch_name,
            "composition": composition,
        }

    return {
        "status": "on_boundary",
        "candidate_classes": sorted(candidates),
        "composition": composition,
    }
