"""Restricted Ic mapping for the lutum+silt-zand-grind triangle's gravel-main-name region.

Scope: this module ONLY builds a representative-Ic field over the 5
"grind" main-name polygons (Gs, Gz1-4). The 3 modifier-only polygons
(g1-g3) intentionally get NO Ic value here - their main name (and hence
Ic) comes from the lutum-silt-zand triangle, not from this one. See
`RestrictedIcSurface` for how this differs from `ic_mapping.IcSurface`
(which requires every polygon in the triangle to have a target).

These are provisional, conceptual Ic targets (min/centre/max per class),
NOT a calibrated correlation - the combined lutum+silt axis does not
resolve whether the fines are lutum-like or silt-like, which particularly
limits the meaning of Gs (see `GRAVEL_CLASS_ENVELOPES["Gs"]["limitation"]`).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

import numpy as np  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

from modules.NEN_graph.classification import classify_composition_generic, get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import composition_to_cartesian  # noqa: E402
from modules.NEN_graph.triangle_definition import TriangleDefinition, load_triangle_definition  # noqa: E402

logger = logging.getLogger(__name__)

GRAVEL_DATA_PATH = Path(__file__).parent / "data" / "nen5104_lutum_silt_zand_grind.json"

# The 5 classes for which "grind" is the main name (Ic is mapped here).
MAPPED_CLASSES = ("Gs", "Gz1", "Gz2", "Gz3", "Gz4")

# g1/g2/g3 are gravel MODIFIERS only - their main name (and Ic) comes from
# the lutum-silt-zand triangle. Kept neutral/unmapped here.
UNMAPPED_CLASSES = ("g1", "g2", "g3")
UNMAPPED_IC_TEXT = "Ic from main-name triangle"

# Approved conceptual targets per user specification: (ic_min, ic_centre, ic_max).
# `ic_centre` is the class-representative control value used to build the
# surface; `ic_min`/`ic_max` are allowed envelopes (diagnostic only - NOT
# used to constrain the surface, exactly like `ic_mapping.IC_CLASS_ENVELOPES`).
GRAVEL_CLASS_ENVELOPES: Dict[str, dict] = {
    "Gz1": {"ic_min": 1.00, "ic_centre": 1.12, "ic_max": 1.25},
    "Gz2": {"ic_min": 1.05, "ic_centre": 1.20, "ic_max": 1.35},
    "Gz3": {"ic_min": 1.15, "ic_centre": 1.30, "ic_max": 1.45},
    "Gz4": {"ic_min": 1.25, "ic_centre": 1.35, "ic_max": 1.45},
    "Gs": {
        "ic_min": 1.20, "ic_centre": 1.40, "ic_max": 1.60,
        "limitation": (
            "The combined lutum+silt axis does not resolve whether Gs's fines are "
            "lutum-like or silt-like; this envelope spans both possibilities rather "
            "than assuming a fixed lutum-silt split. No separate model is introduced."
        ),
    },
}

# Class-representative control values used to build the surface (the "centre" column above).
GRAVEL_CLASS_TARGETS: Dict[str, float] = {code: env["ic_centre"] for code, env in GRAVEL_CLASS_ENVELOPES.items()}

# Explicit Ic value at every shared vertex of the mapped (Gz1-4, Gs) polygons,
# keyed by (grind, sand, lutum_silt) percentages - the repository's native
# component order, so no reordering is needed. This is the SOLE source of
# boundary-control values: class centres are never averaged to derive a
# boundary value (see `RestrictedIcSurface._build_controls`). Two touching
# polygons sharing a vertex therefore always resolve to exactly this value,
# by construction (a single control point is built per unique coordinate).
IC_GRAVEL_VERTEX_VALUES: Dict[Tuple[float, float, float], float] = {
    (100.0, 0.0, 0.0): 1.00,
    (90.0, 10.0, 0.0): 1.05,
    (70.0, 30.0, 0.0): 1.15,
    (50.0, 50.0, 0.0): 1.25,
    (30.0, 70.0, 0.0): 1.35,
    (80.0, 0.0, 20.0): 1.20,
    (70.0, 10.0, 20.0): 1.25,
    (50.0, 30.0, 20.0): 1.35,
    (30.0, 50.0, 20.0): 1.45,
    (30.0, 0.0, 70.0): 1.60,
}


class GravelIcMappingError(ValueError):
    """Raised for `GRAVEL_CLASS_ENVELOPES`/geometry configuration problems."""


class RestrictedIcSurface:
    """Class-anchored, piecewise-linear Ic surface over a SUBSET of a triangle's polygons.

    Unlike `ic_mapping.IcSurface` (which requires a target for every
    polygon in the triangle), this builds controls only from
    `mapped_class_targets`' classes, using every OTHER polygon in the
    triangle solely to determine, at each mapped-polygon vertex, which
    classes are incident there (via the same distance-based boundary test
    `classify_composition_generic` uses - not literal JSON vertex
    matching). Any incident class outside `mapped_class_targets` (e.g. an
    adjacent unmapped modifier-only polygon) simply does not contribute to
    the boundary control's value.

    Boundary values always come from `vertex_values` (an explicit,
    approved value per shared coordinate) - class centres are never
    averaged to derive a boundary control. Every vertex of every mapped
    polygon must have an entry in `vertex_values`, including vertices a
    polygon shares with a neighbor but does not itself declare in the
    triangle JSON (e.g. Gs's fines=20% edge is subdivided by Gz1/Gz2/Gz3's
    own vertices lying on that edge) - this subdivides the interpolation
    mesh without any change to the NEN geometry.

    Because the mapped region here (grind >= 30%) is itself convex, the
    Delaunay convex hull of all controls exactly equals that region, so
    `value_at_cartesian` naturally returns `None` outside it - no separate
    masking step is needed.

    Attributes:
        controls (List[dict]): one entry per control point (`xy`, `ic`,
            `control_type` in {"class_representative", "vertex_control"},
            `classes`, `composition`).
        boundary_conflicts (List[dict]): vertex controls whose value fell
            outside the intersection of its incident mapped classes'
            envelopes - reported, not silently corrected.
    """

    def __init__(
        self,
        triangle: TriangleDefinition,
        mapped_class_targets: Dict[str, float] = GRAVEL_CLASS_TARGETS,
        class_envelopes: Dict[str, dict] = GRAVEL_CLASS_ENVELOPES,
        vertex_values: Dict[Tuple[float, float, float], float] = IC_GRAVEL_VERTEX_VALUES,
        coordinate_tolerance: float = 1e-6,
    ):
        self.triangle = triangle
        self.mapped_class_targets = mapped_class_targets
        self.class_envelopes = class_envelopes
        self.vertex_values = vertex_values
        self.coordinate_tolerance = coordinate_tolerance
        self.polygons = get_triangle_polygons(triangle)
        self.mapped_classes: Set[str] = set(mapped_class_targets)

        unknown = self.mapped_classes - set(self.polygons)
        if unknown:
            raise GravelIcMappingError(f"mapped classes not present in the triangle: {sorted(unknown)}")

        self.boundary_conflicts: List[dict] = []
        self.controls = self._build_controls()

        points = np.array([c["xy"] for c in self.controls], dtype=float)
        self.values = np.array([c["ic"] for c in self.controls], dtype=float)
        self.triangulation = Delaunay(points)

    def _build_controls(self) -> List[dict]:
        controls: List[dict] = []
        seen_xy: Dict[Tuple[float, float], dict] = {}

        def _add(xy, ic, control_type, classes, composition):
            key = (round(xy[0], 9), round(xy[1], 9))
            existing = seen_xy.get(key)
            if existing is not None:
                if abs(existing["ic"] - ic) > 1e-9:
                    raise GravelIcMappingError(
                        f"conflicting control points at {xy}: {existing['ic']} vs {ic}"
                    )
                return
            entry = {"xy": (float(xy[0]), float(xy[1])), "ic": float(ic),
                      "control_type": control_type, "classes": classes, "composition": composition}
            seen_xy[key] = entry
            controls.append(entry)

        # 1. one representative control per mapped class, at its polygon's centre.
        for code in self.mapped_classes:
            poly = self.polygons[code]
            shapely_poly = poly.shapely_polygon
            centroid = shapely_poly.centroid
            point = (centroid.x, centroid.y) if shapely_poly.contains(centroid) else \
                (shapely_poly.representative_point().x, shapely_poly.representative_point().y)
            _add(point, self.mapped_class_targets[code], "class_representative", (code,), None)

        # 2. every vertex of every MAPPED polygon (own vertices + shared boundaries
        #    with neighbors, mapped or not) must have an explicit value in vertex_values.
        vertex_locations: Dict[Tuple[float, ...], dict] = {}
        for poly_def in self.triangle.polygons:
            if poly_def.code not in self.mapped_classes:
                continue
            for vertex in poly_def.vertices:
                key = tuple(round(vertex[k], 6) for k in self.triangle.component_keys)
                vertex_locations.setdefault(key, vertex)

        value_lookup: Dict[Tuple[float, ...], float] = {}
        for coords, value in self.vertex_values.items():
            vertex = dict(zip(self.triangle.component_keys, coords))
            key = tuple(round(vertex[k], 6) for k in self.triangle.component_keys)
            value_lookup[key] = value

        missing = set(vertex_locations) - set(value_lookup)
        if missing:
            raise GravelIcMappingError(f"missing explicit vertex_values for mapped-polygon vertices: {sorted(missing)}")
        unused = set(value_lookup) - set(vertex_locations)
        if unused:
            raise GravelIcMappingError(f"vertex_values not on any mapped-polygon vertex: {sorted(unused)}")

        for key, vertex in vertex_locations.items():
            result = classify_composition_generic(vertex, self.triangle, tolerance=self.coordinate_tolerance)
            if result["status"] == "inside":
                incident = [result["class_code"]]
            elif result["status"] == "on_boundary":
                incident = result["candidate_classes"]
            else:
                raise GravelIcMappingError(f"vertex {vertex} did not classify (status={result['status']!r})")

            mapped_incident = sorted(set(incident) & self.mapped_classes)
            if not mapped_incident:
                continue  # shouldn't happen (vertex belongs to a mapped polygon by construction)

            ic = value_lookup[key]
            self._check_envelope_conflict(vertex, mapped_incident, ic)

            xy = composition_to_cartesian(vertex, self.triangle)
            _add(xy, ic, "vertex_control", tuple(mapped_incident), vertex)

        return controls

    def _check_envelope_conflict(self, vertex: dict, classes: List[str], ic: float):
        lo = max(self.class_envelopes[c]["ic_min"] for c in classes)
        hi = min(self.class_envelopes[c]["ic_max"] for c in classes)
        if not (lo - 1e-9 <= ic <= hi + 1e-9):
            self.boundary_conflicts.append({
                "classes": tuple(classes), "location": dict(vertex), "assigned_ic": ic,
                "intersection_envelope": (lo, hi),
            })

    def value_at_cartesian(self, x: float, y: float) -> Optional[float]:
        """Linear barycentric interpolation, or `None` outside the mapped (grind>=30%) region."""
        simplex = self.triangulation.find_simplex((x, y))
        if simplex == -1:
            return None
        transform = self.triangulation.transform[simplex]
        delta = np.array([x, y]) - transform[2]
        bary_ab = transform[:2].dot(delta)
        bary = np.array([bary_ab[0], bary_ab[1], 1.0 - bary_ab.sum()])
        vertex_ids = self.triangulation.simplices[simplex]
        return float(bary.dot(self.values[vertex_ids]))

    def to_js_field(self) -> dict:
        return {
            "points": self.triangulation.points.tolist(),
            "simplices": self.triangulation.simplices.tolist(),
            "values": self.values.tolist(),
        }


_GRAVEL_TRIANGLE_CACHE: Optional[TriangleDefinition] = None
_GRAVEL_SURFACE_CACHE: Optional[RestrictedIcSurface] = None


def get_gravel_triangle() -> TriangleDefinition:
    global _GRAVEL_TRIANGLE_CACHE
    if _GRAVEL_TRIANGLE_CACHE is None:
        _GRAVEL_TRIANGLE_CACHE = load_triangle_definition(GRAVEL_DATA_PATH)
    return _GRAVEL_TRIANGLE_CACHE


def get_gravel_ic_surface() -> RestrictedIcSurface:
    global _GRAVEL_SURFACE_CACHE
    if _GRAVEL_SURFACE_CACHE is None:
        _GRAVEL_SURFACE_CACHE = RestrictedIcSurface(get_gravel_triangle())
    return _GRAVEL_SURFACE_CACHE


def calculate_gravel_ic(
    composition: Dict[str, float],
    triangle: Optional[TriangleDefinition] = None,
    surface: Optional[RestrictedIcSurface] = None,
) -> Optional[float]:
    """Representative Ic at a composition, or `None` outside the grind>=30% mapped region."""
    triangle = triangle or get_gravel_triangle()
    surface = surface or get_gravel_ic_surface()
    x, y = composition_to_cartesian(composition, triangle)
    return surface.value_at_cartesian(x, y)
