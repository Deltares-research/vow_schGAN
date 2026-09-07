"""Ic (soil behaviour type index) mapping for the NEN 5104 triangle.

This module builds ONE canonical, class-anchored, piecewise-linear
"representative Ic" surface over the NEN 5104 lutum-silt-zand composition
triangle. It replaces a series of earlier experimental models (an original
three-corner interpolation, a per-polygon fan triangulation, a global
Clough-Tocher cubic surface, and a constrained thin-plate-spline surface) -
none of those are kept; there is exactly one method now.

Methodology
-----------
1. Representative Ic values (`IC_CLASS_TARGETS`) are assigned once per NEN
   class - these are the "editable source of truth" for how the field
   behaves near the middle of each polygon.
2. Each class contributes ONE control point, placed at its polygon's
   geometric area centroid (or, if the centroid does not lie strictly
   inside the polygon - possible for concave shapes - at a deterministic
   interior "representative point" instead; any such fallback is recorded
   on `IcSurface.representative_point_fallbacks`).
3. Every distinct NEN polygon vertex that lies on the OUTER triangle
   perimeter (i.e. at least one component is ~0), excluding the three pure
   corners, becomes a "perimeter anchor". Its Ic value is the arithmetic
   mean of `IC_CLASS_TARGETS[code]` for every class whose polygon closure
   touches that exact location (found the same way
   `classify_composition_generic` finds "on_boundary" candidates - by
   geometric distance, not by literal JSON vertex membership - so a class
   whose edge merely passes through the point, without listing it as one
   of its own vertices, is still included). Internal (non-perimeter) NEN
   vertices are NOT used as interpolation controls in this method.
4. The three pure-component corners get explicit, hand-set values
   (`IC_CORNER_TARGETS`), which take precedence over the perimeter-anchor
   averaging rule above.
5. All control points (class representatives + perimeter anchors +
   corners) are converted to Cartesian coordinates and fed, together, into
   ONE global `scipy.spatial.Delaunay` triangulation. The Ic value at any
   composition is the linear (barycentric) interpolation of the surrounding
   simplex's three control values - see `IcSurface.value_at_cartesian`.

This construction is C0-continuous everywhere (adjacent simplices always
share the same two control values along their common edge), exactly
reproduces every control value, and never extrapolates: compositions
outside the triangle return `None` instead of a numerically extrapolated
value. It is NOT guaranteed to be monotonic in any particular composition
direction; check that separately (see `ic_diagnostic.py`) rather than
assuming it.

`IC_CLASS_ENVELOPES` (renamed from the old `IC_CLASS_MAPPING`) is kept as
diagnostic-only reference data (each class's plausible `[ic_min, ic_max]`
range and intended Robertson zone(s)); it is NOT used to constrain the
surface in any way.

None of this is a calibrated correlation; it is an experimental
visualization aid and must not be used for engineering interpretation.

Reuse for other triangles
--------------------------
`IcSurface` itself has no NEN-specific knowledge beyond the constants it is
given - to build an analogous surface for a different `TriangleDefinition`,
supply `class_targets` (class code -> representative Ic) and
`corner_targets` (a 3-entry dict keyed by tuples in the same order as
`triangle.component_keys`) for that triangle.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

import numpy as np  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

from modules.NEN_graph.classification import (  # noqa: E402
    classify_composition_generic,
    get_nen_triangle,
    get_triangle_polygons,
)
from modules.NEN_graph.coordinates import composition_to_cartesian  # noqa: E402
from modules.NEN_graph.triangle_definition import TriangleDefinition  # noqa: E402

logger = logging.getLogger(__name__)

# Per-class representative ("target") Ic value - the editable source of
# truth for where the surface is anchored near the middle of each polygon.
IC_CLASS_TARGETS: Dict[str, float] = {
    "Ks1": 3.30,
    "Ks2": 3.10,
    "Ks3": 2.90,
    "Ks4": 2.75,

    "Kz1": 2.85,
    "Kz2": 2.70,
    "Kz3": 2.65,

    "Lz1": 2.80,
    "Lz3": 2.55,

    "Zk": 1.85,
    "Zs1": 1.65,
    "Zs2": 1.80,
    "Zs3": 2.00,
    "Zs4": 2.30,
}

# Explicit Ic value for the 3 pure-component corners, keyed by
# (sand, silt, lutum) tuples (i.e. in `triangle.component_keys` order for
# the NEN triangle). These take precedence over the perimeter-anchor
# averaging rule for any other outer-edge vertex.
IC_CORNER_TARGETS: Dict[Tuple[float, float, float], float] = {
    (100.0, 0.0, 0.0): 1.60,   # pure sand
    (0.0, 100.0, 0.0): 2.80,   # pure silt
    (0.0, 0.0, 100.0): 3.45,   # pure lutum
}

# Diagnostic-only reference envelope per class (NOT used to constrain the
# surface): a plausible [ic_min, ic_max] range and the intended Robertson
# soil-behaviour-type zone(s), for documentation/QA purposes only.
IC_CLASS_ENVELOPES: Dict[str, dict] = {
    "Ks1": {"ic_min": 2.95, "ic_max": 3.55, "zones": [3]},
    "Ks2": {"ic_min": 2.80, "ic_max": 3.30, "zones": [3, 4]},
    "Ks3": {"ic_min": 2.65, "ic_max": 3.10, "zones": [4, 3]},
    "Ks4": {"ic_min": 2.45, "ic_max": 2.95, "zones": [4, 5]},

    "Kz1": {"ic_min": 2.55, "ic_max": 3.00, "zones": [4, 3]},
    "Kz2": {"ic_min": 2.50, "ic_max": 2.85, "zones": [4, 5]},
    "Kz3": {"ic_min": 1.95, "ic_max": 2.75, "zones": [4, 5]},

    "Lz1": {"ic_min": 2.55, "ic_max": 2.95, "zones": [4]},
    "Lz3": {"ic_min": 2.30, "ic_max": 2.75, "zones": [5, 4]},

    "Zk": {"ic_min": 1.60, "ic_max": 2.10, "zones": [6, 5]},
    "Zs1": {"ic_min": 1.50, "ic_max": 1.80, "zones": [6]},
    "Zs2": {"ic_min": 1.60, "ic_max": 1.95, "zones": [6]},
    "Zs3": {"ic_min": 1.80, "ic_max": 2.30, "zones": [6, 5]},
    "Zs4": {"ic_min": 2.05, "ic_max": 2.65, "zones": [5, 4]},
}


class IcClassMappingError(ValueError):
    """Raised for `IC_CLASS_TARGETS`/`IC_CORNER_TARGETS` configuration problems.

    Raised when a class is missing (or has an unexpected extra) target, or
    when two control points would land at the same location with
    conflicting Ic values.
    """


class IcSurface:
    """Global, class-anchored, piecewise-linear representative-Ic surface.

    See the module docstring for the full construction methodology.

    Attributes:
        controls (List[dict]): one entry per control point, each with keys
            `xy` (Cartesian tuple), `ic` (assigned value), `control_type`
            (`"class_representative"`, `"perimeter_anchor"`, or `"corner"`),
            `classes` (contributing class code(s), or `None` for corners),
            `composition` (the NEN composition tuple/key that produced this
            control, where applicable), and `is_fallback` (bool, only
            meaningful for `"class_representative"` controls).
        triangulation (scipy.spatial.Delaunay): the single global
            triangulation over every control point's Cartesian coordinates.
        values (np.ndarray): control Ic values, in the same order as
            `triangulation.points`.
        representative_point_fallbacks (List[str]): class codes whose
            centroid fell outside their own polygon, so a
            `representative_point()` fallback was used instead.
    """

    def __init__(
        self,
        triangle: Optional[TriangleDefinition] = None,
        class_targets: Dict[str, float] = IC_CLASS_TARGETS,
        corner_targets: Dict[Tuple[float, float, float], float] = IC_CORNER_TARGETS,
        class_envelopes: Dict[str, dict] = IC_CLASS_ENVELOPES,
        coordinate_tolerance: float = 1e-6,
    ):
        self.triangle = triangle or get_nen_triangle()
        self.class_targets = class_targets
        self.class_envelopes = class_envelopes
        self.coordinate_tolerance = coordinate_tolerance
        self.polygons = get_triangle_polygons(self.triangle)
        self.representative_point_fallbacks: List[str] = []

        self.controls = self._build_controls(corner_targets)

        points = np.array([c["xy"] for c in self.controls], dtype=float)
        self.values = np.array([c["ic"] for c in self.controls], dtype=float)
        self.triangulation = Delaunay(points)

    # -- construction ---------------------------------------------------

    def _build_controls(self, corner_targets: Dict[Tuple[float, float, float], float]) -> List[dict]:
        class_codes = set(self.polygons)
        missing_targets = class_codes - set(self.class_targets)
        extra_targets = set(self.class_targets) - class_codes
        if missing_targets or extra_targets:
            raise IcClassMappingError(
                f"IC_CLASS_TARGETS does not match the triangle's polygons: "
                f"missing={sorted(missing_targets)}, unexpected={sorted(extra_targets)}"
            )

        controls: List[dict] = []
        seen_xy: Dict[Tuple[float, float], dict] = {}

        def _add(xy, ic, control_type, classes, composition, is_fallback=False):
            key = (round(xy[0], 9), round(xy[1], 9))
            existing = seen_xy.get(key)
            if existing is not None:
                if existing["control_type"] != control_type or abs(existing["ic"] - ic) > 1e-9:
                    raise IcClassMappingError(
                        f"conflicting control points at {xy}: "
                        f"{existing['control_type']}={existing['ic']} vs {control_type}={ic}"
                    )
                return
            entry = {
                "xy": (float(xy[0]), float(xy[1])),
                "ic": float(ic),
                "control_type": control_type,
                "classes": classes,
                "composition": composition,
                "is_fallback": is_fallback,
            }
            seen_xy[key] = entry
            controls.append(entry)

        # 1. one representative control point per class.
        for code, poly in self.polygons.items():
            shapely_poly = poly.shapely_polygon
            centroid = shapely_poly.centroid
            if shapely_poly.contains(centroid):
                point = (centroid.x, centroid.y)
                is_fallback = False
            else:
                rep = shapely_poly.representative_point()
                point = (rep.x, rep.y)
                is_fallback = True
                self.representative_point_fallbacks.append(code)
            _add(point, self.class_targets[code], "class_representative", [code], None, is_fallback)

        # 2. perimeter anchors: distinct outer-edge NEN vertices, excluding corners.
        corner_keys = {tuple(round(v, 6) for v in key) for key in corner_targets}
        perimeter_vertices: Dict[Tuple[float, float, float], Dict[str, float]] = {}
        for poly_def in self.triangle.polygons:
            for vertex in poly_def.vertices:
                values = tuple(vertex[k] for k in self.triangle.component_keys)
                if min(values) > self.coordinate_tolerance:
                    continue  # strictly internal vertex - not an interpolation control
                key = tuple(round(v, 6) for v in values)
                if key in corner_keys:
                    continue  # corners are handled explicitly below
                perimeter_vertices.setdefault(key, vertex)

        for key, vertex in perimeter_vertices.items():
            classes = sorted(set(self._incident_classes(vertex)))
            ic = sum(self.class_targets[c] for c in classes) / len(classes)
            xy = composition_to_cartesian(vertex, self.triangle)
            _add(xy, ic, "perimeter_anchor", classes, key)

        # 3. explicit corner values (precedence over perimeter averaging).
        for corner_tuple, ic in corner_targets.items():
            composition = dict(zip(self.triangle.component_keys, corner_tuple))
            xy = composition_to_cartesian(composition, self.triangle)
            _add(xy, ic, "corner", None, corner_tuple)

        return controls

    def _incident_classes(self, composition: Dict[str, float]) -> List[str]:
        """Every class whose polygon closure contains `composition` (by distance, not JSON membership)."""
        result = classify_composition_generic(composition, self.triangle, tolerance=self.coordinate_tolerance)
        if result["status"] == "inside":
            return [result["class_code"]]
        if result["status"] == "on_boundary":
            return result["candidate_classes"]
        raise IcClassMappingError(
            f"perimeter anchor at {composition} does not classify against any polygon (status={result['status']!r})"
        )

    # -- evaluation -------------------------------------------------------

    def value_at_cartesian(self, x: float, y: float) -> Optional[float]:
        """Linear barycentric interpolation at (x, y), or `None` if outside the triangle.

        Never extrapolates: a composition strictly outside the outer
        triangle (i.e. outside the convex hull of the control points, which
        coincides with the outer triangle since the 3 corners are always
        control points) returns `None`.
        """
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
        """Serializable `{"points", "simplices", "values"}` for the hover script.

        `visualization.py`'s client-side hover evaluates the EXACT SAME
        Delaunay triangulation + linear barycentric interpolation as
        `value_at_cartesian` from this - there is no separate/approximate
        JS scheme.
        """
        return {
            "points": self.triangulation.points.tolist(),
            "simplices": self.triangulation.simplices.tolist(),
            "values": self.values.tolist(),
        }


_DEFAULT_SURFACE: Optional[IcSurface] = None


def get_ic_surface() -> IcSurface:
    """Return the cached default `IcSurface` for the NEN triangle."""
    global _DEFAULT_SURFACE
    if _DEFAULT_SURFACE is None:
        _DEFAULT_SURFACE = IcSurface()
    return _DEFAULT_SURFACE


def calculate_mean_ic(
    composition: Dict[str, float],
    triangle: Optional[TriangleDefinition] = None,
    surface: Optional[IcSurface] = None,
) -> Optional[float]:
    """Representative Ic at a composition, via the canonical `IcSurface`.

    Args:
        composition (dict): component key -> percentage, summing to `triangle.total`.
        triangle (TriangleDefinition, optional): defaults to the NEN triangle.
        surface (IcSurface, optional): defaults to the cached default surface for `triangle`.

    Returns:
        float: the interpolated representative Ic value, or `None` if the
            composition falls outside the triangle (no extrapolation).
    """
    triangle = triangle or get_nen_triangle()
    surface = surface or get_ic_surface()
    x, y = composition_to_cartesian(composition, triangle)
    return surface.value_at_cartesian(x, y)
