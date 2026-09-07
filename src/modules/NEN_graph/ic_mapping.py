"""Ic (soil behaviour type index) mapping for the NEN 5104 triangle.

Two models are provided:

- `calculate_mean_ic_legacy_three_anchor`: the original three-corner linear
  interpolation (kept only for comparison against the new model).
- `calculate_mean_ic_class_constrained` (the default): a continuous,
  piecewise-linear surface built from `IC_CLASS_MAPPING` (per-class target
  Ic range/centre) and `IC_VERTEX_VALUES` (an explicit Ic value for every
  NEN polygon vertex) - see `ClassConstrainedIcSurface`.

Neither model is a calibrated correlation; both are experimental
visualization aids and must not be used for engineering interpretation.
`calculate_ic_rw` (Robertson-Wride) is kept purely as a separate reference
calculation - it is not used to build either Ic surface above, and it is
not a fitting target for `IC_CLASS_MAPPING` or `IC_VERTEX_VALUES`.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from shapely.geometry import Point  # noqa: E402

from modules.NEN_graph.classification import get_nen_triangle, get_triangle_polygons  # noqa: E402
from modules.NEN_graph.coordinates import composition_to_cartesian  # noqa: E402
from modules.NEN_graph.triangle_definition import TriangleDefinition  # noqa: E402

IC_ANCHORS = {
    "sand": 1.68,
    "silt": 2.78,
    "lutum": 3.28,
}


def calculate_mean_ic_legacy_three_anchor(
    composition: Dict[str, float],
    anchors: Dict[str, float] = IC_ANCHORS,
    total: float = 100.0,
    tolerance: float = 1e-6,
) -> float:
    """Legacy: linearly interpolate a mean Ic value from just the 3 triangle corners.

    mean_ic = sum(fraction[key] * anchors[key] for key in anchors), where
    fraction[key] = composition[key] / total.

    Args:
        composition (dict): component key -> percentage, summing to `total`.
            Must provide a value for every key in `anchors`.
        anchors (dict): component key -> experimental Ic anchor value.
        total (float): the expected sum of `composition`'s values (100 for NEN).
        tolerance (float): allowed deviation from `total` for the sum check.

    Returns:
        float: the linearly interpolated mean Ic.

    Raises:
        ValueError: if composition is missing an anchor key, or its values
            (restricted to the anchor keys) don't sum to `total`.
    """
    missing = [k for k in anchors if k not in composition]
    if missing:
        raise ValueError(f"composition is missing component(s) {missing} required by anchors")

    composition_sum = sum(composition[k] for k in anchors)
    if abs(composition_sum - total) > tolerance:
        raise ValueError(f"composition must sum to {total} (got {composition_sum})")

    return sum((composition[k] / total) * anchors[k] for k in anchors)


def calculate_ic_rw(fines: float) -> float:
    """Robertson-Wride Ic benchmark from a fines percentage (silt + lutum).

    Reference calculation only - NEN 5104 uses a 63 micron lutum/silt
    boundary while the published Robertson-Wride fines relationship uses
    75 micron, so this is an approximate benchmark, not a like-for-like
    value. It does not influence `IC_CLASS_MAPPING`, `IC_VERTEX_VALUES`, or
    the class-constrained surface below.
    """
    return ((fines + 3.7) / 1.75) ** (1 / 3.25)


# Per-class target Ic range/centre. This is the editable "source of truth"
# for the class-constrained model's representative-point values - edit
# these to retune each polygon's centre. `zones` records the intended
# Robertson soil-behaviour-type zone(s) for documentation purposes only; it
# is not used in any calculation.
IC_CLASS_MAPPING = {
    "Ks1": {"ic_min": 2.95, "ic_center": 3.30, "ic_max": 3.55, "zones": [3]},
    "Ks2": {"ic_min": 2.80, "ic_center": 3.10, "ic_max": 3.30, "zones": [3, 4]},
    "Ks3": {"ic_min": 2.65, "ic_center": 2.90, "ic_max": 3.10, "zones": [4, 3]},
    "Ks4": {"ic_min": 2.45, "ic_center": 2.75, "ic_max": 2.95, "zones": [4, 5]},

    "Kz1": {"ic_min": 2.55, "ic_center": 2.85, "ic_max": 3.00, "zones": [4, 3]},
    "Kz2": {"ic_min": 2.50, "ic_center": 2.70, "ic_max": 2.85, "zones": [4, 5]},
    "Kz3": {"ic_min": 1.95, "ic_center": 2.65, "ic_max": 2.75, "zones": [4, 5]},

    "Lz1": {"ic_min": 2.55, "ic_center": 2.80, "ic_max": 2.95, "zones": [4]},
    "Lz3": {"ic_min": 2.30, "ic_center": 2.55, "ic_max": 2.75, "zones": [5, 4]},

    "Zk": {"ic_min": 1.60, "ic_center": 1.85, "ic_max": 2.10, "zones": [6, 5]},
    "Zs1": {"ic_min": 1.50, "ic_center": 1.65, "ic_max": 1.80, "zones": [6]},
    "Zs2": {"ic_min": 1.60, "ic_center": 1.80, "ic_max": 1.95, "zones": [6]},
    "Zs3": {"ic_min": 1.80, "ic_center": 2.00, "ic_max": 2.30, "zones": [6, 5]},
    "Zs4": {"ic_min": 2.05, "ic_center": 2.30, "ic_max": 2.65, "zones": [5, 4]},
}

# Explicit Ic value for every NEN polygon vertex, keyed by its exact
# (sand, silt, lutum) percentages. This is the editable "source of truth"
# for the class-constrained model's boundary values: every vertex that
# appears in the NEN JSON must have exactly one entry here (checked at
# `ClassConstrainedIcSurface` construction time). Because neighbouring
# polygons share the same vertex key, they automatically get the same
# boundary Ic value, which is what makes the surface continuous.
IC_VERTEX_VALUES = {
    # Outer corners
    (100.0, 0.0, 0.0): 1.60,   # pure sand
    (0.0, 100.0, 0.0): 2.80,   # pure silt
    (0.0, 0.0, 100.0): 3.45,   # pure lutum

    # Clay-rich boundaries
    (50.0, 0.0, 50.0): 3.15,
    (0.0, 50.0, 50.0): 3.25,
    (65.0, 0.0, 35.0): 3.00,
    (0.0, 65.0, 35.0): 3.05,
    (75.0, 0.0, 25.0): 2.85,
    (50.0, 25.0, 25.0): 2.75,
    (0.0, 75.0, 25.0): 2.85,

    # Loam boundaries
    (15.0, 65.1, 19.9): 2.70,
    (15.0, 85.0, 0.0): 2.70,

    # Sandy-clay boundaries
    (82.5, 0.0, 17.5): 2.65,
    (50.0, 32.5, 17.5): 2.75,
    (88.0, 0.0, 12.0): 2.50,
    (50.0, 38.0, 12.0): 2.65,

    # Lutum = 8% transition
    (92.0, 0.0, 8.0): 2.00,
    (82.5, 9.5, 8.0): 2.05,
    (67.5, 24.5, 8.0): 2.25,
    (50.0, 42.0, 8.0): 2.55,

    # Nearly pure sand
    (95.0, 0.0, 5.0): 1.70,
    (90.0, 5.0, 5.0): 1.75,
    (82.5, 12.5, 5.0): 1.90,

    # Lutum = 0% sand-silt edge
    (90.0, 10.0, 0.0): 1.70,
    (82.5, 17.5, 0.0): 1.85,
    (67.5, 32.5, 0.0): 2.15,
    (50.0, 50.0, 0.0): 2.45,
}


class IcClassMappingError(ValueError):
    """Raised for `IC_VERTEX_VALUES`/`IC_CLASS_MAPPING` configuration problems.

    Either a NEN polygon vertex has no corresponding `IC_VERTEX_VALUES`
    entry, or classes sharing a vertex have non-overlapping Ic ranges.
    """


def _vertex_lookup_key(vertex: Dict[str, float]) -> Tuple[float, float, float]:
    """(sand, silt, lutum) lookup key for `IC_VERTEX_VALUES`, rounded for float-safety."""
    return (round(vertex["sand"], 6), round(vertex["silt"], 6), round(vertex["lutum"], 6))


def _normalize_vertex_values_table(
    vertex_values: Dict[Tuple[float, float, float], float],
) -> Dict[Tuple[float, float, float], float]:
    return {tuple(round(v, 6) for v in key): value for key, value in vertex_values.items()}


def _barycentric(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    """Barycentric coordinates of `p` with respect to triangle (a, b, c), or None if degenerate."""
    x, y = p
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-15:
        return None
    wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom
    wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / denom
    return wa, wb, 1.0 - wa - wb


class ClassConstrainedIcSurface:
    """Continuous, piecewise-linear Ic surface built from explicit values.

    Construction:
        1. Each polygon's representative point is assigned its
           `IC_CLASS_MAPPING[code]["ic_center"]`.
        2. Every NEN polygon vertex is assigned its explicit
           `IC_VERTEX_VALUES` entry (looked up by exact (sand, silt, lutum)
           percentages) - no averaging is performed.
        3. Each polygon is fan-triangulated from its representative point to
           every edge; Ic is interpolated linearly (barycentric) within each
           fan triangle. Because adjacent polygons share identical vertex
           values along their common edge (the same `IC_VERTEX_VALUES`
           entry), the field is continuous across NEN polygon boundaries.

    Raises:
        IcClassMappingError: at construction time, if any NEN polygon
            vertex has no corresponding `IC_VERTEX_VALUES` entry.

    Attributes:
        validation_warnings (List[str]): non-fatal warnings noting any
            vertex whose explicit value falls outside a touching class's
            configured `[ic_min, ic_max]` range. Construction does not fail
            on these; inspect/report them explicitly (see `ic_diagnostic.py`).
    """

    def __init__(
        self,
        triangle: Optional[TriangleDefinition] = None,
        class_mapping: Dict[str, dict] = IC_CLASS_MAPPING,
        vertex_values: Dict[Tuple[float, float, float], float] = IC_VERTEX_VALUES,
    ):
        self.triangle = triangle or get_nen_triangle()
        self.class_mapping = class_mapping
        self.polygons = get_triangle_polygons(self.triangle)
        self._vertex_values, vertex_codes = self._resolve_vertex_values(vertex_values)
        self.validation_warnings = self._validate_vertex_ranges(vertex_codes)
        self._fans = self._build_fans()

    def _resolve_vertex_values(
        self, vertex_values: Dict[Tuple[float, float, float], float]
    ) -> Tuple[Dict[tuple, float], Dict[tuple, dict]]:
        normalized = _normalize_vertex_values_table(vertex_values)

        owners: Dict[tuple, dict] = {}
        for poly_def in self.triangle.polygons:
            for vertex in poly_def.vertices:
                key = _vertex_lookup_key(vertex)
                entry = owners.setdefault(key, {"vertex": vertex, "codes": set()})
                entry["codes"].add(poly_def.code)

        missing = [entry for key, entry in owners.items() if key not in normalized]
        if missing:
            details = "; ".join(
                f"{entry['vertex']} (used by {', '.join(sorted(entry['codes']))})" for entry in missing
            )
            raise IcClassMappingError(f"IC_VERTEX_VALUES is missing an entry for vertex/vertices: {details}")

        values = {key: normalized[key] for key in owners}
        return values, owners

    def _validate_vertex_ranges(self, owners: Dict[tuple, dict]) -> List[str]:
        warnings = []
        for key, entry in owners.items():
            value = self._vertex_values[key]
            for code in sorted(entry["codes"]):
                lo, hi = self.class_mapping[code]["ic_min"], self.class_mapping[code]["ic_max"]
                if not (lo - 1e-9 <= value <= hi + 1e-9):
                    warnings.append(
                        f"vertex {entry['vertex']} = {value:.3f} is outside {code}'s configured range [{lo}, {hi}]"
                    )
        return warnings

    def _build_fans(self) -> Dict[str, List[Tuple]]:
        fans = {}
        for poly_def in self.triangle.polygons:
            code = poly_def.code
            class_polygon = self.polygons[code]
            cartesian_vertices = class_polygon.cartesian_vertices
            vertex_ic = [self._vertex_values[_vertex_lookup_key(v)] for v in poly_def.vertices]
            rep_point = class_polygon.shapely_polygon.representative_point().coords[0]
            rep_ic = self.class_mapping[code]["ic_center"]

            n = len(cartesian_vertices)
            triangles = []
            for i in range(n):
                j = (i + 1) % n
                triangles.append(
                    (rep_point, cartesian_vertices[i], cartesian_vertices[j], rep_ic, vertex_ic[i], vertex_ic[j])
                )
            fans[code] = triangles
        return fans

    def locate_polygon(self, x: float, y: float) -> Optional[str]:
        """Nearest (or containing) NEN polygon code for an (x, y) point."""
        point = Point(x, y)
        best_code = None
        best_dist = float("inf")
        for code, poly in self.polygons.items():
            dist = poly.shapely_polygon.distance(point)
            if dist < best_dist:
                best_dist = dist
                best_code = code
            if dist == 0.0:
                break
        return best_code

    def value_at_cartesian(self, x: float, y: float) -> Optional[float]:
        """Interpolated Ic value at an (x, y) point, or None if outside the triangle."""
        code = self.locate_polygon(x, y)
        if code is None:
            return None
        best_value = None
        best_score = float("-inf")
        for rep_point, v_i, v_j, rep_ic, ic_i, ic_j in self._fans[code]:
            bary = _barycentric((x, y), rep_point, v_i, v_j)
            if bary is None:
                continue
            wa, wb, wc = bary
            score = min(wa, wb, wc)
            if score > best_score:
                best_score = score
                best_value = wa * rep_ic + wb * ic_i + wc * ic_j
        return best_value

    def to_js_fans(self) -> Dict[str, list]:
        """Serializable fan-triangle data for the client-side hover script."""
        return {
            code: [
                [[rp[0], rp[1], rep_ic], [vi[0], vi[1], ic_i], [vj[0], vj[1], ic_j]]
                for rp, vi, vj, rep_ic, ic_i, ic_j in triangles
            ]
            for code, triangles in self._fans.items()
        }


_DEFAULT_SURFACE: Optional[ClassConstrainedIcSurface] = None


def get_class_constrained_ic_surface() -> ClassConstrainedIcSurface:
    """Return the cached default `ClassConstrainedIcSurface` for the NEN triangle."""
    global _DEFAULT_SURFACE
    if _DEFAULT_SURFACE is None:
        _DEFAULT_SURFACE = ClassConstrainedIcSurface()
    return _DEFAULT_SURFACE


def calculate_mean_ic_class_constrained(
    composition: Dict[str, float],
    triangle: Optional[TriangleDefinition] = None,
    surface: Optional[ClassConstrainedIcSurface] = None,
) -> Optional[float]:
    """Default: continuous, class-constrained Ic via `ClassConstrainedIcSurface`.

    Args:
        composition (dict): component key -> percentage, summing to `triangle.total`.
        triangle (TriangleDefinition, optional): defaults to the NEN triangle.
        surface (ClassConstrainedIcSurface, optional): defaults to the cached
            default surface for `triangle`.

    Returns:
        float: the interpolated Ic value, or None if the composition falls
            outside the triangle.
    """
    triangle = triangle or get_nen_triangle()
    surface = surface or get_class_constrained_ic_surface()
    x, y = composition_to_cartesian(composition, triangle)
    return surface.value_at_cartesian(x, y)


def summarize_vertex_trend(
    vertex_values: Dict[Tuple[float, float, float], float] = IC_VERTEX_VALUES,
) -> float:
    """Pearson correlation between fines fraction (silt + lutum) and Ic across `IC_VERTEX_VALUES`.

    Informational only ("values generally increase when sand is replaced by
    silt or lutum"): this is not enforced anywhere in the surface
    construction, just reported by `ic_diagnostic.py` as a sanity check.
    """
    fines = [silt + lutum for (_sand, silt, lutum) in vertex_values]
    ic = list(vertex_values.values())
    n = len(fines)
    mean_f = sum(fines) / n
    mean_i = sum(ic) / n
    cov = sum((f - mean_f) * (i - mean_i) for f, i in zip(fines, ic))
    var_f = sum((f - mean_f) ** 2 for f in fines)
    var_i = sum((i - mean_i) ** 2 for i in ic)
    if var_f == 0 or var_i == 0:
        return 0.0
    return cov / (var_f * var_i) ** 0.5


