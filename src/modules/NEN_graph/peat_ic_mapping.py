"""Representative Ic mapping for the organic stof-lutum-silt+zand (peat/veen) triangle.

Scope: builds a representative-Ic field over the 5 "veen" main-name
polygons (Vm, Vk1, Vk3, Vz1, Vz3) only. The 3 modifier-only polygons
(h1-h3) intentionally get NO Ic value here - their main name (and hence
Ic) comes from the lutum-silt-zand(-grind) triangle, not from this one.

Reuses `gravel_ic_mapping.RestrictedIcSurface` as-is (it is already fully
generic over any `TriangleDefinition` - see that class's docstring for the
shared-boundary-control / class-centroid method). No changes are made to
`gravel_ic_mapping.py` or the gravel chart.

These are assumed representative Ic values for this review round, NOT a
CPT-calibrated relationship. Vm's upper limit of 4.40 is a modelling
choice, not a measured value.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.coordinates import composition_to_cartesian  # noqa: E402
from modules.NEN_graph.gravel_ic_mapping import GravelIcMappingError, RestrictedIcSurface  # noqa: E402
from modules.NEN_graph.peat_classification import MAIN_NAME_CLASSES, MODIFIER_CLASSES, get_peat_triangle  # noqa: E402
from modules.NEN_graph.triangle_definition import TriangleDefinition  # noqa: E402

# Re-exported under a peat-appropriate name; the error class has no gravel-specific logic.
PeatIcMappingError = GravelIcMappingError

MAPPED_CLASSES = MAIN_NAME_CLASSES  # Vm, Vk1, Vk3, Vz1, Vz3
UNMAPPED_CLASSES = MODIFIER_CLASSES  # h1, h2, h3 - Ic comes from the main-name triangle instead
UNMAPPED_IC_TEXT = "Ic from main-name triangle"

# Approved conceptual targets (assumed representative values, NOT CPT-calibrated).
# "ic_centre" is the class-representative control value used to build the
# surface; "ic_min"/"ic_max" are allowed envelopes (diagnostic only, used to
# flag boundary-control conflicts - never to constrain the surface itself).
PEAT_CLASS_ENVELOPES: Dict[str, dict] = {
    "Vm": {"ic_min": 3.90, "ic_centre": 4.15, "ic_max": 4.40},
    "Vk1": {"ic_min": 3.75, "ic_centre": 3.95, "ic_max": 4.10},
    "Vk3": {"ic_min": 3.60, "ic_centre": 3.78, "ic_max": 3.90},
    "Vz1": {"ic_min": 3.70, "ic_centre": 3.82, "ic_max": 3.95},
    "Vz3": {"ic_min": 3.55, "ic_centre": 3.65, "ic_max": 3.75},
}

# Class-representative control values used to build the surface (the "centre" column above).
PEAT_CLASS_TARGETS: Dict[str, float] = {code: env["ic_centre"] for code, env in PEAT_CLASS_ENVELOPES.items()}

# Explicit Ic value at every shared vertex of the mapped (Vm, Vk1, Vk3, Vz1, Vz3)
# polygons, keyed by (silt_zand, lutum, organic_matter) - this triangle's declared
# component order (see the JSON's "components" list) - the repository convention
# also used by `gravel_ic_mapping.IC_GRAVEL_VERTEX_VALUES`. This is the SOLE source
# of boundary-control values; class centres are never averaged to derive one.
# R1/R2/R3 reuse the exact (already full-precision, 4-decimal-rounded) coordinates
# stored in the triangle JSON - no new nearby vertices are introduced.
IC_PEAT_VERTEX_VALUES: Dict[Tuple[float, float, float], float] = {
    (0.0, 0.0, 100.0): 4.40,             # apex
    (65.0, 0.0, 35.0): 3.90,             # Vm/Vz1 boundary (outer edge)
    (54.6951, 4.7561, 40.5488): 3.95,    # R1: Vz1/Vk1 boundary
    (0.0, 30.0, 70.0): 4.10,             # Vm/Vk1 boundary (outer edge)
    (77.5, 0.0, 22.5): 3.70,             # Vz1/Vz3 boundary (outer edge)
    (69.0405, 6.0035, 24.956): 3.75,     # R2: Vz1/Vk1/Vz3/Vk3 four-way point
    (0.0, 55.0, 45.0): 3.90,             # Vk1/Vk3 boundary (outer edge)
    (85.0, 0.0, 15.0): 3.55,             # Vz3/h3 boundary (outer edge)
    (76.882, 6.6854, 16.4326): 3.60,     # R3: Vz3/Vk3 boundary (at the h3 edge)
    (0.0, 70.0, 30.0): 3.70,             # Vk3/h3 boundary (outer edge)
}


_PEAT_IC_SURFACE_CACHE: Optional[RestrictedIcSurface] = None


def get_peat_ic_surface() -> RestrictedIcSurface:
    """Return the cached representative-Ic surface for the peat/veen triangle."""
    global _PEAT_IC_SURFACE_CACHE
    if _PEAT_IC_SURFACE_CACHE is None:
        _PEAT_IC_SURFACE_CACHE = RestrictedIcSurface(
            get_peat_triangle(),
            mapped_class_targets=PEAT_CLASS_TARGETS,
            class_envelopes=PEAT_CLASS_ENVELOPES,
            vertex_values=IC_PEAT_VERTEX_VALUES,
        )
    return _PEAT_IC_SURFACE_CACHE


def calculate_peat_ic(
    composition: Dict[str, float],
    triangle: Optional[TriangleDefinition] = None,
    surface: Optional[RestrictedIcSurface] = None,
) -> Optional[float]:
    """Representative Ic at a composition, or `None` outside Vm/Vk1/Vk3/Vz1/Vz3."""
    triangle = triangle or get_peat_triangle()
    surface = surface or get_peat_ic_surface()
    x, y = composition_to_cartesian(composition, triangle)
    return surface.value_at_cartesian(x, y)
