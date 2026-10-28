"""Registry of the three NEN 5104 triangles used by the borehole-to-Ic workflow.

Routing is by the MAIN soil noun of the BRO `soilNameNEN5104` name:

    klei / zand / leem -> "mineral"  (lutum-silt-zand triangle)
    veen               -> "peat"     (organische stof-lutum-silt+zand triangle)
    grind              -> "gravel"   (lutum+silt-zand-grind triangle)

Only the classes that supply a MAIN name with their own Ic field are routable
(peat: Vm, Vk1, Vk3, Vz1, Vz3; gravel: Gs, Gz1-Gz4). The modifier-only regions
(h1-h3 humeus, g1-g3 grindig) are not routed: their main name and Ic come from
another triangle, and combining modifiers is not implemented.

Generic classes ("Klei*", "Zand*", "Leem*", "Veen*", "Grind*") cover names
with a recognisable main noun but no exact class; they sample the union of
that noun's main-name classes (`class_polygon`).

Exact class codes are unique across the three triangles, so a class code
alone identifies its triangle (`spec_for_class`).
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import get_nen_triangle, get_triangle_polygons  # noqa: E402
from modules.NEN_graph.gravel_ic_mapping import (  # noqa: E402
    MAPPED_CLASSES as GRAVEL_MAIN_CLASSES, get_gravel_ic_surface, get_gravel_triangle,
)
from modules.NEN_graph.ic_mapping import get_ic_surface  # noqa: E402
from modules.NEN_graph.peat_classification import MAIN_NAME_CLASSES as PEAT_MAIN_CLASSES, get_peat_triangle  # noqa: E402
from modules.NEN_graph.peat_ic_mapping import get_peat_ic_surface  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

BASE_NOUN_TO_TRIANGLE = {"klei": "mineral", "zand": "mineral", "leem": "mineral", "veen": "peat", "grind": "gravel"}

# Per-triangle prefix for the composition columns in the samples table. The
# mineral triangle keeps its original unprefixed columns (sand_pct, silt_pct,
# lutum_pct) so existing outputs and consumers are unchanged.
COLUMN_PREFIX = {"mineral": "", "peat": "peat_", "gravel": "gravel_"}

# Display colours for borehole class bands, by main name: zand = yellows,
# klei = greens, veen = reds, leem = browns, grind = slate greys. Within a
# family the shade darkens with the secondary fraction (more lutum / more
# fines / less mineral). Generic guessed classes ("Zand*", ...) get a mid
# shade of their family. Used only by the borehole plots; the triangle
# charts keep their own palettes.
CLASS_COLORS = {
    "Ks1": "#1b5e20", "Ks2": "#2e7d32", "Ks3": "#43a047", "Ks4": "#66bb6a",
    "Kz1": "#7cb342", "Kz2": "#9ccc65", "Kz3": "#c5e1a5",
    "Zk": "#f9a825", "Zs4": "#fbc02d", "Zs3": "#fdd835", "Zs2": "#ffee58", "Zs1": "#fff59d",
    "Vm": "#7f0000", "Vk3": "#b71c1c", "Vk1": "#d32f2f", "Vz3": "#e57373", "Vz1": "#ef9a9a",
    "Lz1": "#8d6e63", "Lz3": "#bcaaa4",
    "Gs": "#546e7a", "Gz1": "#607d8b", "Gz2": "#78909c", "Gz3": "#90a4ae", "Gz4": "#b0bec5",
    "Klei*": "#81c784", "Zand*": "#ffe082", "Veen*": "#e53935", "Leem*": "#a1887f", "Grind*": "#90a4ae",
}

# Generic class per main noun, used when a raw soil name contains a recognisable
# main noun but does not match any NEN class exactly (e.g. "Zand", "Veen",
# "matigKleiigVeen"). The asterisk marks it as a guess. Its sampling region is
# the union of all main-name classes of that noun in the routed triangle.
GENERIC_CLASS_BY_NOUN = {"klei": "Klei*", "zand": "Zand*", "leem": "Leem*", "veen": "Veen*", "grind": "Grind*"}
_NOUN_BY_GENERIC_CLASS = {code: noun for noun, code in GENERIC_CLASS_BY_NOUN.items()}


@dataclass(frozen=True)
class TriangleSpec:
    key: str
    get_triangle: Callable
    get_surface: Callable
    main_classes: Optional[Tuple[str, ...]]  # None = every class in the triangle supplies a main name

    @property
    def triangle(self):
        return self.get_triangle()

    @property
    def surface(self):
        return self.get_surface()

    @property
    def polygons(self) -> Dict[str, object]:
        return _polygons(self.key)

    @property
    def component_keys(self) -> Tuple[str, ...]:
        return tuple(c.key for c in self.triangle.components)

    def column(self, component_key: str) -> str:
        return f"{COLUMN_PREFIX[self.key]}{component_key}_pct"

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self.column(k) for k in self.component_keys)


TRIANGLES: Dict[str, TriangleSpec] = {
    "mineral": TriangleSpec("mineral", get_nen_triangle, get_ic_surface, None),
    "peat": TriangleSpec("peat", get_peat_triangle, get_peat_ic_surface, tuple(PEAT_MAIN_CLASSES)),
    "gravel": TriangleSpec("gravel", get_gravel_triangle, get_gravel_ic_surface, tuple(GRAVEL_MAIN_CLASSES)),
}


@lru_cache(maxsize=None)
def _polygons(key: str):
    return get_triangle_polygons(TRIANGLES[key].triangle)


@lru_cache(maxsize=None)
def dutch_name_to_code(key: str) -> Dict[str, str]:
    """`dutch_name -> class_code` for the routable main-name classes of one triangle."""
    spec = TRIANGLES[key]
    return {
        poly.dutch_name: code for code, poly in spec.polygons.items()
        if spec.main_classes is None or code in spec.main_classes
    }


@lru_cache(maxsize=None)
def spec_for_class(class_code: str) -> TriangleSpec:
    if class_code in _NOUN_BY_GENERIC_CLASS:
        return TRIANGLES[BASE_NOUN_TO_TRIANGLE[_NOUN_BY_GENERIC_CLASS[class_code]]]
    for spec in TRIANGLES.values():
        if class_code in spec.polygons:
            return spec
    raise KeyError(f"class code {class_code!r} not found in any registered triangle")


def is_generic_class(class_code: str) -> bool:
    return class_code in _NOUN_BY_GENERIC_CLASS


@lru_cache(maxsize=None)
def generic_class_members(class_code: str) -> Tuple[str, ...]:
    """Main-name classes whose Dutch name starts with the generic class's noun (e.g. Zand* -> Zk, Zs1-Zs4)."""
    noun = _NOUN_BY_GENERIC_CLASS[class_code]
    names = dutch_name_to_code(BASE_NOUN_TO_TRIANGLE[noun])
    return tuple(code for name, code in names.items() if name.split(",")[0].strip() == noun)


@lru_cache(maxsize=None)
def class_polygon(class_code: str) -> Polygon:
    """Shapely polygon to sample for a class: the NEN polygon, or the union of members for a generic class."""
    spec = spec_for_class(class_code)
    if not is_generic_class(class_code):
        return spec.polygons[class_code].shapely_polygon
    # Members share edges, but T-junctions (a neighbour's vertex lying mid-edge,
    # e.g. Zk/Zs2 on Zs3's sand=82.5 edge) leave float-level slits that stop a plain
    # union from merging. A 1e-9 mitred dilate/erode closes them without moving corners.
    eps = 1e-9
    union = unary_union(
        [spec.polygons[c].shapely_polygon.buffer(eps, join_style=2) for c in generic_class_members(class_code)]
    ).buffer(-eps, join_style=2)
    if union.geom_type == "MultiPolygon":
        # Rounded shared vertices can leave numerically negligible slivers; keep the main region.
        parts = sorted(union.geoms, key=lambda g: g.area, reverse=True)
        if sum(p.area for p in parts[1:]) > 1e-6 * parts[0].area:
            raise ValueError(f"members of {class_code} do not form one contiguous region")
        union = parts[0]
    return union


def all_composition_columns() -> Tuple[str, ...]:
    return tuple(col for spec in TRIANGLES.values() for col in spec.columns)
