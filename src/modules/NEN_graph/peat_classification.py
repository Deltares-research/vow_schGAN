"""Domain-specific wrapper for the organic stof-lutum-silt+zand (peat/veen) triangle.

Builds on the generic ternary-triangle engine (`triangle_definition`,
`coordinates`, `classification`) exactly like `gravel_ic_mapping.py` does for
the grind triangle - no changes to the generic engine or to either of the
other two (mineral, gravel) triangles.

Two class groups (see `data/nen5104_organic_lutum_silt_zand.json`):
    MAIN_NAME_CLASSES (Vm, Vk1, Vk3, Vz1, Vz3): "veen" main names, supplied
        directly by this triangle.
    MODIFIER_CLASSES (h1, h2, h3): humus modifiers only. The main name for a
        point in one of these regions is determined by the lutum-silt-zand
        (or lutum-silt-zand-grind) triangle instead; that lookup/routing is
        out of scope here (geometry/classification/visualization only).

`classify_peat_composition` additionally special-cases organic_matter == 0
(within tolerance): Figuur 3 gives no distinct, nonzero threshold for a
region with "no humus modifier" below h1 - h1's own lower boundary is the
organic_matter=0 edge itself (see the JSON's "h1_lower_edge_note"). A
composition exactly on that edge is reported as `humus_modifier=None`
("no_modifier" status) rather than "h1".
"""

import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import (  # noqa: E402
    DEFAULT_TOLERANCE,
    classify_composition_generic,
    get_triangle_polygons,
)
from modules.NEN_graph.triangle_definition import TriangleDefinition, load_triangle_definition  # noqa: E402

PEAT_DATA_PATH = Path(__file__).parent / "data" / "nen5104_organic_lutum_silt_zand.json"

MAIN_NAME_CLASSES = ("Vm", "Vk1", "Vk3", "Vz1", "Vz3")
MODIFIER_CLASSES = ("h1", "h2", "h3")

_PEAT_TRIANGLE_CACHE: Optional[TriangleDefinition] = None


def get_peat_triangle() -> TriangleDefinition:
    """Return the cached `TriangleDefinition` for the peat/veen triangle."""
    global _PEAT_TRIANGLE_CACHE
    if _PEAT_TRIANGLE_CACHE is None:
        _PEAT_TRIANGLE_CACHE = load_triangle_definition(PEAT_DATA_PATH)
    return _PEAT_TRIANGLE_CACHE


def get_peat_polygons():
    """Return (code -> ClassPolygon) for the peat/veen triangle."""
    return get_triangle_polygons(get_peat_triangle())


def classify_peat_composition(
    organic_matter: float,
    lutum: float,
    silt_zand: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Classify an organic_matter/lutum/silt_zand composition.

    Args:
        organic_matter (float): organic matter percentage.
        lutum (float): lutum (clay-sized fraction) percentage.
        silt_zand (float): silt+sand percentage.
        tolerance (float): tolerance (percentage points) for the sum-to-100
            check, the negative-value check, and boundary detection.

    Returns:
        dict: the generic `classify_composition_generic` result (see that
        function's docstring for the "invalid"/"inside"/"on_boundary"/
        "unclassified" statuses), extended with:
            - `supplies_main_name` (bool or None): True for Vm/Vk1/Vk3/Vz1/Vz3,
              False for h1/h2/h3 (or the no-modifier edge case), None when
              the composition is invalid/unclassified/on a boundary.
            - `humus_modifier` (str or None): "h1"/"h2"/"h3" or None.
        organic_matter<=`tolerance` overrides any h1 classification to a
        dedicated "no_modifier" status (see module docstring).
    """
    triangle = get_peat_triangle()
    composition = {"organic_matter": organic_matter, "lutum": lutum, "silt_zand": silt_zand}
    result = classify_composition_generic(composition, triangle, tolerance)

    if result["status"] == "inside":
        code = result["class_code"]
        if code in MODIFIER_CLASSES and organic_matter <= tolerance:
            return {
                "status": "no_modifier",
                "class_code": None,
                "supplies_main_name": False,
                "humus_modifier": None,
                "note": (
                    "organic matter content is at the 0% edge - no humus modifier applies; "
                    "main name is determined by the lutum-silt-zand(-grind) triangle"
                ),
                "composition": result["composition"],
            }
        result["supplies_main_name"] = code in MAIN_NAME_CLASSES
        result["humus_modifier"] = code if code in MODIFIER_CLASSES else None
    else:
        result["supplies_main_name"] = None
        result["humus_modifier"] = None

    return result
