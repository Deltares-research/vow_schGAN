"""Map BRO `soilNameNEN5104` compound Dutch soil names to NEN 5104 triangle class codes.

The BRO codelist encodes each soil description as a single camelCase compound
string, e.g. ``"zwakZandigeKlei"`` = "klei, zwak zandig" (gradation + adjective
+ base noun). This module decomposes that compound *generically* (no lookup
table of every possible compound) into ``(gradation, modifier, base)`` and
reconstructs the same ``"<base>, <gradation> <modifier>"`` string used by the
digitized triangle's `dutch_name` field, then looks it up directly against
`get_triangle_polygons()` - so the mapping is validated against the actual
polygon set rather than against a hand-maintained table.

This module does NOT modify the triangle geometry or Ic mapping; it only
consumes `classification.get_nen_triangle()` / `get_triangle_polygons()`.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.NEN_graph.classification import get_nen_triangle, get_triangle_polygons  # noqa: E402

# Base nouns as they appear at the END of the BRO camelCase compound, and the
# lowercase word used to rebuild the triangle's "<base>, ..." dutch_name.
_BASE_SUFFIXES = {"Klei": "klei", "Zand": "zand", "Leem": "leem", "Veen": "veen", "Grind": "grind"}

# Base nouns that are NOT represented at all in the digitized klei/silt/zand
# triangle (organic and coarse-grained material are explicitly out of scope
# for that triangle - see `NEN_graph/README.md`).
_UNSUPPORTED_BASES = {
    "veen": "organic material (veen/peat) is not represented in the digitized NEN 5104 lutum-silt-zand triangle",
    "grind": "gravel (grind) is not represented in the digitized NEN 5104 lutum-silt-zand triangle",
}

_GRADATIONS = ["uiterst", "sterk", "matig", "zwak"]  # longest-safe order; none is a prefix of another

# Canonical modifier spellings accepted after gradation-stripping and
# grammatical "-e" agreement normalization (e.g. "zandige" -> "zandig").
_MODIFIERS = {"zandig", "siltig", "kleiig", "grindig"}


def _decompose(soil_name: str) -> Optional[Dict[str, Optional[str]]]:
    """Split a BRO `soilNameNEN5104` compound into (base, gradation, modifier).

    Returns `None` if the base noun is not recognized at all (unparseable
    value - should be flagged as unsupported/ambiguous by the caller, not
    silently dropped).
    """
    name = soil_name.strip()
    low_full = name.lower()
    if low_full in ("klei", "zand", "leem", "veen", "grind"):
        return {"base": low_full, "gradation": None, "modifier": None}

    base = None
    prefix = None
    for suffix_cap, base_word in _BASE_SUFFIXES.items():
        if name.endswith(suffix_cap):
            base = base_word
            prefix = name[: -len(suffix_cap)]
            break
    if base is None or not prefix:
        return None

    prefix = prefix[0].lower() + prefix[1:]
    gradation = None
    rest = prefix
    for g in _GRADATIONS:
        if prefix.startswith(g):
            gradation = g
            rest = prefix[len(g):]
            rest = rest[0].lower() + rest[1:] if rest else rest
            break

    modifier_key = rest
    if modifier_key.endswith("ige"):  # "zandige"/"siltige"/"grindige" -> drop the grammatical "-e"
        modifier_key = modifier_key[:-1]
    modifier = modifier_key if modifier_key in _MODIFIERS else None

    return {"base": base, "gradation": gradation, "modifier": modifier}


def _dutch_name_lookup() -> Dict[str, str]:
    """Build `dutch_name -> class_code` from the live triangle (no hardcoded table)."""
    triangle = get_nen_triangle()
    polygons = get_triangle_polygons(triangle)
    return {poly.dutch_name: code for code, poly in polygons.items()}


_DUTCH_NAME_TO_CODE: Optional[Dict[str, str]] = None


def _get_dutch_name_to_code() -> Dict[str, str]:
    global _DUTCH_NAME_TO_CODE
    if _DUTCH_NAME_TO_CODE is None:
        _DUTCH_NAME_TO_CODE = _dutch_name_lookup()
    return _DUTCH_NAME_TO_CODE


def map_soil_name_to_class(soil_name_nen5104: Optional[str]) -> Dict[str, object]:
    """Resolve a raw `soilNameNEN5104` string to a triangle class code.

    Args:
        soil_name_nen5104 (str): the raw BRO compound Dutch soil name, e.g.
            "zwakZandigeKlei". May be `None`/empty for a layer with no
            recorded soil description.

    Returns:
        dict: always contains:
            - "raw" (str or None): the original input, unmodified.
            - "supported" (bool): whether a triangle class was resolved.
            - "class_code" (str or None): e.g. "Kz1", only if supported.
            - "dutch_name" (str or None): the reconstructed "<base>, ..." string.
            - "reason" (str or None): explanation when NOT supported.
    """
    if not soil_name_nen5104:
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": None, "reason": "no soilNameNEN5104 recorded for this layer",
        }

    parts = _decompose(soil_name_nen5104)
    if parts is None:
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": None,
            "reason": f"unrecognized base soil noun in {soil_name_nen5104!r} (not klei/zand/leem/veen/grind)",
        }

    base = parts["base"]
    if base in _UNSUPPORTED_BASES:
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": None, "reason": _UNSUPPORTED_BASES[base],
        }

    if parts["gradation"] is None and parts["modifier"] is None:
        # Bare "klei"/"zand"/"leem" with no qualifier - the digitized triangle
        # has no "pure/clean" polygon for these; do not guess a class.
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": None,
            "reason": f"unqualified '{base}' (no gradation/modifier) has no corresponding polygon in the triangle",
        }

    if parts["modifier"] is None:
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": None,
            "reason": f"could not parse a recognized modifier out of {soil_name_nen5104!r}",
        }

    dutch_name = f"{base}, {parts['gradation'] + ' ' if parts['gradation'] else ''}{parts['modifier']}"
    code = _get_dutch_name_to_code().get(dutch_name)
    if code is None:
        return {
            "raw": soil_name_nen5104, "supported": False, "class_code": None,
            "dutch_name": dutch_name,
            "reason": f"reconstructed dutch_name {dutch_name!r} has no matching polygon in the digitized triangle",
        }

    return {"raw": soil_name_nen5104, "supported": True, "class_code": code, "dutch_name": dutch_name, "reason": None}
