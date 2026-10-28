"""Map BRO `soilNameNEN5104` compound Dutch soil names to NEN 5104 triangle class codes.

The BRO codelist encodes each soil description as a single camelCase compound
string, e.g. ``"zwakZandigeKlei"`` = "klei, zwak zandig" (gradation + adjective
+ base noun). This module decomposes that compound *generically* (no lookup
table of every possible compound) into ``(gradation, modifier, base)`` and
reconstructs the same ``"<base>, <gradation> <modifier>"`` string used by the
digitized triangle's `dutch_name` field, then looks it up directly against
`get_triangle_polygons()` - so the mapping is validated against the actual
polygon set rather than against a hand-maintained table.

This module does NOT modify any triangle geometry or Ic mapping. The main
noun selects the triangle (klei/zand/leem -> mineral, veen -> peat,
grind -> gravel; see `triangles.py`), and the reconstructed name is looked up
against that triangle's main-name classes.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.triangles import BASE_NOUN_TO_TRIANGLE, GENERIC_CLASS_BY_NOUN, dutch_name_to_code  # noqa: E402

# Base nouns as they appear at the END of the BRO camelCase compound, and the
# lowercase word used to rebuild the triangle's "<base>, ..." dutch_name.
_BASE_SUFFIXES = {"Klei": "klei", "Zand": "zand", "Leem": "leem", "Veen": "veen", "Grind": "grind"}

_GRADATIONS = ["uiterst", "sterk", "matig", "zwak"]  # longest-safe order; none is a prefix of another

# Canonical modifier spellings accepted after gradation-stripping and
# grammatical "-e" agreement normalization (e.g. "zandige" -> "zandig").
# "mineraalarm" is the qualifier of peat class Vm ("veen, mineraalarm").
_MODIFIERS = {"zandig", "siltig", "kleiig", "grindig", "mineraalarm"}


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


def _unsupported(raw, reason, dutch_name=None, triangle=None) -> Dict[str, object]:
    return {"raw": raw, "supported": False, "class_code": None, "triangle": triangle,
            "dutch_name": dutch_name, "reason": reason, "is_guess": False, "match_note": None}


# Main noun NOT followed by an adjective ending ("zandig", "kleiige", ...), so
# "zwakZandigeKlei" yields only "klei" and "zandNietGespecificeerd" yields "zand".
_NOUN_PATTERN = re.compile(r"(klei|zand|leem|veen|grind)(?!ig)")


def _guess_base_noun(raw: str) -> Optional[str]:
    """Main noun of a raw name: the decomposed base if parseable, else the last noun occurrence."""
    parts = _decompose(raw)
    if parts is not None:
        return parts["base"]
    nouns = _NOUN_PATTERN.findall(raw.lower())
    return nouns[-1] if nouns else None


def _with_generic_fallback(raw: str, failed: Dict[str, object]) -> Dict[str, object]:
    """Replace an unmatched result by the generic class of the main noun found in `raw`, if any."""
    noun = _guess_base_noun(raw)
    if noun is None:
        return failed
    return {
        "raw": raw, "supported": True, "class_code": GENERIC_CLASS_BY_NOUN[noun],
        "triangle": BASE_NOUN_TO_TRIANGLE[noun], "dutch_name": f"{noun} (generic, guessed)",
        "reason": None, "is_guess": True,
        "match_note": f"no exact NEN class ({failed['reason']}); guessed generic {noun} from the raw name",
    }


def map_soil_name_to_class(soil_name_nen5104: Optional[str]) -> Dict[str, object]:
    """Resolve a raw `soilNameNEN5104` string to a triangle class code.

    Exact NEN classes are tried first. If none matches but the raw name
    contains a main noun (klei/zand/leem/veen/grind, e.g. "Zand",
    "zandNietGespecificeerd", "matigKleiigVeen"), the layer gets that noun's
    generic class ("Zand*", "Veen*", ...; `is_guess=True`). Only names with
    no recognisable main noun stay unsupported.

    Returns:
        dict: always contains:
            - "raw" (str or None): the original input, unmodified.
            - "supported" (bool): whether a class (exact or generic) was resolved.
            - "class_code" (str or None): e.g. "Kz1", "Vk1" or a generic "Zand*".
            - "triangle" (str or None): "mineral", "peat" or "gravel".
            - "dutch_name" (str or None): the reconstructed "<base>, ..." string.
            - "reason" (str or None): explanation when NOT supported.
            - "is_guess" (bool): True for a generic class.
            - "match_note" (str or None): why the generic class was used.
    """
    raw = soil_name_nen5104
    if not raw:
        return _unsupported(raw, "no soilNameNEN5104 recorded for this layer")
    result = _match_exact(raw)
    return result if result["supported"] else _with_generic_fallback(raw, result)


def _match_exact(raw: str) -> Dict[str, object]:
    """Exact NEN class lookup in the triangle of the main noun."""
    parts = _decompose(raw)
    if parts is None:
        return _unsupported(raw, f"unrecognized base soil noun in {raw!r} (not klei/zand/leem/veen/grind)")

    base = parts["base"]
    triangle = BASE_NOUN_TO_TRIANGLE[base]

    if parts["gradation"] is None and parts["modifier"] is None:
        # Bare "klei"/"veen"/... with no qualifier - no single class corresponds (generic fallback applies).
        return _unsupported(
            raw, f"unqualified '{base}' (no gradation/modifier) has no corresponding {triangle}-triangle class",
            triangle=triangle,
        )

    if parts["modifier"] is None:
        return _unsupported(raw, f"could not parse a recognized modifier out of {raw!r}", triangle=triangle)

    dutch_name = f"{base}, {parts['gradation'] + ' ' if parts['gradation'] else ''}{parts['modifier']}"
    code = dutch_name_to_code(triangle).get(dutch_name)
    if code is None:
        return _unsupported(
            raw, f"reconstructed dutch_name {dutch_name!r} has no matching class in the {triangle} triangle",
            dutch_name=dutch_name, triangle=triangle,
        )

    return {"raw": raw, "supported": True, "class_code": code, "triangle": triangle,
            "dutch_name": dutch_name, "reason": None, "is_guess": False, "match_note": None}
