"""Ternary <-> Cartesian coordinate conversion.

Orientation (matches Figuur 4):
    - Top vertex      : lutum = 100 (sand = 0, silt = 0)
    - Bottom-left     : sand  = 100 (silt = 0, lutum = 0)
    - Bottom-right    : silt  = 100 (sand = 0, lutum = 0)

The triangle is rendered as an equilateral triangle with the bottom edge
on y=0 running from x=0 (left corner) to x=1 (right corner), and the apex
at (0.5, sqrt(3)/2). Cartesian coordinates are dimensionless (0-1 range)
and independent of plot size/resolution; scaling to pixels/points is left
entirely to the plotting code.

`composition_to_cartesian` / `cartesian_to_composition` are generic: they
work with any `TriangleDefinition` (three components, each assigned to one
of the three corners) and do not reference specific component names.
`ternary_to_cartesian` / `cartesian_to_ternary` are kept, unchanged, as
sand/silt/lutum-specific convenience wrappers for backward compatibility.
"""

import math
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .triangle_definition import TriangleDefinition

SQRT3_OVER_2 = math.sqrt(3) / 2.0


def composition_to_cartesian(composition: Dict[str, float], triangle: "TriangleDefinition") -> Tuple[float, float]:
    """Convert a composition (mapping of component key -> value) to (x, y).

    The mapping from component to corner (and the triangle's `total`) is
    read entirely from `triangle`; this function contains no assumptions
    about specific component names.

    Args:
        composition (dict): component key -> value, summing to `triangle.total`.
        triangle (TriangleDefinition): the triangle's component/corner layout.

    Returns:
        Tuple[float, float]: (x, y) Cartesian coordinates in the unit triangle.
    """
    total = triangle.total
    w_left = composition[triangle.component_by_corner("left").key] / total
    w_right = composition[triangle.component_by_corner("right").key] / total
    w_top = composition[triangle.component_by_corner("top").key] / total
    x = w_right + 0.5 * w_top
    y = w_top * SQRT3_OVER_2
    return x, y


def cartesian_to_composition(x: float, y: float, triangle: "TriangleDefinition") -> Dict[str, float]:
    """Convert (x, y) Cartesian coordinates back to a composition dict.

    Args:
        x (float): x coordinate in the unit triangle.
        y (float): y coordinate in the unit triangle.
        triangle (TriangleDefinition): the triangle's component/corner layout.

    Returns:
        dict: component key -> value, summing to `triangle.total`.
    """
    total = triangle.total
    w_top = y / SQRT3_OVER_2
    w_right = x - 0.5 * w_top
    w_left = 1.0 - w_right - w_top
    return {
        triangle.component_by_corner("left").key: w_left * total,
        triangle.component_by_corner("right").key: w_right * total,
        triangle.component_by_corner("top").key: w_top * total,
    }


def ternary_to_cartesian(sand: float, silt: float, lutum: float) -> Tuple[float, float]:
    """Convert a sand/silt/lutum composition (percent, summing to 100) to (x, y).

    Args:
        sand (float): sand percentage (0-100).
        silt (float): silt percentage (0-100).
        lutum (float): lutum percentage (0-100).

    Returns:
        Tuple[float, float]: (x, y) Cartesian coordinates in the unit triangle.
    """
    x = (silt + 0.5 * lutum) / 100.0
    y = (lutum * SQRT3_OVER_2) / 100.0
    return x, y


def cartesian_to_ternary(x: float, y: float) -> Tuple[float, float, float]:
    """Convert (x, y) Cartesian coordinates back to sand/silt/lutum percentages.

    Args:
        x (float): x coordinate in the unit triangle.
        y (float): y coordinate in the unit triangle.

    Returns:
        Tuple[float, float, float]: (sand, silt, lutum) percentages summing to 100.
    """
    lutum = (y / SQRT3_OVER_2) * 100.0
    silt = x * 100.0 - 0.5 * lutum
    sand = 100.0 - silt - lutum
    return sand, silt, lutum

