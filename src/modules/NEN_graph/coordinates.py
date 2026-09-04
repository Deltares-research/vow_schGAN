"""Ternary <-> Cartesian coordinate conversion for the NEN 5104 triangle.

Orientation (matches Figuur 4):
    - Top vertex      : lutum = 100 (sand = 0, silt = 0)
    - Bottom-left     : sand  = 100 (silt = 0, lutum = 0)
    - Bottom-right    : silt  = 100 (sand = 0, lutum = 0)

The triangle is rendered as an equilateral triangle with the bottom edge
on y=0 running from x=0 (sand corner) to x=1 (silt corner), and the apex
at (0.5, sqrt(3)/2). Cartesian coordinates are dimensionless (0-1 range)
and independent of plot size/resolution; scaling to pixels/points is left
entirely to the plotting code.
"""

import math
from typing import Tuple

SQRT3_OVER_2 = math.sqrt(3) / 2.0


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
