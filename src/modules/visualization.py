"""
Visualization utilities for VOW SchemaGAN pipeline.

This module provides colormap creation and visualization helper functions.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from typing import Tuple
import numpy as np
from matplotlib.colors import ListedColormap

from typing import Tuple
import numpy as np
import colorsys
from matplotlib.colors import ListedColormap


from typing import Tuple
import numpy as np
from matplotlib.colors import ListedColormap


def _ramp(c0, c1, n):
    """Linear RGBA ramp from c0 -> c1 with n steps."""
    c0 = np.array(c0, dtype=float)
    c1 = np.array(c1, dtype=float)
    return np.linspace(c0, c1, max(int(n), 1))


def create_custom_ic_colormap(
    ic_min: float = 1.0,
    ic_sand_sandmix_boundary: float = 2.05,
    ic_sandmix_siltmix_boundary: float = 2.6,
    ic_siltmix_clay_boundary: float = 2.95,
    ic_clay_organic_boundary: float = 3.4,
    ic_max: float = 4.5,
) -> Tuple[ListedColormap, float, float]:

    vmin = ic_min
    vmax = ic_max

    sand_end = ic_sand_sandmix_boundary
    sandmix_end = ic_sandmix_siltmix_boundary
    siltmix_end = ic_siltmix_clay_boundary
    clay_end = ic_clay_organic_boundary

    n_bins = 256

    # Segment proportions
    sand_prop = (sand_end - vmin) / (vmax - vmin)
    sandmix_prop = (sandmix_end - sand_end) / (vmax - vmin)
    siltmix_prop = (siltmix_end - sandmix_end) / (vmax - vmin)
    clay_prop = (clay_end - siltmix_end) / (vmax - vmin)

    n_sand = int(n_bins * sand_prop)
    n_sandmix = int(n_bins * sandmix_prop)
    n_siltmix = int(n_bins * siltmix_prop)
    n_clay = int(n_bins * clay_prop)
    n_organic = n_bins - n_sand - n_sandmix - n_siltmix - n_clay  # remainder

    # Two vivid tones per class (tweak freely)
    sand_c0, sand_c1 = [1.00, 0.90, 0.10, 1.0], [
        0.95,
        0.70,
        0.00,
        1.0,
    ]  # vivid yellow -> deeper yellow
    sandmix_c0, sandmix_c1 = [1.00, 0.65, 0.10, 1.0], [
        0.90,
        0.40,
        0.00,
        1.0,
    ]  # vivid orange -> deeper orange
    silt_c0, silt_c1 = [0.10, 0.80, 1.00, 1.0], [
        0.00,
        0.45,
        0.95,
        1.0,
    ]  # vivid cyan -> vivid blue
    clay_c0, clay_c1 = [0.20, 0.95, 0.20, 1.0], [
        0.00,
        0.55,
        0.10,
        1.0,
    ]  # vivid green -> deeper green
    org_c0, org_c1 = [1.00, 0.20, 0.20, 1.0], [
        0.70,
        0.00,
        0.00,
        1.0,
    ]  # vivid red -> deeper red

    sand = _ramp(sand_c0, sand_c1, n_sand)
    sandmix = _ramp(sandmix_c0, sandmix_c1, n_sandmix)
    siltmix = _ramp(silt_c0, silt_c1, n_siltmix)
    clay = _ramp(clay_c0, clay_c1, n_clay)
    organic = _ramp(org_c0, org_c1, n_organic)

    color_array = np.vstack([sand, sandmix, siltmix, clay, organic])

    cmap = ListedColormap(color_array, name="custom_ic_vivid_ramps")
    cmap.set_under("black")
    cmap.set_over("black")
    cmap.set_bad("black")

    return cmap, vmin, vmax


def create_custom_ic_colormap_OLD(
    ic_min: float = 1.0,
    ic_sand_sandmix_boundary: float = 2.05,
    ic_sandmix_siltmix_boundary: float = 2.6,
    ic_siltmix_clay_boundary: float = 2.95,
    ic_clay_organic_boundary: float = 3.4,
    ic_max: float = 4.5,
) -> Tuple[ListedColormap, float, float]:
    """Create custom segmented colormap for IC values.

    Args:
        ic_min: Start of sand (gold) range
        ic_sand_sandmix_boundary: Sand to sand mix transition
        ic_sandmix_siltmix_boundary: Sand mix to silt mix transition
        ic_siltmix_clay_boundary: Silt mix to clay transition
        ic_clay_organic_boundary: Clay to organic transition
        ic_max: End of organic (red) range

    Returns:
        Tuple of (colormap, vmin, vmax)
            - colormap: ListedColormap with five categories
            - vmin: Minimum IC value
            - vmax: Maximum IC value
    """
    vmin = ic_min
    vmax = ic_max
    sand_end = ic_sand_sandmix_boundary
    sandmix_end = ic_sandmix_siltmix_boundary
    siltmix_end = ic_siltmix_clay_boundary
    clay_end = ic_clay_organic_boundary

    # Total number of discrete colors
    n_bins = 256

    # Proportion of each segment
    sand_prop = (sand_end - vmin) / (vmax - vmin)
    sandmix_prop = (sandmix_end - sand_end) / (vmax - vmin)
    siltmix_prop = (siltmix_end - sandmix_end) / (vmax - vmin)
    clay_prop = (clay_end - siltmix_end) / (vmax - vmin)
    organic_prop = (vmax - clay_end) / (vmax - vmin)

    # Number of bins for each segment
    n_sand = int(n_bins * sand_prop)
    n_sandmix = int(n_bins * sandmix_prop)
    n_siltmix = int(n_bins * siltmix_prop)
    n_clay = int(n_bins * clay_prop)
    n_organic = n_bins - n_sand - n_sandmix - n_siltmix - n_clay

    # Build color array with uniform, saturated colors for high contrast
    color_array = []

    # Sand segment: Pure gold (bright and saturated)
    gold_color = [1.0, 0.843, 0.0, 1.0]
    for i in range(n_sand):
        color_array.append(gold_color)

    # Sand mix segment: Pure orange (bright and saturated)
    orange_color = [1.0, 0.55, 0.0, 1.0]
    for i in range(n_sandmix):
        color_array.append(orange_color)

    # Silt mix segment: Bright cyan/light blue
    cyan_color = [0.0, 0.75, 1.0, 1.0]
    for i in range(n_siltmix):
        color_array.append(cyan_color)

    # Clay segment: Bright green
    green_color = [0.0, 0.8, 0.0, 1.0]
    for i in range(n_clay):
        color_array.append(green_color)

    # Organic segment: Pure red
    red_color = [1.0, 0.0, 0.0, 1.0]
    for i in range(n_organic):
        color_array.append(red_color)

    cmap = ListedColormap(color_array, name="custom_ic")
    cmap.set_under("black")  # Values < vmin
    cmap.set_over("black")  # Values > vmax
    cmap.set_bad("black")  # NaN values

    return cmap, vmin, vmax


# Six soil-behaviour classes on Ic: Robertson zones, with all Ic below 2.05
# treated as sand (no separate gravelly-sand class 7) and the clay zone split
# into 3a (clay) and 3b (potential organic clay) at the proposed Ic = 3.30.
# Each class: (code, label, lower Ic, upper Ic, light colour, dark colour);
# lower bound inclusive, upper exclusive. The outer bounds (1.0 and 4.5) only
# set the displayed range: values below 1.0 are still sand and values above
# 4.5 still class 2.
IC_SOIL_CLASSES = [
    ("6", "sand (incl. Ic < 1.31)", 1.00, 2.05, (1.00, 0.90, 0.10), (0.95, 0.70, 0.00)),
    ("5", "sand mixtures", 2.05, 2.60, (1.00, 0.65, 0.10), (0.90, 0.40, 0.00)),
    ("4", "silt mixtures", 2.60, 2.95, (0.10, 0.80, 1.00), (0.00, 0.45, 0.95)),
    ("3a", "clay - silty clay to clay", 2.95, 3.30, (0.20, 0.95, 0.20), (0.00, 0.55, 0.10)),
    ("3b", "organic clay (potential)", 3.30, 3.60, (0.90, 0.97, 0.88), (0.74, 0.89, 0.72)),
    ("2", "organic soils / peat (potential)", 3.60, 4.50, (1.00, 0.20, 0.20), (0.70, 0.00, 0.00)),
]


def ic_class_boundaries() -> list:
    """Internal class boundaries (1.31 ... 3.60), increasing."""
    return [c[3] for c in IC_SOIL_CLASSES[:-1]]


def create_ic_class_colormap(n_bins: int = 4096) -> Tuple[ListedColormap, float, float]:
    """Colormap for IC_SOIL_CLASSES: one light-to-dark ramp per class, sharp jumps at the class boundaries.

    The many bins keep each boundary within (vmax - vmin) / n_bins (< 0.001 Ic) of its exact value.
    Values below vmin take the lightest sand colour, above vmax the darkest class-2 colour.
    """
    vmin, vmax = IC_SOIL_CLASSES[0][2], IC_SOIL_CLASSES[-1][3]
    centres = vmin + (np.arange(n_bins) + 0.5) * (vmax - vmin) / n_bins
    colors = np.ones((n_bins, 4))
    for _, _, lower, upper, light, dark in IC_SOIL_CLASSES:
        sel = (centres >= lower) & (centres < upper)
        t = ((centres[sel] - lower) / (upper - lower))[:, None]
        colors[sel, :3] = (1 - t) * np.array(light) + t * np.array(dark)
    cmap = ListedColormap(colors, name="ic_soil_classes")
    cmap.set_under(IC_SOIL_CLASSES[0][4])
    cmap.set_over(IC_SOIL_CLASSES[-1][5])
    cmap.set_bad("white")
    return cmap, vmin, vmax


def trace(message: str, level: int = logging.INFO, verbose: bool = True):
    """Emit a verbose message if VERBOSE mode is enabled.

    Args:
        message: The log message to emit.
        level: Logging level (default: logging.INFO).
        verbose: Whether to emit the message (controlled by config).
    """
    if verbose:
        logger.log(level, message)
