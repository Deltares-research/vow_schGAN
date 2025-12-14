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


def create_custom_ic_colormap(
    ic_min: float = 1.0,
    ic_sand_sandmix_boundary: float = 2.05,
    ic_sandmix_siltmix_boundary: float = 2.6,
    ic_siltmix_clay_boundary: float = 2.95,
    ic_clay_organic_boundary: float = 3.6,
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


def trace(message: str, level: int = logging.INFO, verbose: bool = True):
    """Emit a verbose message if VERBOSE mode is enabled.

    Args:
        message: The log message to emit.
        level: Logging level (default: logging.INFO).
        verbose: Whether to emit the message (controlled by config).
    """
    if verbose:
        logger.log(level, message)
