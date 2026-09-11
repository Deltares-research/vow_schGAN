"""Anisotropic, depth-correlated composition sampling within NEN polygons.

For each supported layer, composition points (sand/silt/lutum, represented
as Cartesian (x, y) in the triangle) are drawn as a mean-reverting
Ornstein-Uhlenbeck process centered on the layer's own polygon, so that:
    - samples are denser near the polygon's representative point and thin
      out towards its boundary (an "anisotropic normal-shaped" spread), and
    - consecutive depth samples are correlated (`correlation_length_m`)
      rather than independent white noise.

Containment is enforced by bounded-retry rejection sampling (redraw the
noise until the point falls inside the polygon) - samples are NEVER clipped
or projected onto the polygon boundary, and any unresolved retry is handled
explicitly (falls back to repeating the previous valid point), never
silently.

All constants below are explicit, documented assumptions for this
prototype - not calibrated against real BRO/CPT statistics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

# -- documented, adjustable prototype assumptions ----------------------------
CORRELATION_LENGTH_M = 0.10  # OU process correlation length (uncalibrated assumption)
RANDOM_SEED = 42
SPREAD_FRACTION = 0.35  # fraction of each polygon's oriented-bbox half-extent used as std-dev
MAX_REJECTION_RETRIES = 60  # per-sample containment retries before falling back


@dataclass
class PolygonShape:
    """Anisotropic Gaussian shape (center + oriented std-devs) fit to a polygon."""

    center: Tuple[float, float]
    axis_u: Tuple[float, float]  # unit vector, first principal (oriented-bbox) axis
    axis_v: Tuple[float, float]  # unit vector, orthogonal axis
    std_u: float
    std_v: float


def fit_polygon_shape(polygon: Polygon, spread_fraction: float = SPREAD_FRACTION) -> PolygonShape:
    """Fit an anisotropic center+axes+std shape to `polygon`.

    Uses the polygon's minimum rotated bounding rectangle to get an
    orientation and two extents that follow the polygon's own elongation
    (rather than a single isotropic spread, which behaves poorly for the
    long, narrow class polygons near the triangle's edges).
    """
    center_point = polygon.representative_point() if not polygon.contains(polygon.centroid) else polygon.centroid
    center = (center_point.x, center_point.y)

    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:4]
    edge0 = np.array(coords[1]) - np.array(coords[0])
    edge1 = np.array(coords[2]) - np.array(coords[1])
    len0, len1 = np.linalg.norm(edge0), np.linalg.norm(edge1)
    axis_u = edge0 / len0 if len0 > 1e-12 else np.array([1.0, 0.0])
    axis_v = edge1 / len1 if len1 > 1e-12 else np.array([0.0, 1.0])

    return PolygonShape(
        center=center,
        axis_u=tuple(axis_u), axis_v=tuple(axis_v),
        std_u=max(len0 * 0.5 * spread_fraction, 1e-4),
        std_v=max(len1 * 0.5 * spread_fraction, 1e-4),
    )


class LayerSampler:
    """Generates a correlated (OU-process) sequence of composition points for one layer.

    Call `sample_at(depth_m)` in increasing depth order; each call advances
    the underlying AR(1) state by one step (`dt = depth_m - previous_depth_m`)
    and returns a point guaranteed to lie inside `polygon` (via bounded-retry
    rejection sampling), never clipped to its boundary.
    """

    def __init__(self, polygon: Polygon, seed: int, correlation_length_m: float = CORRELATION_LENGTH_M):
        self.polygon = polygon
        self.shape = fit_polygon_shape(polygon)
        self.correlation_length_m = correlation_length_m
        self.rng = np.random.default_rng(seed)
        self._u_state: Optional[np.ndarray] = None  # standardized principal-axis coordinates
        self._last_depth: Optional[float] = None
        self._last_point: Optional[Tuple[float, float]] = None
        self.fallback_count = 0

    def _to_xy(self, u: np.ndarray) -> Tuple[float, float]:
        shape = self.shape
        dx = u[0] * shape.std_u * shape.axis_u[0] + u[1] * shape.std_v * shape.axis_v[0]
        dy = u[0] * shape.std_u * shape.axis_u[1] + u[1] * shape.std_v * shape.axis_v[1]
        return shape.center[0] + dx, shape.center[1] + dy

    def sample_at(self, depth_m: float) -> Tuple[float, float, bool]:
        """Return (x, y, is_fallback) for the given depth.

        `is_fallback` is True only when all containment retries were
        exhausted and the previous valid point had to be repeated instead
        (documented, not silent).
        """
        if self._u_state is None:
            dt = 0.0
            rho = 0.0
        else:
            dt = depth_m - self._last_depth
            rho = float(np.exp(-dt / self.correlation_length_m)) if dt > 0 else 0.0

        for _ in range(MAX_REJECTION_RETRIES):
            noise = self.rng.standard_normal(2)
            if self._u_state is None:
                candidate_u = noise
            else:
                candidate_u = self._u_state * rho + noise * np.sqrt(max(1.0 - rho ** 2, 0.0))
            x, y = self._to_xy(candidate_u)
            if self.polygon.contains(Point(x, y)):
                self._u_state = candidate_u
                self._last_depth = depth_m
                self._last_point = (x, y)
                return x, y, False

        # Retries exhausted: fall back to repeating the previous valid point
        # (or the polygon center, on the very first sample) rather than
        # clipping/projecting onto the boundary.
        self.fallback_count += 1
        if self._last_point is None:
            self._last_point = self.shape.center
            self._u_state = np.zeros(2)
        self._last_depth = depth_m
        return self._last_point[0], self._last_point[1], True


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Standard smoothstep (3t^2 - 2t^3), clamped to [0, 1] first."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bounded_transition_half_thickness(
    requested_half_thickness_m: float, thickness_above_m: float, thickness_below_m: float,
) -> float:
    """Cap a transition zone's half-thickness so it never exceeds 40% of either neighbor's own thickness.

    Prevents a wide transition from erasing a thin layer's own core interval
    or overlapping an adjacent transition zone.
    """
    cap = 0.4 * min(thickness_above_m, thickness_below_m)
    return max(min(requested_half_thickness_m, cap), 0.0)
