"""Generic ternary-triangle definition: schema, loading, and validation.

This module has no knowledge of any specific triangle's component names
(e.g. sand/silt/lutum) - it only understands the generic shape: exactly
three components, each assigned to one of the triangle's three corners
("left", "right", "top"), plus a set of classification polygons whose
vertices are expressed in those component keys.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

VALID_CORNERS = ("left", "right", "top")


class TriangleDefinitionError(ValueError):
    """Raised when a triangle definition JSON fails validation."""


@dataclass(frozen=True)
class ComponentDefinition:
    """One of the three components of a ternary triangle."""

    key: str
    label: str
    corner: str  # one of VALID_CORNERS


@dataclass(frozen=True)
class PolygonDefinition:
    """A single classification polygon, vertices in compositional coordinates."""

    code: str
    dutch_name: str
    english_name: str
    main_group: str
    vertices: List[Dict[str, float]]
    source_note: str = ""


@dataclass(frozen=True)
class TriangleDefinition:
    """A fully validated, generic ternary-triangle definition."""

    triangle_id: str
    components: List[ComponentDefinition]
    total: float
    polygons: List[PolygonDefinition]
    metadata: dict = field(default_factory=dict)

    @property
    def component_keys(self) -> List[str]:
        return [c.key for c in self.components]

    def component_by_corner(self, corner: str) -> ComponentDefinition:
        for c in self.components:
            if c.corner == corner:
                return c
        raise KeyError(f"no component assigned to corner {corner!r}")

    def label_for(self, key: str) -> str:
        for c in self.components:
            if c.key == key:
                return c.label
        raise KeyError(f"unknown component key {key!r}")


def load_triangle_definition(path: Union[str, Path]) -> TriangleDefinition:
    """Load and validate a triangle definition JSON file.

    Validates:
        - exactly three components
        - unique component keys
        - unique corner assignments (a permutation of left/right/top)
        - a valid (positive numeric) total
        - every vertex contains all three components
        - every vertex sums to the configured total
        - polygon codes are unique

    Raises:
        TriangleDefinitionError: if any of the above checks fail.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_components = data.get("components")
    if not isinstance(raw_components, list) or len(raw_components) != 3:
        raise TriangleDefinitionError(
            f"{path}: 'components' must be a list of exactly 3 entries (got {raw_components!r})"
        )

    components = []
    for entry in raw_components:
        try:
            components.append(
                ComponentDefinition(key=entry["key"], label=entry["label"], corner=entry["corner"])
            )
        except (KeyError, TypeError) as exc:
            raise TriangleDefinitionError(f"{path}: invalid component entry {entry!r}: {exc}") from exc

    keys = [c.key for c in components]
    if len(set(keys)) != len(keys):
        raise TriangleDefinitionError(f"{path}: component keys must be unique (got {keys})")

    corners = [c.corner for c in components]
    if sorted(corners) != sorted(VALID_CORNERS):
        raise TriangleDefinitionError(
            f"{path}: component corners must be a permutation of {VALID_CORNERS} (got {corners})"
        )

    total = data.get("total")
    if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
        raise TriangleDefinitionError(f"{path}: 'total' must be a positive number (got {total!r})")
    total = float(total)

    raw_polygons = data.get("polygons")
    if not isinstance(raw_polygons, list):
        raise TriangleDefinitionError(f"{path}: 'polygons' must be a list (got {type(raw_polygons)})")

    tolerance = 1e-6 * (total / 100.0)  # scale the default NEN tolerance to this triangle's total
    polygons = []
    seen_codes = set()
    for poly in raw_polygons:
        code = poly.get("code")
        if not code:
            raise TriangleDefinitionError(f"{path}: polygon missing 'code': {poly!r}")
        if code in seen_codes:
            raise TriangleDefinitionError(f"{path}: duplicate polygon code {code!r}")
        seen_codes.add(code)

        vertices = poly.get("vertices")
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise TriangleDefinitionError(f"{path}: polygon {code!r} must have at least 3 vertices")
        for v in vertices:
            missing = [k for k in keys if k not in v]
            if missing:
                raise TriangleDefinitionError(
                    f"{path}: polygon {code!r} vertex {v!r} is missing component(s) {missing}"
                )
            vertex_sum = sum(v[k] for k in keys)
            if abs(vertex_sum - total) > tolerance:
                raise TriangleDefinitionError(
                    f"{path}: polygon {code!r} vertex {v!r} sums to {vertex_sum}, expected {total}"
                )

        polygons.append(
            PolygonDefinition(
                code=code,
                dutch_name=poly.get("dutch_name", ""),
                english_name=poly.get("english_name", ""),
                main_group=poly.get("main_group", ""),
                vertices=vertices,
                source_note=poly.get("source_note", ""),
            )
        )

    metadata = {k: v for k, v in data.items() if k not in {"triangle_id", "components", "total", "polygons"}}

    return TriangleDefinition(
        triangle_id=data.get("triangle_id", path.stem),
        components=components,
        total=total,
        polygons=polygons,
        metadata=metadata,
    )
