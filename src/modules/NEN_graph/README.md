# NEN 5104 lutum-silt-zand classification triangle

This module (`src/modules/NEN_graph/`) implements the NEN 5104 soil
classification ternary diagram ("Figuur 4 - Indeling van de
lutum-silt-zanddriehoek") as digital geometry, a point classifier, and a
matplotlib visualization.

## Files

| File | Purpose |
|---|---|
| `coordinates.py` | `ternary_to_cartesian` / `cartesian_to_ternary` conversion utilities. |
| `data/nen5104_triangle.json` | Declarative source of truth: the 14 classification polygons as sand/silt/lutum percent vertices, plus Dutch/English names and metadata. |
| `classification.py` | Loads the JSON, builds shapely polygons, and exposes `classify_composition(sand, silt, lutum)`. |
| `visualization.py` | Draws the full triangle (ticks, fills, labels, legend, hover readout) and saves a PNG when run directly. |
| `verify_geometry.py` | Manual (non-pytest) verification script — run it directly to re-check the geometry. |

## Orientation

- Top vertex: lutum = 100 (sand = 0, silt = 0)
- Bottom-left vertex: sand = 100 (silt = 0, lutum = 0)
- Bottom-right vertex: silt = 100 (sand = 0, lutum = 0)

Cartesian mapping (dimensionless, 0-1 range, independent of plot size):

```
x = (silt + 0.5 * lutum) / 100
y = (lutum * sqrt(3)/2) / 100
```

## Class list

| Code | Dutch name | English name | Main group |
|---|---|---|---|
| Ks1 | klei, zwak siltig | Clay, slightly silty | Klei (clay) |
| Ks2 | klei, matig siltig | Clay, moderately silty | Klei (clay) |
| Ks3 | klei, sterk siltig | Clay, strongly silty | Klei (clay) |
| Ks4 | klei, uiterst siltig | Clay, extremely silty | Klei (clay) |
| Kz1 | klei, zwak zandig | Clay, slightly sandy | Klei (clay) |
| Kz2 | klei, matig zandig | Clay, moderately sandy | Klei (clay) |
| Kz3 | klei, sterk zandig | Clay, strongly sandy | Klei (clay) |
| Lz1 | leem, zwak zandig | Loam, slightly sandy | Leem (loam) |
| Lz3 | leem, sterk zandig | Loam, strongly sandy | Leem (loam) |
| Zk | zand, kleiig | Sand, clayey | Zand (sand) |
| Zs1 | zand, zwak siltig | Sand, slightly silty | Zand (sand) |
| Zs2 | zand, matig siltig | Sand, moderately silty | Zand (sand) |
| Zs3 | zand, sterk siltig | Sand, strongly silty | Zand (sand) |
| Zs4 | zand, uiterst siltig | Sand, extremely silty | Zand (sand) |

Main-group boundary rule (derived from the geometry, used only to explain
polygon layout, not hard-coded anywhere as a separate rule):
- Klei (clay): lutum >= 8
- Leem (loam): lutum < 8 and sand < 50
- Zand (sand): lutum < 8 and sand >= 50

## Digitization method

The polygon vertices were **not** OCR'd or auto-extracted from the source
PDF (it is a scanned, textless fax image). They were reconstructed from:

1. The user-supplied exact grid-line values for all three ternary axes.
2. The user-supplied "Tabel 4" of official Dutch class names/qualifiers.
3. One explicit non-grid exception line: the Lz1/Lz3 divider does not sit
   on a labelled grid line; its position was given explicitly by the user.
4. Cross-checking the visual layout of the original attached figure to
   resolve which side of an internal split belongs to which class code
   (see "Known assumption" below).

Every boundary segment was verified to be shared **exactly** (identical
vertex coordinates) between neighboring polygons — see `verify_geometry.py`,
which checks:
- every vertex sums to 100 (sand + silt + lutum)
- every polygon is valid (no self-intersection)
- no two polygons overlap (pairwise intersection area ~ 0)
- the union of all 14 polygons exactly covers the outer triangle area
- ternary <-> Cartesian round-trip conversion accuracy
- representative points inside each polygon classify to that polygon
- known shared-edge/shared-vertex points classify as `on_boundary` with
  the correct candidate classes
- invalid inputs (negative percentages, sum != 100, NaN) are rejected

Run it with:

```bash
cd src
python modules/NEN_graph/verify_geometry.py
```

All checks currently **PASS**.

## Known assumption / ambiguity

The split between **Ks4** (clay, extremely silty) and **Kz1** (clay,
slightly sandy) occurs along the sand = 15 grid line. This threshold value
was given directly by the user's grid data, but which side of that
line is "Ks4" vs. "Kz1" was inferred from the visual layout of the
original figure (Kz1 sits closer to the sand corner than Ks4) rather than
being stated in words. This is geometrically self-consistent (verified via
edge-matching across the whole triangle) but is flagged here as an
assumption in case it needs to be double-checked against the printed
standard.

## Tolerance

`classify_composition` uses a single tolerance (default `1e-6`, in
percentage points) for:
- the sum-to-100 validity check
- the negative-value validity check
- boundary detection (a point is considered `on_boundary` if its distance
  to a polygon, in percent-equivalent units, is <= tolerance)

This is far smaller than the smallest gap between adjacent class
boundaries in this dataset (2 percentage points), so it will not
misclassify real points while still absorbing floating-point rounding
noise.

## Out of scope (this phase)

Per the task scope, this module intentionally does **not** implement: Ic
(soil behavior index) mapping, probability fields, color gradients,
sampling utilities, peat, gravel, borehole profiles, or layer-transition
logic. It also does not overlay the scanned source image on the render,
since no raster copy of "Figuur 4" exists in the repository to overlay
against — only the rendered digital triangle is produced.

## Usage

```python
from modules.NEN_graph.classification import classify_composition

result = classify_composition(sand=30, silt=60, lutum=10)
# {'status': 'inside', 'class_code': 'Kz3', 'class_name': 'Clay, strongly sandy', ...}
```

```bash
# from src/
python -m modules.NEN_graph.visualization
```
