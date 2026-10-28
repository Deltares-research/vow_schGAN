# Technical Handover: NEN 5104 → Representative-Ic Mapping and Borehole-to-Synthetic-Ic Workflow

**Purpose of this document.** This is a factual technical handover for another
assistant to build a presentation from. It contains no slides, no slide
outline, and no narrative arc — only verified facts about the current,
accepted implementation, collected figures, and mapping tables. No modelling
code was changed to produce this document. Every numeric claim below was
either read directly from source files (marked **[verified]**) or taken from
existing generated artifacts on disk (marked **[artifact]**). Anything that
is an assumption rather than a measured/derived fact is labeled **[assumption]**.

All file paths are relative to the repository root (`d:\codes\vow_schGAN`).

---

## 0. Folder contents

```
handover_nen_to_ic/
├── HANDOVER.md                       <- this file
├── figures/
│   ├── 01_mineral_triangle_ic_interactive.html
│   ├── 01_mineral_triangle_ic_static.png
│   ├── 01_mineral_triangle_ic_static_diagnostic.png
│   ├── 01_mineral_triangle_method_diagram_control_points.png
│   ├── 02_gravel_triangle_ic_static.png
│   ├── 02_gravel_triangle_ic_interactive.html
│   ├── 03_peat_triangle_ic_static.png
│   ├── 03_peat_triangle_ic_interactive.html
│   ├── 04_triangle_sampled_compositions_interactive.html
│   ├── 05_borehole_synthetic_vs_reference_cpt.png
│   └── 06_ensemble_comparison.png
└── tables/
    ├── mineral_class_ic_table.csv
    ├── mineral_control_points_all32.csv
    ├── mineral_ic_diagnostic_summary.csv
    ├── gravel_class_ic_table.csv
    ├── gravel_shared_boundary_controls.csv
    ├── peat_class_ic_table.csv
    ├── peat_shared_boundary_controls.csv
    ├── borehole_example_metadata.json
    ├── borehole_example_verification_report.txt
    ├── borehole_example_layer_table.csv
    ├── ensemble_summary.json
    └── ensemble_comparison_summary.csv
```

See §6 for what each figure/table shows and how to interpret it.

---

## 1. The three completed triangle mappings

There are three separate NEN 5104 classification triangles in this
repository, each with its own JSON geometry file under
`src/modules/NEN_graph/data/`, and each now has a representative-Ic
("Ic" = soil behaviour type index, Robertson) mapping built on top of it.

| Triangle | JSON file | Components (corner: left / right / top) | Total | Code |
|---|---|---|---|---|
| Mineral (lutum-silt-zand) | `nen5104_lutum_silt_zand.json` | sand / silt / lutum | 100 | `ic_mapping.py` |
| Gravel (lutum+silt-zand-grind) | `nen5104_lutum_silt_zand_grind.json` | grind / sand / lutum_silt | 100 | `gravel_ic_mapping.py` |
| Peat/veen (organic-lutum-silt+zand) | `nen5104_organic_lutum_silt_zand.json` | silt_zand / lutum / organic_matter | 100 | `peat_ic_mapping.py` |

**[verified]** Component/corner assignment read directly from each triangle's
JSON `"components"` list (`data/nen5104_lutum_silt_zand.json` lines 1-20,
`data/nen5104_lutum_silt_zand_grind.json` lines 1-15).

Cartesian conversion is identical and generic across all three
(`coordinates.py::composition_to_cartesian`): the triangle is unit
equilateral, bottom-left = "left"-component corner, bottom-right =
"right"-component corner, apex = "top"-component corner; `x = w_right +
0.5*w_top`, `y = w_top * sqrt(3)/2`, weights are fractions of `triangle.total`.

### 1.1 Classes with an Ic value vs. modifier-only regions

Not every polygon in every triangle gets an Ic value:

- **Mineral triangle**: all 14 polygons are "main-name" classes and **all**
  get an Ic value (`IcSurface`, no unmapped classes).
- **Gravel triangle**: 8 polygons total. Only the 5 where grind is the
  main name (**Gs, Gz1, Gz2, Gz3, Gz4**) get an Ic value. The 3
  grind-**modifier** polygons (**g1, g2, g3** = zwak/matig/sterk grindig)
  get no Ic value; the API/hover instead returns the literal text
  `"Ic from main-name triangle"` (`gravel_ic_mapping.py`, `UNMAPPED_IC_TEXT`),
  because their main name (and hence Ic) is determined by the mineral
  triangle instead.
- **Peat triangle**: 8 polygons total. Only the 5 where "veen" is the main
  name (**Vm, Vk1, Vk3, Vz1, Vz3**) get an Ic value. The 3 organic-**modifier**
  polygons (**h1, h2, h3** = zwak/matig/sterk humeus, organic_matter < 15%)
  get no Ic value, same `"Ic from main-name triangle"` text.

**[verified]** `MAPPED_CLASSES`/`UNMAPPED_CLASSES` constants in
`gravel_ic_mapping.py` and `peat_ic_mapping.py`.

### 1.2 Final class min/centre/max tables

Full tables are in `tables/mineral_class_ic_table.csv`,
`tables/gravel_class_ic_table.csv`, `tables/peat_class_ic_table.csv`. Summary:

**Mineral** (`IC_CLASS_TARGETS` = centre, `IC_CLASS_ENVELOPES` = min/max +
Robertson zone list, `ic_mapping.py` lines ~85-141) **[verified]**:

| Class | min | centre | max | Robertson zones |
|---|---|---|---|---|
| Ks1 | 2.95 | **3.30** | 3.55 | [3] |
| Ks2 | 2.80 | **3.10** | 3.30 | [3,4] |
| Ks3 | 2.65 | **2.90** | 3.10 | [4,3] |
| Ks4 | 2.45 | **2.75** | 2.95 | [4,5] |
| Kz1 | 2.55 | **2.85** | 3.00 | [4,3] |
| Kz2 | 2.50 | **2.70** | 2.85 | [4,5] |
| Kz3 | 1.95 | **2.65** | 2.75 | [4,5] |
| Lz1 | 2.55 | **2.80** | 2.95 | [4] |
| Lz3 | 2.30 | **2.55** | 2.75 | [5,4] |
| Zk  | 1.60 | **1.85** | 2.10 | [6,5] |
| Zs1 | 1.50 | **1.65** | 1.80 | [6] |
| Zs2 | 1.60 | **1.80** | 1.95 | [6] |
| Zs3 | 1.80 | **2.00** | 2.30 | [6,5] |
| Zs4 | 2.05 | **2.30** | 2.65 | [5,4] |

Plus 3 explicit pure-corner values (`IC_CORNER_TARGETS`) that override the
perimeter-anchor averaging rule (§2): pure sand (100,0,0) = **1.60**, pure
silt (0,100,0) = **2.80**, pure lutum (0,0,100) = **3.45**.

**Gravel** (`GRAVEL_CLASS_ENVELOPES`, `gravel_ic_mapping.py` lines ~38-53)
**[assumption — "provisional, conceptual Ic targets... NOT a calibrated
correlation"** per the module docstring]:

| Class | min | centre | max |
|---|---|---|---|
| Gz1 | 1.00 | **1.12** | 1.25 |
| Gz2 | 1.05 | **1.20** | 1.35 |
| Gz3 | 1.15 | **1.30** | 1.45 |
| Gz4 | 1.25 | **1.35** | 1.45 |
| Gs  | 1.20 | **1.40** | 1.60 |

Gs carries an explicit documented limitation: the combined lutum+silt axis
cannot resolve whether the fines are lutum-like or silt-like, so its
envelope deliberately spans both possibilities.

**Peat/veen** (`PEAT_CLASS_ENVELOPES`, `peat_ic_mapping.py` lines ~38-45)
**[assumption — "assumed representative Ic values for this review round,
NOT a CPT-calibrated relationship"**]:

| Class | min | centre | max |
|---|---|---|---|
| Vm  | 3.90 | **4.15** | 4.40 |
| Vk1 | 3.75 | **3.95** | 4.10 |
| Vk3 | 3.60 | **3.78** | 3.90 |
| Vz1 | 3.70 | **3.82** | 3.95 |
| Vz3 | 3.55 | **3.65** | 3.75 |

Vm's upper bound of 4.40 (at the pure-organic apex) is explicitly flagged
in the code as "a modelling choice, not a measured value."

### 1.3 How the min/max envelopes are used

For **all three** triangles the `[ic_min, ic_max]` envelope is
**diagnostic-only**: it is never used to constrain the interpolated surface,
only to flag (not fix) cases where an already-built boundary control value
falls outside the envelope of one of its incident classes
(`_check_envelope_conflict` in `gravel_ic_mapping.py`, reused unchanged by
peat; the mineral surface has no equivalent automatic check, but
`ic_diagnostic.py`'s `pct_outside_envelope` column reports the same kind of
information after the fact — see §5).

### 1.4 Main-name classification and secondary modifiers

Classification into a main-name polygon (Ks1, Gz2, Vk1, …) is purely
geometric: a composition's (x, y) point is located inside one of the
triangle's declared polygons (`classification.py::classify_composition`,
reused generically as `classify_composition_generic` in the Ic-mapping
modules). Secondary modifiers (grind content class g1/g2/g3, organic content
class h1/h2/h3, and — in the borehole workflow, see §3 — colour,
mottled, sand_median_class, carbonate_content_class, peat_constituent) are
carried through as **metadata only**; they do not change which Ic value is
looked up. Only the main-name class code determines the Ic value (or the
"Ic from main-name triangle" text if unmapped).

### 1.5 Assumption vs. Robertson-derived values

- **Robertson-derived**: the Robertson soil-behaviour-type zone boundaries
  `(2.05, 2.60, 2.95)` used as contour levels on the mineral and gravel Ic
  charts, and the "zones" column in the mineral class table, are the
  standard Robertson Ic thresholds (SBT zones 3-7 boundaries).
- **Assumption (this project's own choice, not from Robertson or a
  standard)**: every specific `ic_centre`/`ic_min`/`ic_max` number for every
  class in all three tables above, all boundary-control values in §2/§1.2,
  the peat chart's `REFERENCE_IC_CONTOURS=(3.80,4.00,4.20)` (explicitly
  labelled in code as "reference levels, not Robertson boundaries"), and
  all constants in the borehole sampling workflow (§3).

---

## 2. The actual interpolation method

### 2.1 Two implementations exist, sharing one mathematical form

- **`ic_mapping.IcSurface`** — used only by the **mineral** triangle. Every
  polygon must have a target (no unmapped classes).
- **`gravel_ic_mapping.RestrictedIcSurface`** — a generic class reused
  *unchanged* by both the **gravel** and **peat** triangles. Only a
  declared subset of polygons ("mapped classes") get Ic values.

Both reduce to the same generic recipe, and **both build exactly one
global Delaunay triangulation per triangle** — there is no per-polygon
fan/sub-triangulation in either implementation:

1. One **class-representative control** per mapped class, placed at that
   class polygon's centroid (or `representative_point()` if the centroid
   falls outside the polygon — tracked as a fallback).
2. A set of **boundary controls** at shared vertices between polygons, so
   neighbouring polygons agree exactly at their shared edge.
3. All controls (class-representative + boundary + any explicit corners)
   are fed into one `scipy.spatial.Delaunay` triangulation
   (`surface.triangulation`).
4. `value_at_cartesian(x, y)` locates the enclosing Delaunay simplex and
   does ordinary **barycentric linear interpolation** of the 3 corner
   control values; returns `None` outside the convex hull of all controls.
5. `to_js_field()` exports `{"points", "simplices", "values"}` so the
   Plotly HTML hover script can evaluate the *exact same* triangulation
   client-side in JavaScript (§2.4).

### 2.2 Where the mineral and gravel/peat methods genuinely differ

This is the substantive difference behind the phrase "polygon-centroid fans
versus global triangulation" — restated precisely, since neither
implementation is literally a "fan per polygon":

| | Mineral (`IcSurface`) | Gravel/Peat (`RestrictedIcSurface`) |
|---|---|---|
| Classes covered | **all** polygons in the triangle | only the declared mapped subset |
| Boundary control value | **arithmetic mean** of `IC_CLASS_TARGETS` over every class geometrically incident at that vertex (via `classify_composition_generic`) | **single explicit approved value** per vertex from a hand-authored table (`IC_GRAVEL_VERTEX_VALUES` / `IC_PEAT_VERTEX_VALUES`); *never* an average of class centres |
| Internal (non-perimeter) shared vertices | **excluded** — only vertices on the *outer* triangle perimeter become boundary controls; internal vertices are not controls at all | **included** — every vertex of every mapped polygon becomes a control, including internal ones shared only between mapped classes (e.g. peat's R1/R2/R3) |
| Pure corners | 3 explicit hand-set values (`IC_CORNER_TARGETS`) override the perimeter-averaging rule | not applicable (corners are not part of the mapped subset in these triangles) |
| Conflict handling | none automated (see `ic_diagnostic.py` diagnostics instead, §5) | `_check_envelope_conflict()` flags (does not fix) any boundary value falling outside the envelope intersection of its incident classes |

So: mineral's boundary values are *derived* (an average over whichever
classes meet there), while gravel/peat's are *authored* (one number per
vertex, chosen once and reused everywhere that vertex is shared) — the
underlying triangulation-and-barycentric-interpolation machinery afterward
is identical.

`ic_mapping.py`'s module docstring records that this "one global
triangulation" design replaced several earlier experimental approaches
(a three-corner interpolation, a per-polygon fan triangulation, a global
Clough-Tocher cubic surface, a constrained thin-plate spline) — "there is
exactly one method now" for the mineral triangle, and the same method
(generalised) was reused for gravel/peat rather than re-inventing another.

### 2.3 Continuity and class-range handling

- Continuity across a shared class boundary is enforced **by
  construction**: both neighbouring polygons' local triangulation-corner
  values are literally the same control point object, so the piecewise
  linear surface is continuous (not merely close) across every polygon
  edge that has an assigned control.
- Nothing in the interpolation enforces the class `[ic_min, ic_max]`
  envelope — it is diagnostic-only (§1.3). The mineral surface **is not**
  guaranteed monotonic in any composition direction (explicitly documented
  in `ic_mapping.py`); `ic_diagnostic.py` measures this after the fact
  (§5). The peat surface **was** checked for monotonicity as part of its
  own verification script and passed (zero gradient violations across 21
  interpolation triangles, per `verify_peat_ic.py`).

### 2.4 Python/JavaScript consistency

`visualization.py::_build_hover_script()` embeds a literal JavaScript port
of the same barycentric evaluator (`meanIcFromField`) used by
`value_at_cartesian()` in Python, operating on the exact `to_js_field()`
export of the same triangulation — so the interactive HTML hover value and
a direct Python call are mathematically the same algorithm, not two
independently-tuned models. This was independently checked for the peat
surface in `verify_peat_ic.py` (65 sample points, Python vs. an independent
Python re-implementation of the JS algorithm): max difference ≈ 8.88e-16
(floating-point noise only).

### 2.5 Rendering differences between the three charts

All three triangles now export both an interactive Plotly HTML chart and a
static matplotlib PNG, in the same visual format (viridis colormap,
"Representative Ic" colourbar, black polygon boundaries + class-code
labels, axis names on all three edges via the shared `_add_axis_labels`
helper):

- **Mineral**: `plot_nen_triangle_with_ic` (HTML) and the newly added
  `plot_nen_triangle_with_ic_static` (PNG), both in `visualization.py`.
  The static function was added during this handover round specifically
  to close the format gap with gravel/peat — it reuses the exact same
  `IcSurface` evaluator and grid-sampling logic as the HTML chart (a new
  shared `_sample_ic_grid` helper), so the two are numerically identical;
  only the rendering library differs. This is a visualization-only
  addition — the `IcSurface` model itself was not changed.
- **Gravel**: `plot_gravel_triangle_with_ic` (HTML) and
  `plot_gravel_triangle_with_ic_static` (PNG) in `gravel_visualization.py`.
- **Peat**: `plot_peat_triangle_with_ic` (HTML) and
  `plot_peat_triangle_with_ic_static` (PNG) in `peat_visualization.py`.

A separate, older diagnostic script (`ic_diagnostic.py`) also produces a
static PNG for the mineral surface
(`figures/01_mineral_triangle_ic_static_diagnostic.png` in this handover),
kept here alongside the new one because it re-triangulates the sampled
points with matplotlib's own `Triangulation` purely for rendering — the
**Ic values plotted are the same canonical `IcSurface` evaluator output**,
only the mesh used to draw the filled contours differs from the
authoritative interpolation mesh. See §5 for why this matters.

---

## 3. The borehole-to-synthetic-Ic workflow

Pipeline: BRO borehole XML → resolved NEN class per layer → sampled
composition points within that class's polygon → Ic via the mineral
`IcSurface` → optional ensemble of independent realizations. Reference CPT
(Lengkeek) is computed separately and used only for comparison.

### 3.1 Inputs and XML parsing

`xml_readers.py` parses BRO XML with Python's standard
`xml.etree.ElementTree` plus a custom namespace-stripping helper (not
`lxml`). `read_borehole_xml(path)` returns metadata (`bro_id`, `nitg_code`,
RD `x`/`y`, `surface_elevation_m_nap`, `vertical_datum`,
`vertical_reference_point`, `final_depth_m`) and a layer list with raw BRO
fields verbatim (`upper_boundary_m`, `lower_boundary_m`,
`soil_name_nen5104`, `gravel_content_class`, `organic_matter_content_class`,
`carbonate_content_class`, `colour`, `mottled`, `sand_median_class`,
`peat_constituent`). Missing fields become `None` (no fabrication);
malformed numeric fields raise, they are not silently skipped.
`read_cpt_xml_metadata(path)` extracts only `bro_id` + coordinates for
CPT-borehole association; the full CPT interpretation is delegated to
`geolib_plus.BroXmlCpt` (§3.5).

### 3.2 Translating borehole descriptions to NEN classes

`soil_mapping.py::map_soil_name_to_class()` decomposes the BRO camelCase
compound (e.g. `"zwakZandigeKlei"`) with regex into
`(gradation, modifier, base)` — **not** a static lookup table — using
`_BASE_SUFFIXES` (Klei/Zand/Leem/Veen/Grind), `_GRADATIONS`
(uiterst/sterk/matig/zwak), and a modifier set
(zandig/siltig/kleiig/grindig). It reconstructs the same
`"<base>, <gradation> <modifier>"` string convention as the triangle JSON's
`dutch_name` field and looks it up **directly against the live triangle
polygons** (`get_triangle_polygons()`), rather than against a
hand-maintained table — so a class only resolves if it actually exists in
the digitized mineral triangle.

**Key finding: gravel (`grind`) and organic (`veen`) base soil names are
explicitly UNSUPPORTED by this mapping today** — the code's own reason
strings say each "is not represented in the digitized NEN 5104
lutum-silt-zand triangle." This means **the newly-built gravel and peat Ic
triangles (§1) are not yet wired into the borehole workflow** — see §3.9
and §5 for the consequence.

### 3.3 Sampling a composition within a polygon

`composition_sampling.py` — **not** uniform-random over the polygon area.
An anisotropic, depth-correlated Ornstein-Uhlenbeck-like AR(1) process:

- Constants (all explicitly documented as uncalibrated assumptions):
  `CORRELATION_LENGTH_M = 0.10`, `RANDOM_SEED = 42`,
  `SPREAD_FRACTION = 0.35`, `MAX_REJECTION_RETRIES = 60`.
- `fit_polygon_shape()`: center = polygon centroid (or
  `representative_point()` if the centroid falls outside); orientation and
  extents from the polygon's minimum rotated bounding rectangle;
  `std_u = max(len0 * 0.5 * SPREAD_FRACTION, 1e-4)`,
  `std_v = max(len1 * 0.5 * SPREAD_FRACTION, 1e-4)`.
- AR(1) update per depth step: `rho = exp(-dt / CORRELATION_LENGTH_M)` (or
  `0` if `dt <= 0`); `candidate = u_state*rho + noise*sqrt(max(1-rho**2, 0))`.
- Containment: `polygon.contains(Point(x, y))`, up to
  `MAX_REJECTION_RETRIES` redraws; if all retries fail, the sampler
  explicitly repeats the previous valid point (never clips/projects onto
  the boundary), and this is counted in a `fallback_count` (0 for every
  layer in the worked example — see §4).

### 3.4 Depth spacing, transitions, and layer grouping

`profile_builder.py`: `DEPTH_STEP_M = 0.02` m;
`TRANSITION_THICKNESS_M = 0.20` m *requested*, but capped per-boundary to
`0.4 * min(thickness_above, thickness_below)` so a transition never eats
most of a thin layer. Depth grid: `n_steps = round(final_depth_m /
depth_step_m)`, evenly spaced 0..`n_steps*depth_step_m`.

**Consecutive identical classes**: `_group_supported_runs()` merges
consecutive layers that share the same `(supported, class_code)` pair into
one continuous run sampled by a *single* `LayerSampler` (so the AR(1)
process runs continuously across the merged interval, not restarted at the
original BRO layer boundary). Other raw fields differing between the
merged layers (e.g. `organic_matter_content_class`) do not affect grouping
— only `class_code` (and supported/unsupported status) does. The worked
example (§4) contains exactly this case: two adjacent `Ks1` layers at
3.0-7.0 m and 7.0-9.3 m with different humus qualifiers merge into one run.

**Transitions** are only inserted *between two supported runs of different
classes*: a smoothstep-blended interpolation in Cartesian (x, y) over the
capped half-thickness (`smoothstep(t) = t*t*(3-2t)`, clipped to [0,1]).
Between a supported and an unsupported run, there is a sharp gap (no
transition, i.e. `transition_thickness = 0.0`) — unsupported intervals are
left as `NaN` gaps, never interpolated across (§3.9, §5).

### 3.5 Reference CPT (Lengkeek) interpretation

`lengkeek_reference.py` does **not** reimplement any CPT interpretation
equations. It wraps the external `geolib_plus` library (repo docs describe
it as a "VOW modified build"): `BroXmlCpt` +
`RobertsonCptInterpretation(interpretation_method=InterpretationMethod.LENGKEEK_2024)`
+ `cpt_utils.NEN9997.NEN_classification(interpretation_method=2)` (for
Dutch main/qualifier names). The resulting `reference_ic_geolib_plus`
column comes from `geolib_plus`'s own internal `cpt.IC` attribute and is
kept **only as a cross-reference for comparison — it is never used to fit,
calibrate, or constrain the synthetic profile.**

### 3.6 Ensemble realizations

`ensemble.py`: `n_realizations` default 20; `base_seed = RANDOM_SEED = 42`.
`derive_realization_seeds()` uses
`np.random.SeedSequence(base_seed).spawn(n_realizations)`, then
`child.generate_state(1, dtype=np.uint32)[0]` per child — independent,
non-sequential streams (explicitly not `base_seed + i`, "which does not
guarantee independence"). Only the random stream varies across
realizations; borehole layers, depth grid, class mapping, and the
unsupported-interval mask are identical every run. Summary statistics per
depth: `ic_median = np.nanmedian(...)`, `ic_p05 = np.nanpercentile(..., 5)`,
`ic_p95 = np.nanpercentile(..., 95)` — **no min/max, no mean±std** are
computed.

### 3.7 Missing/unsupported intervals

`build_synthetic_profile()`'s output preserves unsupported layers as `NaN`
gaps in the synthetic Ic column rather than fabricating a value; this is
independently verified by `verification.py::check_unsupported_gaps_preserved`
(§5). The specific worked example in §4 happens to contain **zero**
unsupported layers.

### 3.8 Verification checks embedded in this workflow

`verification.py::run_all_checks()` runs 8 checks per built profile:
layer-interval validity, vertical alignment (elevation vs. depth),
composition-sum-to-100, polygon containment (samples fall inside their own
declared polygon, tolerance 1e-6), Ic agreement (recomputed vs. stored,
tolerance 1e-9), reproducibility under the same seed (tolerance 1e-12),
finiteness where supported, and gap preservation where unsupported. Full
results for the worked example are in §5/§6.

### 3.9 Which triangles are actually integrated into this workflow

**Only the mineral (lutum-silt-zand) triangle is wired into the borehole
workflow.** Confirmed by direct import search: `ic_mapping.py` (mineral) is
imported by `profile_builder.py`, `plotting.py`, and `verification.py`
(all via `get_ic_surface()`). `gravel_ic_mapping.py`, `peat_ic_mapping.py`,
and `peat_classification.py` are **not imported anywhere** in
`modules/borehole_ic/*.py` or in either `run_*.py` entry-point script.
`classification.py::get_nen_triangle()` / `get_triangle_polygons()` (used
by `soil_mapping.py` and `profile_builder.py`) resolve only to the mineral
triangle. This is consistent with §3.2's finding that `grind`/`veen` base
names are marked unsupported — the gravel and peat Ic surfaces exist,
are independently built and verified (§5), but have not yet been connected
to this pipeline.

---

## 4. Worked example and ensemble: BHR000000397320 / CPT000000061801

Source scripts: `src/run_borehole_to_ic_profile.py` (single realization) and
`src/run_ensemble_borehole_to_ic_profile.py` (20-realization ensemble).
Outputs live in `res/borehole_ic/BHR000000397320_CPT000000061801/` and
`..._ensemble/` respectively (both copied into `tables/`/`figures/` here).
**[artifact]** — all numbers in this section are read directly from those
generated files, not recomputed.

### 4.1 Borehole selection and site metadata

Three candidate boreholes were checked against the CPT location by RD
horizontal distance; the closest was auto-selected:

| Borehole | Distance to CPT (m) |
|---|---|
| **BHR000000397320 (selected)** | **2.88** |
| BHR000000397307 | 15.30 |
| BHR000000397319 | 52.15 |

Borehole: surface elevation **7.12 m NAP**, final depth **10.3 m**,
vertical datum NAP, vertical reference point "maaiveld" (ground level).
CPT location: RD (153595.28, 440381.53).

### 4.2 What comes from the borehole vs. the measured CPT

- **From the borehole** (`layer_table.csv`, `samples.csv`): the 7-layer
  NEN 5104 class log and the synthetic Ic profile sampled/interpolated
  from it (§3). This is a *model*, not a measurement.
- **From the CPT** (`reference_lengkeek_cpt.csv`): the actual measured CPT
  cone data, interpreted into Ic and Dutch main/qualifier names by
  `geolib_plus`'s Lengkeek-2024 Robertson implementation (§3.5). This is
  the only measured Ic in the comparison.
- The CPT is **only used for comparison, never to fit or calibrate** the
  synthetic sampling model (§3.5, §3.6).

All 7 borehole layers are **fully supported** (all resolve to mineral
triangle classes; no grind/veen descriptions occur in this particular
borehole) — so, for this specific example, the gravel/peat non-integration
noted in §3.9 causes no missing data, but that is a property of this one
borehole, not a general guarantee.

Layer table (`tables/borehole_example_layer_table.csv`):

| upper (m) | lower (m) | soil_name_nen5104 | class_code |
|---|---|---|---|
| 0.0 | 0.2 | kleiigZand | Zk |
| 0.2 | 0.9 | zwakSiltigeKlei | Ks1 |
| 0.9 | 1.5 | matigZandigeKlei | Kz2 |
| 1.5 | 3.0 | matigSiltigeKlei | Ks2 |
| 3.0 | 7.0 | zwakSiltigeKlei | Ks1 |
| 7.0 | 9.3 | zwakSiltigeKlei | Ks1 |
| 9.3 | 10.3 | zwakSiltigZand | Zs1 |

(The two adjacent Ks1 rows at 3.0-9.3 m are merged into one continuous
sampler run — §3.4.)

### 4.3 Elevation reference and overlapping comparison interval

Both profiles are expressed in elevation (m NAP): borehole elevation =
`surface_elevation_m_nap (7.12) - depth`. The reference CPT's own reported
surface elevation is **7.181 m NAP** at depth 0 (`reference_lengkeek_cpt.csv`
row 1) — about 6 cm different from the borehole's 7.12 m, consistent with
the ~2.88 m horizontal separation between the two physical locations (they
are not the same point). The borehole only extends to 10.3 m depth
(elevation ≈ **+7.12 to −3.18 m NAP**), while the reference CPT extends much
deeper (to roughly **−12.8 m NAP**, well into a "Zand Schoon" sand unit
below the organic clay). **The comparison is therefore only meaningful over
the borehole's shallower coverage interval**; below −3.18 m NAP there is no
synthetic Ic to compare against, only the standalone CPT trace. This is
visible directly in `figures/05_borehole_synthetic_vs_reference_cpt.png`
and `figures/06_ensemble_comparison.png`.

### 4.4 Ensemble definition and summary statistics

`n_realizations = 20`, `base_seed = 42`, independent per-realization seeds
via `SeedSequence.spawn` (§3.6; the 20 derived integer seeds are recorded
in `tables/ensemble_summary.json`). Band definition (from
`ensemble_summary.json`, field `band_label`):

> "Central 90% interval across 20 synthetic realizations (sampling-assumption
> variability only, not a calibrated predictive interval; tail estimates
> from n=20 are approximate)"

Available summary statistics per depth: `ic_median`, `ic_p05`, `ic_p95`
(no min/max, no mean/std — §3.6).

### 4.5 What the ensemble variability represents (and excludes)

The ensemble band reflects **only** the variability introduced by the
composition-sampling assumptions (AR(1) noise realizations within each
class's polygon, §3.3) for a *fixed, already-resolved* borehole layer log.
It explicitly does **not** include: uncertainty in the soil-name-to-class
mapping itself, uncertainty in the layer boundary depths as logged, spatial
variability between the borehole and CPT locations (2.88 m apart), or any
uncertainty in the Ic target/envelope values themselves (§1, all flagged as
assumptions). The comparison-summary statistics
(`tables/ensemble_comparison_summary.csv` / `ensemble_summary.json`) make
the resulting gap explicit:

| group | n samples | median band width | median (synthetic − reference) | fraction of reference inside band |
|---|---|---|---|---|
| overall | 515 | 0.120 | −0.086 | **21.7%** |
| transition (all classes) | 48 | 0.114 | +0.090 | 25.0% |
| Ks1 | 331 | 0.127 | −0.086 | 21.5% |
| Ks2 | 65 | 0.106 | −0.127 | 12.3% |
| Kz2 | 20 | 0.117 | −0.226 | 15.0% |
| Zk | 6 | 0.208 | −0.059 | 66.7% |
| Zs1 | 45 | 0.095 | −0.105 | 31.1% |

Only about a fifth of the measured reference Ic samples fall inside the
synthetic ensemble's central-90% band overall — i.e. the sampling-assumption
variability captured by the ensemble is **narrow relative to the actual
gap** between the class-representative Ic targets and the measured CPT at
this site. This is an honest, already-computed result, not a new finding
produced for this handover — see §5.

---

## 5. Verification and limitations

### 5.1 Checks already performed, and their results

**Mineral Ic surface** (`ic_diagnostic.py`, outputs in `figures/` and
`tables/mineral_ic_diagnostic_summary.csv`):
- Per-class sampled Ic statistics (min/median/max over the class polygon)
  compared against the target/envelope table. **Several classes show
  sampled values outside their own diagnostic envelope**: e.g. Zs3 —
  15.0% of sampled points outside envelope (worst departure 0.20 Ic
  units); Zk — 4.3% outside (worst departure 0.029); Zs1 — 2.5% outside;
  Zs2 — 2.9% outside; Kz2 — 2.9% outside (worst departure 0.078). All
  other classes show 0% outside. This is a genuine, already-flagged
  numerical limitation of the mineral surface, not a hypothesis raised by
  this handover — it is directly present in the existing diagnostic
  output.
- The mineral surface is explicitly documented as **not guaranteed
  monotonic** in any composition direction; `ic_diagnostic.py` includes a
  reversal-counting check for this (present in the script; its numeric
  output was not re-run for this handover — treat as "instrumented but not
  re-executed here").

**Peat Ic surface** (`verify_peat_ic.py`, 9 checks, all reported PASS in
the prior session that built it): boundary-control reproduction (10/10
exact), centroid-target reproduction (5/5 exact), no NaN over 1187 interior
samples, continuity at all 6 shared internal vertices, generated
[min,median,max] within target range per class, zero class-range
violations, zero boundary-conflict flags, Python-vs-JS agreement (max
diff ≈ 8.88e-16), correct `None`/text-only behaviour for h1/h2/h3, and zero
gradient (monotonicity) violations across 21 interpolation triangles.

**Borehole worked example** (`verification.py`, `tables/borehole_example_verification_report.txt`):
**8/8 PASS** — valid layer intervals (7 layers), vertical alignment (max
error 0.00e+00), composition sums to 100% (max error 2.84e-14 over 516
samples), all 468 non-transition samples inside their declared polygon,
Ic agreement (0.00e+00), reproducibility under the same seed (0.00e+00),
zero NaN where supported, zero unsupported layers (trivially preserved).

**Ensemble** (`ensemble_summary.json`, `checks` field): reproducibility
under the same base seed (0.00e+00), all 190 realization pairs produce
distinct streams, percentile ordering (`p05 <= median <= p95`) holds
everywhere, finite where supported / gap where unsupported, and same-class
boundaries correctly show no artificial transition (1/1).

### 5.2 Numerical verification vs. validation against measured data

Everything in §5.1 is **internal numerical verification**: it confirms the
code does what it is designed to do (reproduces its own control points,
agrees with itself under the same seed, stays continuous, etc.). It is
**not** validation against independent measured ground truth. The one place
measured data enters at all is the §4 worked-example comparison against the
reference CPT, and there the result is a **large, quantified gap** (§4.5:
only ~22% of measured samples fall inside the synthetic ensemble's
central-90% band) — this should be read as an honest indication that the
current Ic target values (all assumptions, §1.5) and/or the sampling model
(§3.3, uncalibrated) do not yet reproduce this particular measured CPT
well, not as a validated predictive tool.

### 5.3 Remaining assumptions and documented interpretation choices

- All per-class Ic centre/min/max values for all three triangles are
  assumptions, not calibrated correlations (§1.2, §1.5).
- The gravel g1/g2/g3 modifier sub-boundaries (grind = 5%, 15%) were not
  explicitly confirmed against the source figure by the user who digitized
  the triangle — noted in the JSON's own `g1_g2_g3_note` as "this
  digitization's own assumption," used only to keep those regions visible;
  it has no effect on the Ic mapping (which only covers Gs/Gz1-4).
  **[verified: `data/nen5104_lutum_silt_zand_grind.json`]**
  Similarly the mineral triangle's digitization note states some grid
  values and one non-grid divider line (Lz1/Lz3) were user-provided
  readings of a source figure rather than machine-digitized.
- The composition-sampling process (`CORRELATION_LENGTH_M`,
  `SPREAD_FRACTION`) is explicitly documented as an uncalibrated prototype
  assumption, not fit to real BRO/CPT statistics (§3.3).
- The ensemble band is explicitly documented as capturing sampling
  variability only, with n=20 tail estimates called "approximate" in the
  code's own `band_label` string (§4.4).

### 5.4 Incomplete routing / integration gaps

- Gravel and peat Ic surfaces are fully built and independently verified,
  but **not yet connected** to the borehole-to-Ic workflow — grind/veen
  soil descriptions are still marked unsupported by `soil_mapping.py`
  regardless of the newer triangles' existence (§3.2, §3.9).

### 5.5 Discrepancies between implementation and documentation

- **`src/modules/NEN_graph/README.md` is outdated** and should not be
  cited as current scope. It references a non-existent file
  (`data/nen5104_triangle.json`; the actual file is
  `nen5104_lutum_silt_zand.json`) and explicitly lists "Ic mapping" and
  "peat, gravel" triangles under "Out of scope (this phase)" — both of
  which now exist and are documented in this handover. Do not use this
  README as a source of current facts without cross-checking code/JSON.
- **Gravel chart's Robertson contour constants are out of range for
  gravel's own data.** `gravel_visualization.py` defines
  `ROBERTSON_IC_CONTOURS = (2.05, 2.60, 2.95)` (copied from the mineral
  chart's scale) together with `COLORSCALE_IC_MIN = 1.00` /
  `COLORSCALE_IC_MAX = 1.60`. Since the gravel Ic field never reaches
  2.05, these "Robertson contour" lines cannot appear anywhere on the
  gravel chart — confirmed visually in
  `figures/02_gravel_triangle_ic_static.png`, where only the chart's own
  `EXTRA_IC_CONTOURS = (1.31,)` reference line is visible. This is a
  latent, harmless-but-misleading constant (no crash, just an invisible,
  meaningless contour parameter) worth correcting or removing if the
  gravel chart is revisited.
- **Mineral triangle previously had no static-PNG Ic chart from the
  plotting module** (only Plotly HTML). This has been closed during this
  handover round by adding `plot_nen_triangle_with_ic_static()` to
  `visualization.py` (§2.5) — a visualization-only addition using the
  existing `IcSurface` evaluator, not a modelling change. The mineral,
  gravel, and peat triangles now all export matching HTML + PNG pairs.
  The older `ic_diagnostic_figure.png` (matplotlib-internal re-triangulation,
  §2.5) remains as a separate diagnostic artifact, not a replacement.

---

## 6. Relevant existing figures — filenames, contents, and how to read them

All files below are unmodified copies of existing repository artifacts,
except the six CSV tables in §1.2/§2.2 which are freshly transcribed
(not recomputed) directly from the source code constants named in each
file's own header comment.

| File | What it shows | How to interpret it |
|---|---|---|
| `figures/01_mineral_triangle_ic_interactive.html` | Mineral triangle, full Plotly interactive chart with Ic colour field + Robertson contour lines (2.05/2.60/2.95) + NEN polygon boundaries + class labels; hovering any point shows composition and the exact interpolated Ic. | Open in a browser. Source: `plot_nen_triangle_with_ic()` in `visualization.py`. Same evaluator as Python (§2.4). |
| `figures/01_mineral_triangle_ic_static.png` | Static matplotlib PNG of the same canonical Ic field, in the same format as the gravel/peat static charts: viridis colormap, colourbar "Representative Ic" (range 1.50-3.55), black Robertson contours (2.05/2.60/2.95), NEN polygon boundaries + class-code labels, and axis names (Zand/Silt/Lutum %) on all three edges. | Produced by the newly added `plot_nen_triangle_with_ic_static()` in `visualization.py` (§2.5); numerically identical to the interactive HTML (same `IcSurface` evaluator and grid-sampling helper `_sample_ic_grid`). |
| `figures/01_mineral_triangle_ic_static_diagnostic.png` | Static equivalent view of the same canonical Ic field (turbo colormap, 100 fill levels) with Robertson contours and NEN polygon boundaries/labels overlaid. | Produced by `ic_diagnostic.py::plot_ic_diagnostic_figure()`. Values are the canonical surface's; the contour rendering mesh is matplotlib's own re-triangulation (§2.5), not the authoritative one — do not use this image to read off exact interpolation-simplex boundaries. Kept as a separate diagnostic artifact alongside the new static chart above. |
| `figures/01_mineral_triangle_method_diagram_control_points.png` | **Method diagram**: every control point used to build the mineral surface, colour/marker-coded by type (14 class-representative circles, 15 perimeter-anchor squares, 3 corner triangles), with the actual Delaunay triangulation edges drawn in light grey and the NEN polygon boundaries in black. | Produced by `ic_diagnostic.py::plot_control_point_diagnostic_figure()`. This is the clearest existing illustration of the interpolation method described in §2. No equivalent diagram currently exists for the gravel/peat `RestrictedIcSurface` method — the class/boundary-control tables in `tables/gravel_shared_boundary_controls.csv` and `tables/peat_shared_boundary_controls.csv` are the textual equivalent. |
| `figures/02_gravel_triangle_ic_static.png` | Gravel triangle, static Ic field (viridis, colourbar 1.00-1.60) over Gs/Gz1-4 only; g1/g2/g3 shown hatched/unfilled (no Ic); a dashed blue "1.31" reference contour is visible; **no Robertson contour lines are visible** (§5.5 discrepancy). | Produced by `plot_gravel_triangle_with_ic_static()` in `gravel_visualization.py`. |
| `figures/02_gravel_triangle_ic_interactive.html` | Interactive Plotly equivalent of the above, with hover values. | Same evaluator as the static PNG and as Python (§2.4). |
| `figures/03_peat_triangle_ic_static.png` | Peat/veen triangle, static Ic field over Vm/Vk1/Vk3/Vz1/Vz3 only (colourbar 3.55-4.40); h1/h2/h3 unfilled; reference contours at 3.80/4.00/4.20 (explicitly not Robertson boundaries) plus a Robertson-derived 3.60 contour. | Produced by `plot_peat_triangle_with_ic_static()` in `peat_visualization.py`. |
| `figures/03_peat_triangle_ic_interactive.html` | Interactive Plotly equivalent. | Same evaluator as the static PNG and as Python. |
| `figures/04_triangle_sampled_compositions_interactive.html` | Mineral triangle with the actual sampled composition points (from the BHR000000397320/CPT000000061801 worked example, §4) scattered on top of the Ic field, coloured/labelled by source layer — shows the AR(1) sampler's actual output distribution within each class polygon. | Generated as part of `run_borehole_to_ic_profile.py`'s output (`plotting.py`). Reflects `random_seed=42` (§3.3/§4). |
| `figures/05_borehole_synthetic_vs_reference_cpt.png` | Four-panel figure: (1) borehole NEN class log by elevation, (2) single-realization synthetic Ic profile, (3) reference Lengkeek CPT classification + measured Ic, (4) synthetic vs. reference Ic overlay. | Generated by `run_borehole_to_ic_profile.py` / `plotting.py`. Elevation axis in m NAP; only the top ≈13.5 m (borehole's depth range) has a synthetic curve — see §4.3 for why the CPT trace continues much deeper with no synthetic counterpart. Uses `random_seed=42`. |
| `figures/06_ensemble_comparison.png` | Same 3-panel layout but the synthetic panel shows all 20 individual realizations (thin grey), the ensemble median (thick blue), and the central 90% band (light blue shading), against the same reference CPT trace. | Generated by `run_ensemble_borehole_to_ic_profile.py`. `n_realizations=20`, `base_seed=42` (see `tables/ensemble_summary.json` for the 20 derived per-realization seeds). See §4.4-§4.5 for how to read the band and the quantified overlap-with-reference statistics. |

### Numerical tables included

- `tables/mineral_class_ic_table.csv`, `gravel_class_ic_table.csv`,
  `peat_class_ic_table.csv` — final per-class min/centre/max Ic tables (§1.2).
- `tables/mineral_control_points_all32.csv` — every control point (all 32:
  14 class + 15 perimeter-anchor + 3 corner) used to build the mineral
  surface, an unmodified copy of `ic_control_points.csv`.
- `tables/gravel_shared_boundary_controls.csv`,
  `peat_shared_boundary_controls.csv` — the explicit boundary-control
  tables for the gravel/peat `RestrictedIcSurface`s (§1.2, §2.2).
- `tables/mineral_ic_diagnostic_summary.csv` — unmodified copy of the
  mineral surface's per-class diagnostic statistics, including the
  envelope-violation percentages discussed in §5.1.
- `tables/borehole_example_metadata.json`, `borehole_example_verification_report.txt`,
  `borehole_example_layer_table.csv` — unmodified copies of the worked
  example's generated outputs (§4).
- `tables/ensemble_summary.json`, `ensemble_comparison_summary.csv` —
  unmodified copies of the ensemble's generated outputs, including the
  realization seeds, band definition, comparison statistics, and checks
  (§4.4-§4.5).

---

## 7. Source file index (for further reading)

| File | Role |
|---|---|
| `src/modules/NEN_graph/ic_mapping.py` | Mineral triangle `IcSurface` |
| `src/modules/NEN_graph/gravel_ic_mapping.py` | Generic `RestrictedIcSurface`; gravel class/boundary tables |
| `src/modules/NEN_graph/peat_ic_mapping.py` | Peat class/boundary tables (reuses `RestrictedIcSurface`) |
| `src/modules/NEN_graph/visualization.py` | Mineral triangle plotting incl. `_build_hover_script` (shared JS evaluator) |
| `src/modules/NEN_graph/gravel_visualization.py`, `peat_visualization.py` | Gravel/peat plotting |
| `src/modules/NEN_graph/ic_diagnostic.py` | Mineral surface diagnostics (figures + summary CSV used in §5, §6) |
| `src/modules/NEN_graph/verify_peat_ic.py` | Peat surface verification script (§5.1) |
| `src/modules/NEN_graph/classification.py`, `coordinates.py`, `triangle_definition.py` | Shared geometry/classification/coordinate machinery |
| `src/modules/borehole_ic/xml_readers.py` | BRO borehole/CPT XML parsing |
| `src/modules/borehole_ic/soil_mapping.py` | Soil-name → NEN class resolution |
| `src/modules/borehole_ic/composition_sampling.py` | AR(1) composition sampler |
| `src/modules/borehole_ic/profile_builder.py` | Depth-grid + transition + orchestration |
| `src/modules/borehole_ic/lengkeek_reference.py` | Reference CPT via `geolib_plus` |
| `src/modules/borehole_ic/ensemble.py` | Multi-realization ensemble + comparison stats |
| `src/modules/borehole_ic/verification.py` | 8-check verification of a built profile |
| `src/modules/borehole_ic/plotting.py` | Worked-example figures |
| `src/run_borehole_to_ic_profile.py`, `src/run_ensemble_borehole_to_ic_profile.py` | Entry-point scripts for the worked example / ensemble |

---

**No modelling code was changed and no experiments were run to produce this
document; nothing was committed or pushed.**
