# NEN 5104 -> representative Ic workflow

This module (`src/modules/NEN_graph/`) plus its companion module
`src/modules/borehole_ic/` implement:

1. Three NEN 5104 ternary classification triangles, each with geometry,
   classification, a representative-Ic mapping, and static/interactive
   plots: **mineral** (lutum-silt-zand), **gravel** (+grind), **peat/veen**
   (organic stof-lutum-silt+zand).
2. A borehole workflow that reads BRO borehole + CPT XML, samples
   compositions inside the **mineral** triangle along the borehole depth
   axis, and produces synthetic Ic profiles (single-realization and
   ensemble), compared against a Lengkeek reference profile from the CPT.

This is a prototype: plain scripts and hand-rolled `[PASS]/[FAIL]` checks,
no pytest, no CLI framework beyond `argparse` for input/output paths.

## Directory map

```
src/modules/NEN_graph/            # 3 triangles: geometry, Ic mapping, plots
  data/*.json                     # digitized polygon vertices (source of truth, do not regenerate)
  coordinates.py                  # ternary <-> Cartesian conversions (shared)
  classification.py               # mineral triangle: polygons + classify_composition()
  ic_mapping.py                   # mineral triangle: canonical representative-Ic surface
  visualization.py                # mineral triangle plots + shared plotting helpers (used by gravel/peat too)
  ic_diagnostic.py                # mineral Ic surface diagnostics (figures + CSV summaries)
  peat_classification.py          # peat/veen triangle: polygons + classification
  peat_ic_mapping.py              # peat/veen triangle: representative-Ic surface (RestrictedIcSurface)
  peat_visualization.py           # peat/veen triangle plots
  gravel_ic_mapping.py            # gravel triangle: representative-Ic surface (RestrictedIcSurface)
  gravel_visualization.py         # gravel triangle plots
  triangle_definition.py          # shared Triangle/Polygon dataclasses + JSON loader
  verify_geometry.py              # mineral geometry checks (run directly)
  verify_peat_geometry.py         # peat geometry checks (also re-runs verify_geometry.py)
  verify_peat_ic.py               # peat Ic-surface checks (continuity, JS/Python agreement, hover text, ...)

src/modules/borehole_ic/          # borehole XML -> synthetic Ic profile (mineral triangle only)
  xml_readers.py                  # BRO borehole (BHR_G) + CPT XML parsing
  soil_mapping.py                 # BRO soilNameNEN5104 compound -> mineral-triangle class code
  composition_sampling.py         # depth-correlated composition sampling within a polygon
  profile_builder.py              # orchestrates: resolved layers -> samples -> Ic profile (1 realization)
  ensemble.py                     # repeats profile_builder across N independent random streams
  lengkeek_reference.py           # reference CPT classification via geolib_plus (Lengkeek 2024), not reimplemented
  plotting.py                     # 3-panel elevation profile + triangle-with-samples figures
  verification.py                 # checks for the single-realization profile

src/run_borehole_to_ic_profile.py           # entry point: 1 borehole + 1 CPT -> profile
src/run_ensemble_borehole_to_ic_profile.py  # entry point: N-realization ensemble for 1 borehole + 1 CPT

res/NEN_graph/{mineral,gravel,peat,diagnostics}/   # generated figures/HTML/CSVs (gitignored, reproducible)
res/borehole_ic/{borehole}_{cpt}[_ensemble]/       # generated profile outputs (gitignored, reproducible)
```

Everything under `res/` is generated output (matched by the repository's
blanket `res/` rule in `.gitignore`) and can be safely deleted/regenerated
with the commands below. Nothing required to regenerate it lives outside
`src/` and the external input XML files.

## Requirements

- The `.venv` in this repo (it has `plotly`, `shapely`, `geolib_plus`, etc.
  installed - the default system Python does not). Always invoke scripts as:
  ```
  .venv\Scripts\python.exe -m src.modules.NEN_graph.<module>   # from the repo root, or
  .venv\Scripts\python.exe src\modules\NEN_graph\<script>.py
  ```
- `geolib_plus` ("VOW modified build") is only needed for
  `lengkeek_reference.py` (the reference CPT comparison / `run_*` entry
  points). The 3 triangle modules and their verification scripts do not need it.
- BRO borehole + CPT XML files are **not** included in this repository -
  point `run_borehole_to_ic_profile.py` / `run_ensemble_borehole_to_ic_profile.py`
  at your own copies via the `--cpt-xml` / `--borehole-xml` arguments (see below).

## Commands

### Triangle plots

Each command below writes into `res/NEN_graph/<triangle>/` by default;
pass `--output-dir <folder>` to any of them to write elsewhere.

```
# Mineral (lutum-silt-zand): geometry PNG+HTML, and Ic PNG+HTML
.venv\Scripts\python.exe -m src.modules.NEN_graph.visualization

# Gravel (lutum+silt-zand-grind): Ic PNG+HTML (geometry-only chart not implemented separately)
.venv\Scripts\python.exe -m src.modules.NEN_graph.gravel_visualization

# Peat/veen (organic stof-lutum-silt+zand): geometry PNG+HTML, and Ic PNG+HTML
.venv\Scripts\python.exe -m src.modules.NEN_graph.peat_visualization

# Mineral Ic-surface diagnostics: 2 figures + 2 CSVs -> res/NEN_graph/diagnostics/
.venv\Scripts\python.exe -m src.modules.NEN_graph.ic_diagnostic
```

### Borehole XML -> synthetic Ic profile (single realization)

```
.venv\Scripts\python.exe src\run_borehole_to_ic_profile.py ^
    --cpt-xml <path\to\CPT....xml> ^
    --borehole-xml <path\to\BHR....xml> [<path\to\BHR....xml> ...] ^
    [--output-dir res\borehole_ic]
```

If more than one `--borehole-xml` candidate is given, the one closest (RD
coordinates) to the CPT is selected automatically. Outputs are written to
`<output-dir>/<borehole_bro_id>_<cpt_bro_id>/` (a `verification_report.txt`
with 8 `[PASS]/[FAIL]` checks is written there too).

### Ensemble (N realizations)

```
.venv\Scripts\python.exe src\run_ensemble_borehole_to_ic_profile.py ^
    --cpt-xml <path\to\CPT....xml> ^
    --borehole-xml <path\to\BHR....xml> ^
    [--output-dir res\borehole_ic] [--n-realizations 20] [--base-seed 42]
```

Outputs are written to `<output-dir>/<borehole_bro_id>_<cpt_bro_id>_ensemble/`.

### Batch: folder of borehole XMLs -> synthetic Ic profiles (no reference CPT)

```
.venv\Scripts\python.exe src\run_batch_boreholes_to_ic.py --input-dir <folder with BHR*.xml> --output-dir <folder> [--limit N] [--no-plots]
```

Writes `<BRO_ID>_synthetic_ic.csv` per borehole, `_plots\<BRO_ID>.png`
(borehole classes | synthetic Ic) and `_batch_summary.csv` (status, triangles
used, unsupported soil names). Unsupported layers are labelled in the plots
with their raw BRO soil name, prefixed with `?`.

### Verification

No pytest; each script below prints `[PASS]/[FAIL]` lines and exits non-zero
on any failure.

```
.venv\Scripts\python.exe src\modules\NEN_graph\verify_geometry.py      # mineral triangle geometry
.venv\Scripts\python.exe src\modules\NEN_graph\verify_peat_geometry.py # peat triangle geometry (+ re-runs verify_geometry.py)
.venv\Scripts\python.exe src\modules\NEN_graph\verify_peat_ic.py       # peat Ic surface (continuity, JS/Python agreement, hover text)
.venv\Scripts\python.exe -m src.modules.NEN_graph.ic_diagnostic        # mineral Ic surface (diagnostics, not pass/fail only)
```

The single-realization and ensemble `run_*` entry points each run their own
built-in checks and write a report/summary alongside their other outputs
(`verification_report.txt`, `ensemble_summary.json`).

**Known gap:** there is currently no dedicated Ic-surface verification
script for the **gravel** triangle (unlike mineral's `ic_diagnostic.py` and
peat's `verify_peat_ic.py`). This is a pre-existing gap, not introduced by
this cleanup, and gravel's geometry itself is not covered by any
`verify_gravel_geometry.py` either - the digitization note in
`data/nen5104_lutum_silt_zand_grind.json` refers to such a script, but it
does not exist in this repository.

## Supported classes and integration limits

- **Mineral triangle**: 14 classes (Ks1-4, Kz1-3, Lz1, Lz3, Zk, Zs1-4) -
  fully supported end-to-end, including the borehole workflow.
- **Gravel triangle**: 5 gravel main-name classes (Gs, Gz1-Gz4) plus 3
  unmapped gravel-modifier classes (g1-g3, no numerical Ic - shown as
  neutral/hatched). Main-name classes are used by the borehole generator.
- **Peat/veen triangle**: 5 veen main-name classes (Vm, Vk1, Vk3, Vz1, Vz3)
  plus 3 unmapped humus-modifier classes (h1-h3, no numerical Ic).
  Main-name classes are used by the borehole generator.

Borehole routing (`src/modules/borehole_ic/triangles.py`,
`soil_mapping.py`): the main noun of the BRO `soilNameNEN5104` selects the
triangle - klei/zand/leem -> mineral, veen -> peat, grind -> gravel - and the
reconstructed name is matched against that triangle's main-name classes
(e.g. `zwakKleiigVeen` -> "veen, zwak kleiig" -> Vk1, `mineraalarmVeen` ->
Vm, `zwakZandigGrind` -> Gz1). Each layer is sampled inside its class polygon
in its own triangle and evaluated on that triangle's Ic surface.

Still unsupported (kept as NaN gaps, never silently substituted): empty soil
names and names with no recognisable main noun (e.g. `Silt`, `onbekend`).

Generic fallback (`is_guess=True`, marked with `*`): a name that matches no
exact class but contains a main noun - e.g. `Zand`, `zandNietGespecificeerd`,
`Veen`, `matigKleiigVeen`, `humeuzeKlei`, `matigGrindigZand` - gets that
noun's generic class (`Klei*`, `Zand*`, `Leem*`, `Veen*`, `Grind*`). It is
sampled over the union of all main-name classes of that noun in its triangle
(e.g. `Zand*` = Zk + Zs1-Zs4), so its Ic spread is wider than for an exact
class. Any secondary modifier in the raw name (e.g. "matig grindig") is
ignored. Plots show the raw name in the borehole column and the generic code
in the Ic panel; `_batch_summary.csv` lists them in `guessed_names`.

Borehole plot colours: zand = yellows, klei = greens, veen = reds,
leem = browns, grind = slate greys (`CLASS_COLORS` in `triangles.py`; the
triangle charts keep their own palettes).

Transitions between layers in the same triangle blend the sampled
composition points; transitions between different triangles (e.g. Kz1 -> Vk1)
blend the two Ic values with the same smoothstep weight and carry no
composition. The interactive sampled-composition triangle plot shows mineral
samples only.

## Digitization / geometry notes (mineral triangle)

The polygon vertices were reconstructed from user-supplied grid-line values
and the official NEN 5104 class table, then verified to share exact edges
between neighboring polygons (`verify_geometry.py`). The split between
**Ks4** (clay, extremely silty) and **Kz1** (clay, slightly sandy) along the
sand = 15 grid line was inferred from the visual layout of the original
figure rather than stated in words - geometrically self-consistent
(edge-matching passes) but flagged here in case it needs to be
double-checked against the printed standard. See the module docstrings and
`verify_geometry.py`/`verify_peat_geometry.py` source for the full list of
checks. The gravel/peat triangles follow the same digitization and
verification approach (`data/*.json` `digitization_note` fields).

## Usage (Python API)

```python
from modules.NEN_graph.classification import classify_composition

result = classify_composition(sand=30, silt=60, lutum=10)
# {'status': 'inside', 'class_code': 'Kz3', 'class_name': 'Clay, strongly sandy', ...}

from modules.NEN_graph.ic_mapping import get_ic_surface, calculate_mean_ic
ic = calculate_mean_ic({"sand": 30, "silt": 60, "lutum": 10})
```

