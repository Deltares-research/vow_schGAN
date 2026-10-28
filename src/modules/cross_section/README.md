# Cross-sections → SchemaGAN: step-by-step guide

How to turn a cross-section (a profile line + nearby CPTs and boreholes) into a
SchemaGAN input matrix, run SchemaGAN, and mask the result at the ground surface.
This reproduces exactly what was done for `cs_O62,7`. All design decisions are
fixed in the code, so every cross-section is processed the same way.

All commands are run in PowerShell from the repository root `D:\codes\vow_schGAN`.

---

## 1. Folder layout (one-time)

Everything under `data\` is local input/output and is **not** tracked by git.

```
D:\codes\vow_schGAN\
  data\
    cs\
      cs_guide.csv                    master list of cross-sections (one row each)
      cs_O62,7\                       one folder per cross-section (any name)
        cs_O62,7_profile.csv          the profile line of that cross-section
    coords\                           coordinate CSVs (name, x, y) for all tests
    cpts\                             CPT files (.gef or BRO .xml), subfolders allowed
    boreholes\                        borehole BRO .xml files (BHR...)
    models\
      schemaGAN.h5                    the trained SchemaGAN model
```

---

## 2. Add a new cross-section

### 2a. Profile CSV

Put the profile in its own folder under `data\cs\`, e.g.
`data\cs\cs_O63,5\cs_O63,5_profile.csv`.

```
dist,x,y,elev_NAP
0,153579.7,440358.0175,3.026
0.589,153580.0104,440358.5183,3.019
...
```

- `x`, `y`: RD coordinates (required). The first point is **A**, the last point is **B**;
  tests are projected onto the straight line A–B. A warning is printed if the
  profile deviates more than 1 m from that straight line.
- `elev_NAP`: ground level in m NAP (required for the ground line and the mask).
  Empty values are allowed: gaps are filled by straight-line interpolation.
- `dist`: distance along the line (optional; computed from x/y if missing).
- The file name must be unique within `data\cs\` (it is found by name).
- Column names are not case-sensitive; `,` or `;` as separator both work.

### 2b. Row in `cs_guide.csv`

```
cs_id,cs_name,cs_profile,cpts_in_cs
1,"62,7","cs_O62,7_profile.csv","BHR000000397320, CPT000000061801_IMBRO_A, BHR000000397307, BHR000000397319"
2,"63,5","cs_O63,5_profile.csv","CPT..., BHR..., ..."
```

- `cs_profile`: file name of the profile CSV (just the name, not the path).
- `cpts_in_cs`: the tests, comma-separated **inside quotes**.
  - Names starting with `BHR` are boreholes (converted to Ic, see §4).
  - All other names are CPTs (`.gef` or `.xml` in `data\cpts\`).
  - A name may be the full file name without extension (`CPT000000061801_IMBRO_A`)
    or just the BRO id (`CPT000000061801`).

### 2c. Coordinates and files

Every test name must be:
- in one of the coordinate CSVs in `data\coords\` (columns `name, x, y`, any CSV
  there is read), and
- present as a file in `data\cpts\` (CPT) or `data\boreholes\` (borehole).

Missing ones are not fatal: they are reported (see §6) and left out of the cross-section.

---

## 3. Step 1 – cross-section plot and SchemaGAN input (runs anywhere)

```powershell
.\.venv\Scripts\python.exe src\run_cross_section_plot.py --cs-id 2
```

- Without `--cs-id` every row of `cs_guide.csv` is processed.
- `--cs-id` / `--cs-name` can be repeated to select several rows.
- Takes about half a minute per cross-section.

Outputs, written next to the profile CSV (`<stem>` = profile file name without `_profile`):

| File | Content |
|---|---|
| `<stem>_cross_section.png` | tests along the line, distance vs m NAP, ground line, design bottom, SchemaGAN frame, augmented copies |
| `<stem>_tests.csv` | one row per test incl. copies: position, offset, ground level, `status` |
| `<stem>_ic_profiles.csv` | all Ic values per test (incl. copies) |
| `<stem>_profile_filled.csv` | profile with gaps interpolated (only if it had gaps) |
| `<stem>_schemagan_input.csv` | **the SchemaGAN input**: 32 lines × 512 values, no header; Ic or 0 |
| `<stem>_schemagan_input_info.txt` | frame, rules, which test fills which column, row elevations |
| `<stem>_schemagan_input.png` | picture of the input matrix |
| `<stem>_schemagan_frame.json` | frame data used by step 2 |

**Check before step 2:**
1. Open `<stem>_cross_section.png`: are the tests where you expect them, is the ground line right?
2. Open `<stem>_tests.csv`: every real test should have `status = ok`.
3. The terminal prints `occupied columns` (should be about 6) and which test was dropped
   because it shared a column with another one.

---

## 4. Step 2 – run SchemaGAN and mask (run in your own terminal)

TensorFlow needs your user folder, so run this in your own PowerShell window, not
through the Copilot sandbox:

```powershell
.\.venv\Scripts\python.exe src\run_schemagan_generation.py --cs-folder "data\cs\cs_O63,5"
```

`--cs-folder` is the folder containing `<stem>_schemagan_input.csv`. Always run
step 1 first (step 2 reads the frame file step 1 writes).

Outputs in the same folder (overwritten on every run):

| File | Content |
|---|---|
| `<stem>_schemagan_output_raw.csv` | 32 × 512 Ic straight from SchemaGAN |
| `<stem>_schemagan_output_masked.csv` | same, cells above the ground surface empty |
| `<stem>_schemagan_output.png` | input / raw / masked / masked in m NAP (viridis colours) |
| `<stem>_schemagan_classes_output.png` | raw and masked result in the six soil classes |

The terminal prints the output Ic range, the number of masked cells and how well the
output reproduces the input at the test columns (mean difference should be ≈ 0.02).

The three `WARNING:tensorflow: ... is deprecated` lines are harmless.

### All cross-sections at once

```powershell
.\.venv\Scripts\python.exe src\run_cross_section_plot.py
Get-ChildItem data\cs -Recurse -Filter *_schemagan_input.csv | ForEach-Object {
    .\.venv\Scripts\python.exe src\run_schemagan_generation.py --cs-folder $_.DirectoryName
}
```

### Redraw only the class figure (no TensorFlow needed)

```powershell
.\.venv\Scripts\python.exe src\run_schemagan_generation.py --cs-folder "data\cs\cs_O63,5" --classes-plot-only
```

---

## 5. Fixed design decisions (and where to change them)

| Step | Decision | Where |
|---|---|---|
| Line | Straight line from first to last profile point; tests projected perpendicular, all kept | `cs_data.py` |
| Ground | Gaps in `elev_NAP` interpolated linearly (flat beyond the ends) | `cs_data.fill_elevation_gaps` |
| Real CPT Ic | geolib_plus Ic (Lengkeek 2024 interpretation, default groundwater) | `borehole_ic/lengkeek_reference.py` |
| Borehole Ic | Synthetic Ic from the NEN 5104 triangles, one realization, seed 42; unknown names use a generic class (`Zand*`, `Veen*`, …) | `borehole_ic/` |
| Frame top | Highest ground level of the tests | `cs_data.schemagan_frame` |
| Frame bottom | −5 m NAP | `cs_data.DESIGN_BOTTOM_M_NAP` |
| Frame size | 512 columns (start to end of the line) × 32 rows | `cs_data.SCHEMAGAN_N_COLS / _ROWS` |
| Augmentation | Copies of the nearest test at 5 m from the start and 10 m and 50 m from the end; real CPT preferred only when in the same column; copies keep the NAP elevations | `cs_augment.AUGMENT_LEFT_FROM_START_M`, `AUGMENT_RIGHT_FROM_END_M` |
| Column | `floor(distance / column width)`; one test per column: real CPT > borehole > copy | `cs_matrix.build_matrix` |
| Row value | Median of the Ic samples in the row, only if the test covers ≥ 50 % of the row; else 0 | `cs_matrix.MIN_ROW_COVERAGE`, `column_values` |
| Top fill | Rows above a column's first value get that first value (prevents SchemaGAN edge effect) | `cs_matrix.build_matrix` |
| Normalisation | Ic 0–4.3 → −1…1 before the model, back after | `run_schemagan_generation.IC_NORMALISATION_MAX` |
| Mask | Keep a cell if its centre is at or below the ground at the column centre | `cs_mask.surface_mask` |
| Soil classes | 6 sand (< 2.05), 5 sand mixtures (2.05–2.60), 4 silt mixtures (2.60–2.95), 3a clay (2.95–3.30), 3b organic clay (3.30–3.60), 2 peat (≥ 3.60) | `modules/visualization.py`, `IC_SOIL_CLASSES` |

All files are in `src\modules\cross_section\` unless another path is given.

---

## 6. Troubleshooting

| Message / symptom | Meaning | What to do |
|---|---|---|
| `status = no_coordinates` in `_tests.csv` | name not found in `data\coords\*.csv` | add it to a coordinate CSV or fix the spelling in the guide |
| `status = file_not_found` | no matching `.gef`/`.xml` in `data\cpts` / `data\boreholes` | copy the file there |
| `status = error` | the test could not be read; the reason is in the `note` column | check the file |
| `profile file ... not found` / `found more than once` | `cs_profile` name wrong, or the same file name exists in two folders | fix the name / make it unique |
| `WARNING: profile points deviate up to ... m` | the profile is not a straight line | check the profile; tests are still projected on A–B |
| `could not write ...png ... is it open in another program?` | the PNG is open in a viewer | close it and rerun |
| `Expected exactly one '*_schemagan_input.csv'` in step 2 | step 1 was not run for that folder | run step 1 first |
| `PermissionError ... .keras\keras.json` | step 2 was run inside the sandbox | run it in your own terminal |
| Many `WARNING:root:Key ... is not defined in the gef file` | normal geolib_plus messages for GEF files | ignore |
