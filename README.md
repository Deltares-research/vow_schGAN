# VOW SchemaGAN Pipeline

A complete automated workflow for preparing Cone Penetration Test (CPT) data and generating geotechnical subsurface schematics using **SchemaGAN**, a generative adversarial network trained for soil schematization in the **VOW project**.

---

## 🎯 Overview

This pipeline transforms raw CPT data (`.gef` files) into detailed subsurface cross-sections through a 6-step automated process:

1. **Folder Setup** - Creates organized experiment directory structure
2. **Coordinate Extraction** - Validates and extracts CPT locations
3. **Data Processing** - Interprets CPT data and compresses to 32-pixel depth profiles
4. **Section Creation** - Generates overlapping spatial sections ready for GAN input
5. **Schema Generation** - Uses trained SchemaGAN model to create detailed schematics
6. **Mosaic Assembly** - Combines all sections into a complete subsurface visualization

**Key Features:**
- ✅ Fully automated end-to-end pipeline (`main_processing.py`)
- ✅ Comprehensive logging saved with each experiment
- ✅ Handles coordinate validation and correction for Netherlands RD system
- ✅ Configurable sectioning with overlap for seamless mosaics
- ✅ Individual scripts can be run standalone for debugging

---

## 📋 Prerequisites

### Required Software
- **Python 3.10** (recommended)
- **GEOLib-Plus** - Local installation required for CPT interpretation
  - Path must be added in scripts: `D:\GEOLib-Plus` (configurable)
- **Trained SchemaGAN Model** - `.h5` file for schema generation

### Required Data
- CPT data files in **GEF format** (`.gef`)
- Coordinates should be in **Netherlands RD coordinate system**

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
py -3.10 -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Your Experiment

Edit `src/main_processing.py` - **CONFIG section** (lines 40-65):

```python
# Base configuration
RES_DIR = Path(r"C:\VOW\res")          # Where results will be saved
REGION = "north"                        # Region name (north/south/etc.)
EXP_NAME = "exp_1"                      # Experiment identifier
DESCRIPTION = "Baseline with 6 CPTs"   # Optional description

# Input paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts\your_data_folder")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")

# Processing parameters
COMPRESSION_METHOD = "mean"             # "mean" or "max" for IC compression
N_COLS = 512                            # Width of output sections (pixels)
N_ROWS = 32                             # Depth levels (pixels)
CPTS_PER_SECTION = 6                    # CPTs in each section
OVERLAP_CPTS = 2                        # Overlap between sections
LEFT_PAD_FRACTION = 0.10                # Padding fractions
RIGHT_PAD_FRACTION = 0.10
DIR_FROM, DIR_TO = "west", "east"       # Sorting direction

# Optional: depth range (auto-computed if None)
Y_TOP_M = None
Y_BOTTOM_M = None
```

### 3. Run the Complete Pipeline

```bash
python src/main_processing.py
```

**Output Structure:**
```
C:\VOW\res\north\exp_1\
├── pipeline.log                  # Complete execution log
├── README.txt                    # Experiment metadata
├── 1_coords\
│   └── cpt_coordinates.csv      # Validated coordinates
├── 2_compressed_cpt\
│   └── compressed_cpt_data_mean.csv  # 32-row IC profiles
├── 3_sections\
│   ├── section_01_cpts_XXX_to_YYY.csv  # Individual sections
│   ├── manifest_sections.csv          # Section metadata
│   └── cpt_coords_with_distances.csv  # Spatial distances
├── 4_gan_images\
│   ├── section_01_..._gan.csv         # Generated schemas (data)
│   └── section_01_..._gan.png         # Generated schemas (images)
└── 5_mosaic\
    ├── schemaGAN_mosaic.csv           # Combined mosaic (data)
    └── schemaGAN_mosaic.png           # Final visualization
```

---

## 📚 Detailed Script Documentation

### Main Pipeline: `main_processing.py`

**Purpose:** Orchestrates the complete workflow  
**Usage:** Configure CONFIG section and run directly  
**Logging:** Saves detailed logs to `<experiment_folder>/pipeline.log`

**Key Functions:**
- `run_coordinate_extraction()` - Calls `extract_coords.py`
- `run_cpt_data_processing()` - Calls `extract_data.py`
- `run_section_creation()` - Calls `create_schGAN_input_file.py`
- `run_schema_generation()` - Inline SchemaGAN prediction
- `run_mosaic_creation()` - Inline mosaic assembly

---

### Step 1: `utils.py` - `setup_experiment()`

**Purpose:** Creates standardized folder structure for experiments

**Function:**
```python
setup_experiment(base_dir, region, exp_name, description)
```

**Creates:**
- `1_coords/` - For coordinate files
- `2_compressed_cpt/` - For processed CPT data
- `3_sections/` - For GAN input sections
- `4_gan_images/` - For generated schemas
- `5_mosaic/` - For final mosaic output
- `README.txt` - Experiment metadata

---

### Step 2: `extract_coords.py`

**Purpose:** Extract and validate CPT coordinates from GEF files

**Main Function:**
```python
process_cpt_coords(cpt_folder: Path, output_csv: Path)
```

**Process:**
1. Reads all `.gef` files in folder
2. Checks for valid coordinates (filters out 0.0 or NaN values)
3. Validates Netherlands RD format (6-digit coordinates)
4. Auto-corrects common scaling errors (e.g., 123.456 → 123456.000)
5. Moves invalid files to `no_coords/` subfolder
6. Saves results to CSV with columns: `name, x, y, fixed`

**Output:** `cpt_coordinates.csv` with validation flags

**Standalone Usage:**
```python
if __name__ == "__main__":
    GEF_FOLDER = Path(r"C:\VOW\data\test_cpts")
    OUT_CSV = Path(r"C:\VOW\res\coordinates_cpts_test_result.csv")
    process_cpt_coords(GEF_FOLDER, OUT_CSV)
```

---

### Step 3: `extract_data.py`

**Purpose:** Interpret CPT data and compress to 32-pixel depth profiles

**Main Function:**
```python
process_cpts(gef_list) → data_cpts, coords
```

**Process:**
1. **Interpret CPTs** using GEOLib-Plus Robertson method
   - Calculates soil behavior index (IC)
   - Applies unit weight calculations
2. **Equalize Top** - Aligns all CPTs to same starting depth
3. **Equalize Depth** - Extends all CPTs to same ending depth (fills with zeros)
4. **Compress to 32 pixels** - Aggregates IC values using `mean` or `max` method
5. Saves compressed data with columns: `Depth_Index (0-31), CPT1, CPT2, ...`

**Key Functions:**
- `equalize_top(data_cpts)` - Trims to common start depth
- `equalize_depth(data_cpts, target_depth)` - Extends to common end depth
- `compress_to_32px(data_cpts, method="mean")` - Aggregates to 32 rows
- `save_cpt_to_csv(data_cpts, output_folder, filename)` - Exports results

**Output:** `compressed_cpt_data_mean.csv` (or `_max.csv`)

**Compression Methods:**
- `"mean"` - Average IC value within each depth bin (smoother)
- `"max"` - Maximum IC value within each depth bin (preserves extremes)

---

### Step 4: `create_schGAN_input_file.py`

**Purpose:** Create spatial sections with overlapping CPTs for GAN input

**Main Function:**
```python
process_sections(coords_df, cpt_df, out_dir, n_cols, n_rows, 
                per, overlap, left_pad_frac, right_pad_frac, 
                from_where, to_where) → manifest
```

**Process:**
1. **Sort CPTs** by spatial direction (west→east, south→north, etc.)
2. **Compute Distances** between consecutive CPTs
3. **Create Sliding Window** with specified overlap
   - Example: 6 CPTs per section, 2 overlap → sections share 2 CPTs
4. **Map to Pixel Grid** (512 × 32)
   - Interpolates CPT positions to column indices
   - Adds left/right padding for context
5. **Handle Collisions** - When multiple CPTs map to same column, uses weighted average
6. **Save Sections** as CSV matrices (rows=depth, cols=distance)

**Output Files:**
- `section_01_cpts_CPT001_to_CPT006.csv` - Individual section matrices
- `cpt_coords_with_distances.csv` - CPT positions with cumulative distances
- `manifest_sections.csv` - Metadata for each section

**Manifest Columns:**
- `section_index` - Section number
- `start_idx`, `end_idx` - CPT indices in this section
- `span_m` - Real-world span between first and last CPT
- `left_pad_m`, `right_pad_m` - Padding distances
- `skipped_count` - CPTs without data (filled with zeros)
- `csv_path` - Path to section file

---

### Step 5: Schema Generation (Inline in `main_processing.py`)

**Purpose:** Generate detailed subsurface schemas using trained SchemaGAN

**Function:**
```python
run_schema_generation(sections_folder, gan_images_folder, 
                     model_path, y_top_m, y_bottom_m)
```

**Process:**
1. **Load Model** - Trained SchemaGAN generator (`.h5` file)
2. **Set Seed** - Random or fixed for reproducibility
3. **For Each Section:**
   - Load section CSV (512 × 32 matrix with IC values and zeros)
   - **Normalize** IC values from [0, 4.3] to [-1, 1]
   - **Predict** using GAN generator
   - **Denormalize** back to IC values
   - Save as CSV and PNG with proper axes

**PNG Features:**
- **Bottom axis:** Distance along line (meters)
- **Top axis:** Pixel index (0-511)
- **Left axis:** Depth index (0-31)
- **Right axis:** Real depth (meters, e.g., 6.8m to -13.1m)
- **Colorbar:** IC values (0 to 4.5)
- **Colormap:** Viridis (green=low IC, yellow=high IC)

**Output:**
- `section_XX_cpts_YYY_to_ZZZ_seedNNNN_gan.csv` - Schema data
- `section_XX_cpts_YYY_to_ZZZ_seedNNNN_gan.png` - Schema visualization

---

### Step 6: Mosaic Creation (Inline in `main_processing.py`)

**Purpose:** Combine all generated sections into seamless mosaic

**Function:**
```python
run_mosaic_creation(sections_folder, gan_images_folder, 
                   mosaic_folder, y_top_m, y_bottom_m)
```

**Process:**
1. **Find All GAN CSVs** - Matches pattern `section_*_gan.csv`
2. **Compute Global Grid:**
   - Finds min/max extent across all sections
   - Uses median pixel size (dx) for consistency
3. **Accumulate Sections** with bilinear interpolation:
   - Maps each section to global coordinate system
   - Handles overlaps with weighted averaging
4. **Normalize** - Divides by weight accumulator
5. **Plot** with proper axes (same as individual schemas)

**Output:**
- `schemaGAN_mosaic.csv` - Complete mosaic data (32 rows × variable columns)
- `schemaGAN_mosaic.png` - Final visualization

**Mosaic Advantages:**
- Seamless blending at section boundaries
- Consistent spatial scale throughout
- Complete subsurface profile from all available CPTs

---

### Supporting Utilities: `utils.py`

**Key Functions:**

```python
# File operations
read_files(path, extension=".gef") → list[Path]

# Distance calculation
euclid(x1, y1, x2, y2) → float

# IC normalization for GAN (Robertson scale: 0-4.3)
IC_normalization(data) → [src_norm, tar_norm]  # [0,4.3] → [-1,1]
reverse_IC_normalization(data) → data          # [-1,1] → [0,4.3]

# Experiment setup
setup_experiment(base_dir, region, exp_name, description) → folders_dict
```

---

## 🔧 Configuration Guide

### GEOLib-Plus Path

**Location to Update:**
- `extract_coords.py`: Line 11
- `extract_data.py`: Line 14
- `main_processing.py`: Line 35
- `utils.py`: Line 8

```python
sys.path.append(r"D:\GEOLib-Plus")  # Change to your installation path
```

### Processing Parameters

**Compression Method:**
- `"mean"` - Better for smooth transitions, general soil behavior
- `"max"` - Preserves strong layers, better for stiff/dense zones

**Section Configuration:**
- `CPTS_PER_SECTION = 6` - More CPTs = better detail, larger file size
- `OVERLAP_CPTS = 2` - More overlap = smoother mosaic, more computation
- Typical: 6 CPTs with 2 overlap gives 67% coverage overlap

**Padding:**
- `LEFT_PAD_FRACTION = 0.10` - Adds 10% context before first CPT
- `RIGHT_PAD_FRACTION = 0.10` - Adds 10% context after last CPT
- Helps GAN understand boundary conditions

**Sorting Direction:**
- `"west", "east"` - Sorts by X coordinate ascending
- `"east", "west"` - Sorts by X coordinate descending
- `"south", "north"` - Sorts by Y coordinate ascending
- `"north", "south"` - Sorts by Y coordinate descending

---

## 🐛 Troubleshooting

### Problem: "No coordinates found" / Files moved to `no_coords/`

**Causes:**
- GEF file missing `#XYID` header
- Coordinates are 0.0 or NaN
- File corrupted or wrong format

**Solution:**
- Check GEF file structure
- Verify coordinates exist in source data
- Manually add coordinates if needed

---

### Problem: "Model file not found"

**Cause:** SchemaGAN `.h5` file path incorrect

**Solution:**
```python
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")  # Update this
```

---

### Problem: Pipeline stops at coordinate extraction

**Cause:** Logging was disabled (fixed in recent updates)

**Check:**
- `extract_coords.py` line 25: Should NOT have `logging.disable()`
- Should have: `geolib_logger = logging.getLogger('geolib_plus')`

---

### Problem: Sections look empty or incorrect

**Possible Causes:**
1. CPT names don't match between coordinates and data CSV
2. Depth range mismatch
3. Coordinate system issues (not RD)

**Debug Steps:**
1. Check `cpt_coordinates.csv` - names should match GEF filenames
2. Check `compressed_cpt_data_*.csv` - column names should match coordinate names
3. Review `manifest_sections.csv` - look for high `skipped_count`

---

### Problem: GAN generates poor quality schemas

**Possible Causes:**
1. Model not trained on similar soil types
2. Input sections have too many zeros (sparse data)
3. IC values out of expected range

**Solutions:**
- Ensure CPTs are closely spaced
- Check IC value distribution (should be 0-4.3)
- Verify model was trained on similar geological conditions

---

## 📊 Understanding the Outputs

### IC (Soil Behavior Index) Scale

**Robertson (1990) Classification:**
- **IC < 1.31** - Gravelly sand to dense sand
- **IC 1.31-2.05** - Sands: clean to silty
- **IC 2.05-2.60** - Sand mixtures: silty sand to sandy silt
- **IC 2.60-2.95** - Silt mixtures: clayey silt to silty clay
- **IC 2.95-3.60** - Clays: silty clay to clay
- **IC > 3.60** - Organic soils (peat, organic clay)

### Visualization Color Scheme (Viridis)
- **Purple/Blue** - Low IC (≈1-2) - Sands
- **Green/Teal** - Medium IC (≈2-3) - Silts
- **Yellow** - High IC (≈3-4) - Clays
- **White/Bright** - Very high IC (>4) - Organic soils

---

## 🔬 Individual Script Usage

All scripts can be run independently for testing or debugging:

### Extract Coordinates Only
```python
python src/extract_coords.py
# Configure GEF_FOLDER and OUT_CSV at bottom of file
```

### Process CPT Data Only
```python
python src/extract_data.py
# Configure CPT_FOLDER and OUTPUT_FOLDER in __main__ section
```

### Create Sections Only
```python
python src/create_schGAN_input_file.py
# Configure paths in CONFIG section
```

### Generate Schemas Only
```python
python src/create_schema.py
# Configure SECTIONS_DIR, PATH_TO_MODEL, OUT_DIR
```

### Create Mosaic Only
```python
python src/create_mosaic.py
# Configure MANIFEST_CSV, COORDS_WITH_DIST_CSV, GAN_DIR
```

---

## 📁 Repository Structure

```
vow_schGAN/
│
├── src/
│   ├── main_processing.py          # ⭐ Main pipeline orchestrator
│   ├── extract_coords.py            # Step 2: Coordinate extraction
│   ├── extract_data.py              # Step 3: CPT interpretation & compression
│   ├── create_schGAN_input_file.py  # Step 4: Section creation
│   ├── create_schema.py             # Step 5: GAN schema generation (standalone)
│   ├── create_mosaic.py             # Step 6: Mosaic assembly (standalone)
│   └── utils.py                     # Shared utilities
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .github/
    └── copilot-instructions.md      # AI assistant guidelines
```

---

## 🤝 Contributing

### Code Style
- Use descriptive variable names
- Add docstrings to functions
- Log important steps (use `logger.info()`)
- Handle errors gracefully with try/except

### Adding New Features
1. Test standalone first in individual script
2. Integrate into `main_processing.py` if part of main workflow
3. Update this README with new parameters/outputs
4. Update `.github/copilot-instructions.md` for AI context

---

## 📝 Version History

### Current Version (2025)
- ✅ Complete automated pipeline
- ✅ Comprehensive logging system
- ✅ Coordinate validation and auto-correction
- ✅ Flexible sectioning with overlap
- ✅ Mosaic generation with interpolation
- ✅ All scripts can run standalone or integrated

### Previous Versions
- Basic coordinate extraction
- Individual CPT processing
- Manual workflow execution

---

## 📧 Support

This readme was made wiy Copilot. For any questions read out to fabian.campos@deltares.nl

---

## ⚖️ License

Part of the VOW project for geotechnical subsurface modeling.

---

**Last Updated:** October 2025  
**Python Version:** 3.10+  
**Required:** GEOLib-Plus, TensorFlow 2.8+
