# VOW SchemaGAN Pipeline

Automated workflow for generating geotechnical subsurface schematics from CPT data using SchemaGAN.

---

## What It Does

Transforms raw CPT (`.gef`) files into detailed subsurface cross-sections:

**Raw CPT Data** → **Interpreted Soil Profiles** → **GAN-Generated Schemas** → **Complete Mosaic**

### Pipeline Steps
1. **Setup** - Create folder structure
2. **Coordinates** - Extract & validate CPT locations
3. **Compression** - Process CPT data to 64-pixel depth
4. **Sections** - Create overlapping spatial sections
5. **GAN** - Generate detailed schemas
6. **Enhancement** - Sharpen layer boundaries
7. **Mosaic** - Combine into seamless visualization
8. **Uncertainty** - Quantify prediction variance
9. **Validation** - Cross-validation metrics (optional)

### Key Features
- **Control every step** - Enable/disable via config flags
- **One config file** - All settings in `config.py`
- **Interactive outputs** - Zoomable HTML visualizations
- **Statistical validation** - Leave-out cross-validation
- **Uncertainty maps** - Know where predictions are reliable

---

## Requirements

- **Python 3.10+**
- **GEOLib-Plus** - For CPT interpretation
- **SchemaGAN model** - Trained `.h5` file
- **CPT data** - `.gef` files with Netherlands RD coordinates

---

## Quick Start

### 1. Install

```bash
# Create virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure
Edit **`config.py`**:
```python
# Paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")
RES_DIR = Path(r"C:\VOW\res")

# Enable/disable steps
RUN_STEP_5_GAN = True
RUN_STEP_9_VALIDATION = False  # Optional, time-consuming
```

### 3. Run
```bash
python src/main_processing_refactored.py
```

Results saved to: `C:\VOW\res\<region>\<exp_name>\`
├── README.txt                            # Experiment metadata
├── 1_coords\
│   └── cpt_coordinates.csv              # Validated coordinates
├── 2_compressed_cpt\
│   └── compressed_cpt_data_mean_64px.csv  # 64-row IC profiles
├── 3_sections\
│   ├── section_01_z_00_cpts_XXX_to_YYY.csv  # Individual sections
│   ├── manifest_sections.csv               # Section metadata
│   └── cpt_coords_with_distances.csv       # Spatial distances
├── 4_gan_images\
│   ├── section_01_z_00_..._gan.csv         # Generated schemas (data)
│   ├── section_01_z_00_..._gan.png         # Generated schemas (images)
│   └── section_01_z_00_..._gan.html        # Interactive viewers
├── 5_enhance\
│   ├── section_01_z_00_..._enhanced.csv    # Enhanced schemas
│   └── section_01_z_00_..._enhanced.png    # Enhanced visualizations
├── 6_mosaic\
│   ├── schemaGAN_mosaic.csv                # Combined mosaic (data)
│   ├── schemaGAN_mosaic.png                # Mosaic visualization
│   ├── schemaGAN_mosaic.html               # Interactive mosaic
│   ├── enhanced_mosaic.csv                 # Enhanced mosaic (data)
│   ├── enhanced_mosaic.png                 # Enhanced mosaic visualization
│   └── enhanced_mosaic.html                # Interactive enhanced mosaic
├── 7_model_uncert\
│   ├── section_01_z_00_..._uncert.csv      # Uncertainty maps
│   ├── section_01_z_00_..._uncert.png      # Uncertainty visualizations
│   ├── uncertainty_mosaic.csv              # Uncertainty mosaic (data)
│   └── uncertainty_mosaic.png              # Uncertainty mosaic visualization
└── 8_validation\ (optional)
    ├── run_01\ ... run_10\                 # Individual validation runs
    │   ├── removed_cpts.txt                # List of removed CPTs
    │   ├── 4_gan_images\                   # Generated schemas without removed CPTs
    │   └── validation_mosaic.png           # Mosaic with dashed lines at removed CPTs
    └── validation_results.csv              # MAE and MSE metrics per run
```

---

### Output Structure
```
C:\VOW\res\<region>\<exp_name>\
├── 1_coords/                  # Validated CPT coordinates
├── 2_compressed_cpt/          # 64-pixel depth profiles
├── 3_sections/                # GAN input sections
├── 4_gan_images/              # Generated schemas (.csv, .png, .html)
├── 5_enhance/                 # Enhanced schemas
├── 6_mosaic/                  # Combined mosaics (GAN + enhanced)
├── 7_model_uncert/            # Uncertainty maps
└── 8_validation/              # Cross-validation results (optional)

### Design Principles

1. **Separation of Concerns**: Core logic separated from pipeline integration
2. **Configuration-Driven**: All parameters in `config.py`, no hardcoded values
3. **Modular Control**: Enable/disable any step via config flags
4. **Backwards Compatible**: Original scripts preserved in `archive/`
5. **Consistent Visualization**: Centralized plotting with unified styling

---

## 📚 Detailed Documentation

### Main Pipeline: `main_processing_refactored.py`

**Purpose:** Orchestrates the complete 8/9-step workflow with modular architecture  
**Usage:** Configure `config.py` and run directly  
**Logging:** Saves detailed logs to `<experiment_folder>/pipeline.log`

**Pipeline Orchestration:**
- **Step 1:** `setup_experiment()` - Creates folder structure
- **Step 2:** `coordinate_extraction.run_coordinate_extraction()` - Extracts CPT coords
- **Step 3:** `data_compression.run_data_compression()` - Processes CPT data
- **Step 4:** `section_creation.run_section_creation()` - Creates GAN input sections
- **Step 5:** `schema_generation.run_schema_generation()` - Generates schemas with GAN
- **Step 6:** `boundary_enhancement.run_boundary_enhancement()` - Enhances boundaries
- **Step 7:** `mosaic_creation.run_mosaic_creation()` - Builds mosaics
- **Step 8:** `mosaic_creation.run_mosaic_creation()` (uncertainty) - Uncertainty mosaic
- **Step 9:** `validation.run_validation_pipeline()` - Optional cross-validation

**Key Features:**
## Project Structure

```
vow_schGAN/
├── config.py                    # Edit this for all settings
├── src/
│   ├── main_processing_refactored.py  # Run this
│   ├── core/                    # Core implementations
│   ├── modules/                 # Pipeline wrappers
│   │   ├── preprocessing/
│   │   ├── generation/
│   │   ├── postprocessing/
│   │   ├── visualization/
│   │   └── validation/
│   └── archive/                 # Legacy scripts
└── requirements.txt
```

**Design:** Core logic in `core/`, config-driven wrappers in `modules/`, everything controlled via `config.py`
```python
if __name__ == "__main__":
    GEF_FOLDER = Path(r"C:\VOW\data\test_cpts")
    OUT_CSV = Path(r"C:\VOW\res\coordinates_cpts_test_result.csv")
    process_cpt_coords(GEF_FOLDER, OUT_CSV)
```

---

### Step 3: CPT Data Processing (`modules/preprocessing/data_compression.py`)

**Purpose:** Interpret CPT data and compress to configurable depth resolution (32 or 64 pixels)

**Wraps:** `core/extract_data.py`

**Main Function:**
```python
run_data_compression(
    cpt_folder: Path,
    coords_csv: Path,
    output_folder: Path,
    method: str = "mean",
    target_rows: int = 64
)
```

**Process:**
1. **Interpret CPTs** using GEOLib-Plus Robertson method
   - Calculates soil behavior index (IC)
   - Applies unit weight calculations
2. ## Configuration

All settings in **`config.py`**:

### Essential Settings
```python
# Paths
CPT_FOLDER = Path(r"C:\VOW\data\cpts")
SCHGAN_MODEL_PATH = Path(r"D:\schemaGAN\h5\schemaGAN.h5")
RES_DIR = Path(r"C:\VOW\res")

# Processing
COMPRESSION_METHOD = "mean"     # "mean" (smooth) or "max" (preserve peaks)
COMPRESSION_TARGET_ROWS = 64    # 32 or 64 pixels depth
CPTS_PER_SECTION = 6            # CPTs per section
OVERLAP_CPTS = 2                # Overlap between sections
```

### Step Controls
```python
RUN_STEP_5_GAN = True           # Generate schemas
RUN_STEP_7_MOSAIC = True        # Create mosaic
RUN_STEP_9_VALIDATION = False   # Optional validation (~10-15 min/run)
```

### Visualization
```python
PLOT_FONT_SIZE = 8                 # Font size for all plots
ASPECT_RATIO_WIDTH_HEIGHT = 4.17   # Plot dimensions
```

See `config.py` for all options.

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "No coordinates found" | Check GEF file structure
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

## Understanding Results

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

## Advanced Usage

### Running Specific Steps Only

Edit `config.py` to control which steps execute:

```python
# Example: Only run GAN generation and mosaic creation
RUN_STEP_1_FOLDERS = False    # Skip folder setup
RUN_STEP_2_COORDS = False     # Skip coordinate extraction
RUN_STEP_3_COMPRESS = False   # Skip data processing
RUN_STEP_4_SECTIONS = False   # Skip section creation
RUN_STEP_5_GAN = True         # Run GAN generation
RUN_STEP_6_ENHANCE = False    # Skip enhancement
RUN_STEP_7_MOSAIC = True      # Run mosaic creation
RUN_STEP_8_UNCERTAINTY = False
RUN_STEP_9_VALIDATION = False
```

Then run: `python src/main_processing_refactored.py`

### Running Legacy Standalone Scripts

Original scripts preserved in `src/archive/` can still run independently:

```bash
# Old pipeline (monolithic)
python src/archive/main_processing.py

# Individual legacy scripts
python src/archive/create_schema.py
python src/archive/create_mosaic.py
python src/archive/uncertainty_quantification.py
```

**Note:** Legacy scripts have hardcoded paths - edit CONFIG sections within each file.

### Core Scripts (Standalone Mode)

Core implementations can run standalone for testing:

```bash
# Extract coordinates
python src/core/extract_coords.py
# Edit GEF_FOLDER and OUT_CSV at bottom of file

# Process CPT data
python src/core/extract_data.py
# Edit CPT_FOLDER and OUTPUT_FOLDER in __main__ section

# Create sections
python src/core/create_schGAN_input_file.py
# Edit paths in CONFIG section (lines 20-40)

# Create mosaic
python src/core/create_mosaic.py
# Edit MANIFEST_CSV, COORDS_WITH_DIST_CSV, GAN_DIR (lines 13-26)
```

### Validation-Only Run

To run validation on existing results:

```python
# In config.py
RUN_STEP_1_FOLDERS = False
RUN_STEP_2_COORDS = False
RUN_STEP_3_COMPRESS = False
RUN_STEP_4_SECTIONS = False
RUN_STEP_5_GAN = False
RUN_STEP_6_ENHANCE = False
RUN_STEP_7_MOSAIC = False
RUN_STEP_8_UNCERTAINTY = False
RUN_STEP_9_VALIDATION = True   # Only validation

VALIDATION_N_RUNS = 10         # Number of iterations
VALIDATION_N_REMOVE = 12       # CPTs to remove per run
```

**Runtime:** ~10-15 minutes per validation run (depends on CPT count and model size)

---

## Repository Structure

```
vow_schGAN/
│
├── config.py                               # Central configuration file
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
│
├── src/
│   ├── main_processing_refactored.py       # Main pipeline orchestrator
│   │
│   ├── core/                               # Core implementations
│   │   ├── extract_coords.py               # Coordinate extraction logic
│   │   ├── extract_data.py                 # CPT processing & compression
│   │   ├── create_schGAN_input_file.py     # Section creation logic
│   │   ├── create_mosaic.py                # Mosaic building logic
│   │   └── utils.py                        # Shared utilities
│   │
│   ├── modules/                            # Pipeline integration modules
│   │   ├── preprocessing/
│   │   │   ├── coordinate_extraction.py    # Step 2 wrapper
│   │   │   └── data_compression.py         # Step 3 wrapper
│   │   ├── generation/
│   │   │   ├── section_creation.py         # Step 4 wrapper
│   │   │   └── schema_generation.py        # Step 5 implementation
│   │   ├── postprocessing/
│   │   │   ├── boundary_enhancement.py     # Step 6 implementation
│   │   │   └── mosaic_creation.py          # Step 7 wrapper
│   │   ├── visualization/
│   │   │   └── plotting.py                 # Unified plotting functions
│   │   └── validation/
│   │       └── validation.py               # Step 9 cross-validation
│   │
│   └── archive/                            # Legacy standalone scripts
│       ├── main_processing.py              # Original monolithic pipeline
│       ├── boundary_enhancement.py
│       ├── combination_calculation.py
│       ├── create_mosaic_adv.py
│       ├── create_mosaic.py
│       ├── create_schema.py
│       ├── explore_gan_arch.py
│       ├── get_elevation_from_AHN.py
│       ├── uncertainty_quantification.py
│       └── validation.py
│
└── .github/
    └── copilot-instructions.md             # AI assistant guidelines
```

### Key Differences: Core vs Modules

- **`core/`** - Pure implementations with hardcoded constants (can run standalone)
- **`modules/`** - Config-driven wrappers that integrate core logic into pipeline
- **`archive/`** - Original scripts preserved for reference/comparison

---

## Contributing

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

## Visualization Features

### Unified Styling
All plots follow consistent design:
- **Font Size:** 8pt (configurable via `config.PLOT_FONT_SIZE`)
- **Aspect Ratio:** 4.17:1 width/height (configurable)
- **Resolution:** 800 DPI for PNG outputs
- **Colormap:** Custom 5-class IC colormap with soil type boundaries
- **Dual Axes:** Pixel indices + real-world coordinates on all plots

### Interactive HTML Viewers
Every PNG visualization has an accompanying HTML file with:
- **Zoom & Pan:** Mouse wheel and drag
- **Pixel Inspector:** Hover to see exact coordinates
- **No Dependencies:** Pure HTML + base64-encoded images

### Custom IC Colormap
Five distinct colors for soil types:
1. **Sand** (IC 0.0-2.05): Yellow
2. **Sand Mixture** (IC 2.05-2.60): Orange  
3. **Silt Mixture** (IC 2.60-2.95): Light green
4. **Clay** (IC 2.95-3.60): Green
5. **Organic** (IC 3.60-4.5): Dark green

---

## Testing and Validation

### Built-in Validation
Step 9 provides comprehensive model validation:

**Method:** Leave-out cross-validation
- Randomly removes N CPTs (e.g., 12)
- Generates schema without those CPTs
## Understanding Results

### IC Scale (Robertson 1990)
- **IC < 2.05** - Sand (yellow/orange in plots)
- **IC 2.05-2.60** - Sand mixtures
- **IC 2.60-2.95** - Silt mixtures (green)
- **IC 2.95-3.60** - Clay
- **IC > 3.60** - Organic soil (dark green)

### Validation Metrics
```
MAE: 0.29 ± 0.02  # Mean prediction error in IC units
MSE: 0.22 ± 0.03  # Squared error
```
- **MAE < 0.3** = Excellent
- **MAE 0.3-0.5** = Good
- **MAE > 0.5** = Consider retraining

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "No coordinates found" | Check GEF files have `#XYID` header with valid RD coordinates |
| "Model file not found" | Update `SCHGAN_MODEL_PATH` in `config.py` |
| Empty sections | Verify CPT names match between coordinate and data files |
| Poor GAN quality | Ensure CPTs closely spaced, IC values in 0-4.3 range | Specific Steps
Set `RUN_STEP_X = False` in `config.py` to skip steps:
```python
RUN_STEP_5_GAN = True      # Only run GAN + mosaic
RUN_STEP_7_MOSAIC = True
# All others = False
```

### Validation Only
```python
RUN_STEP_9_VALIDATION = True  # Only this True
VALIDATION_N_RUNS = 10        # ~10-15 min per run
```

### Legacy Scripts
Original monolithic scripts available in `src/archive/` (require editing hardcoded paths)

---

## Contact

**Author:** Fabian Campos (fabian.campos@deltares.nl)  
**Project:** VOW - Geotechnical Subsurface Modeling  
**License:** Deltares © 2024-2025

**Version:** 2.0 (December 2025)  
**Python:** 3.10+ | **Dependencies:** TensorFlow 2.8+, GEOLib-Plus