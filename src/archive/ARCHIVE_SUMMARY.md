# Archive Summary

## What Was Moved

The following scripts have been moved to `src/archive/` as they are no longer part of the active pipeline:

### 📦 Archived Scripts (6 files)

1. **`main_processing.py`** (1693 lines)
   - Original monolithic script
   - Replaced by `main_processing_refactored.py`
   - Kept as backup

2. **`create_schema.py`**
   - Functionality moved to `modules/generation/schema_generation.py`

3. **`combination_calculation.py`**
   - Not used in main pipeline

4. **`explore_gan_arch.py`**
   - Research/exploration script

5. **`get_elevation_from_AHN.py`**
   - External utility, not part of main pipeline

6. **`create_mosaic_adv.py`**
   - Advanced/experimental version

## Active Pipeline Structure

### 🚀 Main Scripts (src/)
- `main_processing_refactored.py` - Main orchestration
- `modules/` - Organized functionality
  - `preprocessing/` - Steps 2-4
  - `generation/` - Step 5
  - `postprocessing/` - Steps 6-7
  - `visualization.py` - Utilities

### 🔧 Core Processing (src/)
- `extract_coords.py` - Coordinate extraction
- `extract_data.py` - CPT data processing
- `create_schGAN_input_file.py` - Section creation
- `boundary_enhancement.py` - Enhancement algorithms
- `create_mosaic.py` - Mosaic creation
- `uncertainty_quantification.py` - MC Dropout
- `utils.py` - Utilities
- `validation.py` - Validation

### ⚙️ Configuration (root)
- `config.py` - All settings

## Benefits

✅ **Cleaner structure**: Only active files in main directory
✅ **Preserved history**: All old scripts safely archived
✅ **Clear documentation**: README in archive explains what and why
✅ **Easy restoration**: Can restore any archived script if needed

## File Count

- **Before refactoring**: 17 Python files in src/
- **After archiving**: 11 Python files in src/ (6 moved to archive/)
- **Reduction**: 35% fewer files in active directory

The active directory now contains only the scripts that are actually used in the pipeline!
