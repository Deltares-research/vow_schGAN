# Archive Folder

This folder contains scripts that are no longer part of the active pipeline after the code refactoring (December 2025).

## Archived Scripts

### `main_processing.py` (Original - 1693 lines)
**Reason**: Replaced by `main_processing_refactored.py`

The original monolithic script that contained all functionality in one file. This has been refactored into a modular structure with:
- Configuration extracted to `config.py`
- Functions organized into `modules/` directory
- New orchestration script: `main_processing_refactored.py`

**Status**: Kept for backup and reference. Contains identical functionality to the refactored version.

---

### `create_schema.py`
**Reason**: Functionality moved to `modules/generation/schema_generation.py`

Schema generation logic has been refactored and is now part of the modular structure. The new module provides better separation of concerns and easier maintenance.

---

### `combination_calculation.py`
**Reason**: Not used in the main pipeline

This script appears to be for experimental calculations and is not imported or used by the main processing pipeline.

---

### `explore_gan_arch.py`
**Reason**: Exploration/research script, not part of production pipeline

This script was used for exploring GAN architectures during development. It's not part of the production pipeline workflow.

---

### `get_elevation_from_AHN.py`
**Reason**: External utility, not part of main pipeline

This appears to be a utility script for getting elevation data from AHN (Actueel Hoogtebestand Nederland). It's not integrated into the main SchemaGAN pipeline.

---

### `create_mosaic_adv.py`
**Reason**: Advanced/experimental mosaic creation

This appears to be an alternative or experimental version of mosaic creation. The main pipeline uses `create_mosaic.py` through the `modules/postprocessing/mosaic_creation.py` module.

---

## Active Scripts (Still in src/)

These scripts remain in the `src/` directory as they are actively used by the refactored pipeline:

- **`main_processing_refactored.py`** - Main orchestration script (new)
- **`extract_coords.py`** - CPT coordinate extraction (used by preprocessing module)
- **`extract_data.py`** - CPT data processing (used by preprocessing module)
- **`create_schGAN_input_file.py`** - Section creation (used by preprocessing module)
- **`boundary_enhancement.py`** - Enhancement algorithms (used by postprocessing module)
- **`create_mosaic.py`** - Mosaic creation (used by postprocessing module)
- **`uncertainty_quantification.py`** - MC Dropout uncertainty (used by generation module)
- **`utils.py`** - Utility functions (used throughout)
- **`validation.py`** - Validation functions (if used)

## Restoring Archived Scripts

If you need to restore any archived script:

```bash
# From the archive folder
Move-Item -Path "archive\<script_name>.py" -Destination ".\"
```

Or simply copy it back:

```bash
Copy-Item -Path "archive\<script_name>.py" -Destination ".\"
```

## Note

These scripts are preserved for:
1. **Backup**: In case you need to reference the original implementation
2. **Historical Reference**: Understanding how the code evolved
3. **Emergency Fallback**: If you need to revert to the old structure

The archived scripts are **not maintained** and may become outdated as the active codebase evolves.
