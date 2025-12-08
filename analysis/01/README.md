# RF-Shimming Analysis

This directory contains analysis scripts for comparing different RF shimming methods from the ds004906 (rf-shimming-7t) dataset.

## Overview

This script analyzes **already-shimmed B1 maps** from the dataset. It does NOT compute shimming weights (those were computed during acquisition using Shimming Toolbox). Instead, it:

1. Loads flip angle maps for different shimming methods (CP, CoV, patient, phase, volume, target, SAReff)
2. Converts them to B1+ efficiency in nT/V units
3. Extracts metrics (mean, std, CV) within the spinal cord ROI
4. Compares the different shimming methods

## Quick Start

1. **Edit the dataset path**: Open `01_rf_shimming_exploration.py` and edit the `DATASET_DIR` variable:
   ```python
   DATASET_DIR = "/path/to/your/dataset"
   ```

2. **Run the analysis**:
   ```bash
   cd analysis/01
   python 01_rf_shimming_exploration.py
   ```

3. **View outputs**: Results will be saved in `analysis_outputs/`:
   - `rf_shim_metrics.csv` - Quantitative metrics (mean, std, CV) for each shimming method
   - `rf_shim_comparison.png` - Visualization of B1+ efficiency maps for all methods
   - `rf_shim_cv_comparison.png` - Bar chart comparing coefficient of variation across methods

## Subject Selection

By default, the script auto-selects the first subject found. To analyze a specific subject, edit the `SUBJECT` variable in the script:

```python
SUBJECT = "sub-01"  # or "sub-02", "sub-03", etc.
```

## What This Script Does

1. **Loads flip angle maps** (`famp*_TB1TFL.nii.gz` files) for each shimming method
2. **Converts to B1+ efficiency** in nT/V units using:
   - Reference voltage from JSON sidecars (`TxRefAmp`)
   - Physical constants (gyromagnetic ratio, pulse duration)
   - Power loss correction factor
3. **Determines ROI** using:
   - Spinal cord segmentation masks (if available in derivatives)
   - Auto-thresholding (fallback)
4. **Computes metrics**:
   - Mean B1+ efficiency (nT/V)
   - Standard deviation (nT/V)
   - Coefficient of variation (%)
5. **Compares methods** and identifies the best shimming method (lowest CV)

## Important Notes

- **This script analyzes already-shimmed results**, not raw channel data
- The shimming weights were computed during acquisition using [Shimming Toolbox](https://github.com/shimming-toolbox/shimming-toolbox)
- The analysis works on a single central slice for speed
- This is a simplified version compared to the full reference implementation
- For full 3D analysis with vertebral level labeling, see the [reference notebook](https://github.com/shimming-toolbox/rf-shimming-7t)

## Requirements

Required Python packages:
- `nibabel` - NIfTI file I/O
- `pandas` - Data handling
- `scipy` - Image processing
- `scikit-image` - Image resizing
- `matplotlib` - Visualization
- `numpy` - Numerical operations

Install with:
```bash
pip install nibabel pandas scipy scikit-image matplotlib numpy
```

## Output Files

- **rf_shim_metrics.csv**: Table with metrics for each shimming method
  - Columns: Method, Mean (nT/V), Std (nT/V), CV (%)
  - Sorted by CV (lower is better for homogeneity)

- **rf_shim_comparison.png**: Grid showing B1+ efficiency maps for all methods
  - Each panel shows one shimming method
  - Overlay shows ROI contour
  - Title shows mean B1+ and CV

- **rf_shim_cv_comparison.png**: Bar chart comparing CV across methods
  - Lower bars indicate better homogeneity
  - Labels show CV% and mean B1+ value

## Reference

- Dataset: [ds004906 on OpenNeuro](https://openneuro.org/datasets/ds004906)
- Reference implementation: [rf-shimming-7t repository](https://github.com/shimming-toolbox/rf-shimming-7t)
- Shimming Toolbox: [shimming-toolbox](https://github.com/shimming-toolbox/shimming-toolbox)
