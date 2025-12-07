# RF-Shimming Exploration Analysis

This directory contains exploratory RF-shimming analysis scripts for the ds004906 (rf-shimming-7t) dataset.

## Quick Start

1. **Edit the dataset path**: Open `01_rf_shimming_exploration.py` and edit the `DATASET_DIR` variable to point to your local ds004906 folder:
   ```python
   DATASET_DIR = "/path/to/your/dataset/ds004906"
   ```

2. **Run the analysis**:
   ```bash
   python analysis/01_rf_shimming_exploration.py
   ```

3. **View outputs**: Results will be saved in `analysis/analysis_outputs/`:
   - `rf_shim_metrics.csv` - Quantitative metrics for each shimming method
   - `rf_shim_before_after.png` - Visualization comparing baseline, phase-only, and LS complex shimming

## Subject Selection

By default, the script auto-selects the first subject found. To analyze a specific subject, edit the `SUBJECT` variable in the script:

```python
SUBJECT = "sub-01"  # or "sub-02", "sub-03", etc.
```

## Important Notes

- **This script uses processed dataset files** and is exploratory; it does not re-run the original heavy pipeline (SCT/Shimming Toolbox).
- The analysis works on a single central slice to keep runtime under 15 minutes on Mac M4.
- The script assumes B1 maps and masks are already available in the dataset's `fmap` directory.

## Requirements

Required Python packages:
- `nibabel` - NIfTI file I/O
- `pandas` - Data handling
- `scipy` - Optimization and image processing
- `scikit-image` - Image resizing
- `matplotlib` - Visualization
- `numpy` - Numerical operations

Install with:
```bash
pip install nibabel pandas scipy scikit-image matplotlib numpy
```

