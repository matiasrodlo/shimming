# Fixes Applied to analysis/01

## Summary

The script has been completely rewritten to properly analyze already-shimmed B1 maps instead of attempting to compute shimming weights from non-existent individual channel data.

## What Was Wrong

1. **Fundamental Mismatch**: Script tried to compute RF shimming weights from individual channel B1 maps that don't exist in the dataset
2. **Synthetic Channel Generation**: Created fake channels when real ones weren't available (invalid approach)
3. **Wrong Input Data**: Attempted to shim already-shimmed results (circular logic)
4. **Missing B1+ Conversion**: Didn't convert flip angle maps to B1+ efficiency in nT/V units

## What Was Fixed

### 1. Correct Data Loading
- ✅ Now loads already-shimmed flip angle maps (`famp*_TB1TFL.nii.gz` files)
- ✅ Handles all available shimming methods (CP, CoV, patient, phase, volume, target, SAReff)
- ✅ No more synthetic channel generation

### 2. Proper B1+ Efficiency Conversion
- ✅ Implements conversion from flip angle maps to B1+ efficiency in nT/V units
- ✅ Extracts reference voltage from JSON sidecars (`TxRefAmp`)
- ✅ Applies correct physics formula with power loss correction
- ✅ Follows reference implementation methodology

### 3. Correct Analysis Approach
- ✅ Compares different shimming methods (not computing shimming)
- ✅ Extracts metrics (mean, std, CV) for each method
- ✅ Identifies best shimming method based on CV

### 4. Improved Visualizations
- ✅ Grid view comparing all shimming methods
- ✅ Bar chart comparing coefficient of variation
- ✅ Proper labels and units (nT/V)

### 5. Better Documentation
- ✅ Updated README with correct usage instructions
- ✅ Clear explanation of what the script does
- ✅ References to source dataset and methodology

## Key Changes

| Before | After |
|--------|-------|
| Computed shimming weights | Analyzes already-shimmed results |
| Synthetic channel generation | Uses real shimming method data |
| No B1+ conversion | Proper nT/V conversion |
| Phase-only & LS shimming | Comparison of 7 shimming methods |
| Wrong input data | Correct data structure |

## Script Structure

1. **Configuration**: Dataset path, subject selection
2. **Data Loading**: Find and load flip angle maps for each shimming method
3. **B1+ Conversion**: Convert flip angle maps to B1+ efficiency (nT/V)
4. **ROI Selection**: Determine spinal cord ROI (mask or auto-threshold)
5. **Metrics Extraction**: Compute mean, std, CV for each method
6. **Comparison**: Sort and compare methods
7. **Visualization**: Create comparison plots
8. **Output**: Save CSV and PNG files

## Usage

```bash
cd analysis/01
python 01_rf_shimming_exploration.py
```

## Output Files

- `rf_shim_metrics.csv`: Metrics table for each method
- `rf_shim_comparison.png`: Grid of B1+ maps
- `rf_shim_cv_comparison.png`: CV comparison bar chart

## Validation

The script now:
- ✅ Uses correct input data (already-shimmed B1 maps)
- ✅ Implements proper B1+ efficiency conversion
- ✅ Follows reference implementation methodology
- ✅ Produces valid, meaningful results
- ✅ Can be compared to reference implementation

## Next Steps (Optional Enhancements)

For a more complete analysis matching the reference:
- Add 3D processing (currently single slice)
- Add vertebral level labeling (C3-T2)
- Add multi-subject analysis
- Add statistical comparisons
- Add along-cord profile extraction

But the current version is now **correct** and produces **valid results** for single-subject, single-slice analysis.

