# Analysis of RF-Shimming Best Practices

## Executive Summary

This document compares the current `analysis/01` implementation against the best practices established in the `rf-shimming-7t` reference implementation (ds004906 dataset). The reference represents a published, peer-reviewed methodology for RF shimming in the cervical spinal cord at 7T.

## Key Findings

### ✅ **Strengths of Current Implementation**

1. **Modular code structure** - Well-organized functions for different shimming methods
2. **Dependency checking** - Includes helpful dependency validation
3. **Flexible ROI selection** - Multiple fallback strategies for ROI determination
4. **Documentation** - Good inline comments and docstrings
5. **Output visualization** - Creates before/after comparison plots

### ❌ **Critical Gaps vs. Best Practices**

#### 1. **Data Processing Pipeline**

**Reference Implementation:**
- Uses Spinal Cord Toolbox (SCT) for proper segmentation
- Registers B1 maps to anatomical space using `sct_register_multimodal`
- Processes full 3D volumes, not single slices
- Uses vertebral level labeling (C3-T2) for consistent analysis
- Properly converts flip angle maps to B1+ efficiency in nT/V units

**Current Implementation:**
- ❌ Works on single central slice only (line 221-233)
- ❌ No registration between B1 maps and anatomical images
- ❌ No vertebral level analysis
- ❌ No proper B1+ efficiency conversion (missing nT/V conversion)
- ❌ Synthetic channel generation when real channels unavailable (line 256-273)

**Impact:** Results are not anatomically meaningful and cannot be compared across subjects or vertebral levels.

#### 2. **Multi-Subject Analysis**

**Reference Implementation:**
- Processes all 5 subjects
- Performs statistical analysis across subjects (Repeated Measures ANOVA)
- Post-hoc paired t-tests with Benjamini-Hochberg FDR correction
- Reports group-level statistics

**Current Implementation:**
- ❌ Single subject only (auto-selects first subject)
- ❌ No statistical analysis
- ❌ No group-level reporting

**Impact:** Cannot assess reproducibility or generalizability of findings.

#### 3. **B1+ Efficiency Calculation**

**Reference Implementation (Cell 40):**
```python
GAMMA = 2.675e8  # [rad / (s T)]
requested_fa = 90  # saturation flip angle
ref_voltage = metadata.get("TxRefAmp", "N/A")
acquired_fa = acquired_fa / 10  # Siemens maps are in units of flip angle * 10
voltage_at_socket = ref_voltage * 10 ** -0.095  # Account for power loss
b1_map = (acquired_fa / requested_fa) * (np.pi / (GAMMA * 1e-3 * voltage_at_socket))
b1_map = b1_map * 1e9  # Convert to [nT/V]
```

**Current Implementation:**
- ❌ No B1+ efficiency conversion
- ❌ Works with raw magnitude/phase maps
- ❌ No reference voltage extraction from JSON sidecars

**Impact:** Results are not in standardized units and cannot be compared to literature values.

#### 4. **ROI Definition**

**Reference Implementation:**
- Uses spinal cord segmentation from SCT
- Restricts analysis to C3-T2 vertebral levels (where RF shimming was prescribed)
- Creates cylindrical mask around centerline (28mm diameter)
- Uses proper vertebral level labeling

**Current Implementation:**
- ✅ Multiple fallback strategies (good)
- ❌ No vertebral level restriction
- ❌ No anatomical validation of ROI
- ❌ May include regions outside the shimming prescription

**Impact:** ROI may not match the actual shimming prescription region.

#### 5. **Metrics Extraction**

**Reference Implementation:**
- Extracts metrics per slice along spinal cord (C3-T2)
- Uses `sct_extract_metric` with vertebral level file
- Computes mean, std, and CV along the cord
- Smooths data for visualization

**Current Implementation:**
- ✅ Computes mean, std, CV (good)
- ❌ Single slice only
- ❌ No along-cord analysis
- ❌ No vertebral level correspondence

**Impact:** Cannot assess spatial variation along the cord.

#### 6. **Visualization**

**Reference Implementation:**
- Multi-subject plots with vertebral level labels
- B1+ maps with statistical overlays
- GRE signal intensity comparisons
- Interactive Plotly figures
- Publication-quality figures

**Current Implementation:**
- ✅ Before/after comparison (good)
- ❌ Single subject only
- ❌ No vertebral level context
- ❌ No statistical overlays

#### 7. **Statistical Analysis**

**Reference Implementation:**
- Repeated Measures ANOVA across subjects
- Pairwise comparisons with FDR correction
- Group-level statistics (mean ± SD)
- Saves statistical results to CSV

**Current Implementation:**
- ❌ No statistical analysis
- ❌ No comparison across shimming methods
- ❌ No significance testing

## Recommendations

### High Priority (Critical for Valid Results)

1. **Implement proper B1+ efficiency conversion**
   - Extract `TxRefAmp` from JSON sidecars
   - Convert flip angle maps to nT/V units
   - Account for power loss (10^-0.095 factor)

2. **Add registration step**
   - Register B1 maps to anatomical space
   - Use SCT registration tools or equivalent
   - Ensure spatial correspondence

3. **Process full 3D volumes**
   - Remove single-slice limitation
   - Process entire spinal cord region
   - Extract metrics along the cord

4. **Implement vertebral level analysis**
   - Use manual disc labels from derivatives
   - Restrict analysis to C3-T2 levels
   - Extract metrics per vertebral level

5. **Multi-subject analysis**
   - Process all available subjects
   - Implement statistical analysis
   - Report group-level results

### Medium Priority (Improve Quality)

6. **Use proper segmentation**
   - Integrate SCT for spinal cord segmentation
   - Use CSF masks from derivatives
   - Validate ROI against anatomical structures

7. **Improve metrics extraction**
   - Extract along-cord profiles
   - Compute spatial statistics
   - Compare across vertebral levels

8. **Enhanced visualization**
   - Add vertebral level labels
   - Multi-subject comparisons
   - Statistical overlays

### Low Priority (Nice to Have)

9. **Interactive visualizations**
   - Plotly figures for exploration
   - Interactive B1+ maps

10. **Comprehensive documentation**
    - Method comparison table
    - Parameter sensitivity analysis

## Code Structure Comparison

### Reference Implementation Structure
```
1. Environment setup
2. Data download/loading
3. Process anat/T2starw (GRE)
   - Segmentation
   - Vertebral labeling
   - Registration
   - Metrics extraction
4. Process fmap/TFL (B1 maps)
   - Registration to GRE
   - B1+ efficiency conversion
   - Metrics extraction
5. Statistics
   - ANOVA
   - Post-hoc tests
6. Visualization
   - B1+ maps
   - GRE images
   - Along-cord profiles
```

### Current Implementation Structure
```
1. Configuration
2. Helper functions
3. Load maps (single slice)
4. ROI selection
5. Shimming methods
   - Phase-only
   - LS complex
6. Metrics computation
7. Visualization
```

## Conclusion

The current `analysis/01` implementation is a good **exploratory** script but does **not follow best practices** for RF shimming analysis. To align with the reference implementation:

1. **Critical:** Implement proper B1+ efficiency conversion
2. **Critical:** Add registration and 3D processing
3. **Critical:** Add vertebral level analysis
4. **Important:** Multi-subject statistical analysis
5. **Important:** Use proper segmentation tools (SCT)

The current script appears designed for quick exploration on a single slice, which is fine for initial development but should be expanded for publication-quality analysis.

## References

- Reference implementation: `rf-shimming-7t` (ds004906 dataset)
- Dataset: https://openneuro.org/datasets/ds004906
- Spinal Cord Toolbox: https://spinalcordtoolbox.com
- Shimming Toolbox: https://github.com/shimming-toolbox/shimming-toolbox

