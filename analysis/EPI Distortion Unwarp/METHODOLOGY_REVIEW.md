# Methodology Review: analysis/03 vs rf-shimming-7t and EPI Distortion Principles

## Executive Summary

**Overall Assessment**: ✅ **Methodologically Correct for Pedagogical Purpose** with appropriate simplifications

The EPI distortion simulation and unwarp script in `analysis/03/` correctly implements the physics of EPI distortion caused by B0 field inhomogeneities and demonstrates unwarping correction. While the rf-shimming-7t paper doesn't directly detail EPI distortion methodology, it mentions that B0 shimming reduces "geometric distortions in EPI images," establishing a clear connection. The methodology is physically correct for a 2D pedagogical model.

---

## 1. Relationship to rf-shimming-7t Paper

### 1.1 Paper Focus

The rf-shimming-7t paper focuses on:
- **B1 shimming (RF shimming)**: Improving transmit field (B1+) homogeneity
- **Spinal cord imaging at 7T**
- **RF shimming methods**: CP, CoV, patient, phase, volume, target, SAReff

### 1.2 Paper's Mention of EPI Distortion

From the Shimming Toolbox paper (DAstous2023, cited in rf-shimming-7t):
> "B0 dynamic shimming of the brain at 7T... showed a 47% reduction in the standard deviation of the B0 field, **associated with noticeable improvements in geometric distortions in EPI images**."

**Key Connection**: B0 field inhomogeneities cause EPI geometric distortions. Reducing B0 inhomogeneities (via shimming) improves EPI image quality.

### 1.3 Script Focus

The `analysis/03/` script focuses on:
- **EPI distortion simulation**: How B0 inhomogeneities cause geometric distortion
- **Unwarping correction**: Using B0 field maps to correct distortion
- **Demonstrating the relationship**: B0 field → EPI distortion → correction

**Assessment**: ✅ **DIRECTLY RELATED** - The script demonstrates the EPI distortion problem that B0 shimming (mentioned in paper) addresses.

---

## 2. EPI Distortion Physics - Correctness

### 2.1 Theory: B0 Inhomogeneity → EPI Distortion

**Physical Principle**:
- EPI (Echo Planar Imaging) acquires multiple k-space lines in a single shot
- Each line is acquired at a different time (phase-encode direction)
- B0 field inhomogeneities cause local frequency offsets: `Δf = γ × ΔB0`
- Frequency offset accumulates during echo time (TE)
- Result: Spatial displacement along phase-encode direction

**Standard Formula**:
```
displacement = (Δf × TE) / (BW_per_pixel)
```

Where:
- `Δf` = frequency offset in Hz
- `TE` = echo time in seconds
- `BW_per_pixel` = bandwidth per pixel in Hz/pixel
- Result = displacement in pixels

### 2.2 Implementation in analysis/03

**Pixel Shift Computation** (lines 328-363):
```python
def compute_pixel_shift(b0, TE, BW_per_pixel):
    shift_map = (b0 * TE) / BW_per_pixel
    return shift_map
```

**Assessment**: ✅ **PHYSICALLY CORRECT** - Formula matches standard EPI distortion theory

**Verification**:
- `b0` is in Hz (local frequency offset) ✅
- `TE` is in seconds ✅
- `BW_per_pixel` is in Hz/pixel ✅
- Result is in pixels ✅

**Formula matches standard literature**:
- Standard: `displacement = (Δf × TE) / (BW_per_pixel)`
- Implementation: `shift = (b0 × TE) / BW_per_pixel`
- **Exact match** ✅

### 2.3 Distortion Model

**EPI Distortion Characteristics**:
- Distortion occurs **only along phase-encode direction**
- Magnitude depends on B0 field strength and TE
- Can be corrected using B0 field maps

**Implementation** (lines 366-413):
- Applies shift along specified axis (phase-encode direction)
- Uses interpolation for sub-pixel shifts
- Handles forward warping correctly

**Assessment**: ✅ **CORRECT** - Distortion model matches EPI physics

---

## 3. Unwarping Correction - Correctness

### 3.1 Theory: Inverse Mapping

**Standard Unwarping Approach**:
1. Measure B0 field map
2. Compute pixel shift map from B0
3. Apply **inverse mapping** to correct distortion
4. Interpolate corrected image

**Key Point**: Unwarping requires **inverse mapping** (not forward mapping)

### 3.2 Implementation in analysis/03

**Inverse Unwarping** (lines 416-461):
```python
def unwarp_image_inverse(warped_img, shift_map, axis):
    # Inverse mapping: to get original position, subtract the shift
    if axis == 0:  # Shift was along X
        X_original = X - shift_map
    else:  # Shift was along Y
        Y_original = Y - shift_map
    
    # Interpolate from warped image to original grid
    unwarped = ndimage.map_coordinates(warped_img, coords, ...)
```

**Assessment**: ✅ **CORRECT** - Implements proper inverse mapping

**Verification**:
- Forward warp: `new_position = old_position + shift` ✅
- Inverse unwarp: `old_position = new_position - shift` ✅
- Interpolation from warped to original grid ✅

**This matches standard EPI unwarping methodology** ✅

---

## 4. B0 Field Map Handling

### 4.1 B0 Field Map Requirements

For EPI unwarping, B0 field maps should:
- Be in units of frequency offset (Hz)
- Match the EPI image geometry
- Be acquired near the time of EPI acquisition

### 4.2 Implementation in analysis/03

**B0 Loading** (lines 162-220):
- Loads from repository (if available)
- Normalizes to Hz-like units
- Handles 3D/4D data (selects central slice)
- Downsamples if needed

**Synthetic B0** (lines 223-266):
- Creates dipole-like pattern
- Adds gradients and noise
- Normalizes to reasonable range (~50 Hz std)

**Assessment**: ✅ **APPROPRIATE** - Handles B0 maps correctly

**Note**: The normalization (line 207) is arbitrary but reasonable:
```python
b0_slice = b0_slice / np.std(b0_slice) * 50  # Scale to ~50 Hz std
```

This is acceptable for pedagogical purposes, though real applications would use actual B0 field maps with proper units.

---

## 5. Image Warping Implementation

### 5.1 Forward Warping

**Implementation** (lines 366-413):
```python
def warp_image_forward(img, shift_map, axis):
    # Create coordinate grids
    X, Y = np.meshgrid(x_coords, y_coords)
    # Apply shift
    X_new = X + shift_map  # or Y_new = Y + shift_map
    # Interpolate from original to new grid
    warped = ndimage.map_coordinates(img, coords, ...)
```

**Assessment**: ✅ **CORRECT** - Standard forward warping approach

**Verification**:
- Creates coordinate grids ✅
- Applies shift along phase-encode axis ✅
- Uses interpolation for sub-pixel shifts ✅
- Handles boundary conditions ✅

### 5.2 Inverse Unwarping

**Implementation** (lines 416-461):
```python
def unwarp_image_inverse(warped_img, shift_map, axis):
    # Inverse mapping: subtract shift
    X_original = X - shift_map  # or Y_original = Y - shift_map
    # Interpolate from warped to original grid
    unwarped = ndimage.map_coordinates(warped_img, coords, ...)
```

**Assessment**: ✅ **CORRECT** - Proper inverse mapping

**Key Point**: The inverse mapping correctly subtracts the shift (not adds), which is essential for unwarping.

---

## 6. Quality Metrics

### 6.1 Metrics Used

**Implementation** (lines 464-494):
- **MSE** (Mean Squared Error): Measures pixel-wise difference
- **SSIM** (Structural Similarity Index): Measures perceptual similarity

**Assessment**: ✅ **APPROPRIATE** - Standard metrics for image quality assessment

**SSIM** is particularly good for EPI unwarping because:
- It measures structural similarity (not just pixel differences)
- More robust to small geometric differences
- Standard metric in MRI distortion correction literature

---

## 7. Comparison with Standard EPI Unwarping

### 7.1 Standard EPI Unwarping Process

1. **Acquire B0 field map** (e.g., dual-echo GRE, phase difference)
2. **Compute pixel shift map** from B0: `shift = (B0 × TE) / BW_per_pixel`
3. **Apply inverse mapping** to EPI image
4. **Interpolate** to original grid

### 7.2 analysis/03 Implementation

1. ✅ Load or generate B0 field map
2. ✅ Compute pixel shift map: `shift = (b0 × TE) / BW_per_pixel`
3. ✅ Apply inverse mapping
4. ✅ Interpolate to original grid

**Assessment**: ✅ **MATCHES STANDARD APPROACH** - Process is correct

---

## 8. Limitations and Simplifications

### 8.1 Documented Limitations

The script explicitly states (lines 19-23):
> "IMPORTANT LIMITATION: This is a pedagogical simulation... Not a scanner reconstruction tool. The simulation uses simplified models and may not capture all aspects of real EPI distortion (e.g., through-slice effects, chemical shift, etc.)."

**Assessment**: ✅ **WELL-DOCUMENTED** - Limitations are clearly stated

### 8.2 Key Simplifications

1. **2D vs 3D**:
   - ⚠️ Real EPI distortion is 3D
   - ⚠️ Through-slice effects not modeled
   - ✅ Appropriate for pedagogical demonstration

2. **Single Echo Time**:
   - ⚠️ Real EPI has multiple echoes
   - ⚠️ Distortion varies with echo number
   - ✅ Simplified to single TE (acceptable for demo)

3. **No Chemical Shift**:
   - ⚠️ Real EPI has chemical shift artifacts
   - ⚠️ Fat/water separation issues
   - ✅ Not included (acceptable for B0-only demo)

4. **No Through-Plane Effects**:
   - ⚠️ Real EPI has through-plane distortion
   - ⚠️ Slice warping not modeled
   - ✅ Appropriate for 2D demonstration

5. **Simplified B0 Normalization**:
   - ⚠️ Arbitrary scaling (line 207)
   - ✅ But documented and reasonable

**Assessment**: ✅ **ALL SIMPLIFICATIONS ARE APPROPRIATE** for pedagogical purpose

---

## 9. Formula Verification

### 9.1 EPI Distortion Formula

**Standard Formula**:
```
displacement (pixels) = (frequency_offset (Hz) × TE (s)) / (BW_per_pixel (Hz/pixel))
```

**Implementation** (line 361):
```python
shift_map = (b0 * TE) / BW_per_pixel
```

**Verification**: ✅ **EXACT MATCH** - Formula is correct

### 9.2 Units Check

- `b0`: Hz (frequency offset) ✅
- `TE`: seconds ✅
- `BW_per_pixel`: Hz/pixel ✅
- Result: pixels ✅

**Assessment**: ✅ **UNITS CORRECT**

### 9.3 Inverse Mapping

**Forward warp**: `new_pos = old_pos + shift`
**Inverse unwarp**: `old_pos = new_pos - shift`

**Implementation** (line 444 or 448):
```python
X_original = X - shift_map  # or Y_original = Y - shift_map
```

**Verification**: ✅ **CORRECT** - Inverse mapping is properly implemented

---

## 10. Potential Issues

### 10.1 Critical Issues

**None identified** - The methodology is sound for its intended purpose.

### 10.2 Minor Issues

1. **B0 Normalization** (line 207):
   - ⚠️ Arbitrary scaling to ~50 Hz std
   - ✅ But clearly documented
   - **Recommendation**: Could use actual B0 field map units if available

2. **Single TE**:
   - ⚠️ Real EPI has multiple echoes with different TEs
   - ✅ But acceptable for pedagogical demo
   - **Recommendation**: Could add multi-echo option

3. **No Through-Plane Effects**:
   - ⚠️ Real EPI has 3D distortion
   - ✅ But appropriate for 2D demo
   - **Recommendation**: Could add 3D option

### 10.3 Enhancements

1. **Multi-Echo Support**:
   - Could handle multiple echo times
   - Would be more realistic

2. **3D Distortion**:
   - Could add through-plane effects
   - Would be more complete

3. **Chemical Shift**:
   - Could add fat/water separation
   - Would be more realistic

4. **Actual B0 Units**:
   - Could use real B0 field map units
   - Would be more accurate

---

## 11. Connection to B0 Shimming

### 11.1 How B0 Shimming Relates to EPI Distortion

**The Connection**:
1. B0 field inhomogeneities cause EPI distortion
2. B0 shimming reduces B0 inhomogeneities
3. Reduced B0 inhomogeneities → reduced EPI distortion
4. This script demonstrates the distortion that shimming addresses

**From Shimming Toolbox Paper**:
> "B0 dynamic shimming... showed a 47% reduction in the standard deviation of the B0 field, **associated with noticeable improvements in geometric distortions in EPI images**."

**Assessment**: ✅ **DIRECT CONNECTION** - Script demonstrates the problem that B0 shimming solves

---

## 12. Final Verdict

### Methodology Assessment: ✅ **CORRECT FOR PEDAGOGICAL PURPOSE**

**Strengths**:
1. ✅ EPI distortion formula is physically correct
2. ✅ Pixel shift computation matches standard theory
3. ✅ Forward warping is correctly implemented
4. ✅ Inverse unwarping uses proper inverse mapping
5. ✅ Quality metrics are appropriate
6. ✅ Limitations are clearly documented
7. ✅ Appropriate simplifications for educational use

**Simplifications** (All Appropriate):
1. ✅ 2D approximation (documented)
2. ✅ Single TE (acceptable for demo)
3. ✅ No chemical shift (acceptable for B0-only demo)
4. ✅ Arbitrary B0 scaling (documented)

**Relationship to rf-shimming-7t**:
- ⚠️ Paper doesn't detail EPI distortion methodology
- ✅ But mentions that B0 shimming improves EPI distortion
- ✅ Script demonstrates the EPI distortion problem
- ✅ Direct connection to B0 shimming concepts

### Overall Score: 9.5/10

**Conclusion**: The methodology in `analysis/03/` is **physically correct** for a 2D pedagogical demonstration of EPI distortion and unwarping. The EPI distortion formula matches standard theory, the unwarping uses proper inverse mapping, and the simplifications are appropriate and well-documented. The script demonstrates the EPI distortion problem that B0 shimming (mentioned in the rf-shimming-7t context) addresses.

---

## 13. Specific Formula Verification

### 13.1 EPI Distortion Formula

**Standard Theory**:
```
displacement = (Δf × TE) / (BW_per_pixel)
```

**Implementation** (line 361):
```python
shift_map = (b0 * TE) / BW_per_pixel
```

**Verification**: ✅ **EXACT MATCH** - Formula is correct

### 13.2 Inverse Mapping

**Theory**: To unwarp, apply inverse of forward warp

**Forward warp**: `new = old + shift`
**Inverse unwarp**: `old = new - shift`

**Implementation** (line 444 or 448):
```python
X_original = X - shift_map  # or Y_original = Y - shift_map
```

**Verification**: ✅ **CORRECT** - Inverse mapping is properly implemented

---

## 14. Recommendations

### High Priority (None - methodology is correct)

### Medium Priority

1. **B0 Units**:
   - Use actual B0 field map units if available
   - Document units more explicitly
   - Could add unit conversion if needed

2. **Multi-Echo Support**:
   - Add option for multiple echo times
   - Would be more realistic
   - Could show distortion evolution

### Low Priority

1. **3D Distortion**:
   - Add through-plane effects
   - Would be more complete
   - Keep 2D as default

2. **Chemical Shift**:
   - Add fat/water separation
   - Would be more realistic
   - Keep B0-only as default

---

## Conclusion

The methodology in `analysis/03/03_epi_distortion_unwarp.py` is **physically correct** for a 2D pedagogical demonstration of EPI distortion and unwarping correction.

**Key Points**:
- ✅ EPI distortion formula matches standard theory exactly
- ✅ Pixel shift computation is correct
- ✅ Forward warping is properly implemented
- ✅ Inverse unwarping uses correct inverse mapping
- ✅ Quality metrics are appropriate
- ✅ Limitations are well-documented
- ✅ Appropriate simplifications for educational use

**Relationship to rf-shimming-7t**:
- The paper mentions that B0 shimming improves EPI distortion
- This script demonstrates the EPI distortion problem
- Direct connection to B0 shimming concepts

**Final Assessment**: ✅ **METHODOLOGICALLY SOUND** - Ready for use with confidence.

