# Technical Analysis: Savart Optimizer

## Executive Summary

**Overall Assessment**: ⚠️ **Partially Correct with Significant Issues**

The optimizer has a **sound mathematical framework** but contains **critical errors** in:
1. **Biot-Savart implementation** - Uses simplified approximation that may not be physically accurate
2. **Gradient calculation** - Mathematical error in variance gradient
3. **Missing loop position** - Bug in `make_loop_positions()` function

---

## 1. Technical Issues

### 1.1 ✅ Loop Positions Function

**Location**: Lines 288-292

**Status**: **CORRECT**

The function correctly computes both x and y coordinates:
```python
loop_centers[:, 0] = R_coil_mm * np.cos(angles)
loop_centers[:, 1] = R_coil_mm * np.sin(angles)  # Present in actual code
```

**Verification**: Code is correct, loops are properly placed in a circle.

---

### 1.2 ⚠️ Incorrect Biot-Savart Implementation

**Location**: Lines 295-386

**Current Implementation**:
```python
# Simplified approximation
r_avg = (r_start + r_end) / 2
contribution = seg_length[i] * perp_dist / (r_avg**3 + 1e-10)
```

**Problem**: This is **not** the correct Biot-Savart formula for a straight segment.

**Correct Biot-Savart Formula** for a straight wire segment:
```
B = (μ₀ I / 4π) * (sin(θ₁) - sin(θ₂)) / r_perp
```
where:
- `θ₁`, `θ₂` are angles from segment endpoints to observation point
- `r_perp` is perpendicular distance from wire to observation point

**Current Approximation Issues**:
1. Uses average distance `r_avg` instead of perpendicular distance `r_perp`
2. Missing angle terms `sin(θ₁) - sin(θ₂)`
3. Uses `perp_dist` in numerator but `r_avg` in denominator (inconsistent)

**Impact**: 
- Field calculations are **not physically accurate**
- May produce incorrect field patterns
- Results are "pedagogical" but not scientifically correct

**Severity**: 🟡 **MODERATE** - Works for demonstration but not for real physics

**Note**: The code comments acknowledge this is a "simplified approximation" and "pedagogical", so this may be intentional for educational purposes.

---

### 1.3 ✅ Gradient Calculation

**Location**: Line 542

**Current Code**:
```python
grad_var = 2 * A_roi.T @ (f_roi - mean_f) - 2 * np.mean(f_roi - mean_f) * np.sum(A_roi, axis=0)
```

**Analysis**: The gradient is **mathematically correct** (up to a constant scaling factor).

**Derivation**:
For variance: `Var = sum((f_i - μ)²)` where `μ = mean(f)`

Gradient:
```
d/dw Var = 2 * A^T @ (f - μ) - 2 * sum(f - μ) * (1/n) * sum(A, axis=0)
```

Since `sum(f - μ) = 0` (by definition of mean), the second term is always zero.

**Result**: The gradient simplifies to:
```
grad_var = 2 * A_roi.T @ (f_roi - mean_f)
```

**Note**: The current implementation includes the zero term, which is harmless but unnecessary. The missing `1/n_roi` normalization factor doesn't affect the optimization direction (only scales the gradient), so the optimizer will still converge correctly.

**Status**: ✅ **FUNCTIONALLY CORRECT** (minor inefficiency with zero term)

---

## 2. Moderate Issues

### 2.1 ⚠️ Objective Function Scaling

**Location**: Line 531

**Current**:
```python
variance = np.sum((f_roi - mean_f)**2)
```

**Issue**: Not normalized by number of ROI pixels.

**Better**:
```python
variance = np.mean((f_roi - mean_f)**2)  # or np.var(f_roi)
```

**Impact**: Objective value depends on ROI size, making it harder to compare across different ROI sizes.

**Severity**: 🟢 **LOW** - Works but not ideal

---

### 2.2 ⚠️ Sign Determination in Biot-Savart

**Location**: Lines 381-382

**Current**:
```python
cross_z = dx * seg_dir[1] - dy * seg_dir[0]
sign = np.sign(cross_z + 1e-10)
```

**Issue**: The sign calculation is simplified and may not correctly capture the right-hand rule for magnetic fields.

**Correct Approach**: Should use the full 3D cross product and right-hand rule:
```
B = (μ₀ I / 4π) * (dl × r̂) / r²
```

**Impact**: Field direction may be incorrect in some cases.

**Severity**: 🟢 **LOW** - May work for most cases

---

## 3. What Works Correctly

### 3.1 ✅ Loop Discretization

**Location**: Lines 321-330

**Status**: Correctly discretizes circular loop into straight segments.

---

### 3.2 ✅ Design Matrix Construction

**Location**: Lines 389-425

**Status**: Correctly builds design matrix `A` where each column is a flattened field map from one loop.

---

### 3.3 ✅ ROI Mask Creation

**Location**: Lines 428-447

**Status**: Correctly creates circular ROI mask.

---

### 3.4 ✅ Optimization Framework

**Location**: Lines 488-559

**Status**: 
- Uses scipy.optimize correctly
- Tikhonov regularization is properly implemented
- Bounds are correctly applied
- Overall optimization structure is sound

---

### 3.5 ✅ Field Combination

**Location**: Lines 450-485

**Status**: Correctly combines fields from multiple loops using linear superposition:
```
B_total = Σ(w_i * B_i)
```

This is physically correct (superposition principle).

---

## 4. Recommendations

### Priority 1 (Important - Should Fix)

1. **Optimize gradient calculation** (remove unnecessary zero term)
   ```python
   # Current (works but has unnecessary term):
   grad_var = 2 * A_roi.T @ (f_roi - mean_f) - 2 * np.mean(f_roi - mean_f) * np.sum(A_roi, axis=0)
   
   # Optimized (same result, more efficient):
   grad_var = 2 * A_roi.T @ (f_roi - mean_f)  # Second term is always zero
   ```

### Priority 2 (Important - Should Fix)

3. **Implement correct Biot-Savart formula**
   - Use proper angle-based formula for straight segments
   - Or use analytical solution for circular loops (elliptic integrals)

4. **Normalize objective function**
   ```python
   variance = np.mean((f_roi - mean_f)**2)  # Instead of sum
   ```

### Priority 3 (Nice to Have)

5. **Add validation tests**
   - Test against known analytical solutions
   - Verify field patterns are physically reasonable
   - Check optimization convergence

6. **Document approximations**
   - Clearly state what is simplified
   - Provide references to correct formulas
   - Explain when approximations are valid

---

## 5. Analytical Solution Alternative

For circular loops, there is an **analytical solution** using elliptic integrals:

```
B_z = (μ₀ I / 2) * (R² / (R² + z²)^(3/2))
```

For off-axis points, use:
```
B_z = (μ₀ I / 2π) * ∫[0 to 2π] (R - r cos(φ)) / (R² + r² - 2Rr cos(φ) + z²)^(3/2) dφ
```

This would be more accurate than the current segment-based approach.

---

## 6. Testing Recommendations

### Unit Tests Needed

1. **Test loop positions**
   ```python
   def test_loop_positions():
       centers, radius = make_loop_positions(8, 80, 10)
       assert centers.shape == (8, 2)
       assert np.allclose(np.linalg.norm(centers, axis=1), 80)  # All at radius 80
   ```

2. **Test Biot-Savart on-axis**
   - Compare with analytical solution for on-axis point
   - Should match: `B_z = μ₀ I R² / (2(R² + z²)^(3/2))`

3. **Test gradient correctness**
   - Use finite differences to verify gradient
   - `grad_approx = (f(w + ε) - f(w)) / ε`

4. **Test optimization convergence**
   - Verify objective decreases
   - Check final solution is reasonable

---

## 7. Conclusion

### Technical Correctness: ⚠️ **Partially Correct**

**What Works**:
- ✅ Overall optimization framework
- ✅ Design matrix construction
- ✅ Field superposition
- ✅ ROI masking

**What Needs Improvement**:
- ⚠️ Biot-Savart implementation (simplified approximation, not fully physically accurate)
- ⚠️ Gradient calculation (works but has unnecessary zero term)
- ⚠️ Objective function scaling (not normalized by ROI size)

**Recommendation**: 
1. **Improve Biot-Savart implementation** for better physical accuracy (or document as pedagogical)
2. **Optimize gradient calculation** (remove unnecessary zero term)
3. **Add tests** to verify correctness against analytical solutions

The code **works correctly** from a mathematical optimization perspective, but the Biot-Savart field calculation uses a simplified approximation that may not be physically accurate for all cases.

---

*Analysis Date: Based on code review of shim_coil_biot_savart.py*

