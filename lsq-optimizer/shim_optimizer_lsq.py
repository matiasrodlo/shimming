"""
Shim-Coil Optimizer using Bounded Least Squares (Standard Method)

This implements the STANDARD optimization approach for shimming tasks:
- Uses scipy.optimize.lsq_linear (Bounded-Variable Least Squares)
- Guaranteed global optimum (convex problem)
- Faster and more stable than general optimizers
- Standard in shimming literature (Juchem et al., Shimming-Toolbox)

The shimming problem is fundamentally a linear least squares problem:
    minimize ||A*w + B0 - constant||² + α*||w||²
    
This is exactly what lsq_linear solves efficiently.
"""

import os
import sys
import json
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path

# Suppress expected numerical warnings from scipy.optimize
# These occur during matrix operations but are handled internally
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize._lsq')

# Import dependencies
try:
    from scipy import optimize
    from scipy.optimize import lsq_linear
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install: pip install numpy scipy matplotlib")
    sys.exit(1)

try:
    from skimage import transform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTDIR = "analysis"
GRID_N = 300  # Higher resolution
GRID_FOV_MM = 200.0
N_LOOPS = 32  # Maximum loops for best performance!
R_COIL_MM = 45.0  # Even closer to ROI for maximum effect
LOOP_RADIUS_MM = 10.0
ROI_RADIUS_MM = 25.0
BOUNDS = (-1000.0, 1000.0)  # Essentially unconstrained
ALPHA = 0.0  # NO regularization - pure optimization!
RANDOM_SEED = 42
DOWNSAMPLE_MAX = 300

# Auto-detect dataset
DATASET_DIR = None
for candidate in ['../dataset', '../../dataset']:
    full_path = os.path.abspath(os.path.join(SCRIPT_DIR, candidate))
    if os.path.exists(full_path):
        DATASET_DIR = full_path
        break

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# ============================================================================
# DATA LOADING (Same as savart-optimizer)
# ============================================================================

def load_bids_fieldmap(dataset_dir, subject='01', acq=None, fmap_type='anat', logger=None):
    """Load BIDS-compliant field map."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading B0 field maps")
    
    # Build path pattern
    subject_dir = os.path.join(dataset_dir, f'sub-{subject}', 'fmap')
    
    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    
    # Build filename pattern
    if acq:
        pattern = f"sub-{subject}_acq-{fmap_type}{acq}_TB1TFL.nii.gz"
    else:
        pattern = f"sub-{subject}_acq-{fmap_type}_TB1TFL.nii.gz"
    
    nii_file = os.path.join(subject_dir, pattern)
    
    if not os.path.exists(nii_file):
        # Try glob pattern
        import glob
        matches = glob.glob(os.path.join(subject_dir, f"sub-{subject}_acq-{fmap_type}*.nii.gz"))
        if matches:
            nii_file = matches[0]
            logger.info(f"Using glob pattern to load: {nii_file}")
        else:
            raise FileNotFoundError(f"No field map found for subject {subject}, acq {acq}")
    
    # Load NIfTI file
    nii = nib.load(nii_file)
    data = nii.get_fdata()
    affine = nii.affine
    
    # Load JSON metadata
    json_file = nii_file.replace('.nii.gz', '.json')
    metadata = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from: {json_file}")
    
    return data, metadata, affine, nii_file


def load_and_resample_b0(dataset_dir, grid_x, grid_y, subject='01', acq=None, logger=None):
    """Load B0 field map and resample to optimization grid."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required")
    
    logger.info(f"\nLoading B0 field map from dataset...")
    logger.info(f"  Subject: {subject}, Acquisition: {acq}")
    
    data, metadata, affine, nii_file = load_bids_fieldmap(
        dataset_dir, subject=subject, acq=acq, fmap_type='anat', logger=logger
    )
    
    logger.info(f"Loaded B0 map: {nii_file}")
    logger.info(f"  Original shape: {data.shape}")
    
    # Handle 3D/4D data
    if len(data.shape) == 3:
        central_slice = data.shape[2] // 2
        b0_slice = data[:, :, central_slice]
        logger.info(f"  Selected central slice: {central_slice}")
    elif len(data.shape) == 4:
        central_slice = data.shape[2] // 2
        b0_slice = data[:, :, central_slice, 0]
        logger.info(f"  Selected central slice: {central_slice}, volume 0")
    else:
        b0_slice = data
    
    # Resample to grid
    target_shape = grid_x.shape
    logger.info(f"  Resampling to grid size: {target_shape[0]} x {target_shape[1]}")
    
    if HAS_SKIMAGE:
        from skimage.transform import resize
        b0_resampled = resize(b0_slice, target_shape, order=1, preserve_range=True, anti_aliasing=True)
    else:
        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / b0_slice.shape[0], target_shape[1] / b0_slice.shape[1])
        b0_resampled = zoom(b0_slice, zoom_factors, order=1)
    
    logger.info(f"  Resampled shape: {b0_resampled.shape}")
    logger.info(f"  B0 range: [{np.min(b0_resampled):.6f}, {np.max(b0_resampled):.6f}]")
    
    return b0_resampled, metadata

# ============================================================================
# FIELD COMPUTATION (Same as savart-optimizer)
# ============================================================================

def make_loop_positions(n_loops, r_coil_mm, loop_radius_mm):
    """Generate loop positions in a circle."""
    angles = np.linspace(0, 2*np.pi, n_loops, endpoint=False)
    loop_centers = np.zeros((n_loops, 2))
    loop_centers[:, 0] = r_coil_mm * np.cos(angles)
    loop_centers[:, 1] = r_coil_mm * np.sin(angles)
    return (loop_centers, loop_radius_mm)


def compute_bz_grid_for_loop(loop_center, loop_radius_mm, grid_x, grid_y, Nseg=64):
    """Compute Bz field from circular loop using Biot-Savart."""
    mu0_over_4pi = 1000.0  # Arbitrary units, scaled later
    
    # Discretize loop
    theta = np.linspace(0, 2*np.pi, Nseg, endpoint=False)
    dtheta = 2*np.pi / Nseg
    
    wire_x = loop_center[0] + loop_radius_mm * np.cos(theta)
    wire_y = loop_center[1] + loop_radius_mm * np.sin(theta)
    
    dl_x = -loop_radius_mm * np.sin(theta) * dtheta
    dl_y = loop_radius_mm * np.cos(theta) * dtheta
    dl_z = np.zeros_like(dl_x)
    
    # Observation points (z=0)
    obs_x = grid_x.flatten()
    obs_y = grid_y.flatten()
    obs_z = np.zeros_like(obs_x)
    
    Bz = np.zeros_like(obs_x)
    
    for i in range(Nseg):
        rx = obs_x - wire_x[i]
        ry = obs_y - wire_y[i]
        rz = obs_z
        
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.maximum(r_mag, loop_radius_mm * 0.1)  # Avoid singularity
        
        cross_x = dl_y[i] * rz - dl_z[i] * ry
        cross_y = dl_z[i] * rx - dl_x[i] * rz
        cross_z = dl_x[i] * ry - dl_y[i] * rx
        
        contribution = mu0_over_4pi * cross_z / (r_mag**3)
        contribution = np.clip(contribution, -10.0, 10.0)
        
        Bz += contribution
    
    Bz = np.nan_to_num(Bz, nan=0.0, posinf=0.0, neginf=0.0)
    Bz = np.clip(Bz, -10.0, 10.0)
    
    return Bz.reshape(grid_x.shape)


def compute_field_matrix(loops, grid_x, grid_y):
    """Compute design matrix A where each column is one loop's field."""
    loop_centers, loop_radius = loops
    n_loops = len(loop_centers)
    
    Ny, Nx = grid_x.shape
    Npix = Ny * Nx
    
    M = np.zeros((n_loops, Ny, Nx))
    A = np.zeros((Npix, n_loops))
    
    for i, center in enumerate(loop_centers):
        Bz_map = compute_bz_grid_for_loop(center, loop_radius, grid_x, grid_y)
        M[i, :, :] = Bz_map
        A[:, i] = Bz_map.flatten()
    
    # Clean up
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.clip(A, -10.0, 10.0)
    
    return M, A


def make_roi_mask(grid_x, grid_y, roi_radius_mm):
    """Create circular ROI mask."""
    r = np.sqrt(grid_x**2 + grid_y**2)
    return r <= roi_radius_mm

# ============================================================================
# LSQ_LINEAR OPTIMIZER (NEW - Standard Method)
# ============================================================================

def optimize_weights_lsq_linear(A, roi_mask, alpha, bounds, baseline_field=None, logger=None):
    """
    Optimize weights using Bounded Least Squares (STANDARD method for shimming).
    
    This is the standard approach used in shimming literature:
    - Juchem et al., MRM 2011
    - Shimming-Toolbox
    - Stockmann & Wald, 2018 review
    
    Problem formulation:
        minimize ||A*w - target||² + α*||w||²
    
    Where target makes the total field uniform in ROI.
    
    Parameters
    ----------
    A : ndarray, shape (Npix, n_loops)
        Design matrix
    roi_mask : ndarray, bool
        ROI mask
    alpha : float
        Regularization parameter
    bounds : tuple
        (lower, upper) bounds for weights
    baseline_field : ndarray, optional
        Baseline B0 field
    logger : logging.Logger, optional
        Logger
    
    Returns
    -------
    w_opt : ndarray
        Optimized weights
    result : OptimizeResult
        Full optimization result
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Extract ROI
    A_roi = A[roi_mask.flatten()]
    n_loops = A.shape[1]
    
    if baseline_field is not None:
        baseline_roi = baseline_field.flatten()[roi_mask.flatten()]
    else:
        baseline_roi = np.zeros(len(A_roi))
    
    logger.info("Using Bounded Least Squares (lsq_linear):")
    logger.info(f"  Method: BVLS (Bounded-Variable Least Squares)")
    logger.info(f"  This is the STANDARD approach for shimming tasks")
    
    # Formulate least squares problem
    # Target: shim field that makes total field uniform
    # We want: baseline + A*w ≈ constant
    # So: A*w ≈ constant - baseline
    # Choose constant = mean(baseline) for simplicity
    target_value = np.mean(baseline_roi)
    target = target_value - baseline_roi
    
    logger.info(f"  Target field value: {target_value:.2f}")
    logger.info(f"  ROI size: {len(baseline_roi)} pixels")
    
    # Add Tikhonov regularization by augmenting the system
    # Original: minimize ||A*w - target||²
    # With reg: minimize ||A*w - target||² + α*||w||²
    # Augmented: minimize ||[A; sqrt(α)*I]*w - [target; 0]||²
    
    scale = np.sqrt(alpha * len(baseline_roi) / n_loops)
    A_aug = np.vstack([A_roi, scale * np.eye(n_loops)])
    b_aug = np.concatenate([target, np.zeros(n_loops)])
    
    logger.info(f"  Augmented system size: {A_aug.shape}")
    logger.info(f"  Regularization scale: {scale:.6f}")
    
    # Solve bounded least squares
    # Suppress expected numerical warnings during optimization
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            result = lsq_linear(
                A_aug, b_aug,
                bounds=(bounds[0], bounds[1]),
                method='bvls',  # Bounded-Variable Least Squares
                lsq_solver='exact',  # Use exact solver (stable for small problems)
                verbose=0  # Suppress scipy's internal verbose output
            )
    
    logger.info(f"\n  Optimization completed:")
    logger.info(f"    Success: {result.success}")
    logger.info(f"    Status: {result.status}")
    logger.info(f"    Message: {result.message}")
    logger.info(f"    Cost: {result.cost:.6f}")
    logger.info(f"    Optimality: {result.optimality:.6e}")
    
    # Check how many weights at bounds
    at_lower = np.sum(result.x <= bounds[0] + 1e-6)
    at_upper = np.sum(result.x >= bounds[1] - 1e-6)
    
    if at_lower > 0 or at_upper > 0:
        logger.warning(f"    {at_lower} weights at lower bound, {at_upper} at upper bound")
        logger.warning(f"    Consider increasing bounds for better results")
    else:
        logger.info(f"    [OK] All weights in interior (optimal!)")
    
    return result.x, result


def compute_metrics(field, roi_mask):
    """Compute field metrics in ROI."""
    roi_field = field[roi_mask]
    return {
        'mean': float(np.mean(roi_field)),
        'std': float(np.std(roi_field)),
        'cv': float(np.std(roi_field) / (np.mean(roi_field) + 1e-10)),
        'min': float(np.min(roi_field)),
        'max': float(np.max(roi_field))
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(field_before, field_after, roi_mask, loops, weights, outpath, logger=None):
    """Create publication-quality comparison plot."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
    })
    
    loop_centers, loop_radius = loops
    
    # Create 2x2 figure for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('white')
    
    # Grid for plotting
    Ny, Nx = field_before.shape
    x = np.linspace(-100, 100, Nx)
    y = np.linspace(-100, 100, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Compute metrics
    std_before = np.std(field_before[roi_mask])
    std_after = np.std(field_after[roi_mask])
    improvement = 100 * (1 - std_after / std_before)
    
    # Extract ROI-only data for focused visualization
    field_before_roi_only = np.where(roi_mask, field_before, np.nan)
    field_after_roi_only = np.where(roi_mask, field_after, np.nan)
    
    # MAXIMUM CONTRAST: Use absolute min/max of ROI values for ultimate visibility
    before_roi_vals = field_before[roi_mask]
    vmin_before = np.min(before_roi_vals)
    vmax_before = np.max(before_roi_vals)
    
    # For after: use absolute min/max (will be narrower range, but fully stretched)
    after_roi_vals = field_after[roi_mask]
    vmin_after = np.min(after_roi_vals)
    vmax_after = np.max(after_roi_vals)
    
    # Center values for diverging colormap
    center_before = np.mean(before_roi_vals)
    center_after = np.mean(after_roi_vals)
    
    # Top-left: Before (ULTRA HIGH CONTRAST)
    ax = axes[0, 0]
    # 50 levels + contour lines for maximum detail
    im = ax.contourf(X, Y, field_before, levels=50, cmap='RdBu_r', 
                     vmin=vmin_before, vmax=vmax_before, extend='both')
    # Add contour lines for extra detail
    cs = ax.contour(X, Y, field_before, levels=15, colors='black', linewidths=0.3, alpha=0.3)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='#00FF00', linewidths=3.5)
    ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c='#FFD700', s=35, alpha=0.85, 
               edgecolors='black', linewidths=1.2, marker='o', zorder=10)
    ax.set_title(f'(A) Before: B0 Baseline\nσ = {std_before:.1f} | Range: {vmax_before-vmin_before:.0f}', 
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, label='B0 (arb. units)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('B0 (arb. units)', fontsize=9, fontweight='bold')
    
    # Top-right: After (ULTRA HIGH CONTRAST - independent scale!)
    ax = axes[0, 1]
    im = ax.contourf(X, Y, field_after, levels=50, cmap='RdBu_r',
                     vmin=vmin_after, vmax=vmax_after, extend='both')
    # Add contour lines for extra detail
    cs = ax.contour(X, Y, field_after, levels=15, colors='black', linewidths=0.3, alpha=0.3)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='#00FF00', linewidths=3.5)
    ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c='#FFD700', s=35, alpha=0.85,
               edgecolors='black', linewidths=1.2, marker='o', zorder=10)
    pct_range_reduction = 100 * (1 - (vmax_after-vmin_after)/(vmax_before-vmin_before))
    ax.set_title(f'(B) After: B0 + LSQ Shim\nσ = {std_after:.1f} ({improvement:.1f}% reduction) | Range: {vmax_after-vmin_after:.0f}', 
                 fontsize=11, fontweight='bold', pad=10, color='#006400')
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, label='B0 (arb. units)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('B0 (arb. units)', fontsize=9, fontweight='bold')
    
    # Bottom-left: ROI Field Distribution
    ax = axes[1, 0]
    roi_before_vals = field_before[roi_mask]
    roi_after_vals = field_after[roi_mask]
    
    bins = 60
    n_before, bins_before, patches_before = ax.hist(roi_before_vals, bins=bins, alpha=0.65, 
            label=f'Before (σ={std_before:.1f})', color='#DC143C', edgecolor='#8B0000', linewidth=1.2)
    n_after, bins_after, patches_after = ax.hist(roi_after_vals, bins=bins, alpha=0.65, 
            label=f'After (σ={std_after:.1f})', color='#4169E1', edgecolor='#00008B', linewidth=1.2)
    ax.axvline(center_before, color='#DC143C', linestyle='--', linewidth=2.5, label='Before mean', alpha=0.9)
    ax.axvline(center_after, color='#4169E1', linestyle='--', linewidth=2.5, label='After mean', alpha=0.9)
    # Add std ranges as shaded areas
    ax.axvspan(center_before - std_before, center_before + std_before, alpha=0.15, color='#DC143C', label='±1σ range')
    ax.axvspan(center_after - std_after, center_after + std_after, alpha=0.15, color='#4169E1')
    ax.set_xlabel('B0 Field Value (arb. units)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Pixel Count', fontsize=10, fontweight='bold')
    ax.set_title(f'(C) ROI Field Distribution\n{improvement:.1f}% reduction in σ', 
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Bottom-right: Shim field applied
    ax = axes[1, 1]
    field_diff = field_after - field_before
    diff_abs_max = np.nanmax(np.abs(field_diff[roi_mask])) if np.any(roi_mask) else np.nanmax(np.abs(field_diff))
    im = ax.contourf(X, Y, field_diff, levels=30, cmap='RdBu_r', 
                     vmin=-diff_abs_max, vmax=diff_abs_max, extend='both')
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=3)
    ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c='gray', s=30, alpha=0.6,
               edgecolors='black', linewidths=1, marker='o', zorder=10)
    ax.set_title(f'(D) Applied Shim Field\n(After - Before)', 
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax, label='ΔB0 (arb. units)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('ΔB0 (arb. units)', fontsize=9, fontweight='bold')
    
    # Add overall figure title
    fig.suptitle(f'LSQ Shim Optimization Results: {improvement:.1f}% Improvement in Field Homogeneity', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved comparison plot: {outpath}")
    
    # Create ULTRA HIGH CONTRAST ROI-focused plot
    roi_path = outpath.replace('.png', '_roi_focus.png')
    fig = plt.figure(figsize=(18, 11))
    
    # Create grid with detail insets
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ULTRA HIGH CONTRAST: Use absolute min/max with many levels
    levels_before = np.linspace(vmin_before, vmax_before, 60)
    levels_after = np.linspace(vmin_after, vmax_after, 60)
    
    # Top row: Full ROI view
    # Before (ROI only) - ULTRA CONTRAST
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X, Y, field_before_roi_only, levels=levels_before, cmap='seismic', extend='both')
    cs = ax.contour(X, Y, field_before_roi_only, levels=20, colors='black', linewidths=0.2, alpha=0.25)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='lime', linewidths=4)
    ax.set_title(f'BEFORE (ROI)\nstd = {std_before:.1f}', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim([-35, 35])
    ax.set_ylim([-35, 35])
    ax.grid(True, alpha=0.2, linestyle='--')
    cbar = plt.colorbar(im, ax=ax, label='Bz', fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    # Move text box lower to avoid overlap with title
    ax.text(0.02, 0.88, f'Range: {vmin_before:.0f} – {vmax_before:.0f}\nSpan: {vmax_before-vmin_before:.0f}', 
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red', linewidth=2))
    
    # After (ROI only) - ULTRA CONTRAST
    ax = fig.add_subplot(gs[0, 1])
    im = ax.contourf(X, Y, field_after_roi_only, levels=levels_after, cmap='seismic', extend='both')
    cs = ax.contour(X, Y, field_after_roi_only, levels=20, colors='black', linewidths=0.2, alpha=0.25)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='lime', linewidths=4)
    ax.set_title(f'AFTER (ROI)\nstd = {std_after:.1f} [OK]', fontsize=12, fontweight='bold', color='darkgreen', pad=8)
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim([-35, 35])
    ax.set_ylim([-35, 35])
    ax.grid(True, alpha=0.2, linestyle='--')
    cbar = plt.colorbar(im, ax=ax, label='Bz', fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    range_reduction_pct = 100 * (1 - (vmax_after-vmin_after)/(vmax_before-vmin_before))
    # Move text box lower to avoid overlap with title
    ax.text(0.02, 0.88, f'Range: {vmin_after:.0f} – {vmax_after:.0f}\nSpan: {vmax_after-vmin_after:.0f} ({range_reduction_pct:.0f}% less!)', 
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen', linewidth=2))
    
    # Normalized deviation view (z-score)
    ax = fig.add_subplot(gs[0, 2])
    # Show standard deviations from mean
    before_zscore = (field_before - center_before) / std_before
    after_zscore = (field_after - center_after) / std_after
    zscore_improvement = np.where(roi_mask, np.abs(before_zscore) - np.abs(after_zscore), np.nan)
    im = ax.contourf(X, Y, zscore_improvement, levels=30, cmap='RdYlGn', extend='both')
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=3)
    ax.set_title(f'Homogeneity Improvement\n(green = better)', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim([-35, 35])
    ax.set_ylim([-35, 35])
    ax.grid(True, alpha=0.2, linestyle='--')
    cbar = plt.colorbar(im, ax=ax, label='σ improvement', fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    
    # Middle row: Detail insets (zoomed regions)
    # Before - Detail zoom
    ax = fig.add_subplot(gs[1, 0])
    zoom_x, zoom_y = [-15, 15], [-10, 10]
    im = ax.contourf(X, Y, field_before_roi_only, levels=levels_before, cmap='seismic', extend='both')
    cs = ax.contour(X, Y, field_before_roi_only, levels=15, colors='black', linewidths=0.3, alpha=0.4)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='lime', linewidths=3)
    ax.set_title('BEFORE - Detail View', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (mm)', fontsize=9)
    ax.set_ylabel('y (mm)', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='Bz', fraction=0.046)
    
    # After - Detail zoom
    ax = fig.add_subplot(gs[1, 1])
    im = ax.contourf(X, Y, field_after_roi_only, levels=levels_after, cmap='seismic', extend='both')
    cs = ax.contour(X, Y, field_after_roi_only, levels=15, colors='black', linewidths=0.3, alpha=0.4)
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='lime', linewidths=3)
    ax.set_title('AFTER - Detail View [OK]', fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_xlabel('x (mm)', fontsize=9)
    ax.set_ylabel('y (mm)', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='Bz', fraction=0.046)
    
    # Difference map (zoomed)
    ax = fig.add_subplot(gs[1, 2])
    field_diff = field_after - field_before
    field_diff_roi = np.where(roi_mask, field_diff, np.nan)
    diff_abs_max = np.nanmax(np.abs(field_diff_roi))
    im = ax.contourf(X, Y, field_diff_roi, levels=30, cmap='RdBu_r', 
                     vmin=-diff_abs_max, vmax=diff_abs_max, extend='both')
    ax.contour(X, Y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=3)
    ax.set_title('Shim Correction Applied\n(After - Before)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (mm)', fontsize=9)
    ax.set_ylabel('y (mm)', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='ΔBz', fraction=0.046)
    
    # Bottom row: Analysis plots
    # Histogram comparison
    ax = fig.add_subplot(gs[2, :2])
    bins = 60
    ax.hist(before_roi_vals, bins=bins, alpha=0.6, label=f'Before (σ={std_before:.1f})', 
            color='red', edgecolor='darkred', linewidth=1.5)
    ax.hist(after_roi_vals, bins=bins, alpha=0.6, label=f'After (σ={std_after:.1f})', 
            color='blue', edgecolor='darkblue', linewidth=1.5)
    ax.axvline(center_before, color='red', linestyle='--', linewidth=3, label='Before mean', alpha=0.8)
    ax.axvline(center_after, color='blue', linestyle='--', linewidth=3, label='After mean', alpha=0.8)
    # Add std ranges
    ax.axvspan(center_before - std_before, center_before + std_before, alpha=0.2, color='red')
    ax.axvspan(center_after - std_after, center_after + std_after, alpha=0.2, color='blue')
    ax.set_xlabel('Field Value (Bz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pixel Count', fontsize=11, fontweight='bold')
    ax.set_title(f'ROI Field Distribution - {improvement:.1f}% std reduction', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = fig.add_subplot(gs[2, 2])
    sorted_before = np.sort(before_roi_vals)
    sorted_after = np.sort(after_roi_vals)
    cdf_x_before = np.linspace(0, 100, len(sorted_before))
    cdf_x_after = np.linspace(0, 100, len(sorted_after))
    ax.plot(cdf_x_before, sorted_before, 'r-', linewidth=2, label='Before')
    ax.plot(cdf_x_after, sorted_after, 'b-', linewidth=2, label='After')
    ax.axhline(center_before, color='red', linestyle='--', alpha=0.5)
    ax.axhline(center_after, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Percentile', fontsize=10, fontweight='bold')
    ax.set_ylabel('Field Value (Bz)', fontsize=10, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add main title with method info
    fig.suptitle(f'Detailed ROI Analysis: {improvement:.1f}% Improvement in Field Homogeneity\n' +
                 f'Method: Bounded Least Squares (BVLS) | 32 Coils | ROI Radius: 25 mm', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.05, hspace=0.35, wspace=0.3)
    plt.savefig(roi_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved ROI-focused plot: {roi_path}")

# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Shim-coil optimizer using Bounded Least Squares (Standard Method)'
    )
    parser.add_argument('--subject', type=str, default='01', help='Subject ID')
    parser.add_argument('--acq', type=str, default='CP', help='Acquisition type')
    parser.add_argument('--dataset-dir', type=str, default=None, help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose)
    
    # Setup paths
    global DATASET_DIR
    if args.dataset_dir:
        DATASET_DIR = os.path.abspath(args.dataset_dir)
    
    if DATASET_DIR is None or not os.path.exists(DATASET_DIR):
        logger.error("Dataset directory not found!")
        sys.exit(1)
    
    outdir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(SCRIPT_DIR, OUTDIR)
    os.makedirs(outdir, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("BOUNDED LEAST SQUARES SHIM OPTIMIZER (Standard Method)")
    logger.info("=" * 70)
    logger.info(f"Method: lsq_linear with BVLS")
    logger.info(f"Dataset: {DATASET_DIR}")
    logger.info(f"Subject: {args.subject}, Acquisition: {args.acq}")
    logger.info(f"Grid: {GRID_N}x{GRID_N}, FOV: {GRID_FOV_MM}mm")
    logger.info(f"Loops: {N_LOOPS}, Radius: {R_COIL_MM}mm")
    logger.info(f"ROI: {ROI_RADIUS_MM}mm")
    logger.info(f"Bounds: {BOUNDS}")
    logger.info(f"Regularization α: {ALPHA}")
    logger.info("=" * 70)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Build grid
    x = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, GRID_N)
    y = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, GRID_N)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Load B0 data
    logger.info("\n" + "=" * 70)
    baseline_b0, metadata = load_and_resample_b0(
        DATASET_DIR, grid_x, grid_y, subject=args.subject, acq=args.acq, logger=logger
    )
    
    # Generate loops
    logger.info("\n" + "=" * 70)
    logger.info(f"Generating {N_LOOPS} loop positions...")
    loops = make_loop_positions(N_LOOPS, R_COIL_MM, LOOP_RADIUS_MM)
    
    # Compute field matrix
    logger.info("\n" + "=" * 70)
    logger.info("Computing field matrix...")
    M, A = compute_field_matrix(loops, grid_x, grid_y)
    logger.info(f"  Matrix shape: {A.shape}")
    
    # Create ROI
    roi_mask = make_roi_mask(grid_x, grid_y, ROI_RADIUS_MM)
    logger.info(f"  ROI pixels: {np.sum(roi_mask)}")
    
    # Normalize and scale field matrix
    logger.info("\n" + "=" * 70)
    logger.info("Normalizing field matrix...")
    A_roi = A[roi_mask.flatten()]
    field_stds = np.std(A_roi, axis=0)
    normalization = np.where(field_stds > 1e-12, field_stds, 1.0)
    A = A / normalization[np.newaxis, :]
    
    # Global scaling
    b0_std = np.std(baseline_b0[roi_mask])
    # Suppress expected numerical warnings during scaling calculation
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        shim_std = np.std((A[roi_mask.flatten()] @ np.ones(N_LOOPS)))
        # Handle potential NaN/Inf from numerical issues
        if not np.isfinite(shim_std) or shim_std <= 1e-10:
            shim_std = 1.0  # Default if calculation fails
    if shim_std > 1e-10:
        scale = min(b0_std / shim_std, 10.0)
        A = A * scale
        logger.info(f"  Scaling factor: {scale:.2f}")
    
    # Baseline metrics
    logger.info("\n" + "=" * 70)
    logger.info("Baseline field (B0 only):")
    metrics_before = compute_metrics(baseline_b0, roi_mask)
    for key, val in metrics_before.items():
        logger.info(f"  {key}: {val:.2f}")
    
    # Optimize with LSQ_LINEAR
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZING WITH BOUNDED LEAST SQUARES...")
    logger.info("=" * 70)
    
    w_opt, result = optimize_weights_lsq_linear(
        A, roi_mask, ALPHA, BOUNDS, baseline_field=baseline_b0, logger=logger
    )
    
    logger.info("\nOptimized weights:")
    for i, w in enumerate(w_opt):
        logger.info(f"  Loop {i}: {w:+.4f}")
    
    # Compute optimized field
    # Suppress expected numerical warnings during matrix multiplication
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        shim_field = (A @ w_opt).reshape(baseline_b0.shape)
        # Clean any NaN/Inf values
        shim_field = np.nan_to_num(shim_field, nan=0.0, posinf=b0_std*3, neginf=-b0_std*3)
    shim_field = np.clip(shim_field, -b0_std * 3, b0_std * 3)  # Conservative clipping
    field_after = baseline_b0 + shim_field
    
    metrics_after = compute_metrics(field_after, roi_mask)
    
    logger.info("\n" + "=" * 70)
    logger.info("Optimized field (B0 + shim):")
    for key, val in metrics_after.items():
        logger.info(f"  {key}: {val:.2f}")
    
    # Improvement
    improvement = 100 * (1 - metrics_after['std'] / metrics_before['std'])
    logger.info("\n" + "=" * 70)
    logger.info(f"IMPROVEMENT: {improvement:.2f}% reduction in ROI std")
    logger.info("=" * 70)
    
    # Save results
    plot_path = os.path.join(outdir, "lsq_comparison.png")
    plot_comparison(baseline_b0, field_after, roi_mask, loops, w_opt, plot_path, logger)
    
    # Save weights
    weights_path = os.path.join(outdir, "lsq_weights.csv")
    with open(weights_path, 'w') as f:
        f.write('loop_index,weight\n')
        for i, w in enumerate(w_opt):
            f.write(f'{i},{w:.6f}\n')
    logger.info(f"Saved weights: {weights_path}")
    
    # Save stats
    stats_path = os.path.join(outdir, "lsq_stats.csv")
    with open(stats_path, 'w') as f:
        f.write('metric,value\n')
        f.write(f'baseline_std,{metrics_before["std"]:.6f}\n')
        f.write(f'optimized_std,{metrics_after["std"]:.6f}\n')
        f.write(f'improvement_percent,{improvement:.2f}\n')
        f.write(f'method,lsq_linear_bvls\n')
    logger.info(f"Saved stats: {stats_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("[OK] OPTIMIZATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

