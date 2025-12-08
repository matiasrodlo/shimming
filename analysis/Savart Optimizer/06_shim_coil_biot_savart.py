"""
2D Toy Shim-Coil Optimizer using Biot-Savart Law

This is a 2D toy shim-coil optimizer that uses a simplified Biot-Savart numeric
approximation for circular loops and optimizes scalar loop currents. It is pedagogical
and omits coil coupling, full 3D effects, and realistic conductor geometry.

The script places circular shim loops around an imaging ROI and optimizes their
currents to minimize field variance within the ROI using Tikhonov regularization.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Import dependencies with error handling
try:
    from scipy import optimize
    from scipy import special
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Suggested: pip install numpy scipy matplotlib")
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

OUTDIR = "analysis/analysis_outputs"  # output folder
GRID_N = 200  # grid resolution (max 300 recommended)
GRID_FOV_MM = 200.0  # field-of-view in mm
N_LOOPS = 8  # number of shim loops placed around the imaging ROI
R_COIL_MM = 80.0  # radius (mm) of coil ring where loops are mounted
LOOP_RADIUS_MM = 10.0  # physical radius of each circular loop conductor (mm)
ROI_RADIUS_MM = 25.0  # radius of central ROI to optimize (mm)
INITIAL_WEIGHT = 0.2  # initial current amplitude for each loop (arbitrary units)
BOUNDS = (-1.0, 1.0)  # allowed current weight bounds for optimizer
ALPHA = 1e-3  # Tikhonov regularization on weights
OPT_METHOD = "L-BFGS-B"  # scipy.optimize method
MAXITER = 500
DOWNSAMPLE_MAX = 300
USE_REPO_B0 = False  # optional: if True and dataset B0 provided, compare improvement on repo B0 (cautious)
DATASET_DIR = None  # local ds004906 path if USE_REPO_B0 True
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_loop_positions(n_loops, R_coil_mm, loop_radius_mm):
    """
    Generate positions and geometry for shim loops placed evenly around a circle.
    
    Parameters
    ----------
    n_loops : int
        Number of loops
    R_coil_mm : float
        Radius of coil ring (mm)
    loop_radius_mm : float
        Physical radius of each loop (mm)
    
    Returns
    -------
    loop_centers : ndarray, shape (n_loops, 2)
        (x, y) positions of loop centers in mm
    loop_radius : float
        Physical loop radius (same for all)
    """
    angles = np.linspace(0, 2 * np.pi, n_loops, endpoint=False)
    loop_centers = np.zeros((n_loops, 2))
    loop_centers[:, 0] = R_coil_mm * np.cos(angles)
    loop_centers[:, 1] = R_coil_mm * np.sin(angles)
    return loop_centers, loop_radius_mm


def compute_bz_grid_for_loop(loop_center, loop_radius_mm, grid_x, grid_y, Nseg=64):
    """
    Numerically compute Bz field on the plane from a circular loop using Biot-Savart.
    
    Approximates the loop as Nseg straight segments and sums Biot-Savart contributions.
    This is a simplified 2D model: loops are in the xy-plane (z=0) with normal along z.
    Field is computed at z=0 (imaging plane).
    
    Parameters
    ----------
    loop_center : array-like, shape (2,)
        (x, y) center of loop in mm
    loop_radius_mm : float
        Radius of loop in mm
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates of grid points in mm
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates of grid points in mm
    Nseg : int
        Number of segments to discretize the loop
    
    Returns
    -------
    Bz : ndarray, shape (Ny, Nx)
        Bz field component (arbitrary units proportional to current)
    """
    # Discretize loop into segments
    seg_angles = np.linspace(0, 2 * np.pi, Nseg, endpoint=False)
    seg_start = np.zeros((Nseg, 2))
    seg_start[:, 0] = loop_center[0] + loop_radius_mm * np.cos(seg_angles)
    seg_start[:, 1] = loop_center[1] + loop_radius_mm * np.sin(seg_angles)
    
    # End points (next segment start)
    seg_end = np.zeros((Nseg, 2))
    seg_end[:, 0] = loop_center[0] + loop_radius_mm * np.cos(seg_angles + 2 * np.pi / Nseg)
    seg_end[:, 1] = loop_center[1] + loop_radius_mm * np.sin(seg_angles + 2 * np.pi / Nseg)
    
    # Segment vectors
    seg_vec = seg_end - seg_start
    seg_length = np.linalg.norm(seg_vec, axis=1)
    
    # Initialize Bz field
    Bz = np.zeros_like(grid_x)
    Ny, Nx = grid_x.shape
    
    # Biot-Savart for each segment
    # For a straight segment: dB = (mu0 * I / 4*pi) * (dl × r) / r^3
    # In our 2D approximation, we compute the z-component
    # Simplified: dBz proportional to segment contribution
    mu0_over_4pi = 1.0  # Arbitrary units (absorbed into scaling)
    
    for i in range(Nseg):
        # Vector from segment start to grid points
        dx = grid_x - seg_start[i, 0]
        dy = grid_y - seg_start[i, 1]
        
        # Vector from segment end to grid points
        dx_end = grid_x - seg_end[i, 0]
        dy_end = grid_y - seg_end[i, 1]
        
        # Distance vectors
        r_start = np.sqrt(dx**2 + dy**2 + 1e-10)  # Add small epsilon to avoid division by zero
        r_end = np.sqrt(dx_end**2 + dy_end**2 + 1e-10)
        
        # Segment direction (unit vector)
        seg_dir = seg_vec[i] / (seg_length[i] + 1e-10)
        
        # Cross product contribution (simplified for z-component)
        # For a segment in xy-plane, Bz comes from the perpendicular component
        # More accurate: use the full 3D Biot-Savart formula projected to z
        # Simplified approximation: Bz ~ (segment_length * perpendicular_distance) / r^3
        
        # Perpendicular distance from grid point to segment line
        # Project grid point onto segment line
        seg_to_point = np.stack([dx, dy], axis=-1)
        proj = np.sum(seg_to_point * seg_dir, axis=-1, keepdims=True) * seg_dir
        perp_vec = seg_to_point - proj
        perp_dist = np.linalg.norm(perp_vec, axis=-1)
        
        # Simplified Bz contribution (ignoring full 3D geometry)
        # This is a pedagogical approximation
        r_avg = (r_start + r_end) / 2
        contribution = seg_length[i] * perp_dist / (r_avg**3 + 1e-10)
        
        # Sign: depends on segment direction relative to observation point
        # Simplified: use sign based on cross product
        cross_z = dx * seg_dir[1] - dy * seg_dir[0]
        sign = np.sign(cross_z + 1e-10)
        
        Bz += sign * contribution * mu0_over_4pi
    
    return Bz


def compute_field_matrix(loops, grid_x, grid_y):
    """
    Compute Bz field maps for all loops and create design matrix.
    
    Parameters
    ----------
    loops : tuple
        (loop_centers, loop_radius) from make_loop_positions
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates
    
    Returns
    -------
    M : ndarray, shape (n_loops, Ny, Nx)
        Stacked field maps for each loop
    A : ndarray, shape (Npix, n_loops)
        Flattened design matrix (Npix = Ny * Nx)
    """
    loop_centers, loop_radius = loops
    n_loops = len(loop_centers)
    Ny, Nx = grid_x.shape
    
    M = np.zeros((n_loops, Ny, Nx))
    
    print(f"Computing field maps for {n_loops} loops...")
    for k in range(n_loops):
        M[k] = compute_bz_grid_for_loop(loop_centers[k], loop_radius, grid_x, grid_y)
        norm_k = np.linalg.norm(M[k])
        print(f"  Loop {k}: L2 norm = {norm_k:.4f}")
    
    # Flatten to design matrix
    Npix = Ny * Nx
    A = M.reshape(n_loops, Npix).T  # Shape: (Npix, n_loops)
    
    return M, A


def make_roi_mask(grid_x, grid_y, roi_radius_mm):
    """
    Create boolean mask for central circular ROI.
    
    Parameters
    ----------
    grid_x : ndarray
        X coordinates
    grid_y : ndarray
        Y coordinates
    roi_radius_mm : float
        ROI radius in mm
    
    Returns
    -------
    mask : ndarray, bool
        True inside ROI
    """
    r = np.sqrt(grid_x**2 + grid_y**2)
    return r <= roi_radius_mm


def baseline_field_and_metrics(A, weights0, roi_mask):
    """
    Compute combined field and metrics inside ROI.
    
    Parameters
    ----------
    A : ndarray, shape (Npix, n_loops)
        Design matrix
    weights0 : ndarray, shape (n_loops,)
        Current weights
    roi_mask : ndarray, bool
        ROI mask
    
    Returns
    -------
    field : ndarray, shape (Ny, Nx)
        Combined field map
    metrics : dict
        Dictionary with 'mean', 'std', 'CV' inside ROI
    """
    Ny, Nx = roi_mask.shape
    field_flat = A @ weights0
    field = field_flat.reshape(Ny, Nx)
    
    roi_field = field[roi_mask]
    mean_val = np.mean(roi_field)
    std_val = np.std(roi_field)
    cv_val = std_val / (np.abs(mean_val) + 1e-10)
    
    metrics = {
        'mean': mean_val,
        'std': std_val,
        'CV': cv_val
    }
    
    return field, metrics


def optimize_weights_tikhonov(A, roi_mask, alpha, bounds, w0, method, maxiter):
    """
    Optimize loop weights to minimize ROI variance with Tikhonov regularization.
    
    Objective: minimize sum((f_roi - mean(f_roi))^2) + alpha * ||w||^2
    
    Parameters
    ----------
    A : ndarray, shape (Npix, n_loops)
        Design matrix
    roi_mask : ndarray, bool
        ROI mask
    alpha : float
        Regularization strength
    bounds : tuple
        (min, max) bounds for weights
    w0 : ndarray
        Initial weights
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
    
    Returns
    -------
    w_opt : ndarray
        Optimized weights
    success : bool
        Optimizer success flag
    obj_value : float
        Final objective value
    """
    Ny, Nx = roi_mask.shape
    roi_flat = roi_mask.flatten()
    n_loops = A.shape[1]
    
    # Extract ROI rows from design matrix
    A_roi = A[roi_flat]
    
    def objective(w):
        """Objective function: variance in ROI + regularization."""
        f_roi = A_roi @ w
        mean_f = np.mean(f_roi)
        variance = np.sum((f_roi - mean_f)**2)
        reg = alpha * np.sum(w**2)
        return variance + reg
    
    def gradient(w):
        """Analytic gradient of objective."""
        f_roi = A_roi @ w
        mean_f = np.mean(f_roi)
        n_roi = len(f_roi)
        
        # Gradient of variance term
        grad_var = 2 * A_roi.T @ (f_roi - mean_f) - 2 * np.mean(f_roi - mean_f) * np.sum(A_roi, axis=0)
        
        # Gradient of regularization term
        grad_reg = 2 * alpha * w
        
        return grad_var + grad_reg
    
    # Optimize
    result = optimize.minimize(
        objective,
        w0,
        method=method,
        jac=gradient,
        bounds=[bounds] * n_loops,
        options={'maxiter': maxiter}
    )
    
    return result.x, result.success, result.fun


def plot_before_after(field_before, field_after, roi_mask, loops, weights_before, weights_after, outpath):
    """
    Create multi-panel before/after comparison figure.
    
    Parameters
    ----------
    field_before : ndarray
        Baseline field map
    field_after : ndarray
        Optimized field map
    roi_mask : ndarray
        ROI mask
    loops : tuple
        (loop_centers, loop_radius)
    weights_before : ndarray
        Initial weights
    weights_after : ndarray
        Optimized weights
    outpath : str
        Output file path
    """
    loop_centers, _ = loops
    Ny, Nx = field_before.shape
    
    # Create grid for plotting
    x_plot = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, Nx)
    y_plot = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, Ny)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Common color scale
    vmin = min(np.min(field_before), np.min(field_after))
    vmax = max(np.max(field_before), np.max(field_after))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: before
    ax = axes[0]
    im = ax.contourf(X_plot, Y_plot, field_before, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.contour(X_plot, Y_plot, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c=weights_before, 
               s=100, cmap='RdBu_r', edgecolors='black', linewidths=1, vmin=-1, vmax=1)
    for k, (x, y) in enumerate(loop_centers):
        ax.annotate(f'{weights_before[k]:.2f}', (x, y), fontsize=8, ha='center', va='center')
    ax.set_title('Before Optimization')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Bz (arb. units)')
    
    # Middle: after
    ax = axes[1]
    im = ax.contourf(X_plot, Y_plot, field_after, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.contour(X_plot, Y_plot, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    ax.scatter(loop_centers[:, 0], loop_centers[:, 1], c=weights_after, 
               s=100, cmap='RdBu_r', edgecolors='black', linewidths=1, vmin=-1, vmax=1)
    for k, (x, y) in enumerate(loop_centers):
        ax.annotate(f'{weights_after[k]:.2f}', (x, y), fontsize=8, ha='center', va='center')
    ax.set_title('After Optimization')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Bz (arb. units)')
    
    # Right: weights bar chart
    ax = axes[2]
    x_pos = np.arange(len(weights_before))
    width = 0.35
    ax.bar(x_pos - width/2, weights_before, width, label='Before', alpha=0.7)
    ax.bar(x_pos + width/2, weights_after, width, label='After', alpha=0.7)
    ax.set_xlabel('Loop Index')
    ax.set_ylabel('Weight')
    ax.set_title('Loop Weights')
    ax.set_xticks(x_pos)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {outpath}")


def save_weights_csv(weights, loops, fname):
    """
    Save weights to CSV with loop positions.
    
    Parameters
    ----------
    weights : ndarray
        Loop weights
    loops : tuple
        (loop_centers, loop_radius)
    fname : str
        Output filename
    """
    loop_centers, _ = loops
    data = {
        'loop_index': np.arange(len(weights)),
        'x_mm': loop_centers[:, 0],
        'y_mm': loop_centers[:, 1],
        'weight': weights
    }
    
    # Simple CSV writing
    with open(fname, 'w') as f:
        f.write('loop_index,x_mm,y_mm,weight\n')
        for i in range(len(weights)):
            f.write(f'{i},{loop_centers[i,0]:.4f},{loop_centers[i,1]:.4f},{weights[i]:.6f}\n')
    
    print(f"Saved weights CSV: {fname}")


def maybe_compare_on_repo_b0(DATASET_DIR, grid_x, grid_y, roi_mask, weights, loops, fname):
    """
    Optional comparison with repository B0 data (if available).
    
    WARNING: This is illustrative only. The coil geometry is a toy model and
    the comparison should be interpreted with caution.
    
    Parameters
    ----------
    DATASET_DIR : str
        Dataset directory path
    grid_x : ndarray
        Grid X coordinates
    grid_y : ndarray
        Grid Y coordinates
    roi_mask : ndarray
        ROI mask
    weights : ndarray
        Optimized weights
    loops : tuple
        (loop_centers, loop_radius)
    fname : str
        Output CSV filename
    """
    if not USE_REPO_B0 or DATASET_DIR is None:
        return
    
    if not HAS_NIBABEL:
        print("Warning: nibabel not available, skipping repo B0 comparison")
        return
    
    print("\n" + "="*60)
    print("WARNING: Comparing with repository B0 data")
    print("This is illustrative only. Coil geometry is a toy model.")
    print("Results should be interpreted with extreme caution.")
    print("="*60 + "\n")
    
    # Try to find a B0 map in the dataset
    # Look for fmap files (field maps)
    fmap_pattern = os.path.join(DATASET_DIR, "sub-*/fmap/*.nii.gz")
    import glob
    fmap_files = glob.glob(fmap_pattern)
    
    if not fmap_files:
        print("No B0 field maps found in dataset, skipping comparison")
        return
    
    # Use first available B0 map
    b0_file = fmap_files[0]
    print(f"Loading B0 map: {b0_file}")
    
    try:
        b0_img = nib.load(b0_file)
        b0_data = b0_img.get_fdata()
        
        # Downsample to match grid
        Ny, Nx = grid_x.shape
        if HAS_SKIMAGE:
            b0_downsampled = transform.resize(b0_data, (Ny, Nx), order=1, anti_aliasing=True)
        else:
            # Simple downsampling
            from scipy.ndimage import zoom
            zoom_factors = (Ny / b0_data.shape[0], Nx / b0_data.shape[1])
            b0_downsampled = zoom(b0_data, zoom_factors, order=1)
        
        # Compute simulated shim field
        _, A = compute_field_matrix(loops, grid_x, grid_y)
        shim_field_flat = A @ weights
        shim_field = shim_field_flat.reshape(Ny, Nx)
        
        # Normalize shim field to have similar scale (arbitrary scaling)
        # This is a toy comparison
        b0_roi = b0_downsampled[roi_mask]
        shim_roi = shim_field[roi_mask]
        if np.std(shim_roi) > 1e-10:
            scale = np.std(b0_roi) / np.std(shim_roi) * 0.1  # Arbitrary scaling factor
            shim_field_scaled = shim_field * scale
        else:
            shim_field_scaled = shim_field
        
        # Apply correction (subtract shim field)
        b0_corrected = b0_downsampled - shim_field_scaled
        
        # Compute metrics
        b0_roi_before = b0_downsampled[roi_mask]
        b0_roi_after = b0_corrected[roi_mask]
        
        std_before = np.std(b0_roi_before)
        std_after = np.std(b0_roi_after)
        percent_reduction = 100 * (1 - std_after / (std_before + 1e-10))
        
        # Save comparison
        import csv
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['b0_std_before', std_before])
            writer.writerow(['b0_std_after', std_after])
            writer.writerow(['percent_reduction', percent_reduction])
            writer.writerow(['note', 'Illustrative comparison only - toy coil model'])
        
        print(f"B0 comparison saved: {fname}")
        print(f"  B0 std before: {std_before:.4f}")
        print(f"  B0 std after: {std_after:.4f}")
        print(f"  Percent reduction: {percent_reduction:.2f}%")
        
    except Exception as e:
        print(f"Error loading/comparing B0 data: {e}")
        print("Skipping repo B0 comparison")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main script execution."""
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directory (OUTDIR is relative to project root)
    project_root = os.path.join(SCRIPT_DIR, "..", "..")
    outdir_full = os.path.abspath(os.path.join(project_root, OUTDIR))
    os.makedirs(outdir_full, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("2D Toy Shim-Coil Optimizer (Biot-Savart)")
    print("="*60)
    print(f"OUTDIR: {OUTDIR}")
    print(f"GRID_N: {GRID_N}")
    print(f"GRID_FOV_MM: {GRID_FOV_MM}")
    print(f"N_LOOPS: {N_LOOPS}")
    print(f"R_COIL_MM: {R_COIL_MM}")
    print(f"LOOP_RADIUS_MM: {LOOP_RADIUS_MM}")
    print(f"ROI_RADIUS_MM: {ROI_RADIUS_MM}")
    print(f"INITIAL_WEIGHT: {INITIAL_WEIGHT}")
    print(f"BOUNDS: {BOUNDS}")
    print(f"ALPHA: {ALPHA}")
    print(f"OPT_METHOD: {OPT_METHOD}")
    print(f"MAXITER: {MAXITER}")
    print(f"USE_REPO_B0: {USE_REPO_B0}")
    print("="*60 + "\n")
    
    # Check grid size
    grid_n = GRID_N
    if grid_n > DOWNSAMPLE_MAX:
        print(f"Warning: GRID_N ({grid_n}) > DOWNSAMPLE_MAX ({DOWNSAMPLE_MAX})")
        print(f"Reducing to {DOWNSAMPLE_MAX}")
        grid_n = DOWNSAMPLE_MAX
    
    # Build imaging grid
    print(f"Creating {grid_n}x{grid_n} imaging grid...")
    x = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    y = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Generate loop positions
    print(f"\nGenerating {N_LOOPS} loop positions...")
    loops = make_loop_positions(N_LOOPS, R_COIL_MM, LOOP_RADIUS_MM)
    loop_centers, loop_radius = loops
    print(f"Loop centers (mm):")
    for k, (x, y) in enumerate(loop_centers):
        print(f"  Loop {k}: ({x:.2f}, {y:.2f})")
    
    # Compute field matrix
    print("\nComputing field matrix...")
    M, A = compute_field_matrix(loops, grid_x, grid_y)
    
    # Create ROI mask
    print("\nCreating ROI mask...")
    roi_mask = make_roi_mask(grid_x, grid_y, ROI_RADIUS_MM)
    n_roi_pixels = np.sum(roi_mask)
    print(f"ROI contains {n_roi_pixels} pixels")
    
    # Baseline field
    print("\nComputing baseline field...")
    weights0 = np.ones(N_LOOPS) * INITIAL_WEIGHT
    field_before, metrics_before = baseline_field_and_metrics(A, weights0, roi_mask)
    print(f"Baseline metrics (ROI):")
    print(f"  Mean: {metrics_before['mean']:.6f}")
    print(f"  Std:  {metrics_before['std']:.6f}")
    print(f"  CV:   {metrics_before['CV']:.6f}")
    
    # Save baseline map (optional)
    baseline_path = os.path.join(outdir_full, "biot_savart_baseline.png")
    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, field_before, levels=20, cmap='RdBu_r')
    plt.contour(grid_x, grid_y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    plt.scatter(loop_centers[:, 0], loop_centers[:, 1], c='red', s=50, marker='o', edgecolors='black')
    plt.colorbar(label='Bz (arb. units)')
    plt.title('Baseline Field')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.savefig(baseline_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline map: {baseline_path}")
    
    # Optimization
    print("\nOptimizing weights...")
    w_opt, success, obj_value = optimize_weights_tikhonov(
        A, roi_mask, ALPHA, BOUNDS, weights0, OPT_METHOD, MAXITER
    )
    print(f"Optimizer success: {success}")
    print(f"Final objective value: {obj_value:.6f}")
    print(f"Optimized weights:")
    for k, w in enumerate(w_opt):
        print(f"  Loop {k}: {w:.6f}")
    
    # Optimized field
    print("\nComputing optimized field...")
    field_after, metrics_after = baseline_field_and_metrics(A, w_opt, roi_mask)
    print(f"Optimized metrics (ROI):")
    print(f"  Mean: {metrics_after['mean']:.6f}")
    print(f"  Std:  {metrics_after['std']:.6f}")
    print(f"  CV:   {metrics_after['CV']:.6f}")
    
    # Percent reduction
    percent_reduction = 100 * (1 - metrics_after['std'] / (metrics_before['std'] + 1e-10))
    print(f"\nImprovement:")
    print(f"  Std reduction: {percent_reduction:.2f}%")
    
    # Save optimized map (optional)
    optimized_path = os.path.join(outdir_full, "biot_savart_optimized.png")
    plt.figure(figsize=(8, 8))
    plt.contourf(grid_x, grid_y, field_after, levels=20, cmap='RdBu_r')
    plt.contour(grid_x, grid_y, roi_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
    plt.scatter(loop_centers[:, 0], loop_centers[:, 1], c=w_opt, s=50, cmap='RdBu_r', 
                edgecolors='black', vmin=-1, vmax=1)
    plt.colorbar(label='Bz (arb. units)')
    plt.title('Optimized Field')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.savefig(optimized_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved optimized map: {optimized_path}")
    
    # Before/after comparison
    print("\nCreating before/after comparison...")
    before_after_path = os.path.join(outdir_full, "biot_savart_before_after.png")
    plot_before_after(field_before, field_after, roi_mask, loops, weights0, w_opt, before_after_path)
    
    # Save weights CSV
    weights_path = os.path.join(outdir_full, "biot_savart_weights.csv")
    save_weights_csv(w_opt, loops, weights_path)
    
    # Optional repo B0 comparison
    if USE_REPO_B0 and DATASET_DIR is not None:
        repo_comparison_path = os.path.join(outdir_full, "biot_savart_repo_comparison.csv")
        maybe_compare_on_repo_b0(DATASET_DIR, grid_x, grid_y, roi_mask, w_opt, loops, repo_comparison_path)
    
    # Save stats CSV
    stats_path = os.path.join(outdir_full, "biot_savart_stats.csv")
    with open(stats_path, 'w') as f:
        f.write('metric,value\n')
        f.write(f'baseline_std,{metrics_before["std"]:.6f}\n')
        f.write(f'optimized_std,{metrics_after["std"]:.6f}\n')
        f.write(f'percent_reduction,{percent_reduction:.2f}\n')
        f.write(f'grid_N,{grid_n}\n')
        f.write(f'roi_radius_mm,{ROI_RADIUS_MM}\n')
        f.write(f'alpha,{ALPHA}\n')
        f.write(f'optimizer_success,{success}\n')
    print(f"Saved stats CSV: {stats_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Saved files:")
    print(f"  - {baseline_path}")
    print(f"  - {optimized_path}")
    print(f"  - {before_after_path}")
    print(f"  - {weights_path}")
    print(f"  - {stats_path}")
    if USE_REPO_B0 and DATASET_DIR is not None:
        print(f"  - {repo_comparison_path}")
    print(f"\nImprovement: {percent_reduction:.2f}% reduction in ROI std")
    print("="*60)


if __name__ == "__main__":
    main()

