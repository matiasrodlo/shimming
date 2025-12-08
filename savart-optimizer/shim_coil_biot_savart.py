"""
2D Shim-Coil Optimizer using Biot-Savart Law

This is a 2D shim-coil optimizer that uses REAL B0 field map data from a BIDS dataset.
It uses the Biot-Savart formula to compute magnetic fields from circular shim loops
and optimizes loop currents to minimize field variance within the ROI.

The script:
1. Loads real B0 field map data from a BIDS dataset
2. Places circular shim loops around an imaging ROI
3. Optimizes loop currents to minimize variance of (B0 + shim) within the ROI
   using Tikhonov regularization

Note: This is a 2D model (loops in xy-plane, field computed at z=0) and omits
coil coupling and full 3D effects for simplicity. The dataset is REQUIRED.
"""

import os
import sys
import json
import logging
import argparse
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

try:
    from bids import BIDSLayout
    HAS_PYBIDS = True
except ImportError:
    HAS_PYBIDS = False

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(verbose=False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTDIR = "analysis"  # output folder
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
USE_REPO_B0 = True  # optional: if True and dataset B0 provided, compare improvement on repo B0 (cautious)
RANDOM_SEED = 42

# Dataset directory - automatically detected from script location
# Try relative path first, then environment variable, then current directory
def find_dataset_directory():
    """Find dataset directory using multiple strategies."""
    # Strategy 1: Relative to script location (one level up)
    relative_dataset = os.path.join(SCRIPT_DIR, "..", "dataset")
    if os.path.exists(relative_dataset):
        return os.path.abspath(relative_dataset)
    
    # Strategy 2: Two levels up (for backward compatibility)
    relative_dataset2 = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
    if os.path.exists(relative_dataset2):
        return os.path.abspath(relative_dataset2)
    
    # Strategy 3: Environment variable
    env_dataset = os.environ.get('BIDS_DATASET_DIR')
    if env_dataset and os.path.exists(env_dataset):
        return os.path.abspath(env_dataset)
    
    # Strategy 4: Current directory
    current_dataset = os.path.join(os.getcwd(), "dataset")
    if os.path.exists(current_dataset):
        return os.path.abspath(current_dataset)
    
    # Strategy 5: Parent of current directory
    parent_dataset = os.path.join(os.path.dirname(os.getcwd()), "dataset")
    if os.path.exists(parent_dataset):
        return os.path.abspath(parent_dataset)
    
    return None

DATASET_DIR = find_dataset_directory()

# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    if GRID_N <= 0:
        errors.append(f"GRID_N must be positive, got {GRID_N}")
    if GRID_N > DOWNSAMPLE_MAX:
        errors.append(f"GRID_N ({GRID_N}) exceeds DOWNSAMPLE_MAX ({DOWNSAMPLE_MAX})")
    
    if GRID_FOV_MM <= 0:
        errors.append(f"GRID_FOV_MM must be positive, got {GRID_FOV_MM}")
    
    if ROI_RADIUS_MM >= GRID_FOV_MM / 2:
        errors.append(f"ROI_RADIUS_MM ({ROI_RADIUS_MM}) must be < GRID_FOV_MM/2 ({GRID_FOV_MM/2})")
    
    if N_LOOPS <= 0:
        errors.append(f"N_LOOPS must be positive, got {N_LOOPS}")
    
    if R_COIL_MM <= ROI_RADIUS_MM:
        errors.append(f"R_COIL_MM ({R_COIL_MM}) should be > ROI_RADIUS_MM ({ROI_RADIUS_MM})")
    
    if ALPHA < 0:
        errors.append(f"ALPHA must be non-negative, got {ALPHA}")
    
    if len(BOUNDS) != 2 or BOUNDS[0] >= BOUNDS[1]:
        errors.append(f"BOUNDS must be (min, max) with min < max, got {BOUNDS}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def load_bids_fieldmap(dataset_dir, subject='01', acq=None, fmap_type='anat', logger=None):
    """
    Load BIDS-compliant field map with metadata.
    
    Parameters
    ----------
    dataset_dir : str
        Path to BIDS dataset
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV')
    fmap_type : str
        'anat' or 'famp' for field map type
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    data : ndarray
        Field map data
    metadata : dict
        BIDS metadata from JSON sidecar
    affine : ndarray
        Affine transformation matrix
    nii_file : str
        Path to NIfTI file
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading field maps")
    
    # Try using pybids if available
    if HAS_PYBIDS:
        try:
            layout = BIDSLayout(dataset_dir, validate=False)
            
            # Build query
            query = {
                'subject': subject,
                'datatype': 'fmap',
                'suffix': 'TB1TFL',
                'extension': '.nii.gz',
                'return_type': 'filename'
            }
            
            if acq:
                query['acquisition'] = acq
            
            # Filter by fmap_type (anat or famp)
            files = layout.get(**query)
            if files:
                # Filter by acquisition name pattern
                pattern = f"acq-{fmap_type}" if not acq else f"acq-{fmap_type}{acq}"
                files = [f for f in files if pattern in os.path.basename(f)]
            
            if not files:
                raise FileNotFoundError(
                    f"No field map found for subject {subject}, "
                    f"acq {acq}, type {fmap_type}"
                )
            
            nii_file = files[0]
            logger.info(f"Using pybids to load: {nii_file}")
            
        except Exception as e:
            logger.warning(f"pybids failed ({e}), falling back to glob pattern")
            nii_file = None
    else:
        nii_file = None
    
    # Fallback to glob pattern if pybids not available or failed
    if nii_file is None:
        import glob
        subject_str = f"sub-{subject:02d}" if isinstance(subject, int) else f"sub-{subject}"
        pattern = os.path.join(dataset_dir, subject_str, "fmap", f"*_acq-{fmap_type}*_TB1TFL.nii.gz")
        if acq:
            pattern = os.path.join(dataset_dir, subject_str, "fmap", f"*_acq-{fmap_type}{acq}_TB1TFL.nii.gz")
        
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No field map found matching pattern: {pattern}"
            )
        nii_file = files[0]
        logger.info(f"Using glob pattern to load: {nii_file}")
    
    # Load NIfTI file
    try:
        img = nib.load(nii_file)
        data = img.get_fdata()
        affine = img.affine
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {nii_file}: {e}")
    
    # Load JSON metadata
    json_file = nii_file.replace('.nii.gz', '.json')
    if not os.path.exists(json_file):
        logger.warning(f"JSON sidecar not found: {json_file}, using empty metadata")
        metadata = {}
    else:
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from: {json_file}")
        except Exception as e:
            logger.warning(f"Error loading JSON metadata: {e}, using empty metadata")
            metadata = {}
    
    return data, metadata, affine, nii_file


def load_and_resample_b0(dataset_dir, grid_x, grid_y, subject='01', acq=None, logger=None):
    """
    Load B0 field map from dataset and resample to match optimization grid.
    
    Parameters
    ----------
    dataset_dir : str
        Path to BIDS dataset directory
    grid_x : ndarray, shape (Ny, Nx)
        X coordinates of optimization grid (in mm)
    grid_y : ndarray, shape (Ny, Nx)
        Y coordinates of optimization grid (in mm)
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV', 'patient')
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    b0_resampled : ndarray, shape (Ny, Nx)
        B0 field map resampled to match grid
    metadata : dict
        BIDS metadata
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading B0 field maps")
    
    # Load B0 field map
    logger.info(f"\nLoading B0 field map from dataset...")
    logger.info(f"  Subject: {subject}, Acquisition: {acq}")
    b0_data, metadata, affine, nii_file = load_bids_fieldmap(
        dataset_dir, subject=subject, acq=acq, fmap_type='anat', logger=logger
    )
    
    logger.info(f"Loaded B0 map: {nii_file}")
    logger.info(f"  Original shape: {b0_data.shape}")
    
    # Handle 3D/4D data - select central slice
    if len(b0_data.shape) == 3:
        # 3D: select central slice
        central_slice = b0_data.shape[2] // 2
        b0_slice = b0_data[:, :, central_slice]
        logger.info(f"  Selected central slice: {central_slice}")
    elif len(b0_data.shape) == 4:
        # 4D: select central slice and first volume
        central_slice = b0_data.shape[2] // 2
        b0_slice = b0_data[:, :, central_slice, 0]
        logger.info(f"  Selected central slice: {central_slice}, volume 0")
    else:
        b0_slice = b0_data
    
    # Get voxel size from metadata or affine
    if metadata and 'SliceThickness' in metadata:
        voxel_size_z = metadata.get('SliceThickness', 1.0)
        voxel_size_xy = np.abs(np.diag(affine[:2, :2]))
        if len(voxel_size_xy) == 2:
            voxel_size_x, voxel_size_y = voxel_size_xy
        else:
            voxel_size_x = voxel_size_y = voxel_size_xy[0] if len(voxel_size_xy) > 0 else 1.0
    else:
        voxel_sizes = np.abs(np.diag(affine[:3, :3]))
        if len(voxel_sizes) >= 2:
            voxel_size_x, voxel_size_y = voxel_sizes[0], voxel_sizes[1]
        else:
            voxel_size_x = voxel_size_y = 1.0
    
    logger.info(f"  Estimated voxel size: {voxel_size_x:.2f} x {voxel_size_y:.2f} mm")
    
    # Resample to match grid
    Ny, Nx = grid_x.shape
    logger.info(f"  Resampling to grid size: {Ny} x {Nx}")
    
    if HAS_SKIMAGE:
        b0_resampled = transform.resize(b0_slice, (Ny, Nx), order=1, anti_aliasing=True)
    else:
        # Simple downsampling using scipy
        from scipy.ndimage import zoom
        zoom_factors = (Ny / b0_slice.shape[0], Nx / b0_slice.shape[1])
        b0_resampled = zoom(b0_slice, zoom_factors, order=1)
    
    logger.info(f"  Resampled shape: {b0_resampled.shape}")
    logger.info(f"  B0 range: [{np.min(b0_resampled):.6f}, {np.max(b0_resampled):.6f}]")
    
    return b0_resampled, metadata


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
    This is a 2D model: loops are in the xy-plane (z=0) with normal along z.
    Field is computed at z=0 (imaging plane).
    
    Uses a numerically stable Biot-Savart formula for straight wire segments:
    Bz = (μ₀ I / 4π) * segment_length * (r × dl) / |r|³
    
    where:
    - r is vector from segment midpoint to observation point
    - dl is segment direction vector
    - Cross product gives field direction (right-hand rule)
    
    This formulation is more numerically stable than angle-based formulas,
    especially for points near the wire.
    
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
    
    # Biot-Savart constant (arbitrary units, absorbed into scaling)
    mu0_over_4pi = 1.0
    
    # For each segment, compute field using numerically stable Biot-Savart formula
    # For a straight wire segment in 2D (xy-plane), Bz is computed as:
    # Bz = (μ₀ I / 4π) * segment_length * (r × dl) / |r|³
    # where r is vector from segment to observation point
    for i in range(Nseg):
        # Unit vector along segment direction
        seg_dir = seg_vec[i] / (seg_length[i] + 1e-12)
        
        # Vector from segment midpoint to each grid point (more stable than endpoints)
        seg_mid = (seg_start[i] + seg_end[i]) / 2
        r = np.stack([
            grid_x - seg_mid[0],
            grid_y - seg_mid[1]
        ], axis=-1)  # Shape: (Ny, Nx, 2)
        
        # Distance from segment midpoint to observation points
        r_mag = np.linalg.norm(r, axis=-1)  # Shape: (Ny, Nx)
        r_mag = np.maximum(r_mag, 1e-6)  # Avoid division by zero
        
        # Cross product (r × seg_dir) for z-component
        # For 2D: cross_z = r_x * seg_dir_y - r_y * seg_dir_x
        cross_z = r[..., 0] * seg_dir[1] - r[..., 1] * seg_dir[0]
        
        # Biot-Savart contribution: Bz ∝ (r × dl) / |r|³
        # Scale by segment length and use r³ for proper field decay
        contribution = seg_length[i] * cross_z / (r_mag**3 + 1e-10)
        
        # Clip to avoid numerical overflow
        contribution = np.clip(contribution, -1e4, 1e4)
        
        # Apply scaling factor
        Bz += contribution * mu0_over_4pi
    
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
    
    logging.getLogger(__name__).info(f"Computing field maps for {n_loops} loops...")
    for k in range(n_loops):
        M[k] = compute_bz_grid_for_loop(loop_centers[k], loop_radius, grid_x, grid_y)
        norm_k = np.linalg.norm(M[k])
        logging.getLogger(__name__).debug(f"  Loop {k}: L2 norm = {norm_k:.4f}")
    
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


def baseline_field_and_metrics(A, weights0, roi_mask, baseline_field=None):
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
    baseline_field : ndarray, shape (Ny, Nx), optional
        Baseline B0 field map. If provided, total field = baseline + shim.
    
    Returns
    -------
    field : ndarray, shape (Ny, Nx)
        Combined field map (baseline + shim if baseline provided, else just shim)
    metrics : dict
        Dictionary with 'mean', 'std', 'CV' inside ROI
    """
    Ny, Nx = roi_mask.shape
    shim_flat = A @ weights0
    shim_field = shim_flat.reshape(Ny, Nx)
    
    if baseline_field is not None:
        field = baseline_field + shim_field
    else:
        field = shim_field
    
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


def optimize_weights_tikhonov(A, roi_mask, alpha, bounds, w0, method, maxiter, baseline_field=None):
    """
    Optimize loop weights to minimize ROI variance with Tikhonov regularization.
    
    If baseline_field is provided, optimizes: minimize variance of (baseline + shim) in ROI
    Otherwise, optimizes: minimize variance of shim field in ROI
    
    Objective: minimize sum((f_total_roi - mean(f_total_roi))^2) + alpha * ||w||^2
    
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
    baseline_field : ndarray, shape (Ny, Nx), optional
        Baseline B0 field map (e.g., from dataset). If provided, optimization
        minimizes variance of (baseline + shim) instead of just shim.
    
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
    
    # Extract baseline field in ROI if provided
    if baseline_field is not None:
        baseline_roi = baseline_field.flatten()[roi_flat]
    else:
        baseline_roi = np.zeros(len(A_roi))
    
    def objective(w):
        """Objective function: variance in ROI + regularization."""
        shim_roi = A_roi @ w
        f_total_roi = baseline_roi + shim_roi
        mean_f = np.mean(f_total_roi)
        variance = np.sum((f_total_roi - mean_f)**2)
        reg = alpha * np.sum(w**2)
        return variance + reg
    
    def gradient(w):
        """Analytic gradient of objective."""
        shim_roi = A_roi @ w
        f_total_roi = baseline_roi + shim_roi
        mean_f = np.mean(f_total_roi)
        
        # Gradient of variance term: d/dw sum((f_total - mean)^2)
        # = 2 * A_roi.T @ (f_total - mean)
        grad_var = 2 * A_roi.T @ (f_total_roi - mean_f)
        
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
    
    logging.getLogger(__name__).info(f"Saved weights CSV: {fname}")


def maybe_compare_on_repo_b0(DATASET_DIR, grid_x, grid_y, roi_mask, weights, loops, fname, 
                              subject='01', acq=None, logger=None):
    """
    Optional comparison with repository B0 data (if available).
    
    WARNING: This is illustrative only. The comparison should be interpreted with caution.
    
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
    subject : str
        Subject ID (e.g., '01')
    acq : str, optional
        Acquisition type (e.g., 'CP', 'CoV')
    logger : logging.Logger, optional
        Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not USE_REPO_B0 or DATASET_DIR is None:
        logger.debug("Skipping repo B0 comparison: USE_REPO_B0=False or DATASET_DIR=None")
        return
    
    if not HAS_NIBABEL:
        logger.warning("nibabel not available, skipping repo B0 comparison")
        return
    
    logger.info("\n" + "="*60)
    logger.info("WARNING: Comparing with repository B0 data")
    logger.info("This is illustrative only.")
    logger.info("Results should be interpreted with extreme caution.")
    logger.info("="*60 + "\n")
    
    try:
        # Use BIDS-compliant loading
        b0_data, metadata, affine, nii_file = load_bids_fieldmap(
            DATASET_DIR, subject=subject, acq=acq, fmap_type='anat', logger=logger
        )
        
        logger.info(f"Loaded B0 map: {nii_file}")
        if metadata:
            logger.info(f"  SliceThickness: {metadata.get('SliceThickness', 'N/A')} mm")
            logger.info(f"  SpacingBetweenSlices: {metadata.get('SpacingBetweenSlices', 'N/A')} mm")
        
        # Handle 3D/4D data - select central slice
        if b0_data.ndim == 3:
            # Find slice with largest non-zero area
            slice_areas = [np.sum(np.abs(b0_data[:, :, z]) > 0) for z in range(b0_data.shape[2])]
            z_slice = np.argmax(slice_areas) if max(slice_areas) > 0 else b0_data.shape[2] // 2
            b0_slice = b0_data[:, :, z_slice]
            logger.info(f"Selected slice {z_slice} from 3D data")
        elif b0_data.ndim == 4:
            z_slice = b0_data.shape[2] // 2
            t_slice = 0  # take first time point
            b0_slice = b0_data[:, :, z_slice, t_slice]
            logger.info(f"Selected slice {z_slice}, time {t_slice} from 4D data")
        elif b0_data.ndim == 2:
            b0_slice = b0_data
            logger.info("Using 2D data directly")
        else:
            logger.warning(f"Unsupported data dimensions: {b0_data.ndim}D")
            return
        
        # Get voxel size from metadata or affine
        if metadata and 'SliceThickness' in metadata:
            # Use metadata if available
            voxel_size_z = metadata.get('SliceThickness', 1.0)
            # Estimate in-plane voxel size from affine
            voxel_size_xy = np.abs(np.diag(affine[:2, :2]))
            if len(voxel_size_xy) == 2:
                voxel_size_x, voxel_size_y = voxel_size_xy
            else:
                voxel_size_x = voxel_size_y = voxel_size_xy[0] if len(voxel_size_xy) > 0 else 1.0
        else:
            # Fallback: estimate from affine
            voxel_sizes = np.abs(np.diag(affine[:3, :3]))
            if len(voxel_sizes) >= 2:
                voxel_size_x, voxel_size_y = voxel_sizes[0], voxel_sizes[1]
            else:
                voxel_size_x = voxel_size_y = 1.0
        
        logger.info(f"Estimated voxel size: {voxel_size_x:.2f} x {voxel_size_y:.2f} mm")
        
        # Downsample to match grid
        Ny, Nx = grid_x.shape
        if HAS_SKIMAGE:
            b0_downsampled = transform.resize(b0_slice, (Ny, Nx), order=1, anti_aliasing=True)
        else:
            # Simple downsampling
            from scipy.ndimage import zoom
            zoom_factors = (Ny / b0_slice.shape[0], Nx / b0_slice.shape[1])
            b0_downsampled = zoom(b0_slice, zoom_factors, order=1)
        
        # Compute simulated shim field
        _, A = compute_field_matrix(loops, grid_x, grid_y)
        shim_field_flat = A @ weights
        shim_field = shim_field_flat.reshape(Ny, Nx)
        
        # Normalize shim field to have similar scale (arbitrary scaling)
        # This is an illustrative comparison
        b0_roi = b0_downsampled[roi_mask]
        shim_roi = shim_field[roi_mask]
        if np.std(shim_roi) > 1e-10:
            scale = np.std(b0_roi) / np.std(shim_roi) * 0.1  # Arbitrary scaling factor
            shim_field_scaled = shim_field * scale
            logger.info(f"Applied scaling factor: {scale:.6f} (arbitrary, for illustration only)")
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
            writer.writerow(['subject', subject])
            writer.writerow(['acquisition', acq if acq else 'default'])
            writer.writerow(['voxel_size_x_mm', voxel_size_x])
            writer.writerow(['voxel_size_y_mm', voxel_size_y])
            writer.writerow(['note', 'Illustrative comparison only'])
        
        logger.info(f"B0 comparison saved: {fname}")
        logger.info(f"  B0 std before: {std_before:.4f}")
        logger.info(f"  B0 std after: {std_after:.4f}")
        logger.info(f"  Percent reduction: {percent_reduction:.2f}%")
        
    except FileNotFoundError as e:
        logger.warning(f"Field map not found: {e}")
        logger.info("Skipping repo B0 comparison")
    except ImportError as e:
        logger.warning(f"Import error: {e}")
        logger.info("Skipping repo B0 comparison")
    except Exception as e:
        logger.error(f"Error loading/comparing B0 data: {e}", exc_info=True)
        logger.info("Skipping repo B0 comparison")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='2D Shim-Coil Optimizer using Biot-Savart Law - Uses REAL B0 data from dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to BIDS dataset directory (REQUIRED - overrides auto-detection)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        default='01',
        help='Subject ID for B0 field map (e.g., "01", "02")'
    )
    
    parser.add_argument(
        '--acq',
        type=str,
        default=None,
        help='Acquisition type for B0 field map (e.g., "CP", "CoV", "patient")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides OUTDIR config)'
    )
    
    parser.add_argument(
        '--no-repo-b0',
        action='store_true',
        help='Disable repository B0 comparison'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main script execution."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Override dataset directory if provided
    global DATASET_DIR
    if args.dataset_dir:
        if os.path.exists(args.dataset_dir):
            DATASET_DIR = os.path.abspath(args.dataset_dir)
            logger.info(f"Using dataset directory from command line: {DATASET_DIR}")
        else:
            logger.error(f"Dataset directory does not exist: {args.dataset_dir}")
            sys.exit(1)
    
    # Override USE_REPO_B0 if requested
    global USE_REPO_B0
    if args.no_repo_b0:
        USE_REPO_B0 = False
        logger.info("Repository B0 comparison disabled by command line")
    
    # Require dataset directory
    if DATASET_DIR is None:
        logger.error("Dataset directory not found. Please provide --dataset-dir or ensure dataset/ exists.")
        logger.error("The optimizer requires B0 field map data from the dataset.")
        sys.exit(1)
    
    # Validate configuration
    try:
        validate_config()
        logger.debug("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directory
    if args.output_dir:
        outdir_full = os.path.abspath(args.output_dir)
    else:
        outdir_full = os.path.abspath(os.path.join(SCRIPT_DIR, OUTDIR))
    os.makedirs(outdir_full, exist_ok=True)
    logger.info(f"Output directory: {outdir_full}")
    
    # Log configuration
    logger.info("="*60)
    logger.info("2D Shim-Coil Optimizer (Biot-Savart)")
    logger.info("Using REAL B0 data from dataset (not synthetic)")
    logger.info("="*60)
    logger.info(f"OUTDIR: {outdir_full}")
    logger.info(f"GRID_N: {GRID_N}")
    logger.info(f"GRID_FOV_MM: {GRID_FOV_MM}")
    logger.info(f"N_LOOPS: {N_LOOPS}")
    logger.info(f"R_COIL_MM: {R_COIL_MM}")
    logger.info(f"LOOP_RADIUS_MM: {LOOP_RADIUS_MM}")
    logger.info(f"ROI_RADIUS_MM: {ROI_RADIUS_MM}")
    logger.info(f"INITIAL_WEIGHT: {INITIAL_WEIGHT}")
    logger.info(f"BOUNDS: {BOUNDS}")
    logger.info(f"ALPHA: {ALPHA}")
    logger.info(f"OPT_METHOD: {OPT_METHOD}")
    logger.info(f"MAXITER: {MAXITER}")
    logger.info(f"DATASET_DIR: {DATASET_DIR}")
    logger.info(f"Subject: {args.subject}, Acquisition: {args.acq}")
    logger.info("="*60 + "\n")
    
    # Check grid size
    grid_n = GRID_N
    if grid_n > DOWNSAMPLE_MAX:
        logger.warning(f"GRID_N ({grid_n}) > DOWNSAMPLE_MAX ({DOWNSAMPLE_MAX})")
        logger.warning(f"Reducing to {DOWNSAMPLE_MAX}")
        grid_n = DOWNSAMPLE_MAX
    
    # Build imaging grid
    logger.info(f"Creating {grid_n}x{grid_n} imaging grid...")
    x = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    y = np.linspace(-GRID_FOV_MM/2, GRID_FOV_MM/2, grid_n)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Load B0 field map from dataset
    try:
        baseline_b0, b0_metadata = load_and_resample_b0(
            DATASET_DIR, grid_x, grid_y, 
            subject=args.subject, acq=args.acq, logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to load B0 field map from dataset: {e}")
        logger.error("The optimizer requires B0 field map data from the dataset.")
        sys.exit(1)
    
    # Generate loop positions
    logger.info(f"\nGenerating {N_LOOPS} loop positions...")
    loops = make_loop_positions(N_LOOPS, R_COIL_MM, LOOP_RADIUS_MM)
    loop_centers, loop_radius = loops
    logger.info("Loop centers (mm):")
    for k, (x, y) in enumerate(loop_centers):
        logger.info(f"  Loop {k}: ({x:.2f}, {y:.2f})")
    
    # Compute field matrix
    logger.info("\nComputing field matrix...")
    M, A = compute_field_matrix(loops, grid_x, grid_y)
    
    # Create ROI mask
    logger.info("\nCreating ROI mask...")
    roi_mask = make_roi_mask(grid_x, grid_y, ROI_RADIUS_MM)
    n_roi_pixels = np.sum(roi_mask)
    logger.info(f"ROI contains {n_roi_pixels} pixels")
    
    # Baseline field (using real B0 data, no shim initially)
    logger.info("\nComputing baseline field (real B0 data, no shim)...")
    weights0 = np.zeros(N_LOOPS)  # Start with zero shim
    field_before, metrics_before = baseline_field_and_metrics(A, weights0, roi_mask, baseline_field=baseline_b0)
    logger.info("Baseline metrics (ROI):")
    logger.info(f"  Mean: {metrics_before['mean']:.6f}")
    logger.info(f"  Std:  {metrics_before['std']:.6f}")
    logger.info(f"  CV:   {metrics_before['CV']:.6f}")
    
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
    logger.info(f"Saved baseline map: {baseline_path}")
    
    # Optimization
    logger.info("\nOptimizing weights to minimize variance of (B0 + shim)...")
    w_opt, success, obj_value = optimize_weights_tikhonov(
        A, roi_mask, ALPHA, BOUNDS, weights0, OPT_METHOD, MAXITER, baseline_field=baseline_b0
    )
    logger.info(f"Optimizer success: {success}")
    logger.info(f"Final objective value: {obj_value:.6f}")
    logger.info("Optimized weights:")
    for k, w in enumerate(w_opt):
        logger.info(f"  Loop {k}: {w:.6f}")
    
    # Optimized field
    logger.info("\nComputing optimized field (B0 + optimized shim)...")
    field_after, metrics_after = baseline_field_and_metrics(A, w_opt, roi_mask, baseline_field=baseline_b0)
    logger.info("Optimized metrics (ROI):")
    logger.info(f"  Mean: {metrics_after['mean']:.6f}")
    logger.info(f"  Std:  {metrics_after['std']:.6f}")
    logger.info(f"  CV:   {metrics_after['CV']:.6f}")
    
    # Percent reduction
    percent_reduction = 100 * (1 - metrics_after['std'] / (metrics_before['std'] + 1e-10))
    logger.info("\nImprovement:")
    logger.info(f"  Std reduction: {percent_reduction:.2f}%")
    
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
    logger.info(f"Saved optimized map: {optimized_path}")
    
    # Before/after comparison
    logger.info("\nCreating before/after comparison...")
    before_after_path = os.path.join(outdir_full, "biot_savart_before_after.png")
    plot_before_after(field_before, field_after, roi_mask, loops, weights0, w_opt, before_after_path)
    
    # Save weights CSV
    weights_path = os.path.join(outdir_full, "biot_savart_weights.csv")
    save_weights_csv(w_opt, loops, weights_path)
    
    # Optional repo B0 comparison
    if USE_REPO_B0 and DATASET_DIR is not None:
        repo_comparison_path = os.path.join(outdir_full, "biot_savart_repo_comparison.csv")
        maybe_compare_on_repo_b0(
            DATASET_DIR, grid_x, grid_y, roi_mask, w_opt, loops, 
            repo_comparison_path, subject=args.subject, acq=args.acq, logger=logger
        )
    
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
    logger.info(f"Saved stats CSV: {stats_path}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("Saved files:")
    logger.info(f"  - {baseline_path}")
    logger.info(f"  - {optimized_path}")
    logger.info(f"  - {before_after_path}")
    logger.info(f"  - {weights_path}")
    logger.info(f"  - {stats_path}")
    if USE_REPO_B0 and DATASET_DIR is not None:
        logger.info(f"  - {repo_comparison_path}")
    logger.info(f"\nImprovement: {percent_reduction:.2f}% reduction in ROI std")
    logger.info("="*60)


if __name__ == "__main__":
    main()

