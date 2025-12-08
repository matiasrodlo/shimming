"""
EPI Distortion Simulation and Unwarp Script

This script simulates EPI (Echo Planar Imaging) distortion caused by B0 field inhomogeneities
and demonstrates unwarping correction using B0 field maps.

Purpose:
- Load or generate a B0 field map
- Create a test phantom image
- Simulate EPI distortion by warping the phantom along the phase-encode axis
- Apply unwarping correction to restore the original image
- Compute quality metrics (MSE, SSIM) to assess correction effectiveness

Distortion Model:
- EPI distortion occurs along the phase-encode direction due to B0 field inhomogeneities
- Pixel shifts are proportional to the local B0 field offset and echo time (TE)
- Formula: shift_pixels ≈ (B0 * TE) / (BW_per_pixel) where B0 is in Hz, TE in seconds

IMPORTANT LIMITATION:
This is a pedagogical simulation of EPI distortion and unwarp using processed B0 estimates
and simple geometric models. Not a scanner reconstruction tool. The simulation uses simplified
models and may not capture all aspects of real EPI distortion (e.g., through-slice effects,
chemical shift, etc.).

Data source: ds004906 (rf-shimming-7t) — optional B0 source
Dataset available at: https://openneuro.org/datasets/ds004906
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import checks with helpful error messages
try:
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator, griddata
except ImportError:
    print("ERROR: Missing scipy. Install with: pip install numpy scipy matplotlib scikit-image nibabel")
    sys.exit(1)

try:
    from skimage.transform import resize
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("ERROR: Missing scikit-image. Install with: pip install numpy scipy matplotlib scikit-image nibabel")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("WARNING: nibabel not available. Repository B0 loading will be skipped.")
    nib = None

try:
    import pandas as pd
except ImportError:
    print("ERROR: Missing pandas. Install with: pip install numpy scipy matplotlib scikit-image nibabel pandas")
    sys.exit(1)

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory - automatically detected from script location
# From analysis/03/, dataset is at ../../dataset
_relative_dataset = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
_absolute_dataset = "/Users/matiasrodlo/Documents/github/shiming/dataset"

if os.path.exists(_relative_dataset):
    DATASET_DIR = os.path.abspath(_relative_dataset)
elif os.path.exists(_absolute_dataset):
    DATASET_DIR = _absolute_dataset
else:
    # Fallback: user must edit this path if dataset is elsewhere
    DATASET_DIR = None  # Will use synthetic B0 if None

# Subject selection: None = auto-select first subject found
SUBJECT = None  # default None -> auto-select first subject in dataset

# Output directory (will be created if missing)
OUTDIR = os.path.join(SCRIPT_DIR, "..", "analysis_outputs")
OUTDIR = os.path.abspath(OUTDIR)

# Image processing parameters
DOWNSAMPLE_MAX = 256  # image max side to keep runtime low

# EPI sequence parameters
TE = 0.03  # echo time in seconds (default 30 ms)
BW_PER_PIXEL = 1500.0  # Hz per pixel (example; used to compute pixel shifts)

# Distortion parameters
PHASE_ENCODE_AXIS = 1  # 0 = X axis, 1 = Y axis (choose which axis distorts)

# Data source options
USE_REPO_B0 = True  # try to load a B0/fmap from DATASET_DIR if available, otherwise use synthetic

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_repo_b0(dataset_dir, subject):
    """
    Search DATASET_DIR/sub-*/fmap/ for candidate B0 or TB1map files.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset root directory
    subject : str or None
        Subject ID (e.g., 'sub-01'). If None, searches all subjects.
    
    Returns:
    --------
    path : str or None
        Path to B0/fmap file if found, None otherwise
    """
    if not os.path.exists(dataset_dir):
        return None
    
    # Search patterns for B0/fmap files
    patterns = [
        "*_TB1map.nii*",
        "*famp*_TB1TFL.nii*",
        "*fmap*.nii*",
        "*fieldmap*.nii*",
        "*B0*.nii*"
    ]
    
    if subject is None:
        # Search all subjects
        subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sub-*")))
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
        if not subject_dirs:
            return None
        # Use first subject
        subject = os.path.basename(subject_dirs[0])
    
    subject_dir = os.path.join(dataset_dir, subject)
    fmap_dir = os.path.join(subject_dir, "fmap")
    
    if not os.path.exists(fmap_dir):
        return None
    
    # Search for matching files
    for pattern in patterns:
        matches = glob.glob(os.path.join(fmap_dir, pattern))
        if matches:
            return matches[0]  # Return first match
    
    return None


def load_b0_slice(path, downsample_to=None):
    """
    Load NIfTI with nibabel, pick central slice with largest mask area (or central z),
    return 2D array, downsample to downsample_to if set.
    
    Parameters:
    -----------
    path : str
        Path to NIfTI file
    downsample_to : int, optional
        Target size for downsampling (applied to both dimensions)
    
    Returns:
    --------
    b0_slice : np.ndarray
        2D B0 field map (in Hz-like units, normalized if metadata not available)
    """
    if nib is None:
        return None
    
    try:
        img = nib.load(path)
        data = img.get_fdata()
        
        # Handle 3D/4D data - select central slice
        if data.ndim == 3:
            # Find slice with largest non-zero area
            slice_areas = [np.sum(np.abs(data[:, :, z]) > 0) for z in range(data.shape[2])]
            z_slice = np.argmax(slice_areas) if max(slice_areas) > 0 else data.shape[2] // 2
            b0_slice = data[:, :, z_slice]
        elif data.ndim == 4:
            z_slice = data.shape[2] // 2
            t_slice = 0  # take first time point
            b0_slice = data[:, :, z_slice, t_slice]
        elif data.ndim == 2:
            b0_slice = data
        else:
            print(f"  WARNING: Unsupported data dimensions: {data.ndim}D")
            return None
        
        # Normalize to Hz-like units
        # If this is a flip angle map or B1 map, we'd need conversion
        # For simplicity, assume it's already in reasonable units or normalize
        # Remove DC offset and scale to reasonable range
        b0_slice = b0_slice - np.mean(b0_slice)
        b0_slice = b0_slice / np.std(b0_slice) * 50  # Scale to ~50 Hz std (arbitrary but reasonable)
        
        # Downsample if requested
        if downsample_to is not None:
            h, w = b0_slice.shape
            if max(h, w) > downsample_to:
                scale = downsample_to / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                b0_slice = resize(b0_slice, (new_h, new_w), preserve_range=True, anti_aliasing=True)
        
        return b0_slice
    except Exception as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return None


def make_synthetic_b0(N=256):
    """
    If no repo B0 available, generate a synthetic B0 field.
    
    Creates a combination of a smooth dipole-like spatial pattern and random
    low-amplitude noise field.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    
    Returns:
    --------
    b0 : np.ndarray
        2D B0 field in Hz-like units
    """
    np.random.seed(RANDOM_SEED)
    
    # Create coordinate grids
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Dipole-like pattern (smooth spatial variation)
    r = np.sqrt(X**2 + Y**2)
    dipole = (X / (r + 0.1)) * 30  # ~30 Hz amplitude
    
    # Add some smooth gradients
    gradient_x = X * 20
    gradient_y = Y * 15
    
    # Low-amplitude random noise (smoothed)
    noise = np.random.randn(N, N) * 5
    noise = ndimage.gaussian_filter(noise, sigma=5)
    
    # Combine components
    b0 = dipole + gradient_x + gradient_y + noise
    
    # Normalize to reasonable range (~50 Hz std)
    b0 = b0 - np.mean(b0)
    b0 = b0 / np.std(b0) * 50
    
    return b0


def make_test_phantom(N, downsample_to=None):
    """
    Create a visually informative test image (grid phantom or Shepp-Logan-like phantom)
    sized to match B0 dimensions.
    
    Parameters:
    -----------
    N : int or tuple
        Target size (if int, creates N x N; if tuple, uses that shape)
    downsample_to : int, optional
        If N is larger, downsample to this size
    
    Returns:
    --------
    phantom : np.ndarray
        Test image normalized to [0, 1]
    """
    if isinstance(N, tuple):
        h, w = N
    else:
        h, w = N, N
    
    # Create coordinate grids
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # Create grid pattern (checkerboard-like)
    grid_size = 8
    grid_x = np.sin(X * np.pi * grid_size)
    grid_y = np.sin(Y * np.pi * grid_size)
    grid = (grid_x * grid_y + 1) / 2  # Normalize to [0, 1]
    
    # Add circular features (Shepp-Logan-like)
    phantom = grid.copy()
    
    # Add some ellipses
    centers = [(-0.3, 0.0), (0.3, 0.0), (0.0, -0.3), (0.0, 0.3)]
    for cx, cy in centers:
        ellipse = ((X - cx) / 0.2)**2 + ((Y - cy) / 0.15)**2
        phantom[ellipse < 1] = 0.8
    
    # Add central circle
    r = np.sqrt(X**2 + Y**2)
    phantom[r < 0.15] = 0.3
    
    # Normalize to [0, 1]
    phantom = (phantom - np.min(phantom)) / (np.max(phantom) - np.min(phantom))
    
    # Downsample if needed
    if downsample_to is not None:
        if max(h, w) > downsample_to:
            scale = downsample_to / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            phantom = resize(phantom, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    
    return phantom


def compute_pixel_shift(b0, TE, BW_per_pixel):
    """
    Compute pixel shift map along phase-encode axis using approximate formula.
    
    Formula explanation:
    - Pixel shift (in pixels) is proportional to local frequency offset Δf = b0 (Hz)
    - The frequency offset accumulates during echo time TE (seconds)
    - For EPI, displacement in pixels ≈ (Δf * TE) / (BW_per_pixel)
    - Simplified: shift_pixels = b0 * TE * BW_per_pixel_factor
    
    Where:
    - b0 is in Hz (local field offset)
    - TE is in seconds (echo time)
    - BW_per_pixel is in Hz/pixel (bandwidth per pixel)
    - Result is in pixels (may be fractional)
    
    Parameters:
    -----------
    b0 : np.ndarray
        B0 field map in Hz
    TE : float
        Echo time in seconds
    BW_per_pixel : float
        Bandwidth per pixel in Hz/pixel
    
    Returns:
    --------
    shift_map : np.ndarray
        Pixel shift map (in pixels, may be fractional)
    """
    # Formula: shift = (b0 * TE) / (1 / BW_per_pixel) = b0 * TE * BW_per_pixel
    # But more accurately: shift = (b0 * TE) / (BW_per_pixel)
    # Since BW_per_pixel is already in Hz/pixel, we use:
    shift_map = (b0 * TE) / BW_per_pixel
    
    return shift_map


def warp_image_forward(img, shift_map, axis):
    """
    Apply forward warp (displace pixels along axis by shift_map) to produce distorted image.
    
    Parameters:
    -----------
    img : np.ndarray
        Original image to warp
    shift_map : np.ndarray
        Pixel shift map (in pixels, may be fractional)
    axis : int
        Axis along which to apply shift (0 = X, 1 = Y)
    
    Returns:
    --------
    warped : np.ndarray
        Warped (distorted) image
    """
    h, w = img.shape
    
    # Create coordinate grids
    if axis == 0:  # Shift along X axis
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        X, Y = np.meshgrid(x_coords, y_coords)
        # Apply shift along X
        X_new = X + shift_map
        Y_new = Y
    else:  # Shift along Y axis (default for EPI)
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        X, Y = np.meshgrid(x_coords, y_coords)
        # Apply shift along Y
        X_new = X
        Y_new = Y + shift_map
    
    # Use map_coordinates for interpolation
    coords = np.array([Y_new.ravel(), X_new.ravel()])
    warped = ndimage.map_coordinates(
        img,
        coords,
        order=1,  # linear interpolation
        mode='constant',
        cval=0.0,  # fill value
        prefilter=False
    ).reshape(img.shape)
    
    return warped


def unwarp_image_inverse(warped_img, shift_map, axis):
    """
    Implement inverse mapping to reconstruct original coordinates from shift_map
    and sample warped image back to original grid.
    
    Parameters:
    -----------
    warped_img : np.ndarray
        Warped (distorted) image
    shift_map : np.ndarray
        Pixel shift map (in pixels, may be fractional)
    axis : int
        Axis along which shift was applied (0 = X, 1 = Y)
    
    Returns:
    --------
    unwarped : np.ndarray
        Unwarped (corrected) image
    """
    h, w = warped_img.shape
    
    # Create original coordinate grid
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Inverse mapping: to get original position, subtract the shift
    if axis == 0:  # Shift was along X
        X_original = X - shift_map
        Y_original = Y
    else:  # Shift was along Y
        X_original = X
        Y_original = Y - shift_map
    
    # Use map_coordinates for interpolation
    coords = np.array([Y_original.ravel(), X_original.ravel()])
    unwarped = ndimage.map_coordinates(
        warped_img,
        coords,
        order=1,  # linear interpolation
        mode='constant',
        cval=0.0,  # fill value
        prefilter=False
    ).reshape(warped_img.shape)
    
    return unwarped


def compute_quality_metrics(original, corrected):
    """
    Compute MSE and SSIM between original phantom and corrected image.
    
    Parameters:
    -----------
    original : np.ndarray
        Original phantom image
    corrected : np.ndarray
        Corrected (unwarped) image
    
    Returns:
    --------
    mse : float
        Mean squared error
    ssim_val : float
        Structural similarity index (0-1, higher is better)
    """
    # Ensure same size
    if original.shape != corrected.shape:
        # Resize corrected to match original
        corrected = resize(corrected, original.shape, preserve_range=True, anti_aliasing=True)
    
    # MSE
    mse = np.mean((original - corrected)**2)
    
    # SSIM (requires data range)
    data_range = original.max() - original.min()
    ssim_val = ssim(original, corrected, data_range=data_range)
    
    return mse, ssim_val


def save_visuals(original, warped, unwarped, shift_map, mse, ssim_val, max_shift, outdir):
    """
    Make and save figure containing original, warped, and unwarped images with metrics.
    
    Parameters:
    -----------
    original : np.ndarray
        Original phantom
    warped : np.ndarray
        Warped (distorted) image
    unwarped : np.ndarray
        Unwarped (corrected) image
    shift_map : np.ndarray
        Pixel shift map
    mse : float
        Mean squared error
    ssim_val : float
        SSIM value
    max_shift : float
        Maximum shift in pixels
    outdir : str
        Output directory
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Main panels: original, warped, unwarped
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(original, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax1.set_title('Original Phantom', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(warped, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax2.set_title('EPI Distorted (Warped)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(unwarped, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax3.set_title('Unwarped (Corrected)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Shift map
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(shift_map, cmap='seismic', origin='lower')
    ax4.set_title('Pixel Shift Map', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Pixels')
    
    # Metrics text
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    metrics_text = f"""
    Quality Metrics:
    
    MSE: {mse:.6f}
    SSIM: {ssim_val:.4f}
    Max Shift: {max_shift:.2f} pixels
    
    Parameters:
    TE: {TE*1000:.1f} ms
    BW/pixel: {BW_PER_PIXEL:.0f} Hz
    Phase-encode axis: {'X' if PHASE_ENCODE_AXIS == 0 else 'Y'}
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Difference image (original vs corrected)
    ax6 = plt.subplot(2, 3, 6)
    diff = np.abs(original - unwarped)
    im6 = ax6.imshow(diff, cmap='hot', origin='lower')
    ax6.set_title('Absolute Difference\n(Original - Corrected)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.suptitle('EPI Distortion Simulation and Unwarp', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(outdir, "epi_warp_before_after.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epi_warp_before_after.png")
    
    # Save shift map separately
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(shift_map, cmap='seismic', origin='lower')
    ax.set_title('Pixel Shift Map (Phase-Encode Direction)', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Shift (pixels)')
    plt.tight_layout()
    
    shift_path = os.path.join(outdir, "epi_shift_map.png")
    plt.savefig(shift_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epi_shift_map.png")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("EPI Distortion Simulation and Unwarp Script")
    print("Pedagogical simulation of EPI distortion and correction")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\nOutput directory: {OUTDIR}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  DATASET_DIR: {DATASET_DIR if DATASET_DIR else 'None (will use synthetic B0)'}")
    print(f"  SUBJECT: {SUBJECT if SUBJECT else 'Auto-select first'}")
    print(f"  DOWNSAMPLE_MAX: {DOWNSAMPLE_MAX}")
    print(f"  TE: {TE*1000:.1f} ms")
    print(f"  BW_PER_PIXEL: {BW_PER_PIXEL} Hz/pixel")
    print(f"  PHASE_ENCODE_AXIS: {'X' if PHASE_ENCODE_AXIS == 0 else 'Y'}")
    print(f"  USE_REPO_B0: {USE_REPO_B0}")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # ========================================================================
    # Step 1: Load or generate B0 field map
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 1: Loading B0 field map")
    print(f"{'='*70}")
    
    b0 = None
    b0_source = None
    
    if USE_REPO_B0 and DATASET_DIR and os.path.exists(DATASET_DIR):
        b0_path = find_repo_b0(DATASET_DIR, SUBJECT)
        if b0_path and nib is not None:
            print(f"  Found repository B0 file: {os.path.basename(b0_path)}")
            b0 = load_b0_slice(b0_path, downsample_to=DOWNSAMPLE_MAX)
            if b0 is not None:
                b0_source = "repository"
                print(f"  Loaded B0 slice: {b0.shape}")
    
    if b0 is None:
        print("  Using synthetic B0 field (repository B0 not available or nibabel not installed)")
        b0 = make_synthetic_b0(DOWNSAMPLE_MAX)
        b0_source = "synthetic"
        print(f"  Generated synthetic B0: {b0.shape}")
    
    print(f"  B0 statistics: mean={np.mean(b0):.2f} Hz, std={np.std(b0):.2f} Hz")
    
    # ========================================================================
    # Step 2: Create test phantom
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 2: Creating test phantom")
    print(f"{'='*70}")
    
    phantom = make_test_phantom(b0.shape, downsample_to=None)
    print(f"  Phantom created: {phantom.shape}")
    
    # ========================================================================
    # Step 3: Compute pixel shift map
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 3: Computing pixel shift map")
    print(f"{'='*70}")
    
    shift_map = compute_pixel_shift(b0, TE, BW_PER_PIXEL)
    
    max_shift = np.max(np.abs(shift_map))
    min_shift = np.min(shift_map)
    print(f"  Shift range: [{min_shift:.2f}, {max_shift:.2f}] pixels")
    print(f"  Max shift: {max_shift:.2f} pixels")
    # Assume 1 mm voxel size for mm conversion
    print(f"  Max shift: {max_shift * 1.0:.2f} mm (assuming 1 mm voxel size)")
    
    # ========================================================================
    # Step 4: Warp phantom forward (simulate EPI distortion)
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 4: Applying forward warp (simulating EPI distortion)")
    print(f"{'='*70}")
    
    warped = warp_image_forward(phantom, shift_map, PHASE_ENCODE_AXIS)
    print(f"  Warped image created: {warped.shape}")
    
    # Save warped image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(warped, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax.set_title('EPI Distorted Image', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    warped_path = os.path.join(OUTDIR, "epi_warped.png")
    plt.savefig(warped_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epi_warped.png")
    
    # ========================================================================
    # Step 5: Unwarp (correct distortion)
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 5: Applying inverse unwarp (correcting distortion)")
    print(f"{'='*70}")
    
    unwarped = unwarp_image_inverse(warped, shift_map, PHASE_ENCODE_AXIS)
    print(f"  Unwarped image created: {unwarped.shape}")
    
    # Save unwarped image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(unwarped, cmap='gray', vmin=0, vmax=1, origin='lower')
    ax.set_title('Unwarped (Corrected) Image', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    unwarped_path = os.path.join(OUTDIR, "epi_unwarped.png")
    plt.savefig(unwarped_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epi_unwarped.png")
    
    # ========================================================================
    # Step 6: Compute quality metrics
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 6: Computing quality metrics")
    print(f"{'='*70}")
    
    mse, ssim_val = compute_quality_metrics(phantom, unwarped)
    
    print(f"  MSE: {mse:.6f}")
    print(f"  SSIM: {ssim_val:.4f} (higher is better, max=1.0)")
    
    # Save metrics CSV
    metrics_data = {
        'Metric': ['MSE', 'SSIM', 'max_shift_pixels'],
        'Value': [mse, ssim_val, max_shift]
    }
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(OUTDIR, "epi_warp_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved: epi_warp_metrics.csv")
    
    # ========================================================================
    # Step 7: Save visualization
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 7: Saving visualization")
    print(f"{'='*70}")
    
    save_visuals(phantom, warped, unwarped, shift_map, mse, ssim_val, max_shift, OUTDIR)
    
    # ========================================================================
    # Step 8: Optional B0 comparison
    # ========================================================================
    if b0_source == "repository" and USE_REPO_B0:
        print(f"\n{'='*70}")
        print("Step 8: Saving B0 comparison")
        print(f"{'='*70}")
        
        # Create synthetic B0 for comparison
        synthetic_b0 = make_synthetic_b0(b0.shape[0])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        im1 = axes[0].imshow(b0, cmap='seismic', origin='lower')
        axes[0].set_title('Repository B0 (Downsampled)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='B0 (Hz)')
        
        im2 = axes[1].imshow(synthetic_b0, cmap='seismic', origin='lower')
        axes[1].set_title('Synthetic B0 (for comparison)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='B0 (Hz)')
        
        plt.suptitle('B0 Field Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        b0_comp_path = os.path.join(OUTDIR, "epi_b0_repo_slice.png")
        plt.savefig(b0_comp_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: epi_b0_repo_slice.png")
        print(f"  Note: Repository B0 is downsampled — for visual comparison only")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    
    print(f"\nOutputs saved to: {OUTDIR}/")
    print(f" - epi_warped.png")
    print(f" - epi_unwarped.png")
    print(f" - epi_warp_before_after.png")
    print(f" - epi_shift_map.png")
    print(f" - epi_warp_metrics.csv")
    if b0_source == "repository":
        print(f" - epi_b0_repo_slice.png")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

