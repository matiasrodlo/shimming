"""
Gradient Nonlinearity Simulation Script

This script demonstrates gradient nonlinearity effects in MRI, which cause spatial
distortion and voxel size variations due to imperfect gradient field linearity.

Purpose:
- Create a coordinate grid representing the imaging field-of-view
- Simulate nonlinear gradient fields using a simple radial model
- Compute warped coordinates due to gradient nonlinearity
- Visualize the geometric distortion and local scale changes
- Optionally apply the warp to a real anatomical image for comparison

What is Simulated:
- Nonlinear gradient field: G_eff = G0 * (1 + alpha * r^2) where r is distance from center
- Coordinate warping: spatial positions are distorted by the nonlinear gradient
- Local scale changes: voxel sizes vary across the field-of-view (Jacobian determinant)
- Grid visualization: shows how a regular grid is distorted by the nonlinearity

IMPORTANT LIMITATION:
This is a pedagogical 2D demonstration of gradient nonlinearity and how it warps spatial
coordinates and voxel sizes. It is not a full coil or scanner model. The simulation uses
simplified models and does not capture all aspects of real gradient coil behavior (e.g.,
3D effects, higher-order terms, eddy currents, etc.).

Data source: ds004906 (rf-shimming-7t) — optional image source
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
except ImportError:
    print("ERROR: Missing scipy. Install with: pip install numpy scipy matplotlib scikit-image nibabel")
    sys.exit(1)

try:
    from skimage.transform import warp, resize
except ImportError:
    print("ERROR: Missing scikit-image. Install with: pip install numpy scipy matplotlib scikit-image nibabel")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("WARNING: nibabel not available. Image loading will be skipped.")
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

# Output folder
OUTDIR = os.path.join(SCRIPT_DIR, "..", "analysis_outputs")
OUTDIR = os.path.abspath(OUTDIR)

# Dataset directory - automatically detected from script location
# From analysis/04/, dataset is at ../../dataset
_relative_dataset = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
_absolute_dataset = "/Users/matiasrodlo/Documents/github/shiming/dataset"

if os.path.exists(_relative_dataset):
    DATASET_DIR = os.path.abspath(_relative_dataset)
elif os.path.exists(_absolute_dataset):
    DATASET_DIR = _absolute_dataset
else:
    DATASET_DIR = None  # Will skip image loading if not found

# If DATASET_DIR set, choose subject or auto-select first
SUBJECT = None  # if DATASET_DIR set, choose subject or auto-select first

# Grid parameters
GRID_SIZE = 200  # number of grid points per axis for visualization (max 400)
FOV_MM = 220.0  # field-of-view in mm (physical size of grid)

# Gradient parameters
G0 = 1.0  # nominal gradient scale factor (arbitrary units)
ALPHA = 1e-4  # nonlinearity coefficient (controls strength of radial distortion)

# Image application
APPLY_TO_IMAGE = True  # if True and DATASET_DIR set, apply warp to an anatomical slice for comparison
DOWNSAMPLE_MAX = 256  # max image side for applying to real images

# Reproducibility
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_coordinate_grid(grid_size, fov_mm):
    """
    Return two 2D arrays X_mm, Y_mm with coordinates in mm centered at 0.
    
    Parameters:
    -----------
    grid_size : int
        Number of grid points per axis
    fov_mm : float
        Field-of-view in millimeters
    
    Returns:
    --------
    X_mm : np.ndarray
        2D array of X coordinates in mm (extent [-FOV/2, FOV/2])
    Y_mm : np.ndarray
        2D array of Y coordinates in mm (extent [-FOV/2, FOV/2])
    """
    # Create coordinate arrays
    coords = np.linspace(-fov_mm / 2, fov_mm / 2, grid_size)
    X_mm, Y_mm = np.meshgrid(coords, coords)
    
    return X_mm, Y_mm


def nonlinear_gradient_field(X, Y, G0, alpha):
    """
    Compute an effective gradient scale factor field G_eff = G0 * (1 + alpha * r^2).
    
    Parameters:
    -----------
    X : np.ndarray
        X coordinates in mm
    Y : np.ndarray
        Y coordinates in mm
    G0 : float
        Nominal gradient scale factor (arbitrary units)
    alpha : float
        Nonlinearity coefficient
    
    Returns:
    --------
    G_eff : np.ndarray
        Effective gradient field (same units as G0, arbitrary scale)
    """
    # Compute radial distance squared (r^2 = X^2 + Y^2) in mm^2
    r_sq = X**2 + Y**2
    
    # Effective gradient: G_eff = G0 * (1 + alpha * r^2)
    # This models radial nonlinearity: gradient strength increases with distance from center
    G_eff = G0 * (1 + alpha * r_sq)
    
    return G_eff


def compute_warped_coordinates(X, Y, G_eff):
    """
    Compute warped coordinates by applying gradient nonlinearity model.
    
    The warping model: assume nominal coordinate mapping is x_phys, and nonlinear
    gradient compresses/expands coordinate by dividing by G_eff relative to G0.
    Approximation: X_warp = X / (G_eff / G0) = X / (1 + alpha*r^2)
    
    This is a simplified model where the gradient nonlinearity causes spatial
    positions to be scaled inversely with the effective gradient strength.
    
    Parameters:
    -----------
    X : np.ndarray
        Original X coordinates in mm
    Y : np.ndarray
        Original Y coordinates in mm
    G_eff : np.ndarray
        Effective gradient field
    
    Returns:
    --------
    X_warp : np.ndarray
        Warped X coordinates in mm
    Y_warp : np.ndarray
        Warped Y coordinates in mm
    """
    # Normalize by G0 to get relative gradient strength
    G_rel = G_eff / G0  # G_rel = 1 + alpha * r^2
    
    # Warped coordinates: divide by relative gradient (inverse scaling)
    # This models how nonlinear gradients compress/expand spatial coordinates
    X_warp = X / G_rel
    Y_warp = Y / G_rel
    
    return X_warp, Y_warp


def compute_local_scale_change(X, Y, Xw, Yw):
    """
    Compute local Jacobian determinant approximation using central differences.
    
    The Jacobian determinant represents the local area scaling factor.
    Percent change = (det(J) - 1) * 100, where det(J) = 1 means no distortion.
    
    Parameters:
    -----------
    X : np.ndarray
        Original X coordinates
    Y : np.ndarray
        Original Y coordinates
    Xw : np.ndarray
        Warped X coordinates
    Yw : np.ndarray
        Warped Y coordinates
    
    Returns:
    --------
    scale_map : np.ndarray
        Percent change in local area (approximate voxel-size distortion)
    """
    h, w = X.shape
    
    # Compute partial derivatives using central differences
    # dXw/dX, dXw/dY, dYw/dX, dYw/dY
    dXw_dX = np.gradient(Xw, axis=1)  # derivative of Xw with respect to X
    dXw_dY = np.gradient(Xw, axis=0)  # derivative of Xw with respect to Y
    dYw_dX = np.gradient(Yw, axis=1)  # derivative of Yw with respect to X
    dYw_dY = np.gradient(Yw, axis=0)  # derivative of Yw with respect to Y
    
    # Jacobian determinant: det(J) = dXw/dX * dYw/dY - dXw/dY * dYw/dX
    det_J = dXw_dX * dYw_dY - dXw_dY * dYw_dX
    
    # Percent change: (det(J) - 1) * 100
    scale_map = (det_J - 1.0) * 100.0
    
    return scale_map


def plot_grid_and_warp(X, Y, Xw, Yw, outpath):
    """
    Plot original grid lines and warped grid lines overlayed.
    
    Parameters:
    -----------
    X : np.ndarray
        Original X coordinates
    Y : np.ndarray
        Original Y coordinates
    Xw : np.ndarray
        Warped X coordinates
    Yw : np.ndarray
        Warped Y coordinates
    outpath : str
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot original grid (thin gray lines)
    step = max(1, X.shape[0] // 20)  # Show every Nth grid line
    for i in range(0, X.shape[0], step):
        ax.plot(X[i, :], Y[i, :], 'gray', linewidth=0.5, alpha=0.5, label='Original' if i == 0 else '')
    for j in range(0, X.shape[1], step):
        ax.plot(X[:, j], Y[:, j], 'gray', linewidth=0.5, alpha=0.5)
    
    # Plot warped grid (thicker blue lines)
    for i in range(0, Xw.shape[0], step):
        ax.plot(Xw[i, :], Yw[i, :], 'b-', linewidth=1.5, alpha=0.7, label='Warped' if i == 0 else '')
    for j in range(0, Xw.shape[1], step):
        ax.plot(Xw[:, j], Yw[:, j], 'b-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Nonlinearity: Grid Distortion', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_scale_map(scale_map, extent_mm, outpath):
    """
    Plot percent distortion map as an image with diverging colormap and colorbar.
    
    Parameters:
    -----------
    scale_map : np.ndarray
        Percent change in local area
    extent_mm : float
        Field-of-view extent in mm (for axis labels)
    outpath : str
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use diverging colormap (red-white-blue) centered at 0%
    vmin = np.min(scale_map)
    vmax = np.max(scale_map)
    vabs = max(abs(vmin), abs(vmax))
    
    im = ax.imshow(scale_map, cmap='RdBu_r', vmin=-vabs, vmax=vabs, 
                   extent=[-extent_mm/2, extent_mm/2, -extent_mm/2, extent_mm/2],
                   origin='lower')
    
    ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Local Voxel Size Distortion (% Change)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='% Change')
    
    # Add min/max labels
    ax.text(0.02, 0.98, f'Min: {vmin:.2f}%\nMax: {vmax:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


def maybe_apply_warp_to_image(image, X, Y, Xw, Yw, outpath):
    """
    If APPLY_TO_IMAGE True and image is provided, remap the image from original
    coords to warped coords and save original vs warped comparison image.
    
    Parameters:
    -----------
    image : np.ndarray
        Original image to warp
    X : np.ndarray
        Original X coordinates
    Y : np.ndarray
        Original Y coordinates
    Xw : np.ndarray
        Warped X coordinates
    Yw : np.ndarray
        Warped Y coordinates
    outpath : str
        Output file path
    """
    if image is None:
        return
    
    h, w = image.shape
    
    # Create interpolator for original image
    from scipy.interpolate import RegularGridInterpolator
    
    x_orig = np.linspace(-FOV_MM/2, FOV_MM/2, w)
    y_orig = np.linspace(-FOV_MM/2, FOV_MM/2, h)
    interp = RegularGridInterpolator((y_orig, x_orig), image, method='linear', bounds_error=False, fill_value=0)
    
    # For each point in the warped grid, find where it maps to in original space
    # The warped coordinates Xw, Yw represent where points end up
    # We need to sample the original image at positions that map to these warped positions
    # Inverse mapping: given warped position (Xw, Yw), find original position
    # Approximate: if Xw = X / (1 + alpha*r^2), then X ≈ Xw * (1 + alpha*rw^2)
    rw_sq = Xw**2 + Yw**2
    X_source = Xw * (1 + ALPHA * rw_sq)
    Y_source = Yw * (1 + ALPHA * rw_sq)
    
    # Sample original image at source coordinates
    source_points = np.column_stack([Y_source.ravel(), X_source.ravel()])
    warped_image = interp(source_points).reshape(image.shape)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = axes[0].imshow(image, cmap='gray', origin='lower')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(warped_image, cmap='gray', origin='lower')
    axes[1].set_title('Warped Image (Gradient Nonlinearity)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle('Gradient Nonlinearity: Image Distortion', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(outpath)}")


def maybe_load_repo_slice(dataset_dir, subject, target_size):
    """
    If DATASET_DIR provided, attempt to find a reasonable anatomical slice,
    load central z-slice with nibabel, downsample to target_size, and return.
    
    Parameters:
    -----------
    dataset_dir : str or None
        Path to dataset directory
    subject : str or None
        Subject ID (auto-select if None)
    target_size : int
        Target size for downsampling
    
    Returns:
    --------
    image : np.ndarray or None
        Loaded and downsampled image slice
    filename : str or None
        Filename that was loaded
    """
    if dataset_dir is None or not os.path.exists(dataset_dir):
        return None, None
    
    if nib is None:
        print("  WARNING: nibabel not available. Skipping image loading.")
        return None, None
    
    # Find subject
    if subject is None:
        subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sub-*")))
        subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
        if not subject_dirs:
            return None, None
        subject = os.path.basename(subject_dirs[0])
    
    subject_dir = os.path.join(dataset_dir, subject, "anat")
    if not os.path.exists(subject_dir):
        return None, None
    
    # Search for anatomical images (prefer T2starw or T1w)
    patterns = [
        f"{subject}_acq-*_T2starw.nii*",
        f"{subject}_acq-*_T1w.nii*",
        f"{subject}_*.nii*"
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(subject_dir, pattern))
        if matches:
            filename = matches[0]
            try:
                img = nib.load(filename)
                data = img.get_fdata()
                
                # Handle 3D/4D - take central slice
                if data.ndim == 3:
                    z_slice = data.shape[2] // 2
                    image = data[:, :, z_slice]
                elif data.ndim == 4:
                    z_slice = data.shape[2] // 2
                    t_slice = 0
                    image = data[:, :, z_slice, t_slice]
                elif data.ndim == 2:
                    image = data
                else:
                    continue
                
                # Downsample if needed
                if max(image.shape) > target_size:
                    scale = target_size / max(image.shape)
                    new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
                    from skimage.transform import resize
                    image = resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)
                
                # Normalize to [0, 1]
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                
                return image, os.path.basename(filename)
            except Exception as e:
                print(f"  WARNING: Could not load {filename}: {e}")
                continue
    
    return None, None


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Gradient Nonlinearity Simulation Script")
    print("Pedagogical 2D demonstration of gradient nonlinearity effects")
    print("=" * 70)
    
    # Safety check: limit GRID_SIZE
    if GRID_SIZE > 400:
        print(f"WARNING: GRID_SIZE={GRID_SIZE} is too large. Reducing to 400 for performance.")
        grid_size_actual = 400
    else:
        grid_size_actual = GRID_SIZE
    
    # Create output directory
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\nOutput directory: {OUTDIR}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  GRID_SIZE: {grid_size_actual}")
    print(f"  FOV_MM: {FOV_MM}")
    print(f"  G0: {G0}")
    print(f"  ALPHA: {ALPHA}")
    print(f"  DATASET_DIR: {DATASET_DIR if DATASET_DIR else 'None'}")
    print(f"  SUBJECT: {SUBJECT if SUBJECT else 'Auto-select first'}")
    print(f"  APPLY_TO_IMAGE: {APPLY_TO_IMAGE}")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # ========================================================================
    # Step 1: Build coordinate grid
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 1: Creating coordinate grid")
    print(f"{'='*70}")
    
    X, Y = create_coordinate_grid(grid_size_actual, FOV_MM)
    print(f"  Grid created: {X.shape}, extent: [{np.min(X):.1f}, {np.max(X):.1f}] mm")
    
    # ========================================================================
    # Step 2: Compute nonlinear gradient field
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 2: Computing nonlinear gradient field")
    print(f"{'='*70}")
    
    G_eff = nonlinear_gradient_field(X, Y, G0, ALPHA)
    print(f"  Gradient field range: [{np.min(G_eff):.6f}, {np.max(G_eff):.6f}]")
    print(f"  Center gradient: {G_eff[grid_size_actual//2, grid_size_actual//2]:.6f}")
    print(f"  Edge gradient: {G_eff[0, 0]:.6f}")
    
    # ========================================================================
    # Step 3: Compute warped coordinates
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 3: Computing warped coordinates")
    print(f"{'='*70}")
    
    Xw, Yw = compute_warped_coordinates(X, Y, G_eff)
    
    # Compute displacement statistics
    displacement = np.sqrt((Xw - X)**2 + (Yw - Y)**2)
    max_displacement = np.max(displacement)
    mean_displacement = np.mean(displacement)
    
    print(f"  Max displacement: {max_displacement:.4f} mm")
    print(f"  Mean displacement: {mean_displacement:.4f} mm")
    
    # ========================================================================
    # Step 4: Compute local scale change
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 4: Computing local scale change (voxel size distortion)")
    print(f"{'='*70}")
    
    scale_map = compute_local_scale_change(X, Y, Xw, Yw)
    min_scale = np.min(scale_map)
    max_scale = np.max(scale_map)
    
    print(f"  Min percent change: {min_scale:.2f}%")
    print(f"  Max percent change: {max_scale:.2f}%")
    
    # ========================================================================
    # Step 5: Save visualizations
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 5: Saving visualizations")
    print(f"{'='*70}")
    
    grid_path = os.path.join(OUTDIR, "gradient_warp_grid.png")
    plot_grid_and_warp(X, Y, Xw, Yw, grid_path)
    
    scale_path = os.path.join(OUTDIR, "gradient_scale_map.png")
    plot_scale_map(scale_map, FOV_MM, scale_path)
    
    # ========================================================================
    # Step 6: Optional image application
    # ========================================================================
    if APPLY_TO_IMAGE and DATASET_DIR:
        print(f"\n{'='*70}")
        print("Step 6: Applying warp to repository image")
        print(f"{'='*70}")
        
        image, filename = maybe_load_repo_slice(DATASET_DIR, SUBJECT, DOWNSAMPLE_MAX)
        
        if image is not None:
            print(f"  Loaded image: {filename}, shape: {image.shape}")
            
            # Resize coordinate grids to match image
            h, w = image.shape
            X_img, Y_img = create_coordinate_grid(max(h, w), FOV_MM)
            X_img = resize(X_img, (h, w), preserve_range=True, anti_aliasing=True)
            Y_img = resize(Y_img, (h, w), preserve_range=True, anti_aliasing=True)
            
            G_eff_img = nonlinear_gradient_field(X_img, Y_img, G0, ALPHA)
            Xw_img, Yw_img = compute_warped_coordinates(X_img, Y_img, G_eff_img)
            
            warp_path = os.path.join(OUTDIR, "image_original_vs_warped.png")
            maybe_apply_warp_to_image(image, X_img, Y_img, Xw_img, Yw_img, warp_path)
        else:
            print("  No image loaded. Skipping image warp application.")
    
    # ========================================================================
    # Step 7: Save statistics CSV
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 7: Saving statistics")
    print(f"{'='*70}")
    
    stats_data = {
        'Parameter': ['max_displacement_mm', 'mean_displacement_mm', 
                     'min_percent_scale_change', 'max_percent_scale_change',
                     'grid_size', 'alpha', 'G0'],
        'Value': [max_displacement, mean_displacement, min_scale, max_scale,
                 grid_size_actual, ALPHA, G0]
    }
    stats_df = pd.DataFrame(stats_data)
    csv_path = os.path.join(OUTDIR, "gradient_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"  Saved: gradient_stats.csv")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    
    print(f"\nParameters:")
    print(f"  GRID_SIZE={grid_size_actual}, FOV_MM={FOV_MM}, ALPHA={ALPHA}, G0={G0}")
    
    print(f"\nOutputs saved to: {OUTDIR}/")
    print(f" - gradient_warp_grid.png")
    print(f" - gradient_scale_map.png")
    print(f" - gradient_stats.csv")
    if APPLY_TO_IMAGE and DATASET_DIR:
        print(f" - image_original_vs_warped.png")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

