"""
B0 Dipole Simulation Script

Simulate susceptibility-induced B0 distortions using a 2D dipole (FFT-based) approximation.
Demonstrate a simple low-order correction (plane or quadratic removal) that mimics low-order shimming.
Optionally compare a downsampled repository B0 slice if COMPARE_DATASET_DIR is set to a local ds004906 path.

IMPORTANT: This is a pedagogical 2D toy model — not a full Maxwell solver — and is intended 
for exploratory use only. The 2D dipole approximation is a simplified approach that does not 
capture full 3D field behavior or complex boundary conditions.

Data source: ds004906 (rf-shimming-7t) — optional comparison
Dataset available at: https://openneuro.org/datasets/ds004906
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import checks with helpful error messages
try:
    from scipy import ndimage
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
except ImportError:
    print("ERROR: Missing scipy. Install with: pip install numpy scipy matplotlib scikit-image nibabel")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: Missing pandas. Install with: pip install numpy scipy matplotlib scikit-image nibabel pandas")
    sys.exit(1)

try:
    from skimage.transform import resize
except ImportError:
    print("WARNING: scikit-image not available. Dataset comparison will be skipped.")
    resize = None

try:
    import nibabel as nib
except ImportError:
    print("WARNING: nibabel not available. Dataset comparison will be skipped.")
    nib = None

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory - relative to script location
# From analysis/02/, analysis_outputs is at ../analysis_outputs
OUTDIR = os.path.join(SCRIPT_DIR, "..", "analysis_outputs")
OUTDIR = os.path.abspath(OUTDIR)

N = 256  # grid resolution (max); if user sets >512 the script reduces to 256
VOXEL_MM = 1.0  # voxel size in millimeters
DELTA_CHI_TISSUE = 0.0  # susceptibility difference for tissue (relative to air)
DELTA_CHI_INCL = 1e-5  # susceptibility difference for inclusion
INCL_RADIUS_MM = 6.0  # radius of circular inclusion in millimeters
INCL_POS = (0.2, 0.0)  # normalized coords in [-0.5..0.5] space for (x_frac, y_frac)
CORRECTION_ORDER = "plane"  # "plane" or "quadratic"
COMPARE_DATASET_DIR = None  # set to local ds004906 path to compare to a repo B0 slice (optional)
RANDOM_SEED = 42  # for reproducibility
DO_SENSITIVITY = False  # set to True to run sensitivity sweep

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_phantom(N, voxel_mm, incl_radius_mm, incl_pos_frac, delta_chi_incl, delta_chi_tissue):
    """
    Builds a 2D elliptical brain-like mask and places a circular susceptibility inclusion.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    voxel_mm : float
        Voxel size in millimeters
    incl_radius_mm : float
        Radius of inclusion in millimeters
    incl_pos_frac : tuple
        Normalized position (x_frac, y_frac) in [-0.5, 0.5] space
    delta_chi_incl : float
        Susceptibility difference for inclusion
    delta_chi_tissue : float
        Susceptibility difference for tissue
    
    Returns:
    --------
    chi_map : np.ndarray
        Susceptibility distribution (float)
    mask : np.ndarray
        Boolean mask of brain-like region
    """
    # Create coordinate grids
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    # Create elliptical brain-like mask (approximate head shape)
    # Ellipse with semi-axes a=0.4, b=0.35 (slightly flattened)
    a, b = 0.4, 0.35
    ellipse = (X / a)**2 + (Y / b)**2
    mask = ellipse <= 1.0
    
    # Initialize susceptibility map
    chi_map = np.zeros((N, N), dtype=np.float64)
    chi_map[mask] = delta_chi_tissue
    
    # Add circular inclusion
    incl_radius_frac = incl_radius_mm / (voxel_mm * N)  # convert mm to normalized units
    x_center = incl_pos_frac[0]
    y_center = incl_pos_frac[1]
    
    # Distance from inclusion center
    r_incl = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    incl_mask = (r_incl <= incl_radius_frac) & mask  # only inside brain mask
    
    chi_map[incl_mask] = delta_chi_incl
    
    return chi_map, mask


def dipole_kernel_2d(N, voxel_mm):
    """
    Builds a 2D k-space dipole kernel for the susceptibility→B0 relationship (2D approximation).
    
    The dipole kernel in k-space is: (1/3 - kz^2/k^2) for 3D, simplified to 2D approximation.
    In 2D, we use a simplified form that captures the main dipole behavior.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    voxel_mm : float
        Voxel size in millimeters
    
    Returns:
    --------
    kernel : np.ndarray
        Complex k-space dipole kernel (in FFT order, not shifted)
    """
    # Create k-space coordinates (in FFT order: 0 to positive, then negative)
    kx = np.fft.fftfreq(N, d=voxel_mm * 1e-3)  # convert mm to meters
    ky = np.fft.fftfreq(N, d=voxel_mm * 1e-3)
    KX, KY = np.meshgrid(kx, ky)
    
    # k-space magnitude squared
    k_sq = KX**2 + KY**2
    
    # 2D dipole kernel approximation
    # In 3D: kernel = 1/3 - kz^2/k^2
    # In 2D (kz=0): we approximate as 1/3 - kx^2/k^2 (angular dependence)
    # Handle k=0 singularity
    k_sq_safe = k_sq.copy()
    k_sq_safe[0, 0] = 1.0  # avoid division by zero
    
    # 2D dipole kernel (simplified from 3D: 1/3 - kz^2/k^2)
    # In 2D, we approximate as: 1/3 - kx^2/k^2
    kernel = (1.0/3.0) - (KX**2 / k_sq_safe)
    
    # Set k=0 to zero (no DC component in field perturbation)
    kernel[0, 0] = 0.0
    
    # Convert to complex (for consistency with FFT operations)
    kernel = kernel.astype(np.complex128)
    
    return kernel


def compute_b0_from_chi(chi_map, kernel):
    """
    Compute FFT of chi_map, multiply by kernel, inverse FFT to get B0 map.
    
    The relationship is: B0 = FFT^-1(FFT(chi) * dipole_kernel)
    This is a convolution in real space, multiplication in k-space.
    
    Parameters:
    -----------
    chi_map : np.ndarray
        Susceptibility distribution
    kernel : np.ndarray
        K-space dipole kernel
    
    Returns:
    --------
    b0_map : np.ndarray
        Real-valued B0 field map (arbitrary units, scaled to nT-like range)
    """
    # FFT of susceptibility
    chi_fft = fft2(chi_map)
    
    # Multiply by dipole kernel
    b0_fft = chi_fft * kernel
    
    # Inverse FFT to get B0 map
    b0_map = np.real(ifft2(b0_fft))
    
    # Scale to interpretable range (nT-like units)
    # The scaling factor depends on field strength and units
    # For 7T, typical B0 perturbations are in the 0-100 nT range
    # We scale to a reasonable range for visualization
    b0_map = b0_map * 1e6  # scale to nT-like units (arbitrary scaling for visualization)
    
    return b0_map


def fit_low_order_surface(b0, mask, order="plane"):
    """
    Fit either a plane (a + b*x + c*y) or quadratic surface to b0[mask].
    
    This mimics low-order shimming correction (e.g., linear/quadratic shim coils).
    
    Parameters:
    -----------
    b0 : np.ndarray
        B0 field map
    mask : np.ndarray
        Boolean mask of region to fit
    order : str
        "plane" for linear fit, "quadratic" for quadratic fit
    
    Returns:
    --------
    fitted_surface : np.ndarray
        2D array of fitted surface values
    coefficients : np.ndarray
        Fitted coefficients
    """
    N = b0.shape[0]
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    
    # Extract coordinates and values within mask
    coords = np.column_stack([X[mask], Y[mask]])
    values = b0[mask]
    
    if order == "plane":
        # Fit plane: z = a + b*x + c*y
        A = np.column_stack([np.ones(len(values)), coords[:, 0], coords[:, 1]])
    elif order == "quadratic":
        # Fit quadratic: z = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
        A = np.column_stack([
            np.ones(len(values)),
            coords[:, 0],  # x
            coords[:, 1],  # y
            coords[:, 0]**2,  # x^2
            coords[:, 1]**2,  # y^2
            coords[:, 0] * coords[:, 1]  # x*y
        ])
    else:
        raise ValueError(f"Unknown order: {order}. Use 'plane' or 'quadratic'")
    
    # Least squares fit
    coefficients, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    
    # Evaluate fitted surface on full grid
    if order == "plane":
        fitted_surface = (coefficients[0] + 
                         coefficients[1] * X + 
                         coefficients[2] * Y)
    else:  # quadratic
        fitted_surface = (coefficients[0] + 
                         coefficients[1] * X + 
                         coefficients[2] * Y +
                         coefficients[3] * X**2 +
                         coefficients[4] * Y**2 +
                         coefficients[5] * X * Y)
    
    return fitted_surface, coefficients


def apply_correction(b0, fitted_surface):
    """
    Subtract fitted surface from b0 and return corrected map.
    
    Parameters:
    -----------
    b0 : np.ndarray
        Original B0 field map
    fitted_surface : np.ndarray
        Fitted low-order surface to subtract
    
    Returns:
    --------
    corrected : np.ndarray
        Corrected B0 map (b0 - fitted_surface)
    """
    return b0 - fitted_surface


def save_figure(img, fname, vmin=None, vmax=None, cmap='seismic', title=None):
    """
    Save a PNG to OUTDIR with colorbar and title.
    
    Parameters:
    -----------
    img : np.ndarray
        Image to save
    fname : str
        Filename (will be saved to OUTDIR)
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    cmap : str
        Colormap name
    title : str, optional
        Title for the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title if title else fname, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    full_path = os.path.join(OUTDIR, fname)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def maybe_load_repo_b0_slice(compare_dir, N):
    """
    If COMPARE_DATASET_DIR is not None, search for common B0/fmap patterns under the dataset,
    load central slice with nibabel, downsample to N x N using skimage.transform.resize if needed.
    
    Parameters:
    -----------
    compare_dir : str or None
        Path to dataset directory
    N : int
        Target grid size for downsampling
    
    Returns:
    --------
    repo_slice : np.ndarray or None
        Downsampled B0 slice from repository
    path_str : str or None
        Path to the file that was loaded
    """
    if compare_dir is None or not os.path.exists(compare_dir):
        return None, None
    
    if nib is None:
        print("  WARNING: nibabel not available. Skipping dataset comparison.")
        return None, None
    
    if resize is None:
        print("  WARNING: scikit-image not available. Skipping dataset comparison.")
        return None, None
    
    # Search for common B0/fmap patterns
    patterns = [
        "*_TB1map.nii*",
        "*famp*_TB1TFL.nii*",
        "*fmap*.nii*",
        "*B0*.nii*"
    ]
    
    found_files = []
    for pattern in patterns:
        matches = list(Path(compare_dir).rglob(pattern))
        found_files.extend(matches)
    
    if not found_files:
        print(f"  WARNING: No B0/fmap files found in {compare_dir}")
        return None, None
    
    # Use first found file
    repo_file = str(found_files[0])
    print(f"  Found repository B0 file: {os.path.basename(repo_file)}")
    
    try:
        img = nib.load(repo_file)
        data = img.get_fdata()
        
        # Handle 3D/4D data - take central slice
        if data.ndim == 3:
            z_slice = data.shape[2] // 2
            repo_slice = data[:, :, z_slice]
        elif data.ndim == 4:
            z_slice = data.shape[2] // 2
            t_slice = 0  # take first time point
            repo_slice = data[:, :, z_slice, t_slice]
        elif data.ndim == 2:
            repo_slice = data
        else:
            print(f"  WARNING: Unsupported data dimensions: {data.ndim}D")
            return None, None
        
        # Downsample to N x N if needed
        if repo_slice.shape[0] != N or repo_slice.shape[1] != N:
            repo_slice = resize(repo_slice, (N, N), preserve_range=True, anti_aliasing=True)
            print(f"  Downsampled to {N}x{N}")
        
        return repo_slice, repo_file
    except Exception as e:
        print(f"  WARNING: Could not load {repo_file}: {e}")
        return None, None


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("B0 Dipole Simulation Script")
    print("Pedagogical 2D toy model for susceptibility-induced B0 distortions")
    print("=" * 70)
    
    # Safety check: limit N for performance
    if N > 512:
        print(f"WARNING: N={N} is too large. Reducing to 256 for speed.")
        N_actual = 256
    else:
        N_actual = N
    
    # Create output directory
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\nOutput directory: {OUTDIR}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  N = {N_actual}")
    print(f"  VOXEL_MM = {VOXEL_MM}")
    print(f"  INCL_RADIUS_MM = {INCL_RADIUS_MM}")
    print(f"  DELTA_CHI_INCL = {DELTA_CHI_INCL}")
    print(f"  INCL_POS = {INCL_POS}")
    print(f"  CORRECTION_ORDER = {CORRECTION_ORDER}")
    print(f"  COMPARE_DATASET_DIR = {COMPARE_DATASET_DIR if COMPARE_DATASET_DIR else 'None'}")
    print(f"  DO_SENSITIVITY = {DO_SENSITIVITY}")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # ========================================================================
    # Step 1: Create phantom
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 1: Creating phantom")
    print(f"{'='*70}")
    
    chi_map, mask = make_phantom(
        N_actual, VOXEL_MM, INCL_RADIUS_MM, INCL_POS,
        DELTA_CHI_INCL, DELTA_CHI_TISSUE
    )
    
    save_figure(chi_map, "b0_phantom.png", cmap='gray', title="Susceptibility Phantom")
    print(f"  Phantom created: {np.sum(mask)} pixels in mask")
    
    # ========================================================================
    # Step 2: Build dipole kernel and compute B0
    # ========================================================================
    print(f"\n{'='*70}")
    print("Step 2: Computing B0 field from susceptibility")
    print(f"{'='*70}")
    
    kernel = dipole_kernel_2d(N_actual, VOXEL_MM)
    b0_map = compute_b0_from_chi(chi_map, kernel)
    
    # Print statistics
    b0_masked = b0_map[mask]
    print(f"  B0 statistics (within mask):")
    print(f"    Mean: {np.mean(b0_masked):.4f} nT")
    print(f"    Std:  {np.std(b0_masked):.4f} nT")
    print(f"    Min:  {np.min(b0_masked):.4f} nT")
    print(f"    Max:  {np.max(b0_masked):.4f} nT")
    
    save_figure(b0_map, "b0_map.png", cmap='seismic', title="B0 Field Map (nT)")
    
    # ========================================================================
    # Step 3: Low-order correction
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Step 3: Applying {CORRECTION_ORDER} correction")
    print(f"{'='*70}")
    
    # Fit low-order surface
    fitted_surface, coeffs = fit_low_order_surface(b0_map, mask, order=CORRECTION_ORDER)
    print(f"  Fitted {CORRECTION_ORDER} coefficients: {coeffs}")
    
    # Apply correction
    b0_corrected = apply_correction(b0_map, fitted_surface)
    
    # Compute metrics
    b0_before = b0_map[mask]
    b0_after = b0_corrected[mask]
    
    mean_before = np.mean(b0_before)
    std_before = np.std(b0_before)
    cv_before = (std_before / mean_before) * 100 if mean_before != 0 else np.inf
    
    mean_after = np.mean(b0_after)
    std_after = np.std(b0_after)
    cv_after = (std_after / mean_after) * 100 if mean_after != 0 else np.inf
    
    print(f"\n  Metrics (within mask):")
    print(f"    Before correction:")
    print(f"      Mean: {mean_before:.4f} nT")
    print(f"      Std:  {std_before:.4f} nT")
    print(f"      CV:   {cv_before:.4f}%")
    print(f"    After {CORRECTION_ORDER} correction:")
    print(f"      Mean: {mean_after:.4f} nT")
    print(f"      Std:  {std_after:.4f} nT")
    print(f"      CV:   {cv_after:.4f}%")
    print(f"    Improvement: {((std_before - std_after) / std_before * 100):.2f}% reduction in std")
    
    # Save metrics CSV
    metrics_data = {
        'metric': ['mean', 'std', 'cv'],
        'before': [mean_before, std_before, cv_before],
        'after': [mean_after, std_after, cv_after]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(OUTDIR, "b0_dipole_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n  Saved metrics: b0_dipole_metrics.csv")
    
    # Save before/after comparison figure
    vmin = min(np.min(b0_map[mask]), np.min(b0_corrected[mask]))
    vmax = max(np.max(b0_map[mask]), np.max(b0_corrected[mask]))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = axes[0].imshow(b0_map, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title("Before Correction", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='B0 (nT)')
    
    im2 = axes[1].imshow(b0_corrected, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f"After {CORRECTION_ORDER.capitalize()} Correction", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='B0 (nT)')
    
    plt.suptitle('B0 Correction Demo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(OUTDIR, "b0_correction_demo.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: b0_correction_demo.png")
    
    # ========================================================================
    # Step 4: Optional comparison with repository data
    # ========================================================================
    if COMPARE_DATASET_DIR:
        print(f"\n{'='*70}")
        print("Step 4: Comparing with repository B0 data")
        print(f"{'='*70}")
        
        repo_slice, repo_path = maybe_load_repo_b0_slice(COMPARE_DATASET_DIR, N_actual)
        
        if repo_slice is not None:
            # Normalize both for comparison (use same scale)
            b0_norm = (b0_map - np.mean(b0_map[mask])) / np.std(b0_map[mask])
            repo_norm = (repo_slice - np.mean(repo_slice)) / np.std(repo_slice)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            im1 = axes[0].imshow(b0_map, cmap='seismic', origin='lower')
            axes[0].set_title("Synthetic B0 (Simulated)", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='B0 (nT)')
            
            im2 = axes[1].imshow(repo_slice, cmap='seismic', origin='lower')
            axes[1].set_title("Repository B0 (Downsampled)", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.suptitle('B0 Comparison: Synthetic vs Repository', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            fig_path = os.path.join(OUTDIR, "b0_comparison_repo.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: b0_comparison_repo.png")
            print(f"  Note: Repository image is downsampled to {N_actual}x{N_actual}")
    
    # ========================================================================
    # Step 5: Optional sensitivity sweep
    # ========================================================================
    if DO_SENSITIVITY:
        print(f"\n{'='*70}")
        print("Step 5: Sensitivity sweep")
        print(f"{'='*70}")
        
        delta_chi_values = [1e-6, 1e-5, 5e-5]
        std_improvements = []
        
        for dchi in delta_chi_values:
            # Create phantom with this susceptibility
            chi_temp, _ = make_phantom(
                N_actual, VOXEL_MM, INCL_RADIUS_MM, INCL_POS,
                dchi, DELTA_CHI_TISSUE
            )
            
            # Compute B0
            b0_temp = compute_b0_from_chi(chi_temp, kernel)
            
            # Apply correction
            fitted_temp, _ = fit_low_order_surface(b0_temp, mask, order=CORRECTION_ORDER)
            b0_corr_temp = apply_correction(b0_temp, fitted_temp)
            
            # Compute improvement
            std_before = np.std(b0_temp[mask])
            std_after = np.std(b0_corr_temp[mask])
            improvement = ((std_before - std_after) / std_before) * 100
            
            std_improvements.append(improvement)
            print(f"  DELTA_CHI_INCL = {dchi:.1e}: {improvement:.2f}% std reduction")
        
        # Plot sensitivity
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(delta_chi_values, std_improvements, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('DELTA_CHI_INCL', fontsize=12, fontweight='bold')
        ax.set_ylabel('Std Reduction (%)', fontsize=12, fontweight='bold')
        ax.set_title('Sensitivity to Susceptibility Difference', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        fig_path = os.path.join(OUTDIR, "b0_sensitivity.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: b0_sensitivity.png")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    
    print(f"\nParameters:")
    print(f"  N={N_actual}, VOXEL_MM={VOXEL_MM}, INCL_RADIUS_MM={INCL_RADIUS_MM}, DELTA_CHI_INCL={DELTA_CHI_INCL}")
    
    print(f"\nOutputs saved to: {OUTDIR}/")
    print(f" - b0_phantom.png")
    print(f" - b0_map.png")
    print(f" - b0_correction_demo.png")
    print(f" - b0_dipole_metrics.csv")
    if COMPARE_DATASET_DIR and repo_slice is not None:
        print(f" - b0_comparison_repo.png")
    if DO_SENSITIVITY:
        print(f" - b0_sensitivity.png")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

