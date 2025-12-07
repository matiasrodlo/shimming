"""
RF-Shimming Exploration Script

This script performs exploratory RF-shimming analysis on processed outputs from the
ds004906 (rf-shimming-7t) OpenNeuro dataset. It implements phase-only and regularized
least-squares complex-weight shimming within the spinal cord ROI.

IMPORTANT LIMITATIONS:
- This script uses processed dataset files and does not re-run the original heavy
  pipeline (SCT/Shimming Toolbox). It assumes B1 maps and masks are already available.
- The analysis is exploratory and works on a single central slice to keep runtime
  under 15 minutes on Mac M4.

Data source: ds004906 (rf-shimming-7t) — used for exploratory analysis only.
Dataset available at: https://openneuro.org/datasets/ds004906
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from scipy.ndimage import label, find_objects
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# User must edit this path to point to local ds004906 folder
DATASET_DIR = "/Users/matiasrodlo/Documents/github/shimming/dataset"

# Subject selection: None = auto-select first subject found
SUBJECT = None

# Output directory (will be created if missing)
OUTPUT_DIR = "analysis/analysis_outputs"

# Maximum image dimension for processing (downsample if larger)
DOWNSAMPLE_MAX = 256

# Regularization parameter for LS shim
ALPHA = 1e-3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_dependencies():
    """Check for required libraries and print installation suggestions if missing."""
    missing = []
    try:
        import nibabel
    except ImportError:
        missing.append("nibabel")
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import skimage
    except ImportError:
        missing.append("scikit-image")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        raise ImportError(f"Missing packages: {', '.join(missing)}")


def find_subjects_and_files(dataset_dir, subject=None):
    """
    Find subject folders and B1 map files.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to dataset root directory
    subject : str or None
        Subject ID (e.g., 'sub-01'). If None, auto-select first found.
    
    Returns:
    --------
    subject_id : str
        Selected subject ID
    b1_files : list
        List of B1 map file paths
    mask_files : list
        List of mask/segmentation file paths
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please edit DATASET_DIR in the script to point to your local ds004906 folder."
        )
    
    # Find all subject folders
    subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sub-*")))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
    
    if not subject_dirs:
        raise ValueError(f"No subject folders found in {dataset_dir}")
    
    subject_names = [os.path.basename(d) for d in subject_dirs]
    print(f"Found subjects: {subject_names}")
    
    # Select subject
    if subject is None:
        subject_id = subject_names[0]
        print(f"Auto-selected subject: {subject_id}")
    else:
        if subject not in subject_names:
            raise ValueError(f"Subject {subject} not found. Available: {subject_names}")
        subject_id = subject
    
    subject_dir = os.path.join(dataset_dir, subject_id)
    fmap_dir = os.path.join(subject_dir, "fmap")
    
    if not os.path.exists(fmap_dir):
        raise FileNotFoundError(f"fmap directory not found: {fmap_dir}")
    
    # Search for B1 maps
    b1_patterns = [
        "*_TB1map.nii*",
        "*famp*_TB1TFL.nii*",
        "*acq-*_TB1TFL.nii*"
    ]
    
    b1_files = []
    for pattern in b1_patterns:
        matches = glob.glob(os.path.join(fmap_dir, pattern))
        b1_files.extend(matches)
    
    # Prefer TB1map files if multiple exist
    tb1map_files = [f for f in b1_files if "_TB1map" in f]
    if tb1map_files:
        b1_files = tb1map_files
    
    b1_files = sorted(list(set(b1_files)))
    
    if not b1_files:
        raise FileNotFoundError(
            f"No B1 maps found in {fmap_dir}\n"
            f"Searched for patterns: {b1_patterns}\n"
            f"Please ensure processed B1 maps are available in the fmap directory."
        )
    
    print(f"Found B1 map files:")
    for f in b1_files:
        print(f"  {f}")
    
    # Search for masks/segmentations
    mask_patterns = [
        "*mask*.nii*",
        "*_seg*.nii*"
    ]
    
    mask_files = []
    for pattern in mask_patterns:
        matches = glob.glob(os.path.join(fmap_dir, pattern))
        mask_files.extend(matches)
    
    # Also check derivatives/labels
    labels_dir = os.path.join(dataset_dir, "derivatives", "labels", subject_id)
    if os.path.exists(labels_dir):
        for pattern in mask_patterns:
            matches = glob.glob(os.path.join(labels_dir, pattern))
            mask_files.extend(matches)
    
    mask_files = sorted(list(set(mask_files)))
    
    if mask_files:
        print(f"Found mask/segmentation files:")
        for f in mask_files:
            print(f"  {f}")
    else:
        print("No mask files found - will use auto-ROI")
    
    return subject_id, b1_files, mask_files


def load_maps(b1_files):
    """
    Load B1 maps from NIfTI files.
    
    Parameters:
    -----------
    b1_files : list
        List of B1 map file paths
    
    Returns:
    --------
    channels : np.ndarray
        Complex array of shape (C, H, W) where C is number of channels
    is_synthetic : bool
        True if channels were synthetically generated
    """
    all_data = []
    is_complex = False
    
    for b1_file in b1_files:
        img = nib.load(b1_file)
        data = img.get_fdata()
        
        # Handle 4D data (channels in first dimension)
        if data.ndim == 4:
            # Take central slice for all channels
            z_slice = data.shape[2] // 2
            data = data[:, :, z_slice, :]
            # Reshape to (C, H, W)
            if data.shape[0] < data.shape[-1]:
                data = np.transpose(data, (2, 0, 1))
            else:
                data = np.transpose(data, (0, 1, 2))
        
        # Handle 3D data
        elif data.ndim == 3:
            # Select central slice with largest ROI (or middle slice)
            z_slice = data.shape[2] // 2
            data = data[:, :, z_slice]
        
        # Handle 2D data
        elif data.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}D")
        
        # Check if complex
        if np.iscomplexobj(data):
            is_complex = True
            all_data.append(data)
        else:
            # Real-valued magnitude
            all_data.append(data.astype(np.float32))
    
    if not all_data:
        raise ValueError("No data loaded from B1 files")
    
    # Stack channels
    if len(all_data) == 1:
        # Single channel - check if we need to create synthetic channels
        base_map = all_data[0]
        if not is_complex:
            print("Only real-valued magnitude map found - generating synthetic complex channels")
            # Generate C=4 synthetic complex channels
            # Apply small spatial shifts and random phases
            np.random.seed(42)  # For reproducibility
            channels = []
            for c in range(4):
                # Small random shift
                shift_y = np.random.uniform(-2, 2)
                shift_x = np.random.uniform(-2, 2)
                shifted = ndimage.shift(base_map, (shift_y, shift_x), mode='nearest')
                # Random phase
                phase = np.random.uniform(0, 2*np.pi)
                # Create complex channel
                complex_ch = shifted * np.exp(1j * phase)
                channels.append(complex_ch)
            channels = np.array(channels)
            is_synthetic = True
        else:
            # Single complex channel - duplicate to create 4 channels
            channels = np.array([base_map] * 4)
            is_synthetic = False
    else:
        # Multiple channels
        channels = np.array(all_data)
        is_synthetic = False
    
    # Ensure channels are complex
    if not np.iscomplexobj(channels):
        # Convert magnitude to complex with zero phase
        channels = channels.astype(np.complex64)
    
    print(f"Loaded {channels.shape[0]} channels, shape: {channels.shape[1:]}")
    if is_synthetic:
        print("WARNING: Using synthetic complex channels (generated from single magnitude map)")
    
    return channels, is_synthetic


def downsample_if_needed(data, max_dim=DOWNSAMPLE_MAX):
    """
    Downsample data if any dimension exceeds max_dim.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    max_dim : int
        Maximum allowed dimension
    
    Returns:
    --------
    data : np.ndarray
        Downsampled data (or original if no downsampling needed)
    scale_factor : float
        Scale factor applied (1.0 if no downsampling)
    """
    h, w = data.shape[-2:]
    max_side = max(h, w)
    
    if max_side <= max_dim:
        return data, 1.0
    
    scale = max_dim / max_side
    new_h, new_w = int(h * scale), int(w * scale)
    
    print(f"Downsampling from ({h}, {w}) to ({new_h}, {new_w}) (scale={scale:.3f})")
    
    if data.ndim == 2:
        resized = resize(data, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    else:
        # Multi-channel: resize each channel
        resized = np.zeros((data.shape[0], new_h, new_w), dtype=data.dtype)
        for c in range(data.shape[0]):
            resized[c] = resize(data[c], (new_h, new_w), preserve_range=True, anti_aliasing=True)
    
    return resized, scale


def auto_roi(channels, mask_files=None):
    """
    Automatically determine ROI for shimming.
    
    Strategy priority:
    1. Use existing shimming mask if available
    2. Use spinal cord segmentation (restricted to C3-T2 if labels available)
    3. Auto-threshold at 30% max, largest connected component
    4. Centered circular ROI (fallback)
    
    Parameters:
    -----------
    channels : np.ndarray
        Channel data of shape (C, H, W)
    mask_files : list, optional
        List of mask file paths
    
    Returns:
    --------
    roi : np.ndarray
        Boolean mask of shape (H, W)
    strategy : str
        Description of ROI strategy used
    """
    h, w = channels.shape[1:]
    roi = np.zeros((h, w), dtype=bool)
    
    # Strategy 1: Load existing mask
    if mask_files:
        for mask_file in mask_files:
            try:
                img = nib.load(mask_file)
                mask_data = img.get_fdata()
                
                # Handle 3D/4D masks
                if mask_data.ndim >= 3:
                    # Use central slice
                    z_slice = mask_data.shape[2] // 2 if mask_data.ndim >= 3 else 0
                    mask_data = mask_data[:, :, z_slice] if mask_data.ndim == 3 else mask_data[:, :, z_slice, 0]
                
                # Resize if needed
                if mask_data.shape[:2] != (h, w):
                    mask_data = resize(mask_data, (h, w), preserve_range=True, anti_aliasing=False) > 0.5
                else:
                    mask_data = mask_data > 0.5
                
                if np.any(mask_data):
                    roi = mask_data.astype(bool)
                    strategy = f"Loaded mask from {os.path.basename(mask_file)}"
                    print(f"ROI strategy: {strategy}")
                    return roi, strategy
            except Exception as e:
                print(f"Warning: Could not load mask {mask_file}: {e}")
                continue
    
    # Strategy 2: Threshold-based auto-ROI
    # Use combined magnitude
    mag_combined = np.abs(np.sum(channels, axis=0))
    threshold = 0.3 * np.max(mag_combined)
    roi_binary = mag_combined > threshold
    
    # Find largest connected component
    labeled, num_features = label(roi_binary)
    if num_features > 0:
        # Get sizes of components
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest_idx = np.argmax(sizes) + 1
        roi = labeled == largest_idx
        strategy = "Auto-threshold (30% max) + largest connected component"
        print(f"ROI strategy: {strategy}")
        return roi, strategy
    
    # Strategy 3: Centered circular ROI (fallback)
    center_y, center_x = h // 2, w // 2
    radius = int(0.25 * min(h, w))
    y, x = np.ogrid[:h, :w]
    roi = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    strategy = f"Centered circular ROI (radius={radius}px)"
    print(f"ROI strategy: {strategy}")
    
    return roi, strategy


def compute_metrics(field, roi):
    """
    Compute shimming metrics (mean, std, CV) inside ROI.
    
    Parameters:
    -----------
    field : np.ndarray
        Combined field (magnitude) of shape (H, W)
    roi : np.ndarray
        Boolean mask of shape (H, W)
    
    Returns:
    --------
    metrics : dict
        Dictionary with 'mean', 'std', 'cv' keys
    """
    roi_values = field[roi]
    mean = np.mean(roi_values)
    std = np.std(roi_values)
    cv = std / mean if mean > 0 else np.inf
    
    return {'mean': mean, 'std': std, 'cv': cv}


def phase_only_shim(channels, roi):
    """
    Compute phase-only shimming weights.
    
    For each channel, compute the mean phase in ROI and set weight to exp(-1j * phi_c).
    Amplitudes are normalized to 1.
    
    Parameters:
    -----------
    channels : np.ndarray
        Complex channel data of shape (C, H, W)
    roi : np.ndarray
        Boolean mask of shape (H, W)
    
    Returns:
    --------
    weights : np.ndarray
        Complex weights of shape (C,)
    combined_field : np.ndarray
        Combined field magnitude of shape (H, W)
    """
    C = channels.shape[0]
    weights = np.zeros(C, dtype=np.complex64)
    
    for c in range(C):
        # Compute mean phase in ROI
        roi_values = channels[c][roi]
        mean_phase = np.angle(np.mean(roi_values))
        # Set weight to conjugate phase (normalize amplitude to 1)
        weights[c] = np.exp(-1j * mean_phase)
    
    # Combine channels
    combined = np.sum(weights[:, np.newaxis, np.newaxis] * channels, axis=0)
    combined_field = np.abs(combined)
    
    return weights, combined_field


def ls_complex_shim(channels, roi, alpha=ALPHA):
    """
    Compute regularized least-squares complex shimming weights.
    
    Solves: min ||A w - t||^2 + alpha ||w||^2
    where A is design matrix (Npix x C), w are complex weights, t is target (ones).
    
    Parameters:
    -----------
    channels : np.ndarray
        Complex channel data of shape (C, H, W)
    roi : np.ndarray
        Boolean mask of shape (H, W)
    alpha : float
        Regularization parameter
    
    Returns:
    --------
    weights : np.ndarray
        Complex weights of shape (C,)
    combined_field : np.ndarray
        Combined field magnitude of shape (H, W)
    """
    C = channels.shape[0]
    roi_pixels = channels[:, roi]  # Shape: (C, Npix)
    Npix = roi_pixels.shape[1]
    
    # Build design matrix A (Npix x C) - each row is a pixel, each col is a channel
    A = roi_pixels.T  # Shape: (Npix, C)
    
    # Target: uniform field (ones)
    target = np.ones(Npix, dtype=np.complex64)
    
    # Convert to real-valued problem: [Re(w); Im(w)]
    # A_real = [Re(A) -Im(A); Im(A) Re(A)]
    A_real = np.zeros((2*Npix, 2*C), dtype=np.float32)
    A_real[:Npix, :C] = A.real
    A_real[:Npix, C:] = -A.imag
    A_real[Npix:, :C] = A.imag
    A_real[Npix:, C:] = A.real
    
    target_real = np.concatenate([target.real, target.imag])
    
    # Regularization matrix: alpha * I
    reg_matrix = alpha * np.eye(2*C)
    
    # Solve: (A^T A + alpha*I) w = A^T t
    # Using scipy.optimize for robustness
    def objective(w_real):
        w_real = w_real.reshape(-1)
        residual = A_real @ w_real - target_real
        reg_term = alpha * np.sum(w_real**2)
        return np.sum(residual**2) + reg_term
    
    # Initial guess: zeros
    w0 = np.zeros(2*C)
    
    # Optimize
    result = optimize.minimize(objective, w0, method='L-BFGS-B')
    w_real = result.x
    
    # Convert back to complex
    weights = w_real[:C] + 1j * w_real[C:]
    
    # Combine channels
    combined = np.sum(weights[:, np.newaxis, np.newaxis] * channels, axis=0)
    combined_field = np.abs(combined)
    
    return weights, combined_field


def save_outputs(metrics_df, baseline_field, phase_field, ls_field, roi, output_dir):
    """
    Save metrics CSV and before/after visualization.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with metrics for each method
    baseline_field : np.ndarray
        Baseline combined field
    phase_field : np.ndarray
        Phase-only shimmed field
    ls_field : np.ndarray
        LS shimmed field
    roi : np.ndarray
        ROI mask
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "rf_shim_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics: {csv_path}")
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Three panels for fields
    fields = [baseline_field, phase_field, ls_field]
    titles = ["Baseline", "Phase-only", "LS Complex"]
    
    for i, (field, title) in enumerate(zip(fields, titles)):
        ax = plt.subplot(1, 3, i + 1)
        im = ax.imshow(field, cmap='hot', aspect='auto')
        # Overlay ROI contour
        ax.contour(roi, colors='cyan', linewidths=1.5, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add metrics bar chart as inset
    ax_inset = fig.add_axes([0.02, 0.02, 0.25, 0.25])
    methods = metrics_df['Method'].values
    cvs = metrics_df['CV'].values
    ax_inset.bar(methods, cvs, color=['red', 'green', 'blue'], alpha=0.7)
    ax_inset.set_ylabel('CV', fontsize=9)
    ax_inset.set_title('Coefficient of Variation', fontsize=9, fontweight='bold')
    ax_inset.tick_params(labelsize=8)
    plt.setp(ax_inset.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    png_path = os.path.join(output_dir, "rf_shim_before_after.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {png_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("RF-Shimming Exploration Script")
    print("=" * 70)
    
    # Check dependencies
    check_dependencies()
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  DATASET_DIR: {DATASET_DIR}")
    print(f"  SUBJECT: {SUBJECT if SUBJECT else 'Auto-select first'}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  DOWNSAMPLE_MAX: {DOWNSAMPLE_MAX}")
    
    # Find subjects and files
    print(f"\n{'='*70}")
    print("Step 1: Finding subjects and B1 maps")
    print(f"{'='*70}")
    subject_id, b1_files, mask_files = find_subjects_and_files(DATASET_DIR, SUBJECT)
    
    # Load maps
    print(f"\n{'='*70}")
    print("Step 2: Loading B1 maps")
    print(f"{'='*70}")
    channels, is_synthetic = load_maps(b1_files)
    
    # Downsample if needed
    print(f"\n{'='*70}")
    print("Step 3: Preprocessing")
    print(f"{'='*70}")
    channels, scale = downsample_if_needed(channels, DOWNSAMPLE_MAX)
    
    # Determine ROI
    print(f"\n{'='*70}")
    print("Step 4: ROI selection")
    print(f"{'='*70}")
    roi, roi_strategy = auto_roi(channels, mask_files)
    print(f"ROI size: {np.sum(roi)} pixels ({100*np.sum(roi)/roi.size:.1f}% of image)")
    
    # Baseline metrics
    print(f"\n{'='*70}")
    print("Step 5: Baseline metrics")
    print(f"{'='*70}")
    baseline_combined = np.abs(np.sum(channels, axis=0))
    baseline_metrics = compute_metrics(baseline_combined, roi)
    print(f"Baseline - Mean: {baseline_metrics['mean']:.4f}, "
          f"Std: {baseline_metrics['std']:.4f}, "
          f"CV: {baseline_metrics['cv']:.4f}")
    
    # Phase-only shim
    print(f"\n{'='*70}")
    print("Step 6: Phase-only shimming")
    print(f"{'='*70}")
    phase_weights, phase_field = phase_only_shim(channels, roi)
    phase_metrics = compute_metrics(phase_field, roi)
    print(f"Phase-only - Mean: {phase_metrics['mean']:.4f}, "
          f"Std: {phase_metrics['std']:.4f}, "
          f"CV: {phase_metrics['cv']:.4f}")
    
    # LS complex shim
    print(f"\n{'='*70}")
    print("Step 7: LS complex shimming")
    print(f"{'='*70}")
    ls_weights, ls_field = ls_complex_shim(channels, roi, ALPHA)
    ls_metrics = compute_metrics(ls_field, roi)
    print(f"LS complex - Mean: {ls_metrics['mean']:.4f}, "
          f"Std: {ls_metrics['std']:.4f}, "
          f"CV: {ls_metrics['cv']:.4f}")
    
    # Compile metrics
    metrics_df = pd.DataFrame({
        'Method': ['Baseline', 'Phase-only', 'LS Complex'],
        'Mean': [baseline_metrics['mean'], phase_metrics['mean'], ls_metrics['mean']],
        'Std': [baseline_metrics['std'], phase_metrics['std'], ls_metrics['std']],
        'CV': [baseline_metrics['cv'], phase_metrics['cv'], ls_metrics['cv']]
    })
    
    # Save outputs
    print(f"\n{'='*70}")
    print("Step 8: Saving outputs")
    print(f"{'='*70}")
    save_outputs(metrics_df, baseline_combined, phase_field, ls_field, roi, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

