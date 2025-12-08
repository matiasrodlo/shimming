"""
RF-Shimming Analysis Script

This script analyzes already-shimmed B1 maps from the ds004906 (rf-shimming-7t) OpenNeuro dataset.
It compares different RF shimming methods (CP, CoV, patient, phase, volume, target, SAReff) by
extracting metrics from the flip angle maps and converting them to B1+ efficiency.

The script follows the methodology from the reference implementation but simplified for
single-subject, single-slice analysis.

Data source: ds004906 (rf-shimming-7t)
Dataset available at: https://openneuro.org/datasets/ds004906
Reference: https://github.com/shimming-toolbox/rf-shimming-7t
"""

import os
import glob
import json
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory - relative to script location
# From analysis/01/, dataset is at ../../dataset
# Falls back to absolute path if relative doesn't exist
_relative_dataset = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
_absolute_dataset = "/Users/matiasrodlo/Documents/github/shiming/dataset"

if os.path.exists(_relative_dataset):
    DATASET_DIR = os.path.abspath(_relative_dataset)
elif os.path.exists(_absolute_dataset):
    DATASET_DIR = _absolute_dataset
else:
    # Default: user must edit this path
    DATASET_DIR = "/path/to/dataset"

# Subject selection: None = auto-select first subject found
SUBJECT = None

# Output directory (will be created if missing)
# Relative to script location
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_outputs")

# Maximum image dimension for processing (downsample if larger)
DOWNSAMPLE_MAX = 256

# Physical constants for B1+ efficiency conversion
GAMMA = 2.675e8  # [rad / (s T)] - gyromagnetic ratio
REQUESTED_FA = 90  # saturation flip angle (degrees) - hard-coded in sequence

# Available shimming methods in the dataset
SHIMMING_METHODS = ["CP", "CoV", "patient", "phase", "volume", "target", "SAReff"]

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
    Find subject folders and available shimming method files.
    
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
    shim_files : dict
        Dictionary mapping shim method names to file paths
    mask_files : list
        List of mask/segmentation file paths
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please edit DATASET_DIR in the script to point to your local dataset folder."
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
    
    # Search for flip angle maps (famp* files) for each shimming method
    shim_files = {}
    for shim_method in SHIMMING_METHODS:
        # Pattern: sub-XX_acq-famp{method}_TB1TFL.nii.gz
        pattern = os.path.join(fmap_dir, f"{subject_id}_acq-famp{shim_method}_TB1TFL.nii.gz")
        if os.path.exists(pattern):
            shim_files[shim_method] = pattern
        else:
            print(f"Warning: {shim_method} shimming method not found for {subject_id}")
    
    if not shim_files:
        raise FileNotFoundError(
            f"No shimming method files found in {fmap_dir}\n"
            f"Expected pattern: {subject_id}_acq-famp*_TB1TFL.nii.gz"
        )
    
    print(f"\nFound {len(shim_files)} shimming methods:")
    for method, filepath in shim_files.items():
        print(f"  {method}: {os.path.basename(filepath)}")
    
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
            matches = glob.glob(os.path.join(labels_dir, "**", pattern), recursive=True)
            mask_files.extend(matches)
    
    mask_files = sorted(list(set(mask_files)))
    
    if mask_files:
        print(f"\nFound {len(mask_files)} mask/segmentation files")
    else:
        print("\nNo mask files found - will use auto-ROI")
    
    return subject_id, shim_files, mask_files


def load_flip_angle_map(famp_file):
    """
    Load flip angle map from NIfTI file.
    
    Parameters:
    -----------
    famp_file : str
        Path to flip angle map NIfTI file
    
    Returns:
    --------
    fa_map : np.ndarray
        Flip angle map (2D or 3D)
    """
    img = nib.load(famp_file)
    fa_map = img.get_fdata()
    
    # Handle 3D data - select central slice
    if fa_map.ndim == 3:
        z_slice = fa_map.shape[2] // 2
        fa_map = fa_map[:, :, z_slice]
    elif fa_map.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported data dimensions: {fa_map.ndim}D")
    
    return fa_map


def convert_to_b1plus_efficiency(famp_file, fa_map):
    """
    Convert flip angle map to B1+ efficiency in nT/V units.
    
    Based on reference implementation (rf-shimming-7t notebook, Cell 40).
    The approach calculates B1+ efficiency using a 1ms, pi-pulse at the acquisition
    voltage, then scales by the ratio of measured to requested flip angle.
    
    Parameters:
    -----------
    famp_file : str
        Path to flip angle map file (used to get JSON metadata)
    fa_map : np.ndarray
        Flip angle map in degrees (Siemens format: actual value * 10)
    
    Returns:
    --------
    b1_map : np.ndarray
        B1+ efficiency map in nT/V
    """
    # Load JSON metadata to get reference voltage
    json_file = famp_file.replace('.nii.gz', '.json').replace('.nii', '.json')
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON sidecar not found: {json_file}")
    
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    ref_voltage = metadata.get("TxRefAmp", None)
    if ref_voltage is None:
        # Try alternative field name
        ref_voltage = metadata.get("TraRefAmpl", None)
        if ref_voltage is None:
            raise ValueError(f"Could not find TxRefAmp or TraRefAmpl in {json_file}")
    
    print(f"  Reference voltage: {ref_voltage} V")
    
    # Siemens maps are in units of flip angle * 10 (in degrees)
    acquired_fa = fa_map / 10.0  # Convert to actual degrees
    
    # Account for power loss between coil and socket (given by Siemens)
    voltage_at_socket = ref_voltage * (10 ** -0.095)
    
    # Compute B1 map in [T/V]
    # Formula: B1 = (acquired_fa / requested_fa) * (pi / (gamma * pulse_duration * voltage))
    # pulse_duration = 1e-3 s (1 ms)
    b1_map = (acquired_fa / REQUESTED_FA) * (np.pi / (GAMMA * 1e-3 * voltage_at_socket))
    
    # Convert to [nT/V]
    b1_map = b1_map * 1e9
    
    return b1_map


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
    
    print(f"  Downsampling from ({h}, {w}) to ({new_h}, {new_w}) (scale={scale:.3f})")
    
    resized = resize(data, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    
    return resized, scale


def auto_roi(b1_map, mask_files=None):
    """
    Automatically determine ROI for analysis.
    
    Strategy priority:
    1. Use existing segmentation mask if available
    2. Auto-threshold at 30% max, largest connected component
    3. Centered circular ROI (fallback)
    
    Parameters:
    -----------
    b1_map : np.ndarray
        B1+ efficiency map of shape (H, W)
    mask_files : list, optional
        List of mask file paths
    
    Returns:
    --------
    roi : np.ndarray
        Boolean mask of shape (H, W)
    strategy : str
        Description of ROI strategy used
    """
    h, w = b1_map.shape
    roi = np.zeros((h, w), dtype=bool)
    
    # Strategy 1: Load existing mask
    if mask_files:
        for mask_file in mask_files:
            try:
                img = nib.load(mask_file)
                mask_data = img.get_fdata()
                
                # Handle 3D/4D masks
                if mask_data.ndim >= 3:
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
    threshold = 0.3 * np.max(b1_map)
    roi_binary = b1_map > threshold
    
    # Find largest connected component
    labeled, num_features = label(roi_binary)
    if num_features > 0:
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


def compute_metrics(b1_map, roi):
    """
    Compute shimming metrics (mean, std, CV) inside ROI.
    
    Parameters:
    -----------
    b1_map : np.ndarray
        B1+ efficiency map of shape (H, W) in nT/V
    roi : np.ndarray
        Boolean mask of shape (H, W)
    
    Returns:
    --------
    metrics : dict
        Dictionary with 'mean', 'std', 'cv' keys
    """
    roi_values = b1_map[roi]
    mean = np.mean(roi_values)
    std = np.std(roi_values)
    cv = (std / mean) * 100 if mean > 0 else np.inf  # CV as percentage
    
    return {'mean': mean, 'std': std, 'cv': cv}


def save_outputs(metrics_df, b1_maps, roi, output_dir, subject_id):
    """
    Save metrics CSV and visualization comparing shimming methods.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with metrics for each shimming method
    b1_maps : dict
        Dictionary mapping shim method names to B1+ efficiency maps
    roi : np.ndarray
        ROI mask
    output_dir : str
        Output directory path
    subject_id : str
        Subject ID
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "rf_shim_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics: {csv_path}")
    
    # Create figure comparing all shimming methods
    n_methods = len(b1_maps)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 4 * n_rows))
    
    methods = sorted(b1_maps.keys())
    
    for i, method in enumerate(methods):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        b1_map = b1_maps[method]
        
        # Get metrics for this method
        method_metrics = metrics_df[metrics_df['Method'] == method].iloc[0]
        mean_val = method_metrics['Mean']
        cv_val = method_metrics['CV']
        
        # Display B1+ map
        im = ax.imshow(b1_map, cmap='viridis', aspect='auto', vmin=5, vmax=30)
        ax.contour(roi, colors='cyan', linewidths=1.5, alpha=0.7)
        ax.set_title(f"{method}\n{mean_val:.2f} nT/V, CV: {cv_val:.2f}%", 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='B1+ [nT/V]')
    
    plt.suptitle(f'B1+ Efficiency Maps - {subject_id}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    png_path = os.path.join(output_dir, "rf_shim_comparison.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {png_path}")
    plt.close()
    
    # Create bar chart comparing CV across methods
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = metrics_df['Method'].values
    cvs = metrics_df['CV'].values
    means = metrics_df['Mean'].values
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, cvs, alpha=0.7, color='steelblue')
    ax.set_xlabel('Shimming Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'B1+ Homogeneity Comparison - {subject_id}', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, cv, mean) in enumerate(zip(bars, cvs, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv:.2f}%\n({mean:.2f} nT/V)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "rf_shim_cv_comparison.png")
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    print(f"Saved bar chart: {bar_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("RF-Shimming Analysis Script")
    print("Analyzing already-shimmed B1 maps from ds004906 dataset")
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
    print("Step 1: Finding subjects and shimming method files")
    print(f"{'='*70}")
    subject_id, shim_files, mask_files = find_subjects_and_files(DATASET_DIR, SUBJECT)
    
    # Process each shimming method
    print(f"\n{'='*70}")
    print("Step 2: Loading and converting flip angle maps to B1+ efficiency")
    print(f"{'='*70}")
    
    b1_maps = {}
    all_metrics = []
    
    for method, famp_file in shim_files.items():
        print(f"\nProcessing {method} shimming method...")
        print(f"  File: {os.path.basename(famp_file)}")
        
        # Load flip angle map
        fa_map = load_flip_angle_map(famp_file)
        
        # Convert to B1+ efficiency
        b1_map = convert_to_b1plus_efficiency(famp_file, fa_map)
        
        # Downsample if needed
        b1_map, scale = downsample_if_needed(b1_map, DOWNSAMPLE_MAX)
        
        b1_maps[method] = b1_map
        print(f"  B1+ map shape: {b1_map.shape}, range: [{np.min(b1_map):.2f}, {np.max(b1_map):.2f}] nT/V")
    
    # Determine ROI (use first B1 map as reference)
    print(f"\n{'='*70}")
    print("Step 3: ROI selection")
    print(f"{'='*70}")
    first_method = list(b1_maps.keys())[0]
    roi, roi_strategy = auto_roi(b1_maps[first_method], mask_files)
    print(f"ROI size: {np.sum(roi)} pixels ({100*np.sum(roi)/roi.size:.1f}% of image)")
    
    # Compute metrics for each method
    print(f"\n{'='*70}")
    print("Step 4: Computing metrics for each shimming method")
    print(f"{'='*70}")
    
    for method, b1_map in b1_maps.items():
        metrics = compute_metrics(b1_map, roi)
        all_metrics.append({
            'Method': method,
            'Mean': metrics['mean'],
            'Std': metrics['std'],
            'CV': metrics['cv']
        })
        print(f"{method:10s} - Mean: {metrics['mean']:7.2f} nT/V, "
              f"Std: {metrics['std']:6.2f} nT/V, CV: {metrics['cv']:6.2f}%")
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Sort by CV (lower is better for homogeneity)
    metrics_df = metrics_df.sort_values('CV')
    print(f"\n{'='*70}")
    print("Results sorted by CV (Coefficient of Variation - lower is better):")
    print(f"{'='*70}")
    print(metrics_df.to_string(index=False))
    
    # Save outputs
    print(f"\n{'='*70}")
    print("Step 5: Saving outputs")
    print(f"{'='*70}")
    save_outputs(metrics_df, b1_maps, roi, OUTPUT_DIR, subject_id)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")
    print(f"\nBest shimming method (lowest CV): {metrics_df.iloc[0]['Method']}")
    print(f"  Mean B1+: {metrics_df.iloc[0]['Mean']:.2f} nT/V")
    print(f"  CV: {metrics_df.iloc[0]['CV']:.2f}%")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
