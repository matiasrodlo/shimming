# Analysis Scripts

This directory contains analysis scripts for RF shimming, B0 field simulations, and EPI distortion correction.

## Scripts

### 01_rf_shimming_exploration.py
Analysis of already-shimmed B1 maps from the ds004906 dataset. Compares different RF shimming methods (CP, CoV, patient, phase, volume, target, SAReff) by extracting metrics from flip angle maps and converting them to B1+ efficiency.

See `01/README.md` for detailed instructions.

### 02_b0_dipole_simulation.py
Simulate susceptibility-induced B0 distortions using a 2D dipole (FFT-based) approximation. Demonstrates a simple low-order correction (plane or quadratic removal) that mimics low-order shimming. Optionally compares a downsampled repository B0 slice if COMPARE_DATASET_DIR is set to a local ds004906 path.

**Note:** This script is a pedagogical 2D toy model for susceptibility-induced B0 and low-order correction; it is not a full Maxwell/Biot–Savart simulation.

#### Run instructions:
- Edit parameters at top of `analysis/02/02_b0_dipole_simulation.py` if desired.
- Optionally set `COMPARE_DATASET_DIR` to your local ds004906 path to compare a downsampled repo B0 slice.
- Run: `python analysis/02/02_b0_dipole_simulation.py`
- Outputs saved to `analysis/analysis_outputs/`

### 03_epi_distortion_unwarp.py
Simulate EPI (Echo Planar Imaging) distortion caused by B0 field inhomogeneities and demonstrate unwarping correction using B0 field maps. The script creates a test phantom, simulates EPI distortion by warping along the phase-encode axis, and applies unwarping correction to restore the original image.

**Note:** This script simulates simple EPI distortion and unwarp using a B0 estimate. It is pedagogical and not a replacement for scanner reconstruction tools.

#### Run instructions:
- Edit `DATASET_DIR` and `SUBJECT` in `analysis/03/03_epi_distortion_unwarp.py` to point to your local ds004906 folder or choose a subject.
- Run: `python analysis/03/03_epi_distortion_unwarp.py`
- Outputs saved to `analysis/analysis_outputs/`

## Output Directory

All analysis outputs are saved to `analysis/analysis_outputs/` by default.

