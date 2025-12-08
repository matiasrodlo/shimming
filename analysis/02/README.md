# Analysis Scripts

This directory contains analysis scripts for RF shimming and B0 field simulations.

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

## Output Directory

All analysis outputs are saved to `analysis/analysis_outputs/` by default.

