# Analysis Scripts

This directory contains analysis scripts for RF shimming, B0 field simulations, EPI distortion correction, and gradient nonlinearity.

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

### 04_gradient_nonlinearity.py
Demonstrate gradient nonlinearity effects in MRI, which cause spatial distortion and voxel size variations due to imperfect gradient field linearity. The script creates a coordinate grid, simulates nonlinear gradient fields, computes warped coordinates, and visualizes geometric distortion and local scale changes.

**Note:** This script is a pedagogical 2D demonstration of gradient nonlinearity and its geometric consequences; it is not a scanner coil model.

#### Run instructions:
- Edit parameters at top of `analysis/04/04_gradient_nonlinearity.py` if desired.
- Optionally set `DATASET_DIR` and `SUBJECT` and set `APPLY_TO_IMAGE=True` to compare with a downsampled repo slice.
- Run: `python analysis/04/04_gradient_nonlinearity.py`
- Outputs saved to `analysis/analysis_outputs/`

### 05_bloch_sinc_pulse.py
Designs a sinc RF pulse (time-domain), shows its frequency profile, and simulates magnetization dynamics using a small Bloch integrator for a 1D slice. The script demonstrates RF pulse design, frequency-domain analysis, and slice-selective excitation through Bloch equation simulation.

**Note:** This script is a pedagogical demonstration of RF pulse design and a 1D Bloch integrator. It is not a clinical-grade simulator.

#### Run instructions:
- Edit parameters at top of `analysis/05/05_bloch_sinc_pulse.py` if desired.
- Run: `python analysis/05/05_bloch_sinc_pulse.py`
- Outputs saved to `analysis/analysis_outputs/`

### 06_shim_coil_biot_savart.py
A 2D toy shim-coil optimizer that uses a simplified Biot-Savart numeric approximation for circular loops and optimizes scalar loop currents. The script places circular shim loops around an imaging ROI and optimizes their currents to minimize field variance within the ROI using Tikhonov regularization.

**Note:** This script is a pedagogical 2D shim-coil optimizer. It omits full 3D coil physics, coil coupling, mutual inductance, and realistic conductor geometry. It is intended for conceptual exploration only.

#### Run instructions:
- Edit parameters at top of `analysis/06/06_shim_coil_biot_savart.py` if desired.
- Run: `python analysis/06/06_shim_coil_biot_savart.py`
- Outputs saved to `analysis/analysis_outputs/`

## Output Directory

All analysis outputs are saved to `analysis/analysis_outputs/` by default.

