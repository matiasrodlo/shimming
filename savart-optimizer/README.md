# Savart Optimizer - 2D Toy Shim-Coil Optimizer

A 2D toy shim-coil optimizer that uses a simplified Biot-Savart numeric approximation for circular loops and optimizes scalar loop currents. This is a pedagogical implementation that omits coil coupling, full 3D effects, and realistic conductor geometry.

## Features

- **Biot-Savart Field Computation**: Numerically computes Bz field from circular loops
- **Tikhonov Regularization**: Optimizes loop currents to minimize ROI variance
- **BIDS-Compatible**: Supports loading data from BIDS-formatted datasets
- **Flexible Configuration**: Command-line arguments and configurable parameters
- **Comprehensive Logging**: Detailed logging with configurable verbosity

## Installation

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `numpy>=1.20.0` - Numerical computations
- `scipy>=1.7.0` - Optimization and scientific computing
- `matplotlib>=3.4.0` - Plotting and visualization

### Optional Dependencies

- `nibabel>=3.2.0` - Neuroimaging file I/O (required for B0 comparison)
- `scikit-image>=0.18.0` - Image processing (improved downsampling)
- `pybids>=0.15.0` - BIDS dataset handling (recommended for proper dataset access)

## Usage

### Basic Usage

Run with default settings:

```bash
python 06_shim_coil_biot_savart.py
```

### Command-Line Options

```bash
python 06_shim_coil_biot_savart.py [OPTIONS]
```

**Options:**

- `--dataset-dir PATH` - Path to BIDS dataset directory (overrides auto-detection)
- `--subject SUBJECT` - Subject ID for B0 comparison (e.g., "01", "02", default: "01")
- `--acq ACQ` - Acquisition type for B0 comparison (e.g., "CP", "CoV", "patient")
- `--output-dir PATH` - Output directory (overrides default OUTDIR)
- `--no-repo-b0` - Disable repository B0 comparison
- `--verbose, -v` - Enable verbose logging

### Examples

**Run with specific subject and acquisition:**

```bash
python 06_shim_coil_biot_savart.py --subject 01 --acq CP
```

**Specify custom dataset directory:**

```bash
python 06_shim_coil_biot_savart.py --dataset-dir /path/to/dataset
```

**Disable B0 comparison:**

```bash
python 06_shim_coil_biot_savart.py --no-repo-b0
```

**Verbose logging:**

```bash
python 06_shim_coil_biot_savart.py --verbose
```

## Dataset Requirements

The script expects a BIDS-compliant dataset with the following structure:

```
dataset/
├── sub-01/
│   ├── anat/
│   │   └── *.nii.gz, *.json
│   └── fmap/
│       └── *_TB1TFL.nii.gz, *.json
├── sub-02/
│   └── ...
└── ...
```

### Dataset Directory Detection

The script automatically detects the dataset directory using the following strategies (in order):

1. Relative path: `../../dataset` (relative to script location)
2. Environment variable: `BIDS_DATASET_DIR`
3. Current directory: `./dataset`
4. Parent directory: `../dataset`
5. Command-line argument: `--dataset-dir`

## Configuration

Configuration parameters are defined at the top of the script:

- `GRID_N` - Grid resolution (default: 200, max: 300)
- `GRID_FOV_MM` - Field-of-view in mm (default: 200.0)
- `N_LOOPS` - Number of shim loops (default: 8)
- `R_COIL_MM` - Radius of coil ring in mm (default: 80.0)
- `LOOP_RADIUS_MM` - Physical radius of each loop in mm (default: 10.0)
- `ROI_RADIUS_MM` - Radius of central ROI in mm (default: 25.0)
- `ALPHA` - Tikhonov regularization strength (default: 1e-3)
- `MAXITER` - Maximum optimization iterations (default: 500)

## Output Files

The script generates the following output files in the output directory:

- `biot_savart_baseline.png` - Baseline field map
- `biot_savart_optimized.png` - Optimized field map
- `biot_savart_before_after.png` - Before/after comparison
- `biot_savart_weights.csv` - Optimized loop weights
- `biot_savart_stats.csv` - Optimization statistics
- `biot_savart_repo_comparison.csv` - B0 comparison results (if enabled)

## Best Practices

### BIDS Compliance

The script now uses BIDS-aware data loading when `pybids` is available:

- Automatically loads JSON metadata sidecars
- Uses proper BIDS entity matching (subject, acquisition, etc.)
- Extracts spatial parameters from metadata

### Logging

All output uses Python's `logging` framework:

- `INFO` level: Standard progress messages
- `DEBUG` level: Detailed debugging information (use `--verbose`)
- `WARNING` level: Non-critical issues
- `ERROR` level: Critical errors

### Error Handling

The script includes comprehensive error handling:

- Configuration validation before execution
- Specific exception types for different error conditions
- Graceful degradation when optional dependencies are missing

## Improvements Made

This version includes the following improvements over the original:

1. ✅ **Removed hardcoded paths** - Uses environment variables and relative paths
2. ✅ **BIDS-compliant data loading** - Uses `pybids` when available
3. ✅ **JSON metadata support** - Loads and uses BIDS metadata
4. ✅ **Command-line interface** - Flexible configuration via arguments
5. ✅ **Logging framework** - Replaces `print()` statements
6. ✅ **Input validation** - Validates configuration before execution
7. ✅ **Better error handling** - Specific exceptions with clear messages
8. ✅ **Requirements file** - Documented dependencies

## Limitations

- **2D Toy Model**: This is a simplified 2D approximation, not a full 3D model
- **No Coil Coupling**: Coil interactions are not modeled
- **Pedagogical Purpose**: Designed for learning, not production use
- **B0 Comparison**: The comparison with repository B0 data is illustrative only

## Citation

If you use this code, please cite the original dataset:

```
RF shimming in the cervical spinal cord at 7T
Dataset DOI: 10.18112/openneuro.ds004906.v2.4.0
```

## License

See the main project LICENSE file.

## Troubleshooting

### Dataset Not Found

If the dataset is not automatically detected:

1. Set the `BIDS_DATASET_DIR` environment variable
2. Use the `--dataset-dir` command-line argument
3. Ensure the dataset follows BIDS structure

### Missing Dependencies

Install missing dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies can be skipped if not needed (script will degrade gracefully).

### Import Errors

If you see import warnings, they are likely for optional dependencies. The script will work without them, but some features may be disabled.

## Contact

For issues or questions, please refer to the main project repository.

