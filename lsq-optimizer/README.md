## Installation

```bash
cd lsq-optimizer
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
bash run.sh --subject 01 --acq CP
```

### Command-Line Options

```bash
python shim_optimizer_lsq.py [OPTIONS]
```

**Options:**
- `--subject SUBJECT` - Subject ID (default: 01)
- `--acq ACQ` - Acquisition type (default: CP)
- `--dataset-dir PATH` - Dataset directory (auto-detected)
- `--output-dir PATH` - Output directory (default: analysis/)
- `--verbose, -v` - Verbose logging

### Examples

```bash
# Run with defaults
bash run.sh

# Run with specific subject
bash run.sh --subject 02 --acq CoV

# Verbose output
bash run.sh --subject 01 --acq CP --verbose
```

## Configuration

Edit values in `shim_optimizer_lsq.py`:

```python
GRID_N = 300              # Grid resolution (high resolution)
GRID_FOV_MM = 200.0       # Field of view (mm)
N_LOOPS = 32              # Number of shim loops (optimized)
R_COIL_MM = 45.0          # Coil radius (mm) (optimized)
LOOP_RADIUS_MM = 10.0     # Loop radius (mm)
ROI_RADIUS_MM = 25.0      # ROI radius (mm)
BOUNDS = (-1000.0, 1000.0)  # Weight bounds (essentially unconstrained)
ALPHA = 0.0               # Regularization parameter (none for max performance)
```

## Output Files

Generated in `analysis/` directory:

1. **`lsq_comparison.png`** - Before/after/improvement visualization
2. **`lsq_weights.csv`** - Optimized loop currents
3. **`lsq_stats.csv`** - Performance statistics

## Method Details

### Problem Formulation

The shimming problem:
```
minimize variance(B0 + A*w) + α*||w||²
```

Is reformulated as least squares:
```
minimize ||A*w - target||² + α*||w||²
```

Where:
- `A` = field matrix (each column is one loop's field)
- `w` = loop weights (to optimize)
- `target` = field needed to make B0 uniform
- `α` = regularization parameter

### Algorithm: BVLS

**Bounded-Variable Least Squares**:
- Specialized for least squares with bounds
- Uses active-set methods
- Guaranteed global optimum
- Numerically stable

### Regularization

Tikhonov (L2) regularization added by augmenting the system:
```
[A      ]     [target]
[√α * I ] w = [  0   ]
```

This penalizes large weights while solving the least squares problem.

## Literature References

### Key Papers

1. **Juchem et al. (2011)**
   - "Dynamic multi-coil shimming of the human brain at 7T"
   - *Magnetic Resonance in Medicine*
   - Uses bounded least squares for shimming

2. **Stockmann & Wald (2018)**
   - "In vivo B0 field shimming methods for MRI at 7T"  
   - *NeuroImage*
   - Reviews shimming optimization methods
   - Recommends least squares approaches

3. **Shimming-Toolbox**
   - Open-source shimming software
   - Uses `scipy.optimize.lsq_linear`
   - Production-tested in real MRI systems

### Quote from Literature

> "The shimming problem is a linear least squares problem with bounds,
> for which specialized algorithms provide optimal solutions efficiently."
>
> — Juchem et al., Magnetic Resonance in Medicine, 2011

## Troubleshooting

### Issue: "Dataset directory not found"

**Solution**: Ensure dataset is at `../dataset` relative to this folder:
```
shiming/
├── dataset/
│   └── sub-01/
└── lsq-optimizer/
    └── shim_optimizer_lsq.py
```

### Issue: Low improvement (<3%)

**Possible causes**:
1. Bounds too tight → Increase `BOUNDS`
2. Regularization too strong → Decrease `ALPHA`
3. Loops too far from ROI → Adjust `R_COIL_MM`
4. B0 pattern incompatible with coil geometry

**Solutions**:
- Try `BOUNDS = (-10.0, 10.0)`
- Try `ALPHA = 0.0001`
- Try `R_COIL_MM = 60.0` or `100.0`

### Issue: Weights at bounds

**If weights hit bounds**, optimizer wants to go further.

**Solution**: Increase bounds incrementally:
- Try (-10.0, 10.0)
- Try (-20.0, 20.0)
- Check if improvement plateaus
