"""
Bloch Simulation of Sinc RF Pulse

This script designs a sinc RF pulse (time-domain), shows its frequency profile,
and simulates magnetization dynamics using a small Bloch integrator for a 1D slice.
It is a pedagogical demo — not a full MRI simulator.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Import dependencies with error handling
try:
    import matplotlib.pyplot as plt
    from scipy import integrate
    from scipy.fft import fft, fftfreq, fftshift
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Suggested: pip install numpy scipy matplotlib")
    sys.exit(1)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTDIR = "analysis/analysis_outputs"
RF_DURATION_MS = 2.0  # total pulse duration in ms (typical small value)
DT_US = 5.0  # time step in microseconds (e.g., 5 µs -> dt = 5e-6 s)
SINC_LOBES = 4  # number of sinc lobes (on each side, typical 4)
WINDOW = "hamming"  # window type for truncation: "hamming", "hann", or None
GRAD_SLICE_M_T_PER_M = 0.01  # gradient amplitude for slice selection (arbitrary units for demo)
SLICE_FOV_MM = 40.0  # simulated slice FOV in mm (across which profile is computed)
NX = 201  # number of spatial points across slice (odd preferred)
T1_MS = 1000.0  # tissue T1 in ms (for relaxation in Bloch sim)
T2_MS = 100.0  # tissue T2 in ms
OFF_RES_HZ = 0.0  # off-resonance frequency for isochromat (Hz)
PLOT_DPI = 200
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def design_sinc_pulse(duration_ms, dt_us, lobes, window):
    """
    Design a sinc RF pulse with windowing.
    
    Creates a complex-valued time-domain RF pulse sampled at dt = dt_us * 1e-6 seconds.
    Builds a sinc baseline centered in time and applies the chosen window.
    Normalizes pulse area for later scaling.
    
    Parameters
    ----------
    duration_ms : float
        Total pulse duration in milliseconds
    dt_us : float
        Time step in microseconds
    lobes : int
        Number of sinc lobes on each side
    window : str or None
        Window type: "hamming", "hann", or None
    
    Returns
    -------
    t_axis_s : ndarray
        Time axis in seconds
    rf_envelope : ndarray, complex
        RF pulse envelope (arbitrary B1 amplitude units)
    """
    # Convert to seconds
    duration_s = duration_ms * 1e-3
    dt_s = dt_us * 1e-6
    
    # Create time axis centered at zero
    n_samples = int(np.round(duration_s / dt_s))
    if n_samples % 2 == 0:
        n_samples += 1  # Prefer odd number for symmetry
    t_axis_s = np.linspace(-duration_s/2, duration_s/2, n_samples)
    
    # Design sinc pulse
    # Sinc with specified number of lobes: sinc(2*pi*BW*t)
    # Bandwidth is related to number of lobes
    # For n lobes, we want sinc to have n zero crossings on each side
    # Approximate: BW ~ n / duration
    bw_approx = lobes / duration_s  # Approximate bandwidth in Hz
    sinc_arg = 2 * np.pi * bw_approx * t_axis_s
    
    # Avoid division by zero at center
    sinc_arg_safe = np.where(np.abs(sinc_arg) < 1e-10, 1e-10, sinc_arg)
    sinc_base = np.sin(sinc_arg_safe) / sinc_arg_safe
    sinc_base[np.abs(sinc_arg) < 1e-10] = 1.0  # Center point
    
    # Apply window
    if window == "hamming":
        window_func = np.hamming(len(t_axis_s))
    elif window == "hann":
        window_func = np.hanning(len(t_axis_s))
    elif window is None:
        window_func = np.ones(len(t_axis_s))
    else:
        print(f"Warning: Unknown window '{window}', using no window")
        window_func = np.ones(len(t_axis_s))
    
    rf_envelope = sinc_base * window_func
    
    # Normalize so that integrated area can be scaled later
    # Area normalization: make integral = 1 (arbitrary units)
    area = np.sum(np.abs(rf_envelope)) * dt_s
    if area > 1e-10:
        rf_envelope = rf_envelope / area
    
    # Make complex (real part is the envelope, imaginary part is zero for now)
    rf_envelope = rf_envelope.astype(complex)
    
    return t_axis_s, rf_envelope


def pulse_spectrum(rf_envelope, dt):
    """
    Compute the frequency spectrum of the RF pulse using FFT.
    
    Parameters
    ----------
    rf_envelope : ndarray, complex
        RF pulse envelope
    dt : float
        Time step in seconds
    
    Returns
    -------
    freqs_hz : ndarray
        Frequency axis in Hz
    spectrum : ndarray
        Magnitude spectrum
    """
    # FFT
    spectrum_fft = fft(rf_envelope)
    freqs_fft = fftfreq(len(rf_envelope), dt)
    
    # Shift to center zero frequency
    spectrum = fftshift(spectrum_fft)
    freqs_hz = fftshift(freqs_fft)
    
    # Magnitude
    spectrum_mag = np.abs(spectrum)
    
    return freqs_hz, spectrum_mag


def bloch_integrator_1d(rf_envelope, t_axis_s, positions_mm, grad_amp, dt, T1, T2, 
                        off_res_hz=0.0, method='RK4'):
    """
    Integrate Bloch equations for 1D slice with RF pulse and gradient.
    
    For each isochromat at positions_mm, computes magnetization evolution
    under the RF pulse and slice-selection gradient.
    
    Parameters
    ----------
    rf_envelope : ndarray, complex
        RF pulse envelope (arbitrary B1 units)
    t_axis_s : ndarray
        Time axis in seconds
    positions_mm : ndarray
        Spatial positions in mm
    grad_amp : float
        Gradient amplitude (arbitrary units, converted to Hz/mm)
    dt : float
        Time step in seconds
    T1 : float
        T1 relaxation time in seconds
    T2 : float
        T2 relaxation time in seconds
    off_res_hz : float
        Off-resonance frequency in Hz
    method : str
        Integration method: 'RK4' or 'Euler'
    
    Returns
    -------
    Mx : ndarray
        Final Mx for each position
    My : ndarray
        Final My for each position
    Mz : ndarray
        Final Mz for each position
    M_central_time : dict, optional
        Time evolution for central isochromat (if requested)
    """
    n_positions = len(positions_mm)
    n_time = len(t_axis_s)
    
    # Initialize magnetization: start at equilibrium (Mx=0, My=0, Mz=1)
    Mx = np.zeros(n_positions)
    My = np.zeros(n_positions)
    Mz = np.ones(n_positions)
    
    # Physical constants (illustrative units)
    # gamma = 2 * pi * 42.577e6 rad/s/T (proton gyromagnetic ratio)
    gamma_rad_per_s_per_T = 2 * np.pi * 42.577e6
    
    # Convert gradient to frequency offset per position
    # For simplicity, assume grad_amp is in arbitrary units
    # Convert to Hz/mm: omega_z = gamma * G * z
    # For demo, use simplified scaling
    grad_hz_per_mm = grad_amp * gamma_rad_per_s_per_T / (2 * np.pi) * 1e-3  # Simplified scaling
    
    # RF envelope is assumed to be already scaled appropriately
    # (scaling is done in main script before calling this function)
    
    # Store time evolution for central isochromat
    central_idx = n_positions // 2
    M_central_time = {
        't': [],
        'Mx': [],
        'My': [],
        'Mz': []
    }
    
    # Bloch equations: dM/dt = M × omega - relaxation
    # omega = [omega_x, omega_y, omega_z]
    # omega_x = gamma * B1x(t) = gamma * Re(B1(t))
    # omega_y = gamma * B1y(t) = gamma * Im(B1(t))
    # omega_z = gamma * G * z + off_resonance
    
    # Integrate for each position
    # Vectorized approach: integrate all positions simultaneously
    M_all = np.zeros((n_positions, 3))
    M_all[:, 2] = 1.0  # Start with Mz=1
    
    for i, pos_mm in enumerate(positions_mm):
        M = np.array([0.0, 0.0, 1.0])  # Initial magnetization
        
        for t_idx in range(n_time):
            t = t_axis_s[t_idx]
            
            # Get RF at current time
            B1_complex = rf_envelope[t_idx]  # Already scaled
            omega_x = gamma_rad_per_s_per_T * np.real(B1_complex)
            omega_y = gamma_rad_per_s_per_T * np.imag(B1_complex)
            
            # Position-dependent frequency offset
            omega_z = 2 * np.pi * (grad_hz_per_mm * pos_mm + off_res_hz)
            
            # Bloch equations
            if method == 'RK4':
                # Runge-Kutta 4th order
                k1 = dt * np.array([
                    M[1] * omega_z - M[2] * omega_y - M[0] / T2,
                    M[2] * omega_x - M[0] * omega_z - M[1] / T2,
                    M[0] * omega_y - M[1] * omega_x - (M[2] - 1.0) / T1
                ])
                
                M_temp = M + k1/2
                k2 = dt * np.array([
                    M_temp[1] * omega_z - M_temp[2] * omega_y - M_temp[0] / T2,
                    M_temp[2] * omega_x - M_temp[0] * omega_z - M_temp[1] / T2,
                    M_temp[0] * omega_y - M_temp[1] * omega_x - (M_temp[2] - 1.0) / T1
                ])
                
                M_temp = M + k2/2
                k3 = dt * np.array([
                    M_temp[1] * omega_z - M_temp[2] * omega_y - M_temp[0] / T2,
                    M_temp[2] * omega_x - M_temp[0] * omega_z - M_temp[1] / T2,
                    M_temp[0] * omega_y - M_temp[1] * omega_x - (M_temp[2] - 1.0) / T1
                ])
                
                M_temp = M + k3
                k4 = dt * np.array([
                    M_temp[1] * omega_z - M_temp[2] * omega_y - M_temp[0] / T2,
                    M_temp[2] * omega_x - M_temp[0] * omega_z - M_temp[1] / T2,
                    M_temp[0] * omega_y - M_temp[1] * omega_x - (M_temp[2] - 1.0) / T1
                ])
                
                M = M + (k1 + 2*k2 + 2*k3 + k4) / 6
            else:
                # Euler method
                dM_dt = np.array([
                    M[1] * omega_z - M[2] * omega_y - M[0] / T2,
                    M[2] * omega_x - M[0] * omega_z - M[1] / T2,
                    M[0] * omega_y - M[1] * omega_x - (M[2] - 1.0) / T1
                ])
                M = M + dt * dM_dt
            
            # Store central isochromat time evolution
            if i == central_idx:
                M_central_time['t'].append(t * 1e3)  # Convert to ms
                M_central_time['Mx'].append(M[0])
                M_central_time['My'].append(M[1])
                M_central_time['Mz'].append(M[2])
        
        M_all[i] = M
    
    Mx = M_all[:, 0]
    My = M_all[:, 1]
    Mz = M_all[:, 2]
    
    return Mx, My, Mz, M_central_time


def compute_slice_profile(Mz, positions_mm):
    """
    Compute excitation profile from Mz and estimate FWHM.
    
    Parameters
    ----------
    Mz : ndarray
        Final Mz magnetization across positions
    positions_mm : ndarray
        Spatial positions in mm
    
    Returns
    -------
    profile : ndarray
        Excitation profile (1 - Mz, since we start from Mz=1)
    fwhm_mm : float
        Full width at half maximum in mm (estimated)
    """
    # Excitation profile: flip angle is related to reduction from equilibrium
    # For 90° flip, Mz goes from 1 to 0, so profile = 1 - Mz
    profile = 1.0 - Mz
    
    # Estimate FWHM
    max_profile = np.max(profile)
    half_max = max_profile / 2.0
    
    # Find positions where profile crosses half-maximum
    above_half = profile >= half_max
    if np.sum(above_half) == 0:
        fwhm_mm = 0.0
    else:
        # Find indices where profile crosses half-maximum
        indices = np.where(above_half)[0]
        if len(indices) > 0:
            # Use actual positions for accurate FWHM
            pos_left = positions_mm[indices[0]]
            pos_right = positions_mm[indices[-1]]
            fwhm_mm = pos_right - pos_left
        else:
            fwhm_mm = 0.0
    
    return profile, fwhm_mm


def plot_pulse_and_profile(t_axis_s, rf_envelope, freqs_hz, spectrum, positions_mm, 
                          Mx, My, Mz, profile, fwhm_mm, M_central_time, outpath):
    """
    Create multi-panel figure showing pulse design and slice profile.
    
    Parameters
    ----------
    t_axis_s : ndarray
        Time axis in seconds
    rf_envelope : ndarray, complex
        RF pulse envelope
    freqs_hz : ndarray
        Frequency axis in Hz
    spectrum : ndarray
        Magnitude spectrum
    positions_mm : ndarray
        Spatial positions in mm
    Mx, My, Mz : ndarray
        Final magnetization components
    profile : ndarray
        Excitation profile
    fwhm_mm : float
        FWHM in mm
    M_central_time : dict
        Time evolution for central isochromat
    outpath : str
        Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: RF envelope vs time
    ax = axes[0, 0]
    t_ms = t_axis_s * 1e3
    ax.plot(t_ms, np.abs(rf_envelope), 'b-', linewidth=2, label='Magnitude')
    ax.plot(t_ms, np.real(rf_envelope), 'r--', linewidth=1, alpha=0.7, label='Real')
    ax.plot(t_ms, np.imag(rf_envelope), 'g--', linewidth=1, alpha=0.7, label='Imag')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('RF Amplitude (arb. units)')
    ax.set_title('RF Pulse Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: RF spectrum
    ax = axes[0, 1]
    # Plot only central portion
    freq_lim = 5000  # Hz
    mask = np.abs(freqs_hz) <= freq_lim
    ax.plot(freqs_hz[mask], spectrum[mask], 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('RF Pulse Spectrum')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Slice profile
    ax = axes[1, 0]
    Mxy_mag = np.sqrt(Mx**2 + My**2)
    ax.plot(positions_mm, profile, 'b-', linewidth=2, label='Excitation (1-Mz)')
    ax.plot(positions_mm, Mxy_mag, 'r--', linewidth=2, label='Mxy magnitude')
    
    # Mark FWHM
    max_profile = np.max(profile)
    half_max = max_profile / 2.0
    ax.axhline(y=half_max, color='gray', linestyle=':', linewidth=1, label='Half-max')
    
    # Find FWHM positions (simplified)
    above_half = profile >= half_max
    if np.sum(above_half) > 0:
        indices = np.where(above_half)[0]
        if len(indices) > 0:
            pos_left = positions_mm[indices[0]]
            pos_right = positions_mm[indices[-1]]
            ax.axvline(x=pos_left, color='gray', linestyle=':', linewidth=1)
            ax.axvline(x=pos_right, color='gray', linestyle=':', linewidth=1)
            ax.text(0.5, 0.95, f'FWHM = {fwhm_mm:.2f} mm', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Magnetization')
    ax.set_title('Slice Excitation Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Time evolution of central isochromat
    ax = axes[1, 1]
    if len(M_central_time['t']) > 0:
        t_plot = np.array(M_central_time['t'])
        ax.plot(t_plot, M_central_time['Mx'], 'r-', linewidth=1.5, label='Mx')
        ax.plot(t_plot, M_central_time['My'], 'g-', linewidth=1.5, label='My')
        ax.plot(t_plot, M_central_time['Mz'], 'b-', linewidth=1.5, label='Mz')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Magnetization')
        ax.set_title('Central Isochromat Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No time evolution data', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Central Isochromat Evolution')
    
    plt.suptitle('Bloch Simulation: Sinc RF Pulse and Slice Profile', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(outpath, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {outpath}")


def save_metrics_csv(metrics_dict, fname):
    """
    Save metrics to CSV file.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric names and values
    fname : str
        Output filename
    """
    with open(fname, 'w') as f:
        f.write('metric,value\n')
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f'{key},{value:.6f}\n')
            else:
                f.write(f'{key},{value}\n')
    print(f"Saved metrics CSV: {fname}")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Main script execution."""
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directory (OUTDIR is relative to project root)
    project_root = os.path.join(SCRIPT_DIR, "..", "..")
    outdir_full = os.path.abspath(os.path.join(project_root, OUTDIR))
    os.makedirs(outdir_full, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("Bloch Simulation: Sinc RF Pulse")
    print("="*60)
    print(f"OUTDIR: {OUTDIR}")
    print(f"RF_DURATION_MS: {RF_DURATION_MS}")
    print(f"DT_US: {DT_US}")
    print(f"SINC_LOBES: {SINC_LOBES}")
    print(f"WINDOW: {WINDOW}")
    print(f"GRAD_SLICE_M_T_PER_M: {GRAD_SLICE_M_T_PER_M}")
    print(f"SLICE_FOV_MM: {SLICE_FOV_MM}")
    print(f"NX: {NX}")
    print(f"T1_MS: {T1_MS}")
    print(f"T2_MS: {T2_MS}")
    print(f"OFF_RES_HZ: {OFF_RES_HZ}")
    print("="*60 + "\n")
    
    # Performance checks
    dt_s = DT_US * 1e-6
    n_samples_estimate = int(np.round(RF_DURATION_MS * 1e-3 / dt_s))
    if n_samples_estimate > 20000:
        print(f"Warning: Estimated samples ({n_samples_estimate}) > 20000")
        print("Consider increasing DT_US to reduce computation time")
    
    if NX > 501:
        print(f"Warning: NX ({NX}) > 501, reducing to 501")
        nx_use = 501
    else:
        nx_use = NX
    
    # Design RF pulse
    print("Designing sinc RF pulse...")
    t_axis_s, rf_envelope = design_sinc_pulse(RF_DURATION_MS, DT_US, SINC_LOBES, WINDOW)
    print(f"  Pulse duration: {RF_DURATION_MS} ms")
    print(f"  Number of samples: {len(t_axis_s)}")
    print(f"  Time step: {dt_s*1e6:.2f} µs")
    
    # Scale RF for ~90° flip angle (illustrative)
    # For 90° flip: integral(gamma * B1) = pi/2
    # We'll scale the pulse so that on-resonance flip is approximately pi/2
    gamma_rad_per_s_per_T = 2 * np.pi * 42.577e6
    pulse_area = np.sum(np.abs(rf_envelope)) * dt_s
    # Scale factor to achieve pi/2 flip (illustrative)
    B1_scale = np.pi / (2 * gamma_rad_per_s_per_T * pulse_area + 1e-10)
    rf_envelope_scaled = rf_envelope * B1_scale
    print(f"  Scaled for ~90° flip angle (illustrative)")
    print(f"  B1 scale factor: {B1_scale:.6e}")
    
    # Compute pulse spectrum
    print("\nComputing pulse spectrum...")
    freqs_hz, spectrum = pulse_spectrum(rf_envelope_scaled, dt_s)
    
    # Estimate spectral bandwidth (-3 dB width)
    max_spectrum = np.max(spectrum)
    half_max_db = max_spectrum / np.sqrt(2)  # -3 dB
    above_half = spectrum >= half_max_db
    if np.sum(above_half) > 0:
        freq_indices = np.where(above_half)[0]
        bw_3db_hz = freqs_hz[freq_indices[-1]] - freqs_hz[freq_indices[0]]
        print(f"  Estimated -3 dB bandwidth: {bw_3db_hz:.1f} Hz")
    else:
        bw_3db_hz = 0.0
        print("  Could not estimate bandwidth")
    
    # Prepare spatial positions
    print(f"\nPreparing spatial grid...")
    positions_mm = np.linspace(-SLICE_FOV_MM/2, SLICE_FOV_MM/2, nx_use)
    spatial_step_mm = SLICE_FOV_MM / (nx_use - 1)
    print(f"  NX={nx_use} positions across {SLICE_FOV_MM} mm")
    print(f"  Spatial step: {spatial_step_mm:.3f} mm")
    
    # Convert T1/T2 to seconds
    T1_s = T1_MS * 1e-3
    T2_s = T2_MS * 1e-3
    
    # Bloch simulation
    print("\nRunning Bloch simulation...")
    print(f"  T1={T1_MS} ms, T2={T2_MS} ms")
    print(f"  Gradient: {GRAD_SLICE_M_T_PER_M} (arbitrary units)")
    print(f"  Off-resonance: {OFF_RES_HZ} Hz")
    
    Mx, My, Mz, M_central_time = bloch_integrator_1d(
        rf_envelope_scaled, t_axis_s, positions_mm, GRAD_SLICE_M_T_PER_M,
        dt_s, T1_s, T2_s, OFF_RES_HZ, method='RK4'
    )
    
    print("  Simulation complete")
    
    # Compute slice profile
    print("\nComputing slice profile...")
    profile, fwhm_mm = compute_slice_profile(Mz, positions_mm)
    print(f"  Computed FWHM: {fwhm_mm:.2f} mm")
    
    # Note on FWHM relationship
    print(f"  Note: FWHM depends on pulse bandwidth and gradient amplitude")
    print(f"        Expected relationship: FWHM ~ BW / (gamma * G)")
    
    # Plot and save
    print("\nCreating plots...")
    plot_path = os.path.join(outdir_full, "bloch_sinc_profile.png")
    plot_pulse_and_profile(
        t_axis_s, rf_envelope_scaled, freqs_hz, spectrum, positions_mm,
        Mx, My, Mz, profile, fwhm_mm, M_central_time, plot_path
    )
    
    # Save metrics
    metrics = {
        'pulse_duration_ms': RF_DURATION_MS,
        'sinc_lobes': SINC_LOBES,
        'window_type': WINDOW if WINDOW else 'none',
        'fwhm_mm': fwhm_mm,
        'bandwidth_3db_hz': bw_3db_hz,
        'peak_flip_angle_deg': 90.0,  # Approximate, since we scaled for 90°
        'T1_ms': T1_MS,
        'T2_ms': T2_MS,
        'gradient_arb_units': GRAD_SLICE_M_T_PER_M,
        'nx_positions': nx_use,
        'spatial_step_mm': spatial_step_mm
    }
    
    metrics_path = os.path.join(outdir_full, "bloch_sinc_metrics.csv")
    save_metrics_csv(metrics, metrics_path)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Pulse design:")
    print(f"  - Duration: {RF_DURATION_MS} ms, lobes: {SINC_LOBES}, window: {WINDOW if WINDOW else 'none'}")
    print("\nSimulation:")
    print(f"  - NX={nx_use} positions across {SLICE_FOV_MM} mm => spatial step {spatial_step_mm:.3f} mm")
    print(f"  - Computed FWHM: {fwhm_mm:.2f} mm")
    print(f"\nOutputs saved to: {OUTDIR}/")
    print(f"  - bloch_sinc_profile.png")
    print(f"  - bloch_sinc_metrics.csv")
    print("="*60)


if __name__ == "__main__":
    main()

