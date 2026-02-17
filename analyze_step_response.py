"""
Analyze Step Response Data to Extract Motor Time Constant
Fits exponential response to step changes in throttle
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ============================================================
# USER CONFIGURATION
# ============================================================

# Step test data file
STEP_FILE = "./StepsTestV2_2024-07-16_164339.csv"

# Minimum step size to detect (in µs)
MIN_STEP_SIZE = 30  # Detect steps > 30 µs change

# Settling criteria (what % of final value is considered "settled")
SETTLING_THRESHOLD = 0.95  # 95% of steady-state

# ============================================================
# FUNCTIONS
# ============================================================

def first_order_response(t, tau, T_initial, T_final, t0):
    """
    First-order exponential response to step input
    T(t) = T_final + (T_initial - T_final) * exp(-(t - t0)/tau)
    
    Parameters:
    -----------
    t : array
        Time vector
    tau : float
        Time constant (what we're trying to find)
    T_initial : float
        Initial thrust before step
    T_final : float
        Final steady-state thrust after step
    t0 : float
        Time when step occurs
    """
    response = np.where(t < t0, 
                       T_initial,
                       T_final + (T_initial - T_final) * np.exp(-(t - t0) / tau))
    return response

def detect_steps(time, esc_signal, min_step=30):
    """
    Detect step changes in ESC signal
    Returns list of (step_index, step_size) tuples
    """
    # Calculate discrete derivative
    d_esc = np.diff(esc_signal)
    d_time = np.diff(time)
    
    # Rate of change (µs per second)
    rate = d_esc / d_time
    
    # Find large sudden changes (steps)
    threshold = min_step / np.median(d_time)  # Adjust for sampling rate
    
    step_indices = []
    i = 0
    while i < len(rate) - 1:
        if abs(rate[i]) > threshold:
            # Found a step - skip ahead to avoid detecting same step multiple times
            step_indices.append(i)
            i += int(0.5 / np.median(d_time))  # Skip 0.5 seconds
        i += 1
    
    return step_indices

def fit_step_response(time, thrust, step_idx, window_before=0.2, window_after=1.5):
    """
    Fit exponential to a single step response
    
    Parameters:
    -----------
    time, thrust : arrays
        Full time series data
    step_idx : int
        Index where step occurs
    window_before : float
        Seconds before step to include (for initial value)
    window_after : float
        Seconds after step to include (for fitting)
    
    Returns:
    --------
    tau : float
        Fitted time constant
    params : dict
        All fitted parameters
    fit_quality : float
        R² value of fit
    """
    t_step = time[step_idx]
    
    # Extract window around step
    mask = (time >= t_step - window_before) & (time <= t_step + window_after)
    t_window = time[mask]
    thrust_window = thrust[mask]
    
    if len(t_window) < 10:
        return None, None, None
    
    # Initial guesses
    T_initial_guess = np.mean(thrust_window[t_window < t_step])
    T_final_guess = np.mean(thrust_window[t_window > t_step + 0.5])
    tau_guess = 0.05  # 50 ms initial guess
    
    # Fit exponential
    try:
        bounds = ([0.001, -np.inf, -np.inf, t_step - 0.1],  # Lower bounds
                  [0.5, np.inf, np.inf, t_step + 0.1])      # Upper bounds
        
        popt, pcov = curve_fit(
            first_order_response,
            t_window,
            thrust_window,
            p0=[tau_guess, T_initial_guess, T_final_guess, t_step],
            bounds=bounds,
            maxfev=5000
        )
        
        tau_fit, T_init_fit, T_final_fit, t0_fit = popt
        
        # Calculate R²
        thrust_pred = first_order_response(t_window, *popt)
        ss_res = np.sum((thrust_window - thrust_pred) ** 2)
        ss_tot = np.sum((thrust_window - np.mean(thrust_window)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        params = {
            'tau': tau_fit,
            'T_initial': T_init_fit,
            'T_final': T_final_fit,
            't0': t0_fit,
            'step_size': T_final_fit - T_init_fit,
            'step_time': t_step
        }
        
        return tau_fit, params, r_squared
        
    except Exception as e:
        print(f"  Fit failed at t={t_step:.2f}s: {e}")
        return None, None, None

def plot_step_fits(time, thrust, esc_signal, step_results):
    """
    Plot all detected steps with fitted responses
    """
    n_steps = len(step_results)
    if n_steps == 0:
        print("No valid steps found to plot")
        return
    
    # Create subplot grid
    n_cols = min(3, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_steps == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, result in enumerate(step_results):
        ax = axes[i]
        
        params = result['params']
        t_step = params['step_time']
        tau = params['tau']
        
        # Plot window
        window = 1.5
        mask = (time >= t_step - 0.2) & (time <= t_step + window)
        t_win = time[mask]
        thrust_win = thrust[mask]
        
        # Plot data
        ax.plot(t_win - t_step, thrust_win, 'b.', markersize=2, label='Measured', alpha=0.6)
        
        # Plot fit
        t_fit = np.linspace(t_win[0], t_win[-1], 200)
        thrust_fit = first_order_response(t_fit, tau, params['T_initial'], 
                                          params['T_final'], params['t0'])
        ax.plot(t_fit - t_step, thrust_fit, 'r-', linewidth=2, label='Exponential Fit')
        
        # Plot time constant markers
        if params['step_size'] > 0:  # Step up
            T_63 = params['T_initial'] + 0.63 * params['step_size']
            ax.axhline(T_63, color='g', linestyle='--', alpha=0.5, label=f'63% (τ={tau*1000:.1f}ms)')
        else:  # Step down
            T_63 = params['T_initial'] + 0.63 * params['step_size']
            ax.axhline(T_63, color='g', linestyle='--', alpha=0.5, label=f'63% (τ={tau*1000:.1f}ms)')
        
        ax.axvline(0, color='k', linestyle=':', alpha=0.5, label='Step')
        ax.axvline(tau, color='g', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Time from step (s)')
        ax.set_ylabel('Thrust (N)')
        ax.set_title(f'Step {i+1}: Δ={params["step_size"]:.2f}N, τ={tau*1000:.1f}ms, R²={result["r_squared"]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('step_response_fits.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: step_response_fits.png")
    plt.show()

def main():
    """Main analysis workflow"""
    
    print("="*60)
    print("Motor Step Response Analysis")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {STEP_FILE}")
    df = pl.read_csv(
        STEP_FILE,
        schema_overrides={
            "Time (s)": pl.Float64,
            "ESC signal (µs)": pl.Float64,
            "Thrust (N)": pl.Float64,
        }
    )
    
    time = df["Time (s)"].to_numpy()
    esc_signal = df["ESC signal (µs)"].to_numpy()
    thrust = df["Thrust (N)"].to_numpy()
    
    print(f"  Duration: {time[-1] - time[0]:.1f} seconds")
    print(f"  Data points: {len(time)}")
    print(f"  Sampling rate: ~{len(time)/(time[-1] - time[0]):.1f} Hz")
    
    # Detect steps
    print(f"\nDetecting step changes (threshold: {MIN_STEP_SIZE} µs)...")
    step_indices = detect_steps(time, esc_signal, MIN_STEP_SIZE)
    print(f"  Found {len(step_indices)} potential steps")
    
    # Fit each step
    print("\nFitting exponential response to each step...")
    step_results = []
    
    for i, step_idx in enumerate(step_indices):
        t_step = time[step_idx]
        esc_before = np.mean(esc_signal[max(0, step_idx-10):step_idx])
        esc_after = np.mean(esc_signal[step_idx:min(len(time), step_idx+10)])
        
        print(f"\n  Step {i+1} at t={t_step:.2f}s: {esc_before:.0f} → {esc_after:.0f} µs")
        
        tau, params, r_squared = fit_step_response(time, thrust, step_idx)
        
        if tau is not None and r_squared > 0.8:  # Only accept good fits
            print(f"    τ = {tau*1000:.2f} ms")
            print(f"    Thrust: {params['T_initial']:.3f} → {params['T_final']:.3f} N")
            print(f"    R² = {r_squared:.4f}")
            
            step_results.append({
                'step_num': i+1,
                'tau': tau,
                'params': params,
                'r_squared': r_squared
            })
        else:
            print(f"    Fit rejected (R² = {r_squared:.4f if r_squared else 'N/A'})")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(step_results) == 0:
        print("No valid step responses found!")
        return
    
    taus = [r['tau'] for r in step_results]
    step_sizes = [r['params']['step_size'] for r in step_results]
    
    print(f"\nNumber of valid steps: {len(step_results)}")
    print(f"\nTime Constant Statistics:")
    print(f"  Mean τ:    {np.mean(taus)*1000:.2f} ms")
    print(f"  Median τ:  {np.median(taus)*1000:.2f} ms")
    print(f"  Std Dev:   {np.std(taus)*1000:.2f} ms")
    print(f"  Range:     {np.min(taus)*1000:.2f} - {np.max(taus)*1000:.2f} ms")
    
    # Actuator bandwidth (f = 1/(2π*τ))
    tau_median = np.median(taus)
    bw_hz = 1 / (2 * np.pi * tau_median)
    print(f"\nActuator Bandwidth: {bw_hz:.2f} Hz (-3dB point)")
    
    # Recommended control bandwidth
    safe_bw = bw_hz / 4  # Stay 4× below actuator limit
    print(f"Recommended Rate Loop BW: {safe_bw:.2f} Hz (4× safety margin)")
    
    # Check for nonlinearity
    print(f"\nNonlinearity Check:")
    
    # Separate up vs down steps
    up_steps = [r for r in step_results if r['params']['step_size'] > 0]
    down_steps = [r for r in step_results if r['params']['step_size'] < 0]
    
    if len(up_steps) > 0 and len(down_steps) > 0:
        tau_up = np.mean([r['tau'] for r in up_steps])
        tau_down = np.mean([r['tau'] for r in down_steps])
        print(f"  τ (step up):   {tau_up*1000:.2f} ms")
        print(f"  τ (step down): {tau_down*1000:.2f} ms")
        print(f"  Asymmetry:     {abs(tau_up - tau_down)/tau_median * 100:.1f}%")
    
    # Correlation with step size
    if len(step_results) > 2:
        correlation = np.corrcoef(taus, np.abs(step_sizes))[0, 1]
        print(f"  Correlation (τ vs step size): {correlation:.3f}")
        if abs(correlation) > 0.5:
            print(f"    → Significant dependency on operating point!")
    
    # Config file recommendation
    print(f"\n" + "="*60)
    print("RECOMMENDATION FOR VEHICLE CONFIG")
    print("="*60)
    print(f'\nAdd to your vehicle config JSON:')
    print(f'  "motor_time_constant": {tau_median:.4f},  // {tau_median*1000:.1f} ms')
    print(f'  "actuator_bandwidth_hz": {bw_hz:.2f},')
    print(f'  "max_rate_bandwidth_hz": {safe_bw:.1f}')
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_step_fits(time, thrust, esc_signal, step_results)

if __name__ == "__main__":
    main()
