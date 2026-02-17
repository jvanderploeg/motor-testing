"""
Extract Motor Time Constant from Step Response Data
Works with both pre-computed settling times and raw transient data
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# CONFIGURATION
# ============================================================

# Data file with step response tests
# Use the file with high-rate time-series data
DATA_FILE = "./90PERCENT_2024-08-13_185523.csv"

# Alternative: If you have raw time-series data without pre-computed metrics
# DATA_FILE = "./your_raw_step_test.csv"

# ============================================================
# ANALYSIS
# ============================================================

def settling_time_to_time_constant(t_settle_90):
    """
    Convert 90% settling time to first-order time constant
    
    For first-order system: y(t) = 1 - exp(-t/tau)
    At 90%: 0.90 = 1 - exp(-t_90/tau)
    Solving: tau = t_90 / ln(1/(1-0.90)) = t_90 / 2.303
    """
    return t_settle_90 / 2.303

def exp_rise(t, tau, y0, yinf, t0):
    """Exponential rise: y = yinf + (y0 - yinf)*exp(-(t-t0)/tau)"""
    return np.where(t < t0, y0, yinf + (y0 - yinf) * np.exp(-(t - t0) / tau))

def analyze_with_settling_times(df):
    """
    Analyze using pre-computed 90% settling times
    (If your test software already calculated these)
    """
    print("\n" + "="*60)
    print("ANALYSIS USING PRE-COMPUTED SETTLING TIMES")
    print("="*60)
    
    # Filter out rows where settling time is null
    df_valid = df.filter(pl.col("90% settling time (s)").is_not_null())
    
    if len(df_valid) == 0:
        print("  No settling time data found!")
        return None
    
    settling_times = df_valid["90% settling time (s)"].to_numpy()
    esc_signals = df_valid["ESC signal (µs)"].to_numpy()
    
    # Convert 90% settling time to time constant
    time_constants = settling_time_to_time_constant(settling_times)
    
    print(f"\nFound {len(time_constants)} steps with settling time data:")
    print(f"\n{'ESC (µs)':<12} {'t_90% (ms)':<15} {'τ (ms)':<10}")
    print("-" * 40)
    for esc, t90, tau in zip(esc_signals, settling_times*1000, time_constants*1000):
        print(f"{esc:<12.0f} {t90:<15.2f} {tau:<10.2f}")
    
    # Statistics
    tau_mean = np.mean(time_constants)
    tau_median = np.median(time_constants)
    tau_std = np.std(time_constants)
    
    print(f"\n" + "="*60)
    print("TIME CONSTANT STATISTICS")
    print("="*60)
    print(f"  Mean:     {tau_mean*1000:.2f} ms")
    print(f"  Median:   {tau_median*1000:.2f} ms")
    print(f"  Std Dev:  {tau_std*1000:.2f} ms")
    print(f"  Range:    {np.min(time_constants)*1000:.2f} - {np.max(time_constants)*1000:.2f} ms")
    
    # Actuator bandwidth
    bw_hz = 1 / (2 * np.pi * tau_median)
    safe_bw = bw_hz / 4
    
    print(f"\n" + "="*60)
    print("BANDWIDTH RECOMMENDATIONS")
    print("="*60)
    print(f"  Actuator -3dB Bandwidth: {bw_hz:.2f} Hz")
    print(f"  Max Safe Rate Loop BW:   {safe_bw:.2f} Hz (4× margin)")
    print(f"  Recommended Rate BW:     {safe_bw*0.8:.2f} Hz (5× margin, conservative)")
    
    # Check for throttle dependency
    if len(time_constants) > 3:
        correlation = np.corrcoef(esc_signals, time_constants)[0, 1]
        print(f"\n  Correlation (ESC vs τ): {correlation:.3f}")
        if abs(correlation) > 0.5:
            print(f"    ⚠ Significant throttle dependency detected!")
            print(f"    → Consider using average τ or τ at hover throttle")
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(esc_signals, time_constants * 1000, 'bo-', markersize=8)
    plt.axhline(tau_median * 1000, color='r', linestyle='--', 
                label=f'Median = {tau_median*1000:.1f} ms')
    plt.xlabel('ESC Signal (µs)')
    plt.ylabel('Time Constant τ (ms)')
    plt.title('Motor Time Constant vs Throttle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(time_constants * 1000, bins=max(5, len(time_constants)//2), 
             color='skyblue', edgecolor='black')
    plt.axvline(tau_median * 1000, color='r', linestyle='--', linewidth=2,
                label=f'Median = {tau_median*1000:.1f} ms')
    plt.xlabel('Time Constant τ (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Constants')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('motor_time_constant_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: motor_time_constant_analysis.png")
    plt.show()
    
    return tau_median

def analyze_from_transients(df):
    """
    Analyze by fitting exponentials to thrust transients
    (If you have raw time-series data without pre-computed metrics)
    """
    print("\n" + "="*60)
    print("ANALYSIS FROM THRUST TRANSIENTS")  
    print("="*60)
    
    time = df["Time (s)"].to_numpy()
    thrust = df["Thrust (N)"].to_numpy()
    esc = df["ESC signal (µs)"].to_numpy()
    
    print(f"\nData points: {len(time)}")
    print(f"Duration: {time[-1] - time[0]:.2f} seconds")
    
    # Detect step changes in ESC signal
    d_esc = np.diff(esc)
    d_time = np.diff(time)
    
    # Find large changes (steps)
    step_threshold = 20  # µs
    step_indices = []
    i = 0
    while i < len(d_esc):
        if abs(d_esc[i]) > step_threshold:
            step_indices.append(i)
            # Skip ahead to avoid double-counting
            skip_points = int(1.0 / np.median(d_time))  # Skip ~1 second
            i += skip_points
        else:
            i += 1
    
    print(f"\nDetected {len(step_indices)} step changes")
    
    if len(step_indices) == 0:
        print("  ⚠ No steps detected - data may not have step changes")
        return None
    
    # Fit each step
    time_constants = []
    
    for step_idx in step_indices:
        t_step = time[step_idx]
        
        # Extract window around step
        window_before = 0.2  # seconds
        window_after = 1.0   # seconds
        mask = (time >= t_step - window_before) & (time <= t_step + window_after)
        
        if np.sum(mask) < 20:
            continue  # Not enough data points
        
        t_win = time[mask]
        thrust_win = thrust[mask]
        
        # Initial guesses
        T0 = np.mean(thrust_win[t_win < t_step])
        Tinf = np.mean(thrust_win[t_win > t_step + 0.5])
        tau_guess = 0.05  # 50 ms
        
        try:
            # Fit exponential
            popt, _ = curve_fit(
                exp_rise,
                t_win,
                thrust_win,
                p0=[tau_guess, T0, Tinf, t_step],
                bounds=([0.001, -10, -10, t_step-0.1],
                       [0.5, 50, 50, t_step+0.1]),
                maxfev=5000
            )
            
            tau = popt[0]
            
            # Sanity check
            if 0.005 < tau < 0.3:  # Between 5 ms and 300 ms
                time_constants.append(tau)
                print(f"  Step at t={t_step:.2f}s: τ = {tau*1000:.2f} ms")
            
        except Exception as e:
            print(f"  Step at t={t_step:.2f}s: Fit failed")
            continue
    
    if len(time_constants) == 0:
        print("\n  ⚠ No valid fits obtained")
        return None
    
    # Report statistics
    tau_median = np.median(time_constants)
    print(f"\nMedian time constant: {tau_median*1000:.2f} ms")
    
    return tau_median

def main():
    """Main analysis workflow"""
    
    print("="*60)
    print("MOTOR TIME CONSTANT EXTRACTION")
    print("="*60)
    print(f"\nInput file: {DATA_FILE}")
    
    # Load data
    df = pl.read_csv(DATA_FILE)
    
    print(f"Columns found: {df.columns}")
    
    # Check which type of analysis to use
    has_settling_time = "90% settling time (s)" in df.columns
    has_time_series = "Time (s)" in df.columns and len(df) > 20
    
    tau_result = None
    
    if has_settling_time:
        print("\n✓ Found pre-computed settling time data")
        tau_result = analyze_with_settling_times(df)
    elif has_time_series:
        print("\n✓ Found time-series data - will fit transients")
        tau_result = analyze_from_transients(df)
    else:
        print("\n✗ Data format not recognized!")
        print("  Need either:")
        print("    - Column '90% settling time (s)' with pre-computed values")
        print("    - High-rate time series data with 'Time (s)' and 'Thrust (N)'")
        return
    
    if tau_result is not None:
        # Final recommendation
        bw_hz = 1 / (2 * np.pi * tau_result)
        safe_bw = bw_hz / 4
        
        print(f"\n" + "="*60)
        print("CONFIGURATION RECOMMENDATION")
        print("="*60)
        print(f"\nAdd to vehicle config JSON:")
        print(f'  "motor_time_constant": {tau_result:.4f},  // {tau_result*1000:.1f} ms')
        print(f'  "actuator_bandwidth_hz": {bw_hz:.2f},')
        print(f'  "max_rate_bandwidth_hz": {safe_bw:.1f}  // Safe limit')
        
        print(f"\nRecommended PID bandwidths:")
        print(f"  Rate loop:     {safe_bw*0.8:.1f} Hz  (conservative start)")
        print(f"  Angle loop:    {safe_bw*0.8/4:.1f} Hz  (4× slower)")
        print(f"  Velocity loop: {safe_bw*0.8/8:.1f} Hz  (2× slower than angle)")
        print(f"  Position loop: {safe_bw*0.8/16:.1f} Hz (2× slower than velocity)")

if __name__ == "__main__":
    main()
