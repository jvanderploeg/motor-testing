# Motor Testing Analysis Scripts

This repository contains Python scripts for analyzing motor and propeller test data from dynamometer tests. These tools help extract key parameters needed for flight controller tuning and vehicle modeling.

## Overview

The scripts process CSV data files exported from motor test stands to calculate:
- Motor time constants and bandwidth limits
- Thrust expo curves for ArduPilot/PX4
- Propeller thrust and torque coefficients (Ct, Cq)
- Motor electrical parameters (resistance, no-load current)
- Dynamic response characteristics

## Python Scripts

### plot_sweep.py
**The primary motor characterization tool** - Comprehensive analysis of ramp/sweep test data.

**What it does:**
- Calculates `MOT_THST_EXPO` parameter for ArduPilot thrust linearization
- Calculates thrust coefficient (Ct) and torque coefficient (Cq) for propellers
- Extracts motor resistance and no-load current (I0)
- Generates detailed plots of thrust, torque, current, and RPM vs ESC signal
- Validates propeller coefficients across RPM range
- Exports simplified thrust curve CSV

**Configuration (edit at top of file):**
```python
file = "./RampTest_2024-07-21_144641.csv"  # Input CSV file
PROP_DIAMETER_INCHES = 6.0                 # Propeller diameter
PROP_PITCH_INCHES = 3.0                    # Propeller pitch
MOTOR_KV = 2300                            # Motor KV rating
MOT_SPIN_MIN = 0.12                        # ArduPilot MOT_SPIN_MIN
MOT_SPIN_MAX = 0.95                        # ArduPilot MOT_SPIN_MAX
MOT_PWM_MIN = 1050                         # ArduPilot MOT_PWM_MIN
MOT_PWM_MAX = 1900                         # ArduPilot MOT_PWM_MAX
```

**Usage:**
```powershell
python ".\Python Scripts\plot_sweep.py"
```

**Outputs:**
- Console: MOT_THST_EXPO value, Ct/Cq coefficients, motor parameters
- Multiple plots showing thrust curves, coefficients, and motor characteristics
- `thrust_curve.csv` - Simplified thrust curve data

**Expected CSV columns:**
- `ESC signal (µs)`, `Thrust (N)`, `Torque (N·m)`, `Current (A)`, `Voltage (V)`, `Motor Optical Speed (RPM)`

---

### analyze_step_response.py
**Detailed step response analysis** - Extracts motor dynamics from step tests.

**What it does:**
- Automatically detects step changes in ESC signal
- Fits first-order exponential responses to each step
- Calculates time constant (τ) for each transient
- Provides statistics (mean, median, std dev)
- Identifies nonlinearities (up vs down steps)
- Recommends actuator bandwidth and control loop limits
- Generates detailed plots showing fits and quality metrics

**Configuration (edit at top of file):**
```python
STEP_FILE = "./StepsTestV2_2024-07-16_164339.csv"  # Input file
MIN_STEP_SIZE = 30         # Minimum step size to detect (µs)
SETTLING_THRESHOLD = 0.95  # 95% settling criterion
```

**Usage:**
```powershell
python ".\Python Scripts\analyze_step_response.py"
```

**Outputs:**
- Console: Time constant statistics, bandwidth recommendations, nonlinearity analysis
- `step_response_fits.png` - Grid of plots showing each step with exponential fit
- Recommendations for `motor_time_constant`, `actuator_bandwidth_hz`, `max_rate_bandwidth_hz`

**Expected CSV columns:**
- `Time (s)`, `ESC signal (µs)`, `Thrust (N)`

---

### extract_motor_dynamics.py
**Versatile time constant extraction** - Works with settling time data or raw transients.

**What it does:**
- **Method 1:** Uses pre-computed 90% settling times (if available)
- **Method 2:** Fits exponentials to raw transient data (fallback)
- Converts settling times to first-order time constants
- Calculates actuator bandwidth
- Provides PID tuning recommendations for cascaded control loops
- Checks for throttle-dependent dynamics

**Configuration (edit at top of file):**
```python
DATA_FILE = "./90PERCENT_2024-08-13_185523.csv"  # Input file
```

**Usage:**
```powershell
python ".\Python Scripts\extract_motor_dynamics.py"
```

**Outputs:**
- Console: Time constant statistics, bandwidth limits, PID recommendations
- Plot: Time constant vs ESC signal and histogram
- `motor_time_constant_analysis.png`

**Expected CSV columns:**
- **With pre-computed data:** `ESC signal (µs)`, `90% settling time (s)`, plus other columns
- **Raw data:** `Time (s)`, `ESC signal (µs)`, `Thrust (N)`

---

### plot_discrete.py
**Multi-dataset comparison tool** - Plots motor parameters across multiple test runs.

**What it does:**
- Loads and overlays multiple CSV files
- Plots 6 key parameters vs ESC signal (thrust, torque, power, current, RPM, voltage)
- Fits thrust expo curve to one primary dataset
- Useful for comparing different test conditions or motors

**Configuration (edit in file):**
```python
datasets = [
    "./StepsTestV2_2024-07-16_164339.csv",
    "./StepsTestV2_2024-07-17_105558.csv",
    "./StepsTestV2_2024-07-17_181904.csv",
]
primary_df = dfs[1]  # Which dataset to use for expo fitting
```

**Usage:**
```powershell
python ".\Python Scripts\plot_discrete.py"
```

**Outputs:**
- Interactive plots with multiple datasets overlaid
- Console: Optimal thrust expo parameter (a)

**Expected CSV columns:**
- `ESC signal (µs)`, `Thrust (N)`, `Torque (N·m)`, `Current (A)`, `Voltage (V)`, `Motor Optical Speed (RPM)`, `Electrical Power (W)`

---

### plot_settle.py
**Simple data viewer** - Filters and displays settling time data.

**What it does:**
- Reads CSV file with motor test data
- Filters to show only rows with valid settling time data
- Displays key columns in console

**Configuration (edit in file):**
```python
file = "./90PERCENT_2024-08-13_185523.csv"
```

**Usage:**
```powershell
python ".\Python Scripts\plot_settle.py"
```

**Outputs:**
- Console: Filtered dataframe with settling time information

**Expected CSV columns:**
- `ESC signal (µs)`, `Thrust (N)`, `Voltage (V)`, `Current (A)`, `Motor Electrical Speed (RPM)`, `Electrical Power (W)`, `90% settling time (s)`, `Max acceleration (RPM/s)`

---

## Installation

All scripts use the following Python packages:
- `polars` - Fast dataframe operations
- `matplotlib` - Plotting
- `numpy` - Numerical computations
- `scipy` - Curve fitting and signal processing

Install dependencies:
```powershell
pip install -r requirements.txt
```

---

## Recommended Workflow

### 1. Initial Motor Characterization (Ramp Test)
Run a **ramp/sweep test** from minimum to maximum throttle:
```powershell
python ".\Python Scripts\plot_sweep.py" your_ramp_test.csv
```
This gives you:
- `MOT_THST_EXPO` for ArduPilot
- Propeller Ct and Cq coefficients
- Motor resistance and I0

### 2. Dynamic Response Analysis (Step Test)
Run a **step response test** with various throttle steps:
```powershell
python ".\Python Scripts\analyze_step_response.py"
```
or
```powershell
python ".\Python Scripts\extract_motor_dynamics.py"
```
This gives you:
- Motor time constant (τ)
- Actuator bandwidth limits
- Safe control loop bandwidth recommendations

### 3. Multi-Test Comparison (Optional)
Compare multiple test runs:
```powershell
python ".\Python Scripts\plot_discrete.py"
```

---

## Understanding the Outputs

### MOT_THST_EXPO
ArduPilot parameter that linearizes thrust response. Range 0-1:
- 0.0 = Linear throttle-to-thrust
- 0.5 = Moderate expo
- 1.0 = Full quadratic (thrust ∝ throttle²)

Typical values: 0.5-0.8 for most quadcopters

### Propeller Coefficients (Ct, Cq)
Non-dimensional coefficients describing propeller performance:
- **Ct** (thrust coefficient): Thrust / (ρ × n² × D⁴)
- **Cq** (torque coefficient): Torque / (ρ × n² × D⁵)

Where ρ = air density, n = rotation speed (rev/s), D = diameter

### Motor Time Constant (τ)
Time for thrust to reach 63% of final value after a step input.
- Typical range: 20-100 ms
- Smaller = faster response, higher possible control bandwidth
- Determines maximum achievable rate loop bandwidth

### Actuator Bandwidth
Frequency where thrust response drops by 3dB: **f = 1/(2πτ)**

**Rule of thumb:** Rate loop bandwidth should be 4-5× lower than actuator bandwidth for stability.

---

## Sample CSV Files

The `Sample CSVs/` folder contains example data from various motor tests. Use these to test the scripts or as reference for expected data format.

---

## Troubleshooting

**"Column not found" errors:**
- Check that your CSV has the expected column names (exact spelling/capitalization)
- Some scripts expect different columns (optical vs electrical RPM, etc.)

**Poor exponential fits (low R²):**
- Data may be too noisy
- Steps may be too small or too fast
- Try adjusting `MIN_STEP_SIZE` or fitting windows

**Motor resistance seems too high:**
- Calculated R includes ESC, wiring, and contact resistance
- For actual motor winding resistance: measure phase-to-phase with multimeter, divide by 2
- Typical motor R: 0.03-0.15 Ω (calculated will be 5-20× higher)

**Ct/Cq not constant across RPM range:**
- Normal slight variation due to Reynolds number effects
- Large variations indicate data quality issues or prop damage

---

## Notes

- All scripts have configuration sections at the top - edit these before running
- Most scripts generate plots that must be closed to return to terminal
- CSV files are assumed to be in the same directory or provide full/relative path
- Scripts are designed for Windows (PowerShell) but work on any platform with Python
