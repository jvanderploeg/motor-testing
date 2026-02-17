import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.optimize as opt

# ============================================================
# USER CONFIGURATION
# ============================================================

# MVP File
#file = "./full range motor sn0020.csv"
#BK File
file = "./RampTest_2024-07-21_144641.csv"

# Propeller specifications
PROP_DIAMETER_INCHES = 6.0  # Propeller diameter in inches
PROP_PITCH_INCHES = 3.0     # Propeller pitch in inches

# Motor specifications
MOTOR_KV = 2300  # Motor KV rating (RPM per volt)

# Environment
AIR_DENSITY = 1.225  # kg/m³ at sea level, 15°C

df = pl.read_csv(
    file,
    schema_overrides={
        "ESC signal (µs)": pl.Float64,
        "Thrust (kgf)": pl.Float64,
        "Thrust (N)": pl.Float64,
        "Norm RPM^2": pl.Float64,
        "RPM": pl.Float64,
        "Voltage (V)": pl.Float64,
        "Current (A)": pl.Float64,
    }
)

#Update for MVP
#MOT_SPIN_MIN = 0.1
#MOT_SPIN_MAX = 1.0
#BK Params
MOT_SPIN_MIN = 0.12
MOT_SPIN_MAX = 0.95

#Update for MVP
#MOT_PWM_MIN = 1000
#MOT_PWM_MAX = 2000
#BK Params
MOT_PWM_MIN = 1050
MOT_PWM_MAX = 1900

mot_pwm_thst_min = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MIN
mot_pwm_thst_max = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MAX

# filter out all rows where the ESC PWM is outside [1150, 1900]
df = df.filter(df["ESC signal (µs)"] > mot_pwm_thst_min)
df = df.filter(df["ESC signal (µs)"] < mot_pwm_thst_max)

esc_pwm = df["ESC signal (µs)"]
thrust = df["Thrust (N)"]
torque = df["Torque (N·m)"] #MVP Not Using torque n-m
current = df["Current (A)"]
voltage = df["Voltage (V)"]
motor_rpm = df["Motor Optical Speed (RPM)"] #MVP Not Using motor optical speed

# MVP RPM
#motor_rpm = df["RPM"]

max_thrust = thrust.max()


def f(esc_pwm, a):
    throttle = (esc_pwm - mot_pwm_thst_min) / (mot_pwm_thst_max - mot_pwm_thst_min)
    throttle = a * throttle**2 + (1 - a) * throttle
    return throttle * max_thrust


a, _ = opt.curve_fit(f, esc_pwm, thrust)
print(f"MOT_THST_EXPO = {a[0]}")

esc_pwm_fit = np.linspace(mot_pwm_thst_min, mot_pwm_thst_max, 100)

# ============================================================
# CALCULATE THRUST AND TORQUE COEFFICIENTS (Ct, Cq)
# ============================================================
# Ct = Thrust / (ρ × n² × D⁴)
# Cq = Torque / (ρ × n² × D⁵)
# where:
#   ρ = air density (kg/m³)
#   n = rotational speed (rev/s) = RPM / 60
#   D = propeller diameter (m)

# Convert propeller diameter from inches to meters
prop_diameter_m = PROP_DIAMETER_INCHES * 0.0254

# Calculate rotational speed in rev/s (not rad/s)
motor_rps = motor_rpm / 60.0

# Filter out zero/low RPM to avoid division by zero
#BK filter out RPM < 4000 to focus on the range where the prop is producing significant thrust
rpm_mask = motor_rpm > 4000  # Only calculate for RPM > 4000
#MVP filter out RPM < 400 to focus on the range where the prop is producing significant thrust
#rpm_mask = motor_rpm > 400  # Only calculate for RPM > 400
motor_rps_filtered = motor_rps.filter(rpm_mask)
thrust_filtered = thrust.filter(rpm_mask)
torque_filtered = torque.filter(rpm_mask)

# Calculate Ct and Cq for each data point
ct_values = thrust_filtered / (AIR_DENSITY * motor_rps_filtered**2 * prop_diameter_m**4)
cq_values = torque_filtered / (AIR_DENSITY * motor_rps_filtered**2 * prop_diameter_m**5)

# Calculate average values (use median for robustness to outliers)
ct_mean = ct_values.mean()
ct_median = ct_values.median()
cq_mean = cq_values.mean()
cq_median = cq_values.median()

print("\n" + "="*60)
print("PROPELLER COEFFICIENTS")
print("="*60)
print(f"Propeller: {PROP_DIAMETER_INCHES}x{PROP_PITCH_INCHES} ({prop_diameter_m:.4f} m diameter)")
print(f"Air density: {AIR_DENSITY} kg/m³")
print(f"\nThrust Coefficient (Ct):")
print(f"  Mean:   {ct_mean:.6f}")
print(f"  Median: {ct_median:.6f}")
print(f"\nTorque Coefficient (Cq):")
print(f"  Mean:   {cq_mean:.6f}")
print(f"  Median: {cq_median:.6f}")
print(f"\nRecommended values for vehicle config:")
print(f'  "prop_ct": {ct_median:.4f}')
print(f'  "prop_cq": {cq_median:.5f}')
print("="*60 + "\n")

# ============================================================
# CALCULATE MOTOR PARAMETERS (Resistance and No-Load Current)
# ============================================================

# Method 1: No-load current from Current vs Torque extrapolation
# Linear fit: I = I0 + k*Torque, find I0 (y-intercept)
# Filter to use only mid-to-high torque range for better linear fit
torque_min_fit = torque.quantile(0.3)  # Use data above 30th percentile
torque_fit_mask = torque > torque_min_fit
torque_fit_data = torque.filter(torque_fit_mask)
current_fit_data = current.filter(torque_fit_mask)

# Linear regression: current = i0 + slope * torque
coeffs_i0 = np.polyfit(torque_fit_data.to_numpy(), current_fit_data.to_numpy(), 1)
i0_from_torque = coeffs_i0[1]  # y-intercept

# Method 2: No-load current from lowest thrust/torque points (average of bottom 10%)
low_power_mask = thrust < thrust.quantile(0.1)
i0_from_low_thrust = current.filter(low_power_mask).mean()

# Use the more reasonable value (should be positive and typically 0.3-2 A)
if i0_from_torque > 0 and i0_from_torque < 3.0:
    i0_recommended = i0_from_torque
else:
    i0_recommended = i0_from_low_thrust

# Motor resistance calculation
# From motor model: V = I*R + Ke*omega
# where Ke = 60/(2*pi*KV) (back-EMF constant in V/(rad/s))
# Rearranging: R = (V - Ke*omega) / I
#
# NOTE: This calculation includes total system resistance:
#   - Motor winding resistance (what we want)
#   - ESC FET resistance
#   - Wiring resistance
#   - Contact resistance
# Result is typically 5-20x higher than actual motor resistance!
# For accurate motor R: measure phase-to-phase with multimeter / 2

Ke = 60.0 / (2.0 * np.pi * MOTOR_KV)  # Back-EMF constant [V/(rad/s)]
motor_omega = motor_rpm * 2.0 * np.pi / 60.0  # Angular velocity [rad/s]

# Calculate resistance for each point
voltage_np = voltage.to_numpy()
current_np = current.to_numpy()
motor_omega_np = motor_omega.to_numpy()

# Filter out very low current points to avoid division issues
R_mask = current_np > 2.0  # Only use data where current > 2A
V_masked = voltage_np[R_mask]
I_masked = current_np[R_mask]
omega_masked = motor_omega_np[R_mask]

# Calculate back-EMF
V_bemf = Ke * omega_masked

# Calculate resistance: R = (V - V_bemf) / I
R_values = (V_masked - V_bemf) / I_masked

# Use median for robustness
motor_resistance = np.median(R_values)
motor_resistance_mean = np.mean(R_values)

print("="*60)
print("MOTOR PARAMETERS")
print("="*60)
print(f"Motor KV: {MOTOR_KV} RPM/V")
print(f"Back-EMF constant (Ke): {Ke:.6f} V/(rad/s)")
print(f"\nNo-Load Current (I0):")
print(f"  From torque extrapolation: {i0_from_torque:.3f} A")
print(f"  From low thrust average:   {i0_from_low_thrust:.3f} A")
print(f"  Recommended (auto-select): {i0_recommended:.2f} A")
print(f"\n  NOTE: To measure directly - Remove prop, apply low throttle")
print(f"        via ESC, measure battery current = I0")
print(f"\nMotor Resistance (R):")
print(f"  Mean:   {motor_resistance_mean:.4f} Ohm")
print(f"  Median: {motor_resistance:.4f} Ohm")
print(f"\n  WARNING: Calculated R = {motor_resistance:.3f} Ohm seems HIGH!")
print(f"           Expected range: 0.03-0.15 Ohm for racing motors")
print(f"           This value likely includes:")
print(f"           - ESC internal resistance")
print(f"           - Wiring resistance")
print(f"           - Phase inductance effects")
print(f"\n  To measure motor R directly:")
print(f"    1. Use multimeter with milliohm capability (or Kelvin clips)")
print(f"    2. Measure phase-to-phase resistance (any 2 of 3 wires)")
print(f"    3. Motor R (per phase) = phase-to-phase / 2")
print(f"    4. Typical: 0.05-0.12 Ohm for 2300KV 6-inch motor")
print(f"\nRecommended values for vehicle config:")
print(f'  "motor_resistance": 0.06  # <-- Use multimeter measurement!')
print(f'  "motor_io": {i0_recommended:.2f}')
print("="*60 + "\n")

# ============================================================
# PLOTTING
# ============================================================

fig, axs = plt.subplots(2, 2)
axs[0][0].plot(esc_pwm, thrust, label="Thrust (N)")
axs[0][0].plot(esc_pwm_fit, f(esc_pwm_fit, a), label="Thrust Fit (N)")
axs[0][0].set_ylabel("Thrust (N)")
axs[0][0].set_xlabel("ESC signal (µs)")
axs[0][0].legend()

#MVP not using torque n-m
axs[0][1].plot(esc_pwm, torque, label="Torque (N·m)")
axs[0][1].set_ylabel("Torque (N·m)")
axs[0][1].set_xlabel("ESC signal (µs)")
axs[0][1].legend()

axs[1][0].plot(esc_pwm, current, label="Current (A)")
axs[1][0].set_ylabel("Current (A)")
axs[1][0].set_xlabel("ESC signal (µs)")
axs[1][0].legend()

axs[1][1].plot(esc_pwm, motor_rpm, label="Motor Optical Speed (RPM)")
axs[1][1].set_ylabel("Motor Optical Speed (RPM)")
axs[1][1].set_xlabel("ESC signal (µs)")
axs[1][1].legend()

# Plot Ct and Cq vs RPM to check if they're constant
fig_coeff, axs_coeff = plt.subplots(1, 2, figsize=(12, 5))

motor_rpm_filtered = motor_rpm.filter(rpm_mask)
esc_pwm_filtered = esc_pwm.filter(rpm_mask)

axs_coeff[0].scatter(motor_rpm_filtered, ct_values, alpha=0.5, s=10)
axs_coeff[0].axhline(ct_median, color='r', linestyle='--', linewidth=2, label=f'Median Ct = {ct_median:.4f}')
axs_coeff[0].set_xlabel('Motor RPM')
axs_coeff[0].set_ylabel('Thrust Coefficient (Ct)')
axs_coeff[0].set_title('Ct vs RPM (should be relatively constant)')
axs_coeff[0].legend()
axs_coeff[0].grid(True, alpha=0.3)

axs_coeff[1].scatter(motor_rpm_filtered, cq_values, alpha=0.5, s=10)
axs_coeff[1].axhline(cq_median, color='r', linestyle='--', linewidth=2, label=f'Median Cq = {cq_median:.5f}')
axs_coeff[1].set_xlabel('Motor RPM')
axs_coeff[1].set_ylabel('Torque Coefficient (Cq)')
axs_coeff[1].set_title('Cq vs RPM (should be relatively constant)')
axs_coeff[1].legend()
axs_coeff[1].grid(True, alpha=0.3)

fig_coeff.tight_layout()

# Plot motor parameters - Current vs Torque for I0 extraction
fig_motor, axs_motor = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Current vs Torque with I0 extrapolation
axs_motor[0].scatter(torque, current, alpha=0.5, s=10, label='Measured Data')
torque_fit_range = np.linspace(0, torque.max(), 100)
current_fit_line = coeffs_i0[0] * torque_fit_range + coeffs_i0[1]
axs_motor[0].plot(torque_fit_range, current_fit_line, 'r-', linewidth=2, label=f'Linear Fit (mid-high torque)')
axs_motor[0].axhline(i0_recommended, color='g', linestyle='--', linewidth=2, label=f'I0 (recommended) = {i0_recommended:.2f} A')
axs_motor[0].set_xlabel('Torque (N·m)')
axs_motor[0].set_ylabel('Current (A)')
axs_motor[0].set_title('No-Load Current Extraction')
axs_motor[0].legend()
axs_motor[0].grid(True, alpha=0.3)
axs_motor[0].set_xlim(left=0)

# Plot 2: Resistance distribution
axs_motor[1].hist(R_values, bins=30, alpha=0.7, edgecolor='black')
axs_motor[1].axvline(motor_resistance, color='r', linestyle='--', linewidth=2, label=f'Median R = {motor_resistance:.3f} Ohm')
axs_motor[1].set_xlabel('Motor Resistance (Ohm)')
axs_motor[1].set_ylabel('Frequency')
axs_motor[1].set_title('Motor Resistance Distribution')
axs_motor[1].legend()
axs_motor[1].grid(True, alpha=0.3, axis='y')

fig_motor.tight_layout()

# plot motor rpm vs thrust:

fig, ax = plt.subplots()
ax.plot(motor_rpm, thrust, label="Thrust (N)")
ax.set_ylabel("Thrust (N)")
ax.set_xlabel("Motor Optical Speed (RPM)")
ax.legend()

# plot motor rpm^2 vs thrust:

# get ang vel in rad/s from motor rpm
motor_ang_vel = motor_rpm * 2 * np.pi / 60

# fit a model to thrust vs motor ang vel^2 (assuming 0 intercept)
a_thrust, _ = opt.curve_fit(lambda x, a: a * x**2, motor_ang_vel, thrust)
print(f"thrust motor constant = {a_thrust[0]}")

a_torque, _ = opt.curve_fit(lambda x, a: a * x**2, motor_ang_vel, torque)
print(f"torque motor constant = {a_torque[0]}")

# plot the fit and the data
fig, ax1 = plt.subplots()
ax1.set_title('Thrust and Torque vs Angular Velocity (ω² fits)')

# Thrust on left y-axis
color_thrust = 'tab:blue'
ax1.set_xlabel('Motor Angular Velocity (rad/s)')
ax1.set_ylabel('Thrust (N)', color=color_thrust)
ax1.plot(motor_ang_vel, thrust, 'o', alpha=0.5, color=color_thrust, label="Thrust (N)")
ax1.plot(motor_ang_vel, a_thrust[0] * motor_ang_vel**2, '-', color=color_thrust, linewidth=2, label="Thrust Fit (N)")
ax1.tick_params(axis='y', labelcolor=color_thrust)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Torque on right y-axis
ax2 = ax1.twinx()
color_torque = 'tab:orange'
ax2.set_ylabel('Torque (N·m)', color=color_torque)
ax2.plot(motor_ang_vel, torque, 's', alpha=0.5, color=color_torque, markersize=3, label="Torque (N·m)")
ax2.plot(motor_ang_vel, a_torque[0] * motor_ang_vel**2, '-', color=color_torque, linewidth=2, label="Torque Fit (N·m)")
ax2.tick_params(axis='y', labelcolor=color_torque)
ax2.legend(loc='upper right')

fig.tight_layout()

df = df.select(
    ["ESC signal (µs)", "Thrust (N)", "Motor Optical Speed (RPM)"]
)
#MVP uses RPM
#df = df.select(
#    ["ESC signal (µs)", "Thrust (N)", "RPM"]
#)

df = df.sort("ESC signal (µs)")
df.write_csv("thrust_curve.csv")

plt.show()
