import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

datasets = [
    "./StepsTestV2_2024-07-16_164339.csv",
    "./StepsTestV2_2024-07-17_105558.csv",
    "./StepsTestV2_2024-07-17_181904.csv",
]

dfs = [pl.read_csv(dataset) for dataset in datasets]
fig, axs = plt.subplots(2, 3)

for i, df in enumerate(dfs):
    esc_pwm = df["ESC signal (µs)"]
    thrust = df["Thrust (N)"]
    torque = df["Torque (N·m)"]
    current = df["Current (A)"]
    voltage = df["Voltage (V)"]
    motor_rpm = df["Motor Optical Speed (RPM)"]
    power = df["Electrical Power (W)"]

    axs[0][0].plot(esc_pwm, thrust, label=f"Dataset {i}")
    axs[0][1].plot(esc_pwm, torque, label=f"Dataset {i}")
    axs[0][2].plot(esc_pwm, power, label=f"Dataset {i}")

    axs[1][0].plot(esc_pwm, current, label=f"Dataset {i}")
    axs[1][1].plot(esc_pwm, motor_rpm, label=f"Dataset {i}")
    axs[1][2].plot(esc_pwm, voltage, label=f"Dataset {i}")

axs[0][0].set_ylabel("Thrust (N)")
axs[0][1].set_ylabel("Torque (N·m)")
axs[0][2].set_ylabel("Electrical Power (W)")
axs[1][0].set_ylabel("Current (A)")
axs[1][1].set_ylabel("Motor Optical Speed (RPM)")
axs[1][2].set_ylabel("Voltage (V)")
axs[0][2].set_xlabel("ESC signal (µs)")
axs[1][2].set_xlabel("ESC signal (µs)")
plt.legend()
plt.show()

primary_df = dfs[1]

esc_pwm = primary_df["ESC signal (µs)"]
thrust = primary_df["Thrust (N)"]

# truncate last 2 entries of esc_pwm and thrust:


max_thrust = thrust.max()


def f(esc_pwm, a):
    spin_min = 0.15
    spin_max = 0.7
    throttle = (esc_pwm - 1000) / 1000
    throttle = (throttle - spin_min) / (spin_max - spin_min)
    throttle = a * throttle**2 + (1 - a) * throttle
    return throttle * max_thrust


optimal_params, _ = opt.curve_fit(f, esc_pwm, thrust)

a = optimal_params
# a = 0.78
print(f"Optimal parameters: a = {a}")

plt.plot(esc_pwm, thrust)
plt.xlabel("Normalized ESC signal")
plt.ylabel("Normalized Thrust")

# plot the best fit curve
x = np.linspace(1000, 2000, 100)
y = f(x, a)
plt.plot(x, y, color="red")

plt.show()
