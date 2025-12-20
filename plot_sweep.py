import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.optimize as opt

file = "./full range motor sn0020.csv"
#file = "./RampTest_2024-07-21_144641.csv"

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
MOT_SPIN_MIN = 0.1
MOT_SPIN_MAX = 1.0
#MOT_SPIN_MIN = 0.12
#MOT_SPIN_MAX = 0.95

#Update for MVP
MOT_PWM_MIN = 1000
MOT_PWM_MAX = 2000
#MOT_PWM_MIN = 1050
#MOT_PWM_MAX = 1900

mot_pwm_thst_min = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MIN
mot_pwm_thst_max = MOT_PWM_MIN + (MOT_PWM_MAX - MOT_PWM_MIN) * MOT_SPIN_MAX

# filter out all rows where the ESC PWM is outside [1150, 1900]
df = df.filter(df["ESC signal (µs)"] > mot_pwm_thst_min)
df = df.filter(df["ESC signal (µs)"] < mot_pwm_thst_max)

esc_pwm = df["ESC signal (µs)"]
thrust = df["Thrust (N)"]
#torque = df["Torque (N·m)"] Not Using torque
current = df["Current (A)"]
#motor_rpm = df["Motor Optical Speed (RPM)"]
motor_rpm = df["RPM"]

max_thrust = thrust.max()


def f(esc_pwm, a):
    throttle = (esc_pwm - mot_pwm_thst_min) / (mot_pwm_thst_max - mot_pwm_thst_min)
    throttle = a * throttle**2 + (1 - a) * throttle
    return throttle * max_thrust


a, _ = opt.curve_fit(f, esc_pwm, thrust)
print(f"MOT_THST_EXPO = {a[0]}")

esc_pwm_fit = np.linspace(mot_pwm_thst_min, mot_pwm_thst_max, 100)

fig, axs = plt.subplots(2, 2)
axs[0][0].plot(esc_pwm, thrust, label="Thrust (N)")
axs[0][0].plot(esc_pwm_fit, f(esc_pwm_fit, a), label="Thrust Fit (N)")
axs[0][0].set_ylabel("Thrust (N)")
axs[0][0].set_xlabel("ESC signal (µs)")
axs[0][0].legend()

#axs[0][1].plot(esc_pwm, torque, label="Torque (N·m)")
#axs[0][1].set_ylabel("Torque (N·m)")
#axs[0][1].set_xlabel("ESC signal (µs)")
#axs[0][1].legend()

axs[1][0].plot(esc_pwm, current, label="Current (A)")
axs[1][0].set_ylabel("Current (A)")
axs[1][0].set_xlabel("ESC signal (µs)")
axs[1][0].legend()

axs[1][1].plot(esc_pwm, motor_rpm, label="Motor Optical Speed (RPM)")
axs[1][1].set_ylabel("Motor Optical Speed (RPM)")
axs[1][1].set_xlabel("ESC signal (µs)")
axs[1][1].legend()

# plot motor rpm vs thrust:

# fig, ax = plt.subplots()
# ax.plot(motor_rpm, thrust, label="Thrust (N)")
# ax.set_ylabel("Thrust (N)")
# ax.set_xlabel("Motor Optical Speed (RPM)")
# ax.legend()

# plot motor rpm^2 vs thrust:

# get ang vel in rad/s from motor rpm
motor_ang_vel = motor_rpm * 2 * np.pi / 60

# fit a model to thrust vs motor ang vel^2 (assuming 0 intercept)
a_thrust, _ = opt.curve_fit(lambda x, a: a * x**2, motor_ang_vel, thrust)
print(f"thrust motor constant = {a_thrust[0]}")

#a_torque, _ = opt.curve_fit(lambda x, a: a * x**2, motor_ang_vel, torque)
#print(f"torque motor constant = {a_torque[0]}")

# plot the fit and the data
#fig, ax = plt.subplots()
# ax.plot(motor_ang_vel, thrust, label="Thrust (N)")
# ax.plot(motor_ang_vel, a[0] * motor_ang_vel**2, label="Thrust Fit (N)")

#ax.plot(motor_ang_vel, torque, label="Torque (N·m)")
#ax.plot(motor_ang_vel, a_torque[0] * motor_ang_vel**2, label="Torque Fit (N·m)")

#ax.set_ylabel("Thrust (N) / Torque (N·m)")
#ax.set_xlabel("Motor Angular Velocity (rad/s)")

#ax.legend()

df = df.select(
    ["ESC signal (µs)", "Thrust (N)", "RPM"]
)
df = df.sort("ESC signal (µs)")
df.write_csv("thrust_curve.csv")

plt.show()
