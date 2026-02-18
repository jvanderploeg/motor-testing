import polars as pl
import matplotlib.pyplot as plt
import numpy as np

file = "./90PERCENT_2024-08-13_185523.csv"

df = pl.read_csv(file)
df.drop_nulls()

# ﻿Time (s),ESC signal (µs),Servo 1 (µs),Servo 2 (µs),Servo 3 (µs),AccX (g),AccY (g),AccZ (g),Torque (N·m),Thrust (N),Voltage (V),Current (A),Motor Electrical Speed (RPM),Motor Optical Speed (RPM),Electrical Power (W),Mechanical Power (W),Motor Efficiency (%),Propeller Mech. Efficiency (N/W),Overall Efficiency (N/W),Vibration (g),App message,90% settling time (s),Max acceleration (RPM/s),
# Only keep the following columns: esc signal, thrust, voltage, current, motor electrical speed, electrical power, 90% settling time, max acceleration

df = df.select(
    [
        "ESC signal (µs)",
        "Thrust (N)",
        "Voltage (V)",
        "Current (A)",
        "Motor Electrical Speed (RPM)",
        "Electrical Power (W)",
        "90% settling time (s)",
        "Max acceleration (RPM/s)",
    ]
)
df = df.filter(df["90% settling time (s)"].is_not_null())

print(df)
