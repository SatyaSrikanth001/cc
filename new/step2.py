#!/usr/bin/env python3
"""
resample_session.py
Function to resample a raw session to a target frequency using linear interpolation.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def resample_session(df: pd.DataFrame, target_hz: float = 16.0) -> pd.DataFrame:
    """
    Resample a raw session DataFrame (columns: timestamp, + SENSOR_COLS)
    to a regular 1/target_hz second interval using linear interpolation.
    """
    if df.empty or len(df) < 2:
        return df

    # Original time in seconds (timestamp assumed in milliseconds)
    original_time = df["timestamp"].values / 1000.0
    start_t = original_time[0]
    end_t = original_time[-1]

    # New uniform time grid
    new_time = np.arange(start_t, end_t, 1.0 / target_hz)

    # Interpolate each sensor column
    new_data = {"timestamp": (new_time * 1000).astype(np.int64)}
    for col in SENSOR_COLS:
        f = interp1d(original_time, df[col].values, kind="linear", fill_value="extrapolate")
        new_data[col] = f(new_time)

    return pd.DataFrame(new_data)

# Quick test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python resample_session.py input.csv output.csv")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    resampled = resample_session(df, target_hz=16.0)
    resampled.to_csv(sys.argv[2], index=False)
    print(f"Resampled {len(df)} -> {len(resampled)} samples")
