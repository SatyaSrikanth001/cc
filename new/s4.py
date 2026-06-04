#!/usr/bin/env python3
"""
create_windows.py
Generate 128×6 sliding windows from resampled sessions,
apply per-window channel-wise z-score normalisation, and save as .npy.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

def session_to_windows(df: pd.DataFrame, window_size=128, stride=64):
    """Slide a window over a session and normalise each window independently."""
    data = df[SENSOR_COLS].values.astype(np.float32)
    if len(data) < window_size:
        return []   # too short, skip
    windows = []
    for start in range(0, len(data) - window_size + 1, stride):
        win = data[start:start+window_size]
        # Channel-wise z-score normalisation
        mean = win.mean(axis=0, keepdims=True)
        std = win.std(axis=0, keepdims=True) + 1e-8
        win_norm = (win - mean) / std
        windows.append(win_norm)
    return windows

def process_user(user_dir: Path, window_size=128, stride=64):
    """
    user_dir should contain:
        genuine/  (training sessions)
        impostor/ (test impostor sessions)
    Returns list of genuine windows, impostor windows.
    """
    genuine_wins = []
    impostor_wins = []
    for label, folder, storage in [("genuine", "genuine", genuine_wins),
                                   ("impostor", "impostor", impostor_wins)]:
        folder_path = user_dir / folder
        if not folder_path.exists():
            continue
        for csv_file in folder_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            wins = session_to_windows(df, window_size, stride)
            storage.extend(wins)
    return genuine_wins, impostor_wins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resampled_root", help="Root of resampled dataset (e.g., 17_users_16Hz)")
    parser.add_argument("--output_dir", default="./windows", help="Where to save window arrays")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.resampled_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assume structure: root/userXX/genuine/*.csv and root/userXX/impostor/*.csv
    # If structure differs, adjust accordingly.
    all_genuine = []
    all_impostor = []
    user_windows = {}

    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        user = user_dir.name
        genuine, impostor = process_user(user_dir, args.window_size, args.stride)
        if genuine:
            user_windows[f"{user}_genuine"] = genuine
            all_genuine.extend(genuine)
        if impostor:
            user_windows[f"{user}_impostor"] = impostor
            all_impostor.extend(impostor)

    # Save per-user windows
    for name, wins in user_windows.items():
        arr = np.array(wins)
        np.save(out / f"{name}_windows.npy", arr)
        print(f"{name}: {arr.shape}")

    # Also save combined
    if all_genuine:
        np.save(out / "all_genuine_windows.npy", np.array(all_genuine))
    if all_impostor:
        np.save(out / "all_impostor_windows.npy", np.array(all_impostor))
    print(f"\nTotal genuine windows: {len(all_genuine)}")
    print(f"Total impostor windows: {len(all_impostor)}")

if __name__ == "__main__":
    main()
