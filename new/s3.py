#!/usr/bin/env python3
"""
resample_dataset.py
Resample all CSV sessions in a dataset tree to a target frequency.
"""
import argparse
from pathlib import Path
import pandas as pd
from resample_session import resample_session, SENSOR_COLS

def resample_tree(input_root: str, output_root: str, target_hz: float = 16.0):
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    for csv_file in input_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Check required columns
            if "timestamp" not in df.columns or not set(SENSOR_COLS).issubset(df.columns):
                print(f"  Skipping {csv_file} (missing required columns)")
                continue

            resampled = resample_session(df, target_hz)
            # Preserve relative path
            rel_path = csv_file.relative_to(input_path)
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            resampled.to_csv(out_file, index=False)
            print(f"  OK: {rel_path}  ({len(df)} -> {len(resampled)})")
        except Exception as e:
            print(f"  ERROR: {csv_file} - {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Root directory with raw CSV sessions")
    parser.add_argument("output_dir", help="Output directory for resampled sessions")
    parser.add_argument("--target_hz", type=float, default=16.0, help="Target sampling rate (Hz)")
    args = parser.parse_args()
    resample_tree(args.input_dir, args.output_dir, args.target_hz)

if __name__ == "__main__":
    main()
