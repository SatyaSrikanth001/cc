#!/usr/bin/env python3
"""
analyze_sampling_rates_detailed.py
Extremely detailed sampling‑rate analysis of raw sensor CSV sessions.

For every CSV file in the dataset tree (recursively), this script:
  1. Reads timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  2. Computes the effective sampling frequency as 1000 / median(diff(timestamp))
  3. Also computes the raw timestamp span, duration, and sample count
  4. Optionally infers the user ID from the folder name (one level up)
  5. Saves a detailed CSV report
  6. Prints a rich summary (percentiles, per-user stats)
  7. Writes a text summary
  8. Plots a histogram with mean and median markers
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

# ---------- 1. Core frequency estimator ----------
def estimate_fs(timestamps: np.ndarray) -> float:
    """
    Returns the sampling frequency (Hz) estimated from the median
    inter‑timestamp interval. timestamps must be in milliseconds.
    """
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(timestamps)
    # Remove any zero or negative diffs (can happen with jitter)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median_diff = np.median(diffs)
    if median_diff == 0:
        return 0.0
    return 1000.0 / median_diff


# ---------- 2. Process entire directory tree ----------
def process_root(root_dir: str, infer_user: bool = True):
    """
    Walk root_dir recursively, read every CSV that contains the required
    sensor columns, and compute session statistics.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.
    infer_user : bool
        If True, the parent folder name of the CSV is treated as the user ID.

    Returns
    -------
    pd.DataFrame with columns:
        user, session_file, frequency_hz, duration_sec, samples,
        time_start_ms, time_end_ms, timestamp_range_ms
    """
    root = Path(root_dir)
    records = []
    skipped_no_cols = []
    skipped_read_error = []

    for csv_file in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            skipped_read_error.append((str(csv_file), str(e)))
            continue

        # Check required columns
        required_cols = ["timestamp", "acc_x", "acc_y", "acc_z",
                         "gyro_x", "gyro_y", "gyro_z"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            skipped_no_cols.append((str(csv_file), missing))
            continue

        ts = df["timestamp"].values
        fs = estimate_fs(ts)

        # Compute duration and time boundaries
        if len(ts) > 1:
            start_ms = ts[0]
            end_ms   = ts[-1]
            duration_sec = (end_ms - start_ms) / 1000.0
            range_ms = end_ms - start_ms
        else:
            start_ms = ts[0] if len(ts) == 1 else 0
            end_ms = start_ms
            duration_sec = 0.0
            range_ms = 0

        # Infer user from parent folder name
        user = "unknown"
        if infer_user:
            user = csv_file.parent.name if csv_file.parent != root else "root"

        records.append({
            "user": user,
            "session_file": str(csv_file),
            "frequency_hz": fs,
            "duration_sec": duration_sec,
            "samples": len(df),
            "time_start_ms": start_ms,
            "time_end_ms": end_ms,
            "timestamp_range_ms": range_ms
        })

    # Print any skipped files
    if skipped_no_cols:
        print("\n=== Files missing required columns ===")
        for f, missing in skipped_no_cols[:10]:
            print(f"  {f}  missing: {missing}")
        if len(skipped_no_cols) > 10:
            print(f"  ... and {len(skipped_no_cols)-10} more")

    if skipped_read_error:
        print("\n=== Files that could not be read ===")
        for f, err in skipped_read_error[:10]:
            print(f"  {f}  error: {err}")

    return pd.DataFrame(records)


# ---------- 3. Print detailed summary ----------
def print_summary(df: pd.DataFrame, out_txt_file: str = None):
    """Print and optionally save a rich text summary."""
    if df.empty:
        print("No valid sessions found.")
        return

    # General stats
    freqs = df["frequency_hz"].dropna()
    if len(freqs) == 0:
        print("No frequency data.")
        return

    lines = []
    lines.append("=" * 60)
    lines.append("SAMPLING RATE ANALYSIS - DETAILED SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total sessions analyzed      : {len(df)}")
    lines.append(f"Total unique users           : {df['user'].nunique()}")
    lines.append(f"")
    lines.append("--- Frequency Statistics ---")
    lines.append(f"  Mean   : {freqs.mean():.2f} Hz")
    lines.append(f"  Median : {freqs.median():.2f} Hz")
    lines.append(f"  Std    : {freqs.std():.2f} Hz")
    lines.append(f"  Min    : {freqs.min():.2f} Hz")
    lines.append(f"  Max    : {freqs.max():.2f} Hz")
    lines.append(f"  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        lines.append(f"    {p:2d}%  : {np.percentile(freqs, p):.2f} Hz")

    # Sessions below / above certain thresholds
    below_10 = (freqs < 10).sum()
    above_200 = (freqs > 200).sum()
    lines.append(f"  Sessions < 10 Hz : {below_10}")
    lines.append(f"  Sessions > 200 Hz: {above_200}")

    lines.append(f"")
    lines.append("--- Duration Statistics ---")
    dur = df["duration_sec"]
    lines.append(f"  Mean   : {dur.mean():.2f} s")
    lines.append(f"  Median : {dur.median():.2f} s")
    lines.append(f"  Min    : {dur.min():.2f} s")
    lines.append(f"  Max    : {dur.max():.2f} s")

    lines.append(f"")
    lines.append("--- Sample Counts ---")
    lines.append(f"  Mean   : {df['samples'].mean():.1f}")
    lines.append(f"  Median : {df['samples'].median():.1f}")

    # Per-user summary (if users exist)
    if df['user'].nunique() > 1:
        lines.append(f"")
        lines.append("--- Per-User Frequency Summary (median) ---")
        user_median = df.groupby("user")["frequency_hz"].median().sort_values()
        for user, med in user_median.items():
            cnt = df[df["user"] == user].shape[0]
            lines.append(f"  {user:20s}: {med:6.2f} Hz  ({cnt} sessions)")

    # Print to console
    for line in lines:
        print(line)

    # Save to file if requested
    if out_txt_file:
        with open(out_txt_file, "w") as f:
            f.write("\n".join(lines))
        print(f"\nSummary saved to {out_txt_file}")


# ---------- 4. Enhanced histogram ----------
def plot_histogram(df: pd.DataFrame, output_plot: str):
    """Create a histogram with mean and median lines."""
    freqs = df["frequency_hz"].dropna()
    if len(freqs) == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(freqs, bins=40, edgecolor='black', alpha=0.7)
    mean_val = freqs.mean()
    median_val = freqs.median()

    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.2f} Hz')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2,
                label=f'Median: {median_val:.2f} Hz')

    plt.xlabel("Sampling frequency (Hz)")
    plt.ylabel("Number of sessions")
    plt.title("Detailed Sampling Rate Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Histogram saved to {output_plot}")


# ---------- 5. Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Detailed sampling-rate analysis of raw sensor sessions."
    )
    parser.add_argument("dataset_dir",
                        help="Root folder containing session CSV files")
    parser.add_argument("--output_report", default="sampling_rate_report_detailed.csv",
                        help="CSV report of every session")
    parser.add_argument("--output_plot", default="sampling_rate_histogram_detailed.png",
                        help="Histogram image")
    parser.add_argument("--output_summary", default="sampling_rate_summary.txt",
                        help="Text summary file")
    parser.add_argument("--no_user_inference", action="store_true",
                        help="Do not infer user from parent folder name")
    args = parser.parse_args()

    # Process dataset
    df = process_root(args.dataset_dir, infer_user=not args.no_user_inference)

    if df.empty:
        print("ERROR: No valid CSV files found. Exiting.")
        sys.exit(1)

    # Save detailed CSV
    df.to_csv(args.output_report, index=False)
    print(f"Detailed report saved to {args.output_report}")

    # Print and save summary
    print_summary(df, args.output_summary)

    # Plot
    plot_histogram(df, args.output_plot)


if __name__ == "__main__":
    main()
