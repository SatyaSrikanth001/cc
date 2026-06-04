import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"

session_records = []
user_records = []

json_files = sorted(Path(DATASET_DIR).glob("*.json"))

print(f"Found {len(json_files)} user files")

for json_file in json_files:

    print(f"Processing {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_id = data.get("user_id", json_file.stem)

    sessions = data["data"]["sessions"]["sessions"]

    user_fs = []
    user_duration = []
    user_samples = []

    for session in sessions:

        sensor_events = session.get("sensor_events", [])

        if len(sensor_events) < 2:
            continue

        timestamps = np.array(
            [e["timestamp"] for e in sensor_events],
            dtype=np.int64
        )

        diffs = np.diff(timestamps)

        diffs = diffs[diffs > 0]

        if len(diffs) == 0:
            continue

        median_dt = np.median(diffs)
        mean_dt = np.mean(diffs)
        std_dt = np.std(diffs)

        fs = 1000.0 / median_dt

        duration_sec = (
            timestamps[-1] - timestamps[0]
        ) / 1000.0

        samples = len(timestamps)

        session_records.append({
            "user_id": user_id,
            "session_id": session.get("session_id"),
            "session_type": session.get("session_type"),

            "samples": samples,
            "duration_sec": duration_sec,

            "sampling_rate_hz": fs,

            "median_dt_ms": median_dt,
            "mean_dt_ms": mean_dt,
            "std_dt_ms": std_dt,

            "touch_events":
                len(session.get("touch_events", [])),

            "navigation_events":
                len(session.get("navigation_events", []))
        })

        user_fs.append(fs)
        user_duration.append(duration_sec)
        user_samples.append(samples)

    if len(user_fs):

        user_records.append({
            "user_id": user_id,

            "num_sessions":
                len(user_fs),

            "median_fs":
                np.median(user_fs),

            "mean_fs":
                np.mean(user_fs),

            "median_duration":
                np.median(user_duration),

            "mean_duration":
                np.mean(user_duration),

            "median_samples":
                np.median(user_samples),

            "mean_samples":
                np.mean(user_samples)
        })

# ----------------------------------
# SAVE CSV REPORTS
# ----------------------------------

session_df = pd.DataFrame(session_records)

user_df = pd.DataFrame(user_records)

session_df.to_csv(
    "sampling_rate_report.csv",
    index=False
)

user_df.to_csv(
    "user_statistics.csv",
    index=False
)

print("\nSaved CSV reports")

# ----------------------------------
# GLOBAL SUMMARY
# ----------------------------------

freqs = session_df["sampling_rate_hz"]

print("\n========== GLOBAL STATS ==========")

print(f"Sessions : {len(session_df)}")

print(f"Mean FS  : {freqs.mean():.2f} Hz")
print(f"Median FS: {freqs.median():.2f} Hz")

print(f"Std FS   : {freqs.std():.2f} Hz")

print(f"Min FS   : {freqs.min():.2f} Hz")
print(f"Max FS   : {freqs.max():.2f} Hz")

print("\nPercentiles")

for p in [5,10,25,50,75,90,95]:
    print(
        f"{p}% = "
        f"{np.percentile(freqs,p):.2f} Hz"
    )

# ----------------------------------
# HISTOGRAM 1
# SAMPLING RATE
# ----------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    session_df["sampling_rate_hz"],
    bins=40
)

plt.axvline(
    freqs.mean(),
    linestyle="--",
    linewidth=2,
    label=f"Mean={freqs.mean():.2f}"
)

plt.axvline(
    freqs.median(),
    linewidth=2,
    label=f"Median={freqs.median():.2f}"
)

plt.xlabel("Sampling Rate (Hz)")
plt.ylabel("Sessions")
plt.title("Sampling Rate Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    "sampling_rate_histogram.png",
    dpi=300
)

plt.close()

# ----------------------------------
# HISTOGRAM 2
# DURATION
# ----------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    session_df["duration_sec"],
    bins=40
)

plt.xlabel("Duration (sec)")
plt.ylabel("Sessions")

plt.title(
    "Session Duration Distribution"
)

plt.tight_layout()

plt.savefig(
    "duration_histogram.png",
    dpi=300
)

plt.close()

# ----------------------------------
# HISTOGRAM 3
# SAMPLE COUNT
# ----------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    session_df["samples"],
    bins=40
)

plt.xlabel("Samples")
plt.ylabel("Sessions")

plt.title(
    "Sample Count Distribution"
)

plt.tight_layout()

plt.savefig(
    "sample_count_histogram.png",
    dpi=300
)

plt.close()

# ----------------------------------
# HISTOGRAM 4
# JITTER
# ----------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    session_df["std_dt_ms"],
    bins=40
)

plt.xlabel(
    "Timestamp Jitter (Std DT ms)"
)

plt.ylabel("Sessions")

plt.title(
    "Timestamp Jitter Distribution"
)

plt.tight_layout()

plt.savefig(
    "timestamp_jitter_histogram.png",
    dpi=300
)

plt.close()

print("\nFinished analysis")