Project Handover Document

Sensor-Based Behavioral Authentication for Mobile Payment Fraud Detection

---

1. Core Objective

Original Objective

Develop a sensor-based behavioral biometric authentication system for mobile payment applications (PhonePe-like environment) capable of distinguishing legitimate users from impostors using only smartphone interaction behavior.

Current production-style pipeline:

- Accelerometer
- Gyroscope
- Touch features
- Navigation features

Authentication model:

- One-Class SVM (OCSVM)
- Per-user modeling
- Genuine-only training
- Fraud/impostor detection during testing

Business goal:

- FAR ≤ 0.5%
- Improve authentication performance from ~69-75% toward 90%+

---

New Objective

Develop a Deep Learning alternative that can outperform the handcrafted-feature OCSVM system.

Strategy:

1. Pretrain on HuMIdb
2. Fine-tune on internal 17-user dataset
3. Compare against OCSVM baseline

Target architecture:

6 Sensor Channels
(acc_x, acc_y, acc_z,
gyro_x, gyro_y, gyro_z)

↓
1D CNN

↓
BiLSTM

↓
Embedding Layer

↓
Siamese / Contrastive Loss

---

2. Key Context & Constraints

Internal Dataset

Users:

17

Split:

13 Development Users
4 Held-Out Evaluation Users

Training Data:

~40 genuine sessions/user

Testing Data:

~35-40 genuine sessions
~20-25 impostor sessions

---

Available Raw Data

The user confirmed:

Raw sensor logs exist.

Available fields:

timestamp
acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z

Formats:

- CSV
- JSON

This is critical because the deep learning pipeline will operate on raw sequences rather than engineered features.

---

Existing Feature-Based Pipeline

Feature count:

~441 engineered features

Includes:

- FFT features
- DTW features
- HMOG features
- Touch features
- Navigation features
- Correlation features
- Stability features
- Spectral features
- Statistical features

These remain relevant for OCSVM benchmarking but are NOT the primary input for the deep learning model.

---

HuMIdb

Selected as pretraining dataset.

GitHub:

https://github.com/BiDAlab/HuMIdb

Known facts:

- 600+ users
- 179+ phone models
- Timestamped sensor streams
- Variable sampling frequencies
- Device-dependent sensor rates

Expected frequency range:

Approximately:

100-140 Hz

Must be verified programmatically.

---

3. Critical Technical Discovery

Initially there was concern that:

17-user dataset ≈ 16 Hz

and

HuMIdb ≈ 120 Hz

would cause incompatibility.

After analysis:

Conclusion:

DO NOT upsample the 17-user dataset.

Instead:

Resample BOTH datasets to a common target frequency.

Preferred target:

16 Hz

Reason:

The deployment dataset is the 17-user dataset.

The pretrained model should learn temporal patterns at the same frequency it will later encounter during fine-tuning.

---

4. Sampling Rate Analysis Findings

Histogram supplied by user showed:

Approximate distribution:

12.5 Hz
14-15 Hz
16-17 Hz (majority)
18-20 Hz

Therefore:

The internal dataset is NOT fixed-rate.

It is variable-rate.

Most sessions cluster around 16 Hz.

---

5. Major Decision

Final recommendation:

Do NOT match HuMIdb to the exact histogram distribution.

Instead:

HuMIdb
100-140 Hz
↓
16 Hz

17-user Dataset
12-20 Hz
↓
16 Hz

Result:

Both datasets become:

16 Hz

This creates a unified temporal representation.

---

6. Deep Learning Architecture Decision

Chosen baseline:

Input:

(128, 6)

Where:

128 = time samples

6 = sensor channels

Channels:

acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z

Architecture:

Conv1D(64)
BatchNorm
ReLU

↓

Conv1D(128)
BatchNorm
ReLU

↓

BiLSTM(128)

↓

Dense(128)

↓

Embedding

↓

Contrastive / Siamese Loss

Important decision:

Do NOT start with:

- Transformers
- Attention models
- Very deep architectures

Reason:

Small fine-tuning dataset.

CNN + BiLSTM is expected to generalize better.

---

7. Preprocessing Pipeline

Phase 1

Analyze Sampling Rates

Both datasets must be analyzed.

For every session:

fs = 1000 / median(diff(timestamp))

Outputs:

sampling_rate_report.csv

sampling_rate_histogram.png

Purpose:

Verify actual frequencies.

---

Phase 2

Resampling

Use timestamp-based interpolation.

Technique:

Linear interpolation.

Required channels:

acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z

Target:

16 Hz

---

Phase 3

Synchronization

Create aligned rows:

timestamp
acc_x
acc_y
acc_z
gyro_x
gyro_y
gyro_z

at:

0.0000
0.0625
0.1250
0.1875
...

seconds

---

Phase 4

Normalization

Per window:

x = (x - mean) / std

Channel-wise normalization.

Purpose:

Remove device-specific bias.

---

Phase 5

Window Creation

Chosen starting window:

128 samples

At 16 Hz:

128 samples ≈ 8 seconds

Input tensor:

(128,6)

Stride:

64

Overlap:

50%

---

8. Generated Code Assets

Script 1

analyze_sampling_rates.py

Purpose:

Compute:

- Session frequency
- Duration
- Sample count

Generate:

sampling_rate_report.csv

sampling_rate_histogram.png

---

Script 2

resample_session.py

Purpose:

Convert any session to:

16 Hz

Uses:

scipy.interpolate.interp1d

Method:

Linear interpolation

---

Script 3

resample_dataset.py

Purpose:

Process entire dataset tree.

Input:

HuMIdb/

Output:

HuMIdb_16Hz/

Automatically preserves folder structure.

---

Script 4

create_windows.py

Purpose:

Generate:

(128,6)

training windows.

Outputs:

windows.npy

Per-window normalization included.

---

9. Original OCSVM Project Status

Existing completed pipeline:

Phase 1:
Feature Cleansing

Phase 2A:
Boruta + SHAP

Phase 2B:
Fisher Score

Phase 2C:
RF Leave-One-User-Out

Phase 3:
Consensus Ranking

Phase 5:
Baseline K Selection

Current OCSVM performance:

Approximately:

69-75%

depending on feature subset.

Lead's feature set:

~97-103 features

TAR:

~75%

with tuned hyperparameters.

---

10. Current State

Current focus has shifted from:

Feature Engineering

toward

Deep Learning Representation Learning.

Deep learning path is now considered the most promising route toward significant performance gains.

Raw sensor data availability has been confirmed.

This makes the DL pipeline feasible.

No CNN/BiLSTM training has started yet.

Preprocessing remains the immediate priority.

---

11. Immediate Next Steps

Step 1

Run:

analyze_sampling_rates.py

on:

- Internal 17-user dataset
- HuMIdb

Confirm actual distributions.

---

Step 2

Choose final target frequency.

Expected:

16 Hz

but must be verified.

---

Step 3

Run:

resample_dataset.py

Generate:

HuMIdb_16Hz/

17Users_16Hz/

---

Step 4

Run:

create_windows.py

Generate:

128×6 windows.

---

Step 5

Create Siamese Pair Generator

Positive:

Same user

Negative:

Different users

---

Step 6

Train CNN-BiLSTM Siamese Network

Pretraining:

HuMIdb

Fine-tuning:

17-user dataset

---

Step 7

Evaluate against OCSVM

Metrics:

Accuracy
FAR
FRR
TAR
HTER

Same test users.

Same protocol.

---

12. Critical Notes for Next AI

1. Raw sensor data exists and should be used directly.

2. Do not build the DL model on the 441 engineered features.

3. Verify actual sampling frequencies before hardcoding 16 Hz.

4. Use timestamp-based interpolation.

5. Resample BOTH datasets to a common frequency.

6. CNN + BiLSTM Siamese architecture is the current recommended baseline.

7. OCSVM remains the benchmark model.

8. Final goal is to surpass the current ~75% TAR while maintaining FAR ≤ 0.5%.

9. The preprocessing pipeline is currently more important than architecture experimentation.

10. The next concrete action is sampling-rate analysis of both datasets.