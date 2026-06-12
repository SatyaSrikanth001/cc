#!/usr/bin/env python3
"""
LOUO hyperparameter tuning for OCSVM with 10+ users.
- Uses Leave‑One‑User‑Out cross‑validation.
- Imports SessionOCSVM from your train_session module (same drop_cols).
- Searches nu, gamma (including your current best 0.00055), threshold.
- Enforces FAR ≤ 0.5% and picks hyperparameters that maximise average TAR.
- Saves full grid results and best parameters.
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# **** IMPORT YOUR EXISTING CLASS ****
# Replace 'train_session' with the actual name of your training script (without .py)
from train_session import SessionOCSVM

warnings.filterwarnings('ignore')

# ======================= CONFIGURATION =======================
DATA_DIR = "./features"                     # folder with all user CSVs
TRAIN_PATTERN = "*_training_sessions.csv"
TEST_PATTERN = "*_testing_sessions.csv"     # your files use "testing"

# Use all available users (set this to the exact number if you want a subset)
N_USERS = 10                                # change to the number of users you have

FAR_LIMIT = 0.005                           # 0.5% maximum acceptable FAR

# Hyperparameter search grid
NU_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
THRESHOLD_VALUES = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]

# Gamma candidates: your current best + adaptive scaling (multiplier / n_features)
FIXED_GAMMA = 0.00055
GAMMA_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0]
# =============================================================

# 1. Identify users
train_files = sorted(glob.glob(os.path.join(DATA_DIR, TRAIN_PATTERN)))
all_users = [os.path.basename(f).replace("_training_sessions.csv", "") for f in train_files]

if len(all_users) < N_USERS:
    raise ValueError(f"Only {len(all_users)} users found, but N_USERS = {N_USERS}. Reduce N_USERS or add more data.")
users = all_users[:N_USERS]
print(f"Users ({len(users)}): {users}")

# 2. Pre‑load data for all users (features are selected by the class's drop_cols)
user_data = {}
feature_count = None

for user in users:
    model = SessionOCSVM(user)
    X_train, X_test, y_test, _, _ = model.load_and_prepare_data(
        os.path.join(DATA_DIR, f"{user}_training_sessions.csv"),
        os.path.join(DATA_DIR, f"{user}_testing_sessions.csv")
    )
    # y_test: 1 = genuine, -1 = impostor
    user_data[user] = {"X_train": X_train, "X_test": X_test, "y_test": y_test}

    if feature_count is None:
        feature_count = X_train.shape[1]
    else:
        assert X_train.shape[1] == feature_count, f"Feature count mismatch for user {user}"

n_features = feature_count
print(f"Number of features (from drop_cols): {n_features}")

# 3. Build gamma candidates
gamma_candidates = [FIXED_GAMMA] + [mult / n_features for mult in GAMMA_MULTIPLIERS]
gamma_candidates = sorted(set(gamma_candidates))   # remove duplicates
print(f"Gamma candidates: {[f'{g:.6f}' for g in gamma_candidates]}")

# 4. Leave‑One‑User‑Out search
best_tar = -1
best_params = None
results = []

for nu in NU_VALUES:
    for gamma in gamma_candidates:
        for thr in THRESHOLD_VALUES:
            fold_tars, fold_fars, fold_frrs = [], [], []

            for held_out_user in users:
                # ----- Build training set from all OTHER users -----
                train_parts = []
                for u in users:
                    if u == held_out_user:
                        continue
                    train_parts.append(user_data[u]["X_train"])
                X_train_fold = np.vstack(train_parts)

                # Scale based on this fold's training data only
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)

                # Train OCSVM
                ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
                ocsvm.fit(X_train_scaled)

                # Test on held‑out user
                X_test_held = user_data[held_out_user]["X_test"]
                y_test_held = user_data[held_out_user]["y_test"]
                X_test_scaled = scaler.transform(X_test_held)
                scores = ocsvm.decision_function(X_test_scaled)
                pred = (scores > thr).astype(int)   # 1 = genuine, 0 = fraud

                # True labels: 1 = genuine, -1 = impostor
                genuine = y_test_held == 1
                impostor = y_test_held == -1

                frr = 1.0 - pred[genuine].mean() if genuine.sum() > 0 else 0.0
                far = pred[impostor].mean() if impostor.sum() > 0 else 0.0

                fold_frrs.append(frr)
                fold_fars.append(far)
                fold_tars.append(1.0 - frr)

            avg_frr = np.mean(fold_frrs)
            avg_far = np.mean(fold_fars)
            avg_tar = np.mean(fold_tars)
            results.append((nu, gamma, thr, avg_tar, avg_frr, avg_far))

            # Keep best that satisfies FAR limit
            if avg_far <= FAR_LIMIT and avg_tar > best_tar:
                best_tar = avg_tar
                best_params = {
                    "nu": nu,
                    "gamma": gamma,
                    "threshold": thr,
                    "TAR": avg_tar,
                    "FRR": avg_frr,
                    "FAR": avg_far,
                }

# 5. Output
print("\n" + "=" * 60)
print("   LEAVE-ONE-USER-OUT TUNING COMPLETE")
print("=" * 60)

if best_params is not None:
    print(f"Best hyperparameters (FAR ≤ {FAR_LIMIT}):")
    print(f"  nu        = {best_params['nu']}")
    print(f"  gamma     = {best_params['gamma']:.6f}")
    print(f"  threshold = {best_params['threshold']}")
    print(f"  Average TAR (1-FRR) = {best_params['TAR']:.4f}")
    print(f"  Average FRR         = {best_params['FRR']:.4f}")
    print(f"  Average FAR         = {best_params['FAR']:.4f}")
else:
    print("No hyperparameter combination met the FAR limit.")
    print("Consider relaxing FAR_LIMIT or expanding the search grid.")

# Save full grid results
results_df = pd.DataFrame(
    results,
    columns=["nu", "gamma", "threshold", "avg_TAR", "avg_FRR", "avg_FAR"]
)
results_df.to_csv("louo_tuning_results.csv", index=False)
print("\nFull grid results saved to 'louo_tuning_results.csv'")