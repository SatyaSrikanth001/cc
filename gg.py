#!/usr/bin/env python3
"""
LOUO hyperparameter tuning – expanded grid with FRR limit.
- Uses Leave‑One‑User‑Out on 10 users.
- Searches a wide range of nu, gamma (adaptive + fixed), threshold.
- Enforces FAR ≤ 0.5% AND FRR ≤ 30%.
- Selects the configuration with highest average TAR.
- Data path: features/v2
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# **** IMPORT YOUR EXISTING CLASS ****
from train_session import SessionOCSVM

warnings.filterwarnings('ignore')

# ======================= CONFIGURATION =======================
DATA_DIR = "./features/v2"                # <-- UPDATED PATH
TRAIN_PATTERN = "*_training_sessions.csv"
TEST_PATTERN = "*_testing_sessions.csv"

N_USERS = 10                              # set to your current number of users

FAR_LIMIT = 0.005                         # 0.5% max FAR
FRR_LIMIT = 0.30                          # max 30% FRR (rejections)

# ---- Expanded hyperparameter grid ----
NU_VALUES = [
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    0.15, 0.2, 0.25, 0.3, 0.4, 0.5
]

THRESHOLD_VALUES = [
    -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5
]

# Gamma: your current best value + a wide range of adaptive multipliers
FIXED_GAMMA = 0.00055
GAMMA_MULTIPLIERS = [
    0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0
]
# =============================================================

# 1. Identify users
train_files = sorted(glob.glob(os.path.join(DATA_DIR, TRAIN_PATTERN)))
all_users = [os.path.basename(f).replace("_training_sessions.csv", "") for f in train_files]

if len(all_users) < N_USERS:
    raise ValueError(f"Only {len(all_users)} users found, but N_USERS = {N_USERS}. Check DATA_DIR or reduce N_USERS.")
users = all_users[:N_USERS]
print(f"Users ({len(users)}): {users}")

# 2. Pre‑load data for all users (features selected by drop_cols in SessionOCSVM)
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

# 3. Build gamma candidates (adaptive + fixed)
gamma_candidates = [FIXED_GAMMA] + [mult / n_features for mult in GAMMA_MULTIPLIERS]
gamma_candidates = sorted(set(gamma_candidates))
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

            # Keep best that satisfies BOTH limits
            if avg_far <= FAR_LIMIT and avg_frr <= FRR_LIMIT and avg_tar > best_tar:
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
    print(f"Best hyperparameters (FAR ≤ {FAR_LIMIT}, FRR ≤ {FRR_LIMIT}):")
    print(f"  nu        = {best_params['nu']}")
    print(f"  gamma     = {best_params['gamma']:.6f}")
    print(f"  threshold = {best_params['threshold']}")
    print(f"  Average TAR (1-FRR) = {best_params['TAR']:.4f}")
    print(f"  Average FRR         = {best_params['FRR']:.4f}")
    print(f"  Average FAR         = {best_params['FAR']:.4f}")
else:
    print("No hyperparameter combination met both FAR and FRR limits.")
    print("Try relaxing limits or expanding the grid further.")

# Save full grid results
results_df = pd.DataFrame(
    results,
    columns=["nu", "gamma", "threshold", "avg_TAR", "avg_FRR", "avg_FAR"]
)
results_df.to_csv("louo_tuning_expanded_results.csv", index=False)
print("\nFull grid results saved to 'louo_tuning_expanded_results.csv'")