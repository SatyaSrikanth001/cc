#!/usr/bin/env python3
"""
Naive hyperparameter tuning – trains on each user's own genuine data,
tests on their test set, computes global average TAR.
WARNING: This method overfits to the test set because you use it directly for selection.
Use only for rough exploration.
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ---- import your existing class ----
from your_training_script import SessionOCSVM   # adjust import

warnings.filterwarnings('ignore')

DATA_DIR = "./features"
TRAIN_PATTERN = "*_training_sessions.csv"
TEST_PATTERN = "*_testing_sessions.csv"
N_USERS = 3
FAR_LIMIT = 0.005

NU_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
THRESHOLD_VALUES = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
FIXED_GAMMA = 0.00055
GAMMA_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0]

# 1. User list
train_files = sorted(glob.glob(os.path.join(DATA_DIR, TRAIN_PATTERN)))
all_users = [os.path.basename(f).replace("_training_sessions.csv", "") for f in train_files]
users = all_users[:N_USERS]
print(f"Users: {users}")

# 2. Pre‑load data for each user (features already selected via drop_cols in class)
user_data = {}
feature_count = None
for user in users:
    model = SessionOCSVM(user)
    X_train, X_test, y_test, _, _ = model.load_and_prepare_data(
        os.path.join(DATA_DIR, f"{user}_training_sessions.csv"),
        os.path.join(DATA_DIR, f"{user}_testing_sessions.csv")
    )
    user_data[user] = {'X_train': X_train, 'X_test': X_test, 'y_test': y_test}
    if feature_count is None:
        feature_count = X_train.shape[1]
    else:
        assert X_train.shape[1] == feature_count, "Feature count mismatch"

n_features = feature_count
print(f"Number of features: {n_features}")

# 3. Gamma candidates
gamma_candidates = [FIXED_GAMMA] + [mult / n_features for mult in GAMMA_MULTIPLIERS]
gamma_candidates = sorted(set(gamma_candidates))

# 4. Loop over all combinations
best_tar = -1
best_params = None
results = []

for nu in NU_VALUES:
    for gamma in gamma_candidates:
        for thr in THRESHOLD_VALUES:
            global_tars = []
            global_fars = []
            global_frrs = []
            for user in users:
                X_train = user_data[user]['X_train']
                X_test = user_data[user]['X_test']
                y_test = user_data[user]['y_test']

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                ocsvm.fit(X_train_scaled)
                scores = ocsvm.decision_function(X_test_scaled)
                pred = (scores > thr).astype(int)

                genuine = y_test == 1
                impostor = y_test == -1
                frr = 1.0 - pred[genuine].mean() if genuine.sum() > 0 else 0.0
                far = pred[impostor].mean() if impostor.sum() > 0 else 0.0
                global_frrs.append(frr)
                global_fars.append(far)
                global_tars.append(1.0 - frr)

            avg_frr = np.mean(global_frrs)
            avg_far = np.mean(global_fars)
            avg_tar = np.mean(global_tars)
            results.append((nu, gamma, thr, avg_tar, avg_frr, avg_far))

            if avg_far <= FAR_LIMIT and avg_tar > best_tar:
                best_tar = avg_tar
                best_params = {
                    'nu': nu,
                    'gamma': gamma,
                    'threshold': thr,
                    'TAR': avg_tar,
                    'FRR': avg_frr,
                    'FAR': avg_far
                }

# 5. Output
print("\n=== Global tuning complete ===")
if best_params:
    print(f"Best hyperparameters: nu={best_params['nu']}, gamma={best_params['gamma']:.6f}, threshold={best_params['threshold']}")
    print(f"  Global TAR: {best_params['TAR']:.4f}")
    print(f"  Global FRR: {best_params['FRR']:.4f}")
    print(f"  Global FAR: {best_params['FAR']:.4f}")
else:
    print("No combination met FAR limit.")
pd.DataFrame(results, columns=['nu', 'gamma', 'threshold', 'avg_TAR', 'avg_FRR', 'avg_FAR']) \
    .to_csv('global_tuning_results.csv', index=False)
print("Full results saved to global_tuning_results.csv")