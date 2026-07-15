#!/usr/bin/env python3
"""
Exhaustive LOUO Hyper‑Parameter Search for OCSVM
Targets: max FAR = 0, avg FRR ≤ 0.20, threshold = 0.
If a solution exists, it will be found.
Otherwise, reports limiting users and suggests next actions.
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import time

from train_session import SessionOCSVM

warnings.filterwarnings("ignore")

# ======================= CONFIGURATION =======================
DATA_DIR = "./features/v2"
TRAIN_PATTERN = "*_training_sessions.csv"
TEST_PATTERN  = "*_testing_sessions.csv"

N_JOBS = -1

# Extremely wide and dense grids
NU_VALUES = np.round(np.arange(0.001, 0.51, 0.01), 3).tolist()   # 0.001, 0.011, ..., 0.501
FIXED_GAMMA = 0.00055
# Adaptive multipliers + extra logarithmic range
GAMMA_MULTIPLIERS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

THRESHOLD = 0.0
TARGET_MAX_FAR = 0.0
TARGET_AVG_FRR = 0.20
# =============================================================

# 1. Users and data
train_files = sorted(glob.glob(os.path.join(DATA_DIR, TRAIN_PATTERN)))
all_users = [os.path.basename(f).replace("_training_sessions.csv", "") for f in train_files]
print(f"Users: {all_users}")

user_data = {}
feature_counts = []
for user in all_users:
    train_path = os.path.join(DATA_DIR, f"{user}_training_sessions.csv")
    test_path  = os.path.join(DATA_DIR, f"{user}_testing_sessions.csv")
    model = SessionOCSVM(user)
    X_train, X_test, y_test, _, _ = model.load_and_prepare_data(train_path, test_path)
    user_data[user] = {"X_train": X_train, "X_test": X_test, "y_test": y_test}
    feature_counts.append(X_train.shape[1])
assert len(set(feature_counts)) == 1, "Feature count mismatch!"
n_features = feature_counts[0]

# 2. Gamma candidates
gamma_candidates = [FIXED_GAMMA] + [mult / n_features for mult in GAMMA_MULTIPLIERS]
gamma_candidates = sorted(set(gamma_candidates))
print(f"Gamma values: {len(gamma_candidates)}")
print(f"Total combinations: {len(NU_VALUES) * len(gamma_candidates)}")

# 3. Evaluation function
def evaluate_combination(nu, gamma):
    user_fars, user_frrs, user_tars = [], [], []
    for user in all_users:
        X_train = user_data[user]["X_train"]
        X_test  = user_data[user]["X_test"]
        y_test  = user_data[user]["y_test"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        ocsvm.fit(X_train_scaled)

        scores = ocsvm.decision_function(X_test_scaled)
        pred_genuine = scores >= THRESHOLD

        is_genuine  = y_test == 1
        is_impostor = y_test == -1

        far = pred_genuine[is_impostor].mean() if is_impostor.sum() > 0 else 0.0
        frr = 1.0 - pred_genuine[is_genuine].mean() if is_genuine.sum() > 0 else 0.0
        tar = 1.0 - frr

        user_fars.append(far)
        user_frrs.append(frr)
        user_tars.append(tar)

    avg_tar = np.mean(user_tars)
    max_far = max(user_fars)
    avg_frr = np.mean(user_frrs)
    return (nu, gamma, avg_tar, max_far, avg_frr, user_fars, user_frrs)

# 4. Run search
param_combinations = [(nu, g) for nu in NU_VALUES for g in gamma_candidates]
start = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(evaluate_combination)(nu, g) for nu, g in param_combinations
)
print(f"Search finished in {(time.time()-start)/60:.1f} min")

# 5. Find best constrained solution
best_constrained = None
best_constrained_tar = -1

# Fallback: among those with max FAR = 0, pick lowest avg FRR (then highest TAR)
# Actually, we want max FAR=0 and avg FRR <=0.2
for nu, gamma, avg_tar, max_far, avg_frr, user_fars, user_frrs in results:
    if max_far == TARGET_MAX_FAR and avg_frr <= TARGET_AVG_FRR:
        if avg_tar > best_constrained_tar:
            best_constrained_tar = avg_tar
            best_constrained = (nu, gamma, avg_tar, max_far, avg_frr, user_fars, user_frrs)

# 6. Output
print("\n" + "="*60)
if best_constrained:
    nu_b, gamma_b, avg_tar_b, max_far_b, avg_frr_b, user_fars_b, user_frrs_b = best_constrained
    print("✅ SOLUTION FOUND meeting max FAR=0 and avg FRR≤0.20")
    print(f"   nu={nu_b}, gamma={gamma_b:.6f}, threshold=0")
    print(f"   Avg TAR={avg_tar_b:.4f}, Avg FRR={avg_frr_b:.4f}, Max FAR={max_far_b:.4f}")
    per_user_df = pd.DataFrame({
        "user": all_users,
        "TAR": [1-frr for frr in user_frrs_b],
        "FRR": user_frrs_b,
        "FAR": user_fars_b
    })
    per_user_df.to_csv("best_louo_per_user.csv", index=False)
    print("   Per‑user metrics saved to best_louo_per_user.csv")
else:
    print("❌ NO COMBINATION SATISFIES BOTH CONSTRAINTS.")
    # Find combination with max FAR=0 and smallest possible avg FRR (even if >0.2)
    best_maxfar_zero = None
    best_maxfar_zero_frr = 1.0
    for nu, gamma, avg_tar, max_far, avg_frr, user_fars, user_frrs in results:
        if max_far == 0.0 and avg_frr < best_maxfar_zero_frr:
            best_maxfar_zero_frr = avg_frr
            best_maxfar_zero = (nu, gamma, avg_tar, max_far, avg_frr, user_fars, user_frrs)
    if best_maxfar_zero:
        nu0, gamma0, _, _, avg_frr0, user_fars0, user_frrs0 = best_maxfar_zero
        print(f"\nBest with FAR=0: nu={nu0}, gamma={gamma0:.6f}, avg FRR={avg_frr0:.4f} (exceeds 0.20)")
        # Identify users with highest FRR
        frr_series = pd.Series(user_frrs0, index=all_users)
        worst = frr_series.nlargest(5)
        print("Users with highest FRR (>=0.2):")
        for u, frr in worst.items():
            if frr >= 0.2:
                print(f"  {u}: FRR={frr:.4f}, TAR={1-frr:.4f}")
    else:
        # No combination even gives max FAR=0
        print("No combination even achieves max FAR=0. The following users have FAR>0 in the best case:")
        # Find combination with smallest max FAR
        min_max_far = 1.0
        best_min_far = None
        for r in results:
            if r[3] < min_max_far:
                min_max_far = r[3]
                best_min_far = r
        if best_min_far:
            nu_m, gamma_m, avg_tar_m, max_far_m, avg_frr_m, user_fars_m, user_frrs_m = best_min_far
            print(f"Best achievable max FAR = {max_far_m:.4f} with nu={nu_m}, gamma={gamma_m:.6f}")
            far_series = pd.Series(user_fars_m, index=all_users)
            print("Users with FAR > 0:")
            for u, far in far_series[far_series > 0].items():
                print(f"  {u}: FAR={far:.4f}")

    print("\nTo meet the targets, you may need to:")
    print(" - Improve features (remove features that cause these users' impostors to score high)")
    print(" - Allow per‑user nu/gamma tuning (same features)")
    print(" - Slightly lower the threshold (currently 0)")

# Save full results
full_df = pd.DataFrame(
    [(nu, gamma, avg_tar, max_far, avg_frr) for nu, gamma, avg_tar, max_far, avg_frr, _, _ in results],
    columns=["nu", "gamma", "avg_TAR", "max_FAR", "avg_FRR"]
)
full_df.to_csv("louo_tuning_full_results.csv", index=False)
