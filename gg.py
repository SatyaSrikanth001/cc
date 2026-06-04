#!/usr/bin/env python3
"""
Manual Feature Combination Evaluator
Input:  a CSV with one column 'feature' listing the features to test.
Output: best TAR, FRR, FAR on the 13 dev users after a fast hyperparameter search.
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ======================= CONFIGURATION =======================
DATA_DIR = "./features"
CLEANED_FEATURES_FILE = "cleaned_features.csv"   # from Phase 1 (optional but recommended)
INPUT_FEATURE_LIST = "candidate_features.csv"    # <-- CHANGE THIS TO YOUR FILE
N_DEV_USERS = 13
FAR_LIMIT = 0.005

# Hyperparameter grids (quick but effective)
NU_VALUES = [0.1, 0.2, 0.3]                     # you can extend if needed
GAMMA_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0]  # gamma = multiplier / n_features
THRESHOLD_VALUES = [-0.2, -0.1, 0, 0.1, 0.2]
SCALER = 'standard'                             # or try 'robust' manually by changing
# =============================================================

# 1. Load development users
train_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_training_sessions.csv")))
all_users = [os.path.basename(f).replace("_training_sessions.csv", "") for f in train_files]
dev_users = all_users[:N_DEV_USERS]

# 2. Load cleaned features (if available) to ensure we only use valid features
try:
    cleaned_df = pd.read_csv(CLEANED_FEATURES_FILE)
    valid_features = set(cleaned_df['feature'].tolist())
except FileNotFoundError:
    valid_features = None  # no pre-filtering

# 3. Load candidate feature list
candidate_df = pd.read_csv(INPUT_FEATURE_LIST)
candidate_features = candidate_df['feature'].tolist()
if valid_features:
    candidate_features = [f for f in candidate_features if f in valid_features]
print(f"Candidate features: {len(candidate_features)}")

# 4. Load user data, filter to candidate features
def load_user_data(user):
    train = pd.read_csv(os.path.join(DATA_DIR, f"{user}_training_sessions.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, f"{user}_test_sessions.csv"))
    train.drop(columns=['user'], errors='ignore', inplace=True)
    test.drop(columns=['user'], errors='ignore', inplace=True)
    y_test = test['label'].astype(int)
    test.drop(columns=['label'], inplace=True)
    
    # Keep only candidate features that exist
    common = [c for c in candidate_features if c in train.columns and c in test.columns]
    train = train[common].select_dtypes(include=[np.number]).fillna(train[common].median() if len(common) else 0)
    test = test[common].select_dtypes(include=[np.number]).fillna(test[common].median() if len(common) else 0)
    return train, test, y_test, common

user_data = {}
all_features = None
for user in dev_users:
    train, test, y_test, common = load_user_data(user)
    if all_features is None:
        all_features = common
    else:
        # ensure consistent feature set across users (intersection)
        common_intersection = list(set(all_features) & set(common))
        train = train[common_intersection]
        test = test[common_intersection]
        all_features = common_intersection
    user_data[user] = {'train': train, 'test': test, 'y_test': y_test}

n_features = len(all_features)
if n_features == 0:
    raise ValueError("No common features found across dev users.")
print(f"Effective features after alignment: {n_features}")

# 5. Fast grid search (Leave-One-User-Out)
best_tar = 0
best_params = {}
results = []

for nu in NU_VALUES:
    for mult in GAMMA_MULTIPLIERS:
        gamma = mult / n_features
        for thr in THRESHOLD_VALUES:
            accs, frrs, fars = [], [], []
            for user in dev_users:
                train = user_data[user]['train'].values.astype(float)
                test  = user_data[user]['test'].values.astype(float)
                y_true = user_data[user]['y_test'].values
                
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train)
                test_scaled = scaler.transform(test)
                
                ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                ocsvm.fit(train_scaled)
                scores = ocsvm.decision_function(test_scaled)
                pred = (scores > thr).astype(int)  # 1 = predicted genuine
                
                genuine = y_true == 0
                impostor = y_true == 1
                frr = 1 - pred[genuine].mean() if genuine.sum()>0 else 0
                far = pred[impostor].mean() if impostor.sum()>0 else 0
                # accuracy mapping
                acc = accuracy_score(1 - y_true, pred)
                accs.append(acc); frrs.append(frr); fars.append(far)
            
            avg_acc = np.mean(accs)
            avg_frr = np.mean(frrs)
            avg_far = np.mean(fars)
            tar = 1 - avg_frr
            
            results.append((nu, mult, gamma, thr, avg_acc, avg_frr, avg_far, tar))
            
            if avg_far <= FAR_LIMIT and tar > best_tar:
                best_tar = tar
                best_params = {'nu': nu, 'gamma': gamma, 'thr': thr,
                               'acc': avg_acc, 'frr': avg_frr, 'far': avg_far}

# 6. Output
print("\n=== Best configuration (FAR <= 0.5%) ===")
if best_tar > 0:
    print(f"TAR (1-FRR): {best_tar:.4f}")
    print(f"Accuracy:    {best_params['acc']:.4f}")
    print(f"FRR:         {best_params['frr']:.4f}")
    print(f"FAR:         {best_params['far']:.4f}")
    print(f"Parameters:   nu={best_params['nu']}, gamma={best_params['gamma']:.6f}, threshold={best_params['thr']}")
else:
    print("No configuration met the FAR limit. Try adjusting the grid.")

# Optional: save a full report
report_df = pd.DataFrame(results, columns=['nu', 'mult', 'gamma', 'thr', 'acc', 'frr', 'far', 'tar'])
report_df.to_csv('candidate_evaluation_details.csv', index=False)
print("\nFull grid results saved to candidate_evaluation_details.csv")