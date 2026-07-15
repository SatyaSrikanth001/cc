# compute_l1_stability.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import os

from load_data import load_feature_list, load_user_data

def compute_l1_stability_for_user(user_id, feature_list, all_users, data_dir='features/v2',
                                  C=0.5, n_bootstraps=50, random_state=42):
    """
    Compute L1‑LR stability selection importance for a single user.
    Training set: all users except user_id.
    """
    # Build training set from all other users (same as RF/XGB)
    train_frames = []
    for other_user in all_users:
        if other_user == user_id:
            continue
        try:
            train_other, test_other, common_feats = load_user_data(other_user, feature_list, data_dir)
        except Exception as e:
            print(f"  Skipping {other_user}: {e}")
            continue
        genuine_other = train_other[common_feats].copy()
        genuine_other['label'] = 0
        train_frames.append(genuine_other)
        impostor_mask = test_other['label'] == 1
        if impostor_mask.sum() > 0:
            impostor_other = test_other[impostor_mask][common_feats + ['label']].copy()
            train_frames.append(impostor_other)
    if not train_frames:
        raise RuntimeError(f"No training data available for user {user_id}")
    train_data = pd.concat(train_frames, ignore_index=True)
    X_train = train_data[common_feats].values.astype(float)
    y_train = train_data['label'].values.astype(int)

    # Clean NaNs
    mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    n_features = len(common_feats)

    # We'll scale features for logistic regression (important for L1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Bootstrap stability selection
    rng = np.random.RandomState(random_state)
    selected_counts = np.zeros(n_features)

    for _ in range(n_bootstraps):
        # Stratified resample
        boot_idx = []
        for label in [0, 1]:
            idx = np.where(y_train == label)[0]
            if len(idx) > 0:
                boot_idx.extend(resample(idx, replace=True, random_state=rng))
        if not boot_idx:
            continue
        X_boot = X_train_scaled[boot_idx]
        y_boot = y_train[boot_idx]
        # L1 LR
        lr = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=2000, random_state=rng, n_jobs=-1)
        lr.fit(X_boot, y_boot)
        # Features with non-zero coefficients are selected
        selected = (np.abs(lr.coef_[0]) > 1e-8).astype(int)
        selected_counts += selected

    selection_prob = selected_counts / n_bootstraps
    # Standard deviation of binary indicator: sqrt(p*(1-p)/n_boot)
    selection_std = np.sqrt(selection_prob * (1 - selection_prob) / n_bootstraps)

    result = pd.DataFrame({
        'feature': common_feats,
        'selection_prob': selection_prob,
        'selection_std': selection_std
    })

    # We don't need to save bootstrap arrays separately for this method, but we can for consistency
    # (would require storing all binary masks; not necessary)
    return result

if __name__ == "__main__":
    all_users = [f'user_{i:02d}' for i in range(1, 26)]
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_l1_stability_for_user(user_id, features, all_users)
        df.to_csv(f'l1_stability_{user_id}.csv', index=False)
        print(f"L1 stability scores saved to l1_stability_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")