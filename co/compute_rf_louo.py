# compute_rf_louo.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import os

from load_data import load_feature_list, load_user_data

def compute_rf_louo_for_user(user_id, feature_list, all_users, data_dir='features/v2',
                             n_estimators=300, n_bootstraps=50, random_state=42):
    """
    Compute Random Forest LOUO feature importance for a single user.
    Training set: all users except user_id.
    """
    # Build training set from all other users
    train_frames = []
    train_labels = []
    for other_user in all_users:
        if other_user == user_id:
            continue
        try:
            train_other, test_other, common_feats = load_user_data(other_user, feature_list, data_dir)
        except Exception as e:
            print(f"  Skipping {other_user}: {e}")
            continue
        # Genuine sessions from training (label=0)
        genuine_other = train_other[common_feats].copy()
        genuine_other['label'] = 0
        train_frames.append(genuine_other)
        # Impostor sessions from test (label=1)
        impostor_mask = test_other['label'] == 1
        if impostor_mask.sum() > 0:
            impostor_other = test_other[impostor_mask][common_feats + ['label']].copy()
            train_frames.append(impostor_other)
    if not train_frames:
        raise RuntimeError(f"No training data available for user {user_id}")
    train_data = pd.concat(train_frames, ignore_index=True)
    X_train = train_data[common_feats].values.astype(float)
    y_train = train_data['label'].values.astype(int)

    # Clean any NaNs (shouldn't be any)
    mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Compute raw importance on full training set
    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced',
                                random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance_raw = rf.feature_importances_

    # Bootstrap
    rng = np.random.RandomState(random_state)
    bootstrap_imps = {i: [] for i in range(len(common_feats))}
    for _ in range(n_bootstraps):
        # Stratified resample
        boot_idx = []
        for label in [0, 1]:
            idx = np.where(y_train == label)[0]
            if len(idx) > 0:
                boot_idx.extend(resample(idx, replace=True, random_state=rng))
        if not boot_idx:
            continue
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]
        rf_boot = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced',
                                         random_state=rng, n_jobs=-1)
        rf_boot.fit(X_boot, y_boot)
        for i in range(len(common_feats)):
            bootstrap_imps[i].append(rf_boot.feature_importances_[i])

    importance_mean = np.array([np.mean(bootstrap_imps[i]) if bootstrap_imps[i] else importance_raw[i]
                                for i in range(len(common_feats))])
    importance_std = np.array([np.std(bootstrap_imps[i]) if bootstrap_imps[i] else 0.0
                               for i in range(len(common_feats))])

    result = pd.DataFrame({
        'feature': common_feats,
        'importance_raw': importance_raw,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    })

    os.makedirs('bootstrap', exist_ok=True)
    for i, feat in enumerate(common_feats):
        np.save(f"bootstrap/rf_{user_id}_{feat}.npy", np.array(bootstrap_imps[i]))

    return result

if __name__ == "__main__":
    # Test with a single dummy user, need the full user list
    all_users = [f'user_{i:02d}' for i in range(1, 26)]
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_rf_louo_for_user(user_id, features, all_users)
        df.to_csv(f'rf_importance_{user_id}.csv', index=False)
        print(f"RF LOUO importance saved to rf_importance_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")