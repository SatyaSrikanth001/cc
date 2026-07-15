# compute_xgb_louo.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import resample
import os

from load_data import load_feature_list, load_user_data

def compute_xgb_louo_for_user(user_id, feature_list, all_users, data_dir='features/v2',
                              n_estimators=200, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              n_bootstraps=50, random_state=42):
    """
    Compute XGBoost LOUO feature importance for a single user.
    Training set: all users except user_id.
    """
    # Build training set from all other users (same as RF)
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

    # Clean NaNs
    mask = np.isfinite(X_train).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Class imbalance handling
    n_genuine = (y_train == 0).sum()
    n_impostor = (y_train == 1).sum()
    scale_pos_weight = n_genuine / n_impostor if n_impostor > 0 else 1

    # Base XGBoost model
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'scale_pos_weight': scale_pos_weight,
        'random_state': random_state,
        'use_label_encoder': False,
        'verbosity': 0
    }

    # Compute raw importance on full training set
    model = xgb.XGBClassifier(**base_params)
    model.fit(X_train, y_train)
    importance_raw = model.feature_importances_  # gain-based

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
        model_boot = xgb.XGBClassifier(**base_params)
        model_boot.fit(X_boot, y_boot)
        for i in range(len(common_feats)):
            bootstrap_imps[i].append(model_boot.feature_importances_[i])

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
        np.save(f"bootstrap/xgb_{user_id}_{feat}.npy", np.array(bootstrap_imps[i]))

    return result

if __name__ == "__main__":
    all_users = [f'user_{i:02d}' for i in range(1, 26)]
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_xgb_louo_for_user(user_id, features, all_users)
        df.to_csv(f'xgb_importance_{user_id}.csv', index=False)
        print(f"XGBoost importance saved to xgb_importance_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")