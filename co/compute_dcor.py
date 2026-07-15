# compute_dcor.py
import pandas as pd
import numpy as np
import dcor
from sklearn.utils import resample
import os

from load_data import load_feature_list, load_user_data

def compute_dcor_for_user(user_id, feature_list, data_dir='features/v2',
                          n_bootstraps=50, random_state=42):
    """
    Compute Distance Correlation for a single user.
    Returns a DataFrame with raw dCor, bootstrap mean and std.
    """
    # Load data
    train_df, test_df, common_features = load_user_data(user_id, feature_list, data_dir)

    # Build combined dataset (genuine train + impostor test)
    impostor_mask = test_df['label'] == 1
    impostor_df = test_df[impostor_mask]
    genuine_df = train_df[common_features].copy()
    genuine_df['label'] = 0
    impostor_df = impostor_df[common_features + ['label']].copy()

    combined = pd.concat([genuine_df, impostor_df], ignore_index=True)
    X = combined[common_features].values.astype(float)
    y = combined['label'].values.astype(float)   # 0 or 1

    # Safety: remove rows with NaN (shouldn't be any)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(np.unique(y)) < 2:
        print(f"[{user_id}] Only one class present, returning zeros.")
        return pd.DataFrame({
            'feature': common_features,
            'dcor_raw': 0.0,
            'dcor_mean': 0.0,
            'dcor_std': 0.0
        })

    # Compute raw dCor on full combined dataset
    dcor_raw = np.array([dcor.distance_correlation(X[:, i], y) for i in range(len(common_features))])

    # Bootstrap
    rng = np.random.RandomState(random_state)
    bootstrap_dcor = {i: [] for i in range(len(common_features))}

    for _ in range(n_bootstraps):
        # Stratified resampling
        boot_idx = []
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            boot_idx.extend(resample(idx, replace=True, random_state=rng))
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        for i in range(len(common_features)):
            dcor_val = dcor.distance_correlation(X_boot[:, i], y_boot)
            bootstrap_dcor[i].append(dcor_val)

    dcor_means = np.array([np.mean(bootstrap_dcor[i]) if bootstrap_dcor[i] else dcor_raw[i] for i in range(len(common_features))])
    dcor_stds = np.array([np.std(bootstrap_dcor[i]) if bootstrap_dcor[i] else 0.0 for i in range(len(common_features))])

    result = pd.DataFrame({
        'feature': common_features,
        'dcor_raw': dcor_raw,
        'dcor_mean': dcor_means,
        'dcor_std': dcor_stds
    })

    # Save bootstrap distributions
    os.makedirs('bootstrap', exist_ok=True)
    for i, feat in enumerate(common_features):
        np.save(f"bootstrap/dcor_{user_id}_{feat}.npy", np.array(bootstrap_dcor[i]))

    return result

if __name__ == "__main__":
    user_id = 'user_01'   # adjust as needed
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_dcor_for_user(user_id, features)
        df.to_csv(f'dcor_scores_{user_id}.csv', index=False)
        print(f"Distance Correlation scores saved to dcor_scores_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")