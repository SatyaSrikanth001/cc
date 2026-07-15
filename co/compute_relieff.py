# compute_relieff.py
import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.utils import resample
import os
import warnings

from load_data import load_feature_list, load_user_data

warnings.filterwarnings('ignore')

def compute_relieff_for_user(user_id, feature_list, data_dir='features/v2',
                             n_bootstraps=50, n_neighbors=10, n_iterations=100,
                             random_state=42):
    """
    Compute ReliefF feature weights for a single user.
    Returns a DataFrame with raw weight, bootstrap mean and std.
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
    y = combined['label'].values.astype(int)

    # Handle any NaN that might have slipped through (shouldn't, but safety)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(np.unique(y)) < 2:
        print(f"[{user_id}] Only one class present, returning zeros.")
        return pd.DataFrame({
            'feature': common_features,
            'weight_raw': 0.0,
            'weight_mean': 0.0,
            'weight_std': 0.0
        })

    # Compute raw ReliefF weights on full dataset
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_select=len(common_features),
                 n_iterations=n_iterations)
    fs.fit(X, y)
    weights_raw = fs.feature_importances_  # already an array

    # Bootstrap for stability
    rng = np.random.RandomState(random_state)
    bootstrap_weights = {i: [] for i in range(len(common_features))}

    for _ in range(n_bootstraps):
        # Stratified resampling
        boot_idx = []
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            boot_idx.extend(resample(idx, replace=True, random_state=rng))
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        # Safety: check both classes present
        if len(np.unique(y_boot)) < 2:
            continue
        fs_boot = ReliefF(n_neighbors=n_neighbors, n_features_to_select=len(common_features),
                          n_iterations=n_iterations)
        fs_boot.fit(X_boot, y_boot)
        weights_boot = fs_boot.feature_importances_
        for i, w in enumerate(weights_boot):
            bootstrap_weights[i].append(w)

    weight_means = np.array([np.mean(bootstrap_weights[i]) if bootstrap_weights[i] else weights_raw[i]
                             for i in range(len(common_features))])
    weight_stds = np.array([np.std(bootstrap_weights[i]) if bootstrap_weights[i] else 0.0
                            for i in range(len(common_features))])

    result = pd.DataFrame({
        'feature': common_features,
        'weight_raw': weights_raw,
        'weight_mean': weight_means,
        'weight_std': weight_stds
    })

    # Save bootstrap distributions optionally
    os.makedirs('bootstrap', exist_ok=True)
    for i, feat in enumerate(common_features):
        np.save(f"bootstrap/relieff_{user_id}_{feat}.npy", np.array(bootstrap_weights[i]))

    return result

if __name__ == "__main__":
    # Test with a single dummy user
    user_id = 'user_01'   # adjust as needed
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_relieff_for_user(user_id, features)
        df.to_csv(f'relieff_scores_{user_id}.csv', index=False)
        print(f"ReliefF scores saved to relieff_scores_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")