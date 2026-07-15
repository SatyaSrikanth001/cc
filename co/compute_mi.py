# compute_mi.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
import os
import joblib

from load_data import load_feature_list, load_user_data

def compute_mi_for_user(user_id, feature_list, data_dir='features/v2', n_bootstraps=50, random_state=42):
    """
    Compute Mutual Information scores for a single user.
    Returns a DataFrame with raw MI, bootstrap mean and std.
    """
    # Load data
    train_df, test_df, common_features = load_user_data(user_id, feature_list, data_dir)
    
    # Build combined dataset: genuine training (label 0) + impostor test (label 1)
    # impostor rows: label == 1
    impostor_mask = test_df['label'] == 1
    impostor_df = test_df[impostor_mask]
    genuine_df = train_df[common_features].copy()
    genuine_df['label'] = 0
    impostor_df = impostor_df[common_features + ['label']].copy()
    
    combined = pd.concat([genuine_df, impostor_df], ignore_index=True)
    X = combined[common_features].values
    y = combined['label'].values
    
    # Compute raw MI on the full combined dataset
    mi_raw = mutual_info_classif(X, y, random_state=random_state)
    
    # Bootstrap to estimate variability
    rng = np.random.RandomState(random_state)
    bootstrap_mi = {i: [] for i in range(len(common_features))}
    
    for _ in range(n_bootstraps):
        # Stratified resampling to maintain class balance
        boot_idx = []
        for label in [0, 1]:
            idx = np.where(y == label)[0]
            boot_idx.extend(resample(idx, replace=True, random_state=rng))
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        mi_boot = mutual_info_classif(X_boot, y_boot, random_state=rng)
        for i, val in enumerate(mi_boot):
            bootstrap_mi[i].append(val)
    
    mi_means = np.array([np.mean(bootstrap_mi[i]) for i in range(len(common_features))])
    mi_stds = np.array([np.std(bootstrap_mi[i]) for i in range(len(common_features))])
    
    # Create result DataFrame
    result = pd.DataFrame({
        'feature': common_features,
        'mi_raw': mi_raw,
        'mi_mean': mi_means,
        'mi_std': mi_stds
    })
    
    # Optionally save bootstrap distributions
    os.makedirs('bootstrap', exist_ok=True)
    for i, feat in enumerate(common_features):
        np.save(f"bootstrap/mi_{user_id}_{feat}.npy", np.array(bootstrap_mi[i]))
    
    return result

if __name__ == "__main__":
    # Test with a single dummy user (ensure data exists)
    user_id = 'user_01'   # replace with actual dummy user
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_mi_for_user(user_id, features)
        df.to_csv(f'mi_scores_{user_id}.csv', index=False)
        print(f"MI scores saved to mi_scores_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")