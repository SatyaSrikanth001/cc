# compute_laplacian.py
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import diags
from sklearn.utils import resample
import os

from load_data import load_feature_list, load_user_data

def laplacian_score(feature_vector, W, D):
    """
    Compute the Laplacian Score for a single feature.
    feature_vector: numpy array of shape (n_samples,)
    W: adjacency matrix (sparse)
    D: degree matrix (sparse diag)
    Returns scalar score (lower is better).
    """
    f = feature_vector.reshape(-1, 1)
    f_centered = f - (f.T @ D @ f) / (D.diagonal().sum())  # actually D sum is n if binary? Use formula from paper
    # proper centering: f_tilde = f - (f^T D 1) / (1^T D 1) * 1
    one = np.ones((len(f), 1))
    denominator = one.T @ D @ one
    if denominator == 0:
        return 1.0
    f_tilde = f - (f.T @ D @ one) / denominator * one
    numerator = f_tilde.T @ (D - W) @ f_tilde   # L = D - W
    denominator2 = f_tilde.T @ D @ f_tilde
    if denominator2 == 0:
        return 1.0
    return float(numerator / denominator2)

def compute_laplacian_for_user(user_id, feature_list, data_dir='features/v2',
                               k=5, n_bootstraps=50, random_state=42):
    """
    Compute Laplacian importance for all features for a single user.
    Returns DataFrame with raw importance, bootstrap mean and std.
    """
    train_df, _, common_features = load_user_data(user_id, feature_list, data_dir)
    # train_df contains only genuine sessions (label=0)
    X = train_df[common_features].astype(float).values
    n_samples = X.shape[0]

    if n_samples < k + 1:
        print(f"[{user_id}] Not enough genuine samples ({n_samples}) for k={k}. Returning zeros.")
        return pd.DataFrame({
            'feature': common_features,
            'importance_raw': 0.0,
            'importance_mean': 0.0,
            'importance_std': 0.0
        })

    # Compute raw importance on full genuine set
    # Build graph on all features? Actually we use all features for neighborhood, then score each feature individually.
    # Standard approach: use all features to build graph, then compute Laplacian score per feature.
    W = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    W = (W + W.T) / 2   # make symmetric
    D_diag = np.array(W.sum(axis=1)).flatten()
    D = diags(D_diag)

    raw_scores = {}
    for i, feat in enumerate(common_features):
        f_vec = X[:, i]
        score = laplacian_score(f_vec, W, D)
        raw_scores[feat] = 1.0 - score   # invert so high = important

    # Bootstrap
    rng = np.random.RandomState(random_state)
    bootstrap_vals = {feat: [] for feat in common_features}

    for _ in range(n_bootstraps):
        idx = resample(range(n_samples), replace=True, random_state=rng)
        X_boot = X[idx]
        W_boot = kneighbors_graph(X_boot, k, mode='connectivity', include_self=False)
        W_boot = (W_boot + W_boot.T) / 2
        D_boot_diag = np.array(W_boot.sum(axis=1)).flatten()
        D_boot = diags(D_boot_diag)
        for i, feat in enumerate(common_features):
            f_vec = X_boot[:, i]
            score = laplacian_score(f_vec, W_boot, D_boot)
            bootstrap_vals[feat].append(1.0 - score)

    importance_means = [np.mean(bootstrap_vals[feat]) if bootstrap_vals[feat] else raw_scores[feat] for feat in common_features]
    importance_stds = [np.std(bootstrap_vals[feat]) if bootstrap_vals[feat] else 0.0 for feat in common_features]

    result = pd.DataFrame({
        'feature': common_features,
        'importance_raw': [raw_scores[feat] for feat in common_features],
        'importance_mean': importance_means,
        'importance_std': importance_stds
    })

    os.makedirs('bootstrap', exist_ok=True)
    for feat in common_features:
        np.save(f"bootstrap/laplacian_{user_id}_{feat}.npy", np.array(bootstrap_vals[feat]))

    return result

if __name__ == "__main__":
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_laplacian_for_user(user_id, features)
        df.to_csv(f'laplacian_scores_{user_id}.csv', index=False)
        print(f"Laplacian scores saved to laplacian_scores_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")