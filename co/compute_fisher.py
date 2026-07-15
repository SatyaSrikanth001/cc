# compute_fisher.py
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from load_data import load_feature_list, load_user_data

def compute_fisher_global(user_list, feature_list, data_dir='features/v2'):
    """
    Compute Fisher Score for each feature across all users.
    Returns DataFrame with columns ['feature', 'fisher_score', 'p_value'].
    """
    # Load genuine training data for all users
    user_data = {}
    for user in user_list:
        train_df, _, common = load_user_data(user, feature_list, data_dir)
        # train_df has label=0 for all rows
        user_data[user] = train_df[common]

    # Align all users to the same feature set (common features already returned)
    features = common  # from last call, but all users should have the same set
    # Just to be safe, compute the intersection across all users
    feat_sets = [set(df.columns) - {'label'} for df in user_data.values()]
    common_features = sorted(set.intersection(*feat_sets))
    print(f"Common features across all users: {len(common_features)}")

    # Collect per-user statistics
    user_means = {}  # user -> Series of feature means
    user_vars = {}   # user -> Series of feature variances
    user_counts = {} # user -> number of sessions

    for user, df in user_data.items():
        data = df[common_features].astype(float)
        user_means[user] = data.mean()
        user_vars[user] = data.var(ddof=0)  # population variance within user
        user_counts[user] = len(data)

    # Global mean (weighted by number of sessions)
    total_samples = sum(user_counts.values())
    global_mean = sum(user_means[u] * user_counts[u] for u in user_list) / total_samples

    # Between-user variance
    between_var = sum(
        user_counts[u] * (user_means[u] - global_mean) ** 2
        for u in user_list
    ) / total_samples

    # Within-user variance (pooled)
    within_var = sum(
        user_counts[u] * user_vars[u]
        for u in user_list
    ) / total_samples

    # Fisher score
    fisher = between_var / (within_var + 1e-10)

    # ANOVA p-value (one-way test across the 25 groups)
    p_values = {}
    for feat in common_features:
        groups = [df[feat].values for df in user_data.values()]
        try:
            _, p = f_oneway(*groups)
        except:
            p = np.nan
        p_values[feat] = p

    # Build result DataFrame
    result = pd.DataFrame({
        'feature': common_features,
        'fisher_score': fisher,
        'p_value': [p_values[f] for f in common_features]
    })
    result = result.sort_values('fisher_score', ascending=False)
    return result

if __name__ == "__main__":
    # For testing with dummy data, we need the user list.
    # We'll generate a list of 25 dummy users.
    user_list = [f"user_{i:02d}" for i in range(1, 26)]
    features = load_feature_list('feature_list.txt')  # dummy list
    df = compute_fisher_global(user_list, features)
    df.to_csv('fisher_scores.csv', index=False)
    print("Fisher scores saved to fisher_scores.csv")
    print(df.head())