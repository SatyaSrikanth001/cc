# phase4_consensus.py
import pandas as pd
import numpy as np
import yaml
import logging
import os
from typing import Dict, List
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from skrebate import ReliefF
from scipy.sparse import diags
from sklearn.neighbors import kneighbors_graph

from utils.data_loader import load_and_clean_data   # re‑use the same data loading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------- Safe filter methods per user -----------------------
def within_user_fisher(user_df: pd.DataFrame, feature: str) -> float:
    """
    Fisher discriminant ratio for one user.
    user_df must have 'is_genuine' column.
    """
    genuine = user_df.loc[user_df['is_genuine'] == 1, feature].dropna()
    impostor = user_df.loc[user_df['is_genuine'] == 0, feature].dropna()
    if len(genuine) < 2 or len(impostor) < 2:
        return 0.0
    mu_g = genuine.mean()
    mu_i = impostor.mean()
    var_g = genuine.var(ddof=0)
    var_i = impostor.var(ddof=0)
    denom = var_g + var_i
    if denom == 0:
        return 0.0
    return (mu_g - mu_i) ** 2 / denom

def within_user_mi(user_df: pd.DataFrame, feature: str) -> float:
    """Mutual information between feature and is_genuine for one user."""
    # Drop rows where feature is NaN
    temp = user_df[[feature, 'is_genuine']].dropna()
    if temp.shape[0] < 2 or temp['is_genuine'].nunique() < 2:
        return 0.0
    X = temp[[feature]].values
    y = temp['is_genuine'].values
    mi = mutual_info_classif(X, y, random_state=0)
    return mi[0]

def within_user_relieff(user_df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
    """ReliefF weights for all features. Returns array of weights."""
    temp = user_df[feature_list + ['is_genuine']].dropna()
    if temp.shape[0] < 2 or temp['is_genuine'].nunique() < 2:
        return np.zeros(len(feature_list))
    X = temp[feature_list].values
    y = temp['is_genuine'].values.astype(int)
    try:
        fs = ReliefF(n_neighbors=10, n_features_to_select=len(feature_list), n_iterations=100)
        fs.fit(X, y)
        return fs.feature_importances_
    except:
        return np.zeros(len(feature_list))

def within_user_laplacian(user_df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
    """
    Laplacian importance (1 - score) per feature, using genuine sessions only.
    Returns array of importance values (higher = better).
    """
    genuine = user_df[user_df['is_genuine'] == 1][feature_list].dropna()
    if genuine.shape[0] < 5:
        return np.zeros(len(feature_list))
    X = genuine.values
    k = min(5, genuine.shape[0] - 1)
    W = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    W = (W + W.T) / 2
    D_diag = np.array(W.sum(axis=1)).flatten()
    D = diags(D_diag)
    importance = []
    for i in range(X.shape[1]):
        f = X[:, i]
        # Laplacian score
        f_centered = f - f.mean()
        numerator = f_centered @ (D - W) @ f_centered
        denominator = f_centered @ D @ f_centered
        score = numerator / denominator if denominator != 0 else 1.0
        importance.append(1.0 - score)
    return np.array(importance)

# ----------------------- Main script -----------------------
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load survivors from Phase 2A
    survivors_path = config['output']['filtered_features']
    survivors = pd.read_csv(survivors_path)['feature'].tolist()
    logging.info(f"Survivors from Phase 2A: {len(survivors)} features")

    # Load cleaned user data (all features, but we will subset to survivors)
    user_data, _ = load_and_clean_data(config)

    # 1. Compute per‑user scores for each method
    users = list(user_data.keys())
    # Initialize DataFrames to store per‑user scores for each feature
    fisher_scores = pd.DataFrame(index=survivors, columns=users, dtype=float)
    mi_scores = pd.DataFrame(index=survivors, columns=users, dtype=float)
    relieff_scores = pd.DataFrame(index=survivors, columns=users, dtype=float)
    laplacian_scores = pd.DataFrame(index=survivors, columns=users, dtype=float)

    for user, df in user_data.items():
        # Subset to survivors
        sub = df[survivors + ['is_genuine']].copy()
        # Compute Fisher and MI per feature
        for feat in survivors:
            fisher_scores.loc[feat, user] = within_user_fisher(sub, feat)
            mi_scores.loc[feat, user] = within_user_mi(sub, feat)
        # ReliefF (all features at once)
        rf_weights = within_user_relieff(sub, survivors)
        relieff_scores[user] = rf_weights
        # Laplacian
        lap_weights = within_user_laplacian(sub, survivors)
        laplacian_scores[user] = lap_weights

    # 2. Aggregate to median per feature across users
    def median_agg(scores_df):
        return scores_df.median(axis=1)

    fisher_median = median_agg(fisher_scores)
    mi_median = median_agg(mi_scores)
    relieff_median = median_agg(relieff_scores)
    laplacian_median = median_agg(laplacian_scores)

    # 3. Load Phase 2B ranking (which is based on median AUC)
    # We'll load the full behavioral ranking that contains all survivors with a 'rank' column.
    ranking_path = config['output']['behavioral_ranking']
    rank_df = pd.read_csv(ranking_path, index_col=0)  # index = feature
    # Ensure we have rank column
    if 'rank' not in rank_df.columns:
        # The ranking file from phase2b_elite_rank saved with feature as index and a rank column,
        # but we might need to create rank if not present. We'll just read the order.
        # Actually phase2b_elite_rank saved full_ranked with rank column. Good.
        pass
    our_rank = rank_df.loc[survivors, 'rank'].copy()

    # 4. Convert each median score to a rank (1 = best, higher score = better)
    def score_to_rank(series):
        # Sort descending (higher score gets rank 1)
        return series.rank(ascending=False, method='min').astype(int)

    fisher_rank = score_to_rank(fisher_median)
    mi_rank = score_to_rank(mi_median)
    relieff_rank = score_to_rank(relieff_median)
    laplacian_rank = score_to_rank(laplacian_median)

    # 5. Weighted average rank
    w_our = 2
    w_others = 1
    total_weight = w_our + 4 * w_others  # = 2 + 1+1+1+1 = 6
    consensus_rank = (w_our * our_rank + w_others * (fisher_rank + mi_rank + relieff_rank + laplacian_rank)) / total_weight

    # Sort by consensus rank ascending (lower = better)
    final_rank_df = pd.DataFrame({
        'feature': survivors,
        'consensus_rank_val': consensus_rank
    }).sort_values('consensus_rank_val')

    final_rank_df['final_rank'] = range(1, len(final_rank_df) + 1)

    # Save full consensus ranking
    consensus_path = config['output'].get('consensus_ranking', 'phase4_consensus_ranking.csv')
    final_rank_df.to_csv(consensus_path, index=False)
    logging.info(f"Consensus ranking saved to {consensus_path}")

    # 6. Select top 40
    final_k = config.get('final_k', 40)
    top_final = final_rank_df.head(final_k)['feature'].tolist()
    final_list_path = config['output'].get('final_selected_features', 'phase4_final_selected_features.csv')
    pd.DataFrame({'feature': top_final}).to_csv(final_list_path, index=False)
    logging.info(f"Final selected {len(top_final)} features saved to {final_list_path}")

    # Print top features
    print("\nTop 10 features by consensus:")
    print(final_rank_df.head(10))

if __name__ == "__main__":
    main()
