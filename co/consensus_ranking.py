# consensus_ranking.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------
INPUT_FILE = 'global_importance_summary.csv'
OUTPUT_FILE = 'consensus_ranking.csv'
# List of columns to include in the consensus (should exist in the input)
# These correspond to the global mean of each method's raw importance.
METRIC_COLUMNS = [
    'mi_raw_global_mean',
    'weight_raw_global_mean',   # ReliefF weight
    'laplacian_raw_global_mean',
    'dcor_raw_global_mean',
    'fisher_raw',
    'rf_raw_global_mean',
    'xgb_raw_global_mean',
    'l1_prob_global_mean',
    'perm_mean_global_mean',
    'shap_importance_global_mean'
]
# Optional: weights for each method (sum doesn't need to be 1; will be normalized internally)
# Set all to 1 for equal weighting.
METHOD_WEIGHTS = {
    'mi_raw_global_mean': 1.0,
    'weight_raw_global_mean': 1.0,
    'laplacian_raw_global_mean': 1.0,
    'dcor_raw_global_mean': 1.0,
    'fisher_raw': 1.0,
    'rf_raw_global_mean': 1.0,
    'xgb_raw_global_mean': 1.0,
    'l1_prob_global_mean': 1.0,
    'perm_mean_global_mean': 1.0,
    'shap_importance_global_mean': 1.0
}
# --------------------------------------------------------

def load_global_summary(filepath):
    """Load the global summary and verify columns."""
    df = pd.read_csv(filepath)
    missing = [col for col in METRIC_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input: {missing}")
    return df

def compute_percentile_ranks(df, columns):
    """
    Convert each metric column to a percentile rank (0-1).
    Higher raw score → higher percentile.
    """
    ranks = df.copy()
    for col in columns:
        # Use rank with pct=True; methods that produce exactly 0 for all values will get NaN ranks.
        ranks[col + '_pct'] = df[col].rank(pct=True)
    return ranks

def compute_weighted_apr(df, columns, weights_dict):
    """
    Compute weighted Average Percentile Rank.
    Returns a Series of consensus scores.
    """
    pct_cols = [col + '_pct' for col in columns]
    weights = np.array([weights_dict.get(col, 1.0) for col in columns])
    weights = weights / weights.sum()  # normalize
    apr = np.average(df[pct_cols].values, axis=1, weights=weights)
    return apr

def main():
    df = load_global_summary(INPUT_FILE)
    
    # 1. Percentile transformation
    df_pct = compute_percentile_ranks(df, METRIC_COLUMNS)
    
    # 2. Weighted APR
    df_pct['apr_score'] = compute_weighted_apr(df_pct, METRIC_COLUMNS, METHOD_WEIGHTS)
    df_pct['apr_rank'] = df_pct['apr_score'].rank(ascending=False, method='dense').astype(int)
    
    # 3. Sort and save
    result = df_pct.sort_values('apr_rank')
    # Keep feature, apr_score, apr_rank, and all percentile columns for transparency
    out_cols = ['feature', 'apr_score', 'apr_rank'] + [c + '_pct' for c in METRIC_COLUMNS]
    result[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"Consensus ranking saved to {OUTPUT_FILE}")
    print("Top 10 features by APR:")
    print(result[['feature', 'apr_score', 'apr_rank']].head(10))
    
    # 4. Visualization
    top_n = min(20, len(result))
    top_features = result.head(top_n)
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_features['apr_score'], color='steelblue')
    plt.yticks(range(top_n), top_features['feature'])
    plt.gca().invert_yaxis()
    plt.xlabel('Average Percentile Rank (APR)')
    plt.title('Top 20 Features by Consensus APR')
    plt.tight_layout()
    plt.savefig('consensus_top20.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()