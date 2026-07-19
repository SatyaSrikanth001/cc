# phase2a_noise_purge.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import yaml
import logging
import os
from typing import Dict, List
from utils.data_loader import load_and_clean_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_aucs_per_user(user_data: Dict[str, pd.DataFrame], feature_list: List[str]) -> pd.DataFrame:
    """
    For each user and each feature, compute the ROC‑AUC of the feature to predict is_genuine.
    Returns a DataFrame with columns: feature, user, auc.
    """
    records = []
    for user, df in user_data.items():
        y = df['is_genuine'].values
        if len(np.unique(y)) < 2:
            # Only one class present → AUC is undefined, treat as 0.5
            for feat in feature_list:
                records.append({'feature': feat, 'user': user, 'auc': 0.5})
            continue
        for feat in feature_list:
            x = df[feat].values
            # If feature is constant, AUC = 0.5
            if np.std(x) == 0:
                records.append({'feature': feat, 'user': user, 'auc': 0.5})
            else:
                # Use the feature as score (higher → more likely genuine)
                # We'll compute AUC assuming genuine=1. If the feature scores are reversed,
                # roc_auc_score will still give a value >=0.5 because it can flip the sign.
                try:
                    auc = roc_auc_score(y, x)
                except ValueError:
                    auc = 0.5
                records.append({'feature': feat, 'user': user, 'auc': auc})
    return pd.DataFrame(records)

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load and clean data
    user_data, feature_list = load_and_clean_data(config)
    logging.info(f"Loaded {len(user_data)} users with {len(feature_list)} features.")

    # Compute AUCs
    auc_df = compute_aucs_per_user(user_data, feature_list)
    logging.info("Per‑user AUC computation complete.")

    # Aggregate: median AUC per feature
    agg = auc_df.groupby('feature')['auc'].agg(['median', 'std', 'count', 'min', 'max'])
    agg.columns = ['median_auc', 'std_auc', 'num_users', 'min_auc', 'max_auc']

    # Apply noise threshold
    noise_threshold = config['noise_purge']['noise_threshold']
    survivors = agg[agg['median_auc'] >= noise_threshold].index.tolist()
    dropped = agg[agg['median_auc'] < noise_threshold].index.tolist()

    logging.info(f"Surviving features (median AUC >= {noise_threshold}): {len(survivors)}")
    logging.info(f"Dropped features (noise): {len(dropped)}")

    # Save survivors list
    pd.Series(survivors, name='feature').to_csv(
        config['output']['filtered_features'], index=False
    )
    logging.info(f"Survivors saved to {config['output']['filtered_features']}")

    # Save detailed AUC stats
    agg.to_csv(config['output']['auc_details'])
    logging.info(f"AUC details saved to {config['output']['auc_details']}")

    # Optionally print a quick summary
    print("\n--- Top 10 features by median AUC ---")
    print(agg.sort_values('median_auc', ascending=False).head(10))

if __name__ == "__main__":
    main()
