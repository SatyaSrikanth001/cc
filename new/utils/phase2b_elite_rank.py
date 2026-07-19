# phase2b_elite_rank.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_phase2a_results(config: dict) -> (pd.DataFrame, List[str]):
    """
    Load the survivors list and the AUC details from Phase 2A.

    Returns
    -------
    auc_df : pd.DataFrame
        Index = feature name, columns = ['median_auc', 'std_auc', 'num_users', ...]
    survivors : list
        List of feature names that passed the noise threshold.
    """
    survivors_path = config['output']['filtered_features']
    auc_details_path = config['output']['auc_details']

    if not os.path.exists(survivors_path) or not os.path.exists(auc_details_path):
        raise FileNotFoundError("Phase 2A outputs missing. Run phase2a_noise_purge.py first.")

    survivors = pd.read_csv(survivors_path)['feature'].tolist()
    auc_df = pd.read_csv(auc_details_path, index_col=0)  # assuming feature is the index

    # Ensure we only keep survivors
    auc_df = auc_df.loc[survivors]
    logging.info(f"Loaded {len(auc_df)} survivors from Phase 2A.")
    return auc_df, survivors

def rank_features(auc_df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """
    Rank features by median AUC (descending). Use standard deviation as tie‑breaker
    (lower std is better). Return the top K features with their rankings.

    Parameters
    ----------
    auc_df : pd.DataFrame
        Must contain columns 'median_auc' and 'std_auc'.
    top_k : int
        Number of top features to select.

    Returns
    -------
    ranked_df : pd.DataFrame
        Original DataFrame with added 'rank' and sorted by rank.
    """
    # Sort by median_auc descending, then std_auc ascending
    sorted_df = auc_df.sort_values(by=['median_auc', 'std_auc'],
                                   ascending=[False, True])
    sorted_df['rank'] = range(1, len(sorted_df) + 1)
    top_features = sorted_df.head(top_k)
    logging.info(f"Selected top {top_k} features from {len(sorted_df)} survivors.")
    return top_features

def plot_top_features(top_features: pd.DataFrame, output_path: str = 'phase2b_top20.png'):
    """
    Generate a horizontal bar plot of the top 20 features with median AUC and error bars (std).
    """
    plt.figure(figsize=(12, 8))
    # Take top 20 from the already sorted top_features
    top20 = top_features.head(20)
    features = top20.index
    median_auc = top20['median_auc']
    std_auc = top20['std_auc']

    plt.barh(range(len(features)), median_auc, xerr=std_auc, color='steelblue', alpha=0.8)
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()  # highest at top
    plt.xlabel('Median ROC‑AUC (Genuine vs Impostor)')
    plt.title('Top 20 Features by Within‑Device Discriminability')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logging.info(f"Top 20 features plot saved to {output_path}")
    plt.show()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load Phase 2A survivors and AUC details
    auc_df, survivors = load_phase2a_results(config)
    top_k = config.get('top_k', 100)  # from config.yaml, default 100

    # Rank and select top K
    elite_df = rank_features(auc_df, top_k=top_k)

    # Save the elite list
    elite_list_path = config['output'].get('elite_features', 'phase2b_top100_features.csv')
    elite_df[['rank']].to_csv(elite_list_path)  # save feature names and rank
    logging.info(f"Elite features saved to {elite_list_path}")

    # Save full ranked list (optional)
    ranked_full_path = config['output'].get('behavioral_ranking', 'phase2b_ranking.csv')
    auc_df_sorted = auc_df.loc[elite_df.index]  # keep only elite? No, we want full ranking for later phases. Let's save full ranking of survivors.
    full_ranked = auc_df.sort_values(by=['median_auc', 'std_auc'], ascending=[False, True])
    full_ranked['rank'] = range(1, len(full_ranked) + 1)
    full_ranked.to_csv(ranked_full_path)
    logging.info(f"Full behavioral ranking saved to {ranked_full_path}")

    # Plot top 20
    plot_top_features(elite_df)

if __name__ == "__main__":
    main()
