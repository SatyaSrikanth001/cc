# phase3_mrmr.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os
from typing import List, Dict
from utils.data_loader import load_and_clean_data  # reuse data loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_relevance_scores(config: dict, top100_features: List[str]) -> pd.Series:
    """
    Load median AUC values for the given features from Phase 2A output.
    Returns a Series indexed by feature name.
    """
    auc_details_path = config['output']['auc_details']
    auc_df = pd.read_csv(auc_details_path, index_col=0)
    # Filter to top100 features only
    relevance = auc_df.loc[top100_features, 'median_auc']
    logging.info(f"Loaded relevance scores for {len(relevance)} features.")
    return relevance

def compute_average_correlation(user_data: Dict[str, pd.DataFrame],
                               feature_list: List[str]) -> pd.DataFrame:
    """
    For each user, compute the absolute Pearson correlation matrix of the given features
    using all sessions (both genuine and impostor). Then average these matrices across users.
    Returns a DataFrame (feature x feature) of average absolute correlations.
    """
    sum_corr = None
    n_users = len(user_data)
    for user, df in user_data.items():
        # Subset the data to the desired features
        sub = df[feature_list].copy()
        # Ensure numeric
        sub = sub.apply(pd.to_numeric, errors='coerce')
        # Drop rows with any NaN (shouldn't be many after imputation)
        sub = sub.dropna()
        if sub.shape[0] < 5:
            logging.warning(f"User {user} has only {sub.shape[0]} complete rows; skipping correlation.")
            continue
        corr = sub.corr(method='pearson').abs()  # absolute correlation for redundancy
        if sum_corr is None:
            sum_corr = corr
        else:
            sum_corr += corr
    if sum_corr is None:
        raise RuntimeError("No valid user data to compute correlations.")
    avg_corr = sum_corr / n_users  # each user contributes equally, even if some skipped (they are not counted)
    # Ensure symmetry and fill diagonal with 0
    avg_corr.values[np.diag_indices_from(avg_corr)] = 0.0
    return avg_corr

def mrmr_greedy_selection(relevance: pd.Series,
                          redundancy: pd.DataFrame,
                          final_k: int = 40) -> List[str]:
    """
    Greedy mRMR selection (difference criterion).
    Returns list of selected feature names in order of selection.
    """
    features = relevance.index.tolist()
    # First feature: maximum relevance
    selected = [relevance.idxmax()]
    remaining = set(features) - set(selected)

    while len(selected) < final_k:
        best_score = -np.inf
        best_feat = None
        for f in remaining:
            # Redundancy term: average absolute correlation with already selected features
            red_avg = np.mean([redundancy.loc[f, s] for s in selected])
            score = relevance[f] - red_avg
            if score > best_score:
                best_score = score
                best_feat = f
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        logging.debug(f"Selected {best_feat} (score={best_score:.4f})")

    logging.info(f"Selected {len(selected)} features via mRMR.")
    return selected

def plot_final_correlation(redundancy: pd.DataFrame, selected: List[str], output_path: str):
    """Heatmap of correlations among the final selected features."""
    corr_sub = redundancy.loc[selected, selected]
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_sub, annot=False, cmap='Reds', vmin=0, vmax=1)
    plt.title('Average Absolute Correlation Among Final 40 Features')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logging.info(f"Correlation heatmap saved to {output_path}")
    plt.show()

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load top 100 features list
    elite_file = config['output']['elite_features']  # from phase2b
    if not os.path.exists(elite_file):
        raise FileNotFoundError(f"{elite_file} missing. Run phase2b_elite_rank.py first.")
    elite_df = pd.read_csv(elite_file)
    if 'feature' in elite_df.columns:
        top100 = elite_df['feature'].tolist()
    else:
        # Assume the first column is the feature name
        top100 = elite_df.iloc[:, 0].tolist()
    logging.info(f"Loaded {len(top100)} elite features for mRMR.")

    # Load relevance scores (median AUC) for these features
    relevance = load_relevance_scores(config, top100)

    # Reload cleaned user data (all features) but we'll restrict to top100 later.
    # We can load full data and then subset to top100 features for correlation.
    user_data, _ = load_and_clean_data(config)  # this returns data with all cleaned features
    # For each user, keep only the top100 features (and is_genuine? not needed for correlation)
    # But compute_average_correlation will subset internally.
    redundancy = compute_average_correlation(user_data, top100)

    # mRMR selection
    final_k = config.get('final_k', 40)
    selected = mrmr_greedy_selection(relevance, redundancy, final_k=final_k)

    # Save final list
    final_df = pd.DataFrame({'feature': selected})
    final_path = config['output'].get('final_features', 'phase3_final_40_features.csv')
    final_df.to_csv(final_path, index=False)
    logging.info(f"Final 40 features saved to {final_path}")

    # Save mRMR scores (relevance minus redundancy at time of selection)
    scores = []
    for i, f in enumerate(selected):
        if i == 0:
            score = relevance[f]  # first feature only has relevance
        else:
            red_avg = np.mean([redundancy.loc[f, s] for s in selected[:i]])
            score = relevance[f] - red_avg
        scores.append(score)
    scores_df = pd.DataFrame({'feature': selected, 'mrmr_score': scores})
    scores_path = config['output'].get('mrmr_scores', 'phase3_mrmr_scores.csv')
    scores_df.to_csv(scores_path, index=False)
    logging.info(f"mRMR scores saved to {scores_path}")

    # Plot correlation heatmap of final 40
    plot_final_correlation(redundancy, selected, 'phase3_final_correlation.png')

if __name__ == "__main__":
    main()
