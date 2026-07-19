# phase5_ocsvm_validation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import yaml
import logging
import os
from typing import Dict, List, Tuple

# Reuse the data loader and Phase 2A AUC computation from our existing modules
from utils.data_loader import load_and_clean_data
from phase2a_noise_purge import compute_aucs_per_user   # we need the function to rank features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------- Helper functions -----------------------
def compute_eer(y_true, scores):
    """Compute Equal Error Rate (EER) given true labels (0=impostor, 1=genuine) and decision scores."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # EER is where fpr = 1 - tpr
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

def get_baseline_a_features(user_data: Dict[str, pd.DataFrame], all_features: List[str],
                            top_k: int = 40) -> List[str]:
    """
    Rank all features by median within‑device AUC (genuine vs impostor)
    using the same method as Phase 2B, but applied to the full feature set
    (no noise purge). Returns the top K features.
    """
    # Compute per‑user AUCs for all features
    auc_df = compute_aucs_per_user(user_data, all_features)
    agg = auc_df.groupby('feature')['auc'].median().sort_values(ascending=False)
    baseline_a = agg.head(top_k).index.tolist()
    logging.info(f"Baseline A: top {len(baseline_a)} features selected by median AUC (unfiltered).")
    return baseline_a

def get_baseline_b_features(all_features: List[str], top_k: int = 40, random_state: int = 42) -> List[str]:
    """Randomly select K features from the full list."""
    rng = np.random.RandomState(random_state)
    selected = rng.choice(all_features, size=min(top_k, len(all_features)), replace=False).tolist()
    logging.info(f"Baseline B: randomly selected {len(selected)} features.")
    return selected

def evaluate_ocsvm_for_user(user_id: str,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            features: List[str],
                            nu: float = 0.1,
                            gamma: str = 'scale',
                            threshold: float = 0.0) -> Tuple[float, float]:
    """
    Train OCSVM on the user's genuine training data, evaluate on test data.
    Returns (EER, AUC).
    """
    # Subset features that exist
    available_features = [f for f in features if f in train_df.columns and f in test_df.columns]
    if len(available_features) == 0:
        return np.nan, np.nan

    X_train = train_df[available_features].values.astype(float)
    X_test = test_df[available_features].values.astype(float)
    y_test = test_df['is_genuine'].values  # 1=genuine, 0=impostor

    # Impute any leftover NaNs (safety)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train OCSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    ocsvm.fit(X_train_scaled)

    # Decision scores
    scores = ocsvm.decision_function(X_test_scaled)

    # EER
    eer = compute_eer(y_test, scores)

    # AUC (higher score for genuine)
    try:
        auc_val = roc_auc_score(y_test, scores)
    except ValueError:
        auc_val = np.nan

    return eer, auc_val

def run_validation(config: dict):
    """Main validation routine."""
    # Load cleaned data (basic cleaning only, all features)
    user_data, all_features = load_and_clean_data(config)
    users = list(user_data.keys())
    logging.info(f"Loaded data for {len(users)} users with {len(all_features)} features.")

    # --- Feature sets to evaluate ---
    # Our final set (Phase 4 consensus top 40)
    final_features_path = config['output'].get('final_selected_features',
                                               'phase4_final_selected_features.csv')
    if not os.path.exists(final_features_path):
        # Fallback to Phase 3 output if Phase 4 not run
        final_features_path = config['output'].get('final_features', 'phase3_final_40_features.csv')
    if not os.path.exists(final_features_path):
        raise FileNotFoundError("No final feature set found. Run Phase 3 or Phase 4 first.")
    final_features = pd.read_csv(final_features_path)['feature'].tolist()
    logging.info(f"Final feature set: {len(final_features)} features.")

    # Baseline A: top 40 by median AUC on full (unfiltered) feature set
    baseline_a_features = get_baseline_a_features(user_data, all_features, top_k=40)

    # Baseline B: random 40 from all features
    baseline_b_features = get_baseline_b_features(all_features, top_k=40)

    # Prepare results storage
    results = {'user': [], 'method': [], 'eer': [], 'auc': []}

    # Train/Test split ratio for genuine sessions
    test_ratio = 0.2

    # Iterate over users
    for user_id, df in user_data.items():
        genuine = df[df['is_genuine'] == 1]
        impostor = df[df['is_genuine'] == 0]

        if len(genuine) < 5:
            logging.warning(f"User {user_id} has only {len(genuine)} genuine sessions; skipping.")
            continue

        # Split genuine sessions into train and test
        train_gen, test_gen = train_test_split(genuine, test_size=test_ratio, random_state=42)

        # Test set: remaining genuine + all impostor
        test_df = pd.concat([test_gen, impostor], ignore_index=True)

        # Evaluate each method
        for method_name, features in [('Final 40', final_features),
                                      ('Baseline A', baseline_a_features),
                                      ('Baseline B', baseline_b_features)]:
            eer, auc_val = evaluate_ocsvm_for_user(user_id, train_gen, test_df, features)
            results['user'].append(user_id)
            results['method'].append(method_name)
            results['eer'].append(eer)
            results['auc'].append(auc_val)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save per‑user metrics
    results_path = config['output'].get('eer_results', 'phase5_eer_results.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"EER results saved to {results_path}")

    # ---- Aggregated statistics ----
    agg = results_df.groupby('method').agg(
        median_eer=('eer', 'median'),
        mean_eer=('eer', 'mean'),
        std_eer=('eer', 'std'),
        median_auc=('auc', 'median'),
        mean_auc=('auc', 'mean'),
        std_auc=('auc', 'std')
    ).reset_index()
    print("\n--- Aggregate Validation Metrics ---")
    print(agg.to_string(index=False))
    agg.to_csv('phase5_summary_metrics.csv', index=False)

    # ---- Boxplots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=results_df, x='method', y='eer', ax=axes[0])
    axes[0].set_title('Equal Error Rate (EER) per method')
    axes[0].set_ylabel('EER')
    sns.boxplot(data=results_df, x='method', y='auc', ax=axes[1])
    axes[1].set_title('ROC‑AUC per method')
    axes[1].set_ylabel('AUC')
    plt.tight_layout()
    boxplot_path = config['output'].get('validation_boxplot', 'phase5_validation_boxplots.png')
    plt.savefig(boxplot_path, dpi=150)
    logging.info(f"Boxplots saved to {boxplot_path}")
    plt.show()

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_validation(config)
