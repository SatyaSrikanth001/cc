# k_selection_consensus.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import time
from joblib import Parallel, delayed
from load_data import load_feature_list, load_user_data

# -------------------- Configuration --------------------
DATA_DIR = 'features/v2'
FEATURE_LIST_FILE = 'feature_list.txt'      # your actual feature list
CONSENSUS_FILE = 'consensus_ranking.csv'    # from Script 14
USER_LIST = [f'user_{i:02d}' for i in range(1, 26)]  # 25 dev users
K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # feature set sizes to test
OCSVM_NU = 0.1
OCSVM_GAMMA = 'scale'          # 'scale' or a float
THRESHOLD = 0.0                # decision threshold
FAR_LIMIT = 0.005              # maximum acceptable FAR for selection
N_JOBS = 4                     # parallel jobs for LOUO inner loop? we'll do sequential per K for clarity
# --------------------------------------------------------

def load_consensus_ranking(filepath):
    """Load and return the sorted feature list from consensus ranking."""
    df = pd.read_csv(filepath)
    # assume the file is sorted by 'apr_rank' or 'apr_score' descending
    if 'apr_rank' in df.columns:
        df = df.sort_values('apr_rank')
    elif 'apr_score' in df.columns:
        df = df.sort_values('apr_score', ascending=False)
    features = df['feature'].tolist()
    return features

def evaluate_user(user_id, selected_features, data_dir=DATA_DIR):
    """
    Train OCSVM on user's genuine training data (with given features) and evaluate on test set.
    Returns TAR, FRR, FAR, Accuracy.
    """
    train_df, test_df, _ = load_user_data(user_id, selected_features, data_dir)
    # Ensure we have both genuine and impostor in test
    y_test = test_df['label'].values  # 0 genuine, 1 impostor
    # Only keep the selected features
    common_features = [f for f in selected_features if f in train_df.columns and f in test_df.columns]
    if len(common_features) == 0:
        return None
    X_train = train_df[common_features].astype(float).values
    X_test = test_df[common_features].astype(float).values

    # Imputation (should be done already, but safety)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Scale on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train OCSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma=OCSVM_GAMMA)
    ocsvm.fit(X_train_scaled)

    # Decision scores
    scores = ocsvm.decision_function(X_test_scaled)
    pred = (scores >= THRESHOLD).astype(int)  # 1 = predicted genuine, 0 = impostor

    # Compute metrics
    genuine_mask = (y_test == 0)
    impostor_mask = (y_test == 1)
    frr = 1.0 - pred[genuine_mask].mean() if genuine_mask.sum() > 0 else 0.0
    far = pred[impostor_mask].mean() if impostor_mask.sum() > 0 else 0.0
    tar = 1.0 - frr
    acc = accuracy_score(1 - y_test, pred)  # align labels: we predict 1 for genuine (label 0)
    return tar, frr, far, acc

def evaluate_k(k, ranked_features, user_list):
    """
    For a given K, select top-K features, evaluate all users, return average metrics.
    """
    selected = ranked_features[:k]
    tars, frrs, fars, accs = [], [], [], []
    for user in user_list:
        metrics = evaluate_user(user, selected, DATA_DIR)
        if metrics is not None:
            tar, frr, far, acc = metrics
            tars.append(tar)
            frrs.append(frr)
            fars.append(far)
            accs.append(acc)
    if not tars:
        return None
    return np.mean(tars), np.mean(frrs), np.mean(fars), np.mean(accs), np.max(fars)

def main():
    # Load consensus ranking
    if not os.path.exists(CONSENSUS_FILE):
        raise FileNotFoundError(f"{CONSENSUS_FILE} not found. Run consensus_ranking.py first.")
    ranked_features = load_consensus_ranking(CONSENSUS_FILE)
    print(f"Loaded {len(ranked_features)} features from consensus ranking.")

    # Evaluate each K
    results = []
    for k in K_VALUES:
        print(f"Evaluating K = {k}...")
        res = evaluate_k(k, ranked_features, USER_LIST)
        if res is None:
            print(f"  K={k} failed (no valid users).")
            continue
        avg_tar, avg_frr, avg_far, avg_acc, max_far = res
        results.append((k, avg_tar, avg_frr, avg_far, avg_acc, max_far))
        print(f"  K={k:3d}: TAR={avg_tar:.4f}, FRR={avg_frr:.4f}, FAR={avg_far:.4f}, Acc={avg_acc:.4f}, MaxFAR={max_far:.4f}")

    # Save results
    res_df = pd.DataFrame(results, columns=['K', 'TAR', 'FRR', 'FAR', 'Accuracy', 'MaxFAR'])
    res_df.to_csv('k_selection_results.csv', index=False)
    print("\nResults saved to k_selection_results.csv")

    # Choose best K (highest TAR with MaxFAR <= FAR_LIMIT)
    valid = res_df[res_df['MaxFAR'] <= FAR_LIMIT]
    if not valid.empty:
        best = valid.loc[valid['TAR'].idxmax()]
        print(f"\nBest K = {int(best['K'])} with TAR={best['TAR']:.4f}, FAR={best['FAR']:.4f}")
    else:
        print("\nNo K satisfies FAR limit; consider adjusting threshold or nu.")

    # Plot TAR vs K
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['K'], res_df['TAR'], marker='o', label='TAR', color='green')
    plt.plot(res_df['K'], res_df['FAR'], marker='x', label='FAR', color='red')
    plt.axhline(y=FAR_LIMIT, color='gray', linestyle='--', label=f'FAR limit ({FAR_LIMIT})')
    plt.xlabel('Number of Features (K)')
    plt.ylabel('Rate')
    plt.title('OCSVM Performance vs. Number of Top Consensus Features')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('k_selection_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()