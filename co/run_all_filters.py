# run_all_filters.py
import pandas as pd
import numpy as np
import os
import time
from joblib import Parallel, delayed

# Import the compute functions from the respective scripts
# (Ensure they are in the same directory and the functions are defined)
from compute_mi import compute_mi_for_user
from compute_relieff import compute_relieff_for_user
from compute_laplacian import compute_laplacian_for_user
from compute_dcor import compute_dcor_for_user
from load_data import load_feature_list

# -------------------- Configuration --------------------
DATA_DIR = 'features/v2'
FEATURE_LIST_FILE = 'feature_list.txt'
FISHER_FILE = 'fisher_scores.csv'
N_JOBS = 4   # number of parallel jobs (adjust based on CPU)
N_BOOTSTRAPS = 50
RANDOM_STATE = 42

# Dummy user list (25 users) – replace with real IDs later
USER_LIST = [f'user_{i:02d}' for i in range(1, 26)]

# -------------------------------------------------------

def process_user(user_id):
    """
    Compute all filter importance scores for a single user.
    Returns a DataFrame with all scores merged, or None if an error occurs.
    """
    try:
        print(f"[{user_id}] Starting...")
        # Load feature list
        features = load_feature_list(FEATURE_LIST_FILE)

        # Compute per-user methods
        mi_df = compute_mi_for_user(user_id, features, data_dir=DATA_DIR,
                                    n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        relieff_df = compute_relieff_for_user(user_id, features, data_dir=DATA_DIR,
                                              n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        laplacian_df = compute_laplacian_for_user(user_id, features, data_dir=DATA_DIR,
                                                  k=5, n_bootstraps=N_BOOTSTRAPS,
                                                  random_state=RANDOM_STATE)
        dcor_df = compute_dcor_for_user(user_id, features, data_dir=DATA_DIR,
                                        n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)

        # Merge all on 'feature'
        profile = mi_df.merge(relieff_df, on='feature', suffixes=('_mi', '_relieff'))
        profile = profile.merge(laplacian_df, on='feature')   # laplacian_df columns: feature, importance_raw, importance_mean, importance_std
        # rename laplacian columns to avoid confusion
        profile.rename(columns={'importance_raw': 'laplacian_raw',
                                'importance_mean': 'laplacian_mean',
                                'importance_std': 'laplacian_std'}, inplace=True)
        profile = profile.merge(dcor_df, on='feature', suffixes=('', '_dcor'))
        # dcor_df columns: feature, dcor_raw, dcor_mean, dcor_std
        profile.rename(columns={'dcor_raw': 'dcor_raw', 'dcor_mean': 'dcor_mean', 'dcor_std': 'dcor_std'}, inplace=True)

        # Merge Fisher (global)
        if os.path.exists(FISHER_FILE):
            fisher_df = pd.read_csv(FISHER_FILE)
            profile = profile.merge(fisher_df[['feature', 'fisher_score', 'p_value']], on='feature', how='left')
            profile.rename(columns={'fisher_score': 'fisher_score', 'p_value': 'fisher_pvalue'}, inplace=True)
        else:
            print(f"Warning: {FISHER_FILE} not found, skipping Fisher.")

        # Save per-user CSV
        output_file = f'importance_profile_{user_id}.csv'
        profile.to_csv(output_file, index=False)
        print(f"[{user_id}] Done. Profile saved to {output_file}")
        return profile

    except Exception as e:
        print(f"[{user_id}] ERROR: {e}")
        return None

def main():
    start_time = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_user)(user) for user in USER_LIST
    )
    elapsed = time.time() - start_time
    print(f"\nAll users processed in {elapsed:.1f} seconds")

    # Generate a global summary (average raw scores across users)
    # Collect all profiles (skip None)
    profiles = [r for r in results if r is not None]
    if profiles:
        # For each method, compute mean of raw scores across users
        summary = profiles[0][['feature']].copy()
        methods = ['mi_raw', 'weight_raw', 'laplacian_raw', 'dcor_raw', 'fisher_score']
        for method in methods:
            if method in profiles[0].columns:
                # average across all profiles
                vals = np.mean([p[method].values for p in profiles], axis=0)
                summary[f'{method}_global_mean'] = vals

        summary.to_csv('global_importance_summary.csv', index=False)
        print("Global summary saved to global_importance_summary.csv")

if __name__ == "__main__":
    main()