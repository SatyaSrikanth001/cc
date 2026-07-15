# run_all_methods.py
import pandas as pd
import numpy as np
import os
import time
from joblib import Parallel, delayed

# Import all per‑user compute functions from their respective scripts
from compute_mi import compute_mi_for_user
from compute_relieff import compute_relieff_for_user
from compute_laplacian import compute_laplacian_for_user
from compute_dcor import compute_dcor_for_user
from compute_fisher import compute_fisher_global
from compute_rf_louo import compute_rf_louo_for_user
from compute_xgb_louo import compute_xgb_louo_for_user
from compute_l1_stability import compute_l1_stability_for_user
from compute_permutation_importance import compute_permutation_importance_for_user
from compute_shap_ocsvm import compute_shap_for_user
from load_data import load_feature_list

# -------------------- Configuration --------------------
DATA_DIR = 'features/v2'
FEATURE_LIST_FILE = 'feature_list.txt'
N_JOBS = 4                # number of parallel jobs
N_BOOTSTRAPS = 50
RANDOM_STATE = 42
N_REPEATS_PERM = 50      # for permutation importance
SHAP_NSAMPLES = 100

# Dummy user list (25 users) – replace with real IDs later
USER_LIST = [f'user_{i:02d}' for i in range(1, 26)]
ALL_USERS = USER_LIST     # same list for LOUO methods
# -------------------------------------------------------

def process_user(user_id):
    """
    Compute all importance scores for a single user.
    Returns a merged DataFrame with all scores, or None if an error occurs.
    """
    try:
        print(f"[{user_id}] Starting...")
        features = load_feature_list(FEATURE_LIST_FILE)

        # --- Filter methods ---
        mi_df = compute_mi_for_user(user_id, features, data_dir=DATA_DIR,
                                    n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        relieff_df = compute_relieff_for_user(user_id, features, data_dir=DATA_DIR,
                                              n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        laplacian_df = compute_laplacian_for_user(user_id, features, data_dir=DATA_DIR,
                                                  k=5, n_bootstraps=N_BOOTSTRAPS,
                                                  random_state=RANDOM_STATE)
        dcor_df = compute_dcor_for_user(user_id, features, data_dir=DATA_DIR,
                                        n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)

        # Merge filter methods
        profile = mi_df.merge(relieff_df, on='feature', suffixes=('_mi', '_relieff'))
        laplacian_renamed = laplacian_df.rename(columns={
            'importance_raw': 'laplacian_raw',
            'importance_mean': 'laplacian_mean',
            'importance_std': 'laplacian_std'
        })
        profile = profile.merge(laplacian_renamed, on='feature')
        dcor_renamed = dcor_df.rename(columns={
            'dcor_raw': 'dcor_raw',
            'dcor_mean': 'dcor_mean',
            'dcor_std': 'dcor_std'
        })
        profile = profile.merge(dcor_renamed, on='feature')

        # Global Fisher is merged later outside (we'll add a function for that)
        # We'll save a temporary profile and merge Fisher later in the main loop for simplicity.
        # Actually, we can load the global Fisher CSV here (if exists) and merge.
        if os.path.exists('fisher_scores.csv'):
            fisher_df = pd.read_csv('fisher_scores.csv')
            profile = profile.merge(fisher_df[['feature', 'fisher_score', 'p_value']],
                                    on='feature', how='left')
            profile.rename(columns={'fisher_score': 'fisher_raw', 'p_value': 'fisher_pvalue'}, inplace=True)

        # --- Embedded methods ---
        rf_df = compute_rf_louo_for_user(user_id, features, ALL_USERS, data_dir=DATA_DIR,
                                         n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        rf_renamed = rf_df.rename(columns={
            'importance_raw': 'rf_raw',
            'importance_mean': 'rf_mean',
            'importance_std': 'rf_std'
        })
        profile = profile.merge(rf_renamed, on='feature')

        xgb_df = compute_xgb_louo_for_user(user_id, features, ALL_USERS, data_dir=DATA_DIR,
                                           n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        xgb_renamed = xgb_df.rename(columns={
            'importance_raw': 'xgb_raw',
            'importance_mean': 'xgb_mean',
            'importance_std': 'xgb_std'
        })
        profile = profile.merge(xgb_renamed, on='feature')

        l1_df = compute_l1_stability_for_user(user_id, features, ALL_USERS, data_dir=DATA_DIR,
                                              n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE)
        l1_renamed = l1_df.rename(columns={
            'selection_prob': 'l1_prob',
            'selection_std': 'l1_std'
        })
        profile = profile.merge(l1_renamed, on='feature')

        # --- Model‑specific methods (permutation and SHAP) ---
        try:
            perm_df = compute_permutation_importance_for_user(user_id, features, data_dir=DATA_DIR,
                                                              n_repeats=N_REPEATS_PERM,
                                                              random_state=RANDOM_STATE)
            perm_renamed = perm_df.rename(columns={
                'importance_mean': 'perm_mean',
                'importance_std': 'perm_std'
            })
            profile = profile.merge(perm_renamed, on='feature')
        except Exception as e:
            print(f"  [{user_id}] Permutation importance failed: {e}")

        try:
            shap_df = compute_shap_for_user(user_id, features, data_dir=DATA_DIR, random_state=RANDOM_STATE)
            shap_renamed = shap_df.rename(columns={'shap_importance': 'shap_importance'})
            profile = profile.merge(shap_renamed, on='feature')
        except Exception as e:
            print(f"  [{user_id}] SHAP failed: {e}")

        # Save per‑user full profile
        out_file = f'full_profile_{user_id}.csv'
        profile.to_csv(out_file, index=False)
        print(f"[{user_id}] Full profile saved to {out_file}")
        return profile

    except Exception as e:
        print(f"[{user_id}] FATAL ERROR: {e}")
        return None

def main():
    start_time = time.time()

    # 1. Compute global Fisher once (if fisher_scores.csv doesn't exist)
    if not os.path.exists('fisher_scores.csv'):
        print("Computing global Fisher scores...")
        features = load_feature_list(FEATURE_LIST_FILE)
        fisher_global = compute_fisher_global(USER_LIST, features, data_dir=DATA_DIR)
        fisher_global.to_csv('fisher_scores.csv', index=False)
        print("Fisher scores saved.")
    else:
        print("Fisher scores already exist.")

    # 2. Process all users in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_user)(user) for user in USER_LIST
    )

    elapsed = time.time() - start_time
    print(f"\nAll users processed in {elapsed:.1f} seconds")

    # 3. Generate a global summary (average raw importance across users)
    profiles = [r for r in results if r is not None]
    if not profiles:
        print("No user profiles generated.")
        return

    # Collect all raw scores (the ones ending in _raw, _mean, etc.) and average
    summary = profiles[0][['feature']].copy()
    score_cols = [c for c in profiles[0].columns if c != 'feature']
    for col in score_cols:
        vals = np.nanmean([p[col].values for p in profiles], axis=0)
        summary[f'{col}_global_mean'] = vals

    summary.to_csv('global_importance_summary.csv', index=False)
    print("Global summary saved to global_importance_summary.csv")

if __name__ == "__main__":
    main()