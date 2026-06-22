#!/usr/bin/env python3
"""
Enhanced analysis: find which features make impostors score above the threshold,
for ALL users at once. Generates per‑user rankings and a global ranking.

Usage: python analyze_all_users_v2.py [threshold]
       If threshold is not given, 0.0 is used.
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

from train_session import SessionOCSVM

warnings.filterwarnings('ignore')

# ======================= CONFIGURATION =======================
MODELS_DIR = "models"
FEATURES_DIR = "./features/v2"

# List of all users – adjust to your actual users
USER_LIST = [
    'samanth', 'vinay', 'reddy123', 'harshit', 'Bhargav128',
    'Nikitha18', 'Pranav', 'surya_mangam30', 'gunja', 'skyy',
    'uday', 'visha117', 'tiwari', 'Pratik', 'amishp', 'Diana', 'sarya'
]
# =============================================================

def load_user_artifacts(user_id):
    """
    Load the trained model, scaler, and the preprocessed data
    exactly as used during training. No CSV re‑reading.
    """
    model_path = os.path.join(MODELS_DIR, f"{user_id}_ocsvm.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{user_id}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler missing for {user_id}.")

    ocsvm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    train_csv = os.path.join(FEATURES_DIR, f"{user_id}_training_sessions.csv")
    test_csv  = os.path.join(FEATURES_DIR, f"{user_id}_testing_sessions.csv")

    # Use the exact same loading logic as training – returns clean arrays & DataFrames
    session_model = SessionOCSVM(user_id)
    X_train_raw, X_test_raw, y_test, train_df_proc, test_df_proc = \
        session_model.load_and_prepare_data(train_csv, test_csv)

    feature_cols = session_model.feature_columns

    # train_df_proc already has only feature_cols and NaN filled (by nan_to_num)
    # test_df_proc same
    return ocsvm, scaler, train_df_proc, test_df_proc, y_test, feature_cols


def analyze_one_user(user_id, threshold):
    """Return a DataFrame of features ranked by how similar impostors are to genuine."""
    try:
        ocsvm, scaler, train_df, test_df, y_test, feature_cols = load_user_artifacts(user_id)
    except Exception as e:
        print(f"  Skipping {user_id}: {e}")
        return None, None

    # Scale the test data using the saved scaler (fitted on training data)
    X_test = test_df.values.astype(float)
    X_test_scaled = scaler.transform(X_test)
    scores = ocsvm.decision_function(X_test_scaled)

    imp_mask = (y_test == -1)      # impostor
    high_score_imp = imp_mask & (scores >= threshold)

    print(f"  [{user_id}] impostors above threshold ({threshold}): "
          f"{high_score_imp.sum()}/{imp_mask.sum()}")

    if high_score_imp.sum() == 0:
        return pd.DataFrame(), 0   # empty DataFrame, zero high‑score count

    # Genuine training statistics (on the original feature values)
    gen_mean = train_df.mean()
    gen_std = train_df.std()

    # High‑scoring impostor subset (original values)
    imp_high = test_df.loc[high_score_imp]
    imp_high_mean = imp_high.mean()

    # z‑diff: how many standard deviations the impostor mean is from the genuine mean
    with np.errstate(divide='ignore', invalid='ignore'):
        z_diff = (imp_high_mean - gen_mean).abs() / (gen_std + 1e-8)
    z_diff = z_diff.replace(np.inf, np.nan).fillna(0)

    result = pd.DataFrame({
        'feature': z_diff.index,
        'z_diff': z_diff.values,
        'impostor_high_mean': imp_high_mean.values,
        'genuine_train_mean': gen_mean.values,
        'genuine_train_std': gen_std.values
    }).sort_values('z_diff')

    result['rank'] = range(1, len(result) + 1)
    return result, high_score_imp.sum()


def main():
    # Allow threshold from command line
    if len(sys.argv) > 1:
        try:
            THRESHOLD = float(sys.argv[1])
        except ValueError:
            print("Invalid threshold. Using 0.0")
            THRESHOLD = 0.0
    else:
        THRESHOLD = 0.0
    print(f"Using decision threshold: {THRESHOLD}\n")

    all_user_rankings = {}        # user_id -> DataFrame (all features ranked)
    global_z_sums = {}            # sum of z_diff per feature across users
    global_z_counts = {}          # count of users having this feature
    global_z_medians = {}         # list of z_diff per feature for median/IQR
    z_diff_lists = {}             # to store all z_diff values per feature
    user_count = 0

    for user in USER_LIST:
        rank_df, num_high = analyze_one_user(user, THRESHOLD)
        if rank_df is None:
            continue
        if rank_df.empty:
            # Perfect user – still contribute to counts? No, they have no impostor issues.
            continue
        user_count += 1
        all_user_rankings[user] = rank_df

        # Save per‑user CSV
        rank_df.to_csv(f"feature_similarity_{user}.csv", index=False)

        # Accumulate for global statistics
        for _, row in rank_df.iterrows():
            feat = row['feature']
            z = row['z_diff']
            if feat not in z_diff_lists:
                z_diff_lists[feat] = []
            z_diff_lists[feat].append(z)

    if user_count == 0:
        print("No users with valid data found.")
        return

    # Build global DataFrame
    global_rows = []
    for feat, z_list in z_diff_lists.items():
        avg_z = np.mean(z_list)
        med_z = np.median(z_list)
        iqr_z = np.percentile(z_list, 75) - np.percentile(z_list, 25)
        # critical count: users where z_diff < 1.0 (impostors very close to genuine)
        critical = sum(1 for z in z_list if z < 1.0)
        global_rows.append({
            'feature': feat,
            'avg_z_diff': avg_z,
            'median_z_diff': med_z,
            'iqr_z_diff': iqr_z,
            'critical_count': critical,          # #users where feature is very weak
            'total_users': len(z_list)
        })

    global_rank_df = pd.DataFrame(global_rows)
    global_rank_df.sort_values('avg_z_diff', inplace=True)

    # Top‑20 count (features most often in a user's personal top‑20)
    top20_counts = {}
    for user, df in all_user_rankings.items():
        top20 = set(df.head(20)['feature'])
        for feat in top20:
            top20_counts[feat] = top20_counts.get(feat, 0) + 1
    global_rank_df['top20_count'] = global_rank_df['feature'].map(top20_counts).fillna(0).astype(int)

    # Save global summary
    global_rank_df.to_csv("feature_similarity_global.csv", index=False)

    # Print summaries
    print("\n=== GLOBAL TOP 20 PROBLEMATIC FEATURES (lowest avg z‑diff) ===")
    print(global_rank_df.head(20).to_string(index=False))

    print("\n=== FEATURES WITH HIGHEST 'CRITICAL COUNT' (z‑diff < 1.0) ===")
    critical_sorted = global_rank_df.sort_values('critical_count', ascending=False)
    print(critical_sorted.head(20).to_string(index=False))

    print("\n=== FEATURES MOST OFTEN IN USER TOP‑20 ===")
    top20_global = global_rank_df.sort_values('top20_count', ascending=False)
    print(top20_global.head(20).to_string(index=False))

    print("\nPer‑user files saved as feature_similarity_<user>.csv")
    print("Global file saved as feature_similarity_global.csv")


if __name__ == "__main__":
    main()