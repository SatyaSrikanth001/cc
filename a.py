#!/usr/bin/env python3
"""
Enhanced analysis: find features that cause false acceptances (FAR)
and false rejections (FRR) for each user, with global summaries.

Usage: python analyze_FAR_FRR.py [threshold]
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

from train_session import SessionOCSVM

warnings.filterwarnings("ignore")

# ======================= CONFIGURATION =======================
MODELS_DIR = "models"
FEATURES_DIR = "./features/v2"

# List of all users – adjust to your actual users
USER_LIST = [
    "reddy56", "Nikitha18", "Diana", "Avnish", "skyy",
    "Pranav", "surya", "divya", "Saurabh", "srikanth",
    "Samarth", "harshit"
]
# =============================================================

def load_user_artifacts(user_id):
    """
    Load the trained model, scaler, and the preprocessed data
    exactly as used during training.
    """
    model_path = os.path.join(MODELS_DIR, f"{user_id}_ocsvm.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{user_id}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler missing for {user_id}")

    ocsvm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    train_csv = os.path.join(FEATURES_DIR, f"{user_id}_training_sessions.csv")
    test_csv = os.path.join(FEATURES_DIR, f"{user_id}_testing_sessions.csv")

    session_model = SessionOCSVM(user_id)
    X_train_raw, X_test_raw, y_test, train_df_proc, test_df_proc = \
        session_model.load_and_prepare_data(train_csv, test_csv)

    # Load the actual feature list used during training (saved separately)
    features_path = os.path.join(MODELS_DIR, f"{user_id}_features.pkl")
    feature_cols = joblib.load(features_path)

    print(f"[{user_id}] Loaded saved feature list: {len(feature_cols)} features")
    print(f"[{user_id}] Scaler expects: {scaler.n_features_in_} features")

    if len(feature_cols) != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch for {user_id}: "
            f"features.pkl has {len(feature_cols)}, scaler expects {scaler.n_features_in_}"
        )

    # Keep only the features the scaler expects, in the same order
    train_features = train_df_proc[feature_cols].copy()
    test_features = test_df_proc[feature_cols].copy()

    # Ensure numeric and fill NaN
    train_features = train_features.apply(pd.to_numeric, errors="coerce").fillna(0)
    test_features = test_features.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Feature count: {len(feature_cols)}")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    return ocsvm, scaler, train_features, test_features, y_test, feature_cols


def analyze_one_user(user_id, threshold):
    """
    Return two DataFrames:
      - far_df : features ranked by z_diff_FAR ascending (worst FAR first)
      - frr_df : features ranked by z_diff_FRR descending (worst FRR first)
    """
    try:
        ocsvm, scaler, train_df, test_df, y_test, feature_cols = load_user_artifacts(user_id)
    except Exception as e:
        print(f"Skipping {user_id}: {e}")
        return None, None

    # Scale test data
    X_test = test_df.values.astype(float)
    try:
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"Scaling failed for {user_id}: {e}")
        print("Expected features:", scaler.n_features_in_)
        print("Provided features:", X_test.shape[1])
        return None, None

    scores = ocsvm.decision_function(X_test_scaled)

    imp_mask = (y_test == -1)
    gen_mask = (y_test == 1)

    # False Acceptances: impostors with score >= threshold
    fa_mask = imp_mask & (scores >= threshold)
    # False Rejections: genuine with score < threshold
    fr_mask = gen_mask & (scores < threshold)

    print(f"[{user_id}] False Acceptances: {fa_mask.sum()}/{imp_mask.sum()}")
    print(f"[{user_id}] False Rejections:  {fr_mask.sum()}/{gen_mask.sum()}")

    # Genuine training statistics (on original feature values)
    gen_mean = train_df.mean()
    gen_std = train_df.std()

    far_df = pd.DataFrame()
    frr_df = pd.DataFrame()

    # --- FAR analysis ---
    if fa_mask.sum() > 0:
        imp_high = test_df.loc[fa_mask]
        imp_high_mean = imp_high.mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            z_diff_far = (imp_high_mean - gen_mean).abs() / (gen_std + 1e-8)
        z_diff_far = z_diff_far.replace(np.inf, np.nan).fillna(0)

        far_df = pd.DataFrame({
            "feature": z_diff_far.index,
            "z_diff_FAR": z_diff_far.values,
            "impostor_high_mean": imp_high_mean.values,
            "genuine_train_mean": gen_mean.values,
            "genuine_train_std": gen_std.values
        }).sort_values("z_diff_FAR", ascending=True)  # smallest = most similar

        far_df["rank_FAR"] = range(1, len(far_df) + 1)

    # --- FRR analysis ---
    if fr_mask.sum() > 0:
        gen_low = test_df.loc[fr_mask]
        gen_low_mean = gen_low.mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            z_diff_frr = (gen_low_mean - gen_mean).abs() / (gen_std + 1e-8)
        z_diff_frr = z_diff_frr.replace(np.inf, np.nan).fillna(0)

        frr_df = pd.DataFrame({
            "feature": z_diff_frr.index,
            "z_diff_FRR": z_diff_frr.values,
            "genuine_low_mean": gen_low_mean.values,
            "genuine_train_mean": gen_mean.values,
            "genuine_train_std": gen_std.values
        }).sort_values("z_diff_FRR", ascending=False)  # largest = most different

        frr_df["rank_FRR"] = range(1, len(frr_df) + 1)

    return far_df, frr_df


def main():
    # --- Threshold ---
    if len(sys.argv) > 1:
        try:
            THRESHOLD = float(sys.argv[1])
        except ValueError:
            print("Invalid threshold. Using 0.0")
            THRESHOLD = 0.0
    else:
        THRESHOLD = 0.0
    print(f"Using decision threshold: {THRESHOLD}\n")

    # --- Per‑user analysis ---
    all_far_dfs = {}      # user -> DataFrame
    all_frr_dfs = {}
    user_count = 0

    for user in USER_LIST:
        far_df, frr_df = analyze_one_user(user, THRESHOLD)
        if far_df is None:
            continue
        user_count += 1

        if not far_df.empty:
            all_far_dfs[user] = far_df
            far_df.to_csv(f"FAR_features_{user}.csv", index=False)
        if not frr_df.empty:
            all_frr_dfs[user] = frr_df
            frr_df.to_csv(f"FRR_features_{user}.csv", index=False)

    if user_count == 0:
        print("No users with valid data.")
        return

    # ================== Global FAR aggregation ==================
    z_far_lists = {}      # feature -> list of z_diff_FAR across users
    far_top20_counts = {}

    for user, df in all_far_dfs.items():
        for _, row in df.iterrows():
            feat = row["feature"]
            z = row["z_diff_FAR"]
            if feat not in z_far_lists:
                z_far_lists[feat] = []
            z_far_lists[feat].append(z)

        top20 = set(df.head(20)["feature"])
        for feat in top20:
            far_top20_counts[feat] = far_top20_counts.get(feat, 0) + 1

    global_far_rows = []
    for feat, zlist in z_far_lists.items():
        avg_z = np.mean(zlist)
        med_z = np.median(zlist)
        iqr_z = np.percentile(zlist, 75) - np.percentile(zlist, 25)
        critical = sum(1 for z in zlist if z < 1.0)   # impostor very close
        global_far_rows.append({
            "feature": feat,
            "avg_z_diff_FAR": avg_z,
            "median_z_diff_FAR": med_z,
            "iqr_z_diff_FAR": iqr_z,
            "critical_count_FAR": critical,
            "total_users": len(zlist),
            "top20_count_FAR": far_top20_counts.get(feat, 0)
        })

    far_global_df = pd.DataFrame(global_far_rows).sort_values("avg_z_diff_FAR")
    far_global_df.to_csv("FAR_features_global.csv", index=False)

    # ================== Global FRR aggregation ==================
    z_frr_lists = {}      # feature -> list of z_diff_FRR across users
    frr_top20_counts = {}

    for user, df in all_frr_dfs.items():
        for _, row in df.iterrows():
            feat = row["feature"]
            z = row["z_diff_FRR"]
            if feat not in z_frr_lists:
                z_frr_lists[feat] = []
            z_frr_lists[feat].append(z)

        top20 = set(df.head(20)["feature"])
        for feat in top20:
            frr_top20_counts[feat] = frr_top20_counts.get(feat, 0) + 1

    global_frr_rows = []
    for feat, zlist in z_frr_lists.items():
        avg_z = np.mean(zlist)
        med_z = np.median(zlist)
        iqr_z = np.percentile(zlist, 75) - np.percentile(zlist, 25)
        critical = sum(1 for z in zlist if z > 2.0)   # genuine far from normal
        global_frr_rows.append({
            "feature": feat,
            "avg_z_diff_FRR": avg_z,
            "median_z_diff_FRR": med_z,
            "iqr_z_diff_FRR": iqr_z,
            "critical_count_FRR": critical,
            "total_users": len(zlist),
            "top20_count_FRR": frr_top20_counts.get(feat, 0)
        })

    frr_global_df = pd.DataFrame(global_frr_rows).sort_values("avg_z_diff_FRR", ascending=False)
    frr_global_df.to_csv("FRR_features_global.csv", index=False)

    # --- Print summaries ---
    print("\n=== GLOBAL TOP 20 FAR‑DRIVING FEATURES (lowest avg z‑diff) ===")
    print(far_global_df.head(20).to_string(index=False))

    print("\n=== GLOBAL TOP 20 FRR‑DRIVING FEATURES (highest avg z‑diff) ===")
    print(frr_global_df.head(20).to_string(index=False))

    print("\nPer‑user FAR files: FAR_features_<user>.csv")
    print("Per‑user FRR files: FRR_features_<user>.csv")
    print("Global FAR file: FAR_features_global.csv")
    print("Global FRR file: FRR_features_global.csv")


if __name__ == "__main__":
    main()
