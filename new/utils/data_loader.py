# utils/data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(config: dict) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load per‑user training and testing CSVs, combine them into a single
    DataFrame per user, and apply basic cleaning.

    Parameters
    ----------
    config : dict
        Must contain:
        - data.features_dir (path to folder with per‑user CSVs)
        - data.id_columns (list of columns to treat as non‑feature)
        - cleaning.missing_threshold
        - cleaning.constant_user_threshold

    Returns
    -------
    user_data : dict
        Mapping from device_owner_id -> DataFrame with feature columns + 'is_genuine'.
    final_features : list
        List of feature names that survived basic cleaning.
    """
    features_dir = config['data']['features_dir']
    train_pattern = '*_training_sessions.csv'
    test_pattern = '*_testing_sessions.csv'
    id_cols = config['data']['id_columns']
    missing_thresh = config['cleaning']['missing_threshold']
    const_user_thresh = config['cleaning']['constant_user_threshold']

    # --- 1. Collect per‑user DataFrames ---
    train_files = sorted(glob.glob(os.path.join(features_dir, train_pattern)))
    if not train_files:
        raise FileNotFoundError(f"No training files found in {features_dir}")

    user_dfs = {}
    for train_path in train_files:
        fname = os.path.basename(train_path)
        user_id = fname.replace('_training_sessions.csv', '')

        # Load training data (all genuine)
        train_df = pd.read_csv(train_path)
        # Remove any label columns that may have slipped in (shouldn't)
        for col in ['session_label', 'label', 'test_type']:
            train_df.drop(columns=[col], errors='ignore', inplace=True)
        train_df['is_genuine'] = 1

        # Load testing data
        test_path = os.path.join(features_dir, f'{user_id}_testing_sessions.csv')
        if not os.path.exists(test_path):
            logging.warning(f"Testing file missing for {user_id}, skipping.")
            continue
        test_df = pd.read_csv(test_path)

        # Find label column in test data
        label_col = None
        for col in ['session_label', 'label', 'test_type']:
            if col in test_df.columns:
                label_col = col
                break
        if label_col is None:
            logging.warning(f"No label column found for {user_id}, skipping.")
            continue

        # Normalize labels to 0/1
        def norm_label(x):
            if pd.isna(x):
                return 0
            return 1 if 'genuine' in str(x).lower() else 0

        test_df['is_genuine'] = test_df[label_col].apply(norm_label)
        test_df.drop(columns=[label_col], errors='ignore', inplace=True)

        # Concatenate train and test
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined['device_owner_id'] = user_id
        user_dfs[user_id] = combined

    if not user_dfs:
        raise RuntimeError("No valid user data found.")

    # Concatenate all for global cleaning (we'll split back later)
    all_data = pd.concat(user_dfs.values(), ignore_index=True)
    # Also keep a user_id column (copy of device_owner_id) for compatibility
    all_data['user_id'] = all_data['device_owner_id']

    # --- 2. Identify feature columns ---
    feature_cols = [c for c in all_data.columns if c not in id_cols + ['is_genuine', 'device_owner_id', 'user_id']]
    logging.info(f"Initial feature count: {len(feature_cols)}")

    # --- 3. Drop features with excessive missing values ---
    missing_ratio = all_data[feature_cols].isnull().mean()
    drop_missing = missing_ratio[missing_ratio > missing_thresh].index.tolist()
    all_data.drop(columns=drop_missing, inplace=True)
    feature_cols = [c for c in feature_cols if c not in drop_missing]
    logging.info(f"Dropped {len(drop_missing)} features with >{missing_thresh*100}% missing. "
                 f"Remaining: {len(feature_cols)}")

    # --- 4. Impute remaining missing values per user (median) ---
    for col in feature_cols:
        all_data[col] = all_data.groupby('device_owner_id')[col].transform(
            lambda x: x.fillna(x.median()) if x.notna().sum() > 0 else 0.0
        )
    logging.info("Imputed missing values with per‑user median.")

    # --- 5. Remove features constant in too many users ---
    constant_counts = {}
    for user, grp in all_data.groupby('device_owner_id'):
        for col in feature_cols:
            if grp[col].var() == 0:
                constant_counts[col] = constant_counts.get(col, 0) + 1

    n_users = all_data['device_owner_id'].nunique()
    drop_const = [col for col, cnt in constant_counts.items() if cnt / n_users > const_user_thresh]
    all_data.drop(columns=drop_const, inplace=True)
    feature_cols = [c for c in feature_cols if c not in drop_const]
    logging.info(f"Dropped {len(drop_const)} constant features (variance=0 in >{const_user_thresh*100}% users). "
                 f"Remaining: {len(feature_cols)}")

    # --- 6. Split back into per‑user dict ---
    user_data = {}
    for user, grp in all_data.groupby('device_owner_id'):
        # Keep only feature columns + is_genuine
        user_data[user] = grp[feature_cols + ['is_genuine']].copy()

    logging.info(f"Final dataset: {len(user_data)} users, {len(feature_cols)} features.")
    return user_data, feature_cols
