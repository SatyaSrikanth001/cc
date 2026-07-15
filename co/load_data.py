# load_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def load_feature_list(filepath='feature_list.txt'):
    """Read the list of features to be used."""
    with open(filepath, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features

def load_user_data(user_id, feature_list, data_dir='features/v2'):
    """
    Load training and testing data for a single user.
    Returns (train_df, test_df) with only the features present in feature_list.
    """
    train_path = Path(data_dir) / f"{user_id}_training_sessions.csv"
    test_path = Path(data_dir) / f"{user_id}_testing_sessions.csv"
    
    # Read raw CSV
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    
    # Identify label column in test data (expected: 'label')
    if 'label' not in test_raw.columns:
        raise KeyError(f"'label' column missing in {test_path}")
    
    # For training data, all sessions are genuine → label = 0
    train_raw['label'] = 0
    
    # Find which requested features actually exist in the data
    common_train = [f for f in feature_list if f in train_raw.columns]
    common_test  = [f for f in feature_list if f in test_raw.columns]
    common_features = sorted(set(common_train) & set(common_test))
    
    missing = set(feature_list) - set(common_features)
    if missing:
        print(f"[WARNING] {user_id}: {len(missing)} features missing, e.g. {list(missing)[:5]}")
    
    # Select only the common features + label
    train_df = train_raw[common_features + ['label']].copy()
    test_df  = test_raw[common_features + ['label']].copy()
    
    # Ensure numeric (fill NaN with median later in preprocessing, not here)
    # For now, just convert to numeric, coerce errors
    for col in common_features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col]  = pd.to_numeric(test_df[col], errors='coerce')
    
    return train_df, test_df, common_features

# Quick test (commented out – run manually when data is available)
# if __name__ == "__main__":
#     features = load_feature_list()
#     train, test, _ = load_user_data('user_01', features)
#     print(train.shape, test.shape)