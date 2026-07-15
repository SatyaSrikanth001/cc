# compute_shap_ocsvm.py
import pandas as pd
import numpy as np
import shap
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import os

from load_data import load_feature_list, load_user_data

def compute_shap_for_user(user_id, feature_list, data_dir='features/v2',
                          nu=0.1, gamma='scale', random_state=42):
    """
    Compute SHAP feature importance for OCSVM using KernelExplainer.
    Returns DataFrame with mean absolute SHAP per feature.
    """
    train_df, test_df, common_features = load_user_data(user_id, feature_list, data_dir)
    X_train = train_df[common_features].astype(float).values
    X_test = test_df[common_features].astype(float).values
    # (y_test not needed for SHAP)

    # Clean NaN
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train OCSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    ocsvm.fit(X_train_scaled)

    # Prepare background with k‑means summary of training data
    background = shap.kmeans(X_train_scaled, min(50, len(X_train_scaled)))

    # KernelExplainer
    explainer = shap.KernelExplainer(ocsvm.decision_function, background)

    # Use a subset of test samples to keep computation reasonable
    test_sample = X_test_scaled
    if len(test_sample) > 200:
        test_sample = shap.utils.sample(X_test_scaled, 200, random_state=random_state)

    # Compute SHAP values – nsamples controls the number of model evaluations per test point
    shap_values = explainer.shap_values(test_sample, nsamples=100)

    # Mean absolute SHAP as global importance
    shap_importance = np.mean(np.abs(shap_values), axis=0)

    result = pd.DataFrame({
        'feature': common_features,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)

    return result

if __name__ == "__main__":
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_shap_for_user(user_id, features)
        df.to_csv(f'shap_importance_{user_id}.csv', index=False)
        print(f"SHAP importance saved to shap_importance_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")