# compute_permutation_importance.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

from load_data import load_feature_list, load_user_data

def custom_auc_scorer(estimator, X, y):
    """
    Scorer for OCSVM: computes ROC AUC using decision_function scores.
    y: 0 for genuine, 1 for impostor.
    """
    scores = estimator.decision_function(X)
    # AUC measures how well the scores separate the two classes.
    # Higher scores should correspond to genuine (class 0), but OCSVM may assign higher scores to
    # genuine (inliers). We'll compute AUC treating genuine as positive class (label 0)
    # by using 1 - scores? Actually roc_auc_score expects positive class with higher scores.
    # We'll invert scores for positive class if needed. Let's check: for an inlier, decision_function
    # returns positive, for outlier negative. So genuine (0) gets positive scores, impostor (1) negative.
    # So we can use scores as is: high score -> predicted genuine. So for ROC, we want positive class
    # (genuine) to have higher scores. Thus we can treat y_true=0 as positive class? No, roc_auc_score
    # expects y_true binary and scores where higher score indicates positive class (label 1).
    # So we need to map: let's set positive label as 0 (genuine) and then scores directly indicate
    # genuine. But roc_auc_score expects y_true and scores where positive class is 1. So we'll
    # invert y: y_inv = 1 - y. Then positive class becomes 1 (genuine) and scores are as is.
    y_inv = 1 - y
    if len(np.unique(y_inv)) < 2:
        return 0.0
    return roc_auc_score(y_inv, scores)

def compute_permutation_importance_for_user(user_id, feature_list, data_dir='features/v2',
                                            nu=0.1, gamma='scale', n_repeats=50, random_state=42):
    """
    Compute permutation importance of features for OCSVM.
    Returns DataFrame with mean and std importance.
    """
    train_df, test_df, common_features = load_user_data(user_id, feature_list, data_dir)
    X_train = train_df[common_features].astype(float).values
    # Prepare test data: combine genuine test (label 0) and impostor test (label 1)
    # test_df already contains both
    X_test = test_df[common_features].astype(float).values
    y_test = test_df['label'].values  # 0 genuine, 1 impostor

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

    # Permutation importance
    scorer = make_scorer(custom_auc_scorer, response_method='decision_function')
    perm_imp = permutation_importance(ocsvm, X_test_scaled, y_test, scoring=scorer,
                                      n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    result = pd.DataFrame({
        'feature': common_features,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    })
    result = result.sort_values('importance_mean', ascending=False)
    return result

if __name__ == "__main__":
    user_id = 'user_01'
    features = load_feature_list('feature_list.txt')
    try:
        df = compute_permutation_importance_for_user(user_id, features)
        df.to_csv(f'permutation_importance_{user_id}.csv', index=False)
        print(f"Permutation importance saved to permutation_importance_{user_id}.csv")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")