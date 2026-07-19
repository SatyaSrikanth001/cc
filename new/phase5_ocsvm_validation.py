# phase5_ocsvm_validation.py
"""
Phase 5: Leakage-free One-Class SVM validation.

Training:
    Genuine development sessions only.

Evaluation:
    Untouched holdout sessions from the testing files.

Feature selection and Baseline A:
    Development data only.

The reported EER and ROC-AUC are threshold-independent.
"""

import logging
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from utils.data_loader import load_split_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray
) -> float:
    """
    Compute EER with linear interpolation between the two closest points
    on the ROC curve for better accuracy.

    FIX #15: Interpolation replaces the simple nearest‑point approximation.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr

    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))

    # Edge cases: if the closest point is at the ends, return the average at that point
    if idx == 0 or idx == len(fpr) - 1:
        return float((fpr[idx] + fnr[idx]) / 2.0)

    # Linear interpolation between (fpr[idx-1], fnr[idx-1]) and (fpr[idx], fnr[idx])
    # We want to find the point where fpr == fnr.
    x1, y1 = fpr[idx - 1], fnr[idx - 1]
    x2, y2 = fpr[idx], fnr[idx]

    # Avoid division by zero if points are identical
    if x2 == x1:
        return float((x1 + y1) / 2.0)

    # Slope and intercept of the line connecting the two points
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Solve for x where y = x (i.e., fpr = fnr)
    # x = slope*x + intercept  =>  x*(1 - slope) = intercept  =>  x = intercept / (1 - slope)
    if np.isclose(slope, 1.0):
        # Parallel to the line y=x; take the average of the two point averages
        eer1 = (x1 + y1) / 2.0
        eer2 = (x2 + y2) / 2.0
        return float((eer1 + eer2) / 2.0)

    eer = intercept / (1.0 - slope)

    # Clamp to a valid probability range
    eer = max(0.0, min(1.0, eer))
    return float(eer)


def get_baseline_a_features(
    config: dict,
    all_features: List[str],
    top_k: int
) -> List[str]:
    """
    Baseline A:
    Top K features by direction-agnostic median AUC on development data only.

    FIX P5-B1: Directly loads development-only AUC rankings computed during Phase 2A,
    retaining statistical alignment and removing heavy nested recomputation loops.
    """
    auc_details_path = config["output"]["auc_details"]
    if not os.path.exists(auc_details_path):
        raise FileNotFoundError(
            f"Phase 2A AUC details file is missing at: '{auc_details_path}'. Please run Phase 2A first."
        )

    auc_df = pd.read_csv(auc_details_path, index_col=0)
    # Force index to string to avoid type mismatches
    auc_df.index = auc_df.index.astype(str)

    # Restrict strictly to features presently matching the dynamic cleaned list
    valid_features = auc_df.index.intersection(all_features)
    ranking = auc_df.loc[valid_features, "median_auc"].sort_values(ascending=False)

    selected = ranking.head(min(top_k, len(ranking))).index.tolist()

    logging.info(
        "Baseline A selected %d features based on Phase 2A results (Dev only).",
        len(selected)
    )

    return selected


def get_baseline_b_features(
    all_features: List[str],
    top_k: int,
    random_state: int
) -> List[str]:
    rng = np.random.RandomState(random_state)

    n = min(top_k, len(all_features))

    selected = rng.choice(
        all_features,
        size=n,
        replace=False
    ).tolist()

    logging.info(
        "Baseline B randomly selected %d features.",
        len(selected)
    )

    return selected


def evaluate_ocsvm_for_user(
    train_genuine: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: List[str],
    ocsvm_config: dict
) -> Tuple[float, float]:
    available = [
        feature for feature in features
        if feature in train_genuine.columns
        and feature in holdout_df.columns
    ]

    if not available:
        return np.nan, np.nan

    if holdout_df["is_genuine"].nunique() < 2:
        return np.nan, np.nan

    X_train = train_genuine[available].to_numpy(dtype=float)
    X_test = holdout_df[available].to_numpy(dtype=float)
    y_test = holdout_df["is_genuine"].to_numpy(dtype=int)

    # Defensive NaN handling (should not be needed after imputation)
    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if len(X_train) < 3:
        return np.nan, np.nan

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = OneClassSVM(
        kernel=ocsvm_config.get("kernel", "rbf"),
        nu=ocsvm_config.get("nu", 0.1),
        gamma=ocsvm_config.get("gamma", "scale")
    )

    try:
        model.fit(X_train_scaled)
        scores = model.decision_function(X_test_scaled)

        eer = compute_eer(y_test, scores)
        auc_value = float(roc_auc_score(y_test, scores))

        return eer, auc_value

    except Exception as exc:
        logging.warning("OCSVM evaluation failed: %s", exc)
        return np.nan, np.nan


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(config["output"].get("dir", "out"), exist_ok=True)

    dev_data, holdout_data, all_features = load_split_data(config)

    logging.info(
        "Loaded %d development users and %d holdout users.",
        len(dev_data),
        len(holdout_data)
    )

    final_path = config["output"]["final_selected_features"]

    if not os.path.exists(final_path):
        raise FileNotFoundError(
            "Final Phase 4 feature file is missing. Run Phase 4 first."
        )

    final_features = (
        pd.read_csv(final_path, dtype={"feature": str})["feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Baseline A: top 40 from Phase 2A (dev only)
    baseline_a = get_baseline_a_features(
        config=config,
        all_features=all_features,
        top_k=40
    )

    # Baseline B: random 40
    baseline_b = get_baseline_b_features(
        all_features,
        top_k=40,
        random_state=config.get("split", {}).get("random_state", 42)
    )

    feature_sets = [
        ("Final 40", final_features),
        ("Baseline A - top AUC dev", baseline_a),
        ("Baseline B - random", baseline_b)
    ]

    results = []

    for user, dev_df in dev_data.items():
        if user not in holdout_data:
            logging.warning(
                "No holdout data for user %s; skipping.",
                user
            )
            continue

        holdout_df = holdout_data[user]

        train_genuine = dev_df[
            dev_df["is_genuine"] == 1
        ].copy()

        if len(train_genuine) < 3:
            logging.warning(
                "User %s has too few genuine development sessions.",
                user
            )
            continue

        if holdout_df["is_genuine"].nunique() < 2:
            logging.warning(
                "User %s holdout contains only one class.",
                user
            )
            continue

        for method_name, features in feature_sets:
            eer, auc_value = evaluate_ocsvm_for_user(
                train_genuine=train_genuine,
                holdout_df=holdout_df,
                features=features,
                ocsvm_config=config.get("ocsvm", {})
            )

            results.append({
                "user": user,
                "method": method_name,
                "eer": eer,
                "auc": auc_value
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No validation results were generated."
        )

    results_df.to_csv(
        config["output"]["eer_results"],
        index=False
    )

    summary = (
        results_df
        .groupby("method")
        .agg(
            median_eer=("eer", "median"),
            mean_eer=("eer", "mean"),
            std_eer=("eer", "std"),
            median_auc=("auc", "median"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            n_users=("user", "nunique")
        )
        .reset_index()
    )

    summary.to_csv(
        config["output"]["summary_metrics"],
        index=False
    )

    print("\n--- Leakage-Free Validation Summary ---")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    sns.boxplot(
        data=results_df,
        x="method",
        y="eer",
        ax=axes[0]
    )
    axes[0].set_title("Holdout EER")
    axes[0].set_ylabel("EER - lower is better")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(
        data=results_df,
        x="method",
        y="auc",
        ax=axes[1]
    )
    axes[1].set_title("Holdout ROC-AUC")
    axes[1].set_ylabel("AUC - higher is better")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(
        config["output"]["validation_boxplot"],
        dpi=150
    )
    plt.close()

    logging.info(
        "Validation results saved to %s.",
        config["output"]["eer_results"]
    )


if __name__ == "__main__":
    main()