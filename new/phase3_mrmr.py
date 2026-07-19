# phase3_mrmr.py
"""
Phase 3: Greedy mRMR feature selection.

Relevance:
    Direction-agnostic median AUC from Phase 2A.

Redundancy:
    Average absolute Pearson correlation computed on development data only
    (pairwise deletion of missing values, not listwise).
"""

import logging
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from utils.data_loader import load_split_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_elite_features(config) -> List[str]:
    """Load Phase 2B elite features, ensuring string dtype."""
    path = config["output"]["elite_features"]

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Phase 2B elite feature file is missing. Run Phase 2B first."
        )

    elite_df = pd.read_csv(path, dtype={"feature": str})

    if "feature" not in elite_df.columns:
        raise ValueError(
            "Elite feature file must contain a 'feature' column."
        )

    return elite_df["feature"].dropna().astype(str).tolist()


def load_relevance_scores(
    config,
    features: List[str]
) -> pd.Series:
    """Load median AUC as relevance; ensure string index."""
    auc_df = pd.read_csv(
        config["output"]["auc_details"],
        index_col=0
    )
    auc_df.index = auc_df.index.astype(str)

    missing = [f for f in features if f not in auc_df.index]
    if missing:
        raise ValueError(
            f"{len(missing)} elite features are missing from AUC details."
        )

    relevance = auc_df.loc[features, "median_auc"].astype(float)
    relevance = relevance.replace([np.inf, -np.inf], np.nan).dropna()

    return relevance


def compute_average_correlation(
    user_data: Dict[str, pd.DataFrame],
    feature_list: List[str]
) -> pd.DataFrame:
    """
    Compute average absolute Pearson correlation across users.

    FIX #13: Use pairwise deletion (pandas corr default) instead of
    listwise deletion (.dropna()), which was too aggressive.  Users
    with fewer than 5 total rows are skipped to avoid unstable estimates.
    """
    correlation_sum = None
    contributing_users = 0

    for user, df in user_data.items():
        # Convert to numeric; leave NaNs for corr to handle pairwise
        sub = df[feature_list].apply(pd.to_numeric, errors="coerce")

        if len(sub) < 5:
            logging.warning(
                "User %s has fewer than five rows; skipping correlation.",
                user
            )
            continue

        # corr uses pairwise complete observations by default
        corr = sub.corr(method="pearson").abs()
        corr = corr.fillna(0.0)  # if a whole column is NaN, correlation is 0

        if correlation_sum is None:
            correlation_sum = corr.copy()
        else:
            correlation_sum = correlation_sum.add(corr, fill_value=0.0)

        contributing_users += 1

    if correlation_sum is None or contributing_users == 0:
        raise RuntimeError(
            "No users contributed valid correlation matrices."
        )

    average = correlation_sum / contributing_users
    average = average.reindex(
        index=feature_list,
        columns=feature_list
    ).fillna(0.0)

    values = average.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)   # no self‑redundancy

    logging.info(
        "Average correlation computed from %d users.",
        contributing_users
    )

    return pd.DataFrame(
        values,
        index=feature_list,
        columns=feature_list
    )


def mrmr_greedy_selection(
    relevance: pd.Series,
    redundancy: pd.DataFrame,
    final_k: int
) -> List[str]:
    """
    Greedy mRMR: select features maximising relevance - mean redundancy.
    Ties are broken deterministically by sorted order of remaining features.
    """
    relevance = relevance.dropna()

    if relevance.empty:
        raise RuntimeError("No valid relevance scores available.")

    final_k = min(final_k, len(relevance))

    # Pick first feature with highest relevance (ties broken by stable sort)
    selected = [
        relevance.sort_values(
            ascending=False,
            kind="mergesort"
        ).index[0]
    ]

    remaining = set(relevance.index) - set(selected)

    while len(selected) < final_k and remaining:
        best_feature = None
        best_score = -np.inf

        for feature in sorted(remaining):
            redundancy_values = [
                float(redundancy.loc[feature, selected_feature])
                for selected_feature in selected
            ]

            redundancy_mean = float(np.mean(redundancy_values))
            score = float(relevance.loc[feature]) - redundancy_mean

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def plot_correlation(
    redundancy: pd.DataFrame,
    selected: List[str],
    output_path: str
):
    """Heatmap of average correlations among selected features."""
    subset = redundancy.loc[selected, selected]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        subset,
        cmap="Reds",
        vmin=0,
        vmax=1,
        annot=False
    )
    plt.title(
        f"Average Absolute Correlation Among {len(selected)} Selected Features"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(config["output"].get("dir", "out"), exist_ok=True)

    elite_features = load_elite_features(config)
    relevance = load_relevance_scores(config, elite_features)

    dev_data, _, _ = load_split_data(config)

    # Ensure features exist in all users (defensive; data_loader already ensures)
    available_features = [
        f for f in relevance.index
        if all(f in df.columns for df in dev_data.values())
    ]
    relevance = relevance.loc[available_features]

    redundancy = compute_average_correlation(
        dev_data,
        available_features
    )

    final_k = config.get("final_k", 40)
    selected = mrmr_greedy_selection(
        relevance,
        redundancy,
        final_k=final_k
    )

    # Save final feature list
    pd.DataFrame({"feature": selected}).to_csv(
        config["output"]["final_features"],
        index=False
    )

    # Recompute mRMR scores for diagnostic output
    scores = []
    for i, feature in enumerate(selected):
        if i == 0:
            score = float(relevance.loc[feature])
        else:
            red = [
                redundancy.loc[feature, previous]
                for previous in selected[:i]
            ]
            score = float(relevance.loc[feature]) - float(np.mean(red))
        scores.append(score)

    pd.DataFrame({
        "feature": selected,
        "mrmr_score": scores
    }).to_csv(
        config["output"]["mrmr_scores"],
        index=False
    )

    # Plot correlation heatmap
    plot_path = os.path.join(
        config["output"].get("dir", "out"),
        "phase3_final_correlation.png"
    )
    plot_correlation(redundancy, selected, plot_path)

    logging.info(
        "Phase 3 selected %d features.",
        len(selected)
    )


if __name__ == "__main__":
    main()