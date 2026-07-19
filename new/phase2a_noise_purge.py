# phase2a_noise_purge.py
"""
Phase 2A: Direction-agnostic within-user AUC noise purge.

Feature selection is performed only on the development data returned by
load_split_data(). The holdout data is never used here.
"""

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from utils.data_loader import load_split_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_aucs_per_user(
    user_data: Dict[str, pd.DataFrame],
    feature_list: List[str]
) -> pd.DataFrame:
    """
    Compute direction-agnostic ROC-AUC for every user and feature.

    Direction-agnostic AUC:
        max(raw_auc, 1 - raw_auc)

    FIX #9: Users with single-class dev data are skipped entirely
    (they would otherwise contribute artificial 0.5 AUC values that
    bias the median downward). A warning is logged for each such user.
    """
    records = []
    skipped_users = []

    for user, df in user_data.items():
        y = df["is_genuine"].to_numpy()

        if len(np.unique(y)) < 2:
            logging.warning(f"User {user} has single-class dev data; "
                            "skipping AUC computation for this user.")
            skipped_users.append(user)
            continue  # Do NOT add 0.5 records for this user

        for feature in feature_list:
            x = pd.to_numeric(df[feature], errors="coerce").to_numpy()

            if np.isnan(x).any():
                valid = ~np.isnan(x)
                x_valid = x[valid]
                y_valid = y[valid]
            else:
                x_valid = x
                y_valid = y

            if len(x_valid) == 0 or len(np.unique(y_valid)) < 2:
                raw_auc = 0.5
                direction = "undefined"
                auc_value = 0.5

            elif np.nanstd(x_valid) == 0:
                raw_auc = 0.5
                direction = "constant"
                auc_value = 0.5

            else:
                try:
                    raw_auc = float(roc_auc_score(y_valid, x_valid))
                except ValueError:
                    raw_auc = 0.5

                direction = (
                    "positive" if raw_auc >= 0.5 else "negative"
                )
                auc_value = max(raw_auc, 1.0 - raw_auc)

            records.append({
                "feature": feature,
                "user": user,
                "auc": auc_value,
                "raw_auc": raw_auc,
                "direction": direction
            })

    if skipped_users:
        logging.info(f"Skipped {len(skipped_users)} users with single-class dev data: "
                     f"{', '.join(skipped_users[:5])}{'...' if len(skipped_users)>5 else ''}")

    return pd.DataFrame(records)


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    output_dir = config["output"].get("dir", "out")
    os.makedirs(output_dir, exist_ok=True)

    dev_data, holdout_data, feature_list = load_split_data(config)

    logging.info(
        "Loaded %d development users and %d cleaned features.",
        len(dev_data),
        len(feature_list)
    )

    auc_df = compute_aucs_per_user(dev_data, feature_list)

    aggregate = (
        auc_df
        .groupby("feature")["auc"]
        .agg(["median", "std", "count", "min", "max"])
        .rename(columns={
            "median": "median_auc",
            "std": "std_auc",
            "count": "num_users",
            "min": "min_auc",
            "max": "max_auc"
        })
    )

    # FIX #10: Deterministic mode selection – if multiple modes exist,
    # choose the one that appears first alphabetically after sorting.
    def dominant_direction(series):
        """
        Return the dominant direction with an explicit deterministic tie-break.
        If positive and negative are tied, choose alphabetically (i.e., "negative" then "positive").
        """
        if series.empty:
            return "undefined"

        counts = series.value_counts()
        if counts.empty:
            return "undefined"

        top_count = counts.max()
        tied = sorted(counts[counts == top_count].index.astype(str).tolist())
        return tied[0]

    direction_summary = (
        auc_df[auc_df["direction"].isin(["positive", "negative"])]
        .groupby("feature")["direction"]
        .agg(dominant_direction)
        .rename("dominant_direction")
    )

    aggregate = aggregate.join(direction_summary)

    threshold = config["noise_purge"]["noise_threshold"]

    survivors = (
        aggregate.loc[aggregate["median_auc"] >= threshold]
        .sort_values(
            ["median_auc", "std_auc"],
            ascending=[False, True]
        )
        .index
        .tolist()
    )

    dropped = (
        aggregate.loc[aggregate["median_auc"] < threshold]
        .index
        .tolist()
    )

    logging.info(
        "Surviving features: %d; dropped features: %d.",
        len(survivors),
        len(dropped)
    )

    if not survivors:
        raise RuntimeError(
            "Phase 2A produced zero surviving features. "
            "Lower noise_purge.noise_threshold or inspect the data."
        )

    # Save selected feature list
    pd.DataFrame({"feature": survivors}).to_csv(
        config["output"]["filtered_features"],
        index=False
    )

    # Save full AUC details
    aggregate.sort_values(
        "median_auc",
        ascending=False
    ).to_csv(config["output"]["auc_details"])

    print("\n--- Top 10 features by direction-agnostic median AUC ---")
    print(
        aggregate
        .sort_values("median_auc", ascending=False)
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()