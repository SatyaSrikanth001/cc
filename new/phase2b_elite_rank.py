# phase2b_elite_rank.py
"""
Phase 2B: Rank Phase 2A survivors by median AUC.

Selects the top-K features (default 100) based on median direction-agnostic AUC,
using standard deviation as a tiebreaker (lower std preferred).
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_phase2a_results(config: dict) -> pd.DataFrame:
    """
    Load Phase 2A outputs and filter AUC details to only survivors.

    FIX #11: Specify dtype=str when reading survivor feature names
    to avoid integer coercion (e.g., features named "12345").
    """
    survivors_path = config["output"]["filtered_features"]
    details_path = config["output"]["auc_details"]

    if not os.path.exists(survivors_path):
        raise FileNotFoundError(
            "Phase 2A survivor file is missing. Run Phase 2A first."
        )

    if not os.path.exists(details_path):
        raise FileNotFoundError(
            "Phase 2A AUC details are missing. Run Phase 2A first."
        )

    # Force feature names as strings
    survivors = pd.read_csv(survivors_path, dtype={"feature": str})["feature"].tolist()

    # Load AUC details; force index (feature names) as strings as well
    auc_df = pd.read_csv(details_path, index_col=0)
    # Convert index to string to avoid mismatches (e.g., if some are int)
    auc_df.index = auc_df.index.astype(str)

    # Ensure all survivors are present
    missing = [f for f in survivors if f not in auc_df.index]
    if missing:
        raise ValueError(
            f"{len(missing)} survivor features are missing from AUC details: "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
        )

    # Keep only survivors
    auc_df = auc_df.loc[survivors]
    return auc_df


def rank_features(auc_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Rank survivors by median AUC descending, then by std_auc ascending.
    Returns the top-K rows.
    """
    top_k = min(top_k, len(auc_df))

    # Use stable sort (mergesort) for full determinism in case of ties
    ranked = (
        auc_df
        .sort_values(
            ["median_auc", "std_auc"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last"
        )
        .copy()
    )

    ranked["rank"] = range(1, len(ranked) + 1)

    logging.info(
        "Ranking %d survivors and selecting top %d.",
        len(ranked),
        top_k
    )

    return ranked.head(top_k)


def plot_top_features(top_features: pd.DataFrame, output_path: str):
    """
    Create a horizontal bar chart of the top 20 features with error bars.
    """
    plot_df = top_features.head(20).iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(
        plot_df.index.astype(str),
        plot_df["median_auc"],
        xerr=plot_df["std_auc"],
        color="steelblue",
        alpha=0.85
    )

    plt.xlabel("Direction-agnostic median ROC-AUC")
    plt.ylabel("Feature")
    plt.title("Top Features by Within-Device Discriminability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(config["output"].get("dir", "out"), exist_ok=True)

    auc_df = load_phase2a_results(config)

    # Get top_k from config; fallback to 100 if not specified
    top_k = config.get("top_k", 100)
    elite_df = rank_features(auc_df, top_k=top_k)

    # FIX #12: Robustly create output with 'feature' column.
    # Reset index (which holds feature names) and ensure the column is named 'feature'.
    elite_output = elite_df.reset_index()
    # The reset index column might be named 'index' or already 'feature' if
    # the index had a name. We rename it explicitly to 'feature' for clarity.
    elite_output = elite_output.rename(columns={elite_output.columns[0]: "feature"})

    # Save only the feature and rank columns
    elite_output[["feature", "rank"]].to_csv(
        config["output"]["elite_features"],
        index=False
    )

    # Also produce a full ranked list (all survivors with ranks)
    full_ranked = (
        auc_df
        .sort_values(
            ["median_auc", "std_auc"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last"
        )
        .reset_index()
        .rename(columns={auc_df.index.name if auc_df.index.name else "index": "feature"})
    )
    # If the index had no name, the column will be named 'index'; rename it.
    if "index" in full_ranked.columns:
        full_ranked = full_ranked.rename(columns={"index": "feature"})

    full_ranked["rank"] = range(1, len(full_ranked) + 1)

    full_ranked.to_csv(
        config["output"]["behavioral_ranking"],
        index=False
    )

    # Plot top 20
    plot_path = os.path.join(
        config["output"].get("dir", "out"),
        "phase2b_top20.png"
    )
    plot_top_features(elite_df, plot_path)

    logging.info(
        "Saved %d elite features to %s.",
        len(elite_df),
        config["output"]["elite_features"]
    )


if __name__ == "__main__":
    main()