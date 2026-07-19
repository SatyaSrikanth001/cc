# phase4_consensus.py
"""
Phase 4: Consensus ranking using:

    - Phase 2B AUC ranking
    - Fisher score
    - Mutual information
    - ReliefF (Deterministic)
    - Laplacian score

All scores are computed using development data only.
"""

import logging
import os
from typing import List
import inspect  # Added for version-proof ReliefF parameter detection

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import diags
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import kneighbors_graph

from utils.data_loader import load_split_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def fisher_score(df: pd.DataFrame, feature: str) -> float:
    genuine = df.loc[df["is_genuine"] == 1, feature].dropna()
    impostor = df.loc[df["is_genuine"] == 0, feature].dropna()

    if len(genuine) < 2 or len(impostor) < 2:
        return 0.0

    mean_difference = genuine.mean() - impostor.mean()
    denominator = (
        genuine.var(ddof=0) +
        impostor.var(ddof=0)
    )

    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0

    return float((mean_difference ** 2) / denominator)


def mutual_information_score(
    df: pd.DataFrame,
    feature: str
) -> float:
    temp = df[[feature, "is_genuine"]].dropna()

    if len(temp) < 5 or temp["is_genuine"].nunique() < 2:
        return 0.0

    X = temp[[feature]].to_numpy()
    y = temp["is_genuine"].to_numpy()

    try:
        value = mutual_info_classif(
            X,
            y,
            random_state=42,
            n_neighbors=min(3, len(temp) - 1)
        )[0]

        return float(value) if np.isfinite(value) else 0.0

    except Exception as exc:
        logging.debug("MI failed for %s: %s", feature, exc)
        return 0.0


def relieff_scores(
    df: pd.DataFrame,
    features: List[str]
) -> np.ndarray:
    """
    Compute ReliefF importance scores with version-proof parameter handling.
    FIX: Removed unsupported 'n_iterations' and uses inspect to detect
    supported parameters like 'random_state' and 'n_jobs'.
    """
    try:
        from skrebate import ReliefF
    except ImportError as exc:
        raise ImportError(
            "The 'skrebate' package is required to execute ReliefF consensus scoring in Phase 4. "
            "Please install it using: 'pip install skrebate'"
        ) from exc

    temp = df[features + ["is_genuine"]].dropna()

    if len(temp) < 10 or temp["is_genuine"].nunique() < 2:
        logging.warning("Insufficient data for ReliefF; returning zeros.")
        return np.zeros(len(features))

    try:
        X = temp[features].to_numpy(dtype=float)
        y = temp["is_genuine"].to_numpy(dtype=int)

        # Detect which parameters the installed ReliefF constructor supports
        relief_params = inspect.signature(ReliefF.__init__).parameters

        kwargs = {
            "n_neighbors": min(10, len(temp) - 1),
            "n_features_to_select": len(features),
        }
        # Only add optional parameters if supported by the installed version
        if "n_jobs" in relief_params:
            kwargs["n_jobs"] = 1
        if "random_state" in relief_params:
            kwargs["random_state"] = 42
        # Note: 'n_iterations' is not supported in some versions; we deliberately omit it.
        # If your version requires it, you must update skrebate or add it back.

        model = ReliefF(**kwargs)
        model.fit(X, y)

        scores = np.asarray(model.feature_importances_, dtype=float)

        if len(scores) != len(features):
            logging.warning("ReliefF returned wrong number of scores; using zeros.")
            return np.zeros(len(features))

        scores[~np.isfinite(scores)] = 0.0
        return scores

    except Exception as exc:
        logging.warning("ReliefF failed for a user: %s; using zeros.", exc)
        return np.zeros(len(features))


def laplacian_scores(
    df: pd.DataFrame,
    features: List[str]
) -> np.ndarray:
    genuine = df.loc[
        df["is_genuine"] == 1,
        features
    ].dropna()

    if len(genuine) < 5:
        logging.warning("Too few genuine samples for Laplacian; returning zeros.")
        return np.zeros(len(features))

    try:
        X = genuine.to_numpy(dtype=float)

        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1.0
        X_scaled = (X - means) / stds

        k = min(5, len(genuine) - 1)

        W = kneighbors_graph(
            X_scaled,
            n_neighbors=k,
            mode="connectivity",
            include_self=False
        )

        W = ((W + W.T) / 2).tocsr()

        degree = np.asarray(W.sum(axis=1)).ravel()
        D = diags(degree).toarray()
        L = D - W.toarray()

        output = np.zeros(len(features))

        for j in range(X.shape[1]):
            values = X_scaled[:, j]
            centered = values - values.mean()

            denominator = centered @ D @ centered

            if denominator <= 0 or not np.isfinite(denominator):
                output[j] = 0.0
                continue

            numerator = centered @ L @ centered
            laplacian_score = numerator / denominator

            # FIX #14: Use 1/(1+score) to map to [0,1] where 1 is best
            # (lower laplacian_score => higher importance)
            importance = 1.0 / (1.0 + laplacian_score)
            output[j] = importance if np.isfinite(importance) else 0.0

        return output

    except Exception as exc:
        logging.warning("Laplacian score failed for a user: %s; using zeros.", exc)
        return np.zeros(len(features))


def normalize_user_columns(scores: pd.DataFrame) -> pd.DataFrame:
    result = scores.astype(float).copy()

    for column in result.columns:
        values = result[column]
        low = values.min()
        high = values.max()

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            result[column] = 0.0
        else:
            result[column] = (values - low) / (high - low)

    return result


def rank_descending(scores: pd.Series) -> pd.Series:
    return scores.rank(
        ascending=False,
        method="min"
    ).astype(float)


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    os.makedirs(config["output"].get("dir", "out"), exist_ok=True)

    survivors_path = config["output"]["filtered_features"]

    if not os.path.exists(survivors_path):
        raise FileNotFoundError(
            "Phase 2A survivors are missing. Run Phase 2A first."
        )

    survivors = pd.read_csv(survivors_path, dtype={"feature": str})["feature"].tolist()

    if not survivors:
        raise RuntimeError("Phase 4 received zero survivor features.")

    dev_data, _, _ = load_split_data(config)
    users = list(dev_data.keys())

    fisher = pd.DataFrame(index=survivors, columns=users, dtype=float)
    mutual_info = pd.DataFrame(index=survivors, columns=users, dtype=float)
    relieff = pd.DataFrame(index=survivors, columns=users, dtype=float)
    laplacian = pd.DataFrame(index=survivors, columns=users, dtype=float)

    for user, df in dev_data.items():
        # FIX #5: Use only features actually present in this user's df
        available = [
            f for f in survivors
            if f in df.columns
        ]
        # For missing features, we will leave 0.0 after filling

        # Compute Fisher & MI per feature
        for feature in survivors:
            if feature in available:
                fisher.loc[feature, user] = fisher_score(df, feature)
                mutual_info.loc[feature, user] = mutual_information_score(df, feature)
            else:
                fisher.loc[feature, user] = 0.0
                mutual_info.loc[feature, user] = 0.0

        # Compute ReliefF & Laplacian on available features only
        relief_values = relieff_scores(df, available)
        laplacian_values = laplacian_scores(df, available)

        # Map back to full survivors list (missing features get 0)
        for idx, feature in enumerate(survivors):
            if feature in available:
                pos = available.index(feature)
                relieff.loc[feature, user] = relief_values[pos]
                laplacian.loc[feature, user] = laplacian_values[pos]
            else:
                relieff.loc[feature, user] = 0.0
                laplacian.loc[feature, user] = 0.0

    # Aggregate: per-user min-max normalization, then median across users
    fisher_median = normalize_user_columns(fisher).median(axis=1)
    mi_median = normalize_user_columns(mutual_info).median(axis=1)
    relieff_median = normalize_user_columns(relieff).median(axis=1)
    laplacian_median = normalize_user_columns(laplacian).median(axis=1)

    # Load Phase 2B ranking
    ranking_path = config["output"]["behavioral_ranking"]

    if not os.path.exists(ranking_path):
        raise FileNotFoundError(
            "Phase 2B ranking is missing. Run Phase 2B first."
        )

    ranking_df = pd.read_csv(ranking_path, dtype={"feature": str})

    if "feature" not in ranking_df.columns or "rank" not in ranking_df.columns:
        raise ValueError(
            "Phase 2B ranking must contain 'feature' and 'rank' columns."
        )

    ranking_df = ranking_df.set_index("feature")

    phase2b_rank = ranking_df["rank"].reindex(survivors)
    worst_rank = len(survivors) + 1
    phase2b_rank = phase2b_rank.fillna(worst_rank)

    # Convert scores to ranks (descending: high score = high rank = low number)
    fisher_rank = rank_descending(fisher_median)
    mi_rank = rank_descending(mi_median)
    relieff_rank = rank_descending(relieff_median)
    laplacian_rank = rank_descending(laplacian_median)

    # FIX #2: Explicitly reindex all rank Series to the same order (survivors)
    # to guarantee alignment in the weighted sum.
    fisher_rank = fisher_rank.reindex(survivors)
    mi_rank = mi_rank.reindex(survivors)
    relieff_rank = relieff_rank.reindex(survivors)
    laplacian_rank = laplacian_rank.reindex(survivors)

    weights = config["consensus"]["weights"]

    w_auc = weights["our_phase2b"]
    w_fisher = weights["fisher"]
    w_mi = weights["mutual_info"]
    w_relieff = weights["relieff"]
    w_laplacian = weights["laplacian"]

    total_weight = (
        w_auc +
        w_fisher +
        w_mi +
        w_relieff +
        w_laplacian
    )

    if total_weight <= 0:
        raise ValueError("Consensus weights must sum to a positive value.")

    consensus_value = (
        w_auc * phase2b_rank +
        w_fisher * fisher_rank +
        w_mi * mi_rank +
        w_relieff * relieff_rank +
        w_laplacian * laplacian_rank
    ) / total_weight

    result = pd.DataFrame({
        "feature": survivors,
        "consensus_rank_value": consensus_value.values,
        "phase2b_rank": phase2b_rank.values,
        "fisher_rank": fisher_rank.values,
        "mutual_info_rank": mi_rank.values,
        "relieff_rank": relieff_rank.values,
        "laplacian_rank": laplacian_rank.values
    })

    result = result.sort_values(
        "consensus_rank_value",
        ascending=True
    ).reset_index(drop=True)

    result["final_rank"] = np.arange(1, len(result) + 1)

    result.to_csv(
        config["output"]["consensus_ranking"],
        index=False
    )

    final_k = min(
        config.get("final_k", 40),
        len(result)
    )

    selected = result.head(final_k)["feature"].tolist()

    # Save final selected features (for Phase 5)
    pd.DataFrame({"feature": selected}).to_csv(
        config["output"]["final_selected_features"],
        index=False
    )

    logging.info(
        "Phase 4 selected %d features.",
        len(selected)
    )

    print("\n--- Top 10 consensus features ---")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()