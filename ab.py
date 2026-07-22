#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnose_features.py

Exhaustive, per-user, per-feature quality-control diagnostics for behavioral
biometric features before training a One-Class SVM.

Each user is processed completely independently. No statistics are pooled
across users.

Inputs
------
./selected_features.json
./features/new_app/{username}_training_sessions.csv
./features/new_app/{username}_testing_sessions.csv

Outputs
-------
./diagnostics_output/{username}_feature_diagnostics.csv
./diagnostics_output/{username}_session_level_outliers.csv

Health-score formula
--------------------
Start at 100:
    -15 for every red-flagged metric
    -8  for every blue warning
    -5  for every explicitly configured borderline condition

The score is clipped to [0, 100].

Only independently assessed quality metrics contribute to the score.
Summary fields such as total_red_flags, total_blue_flags, verdict, and
overall_health_score do not recursively affect the score.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

try:
    from diptest import diptest

    DIPTEST_AVAILABLE = True
except ImportError:
    DIPTEST_AVAILABLE = False

try:
    from statsmodels.stats.diagnostic import lilliefors

    LILLIEFORS_AVAILABLE = True
except ImportError:
    LILLIEFORS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

USERS = ["srikanth", "Samarth", "harshit"]

FEATURES_JSON = Path("./selected_features.json")
FEATURE_DIRECTORY = Path("./features/new_app")
OUTPUT_DIRECTORY = Path("./diagnostics_output")

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000
EPSILON = 1e-10
MEAN_NEAR_ZERO = 1e-6
MIN_NORMALITY_SAMPLE_SIZE = 8

RED = "🔴"
BLUE = "🔵"
GREEN = "🟢"

REQUIRED_METRICS = [
    # Basic descriptive
    "count_valid",
    "count_nan",
    "count_inf",
    "mean",
    "median",
    "std_dev",
    "min",
    "max",
    "range",
    "iqr",

    # Shape
    "skewness",
    "kurtosis",
    "is_unimodal",
    "num_peaks",
    "cv",
    "median_to_mean_ratio",

    # Robust spread
    "mad",
    "robust_cv",
    "mad_to_std_ratio",
    "snr_estimate",

    # Outlier detection
    "modified_z_max",
    "modified_z_mean",
    "modified_z_outlier_count",
    "modified_z_outlier_pct",
    "iqr_outlier_count",
    "iqr_outlier_pct",
    "percentile_2.5",
    "percentile_97.5",
    "pct_outside_95_ci",
    "grubbs_statistic",
    "grubbs_p_value",
    "is_grubbs_outlier",

    # Advanced statistical tests
    "shapiro_wilk_stat",
    "shapiro_wilk_p",
    "is_normal_shapiro",
    "dagostino_stat",
    "dagostino_p",
    "anderson_stat",
    "anderson_crit_5pct",
    "is_normal_anderson",
    "levenes_stat",
    "levenes_p",

    # Temporal/drift
    "linear_trend_slope",
    "linear_trend_p",
    "is_drifting",
    "mann_kendall_tau",
    "mann_kendall_p",
    "first_half_vs_second_half_median_diff",

    # Extreme values
    "max_to_median_ratio",
    "min_to_median_ratio",
    "has_near_zero_denominator_evidence",
    "has_numerical_overflow",
    "has_constant_segment",
    "entropy_of_binned_values",

    # Distribution fit
    "ks_normal_stat",
    "ks_normal_p",
    "cvm_normal_stat",
    "cvm_normal_p",

    # Cross-session consistency
    "session_to_session_change_mean",
    "session_to_session_change_std",
    "autocorrelation_lag1",
    "autocorrelation_lag2",

    # Feature quality score
    "overall_health_score",
    "verdict",

    # Flag summary
    "total_red_flags",
    "total_blue_flags",
    "flag_explanation",
]

ADDITIONAL_METRICS = [
    "isolation_forest_score_min",
    "isolation_forest_outlier_count",
    "lof_score_max",
    "lof_outlier_count",
    "mahalanobis_distance_max",
    "mahalanobis_outlier_count",
    "bootstrap_median_ci_low",
    "bootstrap_median_ci_high",
    "bootstrap_median_ci_width",
    "bootstrap_ci_too_wide",
    "lilliefors_stat",
    "lilliefors_p",
    "jarque_bera_stat",
    "jarque_bera_p",
    "successive_ratio_variance",
    "proportion_unique",
    "rank_dispersion_index",
    "peak_over_threshold_count",
    "peak_over_threshold_pct",
]

OUTPUT_METRICS = (
    REQUIRED_METRICS[:-4]
    + ADDITIONAL_METRICS
    + REQUIRED_METRICS[-4:]
)

NON_SCORING_METRICS = {
    "mean",
    "median",
    "min",
    "max",
    "range",
    "iqr",
    "median_to_mean_ratio",
    "mad",
    "modified_z_mean",
    "percentile_2.5",
    "percentile_97.5",
    "grubbs_statistic",
    "shapiro_wilk_stat",
    "dagostino_stat",
    "anderson_stat",
    "anderson_crit_5pct",
    "levenes_stat",
    "linear_trend_slope",
    "mann_kendall_tau",
    "first_half_vs_second_half_median_diff",
    "ks_normal_stat",
    "cvm_normal_stat",
    "session_to_session_change_mean",
    "session_to_session_change_std",
    "autocorrelation_lag1",
    "autocorrelation_lag2",
    "isolation_forest_score_min",
    "lof_score_max",
    "mahalanobis_distance_max",
    "bootstrap_median_ci_low",
    "bootstrap_median_ci_high",
    "bootstrap_median_ci_width",
    "lilliefors_stat",
    "jarque_bera_stat",
    "successive_ratio_variance",
    "rank_dispersion_index",
    "peak_over_threshold_count",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result
    except (TypeError, ValueError):
        return default


def finite_or_nan(value: Any) -> float:
    value = safe_float(value)
    return value if np.isfinite(value) else np.nan


def round_numeric(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if np.isposinf(value):
            return np.inf
        if np.isneginf(value):
            return -np.inf
        return round(float(value), digits)

    return value


def value_to_string(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"

    if value is None:
        return "NA"

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "NA"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{float(value):.6f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return str(value)


def format_metric_cell(metric: str, value: Any, status: str) -> str:
    emoji = {
        "red": RED,
        "blue": BLUE,
        "green": GREEN,
    }.get(status, GREEN)

    return f"{emoji} {metric}={value_to_string(round_numeric(value))}"


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = np.nan
) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    if abs(denominator) <= EPSILON:
        return default
    return float(numerator / denominator)


def status_rank(status: str) -> int:
    return {"green": 0, "blue": 1, "red": 2}.get(status, 0)


def merge_status(existing: str, new_status: str) -> str:
    return new_status if status_rank(new_status) > status_rank(existing) else existing


def safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return np.nan

    left = values[:-lag]
    right = values[lag:]

    if left.size < 2 or np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return np.nan

    return float(np.corrcoef(left, right)[0, 1])


def calculate_entropy(values: np.ndarray, bins: int = 20) -> float:
    if values.size == 0:
        return np.nan

    if np.allclose(values, values[0], rtol=0.0, atol=EPSILON):
        return 0.0

    counts, _ = np.histogram(values, bins=bins)
    counts = counts[counts > 0]

    if counts.size == 0:
        return 0.0

    probabilities = counts / counts.sum()
    return float(stats.entropy(probabilities, base=2))


def mann_kendall_test(values: np.ndarray) -> Tuple[float, float]:
    """
    Direct Mann-Kendall trend test with tie correction.

    Returns Kendall tau and a two-sided normal-approximation p-value.
    """
    n = values.size
    if n < 3:
        return np.nan, np.nan

    s_stat = 0
    for i in range(n - 1):
        s_stat += int(np.sign(values[i + 1:] - values[i]).sum())

    unique_values, tie_counts = np.unique(values, return_counts=True)
    del unique_values

    variance_s = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    ) / 18.0

    denominator = n * (n - 1) / 2.0
    tau = float(s_stat / denominator) if denominator > 0 else np.nan

    if variance_s <= EPSILON:
        return tau, 1.0

    if s_stat > 0:
        z_stat = (s_stat - 1.0) / math.sqrt(variance_s)
    elif s_stat < 0:
        z_stat = (s_stat + 1.0) / math.sqrt(variance_s)
    else:
        z_stat = 0.0

    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    return tau, p_value


def grubbs_test(values: np.ndarray) -> Tuple[float, float, bool, Optional[int]]:
    """
    Two-sided Grubbs test for one outlier.

    The returned index is local to the supplied valid-values array.
    """
    n = values.size
    if n < 3:
        return np.nan, np.nan, False, None

    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=1))

    if std_value <= EPSILON:
        return 0.0, 1.0, False, None

    deviations = np.abs(values - mean_value)
    extreme_index = int(np.argmax(deviations))
    grubbs_statistic = float(deviations[extreme_index] / std_value)

    denominator = (n - 1) ** 2 - n * (grubbs_statistic ** 2)
    numerator = n * (n - 2) * (grubbs_statistic ** 2)

    if denominator <= 0:
        p_value = 0.0
    else:
        t_squared = numerator / denominator
        t_value = math.sqrt(max(t_squared, 0.0))
        p_value = float(min(1.0, 2.0 * n * stats.t.sf(t_value, df=n - 2)))

    return grubbs_statistic, p_value, bool(p_value < 0.05), extreme_index


def detect_kde_modes(values: np.ndarray) -> Tuple[bool, int]:
    """
    Determine unimodality and number of modes.

    If diptest is installed, its p-value is used for the unimodality decision.
    KDE peak counting is always used to estimate the number of modes.
    """
    if values.size < 3 or np.unique(values).size <= 1:
        return True, 1

    value_min = float(np.min(values))
    value_max = float(np.max(values))

    if abs(value_max - value_min) <= EPSILON:
        return True, 1

    try:
        grid = np.linspace(value_min, value_max, 200)
        kde = stats.gaussian_kde(values)
        density = kde(grid)

        density_range = float(np.max(density) - np.min(density))
        prominence = max(0.05 * density_range, EPSILON)

        peak_indices, _ = find_peaks(density, prominence=prominence)
        num_peaks = max(1, int(peak_indices.size))
    except Exception:
        num_peaks = 1

    if DIPTEST_AVAILABLE:
        try:
            _, dip_p = diptest(np.asarray(values, dtype=float))
            is_unimodal = bool(float(dip_p) >= 0.05)
        except Exception:
            is_unimodal = num_peaks <= 1
    else:
        is_unimodal = num_peaks <= 1

    return is_unimodal, num_peaks


def bootstrap_median_ci(
    values: np.ndarray,
    iterations: int,
    seed: int
) -> Tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        low=0,
        high=values.size,
        size=(iterations, values.size),
    )
    bootstrap_samples = values[indices]
    medians = np.median(bootstrap_samples, axis=1)

    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high), float(high - low)


def calculate_modified_z_scores(
    values: np.ndarray,
    median_value: float,
    mad_value: float
) -> np.ndarray:
    """
    Modified Z-score based on median/MAD.

    Degenerate MAD handling:
    - If MAD > 0, use the standard 0.6745 * deviation / MAD formula.
    - If MAD == 0 and every value equals the median, all scores are zero.
    - If MAD == 0 but departures from the median exist, those departures
      receive infinite scores. This prevents a single catastrophic value
      from escaping detection when the majority of sessions are identical.
    """
    deviations = np.abs(values - median_value)

    if mad_value > EPSILON:
        return 0.6745 * deviations / mad_value

    scores = np.zeros(values.size, dtype=float)
    scores[deviations > EPSILON] = np.inf
    return scores


def run_isolation_forest(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = values.size

    if n < 4 or np.unique(values).size <= 1:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)

    model = IsolationForest(
        n_estimators=50,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    matrix = values.reshape(-1, 1)
    model.fit(matrix)

    scores = model.decision_function(matrix)
    flags = scores < -0.5
    return scores.astype(float), flags.astype(bool)


def run_lof(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = values.size

    if n < 3 or np.unique(values).size <= 1:
        return np.ones(n, dtype=float), np.zeros(n, dtype=bool)

    neighbors = min(10, n - 1)
    model = LocalOutlierFactor(n_neighbors=neighbors)
    model.fit_predict(values.reshape(-1, 1))

    lof_scores = -model.negative_outlier_factor_
    flags = lof_scores > 2.0
    return lof_scores.astype(float), flags.astype(bool)


def robust_mahalanobis_1d(
    values: np.ndarray,
    median_value: float,
    mad_value: float,
    std_value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-dimensional robust Mahalanobis-like distance.

    Uses 1.4826*MAD as the robust scale. If MAD is zero, standard deviation
    is used only as a fallback. If both are zero, distances are zero.
    """
    robust_scale = 1.4826 * mad_value

    if robust_scale <= EPSILON:
        robust_scale = std_value

    if robust_scale <= EPSILON:
        distances = np.zeros(values.size, dtype=float)
    else:
        distances = np.abs(values - median_value) / robust_scale

    flags = distances > 3.5
    return distances.astype(float), flags.astype(bool)


# =============================================================================
# STATUS AND EXPLANATION LOGIC
# =============================================================================

def assign_metric_statuses(
    metrics: Dict[str, Any],
    total_sessions: int,
    missing_column: bool,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    statuses = {metric: "green" for metric in OUTPUT_METRICS}
    explanations: Dict[str, str] = {}

    count_valid = int(metrics["count_valid"])
    count_nan = int(metrics["count_nan"])
    count_inf = int(metrics["count_inf"])
    mean_value = metrics["mean"]
    median_value = metrics["median"]
    std_value = metrics["std_dev"]
    iqr_value = metrics["iqr"]

    def set_flag(metric: str, status: str, explanation: Optional[str] = None) -> None:
        statuses[metric] = merge_status(statuses.get(metric, "green"), status)
        if status == "red" and explanation:
            explanations[metric] = explanation

    if missing_column:
        set_flag(
            "count_valid",
            "red",
            "The feature column is missing from at least one source CSV, so no "
            "complete per-session feature series is available for this user.",
        )

    if count_valid < MIN_NORMALITY_SAMPLE_SIZE:
        set_flag(
            "count_valid",
            "red",
            f"Only {count_valid} valid sessions are available out of "
            f"{total_sessions}; fewer than {MIN_NORMALITY_SAMPLE_SIZE} values "
            "is insufficient for reliable feature-quality estimation.",
        )
    elif count_valid < total_sessions:
        set_flag("count_valid", "blue")

    if count_nan > 0:
        percentage = 100.0 * count_nan / max(total_sessions, 1)
        set_flag(
            "count_nan",
            "red",
            f"{count_nan} sessions ({percentage:.1f}%) contain NaN values. "
            "This indicates failed feature extraction or undefined arithmetic "
            "and would require imputation before OCSVM scaling.",
        )

    if count_inf > 0:
        percentage = 100.0 * count_inf / max(total_sessions, 1)
        set_flag(
            "count_inf",
            "red",
            f"{count_inf} sessions ({percentage:.1f}%) contain positive or "
            "negative infinity, indicating numerical overflow or division by "
            "a zero or near-zero denominator.",
        )

    if np.isfinite(std_value):
        median_scale = max(abs(median_value), EPSILON)
        std_to_median = std_value / median_scale

        if std_value <= EPSILON:
            set_flag(
                "std_dev",
                "red",
                f"Feature is constant with standard deviation {std_value:.6f} "
                f"across {count_valid} valid sessions and carries no genuine-"
                "session variability for OCSVM boundary learning.",
            )
        elif std_value > 1e8:
            set_flag(
                "std_dev",
                "red",
                f"Standard deviation is {std_value:.6f}, exceeding the 1e8 "
                "numerical-stability threshold and likely causing StandardScaler "
                "and RBF-distance distortion.",
            )
        elif std_to_median > 1000:
            set_flag(
                "std_dev",
                "red",
                f"Standard deviation ({std_value:.6f}) is "
                f"{std_to_median:.2f} times the absolute median "
                f"({abs(median_value):.6f}), indicating a catastrophically "
                "unstable scale likely driven by extreme sessions.",
            )

    skew_value = metrics["skewness"]
    if np.isfinite(skew_value):
        if abs(skew_value) > 3:
            set_flag(
                "skewness",
                "red",
                f"Absolute skewness is {abs(skew_value):.6f}, above 3.0. "
                "The genuine-session distribution is severely asymmetric and "
                "may produce an unbalanced OCSVM boundary.",
            )
        elif abs(skew_value) > 1:
            set_flag("skewness", "blue")

    kurtosis_value = metrics["kurtosis"]
    if np.isfinite(kurtosis_value):
        if kurtosis_value > 5:
            set_flag(
                "kurtosis",
                "red",
                f"Excess kurtosis is {kurtosis_value:.6f}, above 5.0, showing "
                "heavy tails or extreme concentration that can inflate the "
                "genuine OCSVM acceptance region.",
            )
        elif kurtosis_value > 3:
            set_flag("kurtosis", "blue")

    if metrics["is_unimodal"] is False:
        set_flag(
            "is_unimodal",
            "red",
            f"The per-user distribution is not unimodal; KDE/dip-test analysis "
            f"detected {metrics['num_peaks']} modes. Multiple genuine clusters "
            "can cause unstable one-class boundaries.",
        )

    num_peaks = metrics["num_peaks"]
    if num_peaks > 2:
        set_flag(
            "num_peaks",
            "red",
            f"KDE detected {num_peaks} distribution peaks, indicating multiple "
            "genuine behavioral regimes rather than one stable user pattern.",
        )
    elif num_peaks == 2:
        set_flag("num_peaks", "blue")

    cv_value = metrics["cv"]
    if abs(mean_value) < MEAN_NEAR_ZERO:
        statuses["cv"] = "green"
    elif np.isfinite(cv_value):
        if cv_value > 1.5:
            set_flag(
                "cv",
                "red",
                f"Coefficient of variation is {cv_value:.6f}, meaning standard "
                "deviation is more than 1.5 times the absolute mean. This feature "
                "is highly inconsistent across genuine sessions.",
            )
        elif cv_value > 0.5:
            set_flag("cv", "blue")

    robust_cv_value = metrics["robust_cv"]
    if np.isfinite(robust_cv_value):
        if robust_cv_value > 1.0:
            set_flag(
                "robust_cv",
                "red",
                f"Robust CV is {robust_cv_value:.6f}; MAD exceeds the absolute "
                "median, showing high genuine-session variability even after "
                "reducing sensitivity to extreme values.",
            )
        elif robust_cv_value > 0.5:
            set_flag("robust_cv", "blue")

    mad_std_ratio = metrics["mad_to_std_ratio"]
    if np.isfinite(mad_std_ratio):
        if mad_std_ratio < 0.1 and std_value > EPSILON:
            set_flag(
                "mad_to_std_ratio",
                "red",
                f"MAD-to-standard-deviation ratio is {mad_std_ratio:.6f}, below "
                "0.1. Standard deviation is dominated by heavy tails or isolated "
                "extreme sessions.",
            )
        elif (
            0.1 <= mad_std_ratio < 0.2
            or 0.2 <= mad_std_ratio < 0.4
            or mad_std_ratio > 0.8
        ):
            set_flag("mad_to_std_ratio", "blue")

    snr_value = metrics["snr_estimate"]
    if np.isfinite(snr_value):
        if snr_value < 0.5:
            set_flag(
                "snr_estimate",
                "red",
                f"Signal-to-noise estimate is {snr_value:.6f}, below 0.5. "
                "Session variability is more than twice the feature's typical "
                "mean magnitude.",
            )
        elif snr_value < 1.0:
            set_flag("snr_estimate", "blue")

    modified_z_max = metrics["modified_z_max"]
    if np.isinf(modified_z_max) or (
        np.isfinite(modified_z_max) and modified_z_max > 5
    ):
        set_flag(
            "modified_z_max",
            "red",
            f"Maximum absolute modified Z-score is "
            f"{value_to_string(modified_z_max)}, above 5.0. At least one session "
            "is catastrophically far from the user's median/MAD baseline.",
        )
    elif np.isfinite(modified_z_max) and modified_z_max > 3.5:
        set_flag("modified_z_max", "blue")

    modified_pct = metrics["modified_z_outlier_pct"]
    modified_count = metrics["modified_z_outlier_count"]
    if modified_pct > 5:
        set_flag(
            "modified_z_outlier_pct",
            "red",
            f"{modified_count} sessions ({modified_pct:.1f}%) exceed the "
            "modified Z-score threshold of 3.5, suggesting an unstable feature "
            "that can bloat the genuine OCSVM boundary.",
        )
        set_flag(
            "modified_z_outlier_count",
            "red",
            f"{modified_count} sessions are modified-Z outliers; the affected "
            f"proportion is {modified_pct:.1f}%, above the 5% rejection threshold.",
        )
    elif modified_pct > 2:
        set_flag("modified_z_outlier_pct", "blue")
        set_flag("modified_z_outlier_count", "blue")

    iqr_pct = metrics["iqr_outlier_pct"]
    iqr_count = metrics["iqr_outlier_count"]
    if iqr_pct > 10:
        set_flag(
            "iqr_outlier_pct",
            "red",
            f"{iqr_count} sessions ({iqr_pct:.1f}%) lie outside the 1.5×IQR "
            "fences, indicating excessive cross-session instability.",
        )
        set_flag(
            "iqr_outlier_count",
            "red",
            f"{iqr_count} sessions are outside the 1.5×IQR fences, representing "
            f"{iqr_pct:.1f}% of valid sessions.",
        )
    elif iqr_pct > 5:
        set_flag("iqr_outlier_pct", "blue")
        set_flag("iqr_outlier_count", "blue")

    percentile_pct = metrics["pct_outside_95_ci"]
    if percentile_pct > 10:
        set_flag(
            "pct_outside_95_ci",
            "red",
            f"{percentile_pct:.1f}% of sessions lie outside the empirical "
            "2.5th–97.5th percentile interval, above the 10% rejection threshold.",
        )
    elif percentile_pct > 5:
        set_flag("pct_outside_95_ci", "blue")

    grubbs_p = metrics["grubbs_p_value"]
    if metrics["is_grubbs_outlier"]:
        set_flag(
            "is_grubbs_outlier",
            "red",
            f"Grubbs' test rejects the no-outlier hypothesis at α=0.05 "
            f"(G={metrics['grubbs_statistic']:.6f}, p={grubbs_p:.6f}).",
        )

    if np.isfinite(grubbs_p):
        if grubbs_p < 0.01:
            set_flag(
                "grubbs_p_value",
                "red",
                f"Grubbs p-value is {grubbs_p:.6f}, below 0.01, providing strong "
                "evidence that the most extreme session is not from the same "
                "genuine-session distribution.",
            )
        elif grubbs_p < 0.05:
            set_flag("grubbs_p_value", "blue")

    shapiro_p = metrics["shapiro_wilk_p"]
    if metrics["is_normal_shapiro"] is False:
        set_flag(
            "is_normal_shapiro",
            "red",
            f"Shapiro-Wilk rejects normality with p={shapiro_p:.6f}. Strong "
            "non-normality may indicate heavy tails, multimodality, or unstable "
            "feature extraction.",
        )

    if np.isfinite(shapiro_p):
        if shapiro_p < 0.01:
            set_flag(
                "shapiro_wilk_p",
                "red",
                f"Shapiro-Wilk p-value is {shapiro_p:.6f}, below 0.01, showing "
                "strong evidence against a stable Gaussian-like distribution.",
            )
        elif shapiro_p < 0.05:
            set_flag("shapiro_wilk_p", "blue")

    dagostino_p = metrics["dagostino_p"]
    if np.isfinite(dagostino_p):
        if dagostino_p < 0.01:
            set_flag(
                "dagostino_p",
                "red",
                f"D'Agostino-Pearson p-value is {dagostino_p:.6f}, below 0.01, "
                "showing significant skewness and/or kurtosis.",
            )
        elif dagostino_p < 0.05:
            set_flag("dagostino_p", "blue")

    if metrics["is_normal_anderson"] is True:
        set_flag(
            "is_normal_anderson",
            "red",
            f"Anderson-Darling rejects normality: statistic "
            f"{metrics['anderson_stat']:.6f} exceeds the 5% critical value "
            f"{metrics['anderson_crit_5pct']:.6f}.",
        )

    levene_p = metrics["levenes_p"]
    if np.isfinite(levene_p):
        if levene_p < 0.01:
            set_flag(
                "levenes_p",
                "red",
                f"Levene p-value is {levene_p:.6f}, indicating a significant "
                "variance change between the first and second halves of sessions.",
            )
        elif levene_p < 0.05:
            set_flag("levenes_p", "blue")

    trend_p = metrics["linear_trend_p"]
    if np.isfinite(trend_p):
        if trend_p < 0.01:
            set_flag(
                "is_drifting",
                "red",
                f"Linear drift is highly significant "
                f"(slope={metrics['linear_trend_slope']:.6f}, p={trend_p:.6f}). "
                "The genuine feature baseline changes systematically over time.",
            )
            set_flag(
                "linear_trend_p",
                "red",
                f"Linear trend p-value is {trend_p:.6f}, below 0.01, confirming "
                "strong temporal drift.",
            )
        elif trend_p < 0.05:
            set_flag("is_drifting", "blue")
            set_flag("linear_trend_p", "blue")

    mk_p = metrics["mann_kendall_p"]
    if np.isfinite(mk_p):
        if mk_p < 0.01:
            set_flag(
                "mann_kendall_p",
                "red",
                f"Mann-Kendall p-value is {mk_p:.6f}, below 0.01 "
                f"(tau={metrics['mann_kendall_tau']:.6f}), showing a strong "
                "monotonic temporal trend.",
            )
        elif mk_p < 0.05:
            set_flag("mann_kendall_p", "blue")

    max_ratio = metrics["max_to_median_ratio"]
    if np.isfinite(max_ratio):
        if abs(max_ratio) > 1e6:
            set_flag(
                "max_to_median_ratio",
                "red",
                f"Maximum-to-median ratio is {max_ratio:.6f}, above 1e6. The "
                "largest session value is incompatible with the user's typical "
                "scale and suggests overflow or division by a near-zero value.",
            )
        elif abs(max_ratio) > 1e3:
            set_flag("max_to_median_ratio", "blue")

    min_ratio = metrics["min_to_median_ratio"]
    if np.isfinite(min_ratio):
        if abs(min_ratio) > 1e6:
            set_flag(
                "min_to_median_ratio",
                "red",
                f"Absolute minimum-to-median ratio is {abs(min_ratio):.6f}, above "
                "1e6, indicating a catastrophically scaled negative or low-end "
                "session value.",
            )
        elif abs(min_ratio) > 1e3:
            set_flag("min_to_median_ratio", "blue")

    if metrics["has_near_zero_denominator_evidence"]:
        set_flag(
            "has_near_zero_denominator_evidence",
            "red",
            "At least one absolute value exceeds 1e6 while another non-zero "
            "absolute value is below 1e-3. This scale split is consistent with "
            "division by a near-zero denominator.",
        )

    if metrics["has_numerical_overflow"]:
        set_flag(
            "has_numerical_overflow",
            "red",
            f"At least one session has absolute magnitude above 1e10 "
            f"(observed maximum magnitude "
            f"{max(abs(metrics['min']), abs(metrics['max'])):.6f}), indicating "
            "numerical overflow or a broken feature formula.",
        )

    if metrics["has_constant_segment"]:
        set_flag(
            "has_constant_segment",
            "red",
            "More than 50% of valid session values are identical after rounding "
            "to six decimal places. The feature has a dominant constant segment "
            "and very low effective resolution.",
        )

    entropy_value = metrics["entropy_of_binned_values"]
    if np.isfinite(entropy_value):
        if entropy_value < 0.5:
            set_flag(
                "entropy_of_binned_values",
                "red",
                f"Binned Shannon entropy is {entropy_value:.6f}, below 0.5 bits, "
                "showing almost no information content across sessions.",
            )
        elif entropy_value < 1.5:
            set_flag("entropy_of_binned_values", "blue")

    ks_p = metrics["ks_normal_p"]
    if np.isfinite(ks_p):
        if ks_p < 0.01:
            set_flag(
                "ks_normal_p",
                "red",
                f"Kolmogorov-Smirnov normal-fit p-value is {ks_p:.6f}, below "
                "0.01, indicating severe distribution mismatch.",
            )
        elif ks_p < 0.05:
            set_flag("ks_normal_p", "blue")

    cvm_p = metrics["cvm_normal_p"]
    if np.isfinite(cvm_p):
        if cvm_p < 0.01:
            set_flag(
                "cvm_normal_p",
                "red",
                f"Cramér-von Mises normal-fit p-value is {cvm_p:.6f}, below "
                "0.01, indicating strong whole-distribution deviation.",
            )
        elif cvm_p < 0.05:
            set_flag("cvm_normal_p", "blue")

    # Additional advanced-method thresholds.
    if metrics["isolation_forest_outlier_count"] > 0:
        set_flag(
            "isolation_forest_outlier_count",
            "red",
            f"Isolation Forest assigned score below -0.5 to "
            f"{metrics['isolation_forest_outlier_count']} sessions, indicating "
            "globally isolated feature values.",
        )

    if metrics["lof_outlier_count"] > 0:
        set_flag(
            "lof_outlier_count",
            "red",
            f"Local Outlier Factor exceeded 2.0 for "
            f"{metrics['lof_outlier_count']} sessions, indicating locally "
            "inconsistent values relative to neighboring sessions.",
        )

    if metrics["mahalanobis_outlier_count"] > 0:
        set_flag(
            "mahalanobis_outlier_count",
            "red",
            f"{metrics['mahalanobis_outlier_count']} sessions exceed robust "
            "one-dimensional Mahalanobis distance 3.5 from the session cluster.",
        )

    if metrics["bootstrap_ci_too_wide"]:
        set_flag(
            "bootstrap_ci_too_wide",
            "blue",
        )

    lilliefors_p = metrics["lilliefors_p"]
    if np.isfinite(lilliefors_p):
        if lilliefors_p < 0.01:
            set_flag(
                "lilliefors_p",
                "red",
                f"Lilliefors p-value is {lilliefors_p:.6f}, below 0.01, "
                "providing strong evidence against normality with estimated "
                "mean and variance.",
            )
        elif lilliefors_p < 0.05:
            set_flag("lilliefors_p", "blue")

    jb_p = metrics["jarque_bera_p"]
    if np.isfinite(jb_p):
        if jb_p < 0.01:
            set_flag(
                "jarque_bera_p",
                "red",
                f"Jarque-Bera p-value is {jb_p:.6f}, below 0.01, confirming "
                "significant skewness and/or excess kurtosis.",
            )
        elif jb_p < 0.05:
            set_flag("jarque_bera_p", "blue")

    ratio_variance = metrics["successive_ratio_variance"]
    if np.isfinite(ratio_variance):
        if ratio_variance > 100:
            set_flag(
                "successive_ratio_variance",
                "red",
                f"Variance of successive ratios is {ratio_variance:.6f}, above "
                "100, showing extreme multiplicative session instability.",
            )
        elif ratio_variance > 10:
            set_flag("successive_ratio_variance", "blue")

    unique_proportion = metrics["proportion_unique"]
    if np.isfinite(unique_proportion):
        if unique_proportion < 0.3:
            set_flag(
                "proportion_unique",
                "red",
                f"Only {100 * unique_proportion:.1f}% of valid values are unique, "
                "below the 30% resolution threshold.",
            )
        elif unique_proportion < 0.5:
            set_flag("proportion_unique", "blue")

    dispersion = metrics["rank_dispersion_index"]
    if np.isfinite(dispersion):
        if dispersion > 0.25:
            set_flag(
                "rank_dispersion_index",
                "red",
                f"Rank-based dispersion index is {dispersion:.6f}, above 0.25, "
                "showing sparse gaps relative to the feature's IQR.",
            )
        elif dispersion > 0.10:
            set_flag("rank_dispersion_index", "blue")

    pot_pct = metrics["peak_over_threshold_pct"]
    if pot_pct > 10:
        set_flag(
            "peak_over_threshold_pct",
            "red",
            f"{pot_pct:.1f}% of sessions exceed mean + 2×MAD, above the 10% "
            "rejection threshold for upper-tail instability.",
        )
    elif pot_pct >= 5:
        set_flag("peak_over_threshold_pct", "blue")

    # Unavailable tests are explicit warnings rather than silent omissions.
    unavailable_test_metrics = [
        "shapiro_wilk_p",
        "dagostino_p",
        "anderson_stat",
        "ks_normal_p",
        "cvm_normal_p",
        "lilliefors_p",
        "jarque_bera_p",
    ]
    for metric in unavailable_test_metrics:
        if not np.isfinite(safe_float(metrics[metric])):
            set_flag(metric, "blue")

    return statuses, explanations


def calculate_health_score(
    statuses: Dict[str, str],
) -> Tuple[float, int, int]:
    """
    Start at 100 and subtract:
      - 15 for every red-flagged scoring metric
      - 8 for every blue warning scoring metric
      - 5 for explicitly configured borderline conditions

    The bootstrap CI width warning is treated as a borderline condition and
    therefore incurs 5 points instead of 8. All other blue metrics incur 8.
    """
    red_metrics = []
    blue_metrics = []
    borderline_metrics = []

    for metric, status in statuses.items():
        if metric in NON_SCORING_METRICS:
            continue

        if metric in {
            "overall_health_score",
            "verdict",
            "total_red_flags",
            "total_blue_flags",
            "flag_explanation",
        }:
            continue

        if status == "red":
            red_metrics.append(metric)
        elif status == "blue":
            if metric == "bootstrap_ci_too_wide":
                borderline_metrics.append(metric)
            else:
                blue_metrics.append(metric)

    score = (
        100
        - 15 * len(red_metrics)
        - 8 * len(blue_metrics)
        - 5 * len(borderline_metrics)
    )
    score = float(np.clip(score, 0, 100))

    total_red = sum(status == "red" for status in statuses.values())
    total_blue = sum(status == "blue" for status in statuses.values())

    return score, total_red, total_blue


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def analyze_feature(
    username: str,
    feature_name: str,
    raw_series: pd.Series,
    session_ids: Sequence[Any],
    missing_column: bool,
    feature_seed: int,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    total_sessions = len(raw_series)

    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    raw_values = numeric_series.to_numpy(dtype=float)

    nan_mask = np.isnan(raw_values)
    inf_mask = np.isinf(raw_values)
    valid_mask = np.isfinite(raw_values)

    valid_values = raw_values[valid_mask]
    valid_positions = np.flatnonzero(valid_mask)
    n_valid = valid_values.size

    metrics: Dict[str, Any] = {
        metric: np.nan for metric in OUTPUT_METRICS
    }

    metrics["count_valid"] = int(n_valid)
    metrics["count_nan"] = int(nan_mask.sum())
    metrics["count_inf"] = int(inf_mask.sum())

    # Default per-session arrays, aligned to all source rows.
    modified_z_full = np.full(total_sessions, np.nan, dtype=float)
    iqr_flags_full = np.zeros(total_sessions, dtype=bool)
    percentile_flags_full = np.zeros(total_sessions, dtype=bool)
    grubbs_flags_full = np.zeros(total_sessions, dtype=bool)
    isolation_flags_full = np.zeros(total_sessions, dtype=bool)
    lof_flags_full = np.zeros(total_sessions, dtype=bool)
    mahalanobis_flags_full = np.zeros(total_sessions, dtype=bool)
    pot_flags_full = np.zeros(total_sessions, dtype=bool)

    if n_valid == 0:
        metrics.update({
            "mean": np.nan,
            "median": np.nan,
            "std_dev": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "is_unimodal": False,
            "num_peaks": 0,
            "cv": np.nan,
            "median_to_mean_ratio": np.nan,
            "mad": np.nan,
            "robust_cv": np.nan,
            "mad_to_std_ratio": np.nan,
            "snr_estimate": np.nan,
            "modified_z_max": np.nan,
            "modified_z_mean": np.nan,
            "modified_z_outlier_count": 0,
            "modified_z_outlier_pct": 0.0,
            "iqr_outlier_count": 0,
            "iqr_outlier_pct": 0.0,
            "percentile_2.5": np.nan,
            "percentile_97.5": np.nan,
            "pct_outside_95_ci": 0.0,
            "grubbs_statistic": np.nan,
            "grubbs_p_value": np.nan,
            "is_grubbs_outlier": False,
            "shapiro_wilk_stat": np.nan,
            "shapiro_wilk_p": np.nan,
            "is_normal_shapiro": False,
            "dagostino_stat": np.nan,
            "dagostino_p": np.nan,
            "anderson_stat": np.nan,
            "anderson_crit_5pct": np.nan,
            "is_normal_anderson": False,
            "levenes_stat": np.nan,
            "levenes_p": np.nan,
            "linear_trend_slope": np.nan,
            "linear_trend_p": np.nan,
            "is_drifting": False,
            "mann_kendall_tau": np.nan,
            "mann_kendall_p": np.nan,
            "first_half_vs_second_half_median_diff": np.nan,
            "max_to_median_ratio": np.nan,
            "min_to_median_ratio": np.nan,
            "has_near_zero_denominator_evidence": False,
            "has_numerical_overflow": False,
            "has_constant_segment": False,
            "entropy_of_binned_values": 0.0,
            "ks_normal_stat": np.nan,
            "ks_normal_p": np.nan,
            "cvm_normal_stat": np.nan,
            "cvm_normal_p": np.nan,
            "session_to_session_change_mean": np.nan,
            "session_to_session_change_std": np.nan,
            "autocorrelation_lag1": np.nan,
            "autocorrelation_lag2": np.nan,
            "isolation_forest_score_min": np.nan,
            "isolation_forest_outlier_count": 0,
            "lof_score_max": np.nan,
            "lof_outlier_count": 0,
            "mahalanobis_distance_max": np.nan,
            "mahalanobis_outlier_count": 0,
            "bootstrap_median_ci_low": np.nan,
            "bootstrap_median_ci_high": np.nan,
            "bootstrap_median_ci_width": np.nan,
            "bootstrap_ci_too_wide": False,
            "lilliefors_stat": np.nan,
            "lilliefors_p": np.nan,
            "jarque_bera_stat": np.nan,
            "jarque_bera_p": np.nan,
            "successive_ratio_variance": np.nan,
            "proportion_unique": 0.0,
            "rank_dispersion_index": np.nan,
            "peak_over_threshold_count": 0,
            "peak_over_threshold_pct": 0.0,
        })
    else:
        mean_value = float(np.mean(valid_values))
        median_value = float(np.median(valid_values))
        std_value = (
            float(np.std(valid_values, ddof=1))
            if n_valid > 1
            else 0.0
        )
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        range_value = float(max_value - min_value)

        q1, q3 = np.percentile(valid_values, [25, 75])
        iqr_value = float(q3 - q1)
        mad_value = float(np.median(np.abs(valid_values - median_value)))

        if n_valid >= 3 and std_value > EPSILON:
            skewness = float(stats.skew(valid_values, bias=False))
            kurtosis = float(stats.kurtosis(valid_values, fisher=True, bias=False))
        else:
            skewness = 0.0
            kurtosis = 0.0

        is_unimodal, num_peaks = detect_kde_modes(valid_values)

        cv_value = (
            float(std_value / abs(mean_value))
            if abs(mean_value) >= MEAN_NEAR_ZERO
            else np.nan
        )
        median_to_mean_ratio = (
            float(median_value / mean_value)
            if abs(mean_value) >= MEAN_NEAR_ZERO
            else np.nan
        )
        robust_cv = (
            float(mad_value / abs(median_value))
            if abs(median_value) > EPSILON
            else (
                0.0 if mad_value <= EPSILON else np.inf
            )
        )
        mad_to_std_ratio = (
            float(mad_value / std_value)
            if std_value > EPSILON
            else (
                0.0 if mad_value <= EPSILON else np.inf
            )
        )
        snr_estimate = (
            float(abs(mean_value) / std_value)
            if std_value > EPSILON
            else (
                np.inf if abs(mean_value) > EPSILON else 0.0
            )
        )

        modified_z_valid = calculate_modified_z_scores(
            valid_values,
            median_value,
            mad_value,
        )
        modified_z_full[valid_positions] = modified_z_valid
        modified_flags_valid = modified_z_valid > 3.5

        modified_z_max = float(np.max(modified_z_valid))
        modified_z_mean = float(np.mean(modified_z_valid))
        modified_count = int(modified_flags_valid.sum())
        modified_pct = 100.0 * modified_count / n_valid

        lower_iqr = q1 - 1.5 * iqr_value
        upper_iqr = q3 + 1.5 * iqr_value
        iqr_flags_valid = (
            (valid_values < lower_iqr)
            | (valid_values > upper_iqr)
        )
        iqr_flags_full[valid_positions] = iqr_flags_valid
        iqr_count = int(iqr_flags_valid.sum())
        iqr_pct = 100.0 * iqr_count / n_valid

        percentile_2_5, percentile_97_5 = np.percentile(
            valid_values,
            [2.5, 97.5],
        )
        percentile_flags_valid = (
            (valid_values < percentile_2_5)
            | (valid_values > percentile_97_5)
        )
        percentile_flags_full[valid_positions] = percentile_flags_valid
        percentile_count = int(percentile_flags_valid.sum())
        percentile_pct = 100.0 * percentile_count / n_valid

        (
            grubbs_statistic,
            grubbs_p_value,
            is_grubbs_outlier,
            grubbs_local_index,
        ) = grubbs_test(valid_values)

        if is_grubbs_outlier and grubbs_local_index is not None:
            grubbs_global_position = valid_positions[grubbs_local_index]
            grubbs_flags_full[grubbs_global_position] = True

        # Normality tests are skipped when fewer than 8 valid values exist.
        shapiro_stat = np.nan
        shapiro_p = np.nan
        is_normal_shapiro = False
        dagostino_stat = np.nan
        dagostino_p = np.nan
        anderson_stat = np.nan
        anderson_crit_5pct = np.nan
        is_normal_anderson = False
        ks_stat = np.nan
        ks_p = np.nan
        cvm_stat = np.nan
        cvm_p = np.nan
        lilliefors_statistic = np.nan
        lilliefors_p_value = np.nan
        jb_stat = np.nan
        jb_p = np.nan

        if n_valid >= MIN_NORMALITY_SAMPLE_SIZE and std_value > EPSILON:
            shapiro_result = stats.shapiro(valid_values)
            shapiro_stat = float(shapiro_result.statistic)
            shapiro_p = float(shapiro_result.pvalue)
            is_normal_shapiro = bool(shapiro_p > 0.05)

            dagostino_result = stats.normaltest(valid_values)
            dagostino_stat = float(dagostino_result.statistic)
            dagostino_p = float(dagostino_result.pvalue)

            anderson_result = stats.anderson(valid_values, dist="norm")
            anderson_stat = float(anderson_result.statistic)

            significance_levels = np.asarray(anderson_result.significance_level)
            critical_values = np.asarray(anderson_result.critical_values)
            critical_index = int(np.argmin(np.abs(significance_levels - 5.0)))
            anderson_crit_5pct = float(critical_values[critical_index])

            # Per the requested threshold table, True means reject normality.
            is_normal_anderson = bool(anderson_stat > anderson_crit_5pct)

            standardized = (valid_values - mean_value) / std_value
            ks_result = stats.kstest(standardized, "norm")
            ks_stat = float(ks_result.statistic)
            ks_p = float(ks_result.pvalue)

            cvm_result = stats.cramervonmises(standardized, "norm")
            cvm_stat = float(cvm_result.statistic)
            cvm_p = float(cvm_result.pvalue)

            if LILLIEFORS_AVAILABLE:
                try:
                    lilliefors_statistic, lilliefors_p_value = lilliefors(
                        valid_values,
                        dist="norm",
                    )
                    lilliefors_statistic = float(lilliefors_statistic)
                    lilliefors_p_value = float(lilliefors_p_value)
                except Exception:
                    lilliefors_statistic = np.nan
                    lilliefors_p_value = np.nan

            jb_result = stats.jarque_bera(valid_values)
            jb_stat = float(jb_result.statistic)
            jb_p = float(jb_result.pvalue)

        # Variance stability: first valid half versus second valid half.
        split_index = n_valid // 2
        first_half = valid_values[:split_index]
        second_half = valid_values[split_index:]

        if first_half.size >= 2 and second_half.size >= 2:
            levene_result = stats.levene(
                first_half,
                second_half,
                center="median",
            )
            levene_stat = float(levene_result.statistic)
            levene_p = float(levene_result.pvalue)
        else:
            levene_stat = np.nan
            levene_p = np.nan

        median_difference = (
            float(np.median(second_half) - np.median(first_half))
            if first_half.size > 0 and second_half.size > 0
            else np.nan
        )

        # Temporal trend uses original session positions, not compressed indices.
        if n_valid >= 3 and std_value > EPSILON:
            regression_result = stats.linregress(
                valid_positions.astype(float),
                valid_values,
            )
            linear_slope = float(regression_result.slope)
            linear_p = float(regression_result.pvalue)
        else:
            linear_slope = 0.0 if n_valid >= 1 else np.nan
            linear_p = 1.0 if n_valid >= 1 else np.nan

        is_drifting = bool(np.isfinite(linear_p) and linear_p < 0.05)
        mk_tau, mk_p = mann_kendall_test(valid_values)

        median_denominator = abs(median_value) + EPSILON
        max_to_median_ratio = float(max_value / median_denominator)
        min_to_median_ratio = float(min_value / median_denominator)

        absolute_values = np.abs(valid_values)
        nonzero_absolute = absolute_values[absolute_values > 0]
        has_small_nonzero = bool(
            nonzero_absolute.size > 0
            and np.any(nonzero_absolute < 1e-3)
        )
        has_near_zero_denominator = bool(
            np.any(absolute_values > 1e6)
            and has_small_nonzero
        )
        has_numerical_overflow = bool(np.any(absolute_values > 1e10))

        rounded_counts = Counter(np.round(valid_values, 6).tolist())
        most_common_count = max(rounded_counts.values())
        has_constant_segment = bool(most_common_count / n_valid > 0.5)

        entropy_value = calculate_entropy(valid_values, bins=20)

        if n_valid >= 2:
            successive_changes = np.abs(np.diff(valid_values))
            change_mean = float(np.mean(successive_changes))
            change_std = (
                float(np.std(successive_changes, ddof=1))
                if successive_changes.size > 1
                else 0.0
            )
        else:
            successive_changes = np.array([], dtype=float)
            change_mean = np.nan
            change_std = np.nan

        autocorrelation_lag1 = safe_autocorrelation(valid_values, lag=1)
        autocorrelation_lag2 = safe_autocorrelation(valid_values, lag=2)

        isolation_scores, isolation_flags_valid = run_isolation_forest(
            valid_values
        )
        isolation_flags_full[valid_positions] = isolation_flags_valid

        lof_scores, lof_flags_valid = run_lof(valid_values)
        lof_flags_full[valid_positions] = lof_flags_valid

        mahalanobis_distances, mahalanobis_flags_valid = (
            robust_mahalanobis_1d(
                valid_values,
                median_value,
                mad_value,
                std_value,
            )
        )
        mahalanobis_flags_full[valid_positions] = mahalanobis_flags_valid

        bootstrap_low, bootstrap_high, bootstrap_width = bootstrap_median_ci(
            valid_values,
            BOOTSTRAP_ITERATIONS,
            feature_seed,
        )
        bootstrap_ci_too_wide = bool(
            iqr_value > EPSILON
            and bootstrap_width > 0.5 * iqr_value
        )

        # Successive ratios ignore pairs whose denominator is effectively zero.
        if n_valid >= 2:
            ratio_denominators = valid_values[:-1]
            ratio_numerators = valid_values[1:]
            usable_ratio_mask = np.abs(ratio_denominators) > EPSILON
            successive_ratios = (
                ratio_numerators[usable_ratio_mask]
                / ratio_denominators[usable_ratio_mask]
            )
            successive_ratio_variance = (
                float(np.var(successive_ratios, ddof=1))
                if successive_ratios.size > 1
                else 0.0
            )
        else:
            successive_ratio_variance = np.nan

        proportion_unique = float(
            np.unique(np.round(valid_values, 12)).size / n_valid
        )

        if n_valid >= 2 and iqr_value > EPSILON:
            sorted_values = np.sort(valid_values)
            average_gap = float(np.mean(np.diff(sorted_values)))
            rank_dispersion_index = float(average_gap / iqr_value)
        elif n_valid >= 2:
            rank_dispersion_index = (
                0.0 if range_value <= EPSILON else np.inf
            )
        else:
            rank_dispersion_index = np.nan

        pot_threshold = mean_value + 2.0 * mad_value
        pot_flags_valid = valid_values > pot_threshold
        pot_flags_full[valid_positions] = pot_flags_valid
        pot_count = int(pot_flags_valid.sum())
        pot_pct = 100.0 * pot_count / n_valid

        metrics.update({
            "mean": mean_value,
            "median": median_value,
            "std_dev": std_value,
            "min": min_value,
            "max": max_value,
            "range": range_value,
            "iqr": iqr_value,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_unimodal": is_unimodal,
            "num_peaks": num_peaks,
            "cv": cv_value,
            "median_to_mean_ratio": median_to_mean_ratio,
            "mad": mad_value,
            "robust_cv": robust_cv,
            "mad_to_std_ratio": mad_to_std_ratio,
            "snr_estimate": snr_estimate,
            "modified_z_max": modified_z_max,
            "modified_z_mean": modified_z_mean,
            "modified_z_outlier_count": modified_count,
            "modified_z_outlier_pct": modified_pct,
            "iqr_outlier_count": iqr_count,
            "iqr_outlier_pct": iqr_pct,
            "percentile_2.5": float(percentile_2_5),
            "percentile_97.5": float(percentile_97_5),
            "pct_outside_95_ci": percentile_pct,
            "grubbs_statistic": grubbs_statistic,
            "grubbs_p_value": grubbs_p_value,
            "is_grubbs_outlier": is_grubbs_outlier,
            "shapiro_wilk_stat": shapiro_stat,
            "shapiro_wilk_p": shapiro_p,
            "is_normal_shapiro": is_normal_shapiro,
            "dagostino_stat": dagostino_stat,
            "dagostino_p": dagostino_p,
            "anderson_stat": anderson_stat,
            "anderson_crit_5pct": anderson_crit_5pct,
            "is_normal_anderson": is_normal_anderson,
            "levenes_stat": levene_stat,
            "levenes_p": levene_p,
            "linear_trend_slope": linear_slope,
            "linear_trend_p": linear_p,
            "is_drifting": is_drifting,
            "mann_kendall_tau": mk_tau,
            "mann_kendall_p": mk_p,
            "first_half_vs_second_half_median_diff": median_difference,
            "max_to_median_ratio": max_to_median_ratio,
            "min_to_median_ratio": min_to_median_ratio,
            "has_near_zero_denominator_evidence": has_near_zero_denominator,
            "has_numerical_overflow": has_numerical_overflow,
            "has_constant_segment": has_constant_segment,
            "entropy_of_binned_values": entropy_value,
            "ks_normal_stat": ks_stat,
            "ks_normal_p": ks_p,
            "cvm_normal_stat": cvm_stat,
            "cvm_normal_p": cvm_p,
            "session_to_session_change_mean": change_mean,
            "session_to_session_change_std": change_std,
            "autocorrelation_lag1": autocorrelation_lag1,
            "autocorrelation_lag2": autocorrelation_lag2,
            "isolation_forest_score_min": float(np.min(isolation_scores)),
            "isolation_forest_outlier_count": int(isolation_flags_valid.sum()),
            "lof_score_max": float(np.max(lof_scores)),
            "lof_outlier_count": int(lof_flags_valid.sum()),
            "mahalanobis_distance_max": float(
                np.max(mahalanobis_distances)
            ),
            "mahalanobis_outlier_count": int(
                mahalanobis_flags_valid.sum()
            ),
            "bootstrap_median_ci_low": bootstrap_low,
            "bootstrap_median_ci_high": bootstrap_high,
            "bootstrap_median_ci_width": bootstrap_width,
            "bootstrap_ci_too_wide": bootstrap_ci_too_wide,
            "lilliefors_stat": lilliefors_statistic,
            "lilliefors_p": lilliefors_p_value,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_p": jb_p,
            "successive_ratio_variance": successive_ratio_variance,
            "proportion_unique": proportion_unique,
            "rank_dispersion_index": rank_dispersion_index,
            "peak_over_threshold_count": pot_count,
            "peak_over_threshold_pct": pot_pct,
        })

    statuses, red_explanations = assign_metric_statuses(
        metrics,
        total_sessions,
        missing_column,
    )

    health_score, total_red, total_blue = calculate_health_score(statuses)

    if health_score > 70:
        verdict = "HEALTHY"
        verdict_status = "green"
    elif health_score >= 40:
        verdict = "SUSPICIOUS"
        verdict_status = "blue"
    else:
        verdict = "REJECT"
        verdict_status = "red"

    metrics["overall_health_score"] = health_score
    metrics["verdict"] = verdict
    metrics["total_red_flags"] = total_red
    metrics["total_blue_flags"] = total_blue

    if red_explanations:
        explanation_text = "; ".join(red_explanations.values())
    else:
        explanation_text = (
            "No red-flag thresholds were triggered for this feature."
        )
    metrics["flag_explanation"] = explanation_text

    statuses["overall_health_score"] = verdict_status
    statuses["verdict"] = verdict_status
    statuses["total_red_flags"] = "red" if total_red > 0 else "green"
    statuses["total_blue_flags"] = "blue" if total_blue > 0 else "green"
    statuses["flag_explanation"] = "red" if total_red > 0 else "green"

    formatted_row: Dict[str, str] = {
        "feature_name": f"{verdict_status_to_emoji(verdict_status)} {feature_name}"
    }

    for metric in OUTPUT_METRICS:
        if metric == "flag_explanation":
            emoji = RED if total_red > 0 else GREEN
            formatted_row[metric] = f"{emoji} {explanation_text}"
        else:
            formatted_row[metric] = format_metric_cell(
                metric,
                metrics[metric],
                statuses.get(metric, "green"),
            )

    # Session-level outlier report.
    session_outliers: List[Dict[str, Any]] = []

    for position in range(total_sessions):
        methods: List[str] = []

        if nan_mask[position]:
            methods.append("invalid_nan")
        if inf_mask[position]:
            methods.append("invalid_inf")
        if (
            np.isfinite(modified_z_full[position])
            and modified_z_full[position] > 3.5
        ) or np.isinf(modified_z_full[position]):
            methods.append("modified_z")
        if iqr_flags_full[position]:
            methods.append("iqr")
        if grubbs_flags_full[position]:
            methods.append("grubbs")
        if percentile_flags_full[position]:
            methods.append("percentile_95")
        if isolation_flags_full[position]:
            methods.append("isolation_forest")
        if lof_flags_full[position]:
            methods.append("local_outlier_factor")
        if mahalanobis_flags_full[position]:
            methods.append("robust_mahalanobis")
        if pot_flags_full[position]:
            methods.append("peak_over_threshold")

        if not methods:
            continue

        method_count = len(methods)
        if method_count >= 3:
            confidence = "HIGH"
        elif method_count == 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        feature_value = raw_values[position]
        modified_score = modified_z_full[position]

        session_outliers.append({
            "session_id": session_ids[position],
            "feature_name": feature_name,
            "feature_value": round_numeric(feature_value),
            "modified_z_score": round_numeric(modified_score),
            "is_iqr_outlier": bool(iqr_flags_full[position]),
            "is_grubbs_outlier": bool(grubbs_flags_full[position]),
            "is_percentile_outlier": bool(
                percentile_flags_full[position]
            ),
            "outlier_methods_triggered": ",".join(methods),
            "confidence_level": confidence,
        })

    raw_summary = {
        "feature_name": feature_name,
        "health_score": health_score,
        "verdict": verdict,
        "total_red_flags": total_red,
        "total_blue_flags": total_blue,
        "red_explanations": list(red_explanations.values()),
        "key_stat": build_key_stat(metrics),
    }

    return formatted_row, session_outliers, raw_summary


def verdict_status_to_emoji(status: str) -> str:
    return {
        "red": RED,
        "blue": BLUE,
        "green": GREEN,
    }.get(status, GREEN)


def build_key_stat(metrics: Dict[str, Any]) -> str:
    if metrics["modified_z_outlier_count"] > 0:
        return (
            f"Mod-Z outliers={metrics['modified_z_outlier_count']} "
            f"({metrics['modified_z_outlier_pct']:.1f}%)"
        )

    if metrics["iqr_outlier_count"] > 0:
        return (
            f"IQR outliers={metrics['iqr_outlier_count']} "
            f"({metrics['iqr_outlier_pct']:.1f}%)"
        )

    if np.isfinite(safe_float(metrics["cv"])):
        return f"CV={metrics['cv']:.3f}"

    if np.isfinite(safe_float(metrics["robust_cv"])):
        return f"Robust CV={metrics['robust_cv']:.3f}"

    return f"Valid={metrics['count_valid']}"


# =============================================================================
# USER-LEVEL PROCESSING
# =============================================================================

def load_selected_features(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Selected-feature file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        feature_list = json.load(file)

    if not isinstance(feature_list, list):
        raise ValueError(
            "selected_features.json must contain a JSON array of column names."
        )

    cleaned_features = []
    seen = set()

    for feature in feature_list:
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError(
                "Every selected feature must be a non-empty string."
            )

        feature = feature.strip()
        if feature not in seen:
            cleaned_features.append(feature)
            seen.add(feature)

    if not cleaned_features:
        raise ValueError("selected_features.json contains no features.")

    return cleaned_features


def load_user_data(
    username: str,
    selected_features: Sequence[str],
) -> Tuple[pd.DataFrame, List[Any], Dict[str, bool]]:
    """
    Load and combine only this user's training/testing sessions.

    No other user's data enters this function.
    """
    train_path = (
        FEATURE_DIRECTORY / f"{username}_training_sessions.csv"
    )
    test_path = (
        FEATURE_DIRECTORY / f"{username}_testing_sessions.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing testing CSV: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["__source__"] = "training"
    test_df["__source__"] = "testing"
    train_df["__source_row__"] = np.arange(len(train_df))
    test_df["__source_row__"] = np.arange(len(test_df))

    combined = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if "session_id" in combined.columns:
        original_ids = combined["session_id"].astype(str).tolist()
        duplicate_mask = pd.Series(original_ids).duplicated(keep=False)

        session_ids = []
        for index, original_id in enumerate(original_ids):
            if duplicate_mask.iloc[index]:
                session_ids.append(
                    f"{combined.loc[index, '__source__']}:{original_id}"
                )
            else:
                session_ids.append(original_id)
    else:
        session_ids = [
            (
                f"{combined.loc[index, '__source__']}:"
                f"{int(combined.loc[index, '__source_row__']) + 1}"
            )
            for index in range(len(combined))
        ]

    missing_columns: Dict[str, bool] = {}

    for feature in selected_features:
        absent_from_train = feature not in train_df.columns
        absent_from_test = feature not in test_df.columns
        missing_columns[feature] = absent_from_train or absent_from_test

        if feature not in combined.columns:
            combined[feature] = np.nan

    return combined, session_ids, missing_columns


def print_user_summary(
    username: str,
    session_count: int,
    feature_summaries: List[Dict[str, Any]],
) -> None:
    sorted_summaries = sorted(
        feature_summaries,
        key=lambda item: (
            item["health_score"],
            item["feature_name"].lower(),
        ),
    )

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(
        f"║  USER: {username} — {session_count} sessions loaded"
        .ljust(63) + "║"
    )
    print(
        f"║  Features analyzed: {len(feature_summaries)}"
        .ljust(63) + "║"
    )
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    headers = ["Feature", "Score", "Verdict", "Top Red Flags", "Key Stat"]

    table_rows = []
    for summary in sorted_summaries:
        red_explanations = summary["red_explanations"]
        if red_explanations:
            top_flags = " | ".join(red_explanations[:2])
        else:
            top_flags = "None"

        if len(top_flags) > 110:
            top_flags = top_flags[:107] + "..."

        table_rows.append([
            summary["feature_name"],
            f"{summary['health_score']:.1f}",
            summary["verdict"],
            top_flags,
            summary["key_stat"],
        ])

    print_table(headers, table_rows)

    rejected = [
        item["feature_name"]
        for item in sorted_summaries
        if item["health_score"] < 40
    ]
    suspicious = [
        item["feature_name"]
        for item in sorted_summaries
        if 40 <= item["health_score"] <= 70
    ]
    healthy = [
        item["feature_name"]
        for item in sorted_summaries
        if item["health_score"] > 70
    ]

    print()
    print(f"{RED} REJECTED features (score < 40): {rejected}")
    print(f"{BLUE} SUSPICIOUS features (40-70): {suspicious}")
    print(f"{GREEN} HEALTHY features (> 70): {healthy}")


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        print("| " + " | ".join(headers) + " |")
        return

    maximum_widths = [45, 8, 12, 110, 35]

    widths = []
    for column_index, header in enumerate(headers):
        content_width = max(
            len(str(header)),
            max(
                len(str(row[column_index]))
                for row in rows
            ),
        )
        widths.append(min(content_width, maximum_widths[column_index]))

    def truncate(value: str, width: int) -> str:
        value = str(value)
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[:width - 3] + "..."

    header_line = "| " + " | ".join(
        truncate(header, widths[index]).ljust(widths[index])
        for index, header in enumerate(headers)
    ) + " |"

    separator_line = "|-" + "-|-".join(
        "-" * width for width in widths
    ) + "-|"

    print(header_line)
    print(separator_line)

    for row in rows:
        print(
            "| "
            + " | ".join(
                truncate(row[index], widths[index]).ljust(widths[index])
                for index in range(len(headers))
            )
            + " |"
        )


def process_user(
    username: str,
    selected_features: Sequence[str],
    user_index: int,
) -> None:
    combined_df, session_ids, missing_columns = load_user_data(
        username,
        selected_features,
    )

    diagnostics_rows: List[Dict[str, str]] = []
    all_session_outliers: List[Dict[str, Any]] = []
    feature_summaries: List[Dict[str, Any]] = []

    for feature_index, feature_name in enumerate(selected_features):
        feature_seed = (
            RANDOM_SEED
            + user_index * 100_000
            + feature_index
        )

        formatted_row, outlier_rows, raw_summary = analyze_feature(
            username=username,
            feature_name=feature_name,
            raw_series=combined_df[feature_name],
            session_ids=session_ids,
            missing_column=missing_columns[feature_name],
            feature_seed=feature_seed,
        )

        diagnostics_rows.append(formatted_row)
        all_session_outliers.extend(outlier_rows)
        feature_summaries.append(raw_summary)

    diagnostics_df = pd.DataFrame(
        diagnostics_rows,
        columns=["feature_name"] + OUTPUT_METRICS,
    )

    outlier_columns = [
        "session_id",
        "feature_name",
        "feature_value",
        "modified_z_score",
        "is_iqr_outlier",
        "is_grubbs_outlier",
        "is_percentile_outlier",
        "outlier_methods_triggered",
        "confidence_level",
    ]

    outlier_df = pd.DataFrame(
        all_session_outliers,
        columns=outlier_columns,
    )

    if not outlier_df.empty:
        confidence_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }
        outlier_df["__confidence_order__"] = (
            outlier_df["confidence_level"]
            .map(confidence_order)
            .fillna(3)
        )
        outlier_df = (
            outlier_df
            .sort_values(
                by=[
                    "__confidence_order__",
                    "feature_name",
                    "session_id",
                ],
                kind="stable",
            )
            .drop(columns=["__confidence_order__"])
            .reset_index(drop=True)
        )

    diagnostics_path = (
        OUTPUT_DIRECTORY / f"{username}_feature_diagnostics.csv"
    )
    outliers_path = (
        OUTPUT_DIRECTORY / f"{username}_session_level_outliers.csv"
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
        encoding="utf-8-sig",
    )
    outlier_df.to_csv(
        outliers_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_user_summary(
        username=username,
        session_count=len(combined_df),
        feature_summaries=feature_summaries,
    )

    print()
    print(f"Saved diagnostics: {diagnostics_path}")
    print(f"Saved session outliers: {outliers_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = time.perf_counter()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features(FEATURES_JSON)

    print("=" * 72)
    print("BEHAVIORAL BIOMETRIC FEATURE QUALITY-CONTROL DIAGNOSTICS")
    print("=" * 72)
    print(f"Users: {USERS}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")
    print(f"Dip test available: {DIPTEST_AVAILABLE}")
    print(f"Lilliefors available: {LILLIEFORS_AVAILABLE}")

    failures: List[Tuple[str, str]] = []

    for user_index, username in enumerate(USERS):
        user_start = time.perf_counter()

        try:
            process_user(
                username=username,
                selected_features=selected_features,
                user_index=user_index,
            )
            elapsed = time.perf_counter() - user_start
            print(f"User analysis time: {elapsed:.2f} seconds")
        except Exception as exc:
            failures.append((username, str(exc)))
            print()
            print(f"{RED} Failed to process user {username}: {exc}")

    total_elapsed = time.perf_counter() - start_time

    print()
    print("=" * 72)
    print(f"Total analysis time: {total_elapsed:.2f} seconds")

    if failures:
        print(f"{RED} Users with failures:")
        for username, error in failures:
            print(f"  - {username}: {error}")
        raise SystemExit(1)

    print(f"{GREEN} All per-user diagnostics completed successfully.")


if __name__ == "__main__":
    main()
