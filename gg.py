#!/usr/bin/env python3
"""
diagnostics.py
==============

Feature-stability and temporal-drift diagnostics for user-specific One-Class
SVM behavioral-biometric models.

What this script does
---------------------
For every user in USER_LIST, the script:
1. Loads <user>_training_sessions.csv and <user>_testing_sessions.csv.
2. Keeps only genuine training data for feature-health and temporal-drift analysis.
3. Evaluates each numeric feature for:
   - coefficient of variation,
   - outliers,
   - Shapiro-Wilk normality,
   - multimodality,
   - temporal linear drift,
   - rolling-CV instability,
   - EWMA drift,
   - optional change points.
4. Optionally simulates future/session-late FRR using the same OCSVM parameters
   as the existing training pipeline.
5. Produces per-user reports, plots, global feature recommendations, and a
   human-readable diagnostic summary.

How to configure
----------------
Edit USER_LIST, DATA_DIR, OUTPUT_DIR, and the thresholds in the configuration
block immediately below.

How to run
----------
    python diagnostics.py

Expected input files
--------------------
DATA_DIR/
    <user>_training_sessions.csv
    <user>_testing_sessions.csv

Expected label columns
----------------------
The script automatically looks for one of:
    session_label, label, test_type

Expected session-order columns
------------------------------
The script prefers:
    session_id, session_number

Outputs
-------
OUTPUT_DIR/
    selected_features.json
    diagnostic_summary.txt
    global_drift_summary.csv
    user_specific/<user>/
    global_plots/

Interpretation
--------------
- selected_features.json:
    Features passing quality checks in more than 50% of processed users.
- excluded_features:
    Features failing quality checks in at least 50% of processed users.
- global_excluded_features:
    Globally problematic features meeting the >=50% exclusion rule.
- drift_summary.csv:
    Feature-level drift diagnostics for one user.
- feature_quality.csv:
    Feature distribution-health diagnostics for one user.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


# =============================================================================
# CONFIGURATION BLOCK — EDIT HERE
# =============================================================================

USER_LIST = [
    "srikanth", "Samarth", "harshit",
    # ... add/remove users here ...
]

DATA_DIR = "./features/new_app/"
OUTPUT_DIR = "./diagnostics/"
RANDOM_SEED = 42

# --- Distribution-health thresholds ---
MAX_COEFFICIENT_OF_VARIATION = 0.30
MAX_OUTLIER_FRACTION = 0.05
NORMALITY_ALPHA = 0.05
DIP_TEST_ALPHA = 0.05

# --- Temporal-drift thresholds ---
SLOPE_SIGNIFICANCE_ALPHA = 0.05
MIN_SLOPE_MAGNITUDE = 0.01
WINDOW_SIZE = 5
EXPONENTIAL_FORGETTING_FACTOR = 0.9

# --- Model-performance simulation (optional) ---
TRAIN_SIZE_RATIO = 0.8
SIMULATE_FRR_OVER_TIME = True
OCSVM_THRESHOLD = -0.3

# --- Additional engineering / numerical-safety configuration ---
EPS = 1e-9
MIN_SHAPIRO_SAMPLES = 3
PREFERRED_SHAPIRO_SAMPLES = 8
MIN_SESSIONS_RELIABLE_DRIFT = 20
MIN_SESSIONS_FOR_DRIFT_FIT = 3
MIN_SESSIONS_FOR_ROLLING_CV = WINDOW_SIZE + 1
MIN_SESSIONS_FOR_CHANGEPOINT = 8
HISTOGRAM_MAX_BINS = 200
HISTOGRAM_MIN_BINS = 10
HISTOGRAM_SMOOTHING_WINDOW = 3
HISTOGRAM_PEAK_PROMINENCE_FRACTION = 0.15
ROLLING_CV_INCREASE_FACTOR = 1.30
ROLLING_CV_POSITIVE_DIFF_FRACTION = 0.50
EWMA_PERSISTENCE_FRACTION = 0.30
PROBLEM_USER_DRIFT_FRACTION = 0.40
PROBLEM_USER_MEAN_NORMALIZED_SLOPE = 0.05
GLOBAL_EXCLUSION_FRACTION = 0.50
SELECTED_FEATURE_PASS_FRACTION = 0.50
HIGH_FRR_THRESHOLD = 0.15
RECENT_TRAINING_FRACTION = 0.30
PLOT_DPI = 150
SESSION_AGGREGATION = "mean"  # Supported: "mean", "median"
LABEL_CANDIDATES = ["session_label", "label", "test_type"]
SESSION_COLUMN_CANDIDATES = ["session_id", "session_number"]
OCSVM_NU = 0.2
OCSVM_GAMMA = 0.00055


# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

try:
    import statsmodels.api as sm

    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    STATSMODELS_AVAILABLE = False

try:
    from diptests import diptest

    DIPTEST_AVAILABLE = True
except ImportError:
    diptest = None
    DIPTEST_AVAILABLE = False

try:
    import ruptures as rpt

    RUPTURES_AVAILABLE = True
except ImportError:
    rpt = None
    RUPTURES_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    SKLEARN_AVAILABLE = True
except ImportError:
    StandardScaler = None
    OneClassSVM = None
    SKLEARN_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UserResult:
    """Stores complete per-user diagnostic outputs used for global aggregation."""

    user_id: str
    feature_quality: pd.DataFrame
    drift_summary: pd.DataFrame
    user_summary: Dict[str, Any]
    frr_result: Dict[str, Any]
    genuine_row_count: int
    feature_count: int
    ordering_method: str


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging(output_dir: Path) -> None:
    """Configure timestamped console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "diagnostics.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =============================================================================
# BASIC DATA UTILITIES
# =============================================================================

def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "feature"


def detect_label_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """Detect a supported label column, preferring one found in both dataframes."""
    for column in candidates:
        if column in train_df.columns and column in test_df.columns:
            return column

    for column in candidates:
        if column in train_df.columns:
            logging.warning(
                "Label column '%s' exists only in training data; using it for "
                "genuine-row filtering.",
                column,
            )
            return column

    for column in candidates:
        if column in test_df.columns:
            logging.warning(
                "Label column '%s' exists only in testing data. Training data "
                "will be treated as genuine because no train label exists.",
                column,
            )
            return column

    return None


def normalize_label(value: Any) -> str:
    """Normalize a source label into genuine or impostor."""
    return "genuine" if "genuine" in str(value).lower() else "impostor"


def load_user_data(
    user_id: str,
    data_dir: Path,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load a user's train/test CSV files safely; return None on failure."""
    train_path = data_dir / f"{user_id}_training_sessions.csv"
    test_path = data_dir / f"{user_id}_testing_sessions.csv"

    if not train_path.exists():
        logging.error("Missing training CSV for user '%s': %s", user_id, train_path)
        return None

    if not test_path.exists():
        logging.error("Missing testing CSV for user '%s': %s", user_id, test_path)
        return None

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        logging.error("Could not read CSVs for user '%s': %s", user_id, exc)
        return None

    if train_df.empty:
        logging.error("Training CSV is empty for user '%s'. Skipping.", user_id)
        return None

    logging.info(
        "Loaded user '%s': train=%s rows x %s cols; test=%s rows x %s cols.",
        user_id,
        train_df.shape[0],
        train_df.shape[1],
        test_df.shape[0],
        test_df.shape[1],
    )
    return train_df, test_df


def sort_sessions_chronologically(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[str], str]:
    """
    Sort sessions chronologically.

    Priority:
    1. session_id, optionally ordered by session_number if present.
    2. session_number.
    3. original CSV row order.

    Returns:
        sorted dataframe, grouping column, ordering-method description.
    """
    working_df = df.copy()
    working_df["__original_row_order__"] = np.arange(len(working_df))

    has_session_id = "session_id" in working_df.columns
    has_session_number = "session_number" in working_df.columns

    if has_session_id and has_session_number:
        sorted_df = working_df.sort_values(
            by=["session_number", "__original_row_order__"],
            kind="mergesort",
        )
        return sorted_df, "session_id", "session_id grouped; ordered by session_number"

    if has_session_id:
        first_seen_order = pd.unique(working_df["session_id"])
        order_map = {value: index for index, value in enumerate(first_seen_order)}
        working_df["__session_first_seen_order__"] = working_df["session_id"].map(order_map)
        sorted_df = working_df.sort_values(
            by=["__session_first_seen_order__", "__original_row_order__"],
            kind="mergesort",
        )
        return sorted_df, "session_id", "session_id grouped; ordered by first appearance"

    if has_session_number:
        sorted_df = working_df.sort_values(
            by=["session_number", "__original_row_order__"],
            kind="mergesort",
        )
        return sorted_df, "session_number", "session_number used for grouping and ordering"

    logging.warning(
        "No session_id/session_number found. Falling back to original CSV row order; "
        "each row is treated as a temporal observation."
    )
    return working_df, None, "original CSV row order; each row treated as one session"


def select_numeric_features(
    df: pd.DataFrame,
    label_column: Optional[str],
) -> List[str]:
    """Return numeric model-feature columns after removing identifiers and internals."""
    excluded_columns = {
        "session_id",
        "session_number",
        "impostor_user_id",
        "__label__",
        "__original_row_order__",
        "__session_first_seen_order__",
    }

    if label_column is not None:
        excluded_columns.add(label_column)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns and not column.startswith("__")
    ]

    return feature_columns


def aggregate_sessions(
    df: pd.DataFrame,
    feature: str,
    group_column: Optional[str],
    aggregation: str,
) -> np.ndarray:
    """
    Aggregate a feature into a chronological session-level series.

    Session mean is the default because the OCSVM pipeline operates on
    session-level features. Median is supported as a more outlier-robust option.
    """
    values = pd.to_numeric(df[feature], errors="coerce")

    if group_column is None:
        return values.to_numpy(dtype=float)

    temporary = pd.DataFrame(
        {
            "group": df[group_column].to_numpy(),
            "value": values.to_numpy(),
        }
    )

    if aggregation == "median":
        aggregated = temporary.groupby("group", sort=False)["value"].median()
    else:
        aggregated = temporary.groupby("group", sort=False)["value"].mean()

    return aggregated.to_numpy(dtype=float)


# =============================================================================
# DISTRIBUTION HEALTH
# =============================================================================

def calculate_cv(mean_value: float, std_value: float) -> float:
    """Calculate coefficient of variation safely."""
    if not np.isfinite(mean_value) or abs(mean_value) < EPS:
        return np.inf
    return float(std_value / abs(mean_value))


def iqr_outlier_fraction(values: np.ndarray) -> float:
    """Compute fraction of values outside the 3*IQR bounds."""
    if values.size == 0:
        return np.nan

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if abs(iqr) < EPS:
        return 0.0

    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    return float(np.mean((values < lower) | (values > upper)))


def histogram_multimodality_fallback(values: np.ndarray) -> bool:
    """
    Deterministic fallback multimodality detector.

    The histogram is smoothed using a moving-average kernel. A peak counts only
    if it is a strict local maximum and its height is at least
    HISTOGRAM_PEAK_PROMINENCE_FRACTION of the global smoothed-histogram maximum.
    Two or more qualifying peaks implies multimodality.
    """
    if values.size < HISTOGRAM_MIN_BINS or np.nanstd(values) < EPS:
        return False

    bins = int(
        min(
            HISTOGRAM_MAX_BINS,
            max(HISTOGRAM_MIN_BINS, round(np.sqrt(values.size))),
        )
    )

    hist, _ = np.histogram(values, bins=bins)
    if hist.max() <= 0:
        return False

    kernel = np.ones(HISTOGRAM_SMOOTHING_WINDOW, dtype=float)
    kernel /= kernel.sum()
    smooth_hist = np.convolve(hist.astype(float), kernel, mode="same")

    prominence_threshold = HISTOGRAM_PEAK_PROMINENCE_FRACTION * smooth_hist.max()
    peak_count = 0

    for index in range(1, len(smooth_hist) - 1):
        if (
            smooth_hist[index] > smooth_hist[index - 1]
            and smooth_hist[index] >= smooth_hist[index + 1]
            and smooth_hist[index] >= prominence_threshold
        ):
            peak_count += 1

    return peak_count >= 2


def run_dip_test(values: np.ndarray) -> Tuple[float, bool, str]:
    """Run Hartigan's dip test or deterministic histogram fallback."""
    if values.size < HISTOGRAM_MIN_BINS or np.nanstd(values) < EPS:
        return np.nan, False, "insufficient_or_constant"

    if DIPTEST_AVAILABLE:
        try:
            result = diptest(values)
            p_value = float(result[1]) if isinstance(result, tuple) else float(result)
            return p_value, bool(p_value < DIP_TEST_ALPHA), "diptest"
        except Exception as exc:
            logging.warning("Dip test failed (%s); using histogram fallback.", exc)

    is_multimodal = histogram_multimodality_fallback(values)
    return np.nan, is_multimodal, "histogram_fallback"


def compute_feature_quality(
    user_id: str,
    feature: str,
    values: np.ndarray,
) -> Dict[str, Any]:
    """Compute all distribution-health metrics and exclusion decision for one feature."""
    clean_values = values[np.isfinite(values)]
    n_values = clean_values.size

    record: Dict[str, Any] = {
        "user_id": user_id,
        "feature": feature,
        "n_values": int(n_values),
        "mean": np.nan,
        "std": np.nan,
        "cv": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "outlier_frac": np.nan,
        "shapiro_p": np.nan,
        "dip_p": np.nan,
        "multimodality_method": "not_run",
        "is_constant": False,
        "is_unstable": False,
        "is_multimodal": False,
        "is_outlier_heavy": False,
        "is_non_normal_high_cv": False,
        "exclude": False,
        "exclusion_reason": "",
    }

    if n_values == 0:
        record["exclude"] = True
        record["exclusion_reason"] = "all_values_missing"
        logging.warning("User '%s', feature '%s': all values are missing.", user_id, feature)
        return record

    mean_value = float(np.mean(clean_values))
    std_value = float(np.std(clean_values, ddof=1)) if n_values > 1 else 0.0

    record["mean"] = mean_value
    record["std"] = std_value

    if std_value < EPS:
        record["is_constant"] = True
        record["is_unstable"] = True
        record["exclude"] = True
        record["cv"] = 0.0 if abs(mean_value) >= EPS else np.inf
        record["exclusion_reason"] = "constant_feature"
        logging.warning(
            "User '%s', feature '%s': constant feature; excluding.",
            user_id,
            feature,
        )
        return record

    cv_value = calculate_cv(mean_value, std_value)
    record["cv"] = cv_value

    if abs(mean_value) < EPS:
        record["is_unstable"] = True
        record["exclude"] = True
        record["exclusion_reason"] = "near_zero_mean_cv_infinite"
        logging.warning(
            "User '%s', feature '%s': |mean| < EPS; CV set to infinity and excluded.",
            user_id,
            feature,
        )
        return record

    record["skewness"] = float(stats.skew(clean_values, bias=False)) if n_values >= 3 else np.nan
    record["kurtosis"] = (
        float(stats.kurtosis(clean_values, fisher=False, bias=False))
        if n_values >= 4
        else np.nan
    )
    record["outlier_frac"] = iqr_outlier_fraction(clean_values)
    record["is_unstable"] = bool(cv_value > MAX_COEFFICIENT_OF_VARIATION)
    record["is_outlier_heavy"] = bool(
        np.isfinite(record["outlier_frac"])
        and record["outlier_frac"] > MAX_OUTLIER_FRACTION
    )

    if n_values >= MIN_SHAPIRO_SAMPLES:
        if n_values < PREFERRED_SHAPIRO_SAMPLES:
            logging.warning(
                "User '%s', feature '%s': Shapiro-Wilk uses only %d observations; "
                "result is less reliable.",
                user_id,
                feature,
                n_values,
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record["shapiro_p"] = float(stats.shapiro(clean_values).pvalue)
        except Exception as exc:
            logging.warning(
                "User '%s', feature '%s': Shapiro-Wilk failed: %s",
                user_id,
                feature,
                exc,
            )
    else:
        logging.warning(
            "User '%s', feature '%s': fewer than %d values; Shapiro-Wilk skipped.",
            user_id,
            feature,
            MIN_SHAPIRO_SAMPLES,
        )

    dip_p, is_multimodal, method = run_dip_test(clean_values)
    record["dip_p"] = dip_p
    record["is_multimodal"] = is_multimodal
    record["multimodality_method"] = method

    record["is_non_normal_high_cv"] = bool(
        np.isfinite(record["shapiro_p"])
        and record["shapiro_p"] < NORMALITY_ALPHA
        and cv_value > 0.15
    )

    reasons: List[str] = []
    if record["is_unstable"]:
        reasons.append("high_cv")
    if record["is_outlier_heavy"]:
        reasons.append("outlier_heavy")
    if record["is_multimodal"]:
        reasons.append("multimodal")
    if record["is_non_normal_high_cv"]:
        reasons.append("non_normal_and_cv_gt_0.15")

    record["exclude"] = len(reasons) > 0
    record["exclusion_reason"] = ";".join(reasons)

    return record


def run_distribution_checks(
    user_id: str,
    df_genuine: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Run distribution-health analysis for every numeric feature of one user."""
    records: List[Dict[str, Any]] = []

    for feature in feature_columns:
        numeric_values = pd.to_numeric(df_genuine[feature], errors="coerce").to_numpy(dtype=float)
        records.append(compute_feature_quality(user_id, feature, numeric_values))

    return pd.DataFrame(records)


# =============================================================================
# DRIFT ANALYSIS
# =============================================================================

def manual_linear_regression_with_pvalue(
    series: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Manual OLS fallback returning slope, intercept, p-value, R².

    This is used only if statsmodels is unavailable. The slope p-value uses the
    standard t-test for the OLS coefficient under the standard linear-model
    assumptions.
    """
    y = np.asarray(series, dtype=float)
    x = np.arange(y.size, dtype=float)

    if y.size < MIN_SESSIONS_FOR_DRIFT_FIT:
        return np.nan, np.nan, np.nan, np.nan

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    ss_xx = float(np.sum(x_centered ** 2))

    if ss_xx < EPS:
        return np.nan, np.nan, np.nan, np.nan

    slope = float(np.sum(x_centered * y_centered) / ss_xx)
    intercept = float(y.mean() - slope * x.mean())
    fitted = intercept + slope * x

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = np.nan if ss_tot < EPS else float(1.0 - ss_res / ss_tot)

    degrees_freedom = y.size - 2
    if degrees_freedom <= 0 or ss_res < EPS:
        return slope, intercept, np.nan, r_squared

    residual_variance = ss_res / degrees_freedom
    standard_error = math.sqrt(residual_variance / ss_xx)

    if standard_error < EPS:
        return slope, intercept, np.nan, r_squared

    t_statistic = slope / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees_freedom))
    return slope, intercept, p_value, r_squared


def linear_drift_test(series: np.ndarray) -> Dict[str, Any]:
    """Fit OLS regression and return slope significance and normalized slope."""
    clean_series = np.asarray(series, dtype=float)
    clean_series = clean_series[np.isfinite(clean_series)]

    result: Dict[str, Any] = {
        "n_sessions": int(clean_series.size),
        "slope": np.nan,
        "intercept": np.nan,
        "normalized_slope": np.nan,
        "p_value": np.nan,
        "R2": np.nan,
        "drift_unassessable": False,
        "linear_drift": False,
    }

    if clean_series.size < MIN_SESSIONS_FOR_DRIFT_FIT:
        result["drift_unassessable"] = True
        return result

    if np.std(clean_series) < EPS:
        result["slope"] = 0.0
        result["intercept"] = float(clean_series[0])
        result["R2"] = 1.0
        result["drift_unassessable"] = True
        return result

    if STATSMODELS_AVAILABLE:
        try:
            x = sm.add_constant(np.arange(clean_series.size, dtype=float))
            model = sm.OLS(clean_series, x).fit()
            result["intercept"] = float(model.params[0])
            result["slope"] = float(model.params[1])
            result["p_value"] = float(model.pvalues[1])
            result["R2"] = float(model.rsquared)
        except Exception as exc:
            logging.warning("statsmodels OLS failed (%s); using manual fallback.", exc)
            slope, intercept, p_value, r_squared = manual_linear_regression_with_pvalue(clean_series)
            result.update(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "p_value": p_value,
                    "R2": r_squared,
                }
            )
    else:
        slope, intercept, p_value, r_squared = manual_linear_regression_with_pvalue(clean_series)
        result.update(
            {
                "slope": slope,
                "intercept": intercept,
                "p_value": p_value,
                "R2": r_squared,
            }
        )

    series_mean = float(np.mean(clean_series))
    if abs(series_mean) < EPS:
        result["normalized_slope"] = np.nan
        result["drift_unassessable"] = True
        return result

    result["normalized_slope"] = float(result["slope"] / abs(series_mean))
    result["linear_drift"] = bool(
        np.isfinite(result["p_value"])
        and result["p_value"] < SLOPE_SIGNIFICANCE_ALPHA
        and abs(result["normalized_slope"]) > MIN_SLOPE_MAGNITUDE
    )
    return result


def rolling_cv_drift(series: np.ndarray) -> Tuple[bool, float, List[float]]:
    """
    Detect rolling-CV instability deterministically.

    Drift is True if:
    1. mean(CV_second_half) > 1.30 * mean(CV_first_half), and
    2. at least ceil(50% of rolling-CV differences) are positive.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < MIN_SESSIONS_FOR_ROLLING_CV:
        return False, np.nan, []

    cvs: List[float] = []
    for start in range(values.size - WINDOW_SIZE + 1):
        window = values[start : start + WINDOW_SIZE]
        mean_value = float(np.mean(window))
        std_value = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
        cv_value = calculate_cv(mean_value, std_value)
        cvs.append(cv_value)

    finite_cvs = np.asarray([cv for cv in cvs if np.isfinite(cv)], dtype=float)
    if finite_cvs.size < 2:
        return False, np.nan, cvs

    midpoint = finite_cvs.size // 2
    if midpoint == 0 or midpoint == finite_cvs.size:
        return False, np.nan, cvs

    first_half = finite_cvs[:midpoint]
    second_half = finite_cvs[midpoint:]

    first_mean = float(np.mean(first_half))
    second_mean = float(np.mean(second_half))

    if abs(first_mean) < EPS:
        cv_change = np.inf if second_mean > EPS else 0.0
    else:
        cv_change = float(second_mean / first_mean)

    differences = np.diff(finite_cvs)
    required_positive_count = math.ceil(
        ROLLING_CV_POSITIVE_DIFF_FRACTION * differences.size
    )
    positive_count = int(np.sum(differences > 0))

    detected = bool(
        second_mean > ROLLING_CV_INCREASE_FACTOR * first_mean
        and positive_count >= required_positive_count
    )
    return detected, cv_change, cvs


def ewma_drift(series: np.ndarray) -> Tuple[bool, str, np.ndarray]:
    """
    Detect persistent EWMA directional drift deterministically.

    A drift direction is reported if a consecutive run of EWMA first differences
    with the same non-zero sign has length >= ceil(0.30 * T).
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < MIN_SESSIONS_FOR_DRIFT_FIT:
        return False, "insufficient_data", np.asarray([], dtype=float)

    # alpha is derived directly from the configured forgetting factor.
    alpha = 1.0 - EXPONENTIAL_FORGETTING_FACTOR
    alpha = min(max(alpha, EPS), 1.0)

    ewma_values = np.empty(values.size, dtype=float)
    ewma_values[0] = values[0]

    for index in range(1, values.size):
        ewma_values[index] = alpha * values[index] + (1.0 - alpha) * ewma_values[index - 1]

    differences = np.diff(ewma_values)
    required_run = max(1, math.ceil(EWMA_PERSISTENCE_FRACTION * values.size))

    longest_positive = 0
    longest_negative = 0
    current_positive = 0
    current_negative = 0

    for difference in differences:
        if difference > EPS:
            current_positive += 1
            current_negative = 0
        elif difference < -EPS:
            current_negative += 1
            current_positive = 0
        else:
            current_positive = 0
            current_negative = 0

        longest_positive = max(longest_positive, current_positive)
        longest_negative = max(longest_negative, current_negative)

    if longest_positive >= required_run:
        return True, "upward", ewma_values
    if longest_negative >= required_run:
        return True, "downward", ewma_values

    return False, "none", ewma_values


def changepoint_drift(series: np.ndarray) -> Tuple[bool, Optional[int], float]:
    """Run optional PELT change-point analysis and Welch t-test segment comparison."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]

    if not RUPTURES_AVAILABLE or values.size < MIN_SESSIONS_FOR_CHANGEPOINT:
        return False, None, np.nan

    try:
        model = rpt.Pelt(model="l2").fit(values.reshape(-1, 1))
        penalty = float(np.log(values.size) * np.var(values))
        breakpoints = model.predict(pen=max(penalty, EPS))

        candidate_breakpoints = [
            point for point in breakpoints if 1 < point < values.size - 1
        ]
        if not candidate_breakpoints:
            return False, None, np.nan

        change_point = int(candidate_breakpoints[0])
        left = values[:change_point]
        right = values[change_point:]

        if left.size < 2 or right.size < 2:
            return False, change_point, np.nan

        p_value = float(
            stats.ttest_ind(left, right, equal_var=False, nan_policy="omit").pvalue
        )
        return bool(np.isfinite(p_value) and p_value < SLOPE_SIGNIFICANCE_ALPHA), change_point, p_value

    except Exception as exc:
        logging.warning("Change-point detection failed: %s", exc)
        return False, None, np.nan


def run_drift_analysis(
    user_id: str,
    df_genuine: pd.DataFrame,
    feature_columns: Sequence[str],
    excluded_features: Sequence[str],
    group_column: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Run feature-level temporal drift analysis for one user."""
    records: List[Dict[str, Any]] = []
    plot_data: Dict[str, Dict[str, Any]] = {}
    excluded_set = set(excluded_features)

    for feature in feature_columns:
        series = aggregate_sessions(
            df=df_genuine,
            feature=feature,
            group_column=group_column,
            aggregation=SESSION_AGGREGATION,
        )
        series = series[np.isfinite(series)]

        linear_result = linear_drift_test(series)
        rolling_detected, rolling_change, rolling_values = rolling_cv_drift(series)
        ewma_detected, ewma_trend, ewma_values = ewma_drift(series)
        cp_detected, cp_index, cp_p_value = changepoint_drift(series)

        unreliable = bool(series.size < MIN_SESSIONS_RELIABLE_DRIFT)
        if unreliable:
            logging.warning(
                "User '%s', feature '%s': only %d temporal observations; "
                "drift estimates are unreliable.",
                user_id,
                feature,
                series.size,
            )

        drift_detected = bool(
            linear_result["linear_drift"]
            or rolling_detected
            or ewma_detected
            or cp_detected
        )

        record = {
            "user_id": user_id,
            "feature": feature,
            "excluded_by_distribution": feature in excluded_set,
            "n_sessions": linear_result["n_sessions"],
            "slope": linear_result["slope"],
            "intercept": linear_result["intercept"],
            "normalized_slope": linear_result["normalized_slope"],
            "p_value": linear_result["p_value"],
            "R2": linear_result["R2"],
            "drift_unassessable": linear_result["drift_unassessable"],
            "unreliable_low_session_count": unreliable,
            "linear_drift": linear_result["linear_drift"],
            "rolling_cv_change": rolling_change,
            "rolling_cv_drift": rolling_detected,
            "ewma_trend": ewma_trend,
            "ewma_drift": ewma_detected,
            "change_point": cp_index,
            "change_point_p_value": cp_p_value,
            "change_point_drift": cp_detected,
            "drift_detected": drift_detected,
        }
        records.append(record)

        plot_data[feature] = {
            "series": series,
            "slope": linear_result["slope"],
            "intercept": linear_result["intercept"],
            "rolling_values": rolling_values,
            "ewma_values": ewma_values,
            "drift_detected": drift_detected,
        }

    return pd.DataFrame(records), plot_data


# =============================================================================
# OPTIONAL FRR SIMULATION
# =============================================================================

def simulate_frr_over_time(
    user_id: str,
    df_genuine: pd.DataFrame,
    feature_columns: Sequence[str],
    group_column: Optional[str],
    plot_dir: Path,
) -> Dict[str, Any]:
    """Simulate late-session FRR using early genuine sessions as OCSVM training."""
    result: Dict[str, Any] = {
        "simulated": False,
        "frr_late": np.nan,
        "n_early_sessions": 0,
        "n_late_sessions": 0,
        "score_session_correlation": np.nan,
        "reason": "",
    }

    if not SIMULATE_FRR_OVER_TIME:
        result["reason"] = "simulation_disabled"
        return result

    if not SKLEARN_AVAILABLE:
        logging.warning(
            "scikit-learn is unavailable; FRR simulation skipped for user '%s'.",
            user_id,
        )
        result["reason"] = "sklearn_unavailable"
        return result

    usable_features = list(feature_columns)
    if not usable_features:
        result["reason"] = "no_numeric_features"
        return result

    if group_column is None:
        session_df = df_genuine[usable_features].copy()
    else:
        session_df = (
            df_genuine.groupby(group_column, sort=False)[usable_features]
            .mean()
            .reset_index(drop=True)
        )

    session_df = session_df.apply(pd.to_numeric, errors="coerce")
    if len(session_df) < MIN_SESSIONS_FOR_DRIFT_FIT:
        result["reason"] = "insufficient_sessions"
        return result

    split_index = int(math.floor(len(session_df) * TRAIN_SIZE_RATIO))
    split_index = max(1, min(split_index, len(session_df) - 1))

    early_df = session_df.iloc[:split_index].copy()
    late_df = session_df.iloc[split_index:].copy()

    if early_df.shape[0] < 2 or late_df.shape[0] < 1:
        result["reason"] = "invalid_train_test_split"
        return result

    medians = early_df.median(axis=0, numeric_only=True)
    early_df = early_df.fillna(medians).fillna(0.0)
    late_df = late_df.fillna(medians).fillna(0.0)

    try:
        scaler = StandardScaler()
        x_early = scaler.fit_transform(early_df.to_numpy(dtype=float))
        x_late = scaler.transform(late_df.to_numpy(dtype=float))

        model = OneClassSVM(
            kernel="rbf",
            nu=OCSVM_NU,
            gamma=OCSVM_GAMMA,
        )
        model.fit(x_early)

        scores = model.decision_function(x_late)
        rejected = scores < OCSVM_THRESHOLD
        frr_late = float(np.mean(rejected))

        late_indices = np.arange(scores.size, dtype=float)
        correlation = np.nan
        if scores.size >= 2 and np.std(scores) > EPS:
            correlation = float(stats.pearsonr(late_indices, scores).statistic)

        result.update(
            {
                "simulated": True,
                "frr_late": frr_late,
                "n_early_sessions": int(early_df.shape[0]),
                "n_late_sessions": int(late_df.shape[0]),
                "score_session_correlation": correlation,
                "reason": "success",
            }
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(late_indices, rejected.astype(int), marker="o", color="tab:red", label="FRR event")
        ax.axhline(
            frr_late,
            color="black",
            linestyle="--",
            label=f"Late FRR = {frr_late:.3f}",
        )
        ax.set_title(f"Late Genuine FRR Simulation — User {user_id}")
        ax.set_xlabel("Late-session index")
        ax.set_ylabel("Rejected genuine session (1=True)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(
            plot_dir / "frr_over_time.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )
        plt.close(fig)

    except Exception as exc:
        logging.warning("FRR simulation failed for user '%s': %s", user_id, exc)
        result["reason"] = f"simulation_failed: {exc}"

    return result


# =============================================================================
# PLOTTING
# =============================================================================

def plot_distribution_feature(
    user_id: str,
    feature: str,
    values: np.ndarray,
    session_series: np.ndarray,
    plot_dir: Path,
) -> None:
    """Save histogram and session-aggregation boxplot for a feature."""
    safe_feature = safe_filename(feature)
    clean_values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))
    if clean_values.size > 0:
        bins = int(
            min(
                HISTOGRAM_MAX_BINS,
                max(HISTOGRAM_MIN_BINS, round(np.sqrt(clean_values.size))),
            )
        )
        ax.hist(clean_values, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(f"Distribution Histogram — User {user_id} — {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.savefig(
        plot_dir / f"dist_hist_{safe_feature}.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    clean_series = session_series[np.isfinite(session_series)]
    if clean_series.size > 0:
        ax.boxplot(clean_series, vert=True, showmeans=True)
    ax.set_title(f"Session-Aggregated Boxplot — User {user_id} — {feature}")
    ax.set_ylabel(f"Session {SESSION_AGGREGATION} of {feature}")
    ax.set_xticks([1])
    ax.set_xticklabels([feature])
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(
        plot_dir / f"box_over_sessions_{safe_feature}.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_feature_time_series(
    user_id: str,
    feature: str,
    plot_info: Dict[str, Any],
    plot_dir: Path,
) -> None:
    """Save chronological session means with fitted linear-regression line."""
    series = np.asarray(plot_info["series"], dtype=float)
    safe_feature = safe_filename(feature)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(series.size)

    ax.plot(x, series, marker="o", linewidth=1.25, color="tab:blue", label="Session aggregate")

    slope = plot_info.get("slope", np.nan)
    intercept = plot_info.get("intercept", np.nan)

    if np.isfinite(slope) and np.isfinite(intercept):
        fitted = intercept + slope * x
        ax.plot(x, fitted, color="tab:orange", linewidth=2, label="OLS regression")

    drift_text = "Drift detected" if plot_info.get("drift_detected", False) else "No drift flag"
    ax.set_title(f"Temporal Drift — User {user_id} — {feature} ({drift_text})")
    ax.set_xlabel("Chronological session index")
    ax.set_ylabel(f"Session {SESSION_AGGREGATION}")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(
        plot_dir / f"{safe_feature}_ts.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_drift_heatmap(
    user_id: str,
    drift_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Save binary per-feature drift heatmap."""
    fig, ax = plt.subplots(
        figsize=(max(8, len(drift_df) * 0.45), 3.5)
    )

    if drift_df.empty:
        ax.text(0.5, 0.5, "No drift data available", ha="center", va="center")
        ax.axis("off")
    else:
        values = drift_df["drift_detected"].astype(int).to_numpy().reshape(1, -1)
        labels = drift_df["feature"].tolist()

        if SEABORN_AVAILABLE:
            sns.heatmap(
                values,
                annot=True,
                fmt="d",
                cmap="RdYlGn_r",
                cbar=False,
                xticklabels=labels,
                yticklabels=["Drift detected"],
                ax=ax,
                vmin=0,
                vmax=1,
            )
        else:
            image = ax.imshow(values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticks([0])
            ax.set_yticklabels(["Drift detected"])
            fig.colorbar(image, ax=ax, label="Drift flag")

    ax.set_title(f"Feature Drift Heatmap — User {user_id}")
    fig.savefig(
        plot_dir / "drift_heatmap.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_rolling_cv(
    user_id: str,
    plot_data: Dict[str, Dict[str, Any]],
    plot_dir: Path,
) -> None:
    """Save combined rolling-CV plot across all analyzable features."""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for feature, info in plot_data.items():
        rolling_values = np.asarray(info.get("rolling_values", []), dtype=float)
        if rolling_values.size > 0 and np.isfinite(rolling_values).any():
            ax.plot(
                np.arange(rolling_values.size),
                rolling_values,
                marker="o",
                linewidth=1,
                label=feature,
            )
            plotted = True

    if plotted:
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_xlabel("Rolling-window start index")
        ax.set_ylabel(f"Rolling CV (window={WINDOW_SIZE})")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data for rolling-CV analysis", ha="center", va="center")
        ax.axis("off")

    ax.set_title(f"Rolling CV Across Features — User {user_id}")
    fig.savefig(plot_dir / "rolling_cv.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ewma_overlay(
    user_id: str,
    plot_data: Dict[str, Dict[str, Any]],
    plot_dir: Path,
) -> None:
    """Save combined EWMA overlay plot across all analyzable features."""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for feature, info in plot_data.items():
        series = np.asarray(info.get("series", []), dtype=float)
        ewma_values = np.asarray(info.get("ewma_values", []), dtype=float)

        if series.size > 0 and ewma_values.size == series.size:
            ax.plot(
                np.arange(series.size),
                ewma_values,
                linewidth=1.5,
                label=f"{feature} EWMA",
            )
            plotted = True

    if plotted:
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_xlabel("Chronological session index")
        ax.set_ylabel("EWMA session aggregate")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data for EWMA analysis", ha="center", va="center")
        ax.axis("off")

    ax.set_title(f"EWMA Feature Trends — User {user_id}")
    fig.savefig(plot_dir / "ewma_overlay.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def make_user_plots(
    user_id: str,
    df_genuine: pd.DataFrame,
    feature_columns: Sequence[str],
    group_column: Optional[str],
    drift_df: pd.DataFrame,
    plot_data: Dict[str, Dict[str, Any]],
    plot_dir: Path,
) -> None:
    """Create all required per-user distribution and drift plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    for feature in feature_columns:
        raw_values = pd.to_numeric(df_genuine[feature], errors="coerce").to_numpy(dtype=float)
        session_series = aggregate_sessions(
            df_genuine,
            feature,
            group_column,
            SESSION_AGGREGATION,
        )
        plot_distribution_feature(
            user_id,
            feature,
            raw_values,
            session_series,
            plot_dir,
        )
        plot_feature_time_series(user_id, feature, plot_data[feature], plot_dir)

    plot_drift_heatmap(user_id, drift_df, plot_dir)
    plot_rolling_cv(user_id, plot_data, plot_dir)
    plot_ewma_overlay(user_id, plot_data, plot_dir)


def make_global_plots(
    global_drift_df: pd.DataFrame,
    global_quality_df: pd.DataFrame,
    user_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create global CV-vs-slope scatter and problem-user bar chart."""
    global_plot_dir = output_dir / "global_plots"
    global_plot_dir.mkdir(parents=True, exist_ok=True)

    merged = global_quality_df.merge(
        global_drift_df[
            ["user_id", "feature", "normalized_slope", "drift_detected"]
        ],
        on=["user_id", "feature"],
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    usable = merged[
        np.isfinite(merged["cv"]) & np.isfinite(merged["normalized_slope"])
    ].copy()

    if not usable.empty:
        colors = np.where(usable["drift_detected"], "tab:red", "tab:green")
        ax.scatter(
            usable["cv"],
            usable["normalized_slope"],
            c=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.axvline(
            MAX_COEFFICIENT_OF_VARIATION,
            color="black",
            linestyle="--",
            label=f"CV threshold={MAX_COEFFICIENT_OF_VARIATION}",
        )
        ax.axhline(
            MIN_SLOPE_MAGNITUDE,
            color="gray",
            linestyle=":",
            label=f"+ slope threshold={MIN_SLOPE_MAGNITUDE}",
        )
        ax.axhline(
            -MIN_SLOPE_MAGNITUDE,
            color="gray",
            linestyle=":",
            label=f"- slope threshold={MIN_SLOPE_MAGNITUDE}",
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite CV/slope pairs available", ha="center", va="center")

    ax.set_title("Global Feature Stability: CV vs. Normalized Drift Slope")
    ax.set_xlabel("Coefficient of Variation")
    ax.set_ylabel("Normalized slope")
    ax.grid(alpha=0.3)
    fig.savefig(
        global_plot_dir / "cv_vs_slope_scatter.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    if not user_summary_df.empty:
        ordered = user_summary_df.sort_values(
            "fraction_drifted_features",
            ascending=False,
        )
        colors = np.where(
            ordered["is_problem_user"],
            "tab:red",
            "tab:blue",
        )
        ax.bar(
            ordered["user_id"],
            ordered["fraction_drifted_features"],
            color=colors,
        )
        ax.axhline(
            PROBLEM_USER_DRIFT_FRACTION,
            color="black",
            linestyle="--",
            label=f"Problem threshold={PROBLEM_USER_DRIFT_FRACTION}",
        )
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No user summary data available", ha="center", va="center")

    ax.set_title("Fraction of Drifting Features per User")
    ax.set_xlabel("User")
    ax.set_ylabel("Fraction of drift-detected features")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(
        global_plot_dir / "problem_users_bar.png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_selected_features_json(
    global_quality_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, List[str]]:
    """Write exactly the required selected_features.json schema."""
    output_path = output_dir / "selected_features.json"

    if global_quality_df.empty:
        payload = {
            "selected_features": [],
            "excluded_features": [],
            "global_excluded_features": [],
        }
    else:
        feature_stats = (
            global_quality_df.groupby("feature", dropna=False)["exclude"]
            .agg(["count", "sum"])
            .reset_index()
        )
        feature_stats["exclude_fraction"] = (
            feature_stats["sum"] / feature_stats["count"]
        )
        feature_stats["pass_fraction"] = 1.0 - feature_stats["exclude_fraction"]

        excluded_features = sorted(
            feature_stats.loc[
                feature_stats["exclude_fraction"] >= GLOBAL_EXCLUSION_FRACTION,
                "feature",
            ].astype(str).tolist()
        )

        selected_features = sorted(
            feature_stats.loc[
                feature_stats["pass_fraction"] > SELECTED_FEATURE_PASS_FRACTION,
                "feature",
            ].astype(str).tolist()
        )

        global_excluded_features = sorted(
            feature_stats.loc[
                feature_stats["exclude_fraction"] >= GLOBAL_EXCLUSION_FRACTION,
                "feature",
            ].astype(str).tolist()
        )

        payload = {
            "selected_features": selected_features,
            "excluded_features": excluded_features,
            "global_excluded_features": global_excluded_features,
        }

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    return payload


def write_diagnostic_summary(
    output_dir: Path,
    results: Sequence[UserResult],
    skipped_users: Sequence[str],
    global_quality_df: pd.DataFrame,
    global_drift_df: pd.DataFrame,
    user_summary_df: pd.DataFrame,
    selected_payload: Dict[str, List[str]],
) -> None:
    """Write comprehensive human-readable diagnostic_summary.txt."""
    summary_path = output_dir / "diagnostic_summary.txt"

    total_genuine_rows = sum(result.genuine_row_count for result in results)
    total_features = int(global_quality_df["feature"].nunique()) if not global_quality_df.empty else 0

    lines: List[str] = []
    lines.append("FEATURE-STABILITY & TEMPORAL-DRIFT DIAGNOSTIC SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 72)
    lines.append(f"Users requested: {len(USER_LIST)}")
    lines.append(f"Users successfully processed: {len(results)}")
    lines.append(f"Users skipped: {len(skipped_users)}")
    lines.append(f"Total genuine training rows analyzed: {total_genuine_rows}")
    lines.append(f"Unique numeric features observed: {total_features}")
    lines.append(f"Session aggregation method: {SESSION_AGGREGATION}")
    lines.append("")

    if skipped_users:
        lines.append("SKIPPED USERS")
        lines.append("-" * 72)
        lines.extend(f"- {user}" for user in skipped_users)
        lines.append("")

    lines.append("OPTIONAL DEPENDENCY STATUS")
    lines.append("-" * 72)
    lines.append(f"- statsmodels available: {STATSMODELS_AVAILABLE}")
    lines.append(f"- diptests available: {DIPTEST_AVAILABLE}")
    lines.append(f"- ruptures available: {RUPTURES_AVAILABLE}")
    lines.append(f"- scikit-learn available: {SKLEARN_AVAILABLE}")
    lines.append("")

    if not global_quality_df.empty:
        finite_cv = global_quality_df.loc[np.isfinite(global_quality_df["cv"]), "cv"]
        finite_outlier = global_quality_df.loc[
            np.isfinite(global_quality_df["outlier_frac"]),
            "outlier_frac",
        ]

        normality_tested = global_quality_df["shapiro_p"].notna()
        rejected_normality = (
            global_quality_df.loc[normality_tested, "shapiro_p"] < NORMALITY_ALPHA
        )

        lines.append("GLOBAL DISTRIBUTION HEALTH")
        lines.append("-" * 72)
        lines.append(f"Median CV: {finite_cv.median() if not finite_cv.empty else np.nan:.6f}")
        lines.append(f"Mean CV: {finite_cv.mean() if not finite_cv.empty else np.nan:.6f}")
        lines.append(
            "Mean outlier fraction: "
            f"{finite_outlier.mean() if not finite_outlier.empty else np.nan:.6f}"
        )
        lines.append(
            "Normality rejection rate: "
            f"{rejected_normality.mean() if not rejected_normality.empty else np.nan:.4f}"
        )
        lines.append(
            "Features excluded at user-level: "
            f"{int(global_quality_df['exclude'].sum())}/{len(global_quality_df)}"
        )
        lines.append("")

    lines.append("FEATURE SELECTION")
    lines.append("-" * 72)
    lines.append(f"Selected features ({len(selected_payload['selected_features'])}):")
    lines.append(", ".join(selected_payload["selected_features"]) or "None")
    lines.append("")
    lines.append(f"Excluded features ({len(selected_payload['excluded_features'])}):")
    lines.append(", ".join(selected_payload["excluded_features"]) or "None")
    lines.append("")

    if not user_summary_df.empty:
        lines.append("USER DRIFT SUMMARY")
        lines.append("-" * 72)
        sorted_users = user_summary_df.sort_values(
            ["fraction_drifted_features", "mean_normalized_slope"],
            ascending=False,
        )

        for _, row in sorted_users.iterrows():
            lines.append(
                f"- {row['user_id']}: "
                f"drift_fraction={row['fraction_drifted_features']:.3f}, "
                f"mean_normalized_slope={row['mean_normalized_slope']:.6f}, "
                f"FRR_late={row['frr_late']:.3f}"
                if np.isfinite(row["frr_late"])
                else
                f"- {row['user_id']}: "
                f"drift_fraction={row['fraction_drifted_features']:.3f}, "
                f"mean_normalized_slope={row['mean_normalized_slope']:.6f}, "
                f"FRR_late=not simulated"
            )
        lines.append("")

        problem_users = user_summary_df[user_summary_df["is_problem_user"]]
        lines.append("PROBLEM USERS")
        lines.append("-" * 72)
        if problem_users.empty:
            lines.append("No users exceeded configured problem-user thresholds.")
        else:
            for _, row in problem_users.iterrows():
                lines.append(
                    f"- {row['user_id']}: fraction_drifted_features="
                    f"{row['fraction_drifted_features']:.3f}, "
                    f"mean_normalized_slope={row['mean_normalized_slope']:.6f}"
                )
        lines.append("")

    if not global_quality_df.empty:
        feature_frequency = (
            global_quality_df.groupby("feature")
            .agg(
                mean_cv=("cv", "mean"),
                exclusion_rate=("exclude", "mean"),
                mean_outlier_fraction=("outlier_frac", "mean"),
            )
            .sort_values(["exclusion_rate", "mean_cv"], ascending=False)
            .head(5)
        )

        lines.append("TOP-5 WORST FEATURES")
        lines.append("-" * 72)
        for feature, row in feature_frequency.iterrows():
            lines.append(
                f"- {feature}: exclusion_rate={row['exclusion_rate']:.3f}, "
                f"mean_cv={row['mean_cv']:.6f}, "
                f"mean_outlier_fraction={row['mean_outlier_fraction']:.6f}"
            )
        lines.append("")

    lines.append("DATA-QUALITY CAVEATS")
    lines.append("-" * 72)
    caveats_written = False
    for result in results:
        low_session_features = result.drift_summary[
            result.drift_summary["unreliable_low_session_count"]
        ]
        if not low_session_features.empty:
            lines.append(
                f"- User {result.user_id} has fewer than "
                f"{MIN_SESSIONS_RELIABLE_DRIFT} temporal observations for one or more "
                "features; drift estimates should be interpreted cautiously."
            )
            caveats_written = True

        if "original CSV row order" in result.ordering_method:
            lines.append(
                f"- User {result.user_id} has no session_id/session_number; "
                "original CSV row order was assumed chronological."
            )
            caveats_written = True

    if not caveats_written:
        lines.append("- No major structural caveats detected.")
    lines.append("")

    lines.append("RECOMMENDATIONS")
    lines.append("-" * 72)

    if selected_payload["global_excluded_features"]:
        lines.append(
            "- Remove global_excluded_features from future OCSVM feature sets unless "
            "there is strong domain evidence to retain them."
        )

    if not user_summary_df.empty:
        for _, row in user_summary_df.iterrows():
            high_drift = row["fraction_drifted_features"] > PROBLEM_USER_DRIFT_FRACTION
            high_frr = np.isfinite(row["frr_late"]) and row["frr_late"] > HIGH_FRR_THRESHOLD

            if high_drift and high_frr:
                lines.append(
                    f"- User {row['user_id']}: late FRR is high "
                    f"({row['frr_late']:.3f}) and many features drift. Retrain on the "
                    f"most recent {int(RECENT_TRAINING_FRACTION * 100)}% of genuine "
                    "sessions and consider exponential forgetting."
                )
            elif high_drift:
                lines.append(
                    f"- User {row['user_id']}: many drifting features detected. "
                    "Consider adaptive windows, recent-session retraining, and "
                    "per-user feature exclusion."
                )
            elif high_frr:
                lines.append(
                    f"- User {row['user_id']}: late FRR is high "
                    f"({row['frr_late']:.3f}) even without widespread drift. Review "
                    "feature preprocessing, OCSVM threshold calibration, and sample size."
                )

    lines.append(
        f"- Consider exponential forgetting with factor "
        f"{EXPONENTIAL_FORGETTING_FACTOR} for users whose EWMA trends persist."
    )
    lines.append(
        "- For unstable sensor features, investigate robust scaling, clipping, "
        "sensor calibration, or replacement with more stable derived statistics."
    )
    lines.append(
        "- Collect at least 20 chronological genuine sessions per user when possible "
        "to improve drift-estimation reliability."
    )
    lines.append("")

    with summary_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))


# =============================================================================
# USER PROCESSING
# =============================================================================

def process_user(
    user_id: str,
    data_dir: Path,
    output_dir: Path,
) -> Optional[UserResult]:
    """Run end-to-end diagnostics for one user without mixing external user data."""
    logging.info("Starting diagnostics for user '%s'.", user_id)

    loaded = load_user_data(user_id, data_dir)
    if loaded is None:
        return None

    train_df, test_df = loaded
    label_column = detect_label_column(train_df, test_df, LABEL_CANDIDATES)

    train_working = train_df.copy()

    if label_column is not None and label_column in train_working.columns:
        train_working["__label__"] = train_working[label_column].apply(normalize_label)
        df_genuine = train_working.loc[
            train_working["__label__"] == "genuine"
        ].copy()
    else:
        logging.warning(
            "User '%s': no usable label column in training CSV. "
            "Treating all training rows as genuine.",
            user_id,
        )
        df_genuine = train_working.copy()
        df_genuine["__label__"] = "genuine"

    if df_genuine.empty:
        logging.error(
            "User '%s': no genuine training rows after filtering. Skipping.",
            user_id,
        )
        return None

    df_genuine, group_column, ordering_method = sort_sessions_chronologically(df_genuine)
    feature_columns = select_numeric_features(df_genuine, label_column)

    if not feature_columns:
        logging.error(
            "User '%s': no usable numeric feature columns after preprocessing. Skipping.",
            user_id,
        )
        return None

    user_dir = output_dir / "user_specific" / safe_filename(user_id)
    plot_dir = user_dir / "plots"
    user_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running distribution checks for user '%s'.", user_id)
    feature_quality_df = run_distribution_checks(user_id, df_genuine, feature_columns)
    feature_quality_df.to_csv(user_dir / "feature_quality.csv", index=False)

    excluded_features = feature_quality_df.loc[
        feature_quality_df["exclude"],
        "feature",
    ].tolist()

    logging.info("Running temporal drift analysis for user '%s'.", user_id)
    drift_df, plot_data = run_drift_analysis(
        user_id=user_id,
        df_genuine=df_genuine,
        feature_columns=feature_columns,
        excluded_features=excluded_features,
        group_column=group_column,
    )

    fraction_drifted = (
        float(drift_df["drift_detected"].mean()) if not drift_df.empty else np.nan
    )
    mean_normalized_slope = (
        float(drift_df["normalized_slope"].mean(skipna=True))
        if not drift_df.empty
        else np.nan
    )

    logging.info("Running FRR simulation for user '%s'.", user_id)
    frr_result = simulate_frr_over_time(
        user_id=user_id,
        df_genuine=df_genuine,
        feature_columns=feature_columns,
        group_column=group_column,
        plot_dir=plot_dir,
    )

    drift_df["fraction_drifted_features"] = fraction_drifted
    drift_df["mean_normalized_slope_user"] = mean_normalized_slope
    drift_df["frr_late"] = frr_result["frr_late"]
    drift_df["ordering_method"] = ordering_method
    drift_df.to_csv(user_dir / "drift_summary.csv", index=False)

    logging.info("Generating plots for user '%s'.", user_id)
    make_user_plots(
        user_id=user_id,
        df_genuine=df_genuine,
        feature_columns=feature_columns,
        group_column=group_column,
        drift_df=drift_df,
        plot_data=plot_data,
        plot_dir=plot_dir,
    )

    user_summary = {
        "user_id": user_id,
        "genuine_rows": int(len(df_genuine)),
        "feature_count": int(len(feature_columns)),
        "excluded_feature_count": int(feature_quality_df["exclude"].sum()),
        "fraction_drifted_features": fraction_drifted,
        "mean_normalized_slope": mean_normalized_slope,
        "frr_late": frr_result["frr_late"],
        "frr_simulated": frr_result["simulated"],
        "n_early_sessions": frr_result["n_early_sessions"],
        "n_late_sessions": frr_result["n_late_sessions"],
        "score_session_correlation": frr_result["score_session_correlation"],
        "ordering_method": ordering_method,
    }

    with (user_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(user_summary, file_handle, indent=2, default=str)

    logging.info("Completed diagnostics for user '%s'.", user_id)

    return UserResult(
        user_id=user_id,
        feature_quality=feature_quality_df,
        drift_summary=drift_df,
        user_summary=user_summary,
        frr_result=frr_result,
        genuine_row_count=len(df_genuine),
        feature_count=len(feature_columns),
        ordering_method=ordering_method,
    )


# =============================================================================
# GLOBAL ORCHESTRATION
# =============================================================================

def run_diagnostics(
    user_list: Sequence[str],
    data_dir: str,
    output_dir: str,
) -> None:
    """Run all user-specific diagnostics and final global aggregation."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "user_specific").mkdir(parents=True, exist_ok=True)
    (output_path / "global_plots").mkdir(parents=True, exist_ok=True)

    if not DIPTEST_AVAILABLE:
        logging.warning(
            "Optional dependency 'diptests' is unavailable; histogram peak fallback "
            "will be used for multimodality."
        )
    if not RUPTURES_AVAILABLE:
        logging.warning(
            "Optional dependency 'ruptures' is unavailable; change-point detection "
            "will be skipped."
        )
    if not STATSMODELS_AVAILABLE:
        logging.warning(
            "Optional dependency 'statsmodels' is unavailable; manual OLS slope "
            "p-value calculation will be used."
        )
    if SIMULATE_FRR_OVER_TIME and not SKLEARN_AVAILABLE:
        logging.warning(
            "Optional dependency 'scikit-learn' is unavailable; FRR simulation "
            "will be skipped."
        )

    results: List[UserResult] = []
    skipped_users: List[str] = []

    for user_id in tqdm(user_list, desc="Processing users", unit="user"):
        try:
            result = process_user(user_id, data_path, output_path)
            if result is None:
                skipped_users.append(user_id)
            else:
                results.append(result)
        except Exception as exc:
            logging.exception(
                "Unhandled error while processing user '%s'; skipping user: %s",
                user_id,
                exc,
            )
            skipped_users.append(user_id)

    if results:
        global_quality_df = pd.concat(
            [result.feature_quality for result in results],
            ignore_index=True,
        )
        global_drift_df = pd.concat(
            [result.drift_summary for result in results],
            ignore_index=True,
        )
        user_summary_df = pd.DataFrame(
            [result.user_summary for result in results]
        )
    else:
        global_quality_df = pd.DataFrame()
        global_drift_df = pd.DataFrame()
        user_summary_df = pd.DataFrame()

    if not user_summary_df.empty:
        user_summary_df["is_problem_user"] = (
            (user_summary_df["fraction_drifted_features"] > PROBLEM_USER_DRIFT_FRACTION)
            | (
                user_summary_df["mean_normalized_slope"].abs()
                > PROBLEM_USER_MEAN_NORMALIZED_SLOPE
            )
        )

        if not global_drift_df.empty:
            global_drift_df = global_drift_df.merge(
                user_summary_df[
                    [
                        "user_id",
                        "fraction_drifted_features",
                        "mean_normalized_slope",
                        "is_problem_user",
                    ]
                ],
                on="user_id",
                how="left",
                suffixes=("", "_user"),
            )

    global_drift_df.to_csv(output_path / "global_drift_summary.csv", index=False)

    selected_payload = write_selected_features_json(global_quality_df, output_path)

    make_global_plots(
        global_drift_df=global_drift_df,
        global_quality_df=global_quality_df,
        user_summary_df=user_summary_df,
        output_dir=output_path,
    )

    write_diagnostic_summary(
        output_dir=output_path,
        results=results,
        skipped_users=skipped_users,
        global_quality_df=global_quality_df,
        global_drift_df=global_drift_df,
        user_summary_df=user_summary_df,
        selected_payload=selected_payload,
    )

    logging.info("Diagnostics complete. Outputs written to: %s", output_path.resolve())


def main() -> None:
    """Configure reproducibility and execute diagnostics."""
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    output_path = Path(OUTPUT_DIR)
    configure_logging(output_path)

    logging.info("Starting feature-stability and temporal-drift diagnostics.")
    logging.info("Random seed: %d", RANDOM_SEED)
    logging.info("Data directory: %s", Path(DATA_DIR).resolve())
    logging.info("Output directory: %s", output_path.resolve())

    run_diagnostics(
        user_list=USER_LIST,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
