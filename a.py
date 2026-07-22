#!/usr/bin/env python3
"""
Deep Per‑User Feature Audit with Rich Metrics and Visual Flags
---------------------------------------------------------------
For each user, computes a comprehensive set of metrics per feature,
flags problematic ones with red emojis 🔴, explains why, and saves
separate CSVs for healthy and problematic features.
"""

import os, sys, json, glob, warnings
import numpy as np
import pandas as pd
from scipy.stats import (
    skew, kurtosis, mannwhitneyu, spearmanr, iqr
)
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================= CONFIGURATION =======================
CONFIG = {
    "USER_LIST": [
        'srikanth', 'Samarth', 'harshit', 'avinash', 'deepika',
        'kavya', 'manoj', 'nandini', 'pranay', 'rajesh',
        'sneha', 'varun', 'vijay', 'yash', 'zara'
    ],
    "FEATURES_DIR": "./features/new_app",
    "SELECTED_FEATURES_JSON": "selected_features.json",

    # Thresholds for flagging
    "MISSING_PCT": 10.0,
    "CONSTANT_PCT": 95.0,
    "CV_THRESHOLD": 2.0,
    "SKEW_THRESHOLD": 3.0,
    "KURTOSIS_THRESHOLD": 8.0,
    "MODZ_THRESHOLD": 3.5,
    "OUTLIER_FRACTION_THRESHOLD": 0.1,
    "ABSOLUTE_EXTREME": 1e9,
    "NEAR_ZERO_STD": 1e-10,
    "TEMPORAL_DRIFT_PVAL": 0.01,
    "TRAIN_TEST_SHIFT_PVAL": 0.01,
    "BIMODALITY_COEFF_THRESH": 0.55,
    "DUPLICATE_PCT": 70.0,
    "MAD_RATIO_THRESH": 2.0,
    "IQR_RATIO_THRESH": 3.0,
    "SPREAD_RATIO_THRESH": 1e6,
}

OUTPUT_DIR = "audit_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================

def load_selected_features(json_path):
    with open(json_path, 'r') as f:
        features = json.load(f)
    if not isinstance(features, list):
        raise ValueError("selected_features.json must contain a JSON list of feature names")
    logging.info(f"Loaded {len(features)} features to audit.")
    return features

def load_user_sessions(user, feat_list):
    train_path = os.path.join(CONFIG["FEATURES_DIR"], f"{user}_training_sessions.csv")
    test_path  = os.path.join(CONFIG["FEATURES_DIR"], f"{user}_testing_sessions.csv")
    dfs = []
    for path, label in [(train_path, 'train'), (test_path, 'test')]:
        if not os.path.exists(path):
            logging.warning(f"Missing {path} for user {user}")
            continue
        df = pd.read_csv(path)
        available = [f for f in feat_list if f in df.columns]
        if not available:
            continue
        df = df[available + (['session_id'] if 'session_id' in df.columns else [])]
        if 'session_id' not in df.columns:
            df['session_id'] = [f"{label}_{i}" for i in range(len(df))]
        else:
            df['session_id'] = df['session_id'].astype(str)
        df['_source'] = label
        dfs.append(df)
    if not dfs:
        return None, []
    combined = pd.concat(dfs, ignore_index=True)
    for f in available:
        combined[f] = pd.to_numeric(combined[f], errors='coerce')
    return combined, available

def compute_all_metrics(series, session_ids, source_labels, session_order):
    """Return a dictionary of all computed metrics and a list of issue descriptions."""
    x = series.values.astype(float)
    n = len(x)
    nan_mask = np.isnan(x)
    n_missing = nan_mask.sum()
    clean = x[~nan_mask]
    n_clean = len(clean)

    metrics = {
        'n_total': n,
        'n_missing': n_missing,
        'missing_pct': 100.0 * n_missing / n if n > 0 else 100.0,
        'n_clean': n_clean,
        'mean': np.nan,
        'std': np.nan,
        'min': np.nan,
        'max': np.nan,
        'median': np.nan,
        'mad': np.nan,
        'iqr': np.nan,
        'p01': np.nan,
        'p05': np.nan,
        'p95': np.nan,
        'p99': np.nan,
        'skew': np.nan,
        'kurtosis': np.nan,
        'cv': np.nan,
        'outlier_fraction': np.nan,
        'max_abs_modz': np.nan,
        'temporal_corr': np.nan,
        'temporal_pval': np.nan,
        'train_test_shift_pval': np.nan,
        'bimodality_coeff': np.nan,
        'duplicate_pct': np.nan,
        'mad_ratio': np.nan,
        'iqr_ratio': np.nan,
        'spread_ratio': np.nan,
    }

    reasons = []   # list of strings explaining problems
    flagged_metrics = set()  # set of metric keys that are flagged

    if n_clean == 0:
        metrics['all_missing'] = True
        reasons.append("All values missing")
        flagged_metrics.add('missing_pct')
        return metrics, reasons, flagged_metrics

    # Basic stats
    clean_vals = clean[~np.isnan(clean)]  # just in case
    if len(clean_vals) == 0:
        return metrics, reasons, flagged_metrics

    mean_v = np.mean(clean_vals)
    std_v = np.std(clean_vals, ddof=1)
    med = np.median(clean_vals)
    mad = np.median(np.abs(clean_vals - med)) if len(clean_vals) > 1 else 0.0
    iqr_val = iqr(clean_vals)
    p01, p05, p95, p99 = np.percentile(clean_vals, [1,5,95,99])
    sk = skew(clean_vals)
    kurt = kurtosis(clean_vals)  # excess
    cv = abs(std_v / mean_v) if mean_v != 0 else (np.inf if std_v > 0 else 0.0)

    metrics.update({
        'mean': mean_v,
        'std': std_v,
        'min': np.min(clean_vals),
        'max': np.max(clean_vals),
        'median': med,
        'mad': mad,
        'iqr': iqr_val,
        'p01': p01, 'p05': p05, 'p95': p95, 'p99': p99,
        'skew': sk, 'kurtosis': kurt,
        'cv': cv,
    })

    # Missing check
    if (100.0 * n_missing / n) > CONFIG["MISSING_PCT"]:
        reasons.append(f"High missing ({metrics['missing_pct']:.1f}%)")
        flagged_metrics.add('missing_pct')

    # Inf check
    inf_mask = np.isinf(clean_vals)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        reasons.append(f"{n_inf} Inf values")
        flagged_metrics.add('n_clean')  # placeholder

    # Constant / near‑constant
    if std_v < CONFIG["NEAR_ZERO_STD"]:
        reasons.append("Constant (zero variance)")
        flagged_metrics.add('std')
    else:
        most_common_pct = 100.0 * pd.Series(clean_vals).value_counts().iloc[0] / n_clean
        if most_common_pct >= CONFIG["CONSTANT_PCT"]:
            reasons.append(f"Near‑constant ({most_common_pct:.1f}% identical)")
            flagged_metrics.add('duplicate_pct')

    # Extreme values
    extreme_mask = np.abs(clean_vals) > CONFIG["ABSOLUTE_EXTREME"]
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        ext_sessions = [session_ids[i] for i in np.where(extreme_mask)[0]][:5]
        reasons.append(f"{n_extreme} extreme values (sessions: {', '.join(ext_sessions)})")
        flagged_metrics.add('max')
        flagged_metrics.add('min')

    # High dispersion
    if cv > CONFIG["CV_THRESHOLD"]:
        reasons.append(f"High CV ({cv:.2f})")
        flagged_metrics.add('cv')
    if med != 0 and mad > 0:
        mad_ratio = mad / abs(med)
        iqr_ratio = iqr_val / abs(med)
        metrics['mad_ratio'] = mad_ratio
        metrics['iqr_ratio'] = iqr_ratio
        if mad_ratio > CONFIG["MAD_RATIO_THRESH"]:
            reasons.append(f"High MAD/median ({mad_ratio:.2f})")
            flagged_metrics.add('mad_ratio')
        if iqr_ratio > CONFIG["IQR_RATIO_THRESH"]:
            reasons.append(f"High IQR/median ({iqr_ratio:.2f})")
            flagged_metrics.add('iqr_ratio')
    spread = (p99 - p01) / (abs(med) + 1e-8)
    metrics['spread_ratio'] = spread
    if spread > CONFIG["SPREAD_RATIO_THRESH"]:
        reasons.append(f"Extreme spread ({spread:.2e})")
        flagged_metrics.add('spread_ratio')

    # Outliers
    if n_clean > 3 and mad > 0:
        mod_z = 0.6745 * (clean_vals - med) / mad
        outlier_mask = np.abs(mod_z) > CONFIG["MODZ_THRESHOLD"]
        n_out = outlier_mask.sum()
        outlier_frac = n_out / n_clean
        metrics['outlier_fraction'] = outlier_frac
        metrics['max_abs_modz'] = np.max(np.abs(mod_z))
        if outlier_frac > CONFIG["OUTLIER_FRACTION_THRESHOLD"]:
            out_sessions = [session_ids[i] for i in np.where(outlier_mask)[0]][:10]
            reasons.append(f"Many outliers ({outlier_frac:.1%}, sessions: {', '.join(out_sessions)})")
            flagged_metrics.add('outlier_fraction')
        elif np.max(np.abs(mod_z)) > 5.0:
            out_sessions = [session_ids[i] for i in np.where(outlier_mask)[0]][:5]
            reasons.append(f"Extreme outlier (max modZ={np.max(np.abs(mod_z)):.1f}, sessions: {', '.join(out_sessions)})")
            flagged_metrics.add('max_abs_modz')

    # Distribution shape
    if abs(sk) > CONFIG["SKEW_THRESHOLD"]:
        reasons.append(f"High skewness ({sk:.2f})")
        flagged_metrics.add('skew')
    if kurt > CONFIG["KURTOSIS_THRESHOLD"]:
        reasons.append(f"High kurtosis ({kurt:.2f})")
        flagged_metrics.add('kurtosis')
    if n_clean > 3:
        bc = (sk**2 + 1) / (kurt + 3) if (kurt + 3) != 0 else 0
        metrics['bimodality_coeff'] = bc
        if bc > CONFIG["BIMODALITY_COEFF_THRESH"]:
            reasons.append(f"Bimodal (BC={bc:.2f})")
            flagged_metrics.add('bimodality_coeff')

    # Temporal drift
    if session_order is not None and n_clean > 10:
        mask = ~np.isnan(series)
        corr, pval = spearmanr(session_order[mask], series[mask])
        metrics['temporal_corr'] = corr
        metrics['temporal_pval'] = pval
        if pval < CONFIG["TEMPORAL_DRIFT_PVAL"] and abs(corr) > 0.3:
            reasons.append(f"Temporal drift (r={corr:.2f}, p={pval:.4f})")
            flagged_metrics.add('temporal_pval')

    # Train‑test shift
    train_mask = (np.array(source_labels) == 'train')[~nan_mask]
    test_mask = (np.array(source_labels) == 'test')[~nan_mask]
    if train_mask.sum() > 3 and test_mask.sum() > 3:
        try:
            _, pval = mannwhitneyu(clean_vals[train_mask], clean_vals[test_mask], alternative='two-sided')
            metrics['train_test_shift_pval'] = pval
            if pval < CONFIG["TRAIN_TEST_SHIFT_PVAL"]:
                reasons.append(f"Train‑test shift (p={pval:.4f})")
                flagged_metrics.add('train_test_shift_pval')
        except:
            pass

    # Duplicate dominance (quantization)
    dup_pct = 100.0 * pd.Series(clean_vals).value_counts().iloc[0] / n_clean
    metrics['duplicate_pct'] = dup_pct
    if dup_pct > CONFIG["DUPLICATE_PCT"]:
        reasons.append(f"Quantized ({dup_pct:.1f}% duplicates)")
        flagged_metrics.add('duplicate_pct')

    return metrics, reasons, flagged_metrics

def format_metrics_for_csv(metrics, flagged_metrics):
    """Return a copy of metrics with 🔴 appended to flagged metric values."""
    formatted = {}
    for key, val in metrics.items():
        if key in flagged_metrics and val is not None and not isinstance(val, bool):
            # Convert to string with red emoji
            formatted[key] = f"{val} 🔴"
        else:
            formatted[key] = val
    return formatted

def audit_user(user, features):
    sessions_df, available = load_user_sessions(user, features)
    if sessions_df is None:
        logging.info(f"  {user}: no data found")
        return None, [], []

    session_ids = sessions_df['session_id'].tolist()
    source_labels = sessions_df['_source'].tolist()
    if 'session_order' in sessions_df.columns:
        session_order = sessions_df['session_order'].values
    else:
        session_order = np.arange(len(sessions_df))

    problematic_rows = []
    healthy_rows = []
    global_issue_counts = {}

    for feat in available:
        feat_series = sessions_df[feat]
        metrics, reasons, flagged = compute_all_metrics(feat_series, session_ids, source_labels, session_order)
        # Determine if problematic
        is_problematic = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else ""

        # Prepare row: feature name + formatted metrics + reason
        row = {'feature': feat, 'is_problematic': is_problematic, 'reason': reason_str}
        formatted_metrics = format_metrics_for_csv(metrics, flagged)
        row.update(formatted_metrics)

        if is_problematic:
            problematic_rows.append(row)
            global_issue_counts[feat] = global_issue_counts.get(feat, 0) + 1
        else:
            healthy_rows.append(row)

    # Save separate CSVs
    if problematic_rows:
        df_prob = pd.DataFrame(problematic_rows)
        df_prob.to_csv(os.path.join(OUTPUT_DIR, f"audit_{user}_problematic.csv"), index=False)
    if healthy_rows:
        df_health = pd.DataFrame(healthy_rows)
        df_health.to_csv(os.path.join(OUTPUT_DIR, f"audit_{user}_healthy.csv"), index=False)

    num_prob = len(problematic_rows)
    num_health = len(healthy_rows)
    logging.info(f"  {user}: {num_prob} problematic, {num_health} healthy features")
    return global_issue_counts, num_prob, num_health

def main():
    logging.info("Starting deep per‑user feature audit with enhanced output...")
    features = load_selected_features(CONFIG["SELECTED_FEATURES_JSON"])
    if not features:
        logging.error("No features loaded.")
        return

    all_global_counts = {}
    for user in CONFIG["USER_LIST"]:
        user_counts, _, _ = audit_user(user, features)
        if user_counts:
            for feat, cnt in user_counts.items():
                all_global_counts[feat] = all_global_counts.get(feat, 0) + cnt

    if all_global_counts:
        summary_df = pd.DataFrame(
            [{'feature': f, 'n_users_affected': cnt} for f, cnt in all_global_counts.items()]
        ).sort_values('n_users_affected', ascending=False)
        summary_path = os.path.join(OUTPUT_DIR, "global_audit_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Global summary saved to {summary_path}")
        print("\n=== Top problematic features (most users affected) ===")
        print(summary_df.head(20).to_string(index=False))
    else:
        logging.info("No issues found across all users – excellent data quality!")

if __name__ == "__main__":
    main()
