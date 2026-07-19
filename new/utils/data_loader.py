# utils/data_loader.py
"""
Leakage-free data loading for the per-user training/testing file structure.

File layout expected in data.features_dir:
    <user_id>_training_sessions.csv   -> all genuine sessions
    <user_id>_testing_sessions.csv    -> genuine + impostor, with a label column

Design (leakage-free):
  1. Load raw files, preserve provenance (training file vs testing file).
  2. Split each user's TESTING sessions (stratified by is_genuine) into
     dev portion and holdout portion using split.dev_ratio.
  3. Development set = training-file sessions + dev portion of testing file.
  4. ALL cleaning decisions (missing threshold, constant features) and ALL
     imputation medians are computed on the DEVELOPMENT set only, then
     applied identically to the holdout set.
  5. A split manifest is saved so every phase reuses the identical split.

Public API:
    load_split_data(config) -> (dev_data, holdout_data, feature_list)
        dev_data / holdout_data: {user_id: DataFrame(features + 'is_genuine')}
"""

import warnings

# Must come before importing pandas to catch import-time pandas warnings
warnings.filterwarnings("ignore", message=r".*numexpr.*")
warnings.filterwarnings("ignore", message=r".*highly fragmented.*")

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
import glob
from contextlib import contextmanager

# ----------------------------------------------------------------------
# Localised warning suppression (avoids global side effects)
# ----------------------------------------------------------------------
@contextmanager
def suppress_pd_warnings():
    """Suppress specific pandas PerformanceWarnings (fragmentation, numexpr)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*numexpr.*")
        warnings.filterwarnings("ignore", message=".*highly fragmented.*")
        yield

def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV with common performance warnings suppressed."""
    with suppress_pd_warnings():
        return pd.read_csv(path, **kwargs)

# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LABEL_CANDIDATES = ['session_label', 'label', 'test_type']


def _norm_label(x) -> int:
    """
    Map a raw label to 1 (genuine) / 0 (impostor). NaN -> -1 (invalid).

    FIX (CRITICAL): Exact string match to 'genuine' after stripping,
    avoiding misclassification of 'non-genuine', 'pseudo-genuine', etc.
    """
    if pd.isna(x):
        return -1
    label_str = str(x).lower().strip()
    return 1 if label_str == 'genuine' else 0


def _load_raw_users(config: dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load raw per-user data preserving provenance.

    Returns {user_id: {'train': DataFrame, 'test': DataFrame}}
    Both frames contain an 'is_genuine' column; 'test' additionally reflects
    its original labels.
    """
    features_dir = config['data']['features_dir']
    train_files = sorted(glob.glob(os.path.join(features_dir, '*_training_sessions.csv')))
    if not train_files:
        raise FileNotFoundError(f"No training files found in {features_dir}")

    raw = {}
    for train_path in train_files:
        user_id = os.path.basename(train_path).replace('_training_sessions.csv', '')

        train_df = read_csv_safe(train_path)
        # Drop any label columns from training (all genuine, no label needed)
        for col in LABEL_CANDIDATES:
            train_df.drop(columns=[col], errors='ignore', inplace=True)
        train_df = train_df.copy()          # defragment before column insertion
        train_df['is_genuine'] = 1

        test_path = os.path.join(features_dir, f'{user_id}_testing_sessions.csv')
        if not os.path.exists(test_path):
            logging.warning(f"Testing file missing for {user_id}; skipping user.")
            continue

        test_df = read_csv_safe(test_path)

        # Pick the first label column that exists
        label_col = next((c for c in LABEL_CANDIDATES if c in test_df.columns), None)
        if label_col is None:
            logging.warning(f"No label column for {user_id}; skipping user.")
            continue

        # FIX #2: Drop ALL other label columns to prevent contamination
        for col in LABEL_CANDIDATES:
            if col != label_col and col in test_df.columns:
                test_df = test_df.drop(columns=[col])

        test_df = test_df.copy()            # defragment before column insertion
        test_df['is_genuine'] = test_df[label_col].apply(_norm_label)

        n_invalid = (test_df['is_genuine'] == -1).sum()
        if n_invalid > 0:
            logging.warning(f"{user_id}: dropping {n_invalid} sessions with invalid/NaN labels.")
            test_df = test_df[test_df['is_genuine'] != -1].copy()

        # FIX #3: If all rows were invalid, skip this user
        if len(test_df) == 0:
            logging.warning(f"{user_id}: all test sessions had invalid labels; skipping user.")
            continue

        test_df.drop(columns=[label_col], errors='ignore', inplace=True)
        raw[user_id] = {'train': train_df, 'test': test_df}

    if not raw:
        raise RuntimeError("No valid user data found.")
    logging.info(f"Loaded raw data for {len(raw)} users.")
    return raw


def _identify_feature_cols(raw: Dict[str, Dict[str, pd.DataFrame]],
                           config: dict) -> List[str]:
    """
    Feature columns = intersection of numeric columns across all users,
    minus ID columns.

    OBSERVATION #4: Using intersection is strict. Added logging for columns
    present in some but not all users (for diagnostic purposes).
    """
    id_cols = set(config['data']['id_columns']) | {'is_genuine', 'device_owner_id', 'user_id'}

    # Compute intersection and also collect presence counts for diagnostics
    common = None
    user_sets = {}
    for user_id, parts in raw.items():
        cols = set(parts['train'].columns) & set(parts['test'].columns)
        user_sets[user_id] = cols
        common = cols if common is None else (common & cols)

    # Log features that are not common (present only in some users)
    all_feats = set.union(*user_sets.values()) if user_sets else set()
    missing_in_some = all_feats - common
    if missing_in_some:
        logging.info(f"Features present in some but not all users: {len(missing_in_some)} "
                     f"(e.g. {list(missing_in_some)[:5]})")

    candidates = sorted(c for c in common if c not in id_cols)

    # FIX DL-F1: Check numeric-ness in BOTH train and test portions across all users
    numeric_feats = []
    for c in candidates:
        ok = True
        for parts in raw.values():
            s_train = pd.to_numeric(parts['train'][c], errors='coerce')
            s_test = pd.to_numeric(parts['test'][c], errors='coerce')
            # Reject if either portion is mostly non-numeric
            if s_train.notna().mean() < 0.5 or s_test.notna().mean() < 0.5:
                ok = False
                break
        if ok:
            numeric_feats.append(c)

    dropped = len(candidates) - len(numeric_feats)
    if dropped:
        logging.info(f"Dropped {dropped} non-numeric candidate columns.")
    logging.info(f"Initial feature count: {len(numeric_feats)}")
    return numeric_feats


def _stratified_test_split(test_df: pd.DataFrame,
                           dev_ratio: float,
                           rng: np.random.RandomState
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one user's testing sessions into dev and holdout,
    stratified by is_genuine, guaranteeing >=1 sample of each class in
    each partition whenever the class has >=2 samples.

    OBSERVATION #5: Single-sample classes are always placed in holdout.
    This is intentional: ensures holdout has at least one sample per class
    for evaluation, even if dev has none for that class.
    """
    dev_parts, hold_parts = [], []
    for cls in [1, 0]:
        cls_df = test_df[test_df['is_genuine'] == cls]
        n = len(cls_df)
        if n == 0:
            continue
        # .copy() makes index array writable for rng.shuffle()
        idx = cls_df.index.to_numpy().copy()
        rng.shuffle(idx)
        if n == 1:
            hold_parts.append(cls_df.loc[idx])
            continue
        n_dev = int(round(n * dev_ratio))
        n_dev = min(max(n_dev, 1), n - 1)   # Guarantee >=1 in each partition
        dev_parts.append(cls_df.loc[idx[:n_dev]])
        hold_parts.append(cls_df.loc[idx[n_dev:]])

    dev = pd.concat(dev_parts) if dev_parts else pd.DataFrame(columns=test_df.columns)
    hold = pd.concat(hold_parts) if hold_parts else pd.DataFrame(columns=test_df.columns)
    return dev, hold


def load_split_data(config: dict
                    ) -> Tuple[Dict[str, pd.DataFrame],
                               Dict[str, pd.DataFrame],
                               List[str]]:
    """
    Canonical, leakage-free entry point.
    """
    dev_ratio = config['split']['dev_ratio']
    random_state = config['split']['random_state']
    missing_thresh = config['cleaning']['missing_threshold']
    const_user_thresh = config['cleaning']['constant_user_threshold']

    raw = _load_raw_users(config)
    feature_cols = _identify_feature_cols(raw, config)

    # Guard: no features survive after initial intersection/numeric check
    if not feature_cols:
        raise RuntimeError(
            "No valid numeric feature columns remain after intersection/filtering."
        )

    rng = np.random.RandomState(random_state)

    # ---------- 1. Build dev / holdout per user ----------
    dev_raw, hold_raw, manifest_rows = {}, {}, []
    for user_id, parts in raw.items():
        train_df = parts['train']
        test_df = parts['test']

        test_dev, test_hold = _stratified_test_split(test_df, dev_ratio, rng)

        # FIX #6: Use ignore_index=True to avoid duplicate indices
        dev_df = pd.concat([train_df, test_dev], ignore_index=True)
        dev_raw[user_id] = dev_df
        hold_raw[user_id] = test_hold

        manifest_rows.append({
            'user': user_id,
            'dev_genuine': int((dev_df['is_genuine'] == 1).sum()),
            'dev_impostor': int((dev_df['is_genuine'] == 0).sum()),
            'hold_genuine': int((test_hold['is_genuine'] == 1).sum()),
            'hold_impostor': int((test_hold['is_genuine'] == 0).sum()),
        })

    # ---------- 2. Cleaning decisions on DEV only ----------
    all_dev = pd.concat(
        [df.assign(_uid=uid) for uid, df in dev_raw.items()], ignore_index=True
    )

    all_dev[feature_cols] = all_dev[feature_cols].apply(pd.to_numeric, errors='coerce')

    # 2a. Missing threshold (DEV only)
    # OBSERVATION #7: Global missing rate may hide per‑user issues.
    # We compute per‑user missing rates and log any feature that exceeds
    # threshold for any user, but we still drop based on global rate
    # to avoid over‑aggressive removal. (This is a trade‑off.)
    missing_ratio = all_dev[feature_cols].isnull().mean()
    drop_missing = missing_ratio[missing_ratio > missing_thresh].index.tolist()

    # Log per‑user worst missing rates for dropped features (diagnostic)
    if drop_missing:
        per_user_missing = all_dev.groupby('_uid')[drop_missing].isnull().mean().max(axis=0)
        for feat in drop_missing:
            logging.debug(f"Feature {feat} dropped (global missing: {missing_ratio[feat]:.2f}, "
                          f"worst per‑user: {per_user_missing[feat]:.2f})")

    feature_cols = [c for c in feature_cols if c not in drop_missing]
    logging.info(f"Dropped {len(drop_missing)} features with >{missing_thresh*100:.0f}% "
                 f"missing (dev only). Remaining: {len(feature_cols)}")

    # Guard: no features after missing filtering
    if not feature_cols:
        raise RuntimeError(
            "No features remain after missing-value filtering. "
            "Lower cleaning.missing_threshold or inspect the data."
        )

    # 2b. Per-user medians computed on DEV only
    dev_medians = all_dev.groupby('_uid')[feature_cols].median()

    # 2c. Constant-feature check on DEV only
    constant_counts = pd.Series(0, index=feature_cols, dtype=int)
    for uid, grp in all_dev.groupby('_uid'):
        stds = grp[feature_cols].std(ddof=0).fillna(0.0)
        constant_counts[stds[stds == 0].index] += 1

    n_users = len(dev_raw)
    drop_const = constant_counts[constant_counts / n_users > const_user_thresh].index.tolist()
    feature_cols = [c for c in feature_cols if c not in drop_const]
    logging.info(f"Dropped {len(drop_const)} near-constant features (dev only). "
                 f"Remaining: {len(feature_cols)}")

    # Guard: no features after constant filtering
    if not feature_cols:
        raise RuntimeError(
            "No features remain after constant-feature filtering. "
            "Lower cleaning.constant_user_threshold or inspect the data."
        )

    # ---------- 3. Apply imputation (dev medians) to BOTH partitions ----------
    def _finalize(df: pd.DataFrame, uid: str) -> pd.DataFrame:
        out = df[feature_cols + ['is_genuine']].copy()
        out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors='coerce')
        med = dev_medians.loc[uid, feature_cols]
        out[feature_cols] = out[feature_cols].fillna(med).fillna(0.0)
        return out.reset_index(drop=True)

    dev_data = {uid: _finalize(df, uid) for uid, df in dev_raw.items()}
    holdout_data = {}
    for uid, df in hold_raw.items():
        if len(df) == 0:
            logging.warning(f"User {uid} has empty holdout (no sessions assigned).")
        else:
            holdout_data[uid] = _finalize(df, uid)

    # ---------- 4. Save split manifest for reproducibility ----------
    out_dir = config['output'].get('dir', 'out')
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = config['output'].get('split_manifest',
                                         os.path.join(out_dir, 'split_manifest.csv'))
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    logging.info(f"Split manifest saved to {manifest_path}")

    logging.info(f"Final: {len(dev_data)} users, {len(feature_cols)} features. "
                 f"Dev sessions: {sum(len(d) for d in dev_data.values())}, "
                 f"Holdout sessions: {sum(len(d) for d in holdout_data.values())}.")
    return dev_data, holdout_data, feature_cols


def load_and_clean_data(config: dict) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Deprecated backward-compatible wrapper. Returns DEVELOPMENT data only.
    """
    logging.warning("load_and_clean_data() is deprecated; returning DEV data only. "
                    "Use load_split_data() for explicit dev/holdout access.")
    dev_data, _, feature_cols = load_split_data(config)
    return dev_data, feature_cols