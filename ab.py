import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import logging

logger = logging.getLogger(__name__)


class WithinDeviceDiscriminabilityFilter(BaseEstimator, TransformerMixin):
    """
    Tier 1: Hardware-disentanglement filter using within-device point-biserial correlation.

    For each feature and each device, compute the point-biserial correlation between the feature
    values and the binary label (genuine=1, impostor=0). Then aggregate across devices by taking
    the median absolute correlation. Features with median < threshold are dropped.

    Parameters
    ----------
    threshold : float, default=0.10
        Minimum median absolute point-biserial correlation required to keep a feature.

    device_id_col : str, default='device_id'
        Name of the column in X that contains device identifiers.

    is_genuine_col : str, default='is_genuine'
        Name of the column in X that indicates genuine (1) or impostor (0) session.

    Attributes
    ----------
    feature_mask_ : ndarray of shape (n_features,)
        Boolean mask of kept features.

    selected_features_ : list of str
        Names of the retained feature columns.

    n_features_in_ : int
        Number of features seen in `fit`.

    feature_names_in_ : ndarray of shape (n_features,)
        Names of all feature columns seen in `fit`.
    """

    def __init__(self, threshold=0.10, device_id_col='device_id', is_genuine_col='is_genuine'):
        self.threshold = threshold
        self.device_id_col = device_id_col
        self.is_genuine_col = is_genuine_col

    def fit(self, X, y=None):
        """
        Compute the within-device discriminability for each feature and determine which to keep.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing all feature columns plus metadata columns.
            Must contain columns `device_id_col` and `is_genuine_col`.

        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self
        """
        # Validate input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        required_cols = [self.device_id_col, self.is_genuine_col]
        missing = [c for c in required_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required metadata columns: {missing}")

        # Separate metadata and features
        metadata = X[required_cols]
        feature_df = X.drop(columns=required_cols)
        feature_names = feature_df.columns.tolist()
        self.feature_names_in_ = np.array(feature_names)
        self.n_features_in_ = len(feature_names)

        # Convert to numpy for speed
        feature_values = feature_df.values.astype(np.float64)
        device_ids = metadata[self.device_id_col].values
        is_genuine = metadata[self.is_genuine_col].values.astype(np.int8)

        # Get unique devices
        unique_devices = np.unique(device_ids)
        n_devices = len(unique_devices)
        n_features = feature_values.shape[1]

        # Store absolute point-biserial correlations per feature across devices
        abs_corrs = np.zeros((n_devices, n_features))

        for d_idx, dev in enumerate(unique_devices):
            mask = (device_ids == dev)
            dev_features = feature_values[mask]
            dev_labels = is_genuine[mask]

            # Separate genuine and impostor
            genuine_mask = (dev_labels == 1)
            impostor_mask = (dev_labels == 0)
            n1 = np.sum(genuine_mask)
            n0 = np.sum(impostor_mask)

            # If either group is empty, skip this device (log warning)
            if n1 == 0 or n0 == 0:
                logger.warning(f"Device {dev} has no genuine or no impostor sessions; skipping.")
                abs_corrs[d_idx, :] = np.nan
                continue

            # Compute means and pooled std for each feature
            M1 = np.mean(dev_features[genuine_mask], axis=0)
            M0 = np.mean(dev_features[impostor_mask], axis=0)
            # Pooled standard deviation (unbiased)
            var1 = np.var(dev_features[genuine_mask], axis=0, ddof=1)
            var0 = np.var(dev_features[impostor_mask], axis=0, ddof=1)
            # Avoid division by zero: if variance is zero, set pooled std to a small epsilon
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
            pooled_std = np.maximum(pooled_std, 1e-12)

            # Point-biserial correlation formula
            # r = (M1 - M0) / pooled_std * sqrt(n1*n0/(n*(n-1)))
            n = n1 + n0
            corr = (M1 - M0) / pooled_std * np.sqrt(n1 * n0 / (n * (n - 1)))
            abs_corrs[d_idx, :] = np.abs(corr)

        # Aggregate: median across devices (ignoring NaNs)
        median_corrs = np.nanmedian(abs_corrs, axis=0)

        # Apply threshold
        keep = median_corrs >= self.threshold
        self.feature_mask_ = keep
        self.selected_features_ = [feature_names[i] for i in range(n_features) if keep[i]]

        logger.info(f"Retained {sum(keep)} features out of {n_features} (threshold={self.threshold}).")
        return self

    def transform(self, X):
        """
        Apply the feature mask to X, returning only the selected features (and dropping metadata).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing feature columns. May also contain metadata columns,
            but they will be dropped.

        Returns
        -------
        X_selected : pd.DataFrame
            DataFrame containing only the kept feature columns.
        """
        check_is_fitted(self, 'feature_mask_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        # Drop metadata columns if present
        cols_to_drop = [self.device_id_col, self.is_genuine_col]
        X_features = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

        # Ensure we only keep features that were present during fit
        common_cols = [c for c in self.selected_features_ if c in X_features.columns]
        if len(common_cols) != len(self.selected_features_):
            missing = set(self.selected_features_) - set(X_features.columns)
            logger.warning(f"Some selected features are missing in transform: {missing}")
        return X_features[common_cols]

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str, optional
            Not used, present for pipeline compatibility.

        Returns
        -------
        feature_names_out : ndarray of str
            Names of selected features.
        """
        check_is_fitted(self, 'selected_features_')
        return np.array(self.selected_features_)


# ------------------ Unit Test ------------------
def test_within_device_filter():
    """Simple test using synthetic data."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    n_devices = 3

    # Create a DataFrame with metadata
    device_ids = np.repeat(np.arange(n_devices), n_samples // n_devices)
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    # Generate random features
    X_features = np.random.randn(n_samples, n_features)
    # Make some features discriminative: add device-specific offset to genuine sessions
    for d in range(n_devices):
        mask = (device_ids == d) & (is_genuine == 1)
        X_features[mask, 0] += 2.0  # feature 0 is strongly discriminative
        X_features[mask, 1] += 1.0  # feature 1 moderately discriminative
    # Feature 2 is noise

    # Create DataFrame
    df = pd.DataFrame(X_features, columns=[f'feat_{i}' for i in range(n_features)])
    df['device_id'] = device_ids
    df['is_genuine'] = is_genuine

    # Apply filter
    filter_ = WithinDeviceDiscriminabilityFilter(threshold=0.10)
    filter_.fit(df)
    selected = filter_.transform(df)

    # Expected: feature 0 and feature 1 retained (and maybe others by chance)
    assert 'feat_0' in selected.columns
    assert 'feat_1' in selected.columns
    # At least one of them should be kept
    assert selected.shape[1] >= 2
    print("Unit test passed.")

if __name__ == "__main__":
    test_within_device_filter()






























































































import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
import logging

logger = logging.getLogger(__name__)


class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Tier 2: Global stability selection using consensus of three ranking methods.

    - LOUO Random Forest: leave-one-user-out CV, average feature importances.
    - Fisher Score: (mean_genuine - mean_impostor)^2 / (var_genuine + var_impostor).
    - Boruta-like: random forest with shadow features, count wins over max shadow.

    The final consensus rank is the average rank across the three methods.
    The top `top_k` features are selected.

    Parameters
    ----------
    top_k : int, default=100
        Number of top features to retain.

    user_id_col : str, default='user_id'
        Column name in X identifying the user (genuine owner).

    is_genuine_col : str, default='is_genuine'
        Column name indicating genuine (1) or impostor (0) session.

    n_estimators : int, default=100
        Number of trees in the random forest (for LOUO RF and Boruta).

    max_depth : int or None, default=10
        Maximum depth for the random forest (to avoid overfitting).

    boruta_iterations : int, default=10
        Number of shadow‑feature iterations.

    random_state : int, default=42
        Seed for reproducibility.

    Attributes
    ----------
    selected_features_ : list of str
        Names of the retained feature columns.

    feature_mask_ : ndarray of shape (n_features,)
        Boolean mask of kept features.

    consensus_ranks_ : dict
        Final ranks and contributions for diagnostic purposes.
    """

    def __init__(
        self,
        top_k=100,
        user_id_col='user_id',
        is_genuine_col='is_genuine',
        n_estimators=100,
        max_depth=10,
        boruta_iterations=10,
        random_state=42,
    ):
        self.top_k = top_k
        self.user_id_col = user_id_col
        self.is_genuine_col = is_genuine_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.boruta_iterations = boruta_iterations
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Compute consensus ranking and select top_k features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing feature columns plus metadata
            (`user_id_col` and `is_genuine_col`).

        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required = [self.user_id_col, self.is_genuine_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        # Separate features and metadata
        feature_df = X.drop(columns=required)
        feature_names = feature_df.columns.tolist()
        self.feature_names_in_ = np.array(feature_names)
        self.n_features_in_ = len(feature_names)

        features = feature_df.values.astype(np.float64)
        user_ids = X[self.user_id_col].values
        is_genuine = X[self.is_genuine_col].values.astype(np.int8)

        # ---- 1. LOUO Random Forest Importance ----
        logger.info("Computing LOUO RF importance...")
        rf_importances = np.zeros(self.n_features_in_)
        unique_users = np.unique(user_ids)
        n_users = len(unique_users)

        for u in unique_users:
            # Leave one user out
            train_mask = (user_ids != u)
            test_mask = (user_ids == u)
            X_train = features[train_mask]
            y_train = is_genuine[train_mask]

            # Skip if any class missing in training
            if len(np.unique(y_train)) < 2:
                continue

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            rf_importances += rf.feature_importances_

        # Average over users
        rf_importances /= n_users
        rf_rank = np.argsort(np.argsort(-rf_importances))  # rank 0 = highest

        # ---- 2. Fisher Score ----
        logger.info("Computing Fisher Score...")
        fisher_scores = np.zeros(self.n_features_in_)
        genuine_mask = (is_genuine == 1)
        impostor_mask = (is_genuine == 0)

        if np.sum(genuine_mask) > 0 and np.sum(impostor_mask) > 0:
            mean_g = np.mean(features[genuine_mask], axis=0)
            mean_i = np.mean(features[impostor_mask], axis=0)
            var_g = np.var(features[genuine_mask], axis=0, ddof=1)
            var_i = np.var(features[impostor_mask], axis=0, ddof=1)
            denominator = var_g + var_i + 1e-12
            fisher_scores = (mean_g - mean_i) ** 2 / denominator
        else:
            logger.warning("Only one class present; Fisher score set to zero.")

        fisher_rank = np.argsort(np.argsort(-fisher_scores))

        # ---- 3. Boruta-like with Shadow Features ----
        logger.info("Computing Boruta-like importance...")
        n_iter = self.boruta_iterations
        win_counts = np.zeros(self.n_features_in_)

        for iter_idx in range(n_iter):
            # Create shadow features by shuffling each column
            shadow = np.random.permutation(features.T).T  # shuffle each column independently
            X_combined = np.hstack([features, shadow])
            n_real = self.n_features_in_

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + iter_idx,
                n_jobs=-1,
            )
            rf.fit(X_combined, is_genuine)
            importances = rf.feature_importances_
            real_imp = importances[:n_real]
            shadow_imp = importances[n_real:]
            max_shadow = np.max(shadow_imp)
            # Count if real feature importance > max_shadow
            wins = (real_imp > max_shadow).astype(int)
            win_counts += wins

        # Rank by win count descending (more wins = more important)
        boruta_rank = np.argsort(np.argsort(-win_counts))

        # ---- Consensus: average of ranks ----
        # Ranks are 0-based (0 = best). Average across methods.
        avg_rank = (rf_rank + fisher_rank + boruta_rank) / 3.0
        # Sort by avg_rank ascending
        sorted_indices = np.argsort(avg_rank)
        # Select top_k
        selected_indices = sorted_indices[:self.top_k]

        self.feature_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.feature_mask_[selected_indices] = True
        self.selected_features_ = [feature_names[i] for i in selected_indices]

        # Store diagnostic info
        self.consensus_ranks_ = {
            'rf_rank': rf_rank,
            'fisher_rank': fisher_rank,
            'boruta_rank': boruta_rank,
            'avg_rank': avg_rank,
            'selected_indices': selected_indices,
        }

        logger.info(f"Selected {len(self.selected_features_)} features from {self.n_features_in_}.")
        return self

    def transform(self, X):
        """
        Apply the feature mask, returning only the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame (may contain metadata; they will be dropped).

        Returns
        -------
        X_selected : pd.DataFrame
            DataFrame containing only the kept feature columns.
        """
        check_is_fitted(self, 'feature_mask_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        # Drop metadata columns if present
        cols_to_drop = [self.user_id_col, self.is_genuine_col]
        X_features = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

        # Keep only selected features
        common = [c for c in self.selected_features_ if c in X_features.columns]
        return X_features[common]

    def get_feature_names_out(self, input_features=None):
        """Return names of selected features."""
        check_is_fitted(self, 'selected_features_')
        return np.array(self.selected_features_)


# ------------------ Unit Test ------------------
def test_consensus_selector():
    """Synthetic test with discriminative features."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    n_users = 5

    # Create metadata
    user_ids = np.repeat(np.arange(n_users), n_samples // n_users)
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Features: first 3 are discriminative (vary with user and genuine)
    X = np.random.randn(n_samples, n_features)
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        X[mask, 0] += 2.0 * (u + 1)   # strong genuine signal
        X[mask, 1] += 1.5 * (u + 1)   # moderate
        X[mask, 2] += 1.0 * (u + 1)   # weaker
    # Features 3-9 are noise

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['user_id'] = user_ids
    df['is_genuine'] = is_genuine

    selector = ConsensusFeatureSelector(top_k=5)
    selector.fit(df)
    selected = selector.transform(df)

    # Expect that at least the three discriminative features are in top 5
    selected_cols = selected.columns.tolist()
    assert 'feat_0' in selected_cols
    assert 'feat_1' in selected_cols
    assert 'feat_2' in selected_cols
    assert len(selected_cols) <= 5
    print("Unit test passed.")


if __name__ == "__main__":
    test_consensus_selector()
























































import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
import logging

logger = logging.getLogger(__name__)


class NonLinearRedundancyReducer(BaseEstimator, TransformerMixin):
    """
    Tier 3: Non‑linear redundancy reduction via Spearman‑based hierarchical clustering.

    Features are clustered based on absolute Spearman correlation. For each cluster,
    the feature with the highest LOUO Random Forest importance (genuine vs impostor)
    is selected as the representative.

    Parameters
    ----------
    correlation_threshold : float, default=0.9
        Minimum absolute Spearman correlation to merge features into the same cluster.
        Clusters are formed by cutting the dendrogram at dissimilarity = 1 - threshold.

    max_features : int, default=80
        Maximum number of features to retain. If after clustering we have more clusters
        than max_features, we keep only the top `max_features` clusters by their
        best feature's importance.

    user_id_col : str, default='user_id'
        Column name in X identifying the user (genuine owner).

    is_genuine_col : str, default='is_genuine'
        Column name indicating genuine (1) or impostor (0) session.

    n_estimators : int, default=100
        Number of trees in the random forest for LOUO importance.

    max_depth : int or None, default=10
        Maximum depth for the random forest.

    random_state : int, default=42
        Seed for reproducibility.

    Attributes
    ----------
    selected_features_ : list of str
        Names of the retained features.

    feature_mask_ : ndarray of shape (n_features,)
        Boolean mask of kept features.

    cluster_representatives_ : dict
        Mapping from cluster ID to selected feature name.
    """

    def __init__(
        self,
        correlation_threshold=0.9,
        max_features=80,
        user_id_col='user_id',
        is_genuine_col='is_genuine',
        n_estimators=100,
        max_depth=10,
        random_state=42,
    ):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.user_id_col = user_id_col
        self.is_genuine_col = is_genuine_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Compute clusters and select representative features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing feature columns plus metadata
            (`user_id_col` and `is_genuine_col`).

        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required = [self.user_id_col, self.is_genuine_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        feature_df = X.drop(columns=required)
        feature_names = feature_df.columns.tolist()
        self.feature_names_in_ = np.array(feature_names)
        self.n_features_in_ = len(feature_names)

        features = feature_df.values.astype(np.float64)
        user_ids = X[self.user_id_col].values
        is_genuine = X[self.is_genuine_col].values.astype(np.int8)

        # ---- 1. Spearman correlation and hierarchical clustering ----
        logger.info("Computing Spearman correlation...")
        corr_matrix, _ = spearmanr(features, axis=0)  # shape (n_features, n_features)
        # Handle potential NaNs (if a feature is constant, correlation will be nan)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Dissimilarity: 1 - |correlation|
        dissim = 1 - np.abs(corr_matrix)
        # Ensure symmetry and zero diagonal
        np.fill_diagonal(dissim, 0)
        # Convert to condensed distance matrix (required by linkage)
        condensed = dissim[np.triu_indices_from(dissim, k=1)]
        # Perform hierarchical clustering with average linkage
        linkage_matrix = linkage(condensed, method='average')

        # Cut dendrogram at threshold = 1 - correlation_threshold
        cut_threshold = 1 - self.correlation_threshold
        cluster_labels = fcluster(linkage_matrix, t=cut_threshold, criterion='distance')
        n_clusters = cluster_labels.max()
        logger.info(f"Formed {n_clusters} clusters with threshold {self.correlation_threshold}.")

        # ---- 2. Compute LOUO Random Forest importance for all features ----
        logger.info("Computing LOUO RF importance (for representative selection)...")
        # We compute the same importance as in Tier 2, but here it's used only for intra-cluster selection.
        importances = np.zeros(self.n_features_in_)
        unique_users = np.unique(user_ids)
        n_users = len(unique_users)

        for u in unique_users:
            train_mask = (user_ids != u)
            X_train = features[train_mask]
            y_train = is_genuine[train_mask]
            if len(np.unique(y_train)) < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            importances += rf.feature_importances_
        importances /= n_users

        # ---- 3. For each cluster, select feature with highest importance ----
        cluster_representatives = {}
        for cluster_id in range(1, n_clusters + 1):
            members = np.where(cluster_labels == cluster_id)[0]
            if len(members) == 0:
                continue
            # Find index of member with highest importance
            best_idx = members[np.argmax(importances[members])]
            cluster_representatives[cluster_id] = best_idx

        # Convert to list of indices
        selected_indices = list(cluster_representatives.values())

        # If we have more than max_features, keep only the top max_features clusters
        # by the importance of their representative.
        if len(selected_indices) > self.max_features:
            # Sort clusters by their representative's importance descending
            sorted_by_imp = sorted(
                cluster_representatives.items(),
                key=lambda item: importances[item[1]],
                reverse=True
            )
            # Keep only the top max_features clusters
            keep_clusters = sorted_by_imp[:self.max_features]
            selected_indices = [idx for _, idx in keep_clusters]
            logger.info(f"Limited to {len(selected_indices)} clusters (max_features={self.max_features}).")

        # Build mask and store
        self.feature_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.feature_mask_[selected_indices] = True
        self.selected_features_ = [feature_names[i] for i in selected_indices]
        self.cluster_representatives_ = {
            f"cluster_{cid}": feature_names[idx]
            for cid, idx in cluster_representatives.items() if idx in selected_indices
        }

        logger.info(f"Selected {len(self.selected_features_)} features out of {self.n_features_in_}.")
        return self

    def transform(self, X):
        """Return only the selected features."""
        check_is_fitted(self, 'feature_mask_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        cols_to_drop = [self.user_id_col, self.is_genuine_col]
        X_features = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
        common = [c for c in self.selected_features_ if c in X_features.columns]
        return X_features[common]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_features_')
        return np.array(self.selected_features_)


# ------------------ Unit Test ------------------
def test_redundancy_reducer():
    """Test that the reducer picks features from clusters."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    n_users = 5

    # Create features: first 3 are correlated (clusters), next 2 are correlated, rest independent.
    X = np.random.randn(n_samples, n_features)
    # Cluster 1: features 0,1,2 highly correlated
    base1 = np.random.randn(n_samples) * 2
    X[:, 0] = base1 + 0.1 * np.random.randn(n_samples)
    X[:, 1] = base1 * 0.8 + 0.2 * np.random.randn(n_samples)
    X[:, 2] = base1 * 0.9 + 0.1 * np.random.randn(n_samples)
    # Cluster 2: features 3,4
    base2 = np.random.randn(n_samples) * 2
    X[:, 3] = base2 + 0.2 * np.random.randn(n_samples)
    X[:, 4] = base2 * 0.7 + 0.3 * np.random.randn(n_samples)
    # Features 5-9 are independent noise

    # Add discriminability to some features: make genuine sessions have higher values for feature 0 and 3
    user_ids = np.repeat(np.arange(n_users), n_samples // n_users)
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        X[mask, 0] += 3.0  # feature 0 is discriminative
        X[mask, 3] += 2.0  # feature 3 is discriminative

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['user_id'] = user_ids
    df['is_genuine'] = is_genuine

    reducer = NonLinearRedundancyReducer(correlation_threshold=0.9, max_features=5)
    reducer.fit(df)
    selected = reducer.transform(df)
    # We expect at least one from cluster 1 (likely feat_0 or feat_2) and one from cluster 2 (feat_3 or feat_4)
    selected_cols = selected.columns.tolist()
    # Since feature 0 is the most discriminative in cluster 1, it should be selected.
    # Feature 3 is discriminative in cluster 2.
    assert 'feat_0' in selected_cols
    assert 'feat_3' in selected_cols
    # Should not select all correlated features
    assert len(selected_cols) <= 5
    print("Unit test passed.")


if __name__ == "__main__":
    test_redundancy_reducer()






































import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
import logging
import warnings

logger = logging.getLogger(__name__)


class OCSVMWrapperOptimizer(BaseEstimator, TransformerMixin):
    """
    Tier 4: Backward elimination guided by OCSVM performance.

    Evaluates feature subsets using a per‑user OCSVM (nu=0.1, gamma='scale')
    and a user‑specific threshold (5th percentile of training decision scores).
    The score = mean(TAR) - 0.5 * std(TAR) over users, where TAR is obtained
    via 5‑fold cross‑validation on each user's genuine sessions.

    Parameters
    ----------
    cv_folds : int, default=5
        Number of folds for each user's genuine session split.

    lambda_variance : float, default=0.5
        Penalty weight for standard deviation of TAR across users.

    min_features : int, default=10
        Minimum number of features to keep (stop elimination if reached).

    n_jobs : int, default=-1
        Number of parallel jobs for OCSVM training/evaluation.

    random_state : int, default=42
        Seed for reproducibility in fold splitting.

    user_id_col : str, default='user_id'
        Column name in X identifying the user (genuine owner / attacked user).

    is_genuine_col : str, default='is_genuine'
        Column name indicating genuine (1) or impostor (0) session.

    Attributes
    ----------
    selected_features_ : list of str
        Names of the retained features.

    feature_mask_ : ndarray of shape (n_features,)
        Boolean mask of kept features.

    final_score_ : float
        The wrapper score of the final feature set.
    """

    def __init__(
        self,
        cv_folds=5,
        lambda_variance=0.5,
        min_features=10,
        n_jobs=-1,
        random_state=42,
        user_id_col='user_id',
        is_genuine_col='is_genuine',
    ):
        self.cv_folds = cv_folds
        self.lambda_variance = lambda_variance
        self.min_features = min_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.user_id_col = user_id_col
        self.is_genuine_col = is_genuine_col

    def fit(self, X, y=None):
        """
        Perform backward elimination to select a feature subset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame containing feature columns plus metadata
            (`user_id_col` and `is_genuine_col`). It is assumed that the
            feature columns have already been reduced by previous tiers.

        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required = [self.user_id_col, self.is_genuine_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        # Separate features and metadata
        feature_df = X.drop(columns=required)
        feature_names = feature_df.columns.tolist()
        self.feature_names_in_ = np.array(feature_names)
        self.n_features_in_ = len(feature_names)

        # Convert to numpy for speed
        features = feature_df.values.astype(np.float64)
        user_ids = X[self.user_id_col].values
        is_genuine = X[self.is_genuine_col].values.astype(np.int8)

        # Get unique users
        unique_users = np.unique(user_ids)
        n_users = len(unique_users)

        # Build per‑user indices for genuine and impostor sessions
        user_genuine_indices = {}
        user_impostor_indices = {}
        for u in unique_users:
            mask_user = (user_ids == u)
            mask_gen = mask_user & (is_genuine == 1)
            mask_imp = mask_user & (is_genuine == 0)
            user_genuine_indices[u] = np.where(mask_gen)[0]
            user_impostor_indices[u] = np.where(mask_imp)[0]

            # Ensure each user has at least cv_folds genuine sessions
            if len(user_genuine_indices[u]) < self.cv_folds:
                warnings.warn(
                    f"User {u} has only {len(user_genuine_indices[u])} genuine sessions; "
                    f"needs at least {self.cv_folds} for cross‑validation. Skipping this user."
                )
                # We'll still include them? We'll handle by returning nan TAR.

        # Define internal evaluation function for a given feature mask (boolean array)
        def evaluate_subset(mask):
            """
            Compute the wrapper score for a feature subset defined by a boolean mask.
            Returns (score, TAR_per_user) where TAR_per_user is a dict for diagnostics.
            """
            # Select only the features specified by mask
            X_sub = features[:, mask]
            # Ensure we have at least one feature
            if X_sub.shape[1] == 0:
                return -np.inf, {}

            # Prepare results per user
            tar_values = []
            for u in unique_users:
                gen_indices = user_genuine_indices.get(u, [])
                imp_indices = user_impostor_indices.get(u, [])
                if len(gen_indices) < self.cv_folds:
                    # Not enough genuine sessions for CV – skip this user
                    continue

                # If there are no impostor sessions, we still need to train but TAR only
                # Use 5‑fold CV on genuine sessions
                skf = StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
                # Need labels for stratification – we'll use binary labels (all genuine, but we need a dummy)
                # We'll create dummy labels (all 1) but StratifiedKFold requires at least two classes,
                # so we'll use a simple KFold instead.
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                fold_tars = []

                for train_idx, test_idx in kf.split(gen_indices):
                    train_global = gen_indices[train_idx]
                    test_global = gen_indices[test_idx]

                    # Train OCSVM on training genuine sessions
                    X_train = X_sub[train_global]
                    ocsvm = OneClassSVM(nu=0.1, gamma='scale')
                    ocsvm.fit(X_train)

                    # Compute decision scores on training set to set threshold (5th percentile)
                    train_scores = ocsvm.decision_function(X_train)
                    threshold = np.percentile(train_scores, 5)

                    # Evaluate on held‑out genuine sessions
                    test_scores = ocsvm.decision_function(X_sub[test_global])
                    preds = (test_scores >= threshold).astype(int)
                    # TAR = TP / (TP+FN) = fraction of genuine accepted
                    tar = np.mean(preds) if len(preds) > 0 else 0.0
                    fold_tars.append(tar)

                # Average TAR over folds
                if fold_tars:
                    user_tar = np.mean(fold_tars)
                else:
                    user_tar = 0.0
                tar_values.append(user_tar)

            if not tar_values:
                return -np.inf, {}

            mean_tar = np.mean(tar_values)
            std_tar = np.std(tar_values, ddof=1) if len(tar_values) > 1 else 0.0
            score = mean_tar - self.lambda_variance * std_tar
            return score, dict(zip(unique_users, tar_values))

        # ---- Backward elimination ----
        current_mask = np.ones(self.n_features_in_, dtype=bool)  # start with all features
        current_score, _ = evaluate_subset(current_mask)
        logger.info(f"Initial score (all features): {current_score:.4f}")

        # Keep track of removed features to avoid re-evaluation
        n_features_current = np.sum(current_mask)
        iteration = 0
        improvement_made = True

        while improvement_made and n_features_current > self.min_features:
            improvement_made = False
            # Evaluate removing each feature (that is still present)
            best_removal_score = -np.inf
            best_removal_idx = -1

            # Parallel evaluation of candidates
            def eval_removal(idx):
                if not current_mask[idx]:
                    return idx, -np.inf
                new_mask = current_mask.copy()
                new_mask[idx] = False
                score, _ = evaluate_subset(new_mask)
                return idx, score

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(eval_removal)(i) for i in range(self.n_features_in_)
            )

            for idx, score in results:
                if score >= best_removal_score:
                    best_removal_score = score
                    best_removal_idx = idx

            # If the best removal improves (or does not decrease) the score, apply it
            if best_removal_score >= current_score - 1e-9:
                current_mask[best_removal_idx] = False
                current_score = best_removal_score
                n_features_current = np.sum(current_mask)
                improvement_made = True
                logger.info(
                    f"Iter {iteration}: removed feature {feature_names[best_removal_idx]}, "
                    f"score = {current_score:.4f}, features left = {n_features_current}"
                )
            else:
                logger.info(f"Iter {iteration}: no improvement, stopping.")
                break

            iteration += 1

        # Finalise
        self.feature_mask_ = current_mask
        self.selected_features_ = [feature_names[i] for i in range(self.n_features_in_) if current_mask[i]]
        self.final_score_ = current_score
        logger.info(f"Final selected {len(self.selected_features_)} features, score = {current_score:.4f}")
        return self

    def transform(self, X):
        """Return only the selected features."""
        check_is_fitted(self, 'feature_mask_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        cols_to_drop = [self.user_id_col, self.is_genuine_col]
        X_features = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
        common = [c for c in self.selected_features_ if c in X_features.columns]
        return X_features[common]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_features_')
        return np.array(self.selected_features_)


# ------------------ Unit Test ------------------
def test_wrapper_optimizer():
    """Simple test: verify that the wrapper can select at least one feature."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    n_users = 5

    # Generate data: features 0-2 are discriminative for each user
    X = np.random.randn(n_samples, n_features)
    user_ids = np.repeat(np.arange(n_users), n_samples // n_users)
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Make feature 0 and 1 predictive of genuine for each user
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        X[mask, 0] += 3.0 * (u + 1)
        X[mask, 1] += 2.0 * (u + 1)
    # Feature 2 is slightly predictive
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        X[mask, 2] += 1.0 * (u + 1)

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['user_id'] = user_ids
    df['is_genuine'] = is_genuine

    optimizer = OCSVMWrapperOptimizer(cv_folds=3, min_features=2, n_jobs=1)
    optimizer.fit(df)
    selected = optimizer.transform(df)
    # Should have kept at least features 0 and 1 (or some discriminative ones)
    assert len(selected.columns) >= 2
    print("Unit test passed.")


if __name__ == "__main__":
    test_wrapper_optimizer()
























































import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import logging
import warnings

logger = logging.getLogger(__name__)


class LOGOValidator:
    """
    Leave‑One‑Group‑Out validation for the full feature selection pipeline.

    Parameters
    ----------
    pipeline : Pipeline or list of transformers
        The complete feature selection pipeline (must include an OCSVMWrapperOptimizer
        as the last step, which stores `mean_tar_` after fitting).

    user_id_col : str, default='user_id'
        Column name in X identifying the genuine user.

    device_id_col : str, default='device_id'
        Column name in X identifying the device (hardware).

    is_genuine_col : str, default='is_genuine'
        Column name indicating genuine (1) or impostor (0) session.

    nu : float, default=0.1
        OCSVM nu parameter (fixed for validation).

    gamma : str or float, default='scale'
        OCSVM gamma parameter.

    n_jobs : int, default=-1
        Number of parallel jobs for fitting/evaluating.

    random_state : int, default=42
        Seed for reproducibility.

    Attributes
    ----------
    device_results_ : dict
        Results from device‑level LOGO.

    user_results_ : dict
        Results from user‑level LOGO.
    """

    def __init__(
        self,
        pipeline,
        user_id_col='user_id',
        device_id_col='device_id',
        is_genuine_col='is_genuine',
        nu=0.1,
        gamma='scale',
        n_jobs=-1,
        random_state=42,
    ):
        self.pipeline = pipeline
        self.user_id_col = user_id_col
        self.device_id_col = device_id_col
        self.is_genuine_col = is_genuine_col
        self.nu = nu
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _evaluate_held_out(self, train_df, valid_df):
        """
        Internal: Fit pipeline on train_df, transform valid_df, then evaluate OCSVM performance
        for each user in valid_df (typically one user per held‑out device).

        Returns a dict with:
            - tar: TAR at FAR=0.001 for the user(s) in valid_df
            - far: achieved FAR
            - n_genuine, n_impostor: counts
            - threshold_used: the decision threshold
        """
        # 1. Fit the pipeline on training data
        # The pipeline should contain all transformers including the wrapper.
        # We need to pass metadata columns; they are part of X.
        self.pipeline.fit(train_df)

        # 2. Retrieve the wrapper's mean TAR from internal cross‑validation
        # Assuming the last step is the wrapper; find it by type.
        wrapper = None
        if isinstance(self.pipeline, Pipeline):
            for step_name, step in self.pipeline.steps:
                if hasattr(step, 'mean_tar_') and hasattr(step, 'selected_features_'):
                    wrapper = step
                    break
        else:
            # If pipeline is a list, take the last
            for step in reversed(self.pipeline):
                if hasattr(step, 'mean_tar_') and hasattr(step, 'selected_features_'):
                    wrapper = step
                    break
        if wrapper is None:
            raise ValueError("Pipeline does not contain a transformer with 'mean_tar_' attribute. "
                             "Ensure OCSVMWrapperOptimizer is included and stores mean_tar_ after fit.")

        train_mean_tar = wrapper.mean_tar_

        # 3. Transform the validation set (keep only selected features)
        valid_selected = self.pipeline.transform(valid_df)

        # 4. For each user in valid_df, train OCSVM on their genuine sessions
        # In LOGO, valid_df contains sessions from the held‑out device (device-level)
        # or held‑out user (user-level). In both cases, there is exactly one genuine user
        # per device, but there might be multiple users if we ever have multi-user per device.
        # We'll handle multiple users by iterating.

        # Separate metadata from features in valid_selected
        # valid_selected is a DataFrame with only feature columns (no metadata)
        # We need the user_id and is_genuine from the original valid_df.
        user_ids = valid_df[self.user_id_col].values
        is_genuine = valid_df[self.is_genuine_col].values.astype(np.int8)

        unique_users = np.unique(user_ids)
        results = []

        for u in unique_users:
            mask_user = (user_ids == u)
            indices = np.where(mask_user)[0]
            # Extract features for this user
            X_user = valid_selected.iloc[indices].values.astype(np.float64)
            y_genuine = is_genuine[indices]

            genuine_mask = (y_genuine == 1)
            impostor_mask = (y_genuine == 0)
            X_gen = X_user[genuine_mask]
            X_imp = X_user[impostor_mask]

            if len(X_gen) == 0 or len(X_imp) == 0:
                logger.warning(f"User {u} has no genuine or no impostor sessions; skipping.")
                continue

            # Train OCSVM on genuine sessions
            ocsvm = OneClassSVM(nu=self.nu, gamma=self.gamma)
            ocsvm.fit(X_gen)

            # Compute scores on impostor sessions to set threshold for FAR=0.001
            imp_scores = ocsvm.decision_function(X_imp)
            # Sort impostor scores descending (higher score = more likely genuine)
            imp_scores_sorted = np.sort(imp_scores)[::-1]  # descending

            # Determine the number of allowed false positives: floor(0.001 * N_imp)
            n_imp = len(imp_scores_sorted)
            max_fp = int(np.floor(0.001 * n_imp))
            # Ensure at least 1? We'll take the threshold that gives the closest to 0.001.
            # We want threshold such that FAR <= 0.001, i.e., at most max_fp false accepts.
            # So we choose threshold = imp_scores_sorted[max_fp] if max_fp < n_imp,
            # else the lowest score (allow all impostors? but that would be FAR >0.001? Actually if max_fp >= n_imp, we can set threshold = -inf to accept all, but that's unrealistic.
            # Instead, we set threshold = imp_scores_sorted[n_imp-1] (lowest) but then FAR = 1? We'll handle carefully.

            if max_fp < n_imp:
                threshold = imp_scores_sorted[max_fp]  # this allows exactly max_fp false positives
            else:
                # If 0.001*n_imp < 1, we can't allow any false positive, so set threshold above the highest impostor score
                # to achieve FAR=0.
                threshold = imp_scores_sorted[0] + 1e-6  # slightly above max impostor score

            # Compute achieved FAR
            achieved_far = np.mean(imp_scores >= threshold) if len(imp_scores) > 0 else 0.0

            # Compute TAR on genuine sessions
            gen_scores = ocsvm.decision_function(X_gen)
            tar = np.mean(gen_scores >= threshold) if len(gen_scores) > 0 else 0.0

            results.append({
                'user': u,
                'tar': tar,
                'far': achieved_far,
                'threshold': threshold,
                'n_genuine': len(X_gen),
                'n_impostor': len(X_imp),
                'train_mean_tar': train_mean_tar,
            })

        # Aggregate across users in this fold
        if not results:
            return {'tar': np.nan, 'far': np.nan, 'train_mean_tar': train_mean_tar, 'n_users': 0}

        mean_tar = np.mean([r['tar'] for r in results])
        mean_far = np.mean([r['far'] for r in results])
        # GDP = (train_mean_tar - mean_tar) / train_mean_tar
        gdp = (train_mean_tar - mean_tar) / train_mean_tar if train_mean_tar > 0 else np.nan

        return {
            'tar': mean_tar,
            'far': mean_far,
            'train_mean_tar': train_mean_tar,
            'gdp': gdp,
            'per_user': results,
            'n_users': len(results),
        }

    def validate_device_logo(self, X):
        """
        Device‑level LOGO: hold out each device, train on others, evaluate on held‑out device.

        Parameters
        ----------
        X : pd.DataFrame
            Full dataset with all features and metadata columns.

        Returns
        -------
        dict with keys:
            - 'average_tar': mean TAR across all devices
            - 'average_far': mean FAR (should be close to 0.001)
            - 'average_gdp': mean GDP across devices
            - 'per_device': list of results for each device
            - 'pass': bool, GDP <= 0.15 for all devices?
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required = [self.device_id_col, self.user_id_col, self.is_genuine_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        device_ids = X[self.device_id_col].values
        unique_devices = np.unique(device_ids)

        def evaluate_one_device(dev):
            train_mask = (device_ids != dev)
            valid_mask = (device_ids == dev)
            train_df = X[train_mask].copy()
            valid_df = X[valid_mask].copy()
            # Ensure we don't include devices not in training (they are dropped)
            return self._evaluate_held_out(train_df, valid_df)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_one_device)(dev) for dev in unique_devices
        )

        # Aggregate
        tars = [r['tar'] for r in results if not np.isnan(r['tar'])]
        fares = [r['far'] for r in results if not np.isnan(r['far'])]
        gdps = [r['gdp'] for r in results if not np.isnan(r['gdp'])]

        avg_tar = np.mean(tars) if tars else np.nan
        avg_far = np.mean(fares) if fares else np.nan
        avg_gdp = np.mean(gdps) if gdps else np.nan
        # Pass if all GDP <= 0.15? Or average? Spec says "If GDP > 0.15 → FAIL"
        # We'll check per device.
        all_pass = all(g <= 0.15 for g in gdps if not np.isnan(g))

        self.device_results_ = {
            'average_tar': avg_tar,
            'average_far': avg_far,
            'average_gdp': avg_gdp,
            'per_device': results,
            'pass': all_pass,
        }
        return self.device_results_

    def validate_user_logo(self, X):
        """
        User‑level LOGO (LOUO): hold out each user, train on others, evaluate on held‑out user.

        Returns similar dict.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required = [self.user_id_col, self.is_genuine_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        user_ids = X[self.user_id_col].values
        unique_users = np.unique(user_ids)

        def evaluate_one_user(u):
            train_mask = (user_ids != u)
            valid_mask = (user_ids == u)
            train_df = X[train_mask].copy()
            valid_df = X[valid_mask].copy()
            return self._evaluate_held_out(train_df, valid_df)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_one_user)(u) for u in unique_users
        )

        tars = [r['tar'] for r in results if not np.isnan(r['tar'])]
        fares = [r['far'] for r in results if not np.isnan(r['far'])]
        gdps = [r['gdp'] for r in results if not np.isnan(r['gdp'])]

        avg_tar = np.mean(tars) if tars else np.nan
        avg_far = np.mean(fares) if fares else np.nan
        avg_gdp = np.mean(gdps) if gdps else np.nan
        # Pass if TAR >= 0.70 at FAR=0.001? We'll check per user.
        all_pass = all(tar >= 0.70 for tar in tars if not np.isnan(tar))

        self.user_results_ = {
            'average_tar': avg_tar,
            'average_far': avg_far,
            'average_gdp': avg_gdp,
            'per_user': results,
            'pass': all_pass,
        }
        return self.user_results_


# ------------------ Unit Test ------------------
def test_logo_validator():
    """Test LOGOValidator with synthetic data."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create synthetic data
    np.random.seed(42)
    n_samples = 400
    n_features = 20
    n_devices = 5
    n_users = n_devices  # one user per device

    # Generate features: some discriminative per user
    X = np.random.randn(n_samples, n_features)
    device_ids = np.repeat(np.arange(n_devices), n_samples // n_devices)
    user_ids = device_ids  # one user per device
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Make features 0-4 discriminative for each user's genuine sessions
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        X[mask, :5] += 2.0 * (u + 1)

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['user_id'] = user_ids
    df['device_id'] = device_ids
    df['is_genuine'] = is_genuine

    # Build a simple pipeline with our transformers
    # (we need to import all previous modules; for test we can mock or use actual)
    # We'll create a minimal pipeline: just the wrapper? But we need all tiers for full test.
    # For simplicity, we'll just test that the validator runs without errors.
    # In a real scenario, we'd build the full pipeline.
    # We'll create a dummy pipeline that just selects first 5 features.
    class DummySelector(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.selected_features_ = X.columns[:5].tolist()
            self.mean_tar_ = 0.85  # dummy
            return self
        def transform(self, X):
            return X[self.selected_features_]
        def get_feature_names_out(self, input_features=None):
            return np.array(self.selected_features_)

    dummy_pipeline = Pipeline([
        ('selector', DummySelector())
    ])

    validator = LOGOValidator(dummy_pipeline, n_jobs=1)
    device_res = validator.validate_device_logo(df)
    user_res = validator.validate_user_logo(df)

    print("Device LOGO: TAR =", device_res['average_tar'])
    print("User LOGO: TAR =", user_res['average_tar'])
    assert not np.isnan(device_res['average_tar'])
    assert not np.isnan(user_res['average_tar'])
    print("Unit test passed.")


if __name__ == "__main__":
    test_logo_validator()






























































import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import logging

# Import the modules we built earlier
# (In practice, these would be imported from separate files)
# For clarity, I'm assuming they are already defined in the namespace.

logger = logging.getLogger(__name__)


# ----- Update OCSVMWrapperOptimizer to store mean_tar_ -----
class OCSVMWrapperOptimizer(BaseEstimator, TransformerMixin):
    """
    Tier 4: Backward elimination guided by OCSVM performance.

    (Extended to store mean_tar_ after fitting, as required by LOGOValidator.)
    """

    def __init__(
        self,
        cv_folds=5,
        lambda_variance=0.5,
        min_features=10,
        n_jobs=-1,
        random_state=42,
        user_id_col='user_id',
        is_genuine_col='is_genuine',
    ):
        self.cv_folds = cv_folds
        self.lambda_variance = lambda_variance
        self.min_features = min_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.user_id_col = user_id_col
        self.is_genuine_col = is_genuine_col

    def fit(self, X, y=None):
        # ... (the full fit code from Module 4) ...
        # We'll add:
        # After final selection, compute mean_tar_ from the final subset.
        # We'll use the evaluate_subset function (defined inside) on the final mask.
        # We already have current_mask and we computed the final score.
        # We need to compute the mean TAR from the final subset evaluation.
        # We'll modify the evaluate_subset to also return the per-user TAR list.

        # ... (insert full fit code from Module 4 here, with the following additions) ...
        # At the end, after we have self.feature_mask_, we compute:
        _, tar_list = evaluate_subset(self.feature_mask_)  # tar_list is dict user->TAR
        self.mean_tar_ = np.mean(list(tar_list.values())) if tar_list else 0.0
        self.std_tar_ = np.std(list(tar_list.values())) if tar_list else 0.0
        logger.info(f"Stored mean_tar_ = {self.mean_tar_:.4f}, std_tar_ = {self.std_tar_:.4f}")
        return self

    # ... (rest of methods unchanged) ...


# ----- Full Pipeline Builder -----
def build_full_pipeline(
    tier1_threshold=0.10,
    tier2_top_k=100,
    tier3_correlation_threshold=0.9,
    tier3_max_features=80,
    tier4_cv_folds=5,
    tier4_lambda_variance=0.5,
    tier4_min_features=10,
    tier4_n_jobs=-1,
    random_state=42,
    user_id_col='user_id',
    device_id_col='device_id',
    is_genuine_col='is_genuine',
):
    """
    Construct the complete 4‑Tier feature selection pipeline.

    Returns
    -------
    Pipeline
        A Scikit‑Learn Pipeline with the four transformers.
    """
    steps = [
        ('tier1', WithinDeviceDiscriminabilityFilter(
            threshold=tier1_threshold,
            device_id_col=device_id_col,
            is_genuine_col=is_genuine_col,
        )),
        ('tier2', ConsensusFeatureSelector(
            top_k=tier2_top_k,
            user_id_col=user_id_col,
            is_genuine_col=is_genuine_col,
            random_state=random_state,
        )),
        ('tier3', NonLinearRedundancyReducer(
            correlation_threshold=tier3_correlation_threshold,
            max_features=tier3_max_features,
            user_id_col=user_id_col,
            is_genuine_col=is_genuine_col,
            random_state=random_state,
        )),
        ('tier4', OCSVMWrapperOptimizer(
            cv_folds=tier4_cv_folds,
            lambda_variance=tier4_lambda_variance,
            min_features=tier4_min_features,
            n_jobs=tier4_n_jobs,
            random_state=random_state,
            user_id_col=user_id_col,
            is_genuine_col=is_genuine_col,
        )),
    ]
    return Pipeline(steps)


# ----- LOGO Validator (same as Module 5, but we include it here for completeness) -----
# The LOGOValidator class is as defined in Module 5. We'll just import it if separate.
# To keep this self-contained, I'll paste the LOGOValidator code here (it's unchanged).

# (Insert LOGOValidator class from Module 5 here)

# ----- Convenience function to run LOGO validation -----
def run_logo_validation(pipeline, X, n_jobs=-1, user_id_col='user_id',
                        device_id_col='device_id', is_genuine_col='is_genuine'):
    """
    Run both device‑level and user‑level LOGO validation on the fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        The full feature selection pipeline (must contain OCSVMWrapperOptimizer).
    X : pd.DataFrame
        Full dataset with metadata columns.
    n_jobs : int
        Number of parallel jobs for validation.
    user_id_col, device_id_col, is_genuine_col : str
        Column names for metadata.

    Returns
    -------
    device_results, user_results : dict
        Results from LOGOValidator.
    """
    validator = LOGOValidator(
        pipeline=pipeline,
        user_id_col=user_id_col,
        device_id_col=device_id_col,
        is_genuine_col=is_genuine_col,
        n_jobs=n_jobs,
    )
    device_results = validator.validate_device_logo(X)
    user_results = validator.validate_user_logo(X)
    return device_results, user_results


# ----- Integration Test -----
def test_full_pipeline():
    """Test the complete pipeline on synthetic data and run LOGO validation."""
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 30
    n_devices = 5
    n_users = n_devices  # one user per device

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    # Assign devices and users
    device_ids = np.repeat(np.arange(n_devices), n_samples // n_devices)
    user_ids = device_ids  # one user per device
    # Make some features discriminative for each user's genuine sessions
    is_genuine = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    for u in range(n_users):
        mask = (user_ids == u) & (is_genuine == 1)
        # Feature 0-4 are predictive
        X[mask, 0:5] += 3.0 * (u + 1)
        # Feature 5-9 are moderately predictive
        X[mask, 5:10] += 1.5 * (u + 1)

    # Create DataFrame with metadata
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(n_features)])
    df['user_id'] = user_ids
    df['device_id'] = device_ids
    df['is_genuine'] = is_genuine

    # Build pipeline with small values for quick testing
    pipeline = build_full_pipeline(
        tier1_threshold=0.05,        # lower threshold to keep more
        tier2_top_k=15,
        tier3_correlation_threshold=0.8,
        tier3_max_features=12,
        tier4_cv_folds=3,
        tier4_min_features=5,
        tier4_n_jobs=1,              # single thread for test
        random_state=42,
    )

    # Fit the pipeline (this runs the wrapper backward elimination)
    pipeline.fit(df)

    # Ensure the pipeline produced a selection
    selected = pipeline.transform(df)
    print(f"Selected {selected.shape[1]} features out of {n_features}")

    # Run LOGO validation (device and user)
    device_res, user_res = run_logo_validation(pipeline, df, n_jobs=1)

    print("Device LOGO: mean TAR = {:.3f}, mean GDP = {:.3f}".format(
        device_res.get('average_tar', np.nan),
        device_res.get('average_gdp', np.nan)
    ))
    print("User LOGO: mean TAR = {:.3f}, pass = {}".format(
        user_res.get('average_tar', np.nan),
        user_res.get('pass', False)
    ))

    # Check that we got some results (not NaN)
    assert not np.isnan(device_res['average_tar'])
    assert not np.isnan(user_res['average_tar'])
    print("Integration test passed.")


if __name__ == "__main__":
    test_full_pipeline()























# Load your data as a DataFrame with columns: features, 'user_id', 'device_id', 'is_genuine'
# df = pd.read_csv(...)

# Build the pipeline (adjust hyper‑parameters as needed)
pipeline = build_full_pipeline(
    tier1_threshold=0.10,
    tier2_top_k=100,
    tier3_correlation_threshold=0.9,
    tier3_max_features=80,
    tier4_cv_folds=5,
    tier4_min_features=10,
    tier4_n_jobs=-1,
    random_state=42,
)

# Fit the pipeline – this runs the 4‑Tier feature selection
pipeline.fit(df)

# Get the selected features DataFrame
X_selected = pipeline.transform(df)

# Run LOGO validation to confirm generalisation
device_res, user_res = run_logo_validation(pipeline, df, n_jobs=-1)

# Check if we meet the success criteria
if device_res['pass'] and user_res['pass'] and user_res['average_tar'] >= 0.70:
    print("Pipeline passes LOGO validation!")
else:
    print("Pipeline may need tuning; check individual results.")
