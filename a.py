# Hurst Exponent of Motion

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from typing import Optional, Tuple, Dict, Any, List

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_hurst_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Hurst exponent features from accelerometer and gyroscope data.
        
        Uses Detrended Fluctuation Analysis (DFA) as the primary method (Peng et al., 1994)
        and Rescaled Range (R/S) analysis as secondary validation (Hurst, 1951).
        
        The Hurst exponent H ∈ [0,1] quantifies long-term memory:
        - H = 0.5: Brownian motion (no memory)
        - H > 0.5: Persistent behavior (trends continue)
        - H < 0.5: Anti-persistent behavior (trends reverse)
        
        References:
        - Peng C-K, Havlin S, Stanley HE, Goldberger AL. (1994). 
          "Mosaic organization of DNA nucleotides." Phys Rev E 49: 1685-1689.
        - Hurst HE. (1951). "Long-term storage capacity of reservoirs." 
          Trans. Amer. Soc. Civil Eng. 116: 770-808.
        
        Args:
            session: Session dictionary containing sensor_events
            
        Returns:
            Dictionary with Hurst exponent features:
            - hurst_dfa_acc: DFA exponent from accelerometer magnitude
            - hurst_dfa_gyro: DFA exponent from gyroscope magnitude
            - hurst_rs_acc: R/S Hurst exponent from accelerometer magnitude
            - hurst_dfa_r2_acc: R² of DFA log-log regression (accel)
            - hurst_dfa_r2_gyro: R² of DFA log-log regression (gyro)
        """
        sensors = session.get('sensor_events', [])
        
        if len(sensors) < 20:
            return self._hurst_defaults()
        
        # Parse sensor data
        df = self._parse_sensors_no_interpolation(sensors)
        if df.empty or len(df) < 20:
            return self._hurst_defaults()
        
        # Extract accelerometer magnitude (primary)
        acc_x = df['acc_x'].values
        acc_y = df['acc_y'].values
        acc_z = df['acc_z'].values
        
        # Remove NaNs
        valid_mask = ~(np.isnan(acc_x) | np.isnan(acc_y) | np.isnan(acc_z))
        if np.sum(valid_mask) < 20:
            return self._hurst_defaults()
            
        acc_x = acc_x[valid_mask]
        acc_y = acc_y[valid_mask]
        acc_z = acc_z[valid_mask]
        
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Extract gyroscope magnitude (secondary)
        gyro_x = df['gyro_x'].values
        gyro_y = df['gyro_y'].values
        gyro_z = df['gyro_z'].values
        
        valid_mask_gyro = ~(np.isnan(gyro_x) | np.isnan(gyro_y) | np.isnan(gyro_z))
        gyro_mag = None
        if np.sum(valid_mask_gyro) >= 20:
            gyro_x = gyro_x[valid_mask_gyro]
            gyro_y = gyro_y[valid_mask_gyro]
            gyro_z = gyro_z[valid_mask_gyro]
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        # Estimate sampling rate
        fs = self._estimate_sampling_rate(sensors)
        fs = np.clip(fs, 5.0, 100.0)
        
        # Compute features
        features = {}
        
        # Primary: DFA on accelerometer magnitude
        dfa_acc, r2_acc = self._compute_dfa(acc_mag, fs=fs)
        features['hurst_dfa_acc'] = float(np.clip(dfa_acc, 0.0, 1.5)) if dfa_acc is not None else 0.0
        features['hurst_dfa_r2_acc'] = float(np.clip(r2_acc, 0.0, 1.0)) if r2_acc is not None else 0.0
        
        # Secondary: DFA on gyroscope magnitude (if available)
        if gyro_mag is not None and len(gyro_mag) >= 20:
            dfa_gyro, r2_gyro = self._compute_dfa(gyro_mag, fs=fs)
            features['hurst_dfa_gyro'] = float(np.clip(dfa_gyro, 0.0, 1.5)) if dfa_gyro is not None else 0.0
            features['hurst_dfa_r2_gyro'] = float(np.clip(r2_gyro, 0.0, 1.0)) if r2_gyro is not None else 0.0
        else:
            features['hurst_dfa_gyro'] = 0.0
            features['hurst_dfa_r2_gyro'] = 0.0
        
        # Validation: R/S analysis on accelerometer magnitude
        rs_acc = self._compute_hurst_rs(acc_mag)
        features['hurst_rs_acc'] = float(np.clip(rs_acc, 0.0, 1.5)) if rs_acc is not None else 0.0
        
        return features
    
    def _hurst_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'hurst_dfa_acc': 0.0,
            'hurst_dfa_gyro': 0.0,
            'hurst_rs_acc': 0.0,
            'hurst_dfa_r2_acc': 0.0,
            'hurst_dfa_r2_gyro': 0.0,
        }
    
    def _compute_dfa(self, signal: np.ndarray, fs: float = 15.0, 
                     order: int = 1) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Detrended Fluctuation Analysis (DFA) exponent.
        
        Implements the algorithm from Peng et al. (1994).
        
        Args:
            signal: 1D time series (magnitude or individual axis)
            fs: Sampling rate in Hz (used for window scaling)
            order: Polynomial order for detrending (1 = linear)
            
        Returns:
            Tuple of (DFA exponent α, R² of log-log regression)
            Returns (None, None) if computation fails.
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        # Remove any NaN or inf values
        signal = signal[np.isfinite(signal)]
        
        if len(signal) < 20:
            return None, None
        
        # Remove mean and integrate to get profile
        signal_mean = np.mean(signal)
        profile = np.cumsum(signal - signal_mean)
        
        N = len(profile)
        
        # Generate window sizes on logarithmic scale
        # Minimum: 4 samples (at 15 Hz, ~0.27 seconds)
        # Maximum: N/6 (ensures at least 6 segments)
        w_min = max(4, int(np.sqrt(N)))  # At least 4, scales with sqrt(N)
        w_max = max(w_min + 1, int(N / 6))
        
        if w_max <= w_min:
            w_min = 4
            w_max = max(5, int(N / 6))
        
        if w_max <= w_min:
            return None, None
        
        # Use logarithmic spacing (more weight to small scales)
        n_windows = min(12, int(np.log2(w_max / w_min) * 3) + 2)
        n_windows = max(4, n_windows)  # Need at least 4 points for regression
        
        windows = np.logspace(np.log10(w_min), np.log10(w_max), n_windows)
        windows = np.unique(np.round(windows).astype(int))
        windows = windows[windows >= w_min]
        windows = windows[windows <= w_max]
        
        if len(windows) < 4:
            return None, None
        
        fluctuations = []
        
        for w in windows:
            n_segments = N // w
            if n_segments < 2:
                continue
                
            rms_vals = []
            for i in range(n_segments):
                start = i * w
                end = start + w
                segment = profile[start:end]
                
                # Fit polynomial of given order
                x_vals = np.arange(w)
                
                if order == 1:
                    # Linear detrending (optimized for speed)
                    coeffs = np.polyfit(x_vals, segment, 1)
                    trend = np.polyval(coeffs, x_vals)
                elif order == 2:
                    coeffs = np.polyfit(x_vals, segment, 2)
                    trend = np.polyval(coeffs, x_vals)
                else:
                    # Fallback to linear
                    coeffs = np.polyfit(x_vals, segment, 1)
                    trend = np.polyval(coeffs, x_vals)
                
                # Detrended fluctuation
                residual = segment - trend
                rms = np.sqrt(np.mean(residual ** 2))
                rms_vals.append(rms)
            
            if rms_vals:
                fluctuations.append(np.mean(rms_vals))
            else:
                fluctuations.append(0.0)
        
        fluctuations = np.array(fluctuations)
        
        # Remove any zero or invalid values
        valid = (fluctuations > 1e-12) & np.isfinite(fluctuations)
        if np.sum(valid) < 4:
            return None, None
        
        log_w = np.log(windows[valid])
        log_f = np.log(fluctuations[valid])
        
        # Fit linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_w, log_f)
            r2 = r_value ** 2
            return slope, r2
        except (ValueError, RuntimeError, stats.LinAlgError):
            return None, None
    
    def _compute_hurst_rs(self, signal: np.ndarray) -> Optional[float]:
        """
        Compute Hurst exponent using Rescaled Range (R/S) analysis.
        
        Implements the algorithm from Hurst (1951).
        
        Args:
            signal: 1D time series
            
        Returns:
            Hurst exponent H, or None if computation fails.
        """
        signal = np.asarray(signal, dtype=np.float64)
        signal = signal[np.isfinite(signal)]
        
        if len(signal) < 20:
            return None
        
        N = len(signal)
        
        # Window sizes: logarithmic from 10 to N/3
        w_min = max(5, int(np.sqrt(N)))
        w_max = max(w_min + 1, int(N / 3))
        
        if w_max <= w_min:
            return None
        
        n_windows = min(10, int(np.log2(w_max / w_min) * 3) + 2)
        n_windows = max(4, n_windows)
        
        windows = np.logspace(np.log10(w_min), np.log10(w_max), n_windows)
        windows = np.unique(np.round(windows).astype(int))
        windows = windows[windows >= w_min]
        windows = windows[windows <= w_max]
        
        if len(windows) < 4:
            return None
        
        rs_ratios = []
        
        for w in windows:
            n_segments = N // w
            if n_segments < 2:
                continue
                
            rs_vals = []
            for i in range(n_segments):
                start = i * w
                end = start + w
                segment = signal[start:end]
                
                # Mean
                mean_seg = np.mean(segment)
                
                # Cumulative deviation from mean
                deviation = np.cumsum(segment - mean_seg)
                
                # Range
                R = np.max(deviation) - np.min(deviation)
                
                # Standard deviation
                S = np.std(segment, ddof=1)
                
                if S > 1e-12:
                    rs_vals.append(R / S)
            
            if rs_vals:
                rs_ratios.append(np.mean(rs_vals))
            else:
                rs_ratios.append(0.0)
        
        rs_ratios = np.array(rs_ratios)
        
        valid = (rs_ratios > 1e-12) & np.isfinite(rs_ratios)
        if np.sum(valid) < 4:
            return None
        
        log_w = np.log(windows[valid])
        log_rs = np.log(rs_ratios[valid])
        
        try:
            slope, _, _, _, _ = stats.linregress(log_w, log_rs)
            return slope
        except (ValueError, RuntimeError, stats.LinAlgError):
            return None



























# Fitts' Law Intercept

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_fitts_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Fitts' Law features from touch events.
        
        Fitts' Law describes the speed-accuracy tradeoff in human movement:
            MT = a + b * log2(A/W + 1)
        
        where:
            MT = Movement Time (touch_up_ts - touch_down_ts)
            A = Distance between consecutive touches
            W = Target width (estimated from touch dispersion or default)
            a = Intercept (reaction time + cognitive processing) - PRIMARY FEATURE
            b = Slope (speed-accuracy tradeoff rate)
        
        The intercept 'a' is a stable individual trait reflecting:
        - Neuromotor processing speed
        - Visual-motor integration time
        - Cognitive style (cautious vs. impulsive)
        
        References:
        - Fitts PM. (1954). "The information capacity of the human motor system 
          in controlling the amplitude of movement." J Exp Psychol. 47(6):381-391.
        - MacKenzie IS. (1992). "Fitts' law as a research and design tool in 
          human-computer interaction." Human-Computer Interaction. 7(1):91-139.
        
        Args:
            session: Session dictionary containing touch_events and device_info
            
        Returns:
            Dictionary with Fitts' Law features:
            - fitts_intercept: Y-intercept (a) from regression
            - fitts_slope: Slope (b) from regression
            - fitts_r_squared: R² of the regression (quality metric)
            - fitts_throughput: Throughput = ID / MT (information transfer rate)
            - fitts_n_points: Number of touch pairs used
        """
        touches = session.get('touch_events', [])
        
        if len(touches) < 3:
            return self._fitts_defaults()
        
        # Get screen dimensions for normalization
        screen_w = session.get('device_info', {}).get('screen_width', 1)
        screen_h = session.get('device_info', {}).get('screen_height', 1)
        
        # Filter for tap events only (swipes have different dynamics)
        # If gesture_type is not available, use all touches
        taps = [t for t in touches if t.get('gesture_type') in [None, 'tap', '']]
        if len(taps) < 3:
            taps = touches  # Fallback to all touches
        
        # Collect touch pairs for Fitts' Law
        distances = []
        movement_times = []
        
        for i in range(1, len(taps)):
            t_prev = taps[i - 1]
            t_curr = taps[i]
            
            # Skip if missing required fields
            if not all(k in t_prev for k in ['x', 'y', 'touch_down_ts', 'touch_up_ts']):
                continue
            if not all(k in t_curr for k in ['x', 'y', 'touch_down_ts', 'touch_up_ts']):
                continue
            
            # Normalize coordinates to [0, 1] range (device-independent)
            x1 = t_prev['x'] / screen_w
            y1 = t_prev['y'] / screen_h
            x2 = t_curr['x'] / screen_w
            y2 = t_curr['y'] / screen_h
            
            # Compute distance (in normalized units, then convert to pixels)
            dx = x2 - x1
            dy = y2 - y1
            distance_pixels = np.sqrt(dx**2 + dy**2) * screen_w
            
            # Skip if distance is too small (same key tap, no meaningful movement)
            if distance_pixels < 10.0:
                continue
            
            # Movement Time: time from previous touch up to current touch down
            # This captures the full movement cycle
            mt = t_curr['touch_down_ts'] - t_prev['touch_up_ts']
            
            # Skip unrealistic movement times
            if mt < 20 or mt > 5000:  # 20ms to 5s
                continue
            
            distances.append(distance_pixels)
            movement_times.append(mt)
        
        if len(distances) < 3:
            return self._fitts_defaults()
        
        # Estimate target width
        # Method 1: From touch dispersion (if enough taps)
        if len(taps) >= 5:
            x_positions = [t['x'] / screen_w for t in taps if 'x' in t]
            y_positions = [t['y'] / screen_h for t in taps if 'y' in t]
            if len(x_positions) >= 5:
                # Estimate width as 2 * std of touch positions
                w_est = 2 * np.std(x_positions) * screen_w
                w_est = max(w_est, 20.0)  # Minimum 20 pixels
                target_width = w_est
            else:
                target_width = 50.0  # Default button width in pixels
        else:
            target_width = 50.0  # Default button width in pixels
        
        # Ensure target width is reasonable
        target_width = np.clip(target_width, 20.0, 200.0)
        
        # Compute Index of Difficulty (Shannon formulation)
        # ID = log2(A/W + 1)
        distances = np.array(distances)
        movement_times = np.array(movement_times)
        
        id_values = np.log2(distances / target_width + 1)
        
        # Remove any invalid IDs
        valid = np.isfinite(id_values) & (id_values > 0)
        if np.sum(valid) < 3:
            return self._fitts_defaults()
        
        id_values = id_values[valid]
        movement_times = movement_times[valid]
        
        # Linear regression: MT = a + b * ID
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(id_values, movement_times)
            r_squared = r_value ** 2
        except (ValueError, RuntimeError, stats.LinAlgError):
            return self._fitts_defaults()
        
        # Compute throughput: TP = ID / MT (information transfer rate)
        # Average throughput across all points
        throughputs = id_values / movement_times
        throughput_mean = float(np.mean(throughputs))
        
        return {
            'fitts_intercept': float(intercept),
            'fitts_slope': float(slope),
            'fitts_r_squared': float(r_squared),
            'fitts_throughput': float(throughput_mean),
            'fitts_n_points': int(len(id_values)),
        }
    
    def _fitts_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'fitts_intercept': 0.0,
            'fitts_slope': 0.0,
            'fitts_r_squared': 0.0,
            'fitts_throughput': 0.0,
            'fitts_n_points': 0.0,
        }






























# Mutual Information (Acc ↔ Gyro)


import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_mutual_information_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Mutual Information features between accelerometer and gyroscope.
        
        Mutual Information quantifies the amount of information shared between
        two signals. High MI indicates tight coupling between linear and rotational
        motion, reflecting individual motor coordination patterns.
        
        Mathematical definition (Shannon, 1948):
            I(X;Y) = sum_x sum_y p(x,y) * log2(p(x,y) / (p(x) * p(y)))
        
        References:
        - Shannon CE. (1948). "A Mathematical Theory of Communication."
          Bell System Technical Journal, 27:379-423, 623-656.
        - Cover TM, Thomas JA. (1991). "Elements of Information Theory."
          John Wiley & Sons.
        
        Args:
            session: Session dictionary containing sensor_events
            
        Returns:
            Dictionary with Mutual Information features:
            - mi_acc_gyro: Mutual information between acc and gyro magnitudes
            - mi_acc_gyro_norm: Normalized mutual information [0, 1]
            - mi_acc_gyro_std: Standard deviation across windows
            - mi_acc_gyro_windows: Number of windows used
        """
        sensors = session.get('sensor_events', [])
        
        if len(sensors) < 20:
            return self._mi_defaults()
        
        # Parse sensor data
        df = self._parse_sensors_no_interpolation(sensors)
        if df.empty or len(df) < 20:
            return self._mi_defaults()
        
        # Extract accelerometer magnitude
        acc_x = df['acc_x'].values
        acc_y = df['acc_y'].values
        acc_z = df['acc_z'].values
        
        valid_mask = ~(np.isnan(acc_x) | np.isnan(acc_y) | np.isnan(acc_z))
        if np.sum(valid_mask) < 20:
            return self._mi_defaults()
        
        acc_x = acc_x[valid_mask]
        acc_y = acc_y[valid_mask]
        acc_z = acc_z[valid_mask]
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Extract gyroscope magnitude
        gyro_x = df['gyro_x'].values
        gyro_y = df['gyro_y'].values
        gyro_z = df['gyro_z'].values
        
        valid_mask_gyro = ~(np.isnan(gyro_x) | np.isnan(gyro_y) | np.isnan(gyro_z))
        if np.sum(valid_mask_gyro) < 20:
            return self._mi_defaults()
        
        gyro_x = gyro_x[valid_mask_gyro]
        gyro_y = gyro_y[valid_mask_gyro]
        gyro_z = gyro_z[valid_mask_gyro]
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        # Ensure both signals have the same length
        min_len = min(len(acc_mag), len(gyro_mag))
        acc_mag = acc_mag[:min_len]
        gyro_mag = gyro_mag[:min_len]
        
        if min_len < 20:
            return self._mi_defaults()
        
        # Estimate sampling rate
        fs = self._estimate_sampling_rate(sensors)
        fs = np.clip(fs, 5.0, 100.0)
        
        # Window size: ~10 seconds
        window_size = int(10 * fs)
        window_size = max(16, min(window_size, min_len // 2))
        
        # Ensure at least 2 windows
        if window_size > min_len // 2:
            window_size = min_len // 2
        
        if window_size < 8:
            return self._mi_defaults()
        
        # Number of bins for histogram estimation
        # For 15 Hz data, 8 bins is appropriate (8^2 = 64 joint states)
        n_bins = 8
        
        mi_values = []
        
        # Sliding windows with 50% overlap
        hop_size = window_size // 2
        
        for start in range(0, min_len - window_size + 1, hop_size):
            acc_window = acc_mag[start:start + window_size]
            gyro_window = gyro_mag[start:start + window_size]
            
            mi = self._compute_mutual_information(acc_window, gyro_window, n_bins)
            if mi is not None and mi >= 0:
                mi_values.append(mi)
        
        if not mi_values:
            return self._mi_defaults()
        
        mi_values = np.array(mi_values)
        
        # Compute features
        mi_mean = float(np.mean(mi_values))
        mi_std = float(np.std(mi_values))
        
        # Normalize mutual information
        # I_norm = I / min(H(X), H(Y))
        # For uniform distribution with B bins, max entropy = log2(B)
        max_entropy = np.log2(n_bins)
        
        # Estimate entropy from the data
        h_acc = self._compute_entropy(acc_mag, n_bins)
        h_gyro = self._compute_entropy(gyro_mag, n_bins)
        
        if h_acc > 0 and h_gyro > 0:
            mi_norm = mi_mean / min(h_acc, h_gyro)
            mi_norm = np.clip(mi_norm, 0.0, 1.0)
        else:
            mi_norm = mi_mean / max_entropy
            mi_norm = np.clip(mi_norm, 0.0, 1.0)
        
        return {
            'mi_acc_gyro': mi_mean,
            'mi_acc_gyro_norm': mi_norm,
            'mi_acc_gyro_std': mi_std,
            'mi_acc_gyro_windows': float(len(mi_values)),
        }
    
    def _mi_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'mi_acc_gyro': 0.0,
            'mi_acc_gyro_norm': 0.0,
            'mi_acc_gyro_std': 0.0,
            'mi_acc_gyro_windows': 0.0,
        }
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray, 
                                    bins: int = 8) -> Optional[float]:
        """
        Compute mutual information between two signals using histogram estimation.
        
        Implements the histogram-based estimator described in Cover & Thomas (1991).
        
        Args:
            x: First signal (1D array)
            y: Second signal (1D array)
            bins: Number of bins for discretization
            
        Returns:
            Mutual information in bits, or None if computation fails.
        """
        if len(x) < 10 or len(y) < 10:
            return None
        
        # Remove any NaN or inf values
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 10:
            return None
        
        x = x[valid]
        y = y[valid]
        
        # Compute bin edges based on percentiles for robust discretization
        x_edges = np.percentile(x, np.linspace(0, 100, bins + 1))
        y_edges = np.percentile(y, np.linspace(0, 100, bins + 1))
        
        # Ensure unique edges (handle edge case where all values are equal)
        if len(np.unique(x_edges)) < 2 or len(np.unique(y_edges)) < 2:
            return None
        
        # Discretize
        x_disc = np.digitize(x, x_edges[:-1]) - 1
        y_disc = np.digitize(y, y_edges[:-1]) - 1
        
        # Clamp to [0, bins-1]
        x_disc = np.clip(x_disc, 0, bins - 1)
        y_disc = np.clip(y_disc, 0, bins - 1)
        
        # Joint histogram
        joint_hist = np.zeros((bins, bins), dtype=np.float64)
        for i in range(len(x_disc)):
            joint_hist[x_disc[i], y_disc[i]] += 1
        
        joint_hist = joint_hist / np.sum(joint_hist)
        
        # Marginal distributions
        px = np.sum(joint_hist, axis=1)
        py = np.sum(joint_hist, axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_hist[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                    mi += joint_hist[i, j] * np.log2(joint_hist[i, j] / (px[i] * py[j]))
        
        # Ensure non-negative (numerical stability)
        return max(0.0, mi)
    
    def _compute_entropy(self, signal: np.ndarray, bins: int = 8) -> float:
        """
        Compute Shannon entropy of a signal using histogram estimation.
        
        Args:
            signal: Input signal (1D array)
            bins: Number of bins for discretization
            
        Returns:
            Entropy in bits.
        """
        signal = signal[np.isfinite(signal)]
        if len(signal) < 10:
            return 0.0
        
        # Compute histogram
        hist, _ = np.histogram(signal, bins=bins)
        hist = hist / np.sum(hist)
        
        # Entropy
        entropy = 0.0
        for p in hist:
            if p > 1e-12:
                entropy -= p * np.log2(p)
        
        return entropy





























 # Transfer Entropy (Acc → Gyro)


import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_transfer_entropy_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Transfer Entropy features from accelerometer to gyroscope.
        
        Transfer Entropy (Schreiber, 2000) quantifies the directed information flow
        from a source process (accelerometer) to a destination process (gyroscope).
        It measures how much the past of the source helps predict the future of
        the destination, beyond what the destination's own past already provides.
        
        Mathematical definition:
            T_{X->Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
        
        where H is conditional entropy.
        
        References:
        - Schreiber T. (2000). "Measuring information transfer."
          Physical Review Letters, 85(2), 461-464.
        - Cover TM, Thomas JA. (1991). "Elements of Information Theory."
          John Wiley & Sons.
        
        Args:
            session: Session dictionary containing sensor_events
            
        Returns:
            Dictionary with Transfer Entropy features:
            - te_acc_to_gyro: Transfer entropy from acc to gyro (bits)
            - te_acc_to_gyro_norm: Normalized transfer entropy [0, 1]
            - te_acc_to_gyro_std: Standard deviation across windows
            - te_acc_to_gyro_windows: Number of windows used
        """
        sensors = session.get('sensor_events', [])
        
        if len(sensors) < 20:
            return self._te_defaults()
        
        # Parse sensor data
        df = self._parse_sensors_no_interpolation(sensors)
        if df.empty or len(df) < 20:
            return self._te_defaults()
        
        # Extract accelerometer magnitude
        acc_x = df['acc_x'].values
        acc_y = df['acc_y'].values
        acc_z = df['acc_z'].values
        
        valid_mask = ~(np.isnan(acc_x) | np.isnan(acc_y) | np.isnan(acc_z))
        if np.sum(valid_mask) < 20:
            return self._te_defaults()
        
        acc_x = acc_x[valid_mask]
        acc_y = acc_y[valid_mask]
        acc_z = acc_z[valid_mask]
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Extract gyroscope magnitude
        gyro_x = df['gyro_x'].values
        gyro_y = df['gyro_y'].values
        gyro_z = df['gyro_z'].values
        
        valid_mask_gyro = ~(np.isnan(gyro_x) | np.isnan(gyro_y) | np.isnan(gyro_z))
        if np.sum(valid_mask_gyro) < 20:
            return self._te_defaults()
        
        gyro_x = gyro_x[valid_mask_gyro]
        gyro_y = gyro_y[valid_mask_gyro]
        gyro_z = gyro_z[valid_mask_gyro]
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        # Ensure both signals have the same length
        min_len = min(len(acc_mag), len(gyro_mag))
        acc_mag = acc_mag[:min_len]
        gyro_mag = gyro_mag[:min_len]
        
        if min_len < 20:
            return self._te_defaults()
        
        # Estimate sampling rate
        fs = self._estimate_sampling_rate(sensors)
        fs = np.clip(fs, 5.0, 100.0)
        
        # Window size: ~10 seconds
        window_size = int(10 * fs)
        window_size = max(16, min(window_size, min_len // 2))
        
        # Ensure at least 2 windows
        if window_size > min_len // 2:
            window_size = min_len // 2
        
        if window_size < 8:
            return self._te_defaults()
        
        # Number of bins for histogram estimation
        # For 15 Hz data, 6 bins is appropriate (6^3 = 216 joint states)
        n_bins = 6
        
        te_values = []
        
        # Sliding windows with 50% overlap
        hop_size = window_size // 2
        
        for start in range(0, min_len - window_size + 1, hop_size):
            acc_window = acc_mag[start:start + window_size]
            gyro_window = gyro_mag[start:start + window_size]
            
            te = self._compute_transfer_entropy(acc_window, gyro_window, n_bins)
            if te is not None and te >= 0:
                te_values.append(te)
        
        if not te_values:
            return self._te_defaults()
        
        te_values = np.array(te_values)
        
        # Compute features
        te_mean = float(np.mean(te_values))
        te_std = float(np.std(te_values))
        
        # Normalize transfer entropy
        # Max TE is bounded by min(H(source), H(dest))
        # We use the entropy of the destination as an upper bound
        h_gyro = self._compute_entropy(gyro_mag, n_bins)
        if h_gyro > 0:
            te_norm = te_mean / h_gyro
            te_norm = np.clip(te_norm, 0.0, 1.0)
        else:
            te_norm = 0.0
        
        return {
            'te_acc_to_gyro': te_mean,
            'te_acc_to_gyro_norm': te_norm,
            'te_acc_to_gyro_std': te_std,
            'te_acc_to_gyro_windows': float(len(te_values)),
        }
    
    def _te_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'te_acc_to_gyro': 0.0,
            'te_acc_to_gyro_norm': 0.0,
            'te_acc_to_gyro_std': 0.0,
            'te_acc_to_gyro_windows': 0.0,
        }
    
    def _compute_transfer_entropy(self, x: np.ndarray, y: np.ndarray, 
                                   bins: int = 6) -> Optional[float]:
        """
        Compute transfer entropy from source X to destination Y.
        
        Implements the histogram-based estimator for transfer entropy
        (Schreiber, 2000) with history length = 1 for both source and destination.
        
        Args:
            x: Source signal (accelerometer magnitude)
            y: Destination signal (gyroscope magnitude)
            bins: Number of bins for discretization
            
        Returns:
            Transfer entropy in bits, or None if computation fails.
        """
        if len(x) < 10 or len(y) < 10:
            return None
        
        # Remove any NaN or inf values
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 10:
            return None
        
        x = x[valid]
        y = y[valid]
        
        # Compute bin edges based on percentiles for robust discretization
        x_edges = np.percentile(x, np.linspace(0, 100, bins + 1))
        y_edges = np.percentile(y, np.linspace(0, 100, bins + 1))
        
        # Ensure unique edges (handle edge case where all values are equal)
        if len(np.unique(x_edges)) < 2 or len(np.unique(y_edges)) < 2:
            return None
        
        # Discretize
        x_disc = np.digitize(x, x_edges[:-1]) - 1
        y_disc = np.digitize(y, y_edges[:-1]) - 1
        
        # Clamp to [0, bins-1]
        x_disc = np.clip(x_disc, 0, bins - 1)
        y_disc = np.clip(y_disc, 0, bins - 1)
        
        N = len(x_disc)
        
        # History length = 1 (lag-1)
        # We need: y_{t+1}, y_t, x_t
        # t goes from 0 to N-2
        
        # Count joint configurations: (y_next, y_curr, x_curr)
        joint_counts = defaultdict(int)
        y_curr_y_next_counts = defaultdict(int)
        y_curr_counts = defaultdict(int)
        y_curr_x_curr_counts = defaultdict(int)
        
        for t in range(N - 1):
            y_next = y_disc[t + 1]
            y_curr = y_disc[t]
            x_curr = x_disc[t]
            
            # Joint: (y_next, y_curr, x_curr)
            joint_counts[(y_next, y_curr, x_curr)] += 1
            
            # Marginal: (y_next, y_curr)
            y_curr_y_next_counts[(y_curr, y_next)] += 1
            
            # Marginal: (y_curr)
            y_curr_counts[y_curr] += 1
            
            # Marginal: (y_curr, x_curr)
            y_curr_x_curr_counts[(y_curr, x_curr)] += 1
        
        total = N - 1
        if total == 0:
            return None
        
        # Compute transfer entropy
        te = 0.0
        
        for (y_next, y_curr, x_curr), count_joint in joint_counts.items():
            p_joint = count_joint / total
            
            # p(y_next | y_curr) = p(y_next, y_curr) / p(y_curr)
            p_y_next_given_y_curr = 0.0
            if y_curr_counts[y_curr] > 0:
                p_y_next_y_curr = y_curr_y_next_counts[(y_curr, y_next)] / total
                p_y_curr = y_curr_counts[y_curr] / total
                if p_y_curr > 0:
                    p_y_next_given_y_curr = p_y_next_y_curr / p_y_curr
            
            # p(y_next | y_curr, x_curr) = p(y_next, y_curr, x_curr) / p(y_curr, x_curr)
            p_y_next_given_y_curr_x_curr = 0.0
            if y_curr_x_curr_counts[(y_curr, x_curr)] > 0:
                p_y_curr_x_curr = y_curr_x_curr_counts[(y_curr, x_curr)] / total
                if p_y_curr_x_curr > 0:
                    p_y_next_given_y_curr_x_curr = p_joint / p_y_curr_x_curr
            
            if p_y_next_given_y_curr > 0 and p_y_next_given_y_curr_x_curr > 0:
                te += p_joint * np.log2(p_y_next_given_y_curr_x_curr / p_y_next_given_y_curr)
        
        # Ensure non-negative (numerical stability)
        return max(0.0, te)
    
    def _compute_entropy(self, signal: np.ndarray, bins: int = 6) -> float:
        """
        Compute Shannon entropy of a signal using histogram estimation.
        
        Args:
            signal: Input signal (1D array)
            bins: Number of bins for discretization
            
        Returns:
            Entropy in bits.
        """
        signal = signal[np.isfinite(signal)]
        if len(signal) < 10:
            return 0.0
        
        # Compute histogram
        hist, _ = np.histogram(signal, bins=bins)
        hist = hist / np.sum(hist)
        
        # Entropy
        entropy = 0.0
        for p in hist:
            if p > 1e-12:
                entropy -= p * np.log2(p)
        
        return entropy
































# family Determinism (RQA)
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class SessionLevelFeatureExtractor:
    # ... existing code ...
    
    def extract_rqa_features(self, session: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Recurrence Quantification Analysis (RQA) features.
        
        Determinism (DET) measures the predictability of a dynamical system.
        It is the ratio of recurrence points that form diagonal lines to all
        recurrence points in the recurrence plot (Webber & Zbilut, 1994).
        
        Mathematical definition:
            DET = sum_{l=l_min}^N l * P(l) / sum_{l=1}^N l * P(l)
        
        where P(l) is the histogram of diagonal line lengths.
        
        High DET indicates highly predictable, deterministic motion.
        Low DET indicates stochastic, random motion.
        
        References:
        - Webber CL, Zbilut JP. (1994). "Dynamical assessment of physiological 
          systems and states using recurrence plot strategies." 
          J Appl Physiol. 76(2):965-973.
        - Eckmann JP, Kamphorst SO, Ruelle D. (1987). "Recurrence plots of 
          dynamical systems." Europhys Lett. 4(9):973-977.
        - Marwan N, Romano MC, Thiel M, Kurths J. (2007). "Recurrence plots 
          for the analysis of complex systems." Phys Rep. 438(5-6):237-329.
        
        Args:
            session: Session dictionary containing sensor_events
            
        Returns:
            Dictionary with RQA features:
            - rqa_det_acc: Determinism from accelerometer magnitude
            - rqa_det_gyro: Determinism from gyroscope magnitude
            - rqa_det_acc_std: Standard deviation across windows
            - rqa_det_windows: Number of windows used
        """
        sensors = session.get('sensor_events', [])
        
        if len(sensors) < 20:
            return self._rqa_defaults()
        
        # Parse sensor data
        df = self._parse_sensors_no_interpolation(sensors)
        if df.empty or len(df) < 20:
            return self._rqa_defaults()
        
        # Extract accelerometer magnitude
        acc_x = df['acc_x'].values
        acc_y = df['acc_y'].values
        acc_z = df['acc_z'].values
        
        valid_mask = ~(np.isnan(acc_x) | np.isnan(acc_y) | np.isnan(acc_z))
        if np.sum(valid_mask) < 20:
            return self._rqa_defaults()
        
        acc_x = acc_x[valid_mask]
        acc_y = acc_y[valid_mask]
        acc_z = acc_z[valid_mask]
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Extract gyroscope magnitude
        gyro_x = df['gyro_x'].values
        gyro_y = df['gyro_y'].values
        gyro_z = df['gyro_z'].values
        
        valid_mask_gyro = ~(np.isnan(gyro_x) | np.isnan(gyro_y) | np.isnan(gyro_z))
        gyro_mag = None
        if np.sum(valid_mask_gyro) >= 20:
            gyro_x = gyro_x[valid_mask_gyro]
            gyro_y = gyro_y[valid_mask_gyro]
            gyro_z = gyro_z[valid_mask_gyro]
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        # Estimate sampling rate
        fs = self._estimate_sampling_rate(sensors)
        fs = np.clip(fs, 5.0, 100.0)
        
        # Window size: ~10 seconds
        window_size = int(10 * fs)
        window_size = max(16, min(window_size, len(acc_mag) // 2))
        
        if window_size < 8:
            return self._rqa_defaults()
        
        # RQA parameters adapted for 15 Hz data
        embedding_dim = 2  # Reduced from 3 for 15 Hz data
        time_delay = 1
        radius_factor = 0.25  # Slightly larger than standard 0.2 for noise tolerance
        min_diagonal_length = 2
        theiler_window = 1
        
        det_acc_values = []
        det_gyro_values = []
        
        # Sliding windows with 50% overlap
        hop_size = window_size // 2
        
        for start in range(0, len(acc_mag) - window_size + 1, hop_size):
            acc_window = acc_mag[start:start + window_size]
            
            det = self._compute_determinism(
                acc_window, 
                embedding_dim=embedding_dim,
                time_delay=time_delay,
                radius_factor=radius_factor,
                min_diagonal_length=min_diagonal_length,
                theiler_window=theiler_window
            )
            if det is not None and det >= 0:
                det_acc_values.append(det)
            
            # Compute gyro determinism if available
            if gyro_mag is not None and start + window_size <= len(gyro_mag):
                gyro_window = gyro_mag[start:start + window_size]
                det_gyro = self._compute_determinism(
                    gyro_window,
                    embedding_dim=embedding_dim,
                    time_delay=time_delay,
                    radius_factor=radius_factor,
                    min_diagonal_length=min_diagonal_length,
                    theiler_window=theiler_window
                )
                if det_gyro is not None and det_gyro >= 0:
                    det_gyro_values.append(det_gyro)
        
        if not det_acc_values:
            return self._rqa_defaults()
        
        det_acc_values = np.array(det_acc_values)
        
        features = {
            'rqa_det_acc': float(np.mean(det_acc_values)),
            'rqa_det_acc_std': float(np.std(det_acc_values)),
            'rqa_det_windows': float(len(det_acc_values)),
        }
        
        if det_gyro_values:
            det_gyro_values = np.array(det_gyro_values)
            features['rqa_det_gyro'] = float(np.mean(det_gyro_values))
        else:
            features['rqa_det_gyro'] = 0.0
        
        return features
    
    def _rqa_defaults(self) -> Dict[str, float]:
        """Return default values when insufficient data is available."""
        return {
            'rqa_det_acc': 0.0,
            'rqa_det_gyro': 0.0,
            'rqa_det_acc_std': 0.0,
            'rqa_det_windows': 0.0,
        }
    
    def _compute_determinism(
        self,
        signal: np.ndarray,
        embedding_dim: int = 2,
        time_delay: int = 1,
        radius_factor: float = 0.25,
        min_diagonal_length: int = 2,
        theiler_window: int = 1
    ) -> Optional[float]:
        """
        Compute determinism (DET) from a time series using RQA.
        
        Implements the algorithm from Webber & Zbilut (1994).
        
        Args:
            signal: 1D time series
            embedding_dim: Embedding dimension (m)
            time_delay: Time delay (tau)
            radius_factor: Recurrence threshold as fraction of signal std
            min_diagonal_length: Minimum diagonal line length (l_min)
            theiler_window: Theiler correction window
            
        Returns:
            Determinism (DET) in [0, 1], or None if computation fails.
        """
        signal = signal[np.isfinite(signal)]
        
        if len(signal) < 20:
            return None
        
        # Normalize signal (zero mean, unit variance)
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        if signal_std < 1e-12:
            return None
        signal_norm = (signal - signal_mean) / signal_std
        
        N = len(signal_norm)
        
        # Phase space reconstruction (Takens, 1981)
        M = N - (embedding_dim - 1) * time_delay
        if M < 10:
            return None
        
        states = np.zeros((M, embedding_dim))
        for i in range(M):
            for j in range(embedding_dim):
                states[i, j] = signal_norm[i + j * time_delay]
        
        # Compute recurrence threshold
        # Using a fixed radius factor of the standard deviation
        epsilon = radius_factor  # Since signal is normalized
        
        # Compute recurrence matrix (optimized)
        # Use Theiler correction to exclude temporally adjacent points
        R = np.zeros((M, M), dtype=np.int8)
        for i in range(M):
            for j in range(i + theiler_window + 1, M):
                dist = np.linalg.norm(states[i] - states[j])
                if dist < epsilon:
                    R[i, j] = 1
                    R[j, i] = 1
        
        # Find diagonal lines
        # For each diagonal offset k = j - i
        diagonal_lengths = []
        
        # Main diagonals (k >= 0)
        for k in range(1, M):
            length = 0
            for i in range(M - k):
                if R[i, i + k] == 1:
                    length += 1
                else:
                    if length >= min_diagonal_length:
                        diagonal_lengths.append(length)
                    length = 0
            if length >= min_diagonal_length:
                diagonal_lengths.append(length)
        
        # Off-diagonals (k < 0) - symmetric, so skip
        # (R is symmetric, so we've already counted all diagonals)
        
        if not diagonal_lengths:
            return 0.0
        
        # Compute determinism
        # DET = sum(l * P(l)) / sum(l * P(l)) for all l >= l_min
        # Note: denominator includes ALL recurrence points (including isolated ones)
        # Isolated recurrence points are length 1, which are NOT included in numerator
        
        # Count total recurrence points
        total_recurrence = np.sum(R)
        if total_recurrence == 0:
            return 0.0
        
        # Count recurrence points in diagonal lines
        diag_recurrence = 0
        for length in diagonal_lengths:
            diag_recurrence += length
        
        det = diag_recurrence / total_recurrence
        
        return float(np.clip(det, 0.0, 1.0))












































































