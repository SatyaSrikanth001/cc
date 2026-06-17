def _get_active_window_indices(self, signal, fs, percentile=30):
    """
    Determine which sliding windows contain significant motion.

    Parameters:
        signal : 1D numpy array (the raw, clean sensor signal)
        fs     : estimated sampling rate (Hz)
        percentile : variance threshold percentile (0-100)

    Returns:
        list of start indices (int) for windows with variance >= threshold.
        If too few windows exist, returns all window indices.
    """
    # ---- window parameters (same as in _extract_windowed_fft) ----
    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2

    # ---- collect window variances and their start indices ----
    variances = []
    indices = []
    for i in range(0, len(signal) - window_size + 1, hop_size):
        chunk = signal[i:i+window_size]
        variances.append(np.var(chunk))
        indices.append(i)

    if len(variances) < 2:
        # too few windows, treat all as active
        return indices

    # ---- adaptive threshold: percentile of the window variances ----
    var_thresh = np.percentile(variances, percentile)

    # ---- select active windows ----
    active_indices = [indices[j] for j, v in enumerate(variances) if v >= var_thresh]

    # ---- safety: if almost all windows are removed, fall back to all windows ----
    if len(active_indices) < 2:
        return indices

    return active_indices














def _extract_windowed_fft(self, signal, prefix, fs, active_indices=None):
    signal = np.array(signal, dtype=np.float64)
    if len(signal) < 32:
        return {f'{prefix}_fft_peak_freq': 0.0, ...}  # (existing defaults)

    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2

    peak_freqs, energies, entropies, band_lows, band_mids = [], [], [], [], []
    energy_centroids, energy_medoids = [], []

    # ---- decide which windows to process ----
    if active_indices is None:
        # original behaviour: all windows
        start_indices = list(range(0, len(signal) - window_size + 1, hop_size))
    else:
        start_indices = active_indices

    for i in start_indices:
        chunk = signal[i:i+window_size]
        # ... rest of the processing (unchanged) ...







def _extract_mfcc_features(self, signal, prefix, fs, num_ceps=6, active_indices=None):
    # ... (initial checks and setup) ...
    if active_indices is None:
        start_indices = list(range(0, len(signal) - window_size + 1, hop_size))
    else:
        start_indices = active_indices

    mfccs_all = []
    for i in start_indices:
        chunk = signal[i:i+window_size] * np.hamming(window_size)
        # ... (rest of MFCC computation unchanged) ...







def _extract_multitaper_features(self, signal, prefix, fs, num_tapers=3, active_indices=None):
    # ... (setup) ...
    if active_indices is None:
        start_indices = list(range(0, len(signal) - window_size + 1, hop_size))
    else:
        start_indices = active_indices

    for i in start_indices:
        chunk = signal[i:i+window_size]
        # ... (multitaper processing unchanged) ...








if len(acc_x) > 20:
    # --- Magnitude ---
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    # NO LONGER: acc_mag_active = self._get_active_segments(acc_mag)
    active_idx_mag = self._get_active_window_indices(acc_mag, fs, percentile=30)
    if len(active_idx_mag) > 0:
        features.update(self._extract_windowed_fft(acc_mag, 'acc', fs, active_indices=active_idx_mag))
        features.update(self._extract_mfcc_features(acc_mag, 'acc_mag', fs, active_indices=active_idx_mag))
        features.update(self._extract_multitaper_features(acc_mag, 'acc_mag', fs, active_indices=active_idx_mag))

    # --- Per‑axis X, Y, Z ---
    for axis_name, axis_signal in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z)]:
        active_idx_axis = self._get_active_window_indices(axis_signal, fs, percentile=30)
        if len(active_idx_axis) > 0:
            features.update(self._extract_windowed_fft(axis_signal, axis_name, fs, active_indices=active_idx_axis))
            features.update(self._extract_mfcc_features(axis_signal, axis_name, fs, active_indices=active_idx_axis))
            features.update(self._extract_multitaper_features(axis_signal, axis_name, fs, active_indices=active_idx_axis))






