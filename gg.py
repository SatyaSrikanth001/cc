def _get_active_window_indices(self, signal, fs, percentile=30):
    """
    Returns list of start indices of windows that are above the variance threshold.
    """
    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2
    variances = []
    indices = []
    for i in range(0, len(signal) - window_size + 1, hop_size):
        chunk = signal[i:i+window_size]
        variances.append(np.var(chunk))
        indices.append(i)
    if len(variances) < 2:
        return indices   # keep all if too few
    var_thresh = np.percentile(variances, percentile)
    active_indices = [indices[j] for j, v in enumerate(variances) if v >= var_thresh]
    # safety: if fewer than 2 active windows, return all
    if len(active_indices) < 2:
        return indices
    return active_indices