def _get_active_segments(self, signal, k=1.0):
    """
    Retain samples whose absolute deviation from the mean
    exceeds (mean absolute value + k * standard deviation of absolute values).
    This adaptively removes low‑energy noise while keeping genuine motion.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 10:
        return signal

    # Remove DC component
    signal = signal - np.mean(signal)

    # Work with absolute values
    mag = np.abs(signal)

    # Adaptive threshold
    threshold = np.mean(mag) + k * np.std(mag)

    # Safety: if threshold is extremely high (e.g., all values are nearly constant), fall back
    if threshold >= np.max(mag) or np.std(mag) < 1e-10:
        return signal

    active_mask = mag >= threshold
    active_signal = signal[active_mask]

    # If too few samples survive, return the original to avoid losing all data
    if len(active_signal) < 10:
        return signal

    return active_signal