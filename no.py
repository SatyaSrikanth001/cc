def _zero_fft_features(self, prefix):
    """
    Return a dictionary with zero values for the 7 standard FFT descriptors.
    """
    return {
        f'{prefix}_fft_peak_freq': 0.0,
        f'{prefix}_fft_spectral_energy': 0.0,
        f'{prefix}_fft_spectral_entropy': 0.0,
        f'{prefix}_fft_band_low_0_2hz': 0.0,
        f'{prefix}_fft_band_mid_2_5hz': 0.0,
        f'{prefix}_fft_centroid': 0.0,
        f'{prefix}_fft_medoid': 0.0,
    }




def _extract_multitaper_features(self, signal, prefix, fs, num_tapers=3):
    """
    Compute spectral descriptors using multitaper estimation (DPSS).
    Returns the same seven descriptors as _extract_windowed_fft, but far more stable.
    """
    signal = np.array(signal, dtype=np.float64)
    if len(signal) < 32:
        return self._zero_fft_features(prefix)       # <-- changed

    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2

    from scipy.signal import windows
    tapers = windows.dpss(window_size, NW=num_tapers, return_ratios=False)

    peak_freqs, energies, entropies, band_lows, band_mids = [], [], [], [], []
    energy_centroids, energy_medoids = [], []

    for i in range(0, len(signal) - window_size, hop_size):
        chunk = signal[i:i+window_size]
        psd_mt = np.zeros(window_size // 2 + 1)
        for taper in tapers:
            windowed = chunk * taper
            fft_vals = np.abs(np.fft.rfft(windowed))
            psd_mt += fft_vals ** 2
        psd_mt /= num_tapers

        freqs = np.fft.rfftfreq(window_size, d=1/fs)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = psd_mt[pos_mask]
        power = np.clip(power, 1e-12, None)
        prob = power / np.sum(power)

        # --- descriptors ---
        peak_idx = np.argmax(power)
        peak_freqs.append(freqs[peak_idx])
        energies.append(np.sum(power))

        entropy = -np.sum(prob * np.log(prob))
        entropy /= np.log(len(prob)) if len(prob) > 1 else 1e-10
        entropies.append(entropy)

        centroid = np.sum(freqs * power) / (energies[-1] + 1e-10)
        energy_centroids.append(centroid)

        cum_energy = np.cumsum(power)
        medoid_idx = np.searchsorted(cum_energy, 0.5 * cum_energy[-1])
        medoid_idx = min(medoid_idx, len(freqs)-1)
        energy_medoids.append(freqs[medoid_idx])

        band_low_mask = (freqs >= 0.5) & (freqs <= 2)
        band_mid_mask = (freqs >= 2) & (freqs <= 5)
        band_lows.append(np.sum(power[band_low_mask]) if np.any(band_low_mask) else 0.0)
        band_mids.append(np.sum(power[band_mid_mask]) if np.any(band_mid_mask) else 0.0)

    if not peak_freqs:
        return self._zero_fft_features(prefix)       # <-- changed

    return {
        f'{prefix}_mt_peak_freq': float(np.mean(peak_freqs)),
        f'{prefix}_mt_spectral_energy': float(np.mean(energies)),
        f'{prefix}_mt_spectral_entropy': float(np.mean(entropies)),
        f'{prefix}_mt_band_low_0_2hz': float(np.mean(band_lows)),
        f'{prefix}_mt_band_mid_2_5hz': float(np.mean(band_mids)),
        f'{prefix}_mt_centroid': float(np.mean(energy_centroids)),
        f'{prefix}_mt_medoid': float(np.mean(energy_medoids)),
    }