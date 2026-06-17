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









# ================================================================
# MFCC HELPERS & EXTRACTION
# ================================================================
@staticmethod
def _hz2mel(hz):
    return 2595.0 * np.log10(1 + hz / 700.0)

@staticmethod
def _mel2hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def _extract_mfcc_features(self, signal, prefix, fs, num_ceps=6):
    """
    Extract MFCC features using sliding windows.
    Returns mean, std, and delta (first‑order difference) for each coefficient.
    """
    signal = np.array(signal, dtype=np.float64)
    if len(signal) < 32:
        # Return zeros
        feats = {}
        for i in range(num_ceps):
            feats[f'{prefix}_mfcc_{i}_mean'] = 0.0
            feats[f'{prefix}_mfcc_{i}_std'] = 0.0
            feats[f'{prefix}_mfcc_{i}_delta'] = 0.0
        return feats

    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2

    # Build mel filterbank (adapted for low sampling rate)
    nfilt = 12
    low_freq = 0.5
    high_freq = fs / 2 - 0.5
    mel_points = np.linspace(self._hz2mel(low_freq),
                             self._hz2mel(high_freq),
                             nfilt + 2)
    hz_points = self._mel2hz(mel_points)
    bin = np.floor((window_size // 2 + 1) * hz_points / (fs / 2))

    fbank = np.zeros((nfilt, window_size // 2 + 1))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m+1])
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1] + 1e-10)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m] + 1e-10)

    mfccs_all = []
    for i in range(0, len(signal) - window_size, hop_size):
        chunk = signal[i:i+window_size] * np.hamming(window_size)
        mag_spec = np.abs(np.fft.rfft(chunk))
        mag_spec = np.clip(mag_spec, 1e-12, None)
        fb_energies = np.dot(fbank, mag_spec)
        log_energies = np.log(fb_energies + 1e-12)
        # DCT to get cepstral coefficients
        mfcc = np.zeros(num_ceps)
        for n in range(num_ceps):
            mfcc[n] = np.sum(log_energies *
                             np.cos(np.pi * (n+1) *
                                    (np.arange(nfilt)+0.5) / nfilt))
        mfccs_all.append(mfcc)

    if not mfccs_all:
        feats = {}
        for i in range(num_ceps):
            feats[f'{prefix}_mfcc_{i}_mean'] = 0.0
            feats[f'{prefix}_mfcc_{i}_std'] = 0.0
            feats[f'{prefix}_mfcc_{i}_delta'] = 0.0
        return feats

    mfccs_all = np.array(mfccs_all)      # (n_windows, num_ceps)
    mean_mfcc = np.mean(mfccs_all, axis=0)
    std_mfcc = np.std(mfccs_all, axis=0)
    delta_mfcc = (np.mean(np.diff(mfccs_all, axis=0), axis=0)
                  if mfccs_all.shape[0] > 1 else np.zeros(num_ceps))

    features = {}
    for i in range(num_ceps):
        features[f'{prefix}_mfcc_{i}_mean'] = float(mean_mfcc[i])
        features[f'{prefix}_mfcc_{i}_std'] = float(std_mfcc[i])
        features[f'{prefix}_mfcc_{i}_delta'] = float(delta_mfcc[i])
    return features

# ================================================================
# MULTITAPER FFT EXTRACTION
# ================================================================
def _extract_multitaper_features(self, signal, prefix, fs, num_tapers=3):
    """
    Compute spectral descriptors using multitaper estimation (DPSS).
    Returns the same seven descriptors as _extract_windowed_fft, but far more stable.
    """
    signal = np.array(signal, dtype=np.float64)
    if len(signal) < 32:
        return self._fft_defaults(prefix)  # reuse existing zero dict

    window_size = int(2 * fs)
    window_size = max(16, min(window_size, 64))
    hop_size = window_size // 2

    from scipy.signal import windows
    tapers = windows.dpss(window_size, NW=num_tapers, return_ratios=False)
    # shape (num_tapers, window_size)

    peak_freqs, energies, entropies, band_lows, band_mids = [], [], [], [], []
    energy_centroids, energy_medoids = [], []

    for i in range(0, len(signal) - window_size, hop_size):
        chunk = signal[i:i+window_size]
        # Compute multitaper spectrum (average periodograms)
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

        # --- same descriptors as your original _extract_windowed_fft ---
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
        return self._fft_defaults(prefix)

    return {
        f'{prefix}_mt_peak_freq': float(np.mean(peak_freqs)),
        f'{prefix}_mt_spectral_energy': float(np.mean(energies)),
        f'{prefix}_mt_spectral_entropy': float(np.mean(entropies)),
        f'{prefix}_mt_band_low_0_2hz': float(np.mean(band_lows)),
        f'{prefix}_mt_band_mid_2_5hz': float(np.mean(band_mids)),
        f'{prefix}_mt_centroid': float(np.mean(energy_centroids)),
        f'{prefix}_mt_medoid': float(np.mean(energy_medoids)),
    }










if len(acc_x) > 20:

    # --- Magnitude (existing + new) ---
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_mag_active = self._get_active_segments(acc_mag)
    if len(acc_mag_active) > 20:
        # Keep your original windowed FFT
        features.update(self._extract_windowed_fft(acc_mag_active, 'acc', fs))
        # ADD MFCC and Multitaper
        features.update(self._extract_mfcc_features(acc_mag_active, 'acc_mag', fs))
        features.update(self._extract_multitaper_features(acc_mag_active, 'acc_mag', fs))

    # --- Per-axis: X, Y, Z (if you haven't implemented per-axis, add now) ---
    for axis_name, axis_signal in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z)]:
        axis_active = self._get_active_segments(axis_signal)
        if len(axis_active) > 20:
            # Per-axis windowed FFT (optional, you might already have it)
            # features.update(self._extract_windowed_fft(axis_active, axis_name, fs))
            # MFCC and Multitaper
            features.update(self._extract_mfcc_features(axis_active, axis_name, fs))
            features.update(self._extract_multitaper_features(axis_active, axis_name, fs))


























