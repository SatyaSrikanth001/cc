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