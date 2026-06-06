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