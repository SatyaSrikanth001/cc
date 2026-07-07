def extract_hmog_acc_gyro(self, session):
    touches = session.get("touch_events", [])
    sensors = session.get("sensor_events", [])
    
    # ---------- logging ----------
    import datetime
    session_id = session.get("session_id", "unknown")
    log_msg = f"{datetime.datetime.now()}, session={session_id}, touches={len(touches)}, sensors={len(sensors)}"
    
    if len(touches) == 0 or len(sensors) == 0:
        with open("hmog_debug.log", "a") as f:
            f.write(log_msg + " -> EARLY RETURN {}\n")
        return {}
    
    # ---------- rest of original code (unchanged) ----------
    # Build arrays
    ts = []
    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    for e in sensors:
        t = e.get("timestamp")
        if t is None:
            continue
        ts.append(float(t))
        acc = e.get("accelerometer")
        gyro = e.get("gyroscope")
        acc_x.append(acc.get("x") if acc else np.nan)
        acc_y.append(acc.get("y") if acc else np.nan)
        acc_z.append(acc.get("z") if acc else np.nan)
        gyro_x.append(gyro.get("x") if gyro else np.nan)
        gyro_y.append(gyro.get("y") if gyro else np.nan)
        gyro_z.append(gyro.get("z") if gyro else np.nan)
    ts = np.array(ts, dtype=np.float64)
    acc_x = np.array(acc_x, dtype=np.float64)
    acc_y = np.array(acc_y, dtype=np.float64)
    acc_z = np.array(acc_z, dtype=np.float64)
    gyro_x = np.array(gyro_x, dtype=np.float64)
    gyro_y = np.array(gyro_y, dtype=np.float64)
    gyro_z = np.array(gyro_z, dtype=np.float64)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    signals = {
        "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z, "acc_mag": acc_mag,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z, "gyro_mag": gyro_mag
    }
    WINDOW = 100
    STABILITY_THRESH = 0.15
    feature_store = {k: [] for k in signals.keys()}
    from scipy.signal import stft
    def compute_stft(x, fs):
        x = x[~np.isnan(x)]
        if len(x) < 16:
            return 0.0, 0.0, 0.0
        x = x - np.mean(x)
        nperseg = min(16, len(x))
        f, _, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, boundary=None)
        mag = np.abs(Zxx)
        energy = np.sum(mag ** 2)
        p = mag / (np.sum(mag) + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        spectrum = np.sum(mag, axis=1)
        peak_freq = f[np.argmax(spectrum)]
        return energy, entropy, peak_freq

    skipped_no_time = 0
    processed_touches = 0
    short_window_count = 0
    for t in touches:
        t_start = t.get('touch_down_ts')
        t_end = t.get('touch_up_ts')
        if t_start is None or t_end is None:
            skipped_no_time += 1
            continue
        processed_touches += 1
        for key, signal in signals.items():
            before_idx = np.where((ts >= t_start - WINDOW) & (ts < t_start))[0]
            during_idx = np.where((ts >= t_start) & (ts <= t_end))[0]
            after_idx = np.where((ts > t_end) & (ts <= t_end + WINDOW))[0]
            if len(during_idx) < 2:
                short_window_count += 1
                continue
            before_vals = signal[before_idx]
            during_vals = signal[during_idx]
            after_vals = signal[after_idx]
            before_vals = before_vals[~np.isnan(before_vals)]
            during_vals = during_vals[~np.isnan(during_vals)]
            after_vals = after_vals[~np.isnan(after_vals)]
            if len(during_vals) == 0:
                continue
            if len(before_vals) == 0:
                before_vals = during_vals
            if len(after_vals) == 0:
                after_vals = during_vals
            avg_before = np.mean(before_vals)
            avg_during = np.mean(during_vals)
            avg_after = np.mean(after_vals)
            f1 = avg_during
            f2 = np.std(during_vals)
            f3 = avg_after - avg_before
            f4 = avg_during - avg_before
            f5 = np.max(during_vals) - avg_before
            if len(during_idx) > 0:
                peak_local = np.argmax(during_vals)
                t_peak = ts[during_idx][peak_local]
                t_max = t_peak - t_start
            else:
                t_max = 0.0
            stable_time = WINDOW
            for idx in after_idx:
                if abs(signal[idx] - avg_after) < STABILITY_THRESH:
                    stable_time = ts[idx] - t_end
                    break
            stft_idx = np.where((ts >= t_start - 200) & (ts <= t_end + 200))[0]
            stft_vals = signal[stft_idx]
            stft_vals = stft_vals[~np.isnan(stft_vals)]
            fs = self._estimate_sampling_rate(sensors)
            energy, entropy_val, peak_f = compute_stft(stft_vals, fs)
            feature_store[key].append({
                "mean": f1, "std": f2, "diff": f3, "net_change": f4,
                "max_change": f5, "t_max": t_max, "t_stability": stable_time,
                "stft_energy": energy, "stft_entropy": entropy_val,
                "stft_peak_freq": peak_f
            })
    # Aggregate
    final_features = {}
    for key, vals in feature_store.items():
        if len(vals) == 0:
            continue
        for k in vals[0].keys():
            arr = np.array([v[k] for v in vals])
            final_features[f"touch_LONG_{key}_{k}"] = float(np.mean(arr))

    # ---------- logging result ----------
    with open("hmog_debug.log", "a") as f:
        f.write(f"{datetime.datetime.now()}, session={session_id}, "
                f"processed_touches={processed_touches}, skipped_no_time={skipped_no_time}, "
                f"short_window_count={short_window_count}, final_features_count={len(final_features)}\n")
    return final_features
