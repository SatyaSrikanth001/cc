1. Typical preprocessing in sensor‑based behavioural authentication
A. Low‑pass filtering & noise removal
HMOG (Sitová et al.): Butterworth low‑pass filter, 10 Hz cutoff, because the sampling rate was 100 Hz and relevant motion is below 10 Hz.

SilentSense (Bo et al.): Low‑pass filter at 20 Hz (sampling 50 Hz).

Guo et al. (2019): Similar low‑pass filter, cutoff 10–20 Hz.

Your approach: You already use lowpass_filter with an adaptive cutoff (fs * 0.4, e.g., 6.4 Hz for 16 Hz). This is very reasonable for your low sampling rate.

B. Gravity removal (linear acceleration)
Almost all projects use a high‑pass filter (0.5–1 Hz) or a running‑average subtraction to separate the linear acceleration from the static gravity component.

Example: acc_linear = acc_raw - moving_average(acc_raw, window≈1 s).

This is critical because gravity hides subtle user‑specific motion patterns.

Your approach: You do this only inside extract_acc_features_mag_window (using uniform_filter1d to estimate gravity). However, your other extraction methods (extract_acc_features, extract_fft_features, etc.) do not remove gravity. They work on raw acceleration. This inconsistency can harm the discriminative power of the frequency‑based features because the large DC/gravity component dominates the spectrum.

C. Resampling & timestamp alignment
Most projects resample the sensor streams to a fixed, constant rate using linear interpolation.

This ensures all windows have the same number of samples, and FFT results are comparable across sessions.

Timestamps are sorted, and large gaps are handled by resampling or by inserting NaNs (then interpolated or ignored).

Your approach: You explicitly avoid resampling (_parse_sensors_no_interpolation). This can lead to slightly different effective sampling rates across sessions, making frequency features less stable. Many real systems accept a small interpolation error for the sake of uniformity.

D. Window‑based segmentation vs. session‑based
All papers use sliding windows (e.g., 2‑5 seconds, 50 % overlap) to compute features.

They either aggregate over the whole session (mean, std, percentiles) or keep the per‑window features for sequence modelling.

Your approach: You already slide windows in many extraction methods. Good.

E. Activity detection / idle removal
Similar to your _get_active_segments, they discard windows where motion energy is below a threshold.

But they do it on a window level, not by concatenating active samples. They either drop the window entirely or set its features to zero.

Your approach: Your sample‑concatenation method is different; I recommended switching to window‑based gating (as we discussed). That would match the literature.

F. Missing value handling & outlier removal
Usually, they remove sensor readings with obviously invalid values (e.g., NaN, Inf, extreme outliers > 6σ).

Missing values within a window are interpolated; if too many are missing, the window is discarded.

Your approach: You fill NaNs with median or zero. Median is reasonable, but zero can be problematic if the feature is not zero‑centered. Outlier removal is not explicitly done.

G. Normalisation / scaling
Per‑user z‑score normalisation (mean=0, std=1) is standard, applied after feature extraction and before classifier training.

Some also apply a per‑session normalisation (e.g., divide by the session’s own std) before computing global features, to reduce between‑session variability.

Your approach: You use StandardScaler inside the OCSVM training. That’s correct.
