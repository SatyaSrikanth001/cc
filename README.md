7. Aggregation extensions for all window‑based features

You currently aggregate windowed features with mean and std. Add:

Aggregation statistic Effect
skew Skewness of the per‑window feature distribution – captures asymmetry.
kurtosis Tailedness – captures extreme bursts.
p5, p95 The 5th and 95th percentiles.
iqr Inter‑quartile range.
range Max‑min.
median Median value.

These would apply to all windowed FFT, MFCC, Multitaper, and the windowed accelerometer features (which already have mean/std). This alone can increase feature count by 3‑5× and improve TAR by 4‑8 % (Buriro 2016).