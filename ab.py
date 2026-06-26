1. How scaling is done in your current project
A. Per‑user StandardScaler (z‑score normalisation)
Inside the SessionOCSVM.train() method:

python
self.scaler = StandardScaler()
X_train_scaled = self.scaler.fit_transform(X_train)
The scaler is fitted only on the user’s genuine training data.

It transforms each feature to have mean = 0 and standard deviation = 1 for that user.

The same scaler is later applied to the test data of that user.

This is correct for a one‑class setting – it ensures the user’s normal behaviour is centred and scaled, so the OCSVM learns a boundary that is not biased by large‑magnitude features.

B. No global scaling across users
Before training, your features are not globally scaled across the entire population.

The raw feature values (e.g., MFCC coefficients, spectral energies) can have vastly different ranges (e.g., acc_fft_spectral_energy might be 0–500, while gyro_x_std might be 0.01–0.1).

Because the scaler is per user, these inter‑feature range differences remain. The OCSVM’s RBF kernel uses Euclidean distances; a feature with a large range will dominate the distance metric.

Consequence: Features with larger variance across the user’s sessions will influence the OCSVM more, even if they are less discriminative.

C. Implicit scaling by the RBF kernel’s gamma
The gamma parameter of the RBF kernel acts as an inverse length scale. A small gamma (like 0.00055) means the kernel is very wide, making the model less sensitive to individual feature differences.

Tuning gamma via LOUO partially compensates for the varying feature scales, but it’s a single global parameter applied uniformly to all features. An ideal solution would give each feature its own scaling, but that’s not possible with a single OCSVM.

2. How scaling is handled in advanced projects
A. Global feature normalisation before model training
Most behavioural authentication projects first normalise each feature globally (across all users or across a large development set) to a common range, then apply per‑user scaling.

Common global normalisation methods:

Min‑Max scaling: Scale each feature to [0, 1] or [-1, 1] using the min/max of the entire training dataset (all users).

Z‑score global: Compute global mean and std per feature (again across all users), transform all users with these values.

Robust scaling: Use the median and IQR of each feature globally, then per‑user scaling if needed.

After this global step, the per‑user StandardScaler is applied. This two‑stage process ensures that all features start on a level playing field and the per‑user scaling only captures individual deviations.

Example from HMOG (Sitová et al.):
They first applied a global z‑score normalisation (per feature, across all users) to remove the effect of wildly different feature ranges. Then, for each user, they further normalised using the user’s own training mean and standard deviation.

Example from commercial systems (BioCatch):
They use Quantile Transformer or RobustScaler to make features follow a roughly Gaussian distribution, then apply per‑user scaling. This reduces the impact of outliers.

B. Per‑session normalisation
Some projects first normalise each session individually (e.g., divide by the session’s total energy or by its mean) to remove between‑session amplitude variations (e.g., a user might hold the phone more firmly one day). After that, global or per‑user scaling is applied.

Example: Before computing touch velocity, they normalise touch coordinates by screen size (which you already do). Similarly, they normalise accelerometer readings by subtracting the session’s mean (DC removal) and sometimes by dividing by the session’s standard deviation.

C. Feature‑wise scaling based on importance
In advanced pipelines, they sometimes weight features by their importance score (e.g., RF importance) before scaling, effectively stretching discriminative features. This is rare but shows up in some commercial patents.

3. What you can improve in your scaling pipeline
A. Add a global feature normalisation step
Before training each user’s OCSVM, you could apply one of the following:

Global StandardScaler: Fit a single StandardScaler on all genuine training data from all users (development set). Transform every user’s data with it first, then apply the per‑user scaler. This ensures all features have roughly the same variance across the entire population.

RobustScaler (global): Use median and IQR per feature globally – more robust to outliers.

After this, your per‑user scaler will only capture how that user deviates from the population, making the model more sensitive to unique behaviour.

B. Remove or reduce the influence of very large range features
Check the variance of each feature across the development users’ genuine data. Features with extremely high variance (e.g., acc_fft_spectral_energy) may need to be log‑transformed (e.g., log1p) before scaling. Many papers do this for energy/entropy features.

C. Ensure consistency between training and test scaling
Your current test scaling only uses the scaler fitted on training data – that’s perfect. But if you add a global scaler, it must be fitted only on the development users’ training data and then applied to test users.

D. Tuning gamma in conjunction with scaling
After global scaling, the optimal gamma will likely change because the feature space is better balanced. You already re‑tune via LOUO, so this is automatically handled.

4. Concrete implementation recommendation
Add a method _global_scale to your pipeline that runs once (on the 13 dev users) and saves a global scaler. Then, in your SessionOCSVM training, load that scaler and apply it before the per‑user scaler. The per‑user scaler then fits on the globally‑scaled data.

This two‑stage scaling is exactly what the literature recommends and will make your OCSVM far more effective at using all features.
