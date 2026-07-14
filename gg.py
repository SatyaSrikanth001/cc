<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Architecture – Behavioral Biometrics Blueprint</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1e2b3c;
            max-width: 1100px;
            margin: 40px auto;
            padding: 40px 60px;
            background: white;
        }
        @media print {
            body { margin: 20px auto; padding: 20px 40px; }
            .page-break { page-break-before: always; }
        }
        h1 {
            font-size: 24pt;
            color: #003366;
            border-bottom: 3px solid #003366;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h2 {
            font-size: 17pt;
            color: #004080;
            margin-top: 28px;
            border-bottom: 1px solid #cccccc;
            padding-bottom: 4px;
        }
        h3 {
            font-size: 14pt;
            color: #005599;
            margin-top: 22px;
        }
        h4 {
            font-size: 12pt;
            color: #006699;
            margin-top: 18px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 14px 0 20px 0;
            font-size: 10pt;
        }
        th, td {
            border: 1px solid #aaaaaa;
            padding: 6px 10px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #e6f0fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        code {
            background-color: #f0f2f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }
        pre {
            background-color: #f0f2f5;
            padding: 14px 18px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 10pt;
            border-left: 4px solid #003366;
            margin: 10px 0 16px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .box {
            background-color: #fef9e7;
            border-left: 6px solid #f39c12;
            padding: 12px 18px;
            margin: 14px 0;
            border-radius: 4px;
        }
        .box strong { color: #e67e22; }
        .footer {
            text-align: center;
            font-size: 9pt;
            color: #888888;
            margin-top: 40px;
            border-top: 1px solid #dddddd;
            padding-top: 12px;
        }
        .pipeline-arrow {
            text-align: center;
            font-size: 18pt;
            color: #004080;
            margin: 6px 0;
        }
        .pipeline-box {
            background-color: #f0f5fa;
            border: 1px solid #b0c4de;
            padding: 16px 20px;
            margin: 10px 0;
            border-radius: 6px;
        }
        .pipeline-box strong {
            color: #003366;
            font-size: 12pt;
        }
        ul, ol { padding-left: 24px; margin: 8px 0 12px 0; }
        li { margin-bottom: 4px; }
    </style>
</head>
<body>

<!-- ============================================================ -->
<!-- DOCUMENT HEADER -->
<!-- ============================================================ -->

<h1>MASTER ARCHITECTURE &amp; CONTEXT-TRANSFER BLUEPRINT</h1>
<p><strong>Document Classification:</strong> Internal – Production Readiness Review<br>
<strong>Version:</strong> 2.0 FINAL<br>
<strong>Date:</strong> 2026-07-14<br>
<strong>Document Owner:</strong> Principal ML Architect<br>
<strong>Target Audience:</strong> New Senior Engineering Team (AI Session)</p>

<hr>

<!-- ============================================================ -->
<!-- SECTION 1: THE MISSION & CONTEXT -->
<!-- ============================================================ -->

<h2>SECTION 1: THE MISSION &amp; CONTEXT</h2>

<h3>1.1 The Dataset Paradox</h3>
<p>You are inheriting a behavioral biometrics feature extraction pipeline that produces <strong>1,300+ hand-crafted features</strong> from mobile banking session data. However, the available development dataset consists of:</p>

<table>
    <tr><th>Data Component</th><th>Quantity</th></tr>
    <tr><td>Physical users</td><td>30</td></tr>
    <tr><td>Genuine sessions per user</td><td>~75</td></tr>
    <tr><td>Impostor sessions per user</td><td>~55 (from 10+ attackers)</td></tr>
    <tr><td>Total sessions</td><td>~3,900</td></tr>
    <tr><td>Features per session</td><td>1,300+</td></tr>
    <tr><td>Devices</td><td>30 (one per user, each with unique hardware)</td></tr>
</table>

<h3>1.2 The Catastrophic Risk</h3>
<p><strong>The fatal flaw:</strong> With 1,300 features and only 30 devices, standard feature selection will inevitably <strong>overfit to device hardware signatures</strong> (sensor jitter, screen-digitizer latency, accelerometer DC offsets, gyroscope bias) rather than human behavioral biometrics.</p>
<p><strong>Why this matters:</strong> A feature that models the phone instead of the human will fail catastrophically upon deployment to <strong>lakhs of unseen devices</strong> with different hardware characteristics. This is not a theoretical concern—it is the single most common cause of failure in production behavioral biometrics systems.</p>

<h3>1.3 The Target Classifier &amp; Metric</h3>
<table>
    <tr><th>Component</th><th>Specification</th></tr>
    <tr><td>Classifier</td><td>Per-user One-Class SVM (OCSVM) with RBF kernel</td></tr>
    <tr><td>Primary Metric</td><td>Target Acceptance Rate (TAR) at False Acceptance Rate (FAR) = 0.001</td></tr>
    <tr><td>Secondary Metric</td><td>Equal Error Rate (EER) for comparative reporting</td></tr>
    <tr><td>Success Criterion</td><td>TAR ≥ 0.80 at FAR = 0.001 on held-out LOGO validation</td></tr>
</table>

<h3>1.4 The Tech Stack (Strictly Enforced)</h3>
<table>
    <tr><th>Category</th><th>Allowed Libraries</th><th>Forbidden</th></tr>
    <tr><td>ML Framework</td><td>Scikit-Learn</td><td>PyTorch, TensorFlow, Keras</td></tr>
    <tr><td>Signal Processing</td><td>SciPy, PyWavelets</td><td>Custom C++/CUDA</td></tr>
    <tr><td>Stats</td><td>Statsmodels</td><td>Proprietary stats packages</td></tr>
    <tr><td>Computation</td><td>NumPy, Pandas</td><td>GPU-only libraries</td></tr>
    <tr><td>Visualization (optional)</td><td>Matplotlib, Seaborn</td><td>PowerBI, Tableau</td></tr>
</table>
<p><strong>No deep learning. No PyTorch. No TensorFlow.</strong> The ecosystem is strictly Scikit-Learn, SciPy, PyWavelets, Statsmodels, and their dependencies.</p>

<h3>1.5 Development vs. Production Context</h3>
<table>
    <tr><th>Aspect</th><th>Development</th><th>Production</th></tr>
    <tr><td>Devices</td><td>30 known devices</td><td>Lakhs of unseen devices</td></tr>
    <tr><td>Users</td><td>30 known users</td><td>Tens of thousands of new users</td></tr>
    <tr><td>Attacks</td><td>10+ known attackers</td><td>Unknown adversarial patterns</td></tr>
    <tr><td>Data volume</td><td>~3,900 sessions</td><td>Millions of sessions</td></tr>
    <tr><td>Hardware</td><td>Controlled environment</td><td>Heterogeneous and uncontrolled</td></tr>
</table>
<p><strong>The selected features MUST be device-invariant and transferable to new hardware without retraining.</strong></p>

<hr>

<!-- ============================================================ -->
<!-- SECTION 2: THE FINALIZED 4-TIER PIPELINE ARCHITECTURE -->
<!-- ============================================================ -->

<h2>SECTION 2: THE FINALIZED 4-TIER PIPELINE ARCHITECTURE</h2>

<h3>2.1 Pipeline Overview</h3>

<div class="pipeline-box">
    <div style="text-align:center; font-size:14pt; font-weight:bold; color:#003366;">RAW FEATURES (1,300+)</div>
    <div class="pipeline-arrow">▼</div>
    <div style="background:#e6f0fa; padding:12px; border-radius:4px;">
        <strong>TIER 1: PRE-SELECTION &amp; HARDWARE DISENTANGLEMENT</strong><br>
        • Within-Device Point-Biserial Correlation Filter<br>
        • Cross-Device Discriminability Aggregation<br>
        <em>Output: ~800-900 features</em>
    </div>
    <div class="pipeline-arrow">▼</div>
    <div style="background:#e6f0fa; padding:12px; border-radius:4px;">
        <strong>TIER 2: GLOBAL STABILITY SELECTION</strong><br>
        • LOUO Random Forest Feature Importance<br>
        • Fisher Score (within-user stability)<br>
        • Boruta + SHAP (non-linear importance)<br>
        • Consensus Ranking<br>
        <em>Output: ~100-120 features</em>
    </div>
    <div class="pipeline-arrow">▼</div>
    <div style="background:#e6f0fa; padding:12px; border-radius:4px;">
        <strong>TIER 3: NON-LINEAR DISCRIMINATIVE GATING</strong><br>
        • Distance Correlation Clustering<br>
        • Medoid Selection<br>
        • JMI (Joint Mutual Information) Feature Selection<br>
        <em>Output: ~60-80 features</em>
    </div>
    <div class="pipeline-arrow">▼</div>
    <div style="background:#e6f0fa; padding:12px; border-radius:4px;">
        <strong>TIER 4: OCSVM-GUIDED WRAPPER ELIMINATION</strong><br>
        • Backward Elimination with OCSVM Performance Objective<br>
        • Smoothness Objective: Mean TAR - λ * Std TAR<br>
        • Cross-Validation to Prevent Memorization<br>
        <em>Output: 50-80 FINAL features</em>
    </div>
    <div class="pipeline-arrow">▼</div>
    <div style="text-align:center; font-size:14pt; font-weight:bold; color:#003366;">FINAL FEATURE SUBSET (50-80)</div>
</div>

<h3>2.2 TIER 1: PRE-SELECTION &amp; HARDWARE DISENTANGLEMENT</h3>

<h4>2.2.1 The Core Insight (MANDATORY)</h4>
<div class="box">
    <p><strong>CRITICAL:</strong> Since each device belongs to a unique user, <strong>user identity and device identity are perfectly confounded</strong>. You CANNOT use a shadow classifier that predicts Device-ID, because it will actually be predicting User-ID and you will discard your strongest behavioral signals.</p>
    <p><strong>CORRECTED APPROACH:</strong> Use the impostor sessions on each device to identify features that consistently separate the genuine user from impostors <strong>within each device</strong>. This isolates human behavioral signal from hardware noise.</p>
</div>

<h4>2.2.2 Within-Device Point-Biserial Correlation</h4>
<p><strong>Algorithm:</strong></p>
<p>For each device <code>d</code> (1..30) and each feature <code>f</code> (1..1300):</p>
<ol>
    <li>Extract all sessions from device <code>d</code></li>
    <li>Label each session:
        <ul>
            <li><code>y = 1</code> if session belongs to the genuine owner</li>
            <li><code>y = 0</code> if session belongs to an impostor</li>
        </ul>
    </li>
    <li>Compute the point-biserial correlation:
        <pre>r_pb(f, d) = (M1 - M0) / s * sqrt( n1*n0 / (n*(n-1)) )</pre>
        where:
        <ul>
            <li><code>M1</code> = mean of feature <code>f</code> for genuine sessions on device <code>d</code></li>
            <li><code>M0</code> = mean of feature <code>f</code> for impostor sessions on device <code>d</code></li>
            <li><code>s</code> = pooled standard deviation</li>
            <li><code>n1, n0</code> = number of genuine and impostor sessions</li>
        </ul>
    </li>
    <li>Store <code>|r_pb(f, d)|</code></li>
</ol>

<h4>2.2.3 Cross-Device Aggregation</h4>
<p>For each feature <code>f</code>:</p>
<ul>
    <li>Compute the <strong>median</strong> of <code>|r_pb(f, d)|</code> across all 30 devices:
        <pre>MedianDisc(f) = median_{d in 1..30} |r_pb(f, d)|</pre>
    </li>
</ul>
<p><strong>Decision Rule:</strong></p>
<ul>
    <li>If <code>MedianDisc(f) &lt; 0.10</code> → <strong>DROP</strong> (weak discriminability across all devices)</li>
    <li>If <code>MedianDisc(f) ≥ 0.10</code> → <strong>KEEP</strong> (consistently separates genuine from impostors)</li>
</ul>

<h4>2.2.4 Optional: Cross-Device Ranking Consistency</h4>
<p>For features that pass the median threshold, compute:</p>
<pre>RankConsistency(f) = std( rank_d(f) )</pre>
<ul>
    <li>Lower std = more consistent ranking across devices = more robust feature</li>
    <li>Use this to break ties when selecting features</li>
</ul>

<p><strong>Implementation Guide:</strong></p>
<pre>
class WithinDeviceDiscriminabilityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, point_biserial_threshold=0.10, use_rank_consistency=True):
        self.threshold = point_biserial_threshold
        self.use_rank_consistency = use_rank_consistency
        self.feature_mask_ = None

    def fit(self, X, y, device_ids, is_genuine):
        # Implementation logic here
        return self
</pre>
<p><strong>Expected Output:</strong> 800-900 features retained.</p>

<h3>2.3 TIER 2: GLOBAL STABILITY SELECTION</h3>
<p><strong>PRINCIPLE:</strong> This tier is <strong>already correct</strong> in your existing pipeline. It must be preserved.</p>

<h4>2.3.1 Three-Method Consensus Ranking</h4>
<table>
    <tr><th>Method</th><th>Purpose</th><th>Mathematical Foundation</th></tr>
    <tr><td>LOUO Random Forest</td><td>Direct genuine-impostor discriminability</td><td>Feature importance from RF trained on 29 users, tested on 1</td></tr>
    <tr><td>Fisher Score</td><td>Within-user stability</td><td>between_user_variance / within_user_variance</td></tr>
    <tr><td>Boruta + SHAP</td><td>Non-linear importance</td><td>SHAP values from Boruta-selected features</td></tr>
</table>

<h4>2.3.2 Consensus Logic</h4>
<p>For each feature <code>f</code>:</p>
<ol>
    <li>Compute its rank in each of the three methods: <code>R_RF(f), R_Fisher(f), R_SHAP(f)</code></li>
    <li>Calculate the average rank:
        <pre>R_avg(f) = (R_RF(f) + R_Fisher(f) + R_SHAP(f)) / 3</pre>
    </li>
    <li>Sort features by <code>R_avg(f)</code> (ascending)</li>
    <li><strong>Select the top 100 features</strong> as the starting set for Tier 3</li>
</ol>

<p><strong>Implementation Guide:</strong></p>
<pre>
class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.selected_features_ = None

    def fit(self, X, y, user_ids):
        # Implementation logic here
        return self
</pre>
<p><strong>Expected Output:</strong> 100-120 features selected.</p>

<h3>2.4 TIER 3: NON-LINEAR DISCRIMINATIVE GATING</h3>

<h4>2.4.1 Distance Correlation Clustering</h4>
<p><strong>Objective:</strong> Group features that are highly dependent (linearly or non-linearly) to reduce redundancy.</p>
<p><strong>Algorithm:</strong></p>
<ol>
    <li>Compute the <strong>distance correlation matrix</strong> <code>D</code> of size 100×100:
        <pre>dCor(f_a, f_b) = dCov(f_a, f_b) / sqrt( dCov(f_a, f_a) * dCov(f_b, f_b) )</pre>
    </li>
    <li>Use <code>1 - dCor</code> as the dissimilarity metric.</li>
    <li>Apply <strong>Affinity Propagation</strong> clustering (or hierarchical clustering with average linkage).</li>
    <li><strong>Cluster Retention Rule:</strong>
        <ul>
            <li>Keep clusters with intra-cluster average dCor ≥ 0.60</li>
            <li>For each kept cluster, select the <strong>medoid</strong> (feature with highest average dCor to other cluster members)</li>
        </ul>
    </li>
</ol>

<h4>2.4.2 JMI (Joint Mutual Information) Selection</h4>
<p><strong>Objective:</strong> Select features that are maximally complementary for OCSVM.</p>
<p><strong>Algorithm:</strong></p>
<p>Let <code>S</code> be the set of already selected features, and <code>Y</code> be the target (User-ID, multi-class).</p>
<p>For each candidate feature <code>f ∈ F \ S</code>, compute the <strong>JMI score</strong>:</p>
<pre>JMI(f) = I(f ; Y) + sum_{s in S} I(f ; Y | s)</pre>
<p>where:</p>
<ul>
    <li><code>I(f ; Y)</code> = mutual information between feature <code>f</code> and User-ID</li>
    <li><code>I(f ; Y | s)</code> = conditional mutual information given selected feature <code>s</code></li>
</ul>
<p><strong>Decision:</strong></p>
<ul>
    <li>Start with empty <code>S</code></li>
    <li>Repeatedly add the feature with highest JMI score</li>
    <li>Stop when <code>|S| = 60</code> (or until JMI increment &lt; 0.01)</li>
</ul>

<p><strong>Implementation Guide:</strong></p>
<pre>
class JMIFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=60, k_neighbors=5):
        self.n_features = n_features
        self.k_neighbors = k_neighbors
        self.selected_features_ = None

    def fit(self, X, y):
        # Implementation logic here
        return self
</pre>
<p><strong>Expected Output:</strong> 60-80 features selected.</p>

<h3>2.5 TIER 4: OCSVM-GUIDED WRAPPER ELIMINATION</h3>

<h4>2.5.1 The Objective Function (CRITICAL)</h4>
<div class="box">
    <p><strong>DO NOT use FAR ≤ 0.001 as a hard constraint.</strong> With only ~45 impostor sessions per user, this constraint is impossible and will cause the wrapper to reject all features or behave randomly.</p>
</div>

<p><strong>CORRECTED OBJECTIVE:</strong></p>
<p>For a candidate feature subset <code>S</code>:</p>
<ol>
    <li>For each user <code>u</code> in 1..30:
        <ul>
            <li>Train OCSVM on user <code>u</code>'s genuine sessions (using LOUO: train on other 29 users)</li>
            <li>Test on:
                <ul>
                    <li>User <code>u</code>'s held-out genuine sessions → compute <code>TAR_u</code></li>
                    <li>Impostor sessions targeting user <code>u</code> → compute <code>FAR_u</code></li>
                </ul>
            </li>
            <li>Record <code>TAR_u</code> (at threshold = 0, or user-specific percentile to keep FRR &lt; 0.1)</li>
        </ul>
    </li>
    <li>Compute the <strong>Smoothness Objective</strong>:
        <pre>Score(S) = MeanTAR(S) - λ * StdTAR(S)</pre>
        where:
        <ul>
            <li><code>MeanTAR(S) = (1/30) * sum_{u=1}^{30} TAR_u</code></li>
            <li><code>StdTAR(S) = std(TAR_u)</code></li>
            <li><code>λ = 0.5</code> (penalizes high variance)</li>
            <li><code>Score(S) ∈ [0, 1]</code></li>
        </ul>
    </li>
    <li><strong>Maximize Score(S)</strong> — this selects subsets that give high and consistent OCSVM performance across all users.</li>
</ol>

<h4>2.5.2 Backward Elimination Algorithm</h4>
<p>Starting from the 60-80 features selected in Tier 3:</p>
<ol>
    <li>Let <code>S</code> = current feature set (initial = Tier 3 output)</li>
    <li><strong>Repeat</strong> until no improvement:
        <ul>
            <li>For each feature <code>f</code> in <code>S</code>:
                <ul>
                    <li>Let <code>S' = S \ {f}</code></li>
                    <li>Compute <code>Score(S')</code> using 5-fold LOUO (stratified by user)</li>
                    <li>If <code>Score(S') ≥ Score(S)</code>:
                        <ul>
                            <li>Remove <code>f</code> (feature was harmful or neutral)</li>
                            <li>Mark that an improvement was made</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li>If no feature was removed in a full pass → <strong>convergence</strong></li>
        </ul>
    </li>
    <li>Return the final set <code>S</code></li>
</ol>

<h4>2.5.3 Cross-Validation to Prevent Memorization</h4>
<p>For each evaluation of <code>Score(S)</code>:</p>
<ol>
    <li>Split into <strong>5 folds</strong> stratified by user (each fold contains all users)</li>
    <li>For each fold:
        <ul>
            <li>Train on folds 1-4</li>
            <li>Evaluate on fold 5</li>
            <li>Compute TAR and FAR</li>
        </ul>
    </li>
    <li>Average TAR across folds → <code>MeanTAR</code></li>
    <li>Standard deviation of TAR across folds → <code>StdTAR</code></li>
    <li>Return <code>Score = MeanTAR - 0.5 * StdTAR</code></li>
</ol>

<p><strong>Implementation Guide:</strong></p>
<pre>
class OCSVMWrapperOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, cv_folds=5, lambda_variance=0.5,
                 far_constraint=0.001, max_features=80):
        self.cv_folds = cv_folds
        self.lambda_variance = lambda_variance
        self.max_features = max_features
        self.selected_features_ = None
        self.final_score_ = None

    def fit(self, X, y, user_ids, is_genuine, impostor_user_ids):
        # Implementation logic here
        return self
</pre>
<p><strong>Expected Output:</strong> 50-80 FINAL features.</p>

<hr>

<!-- ============================================================ -->
<!-- SECTION 3: LOGO CROSS-VALIDATION PROTOCOL -->
<!-- ============================================================ -->

<h2>SECTION 3: THE LOGO CROSS-VALIDATION PROTOCOL</h2>

<h3>3.1 What LOGO Validates</h3>
<p><strong>Leave-One-Group-Out (LOGO)</strong> is the gold standard for proving that features generalize to unseen devices and unseen users.</p>

<table>
    <tr><th>Validation Type</th><th>What is Held Out</th><th>What It Proves</th></tr>
    <tr><td>Device-Level LOGO</td><td>All sessions from one device (both genuine and impostor)</td><td>Features generalize to <strong>new hardware</strong></td></tr>
    <tr><td>User-Level LOGO</td><td>All sessions from one user (both genuine and impostor)</td><td>Features generalize to <strong>new users</strong></td></tr>
</table>

<h3>3.2 Device-Level LOGO Protocol</h3>
<p><strong>Procedure:</strong></p>
<ol>
    <li>For each device <code>d</code> in 1..30:
        <ul>
            <li><strong>Training Set:</strong> All sessions from devices ≠ <code>d</code></li>
            <li><strong>Validation Set:</strong> All sessions from device <code>d</code></li>
        </ul>
    </li>
    <li>Run the <strong>ENTIRE 4-Tier Pipeline</strong> on the training set ONLY.</li>
    <li>Train per-user OCSVMs on the training set.</li>
    <li>Evaluate on the validation set (device <code>d</code>):
        <ul>
            <li>For each user <code>u</code> on device <code>d</code>:
                <ul>
                    <li>Test genuine sessions of user <code>u</code> → compute TAR</li>
                    <li>Test impostor sessions targeting user <code>u</code> → compute FAR</li>
                </ul>
            </li>
        </ul>
    </li>
    <li>Compute the <strong>Generalization Degradation Penalty (GDP)</strong>:
        <pre>GDP = (TAR_train - TAR_valid) / TAR_train</pre>
        where TAR_train is the average TAR from cross-validation (from Tier 4).
    </li>
    <li><strong>Decision Rules:</strong>
        <ul>
            <li>If GDP > 0.15 (15% degradation) → <strong>FAIL</strong> (return to Tier 1)</li>
            <li>If GDP ≤ 0.15 → <strong>PASS</strong></li>
        </ul>
    </li>
</ol>

<h3>3.3 User-Level LOGO Protocol (LOUO)</h3>
<p><strong>Procedure:</strong></p>
<ol>
    <li>For each user <code>u</code> in 1..30:
        <ul>
            <li><strong>Training Set:</strong> All sessions from users ≠ <code>u</code></li>
            <li><strong>Validation Set:</strong> All sessions from user <code>u</code></li>
        </ul>
    </li>
    <li>Run the <strong>ENTIRE 4-Tier Pipeline</strong> on the training set ONLY.</li>
    <li>Train per-user OCSVMs on the training set.</li>
    <li>Evaluate on the validation set (user <code>u</code>):
        <ul>
            <li>Test genuine sessions of user <code>u</code> → compute TAR</li>
            <li>Test impostor sessions targeting user <code>u</code> → compute FAR</li>
        </ul>
    </li>
    <li><strong>Decision Rules:</strong>
        <ul>
            <li>If TAR &lt; 0.70 at FAR = 0.001 → <strong>FAIL</strong></li>
            <li>If TAR ≥ 0.70 → <strong>PASS</strong></li>
        </ul>
    </li>
</ol>

<h3>3.4 Combined Validation Matrix</h3>
<table>
    <tr><th>Validation</th><th>What it Tests</th><th>Pass Criterion</th></tr>
    <tr><td>Device LOGO</td><td>Hardware generalization</td><td>GDP ≤ 0.15</td></tr>
    <tr><td>User LOGO</td><td>User generalization</td><td>TAR ≥ 0.70 at FAR=0.001</td></tr>
    <tr><td>Full Pipeline</td><td>Complete robustness</td><td>Both passes</td></tr>
</table>

<hr>

<!-- ============================================================ -->
<!-- SECTION 4: STRICT EXECUTION RULES -->
<!-- ============================================================ -->

<h2>SECTION 4: STRICT EXECUTION RULES FOR THE NEW AI (THE CODING GUIDELINES)</h2>

<h3>4.1 MANDATORY: Interactive Coding Process</h3>
<p>You are <strong>NOT permitted</strong> to output the entire pipeline in a single massive script.</p>
<p>You <strong>WILL</strong> act as an interactive coding pair with the user:</p>
<ul>
    <li><strong>Wait</strong> for the user's prompt to begin coding.</li>
    <li>Write code <strong>ONE MODULE AT A TIME</strong>.</li>
    <li>After each module, explain what you've written and ask: <em>"Shall I proceed to the next module?"</em></li>
    <li><strong>Do NOT proceed</strong> until the user gives explicit approval.</li>
</ul>

<h3>4.2 MANDATORY: Modular Structure</h3>
<p>All code MUST be structured as <strong>Scikit-Learn Custom Transformers</strong>:</p>

<table>
    <tr><th>Module</th><th>Class Name</th><th>Purpose</th></tr>
    <tr><td>1</td><td>WithinDeviceDiscriminabilityFilter</td><td>Tier 1: Hardware decoupling</td></tr>
    <tr><td>2</td><td>ConsensusFeatureSelector</td><td>Tier 2: Global stability selection</td></tr>
    <tr><td>3</td><td>JMIFeatureSelector</td><td>Tier 3: JMI selection</td></tr>
    <tr><td>4</td><td>OCSVMWrapperOptimizer</td><td>Tier 4: OCSVM-guided elimination</td></tr>
    <tr><td>5</td><td>LOGOValidator</td><td>Cross-validation protocol</td></tr>
</table>

<p>Each transformer <strong>MUST</strong>:</p>
<ul>
    <li>Inherit from <code>BaseEstimator</code> and <code>TransformerMixin</code></li>
    <li>Implement <code>fit()</code> and <code>transform()</code></li>
    <li>Store selected feature indices in <code>self.selected_features_</code></li>
    <li>Support <code>get_feature_names_out()</code> (for integration with Scikit-Learn pipelines)</li>
</ul>

<h3>4.3 MANDATORY: Mathematical Completeness</h3>
<p>For each transformer, you <strong>MUST</strong> include:</p>
<ul>
    <li><strong>Docstring</strong> explaining the mathematical algorithm</li>
    <li><strong>Parameter validation</strong> (raise errors for invalid inputs)</li>
    <li><strong>Numerical stability</strong> (epsilon constants, clipping, NaN handling)</li>
    <li><strong>Logging</strong> (using <code>logging.getLogger(__name__)</code>)</li>
</ul>

<h3>4.4 MANDATORY: Testing Requirements</h3>
<p>Each module MUST include a simple unit test using synthetic data:</p>
<pre>
def test_module_name():
    # Generate synthetic data
    # Apply transformer
    # Assert expected behavior
</pre>

<h3>4.5 MANDATORY: Code Style</h3>
<ul>
    <li><strong>PEP 8</strong> compliance</li>
    <li><strong>Type hints</strong> for all function parameters and returns</li>
    <li><strong>No external dependencies</strong> beyond Scikit-Learn, SciPy, Statsmodels, NumPy, Pandas</li>
    <li><strong>No deep learning</strong> (PyTorch, TensorFlow, Keras strictly forbidden)</li>
</ul>

<h3>4.6 FORBIDDEN: What NOT to Do</h3>
<ul>
    <li><strong>DO NOT</strong> generate code that assumes multiple genuine users per device</li>
    <li><strong>DO NOT</strong> use shadow classifiers that predict Device-ID</li>
    <li><strong>DO NOT</strong> enforce the condition FAR ≤ 0.001 in the wrapper objective.</li>
    <li><strong>DO NOT</strong> use L1-logistic regression bootstrapping for stability selection</li>
    <li><strong>DO NOT</strong> output the full pipeline in one response</li>
</ul>

<h3>4.7 The Module Order (Execution Sequence)</h3>
<p>You will build the pipeline in this exact order:</p>
<ol>
    <li><strong>Module 1:</strong> WithinDeviceDiscriminabilityFilter (Tier 1)</li>
    <li><strong>Module 2:</strong> ConsensusFeatureSelector (Tier 2)</li>
    <li><strong>Module 3:</strong> JMIFeatureSelector (Tier 3)</li>
    <li><strong>Module 4:</strong> OCSVMWrapperOptimizer (Tier 4)</li>
    <li><strong>Module 5:</strong> LOGOValidator (Validation Protocol)</li>
    <li><strong>Module 6:</strong> Full Pipeline Integration (Combining all modules into a single Scikit-Learn Pipeline)</li>
</ol>
<p><strong>Do NOT skip ahead. Wait for user approval after each module.</strong></p>

<hr>

<!-- ============================================================ -->
<!-- SECTION 5: FINAL VERIFICATION CHECKLIST -->
<!-- ============================================================ -->

<h2>SECTION 5: FINAL VERIFICATION CHECKLIST</h2>
<p>Before you consider this pipeline complete, verify:</p>

<table>
    <tr><th>Item</th><th>Status</th></tr>
    <tr><td>Tier 1 uses within-device point-biserial correlation</td><td>☐</td></tr>
    <tr><td>Tier 1 threshold = 0.10 for median discriminability</td><td>☐</td></tr>
    <tr><td>Tier 2 uses consensus of RF LOUO + Fisher + Boruta SHAP</td><td>☐</td></tr>
    <tr><td>Tier 3 uses Distance Correlation + JMI</td><td>☐</td></tr>
    <tr><td>Tier 4 uses the formula: Score = Mean TAR – 0.5 * Std TAR (under the NOT FAR constraint)</td><td>☐</td></tr>
    <tr><td>LOGO validation has both device and user holdout</td><td>☐</td></tr>
    <tr><td>GDP ≤ 0.15 is the pass criterion</td><td>☐</td></tr>
    <tr><td>All transformers are Scikit-Learn custom transformers</td><td>☐</td></tr>
    <tr><td>No deep learning libraries are imported</td><td>☐</td></tr>
    <tr><td>Each module includes unit tests.</td><td>☐</td></tr>
</table>

<hr>

<!-- ============================================================ -->
<!-- FINAL TASK SUMMARY -->
<!-- ============================================================ -->

<h2>YOUR TASK (SUMMARY)</h2>
<p>You are the <strong>new Senior Engineering Team</strong> receiving this project. You will build the Python implementation of this <strong>4-Tier Feature Selection Pipeline</strong>, one module at a time, following the specifications above.</p>
<p><strong>You are NOT to output all code at once.</strong> Wait for the user's prompt to begin. <strong>Start with Module 1: WithinDeviceDiscriminabilityFilter.</strong></p>

<hr>

<!-- ============================================================ -->
<!-- FOOTER -->
<!-- ============================================================ -->

<div class="footer">
    <hr>
    <p>Document Classification: Internal – Production Readiness Review<br>
    Version 2.0 FINAL | 2026‑07‑14</p>
</div>

</body>
</html>
