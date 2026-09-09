Yes — for your behavioral biometrics project, this is exactly the stage where feature selection becomes critical. With ~15 users, hundreds of sessions per user, and potentially 5,000+ engineered features, you should not simply throw all features into a model and select based on importance.

Your data has a special structure: many sessions belonging to the same person. So the best method needs to avoid data leakage.

First, I want to clarify your setup

Something like:

15 users

Each user has approximately:

~100 genuine sessions

~80 impostor sessions


Training CSV:

70 genuine sessions per user


Testing CSV:

remaining genuine sessions

impostor/important attack sessions in testing


Each session contains:

keystrokes

touches

navigation

scrolling

gestures

many raw behavioral signals


Feature engineering creates potentially 5,000 features


Correct?


---

🚨 The biggest danger: Feature selection leakage

Suppose you do this:

> Take all sessions → select best 100 features → split train/test → train model



❌ Wrong.

The feature selection algorithm has already seen information from the test set.

Instead:

> Training sessions only → feature selection → freeze selected features → apply same features to test



✅ Correct.


---

The method I would recommend for your project

Don't use just one feature selection technique.

Use a multi-stage feature selection pipeline:

🥇 Stage 1: Remove useless features

From 5,000 features, first remove obvious garbage.

Remove:

1. Constant features

Features that barely change.

Example:

feature_A = 0 for almost everyone

No behavioral information.

Use:

Variance Threshold



---

2. Near-duplicate features

Suppose:

mean_touch_duration
avg_touch_duration
touch_duration_mean

They may essentially contain the same information.

Calculate correlation.

If:

|correlation| > 0.90 or 0.95

Keep only one.

This can easily reduce:

5000 → 1500–2500


---

3. Features with too many missing values

For example:

pinch_gesture_speed

But only 5% of sessions contain pinch gestures.

That feature may be unreliable.

You can:

Remove features missing in >40–60% of sessions

Or treat "gesture absent" carefully as behavioral information


⚠️ Important: In behavioral biometrics, missingness itself can sometimes be useful.

For example:

> User A never uses pinch zoom
User B frequently uses it



So don't blindly delete everything with missing values.

You may create:

pinch_speed
pinch_used_flag


---

🥈 Stage 2: Univariate feature selection

Now ask:

> Does this feature individually help distinguish the target?



Use multiple methods.

For numerical features:

ANOVA F-test

Mutual Information

AUC / ROC discriminatory power


Mutual Information is particularly useful because behavioral patterns may be non-linear.

For every feature, calculate something like:

Feature
↓
ANOVA score
Mutual Information score
Effect size
AUC

Then rank them.

Example:

Feature	MI	AUC	Rank

swipe_velocity_mean	High	0.82	1
typing_hold_time_std	High	0.79	2
screen_transition_entropy	Medium	0.70	15
random_feature	Low	0.51	4800


This might reduce:

2000 → Top 500


---

🥇🥇 Stage 3: The MOST important one — model-based selection

Now use models that naturally discover important features.

I strongly recommend trying:

1. LightGBM / XGBoost

Great for:

nonlinear behavior

interactions

heterogeneous features

large feature spaces


Get feature importance using:

Gain importance

Permutation importance



---

2. Random Forest / Extra Trees

Especially useful as another independent opinion.


---

3. L1 Logistic Regression

This is extremely useful for feature selection.

L1 regularization forces many coefficients to:

0

So:

500 features
↓
maybe 80 useful features


---

🥇 My favorite approach for your case: Feature Selection Voting

Don't trust one algorithm.

Create rankings from multiple methods:

Method A

Mutual Information

Method B

ANOVA / statistical discrimination

Method C

XGBoost importance

Method D

Permutation importance

Method E

L1 regularization

Then calculate:

> How consistently is a feature considered useful?



Example:

Feature	MI	XGB	Permutation	L1	Final

swipe_speed_std	✅	✅	✅	✅	⭐⭐⭐⭐⭐
typing_variance	✅	✅	❌	✅	⭐⭐⭐⭐
random_feature	❌	❌	❌	❌	❌


This gives you stable features, rather than features that only look good because of random training data.


---

🚨 But your dataset has another major issue: only 15 users

This is very important.

You may have:

15 users × 100 sessions

But statistically, you don't really have thousands of completely independent identities.

You have only:

👉 15 identities

So if you generate 5,000 features, you have a serious risk of:

⚠️ Overfitting

The model may learn:

> "This particular feature pattern belongs to these 15 people"



instead of learning genuine behavioral identity.


---

⭐ The BEST technique for your dataset: Stability Selection

I would strongly recommend this.

The process:

Step 1

Randomly sample training sessions.

Step 2

Perform feature selection.

Step 3

Record selected features.

Step 4

Repeat maybe:

50–100 times

Then calculate:

Selection Frequency

Example:

Feature	Selected in

touch_pressure_std	96%
swipe_acceleration_mean	91%
key_hold_std	88%
random_navigation_feature	12%


Keep features selected consistently, for example:

>70% of runs

This is MUCH more trustworthy than selecting features once.


---

🔥 Even better: Group-wise Feature Selection

This is something I particularly recommend for your project.

You told me your features come from different behavioral families:

Feature families

⌨️ Keystroke dynamics

👆 Touch dynamics

📜 Scroll behavior

👉 Swipe behavior

🔄 Gesture behavior

🧭 Navigation behavior

⏱️ Timing behavior

🔀 Transition behavior

🧠 Session-level behavior

📊 Distribution/statistical features


If you simply select the global top 100 features...

You might get:

85 swipe features
10 touch features
5 keyboard features

But that may not be ideal.

Why?

Because highly correlated swipe features can dominate the ranking.


---

Instead use Hierarchical Feature Selection

Level 1 — Select within each family

Example:

Keystroke features: 500
→ Top 30

Touch features: 1000
→ Top 40

Swipe features: 1500
→ Top 40

Navigation features: 500
→ Top 30

Scroll features: 500
→ Top 30

Now:

~170 features

Then...

Level 2 — Global feature selection

Run:

XGBoost

L1

permutation importance

stability selection


Finally:

170
↓
50–150 final features

This gives you behavioral diversity.


---

🎯 The pipeline I would personally use for YOUR project

STEP 0 — Split correctly

TRAIN
│
├── Feature selection
├── Scaling
├── Feature engineering parameters
└── Model training

TEST
│
└── ONLY final evaluation

Test data must remain untouched.


---

STEP 1 — Basic Cleaning

5000
 ↓
Remove constants
 ↓
Remove near-zero variance
 ↓
Handle missing values
 ↓
Remove duplicate features
 ↓
Correlation filtering

Result:

~1500–2500 features


---

STEP 2 — Feature Family Selection

Within each behavioral family:

Keyboard → Top K
Touch → Top K
Swipe → Top K
Scroll → Top K
Navigation → Top K
Gesture → Top K

Result:

~300–500 features


---

STEP 3 — Multi-method Ranking

Use:

Mutual Information
+
XGBoost Gain
+
Permutation Importance
+
L1 Logistic Regression

Create a combined ranking.

Result:

Top 100–300


---

STEP 4 — Stability Selection ⭐

Repeat selection:

50–100 times

using different bootstrap samples of training data.

Keep features that consistently survive.

Result:

~50–150 robust features


---

STEP 5 — Recursive Feature Elimination

Now optionally use:

RFECV

Test:

200 features
150
100
75
50
30

Evaluate cross-validation performance.

Choose the smallest feature set that gives near-maximum performance.

This is important.

Suppose:

50 features → 94.2%
100 features → 94.8%
200 features → 95.0%

Don't necessarily choose 200.

Choose around:

50–100

Because it is:

faster

more robust

less overfitted

easier to explain



---

📊 One thing I strongly recommend: Don't optimize only Accuracy

For behavioral biometrics, measure:

Authentication metrics

FAR — False Acceptance Rate

FRR — False Rejection Rate

EER — Equal Error Rate

ROC-AUC

Precision / Recall


Most importantly:

⭐ EER

Because your system is authentication/security-related.

A model with:

99% Accuracy

can still have poor security performance depending on class balance.


---

🔥 My strongest recommendation for your exact project

Use this architecture:

5,000 Raw Engineered Features
            │
            ▼
 ┌──────────────────────────────┐
 │ Remove Bad Features          │
 │ variance / missing / dupes   │
 └──────────────┬───────────────┘
                ▼
         Correlation Filtering
                │
                ▼
 ┌──────────────────────────────┐
 │ Feature Family Selection     │
 │ keyboard/touch/swipe/etc     │
 └──────────────┬───────────────┘
                ▼
       Mutual Information Ranking
                │
                ▼
      XGBoost Feature Importance
                │
                ▼
       Permutation Importance
                │
                ▼
        Stability Selection
                │
                ▼
         RFECV Optimization
                │
                ▼
       🎯 Final 50–150 Features
                │
                ▼
          FINAL MODEL

One more thing: your session splitting strategy matters enormously

Before deciding the exact feature-selection code, I would want to inspect your actual:

training CSV structure

testing CSV structure

target label

user ID column

session ID column

how you represent genuine vs impostor


Because for your behavioral biometrics project, feature selection must be performed with user/session-aware cross-validation. A normal random train_test_split() can give artificially excellent results due to sessions from the same user leaking behavioral patterns across folds.

If you share your current feature-generation Python file and the columns/structure of the training CSV, I can design the exact feature-selection pipeline for your dataset, including which methods to use and in what order.