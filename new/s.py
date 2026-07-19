import os
import uuid
import numpy as np
import pandas as pd

# ============================================================
# Synthetic HMOG Dataset Generator
# Compatible with load_and_clean_data()
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OUTPUT_DIR = "synthetic_features"

NUM_USERS = 17
NUM_FEATURES = 1350

TRAIN_GENUINE = 25
TEST_GENUINE = 10
TEST_IMPOSTOR = 16

MISSING_FEATURES = 40
CONSTANT_FEATURES = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Feature Names
# ------------------------------------------------------------

feature_names = [
    f"feature_{i:04d}"
    for i in range(1, NUM_FEATURES + 1)
]

missing_feature_names = feature_names[:MISSING_FEATURES]

constant_feature_names = feature_names[
    MISSING_FEATURES:
    MISSING_FEATURES + CONSTANT_FEATURES
]

print("=" * 70)
print("Generating Synthetic Behavioral Biometrics Dataset")
print("=" * 70)

print(f"Users                : {NUM_USERS}")
print(f"Total Features       : {NUM_FEATURES}")
print(f"Training Sessions    : {TRAIN_GENUINE}")
print(f"Testing Genuine      : {TEST_GENUINE}")
print(f"Testing Impostor     : {TEST_IMPOSTOR}")
print()

# ============================================================
# Generate One User at a Time
# ============================================================

for user in range(1, NUM_USERS + 1):

    user_id = f"user{user:03d}"

    print(f"Generating {user_id}")

    # --------------------------------------------------------
    # Each user has a unique behavioural profile
    # --------------------------------------------------------

    user_mean = np.random.uniform(
        low=-3,
        high=3,
        size=NUM_FEATURES
    )

    # --------------------------------------------------------
    # Training Genuine Sessions
    # --------------------------------------------------------

    train = np.random.normal(
        loc=user_mean,
        scale=0.40,
        size=(TRAIN_GENUINE, NUM_FEATURES)
    )

    train_df = pd.DataFrame(
        train,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Testing Genuine Sessions
    # --------------------------------------------------------

    genuine = np.random.normal(
        loc=user_mean,
        scale=0.40,
        size=(TEST_GENUINE, NUM_FEATURES)
    )

    genuine_df = pd.DataFrame(
        genuine,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Testing Impostor Sessions
    # --------------------------------------------------------

    impostor_mean = np.random.uniform(
        low=-3,
        high=3,
        size=NUM_FEATURES
    )

    impostor = np.random.normal(
        loc=impostor_mean,
        scale=0.40,
        size=(TEST_IMPOSTOR, NUM_FEATURES)
    )

    impostor_df = pd.DataFrame(
        impostor,
        columns=feature_names
    )

    # --------------------------------------------------------
    # Add Labels
    # --------------------------------------------------------

    genuine_df["session_label"] = "genuine"
    impostor_df["session_label"] = "impostor"

    test_df = pd.concat(
        [genuine_df, impostor_df],
        ignore_index=True
    )

    # Shuffle testing sessions
    test_df = test_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Insert session_id
    # --------------------------------------------------------

    train_df.insert(
        0,
        "session_id",
        [
            f"{user_id}_{uuid.uuid4().hex[:16]}"
            for _ in range(TRAIN_GENUINE)
        ]
    )

    test_df.insert(
        0,
        "session_id",
        [
            f"{user_id}_{uuid.uuid4().hex[:16]}"
            for _ in range(len(test_df))
        ]
    )

    # --------------------------------------------------------
    # Insert device_owner_id
    # --------------------------------------------------------

    train_df.insert(
        1,
        "device_owner_id",
        user_id
    )

    test_df.insert(
        1,
        "device_owner_id",
        user_id
    )

    # --------------------------------------------------------
    # Create Missing Features
    # --------------------------------------------------------

    for feature in missing_feature_names:

        train_mask = np.random.rand(len(train_df)) < 0.70
        test_mask = np.random.rand(len(test_df)) < 0.70

        train_df.loc[train_mask, feature] = np.nan
        test_df.loc[test_mask, feature] = np.nan

    # --------------------------------------------------------
    # Create Constant Features
    # Constant for first 15 users
    # --------------------------------------------------------

    if user <= 15:

        for feature in constant_feature_names:

            value = np.random.uniform(-5, 5)

            train_df[feature] = value
            test_df[feature] = value

    # --------------------------------------------------------
    # Save CSV Files
    # --------------------------------------------------------

    train_file = os.path.join(
        OUTPUT_DIR,
        f"{user_id}_training_sessions.csv"
    )

    test_file = os.path.join(
        OUTPUT_DIR,
        f"{user_id}_testing_sessions.csv"
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("Synthetic Dataset Generated Successfully")
print("=" * 70)

print(f"Output Folder : {OUTPUT_DIR}")

print()
print("Expected Cleaning Results")
print("-------------------------")
print(f"Initial Features      : {NUM_FEATURES}")
print(f"Missing Removed       : ~{MISSING_FEATURES}")
print(f"Constant Removed      : ~{CONSTANT_FEATURES}")
print(f"Remaining Features    : ~{NUM_FEATURES - MISSING_FEATURES - CONSTANT_FEATURES}")

print()
print("Generated Files")
print("----------------")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(file)

print()
print("Example session_id")
print("------------------")
print(f"user001_{uuid.uuid4().hex[:16]}")