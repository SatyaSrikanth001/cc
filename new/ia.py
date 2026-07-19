import warnings
warnings.filterwarnings("ignore", message=".*numexpr.*")

import yaml
import sys

print("Step 1: Loading config...")
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"  Config loaded. features_dir = {config['data']['features_dir']}")
except Exception as e:
    print(f"  ERROR loading config: {e}")
    sys.exit(1)

print("Step 2: Importing data loader...")
try:
    from utils.data_loader import load_split_data
    print("  Import successful.")
except Exception as e:
    print(f"  ERROR importing: {e}")
    sys.exit(1)

print("Step 3: Loading and splitting data...")
try:
    dev_data, holdout_data, feature_list = load_split_data(config)
    print(f"  Dev users: {len(dev_data)}")
    print(f"  Holdout users: {len(holdout_data)}")
    print(f"  Features: {len(feature_list)}")
except Exception as e:
    print(f"  ERROR in load_split_data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 4: Checking output directory...")
import os
out_dir = config["output"].get("dir", "out")
os.makedirs(out_dir, exist_ok=True)
print(f"  Output dir: {out_dir} (exists: {os.path.exists(out_dir)})")

print("Step 5: Computing AUCs for first user...")
try:
    first_user = list(dev_data.keys())[0]
    df = dev_data[first_user]
    print(f"  User: {first_user}")
    print(f"  Rows: {len(df)}")
    print(f"  Genuine: {(df['is_genuine'] == 1).sum()}")
    print(f"  Impostor: {(df['is_genuine'] == 0).sum()}")
    print(f"  Columns: {len(df.columns)}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll checks passed. The data loader works correctly.")
print("Phase 2A should be able to run.")