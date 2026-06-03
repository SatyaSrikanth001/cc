def load_user_data(user):
    train = pd.read_csv(os.path.join(DATA_DIR, f"{user}_training_sessions.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, f"{user}_testing_sessions.csv"))

    # Remove non-feature columns
    train.drop(columns=['user'], errors='ignore', inplace=True)
    test.drop(columns=['user'], errors='ignore', inplace=True)

    # ---- ADD DIAGNOSTIC HERE (before mapping, before dropping label) ----
    print(f"Labels for {user}:", test['label'].unique())
    # -------------------------------------------------------------------

    # Extract labels (0=genuine, 1=impostor)
    label_map = {'genuine': 0, 'impostor': 1}
    y_test = test['label'].map(label_map)

    test.drop(columns=['label'], inplace=True)

    # ... rest of the function ...