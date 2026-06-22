def load_user_artifacts(user_id):
    model_path = os.path.join(MODELS_DIR, f"{user_id}_ocsvm.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{user_id}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler missing for {user_id}.")

    ocsvm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    train_csv = os.path.join(FEATURES_DIR, f"{user_id}_training_sessions.csv")
    test_csv  = os.path.join(FEATURES_DIR, f"{user_id}_testing_sessions.csv")

    # Use the exact same loading logic as training
    session_model = SessionOCSVM(user_id)
    X_train_raw, X_test_raw, y_test, train_df_proc, test_df_proc = \
        session_model.load_and_prepare_data(train_csv, test_csv)

    feature_cols_all = session_model.feature_columns  # all columns after drop_cols

    # Keep ONLY numeric columns (safety against string metadata)
    train_df = train_df_proc.select_dtypes(include=[np.number])
    test_df  = test_df_proc.select_dtypes(include=[np.number])

    # Report any unexpected drops
    if len(train_df.columns) != len(feature_cols_all):
        removed = set(feature_cols_all) - set(train_df.columns)
        if removed:
            print(f"  [{user_id}] Removed non‑numeric columns: {removed}")

    actual_features = train_df.columns.tolist()
    return ocsvm, scaler, train_df, test_df, y_test, actual_features