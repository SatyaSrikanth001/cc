# --- Debug prints ---
print("\n--- SHAP debug ---")
print("Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    print("Number of classes (len of list):", len(shap_values))
    for i, sv in enumerate(shap_values):
        print(f"  shap_values[{i}] shape: {sv.shape}")
else:
    print("shap_values shape:", shap_values.shape)

print("X_surv_scaled shape:", X_surv_scaled.shape)
print("Number of final_features:", len(final_features))
print("--------------------\n")
# --- end debug ---

# Then the existing lines:
mean_abs = np.atleast_1d(
    np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
)