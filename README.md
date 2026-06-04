# cc
Item Detail
Feature pool You have a code that can generate up to ~441 features. From those, you have currently selected a subset of ~90 features (the ones you listed), after dropping many via a drop_classes array.
Model OCSVM, RBF kernel, per‑user (17 independent models). Training on ~40 genuine sessions per user. Test on ~35‑40 genuine + 20‑25 impostor sessions.
Current metrics Accuracy ~69 %, FRR = 32.25 % (genuine users rejected), FAR = 0.15 % (impostors barely accepted).
Hyperparameters nu=0.05, gamma=0.00055, threshold = 0 (score > 0 → genuine).
Constraints OCSVM cannot be replaced, but you can adjust the decision threshold, tune nu and gamma, and use data from other users for feature selection.


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


https://github.com/BiDAlab/HuMIdb