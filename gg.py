mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))  # average over samples (0) and classes (2)



mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))