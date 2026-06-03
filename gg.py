def plot_shap_summary(shap_values, X_scaled, feature_names):
    """SHAP summary plot (beeswarm) for the first class."""
    # shap_values is 3D (samples, features, classes) – take class 0
    shap_values_class0 = shap_values[:, :, 0]
    shap.summary_plot(shap_values_class0, X_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()



def plot_shap_bar(shap_values, feature_names, top_n=20):
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))   # average over samples & classes
    idx = np.argsort(mean_abs_shap)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(idx)), mean_abs_shap[idx], color='steelblue')
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} SHAP Feature Importance (Boruta survivors)")
    plt.tight_layout()
    plt.savefig("shap_importance.png", dpi=300, bbox_inches='tight')
    plt.show()