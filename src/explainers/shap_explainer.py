"""
SHAP (SHapley Additive exPlanations) implementation for psychometric assessment model.
Generates global feature importance summary plots and local sample explanations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any, List


def generate_shap_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/shap",
    sample_idx: int = 0
):
    """
    Calculates actual SHAP values for the trained model on real psychometric test data.
    Saves global summary, bar plot, and local sample explanation figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Sample up to 500 test instances for fast and statistically representative SHAP calculation
    X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    print(f"\n[SHAP EXPLAINER] Calculating SHAP values on {len(X_test_sample):,} representative test samples...")

    # Initialize TreeExplainer or Explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample)
    except Exception:
        # Fallback to generic Explainer or KernelExplainer background sampling
        background = X_train.sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_test_sample)

    # Handle binary classification SHAP dimensions
    if len(shap_values.shape) == 3:
        # Shape: (samples, features, classes) -> Select class 1 (Moderate-to-Severe Risk)
        sv_class1 = shap_values[:, :, 1]
    else:
        sv_class1 = shap_values

    # 1. Global Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_class1.values, X_test_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (DASS Psychometric Data)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Global Bar Importance Plot
    plt.figure(figsize=(10, 6))
    mean_abs_shap = np.abs(sv_class1.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:]
    plt.barh([feature_names[i] for i in top_indices], mean_abs_shap[top_indices], color="#3182bd")
    plt.xlabel("Mean |SHAP Value| (Impact on Depression Risk Prediction)", fontsize=12)
    plt.title("Top 15 Most Influential Features (SHAP Global Bar Plot)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_bar_plot.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Local Explanation for Sample Observation
    sample_shap = sv_class1[sample_idx]
    plt.figure(figsize=(10, 6))
    sample_values = sample_shap.values
    local_top_idx = np.argsort(np.abs(sample_values))[-10:]
    
    colors = ["#d73027" if val > 0 else "#4575b4" for val in sample_values[local_top_idx]]
    plt.barh([feature_names[i] for i in local_top_idx], sample_values[local_top_idx], color=colors)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("SHAP Value (Contribution to Prediction)", fontsize=12)
    plt.title(f"Local SHAP Explanation for Test Participant #{sample_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    local_path = os.path.join(output_dir, "shap_local_sample.png")
    plt.savefig(local_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print actual calculated local feature impacts
    pred_prob = model.predict_proba(X_test.iloc[[sample_idx]])[0, 1] if hasattr(model, "predict_proba") else model.predict(X_test.iloc[[sample_idx]])[0]
    pred_label = "Moderate-to-Severe Depression Risk (1)" if pred_prob >= 0.5 else "Normal/Mild Depression (0)"

    print("\n" + "="*60)
    print(f"       SHAP LOCAL EXPLANATION REPORT (Participant #{sample_idx})")
    print("="*60)
    print(f"Model Predicted Risk Probability: {pred_prob:.4f}")
    print(f"Predicted Class: {pred_label}")
    print("\nMost Influential Features for this Participant:")
    
    sorted_local = sorted(zip(feature_names, sample_values), key=lambda x: abs(x[1]), reverse=True)
    for rank, (fname, val) in enumerate(sorted_local[:5], 1):
        direction = "Increases Risk (+)" if val > 0 else "Decreases Risk (-)"
        print(f"  {rank}. {fname}: SHAP = {val:+.4f} ({direction})")
    print("="*60 + "\n")

    print(f"[SHAP] Global summary plot saved to: {summary_path}")
    print(f"[SHAP] Global bar plot saved to: {bar_path}")
    print(f"[SHAP] Local explanation plot saved to: {local_path}")

    return {
        "summary_plot": summary_path,
        "bar_plot": bar_path,
        "local_plot": local_path,
        "top_features": sorted_local[:5]
    }
