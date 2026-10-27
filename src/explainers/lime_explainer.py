"""
LIME (Local Interpretable Model-agnostic Explanations) implementation for psychometric assessment.
Generates local tabular feature contribution explanations and visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
from typing import Any, List


def generate_lime_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/lime",
    sample_idx: int = 0
):
    """
    Computes actual LIME local explanation for a real test participant observation.
    Saves visualization plots and interactive HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[LIME EXPLAINER] Initializing LimeTabularExplainer on real dataset...")

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["Normal/Mild (0)", "Mod-Severe (1)"],
        mode="classification",
        random_state=42
    )

    sample_instance = X_test.iloc[sample_idx]
    
    # Generate local explanation for classes (0, 1)
    exp = explainer.explain_instance(
        data_row=np.array(sample_instance),
        predict_fn=model.predict_proba,
        num_features=10,
        labels=(0, 1)
    )

    # Save HTML report
    html_path = os.path.join(output_dir, "lime_explanation.html")
    exp.save_to_file(html_path)

    # Plot & save matplotlib bar chart of LIME weights for Class 1 (or top class)
    plt.figure(figsize=(10, 6))
    label_to_explain = 1 if 1 in exp.available_labels() else exp.available_labels()[0]
    explanation_tuples = exp.as_list(label=label_to_explain)
    
    feat_names = [t[0] for t in explanation_tuples][::-1]
    weights = [t[1] for t in explanation_tuples][::-1]
    colors = ["#d73027" if w > 0 else "#4575b4" for w in weights]

    plt.barh(feat_names, weights, color=colors)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("LIME Weight (Feature Contribution)", fontsize=12)
    plt.title(f"LIME Local Explanation for Test Participant #{sample_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "lime_local_sample.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "="*60)
    print(f"       LIME LOCAL EXPLANATION REPORT (Participant #{sample_idx})")
    print("="*60)
    print("Feature Conditions and Local Weights for Moderate-to-Severe Depression Risk:")
    for rank, (feat_cond, weight) in enumerate(explanation_tuples[:5], 1):
        direction = "Supports Risk Prediction (+)" if weight > 0 else "Opposes Risk Prediction (-)"
        print(f"  {rank}. {feat_cond}: Weight = {weight:+.4f} ({direction})")
    print("="*60 + "\n")

    print(f"[LIME] Local plot figure saved to: {plot_path}")
    print(f"[LIME] Interactive HTML report saved to: {html_path}")

    return {
        "plot_path": plot_path,
        "html_path": html_path,
        "explanation_list": explanation_tuples
    }
