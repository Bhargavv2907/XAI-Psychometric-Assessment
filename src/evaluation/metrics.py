"""
Evaluation metrics calculation and visualization module.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix dynamically from test set predictions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from typing import Dict, Any


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs/figures"
) -> Dict[str, float]:
    """
    Evaluates the model on test data, prints calculated metrics,
    and saves Confusion Matrix and ROC Curve figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*60)
    print("               MODEL PERFORMANCE EVALUATION")
    print("="*60)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*60 + "\n")

    # Plot & save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal/Mild (0)", "Mod-Severe (1)"],
                yticklabels=["Normal/Mild (0)", "Mod-Severe (1)"])
    plt.title("Confusion Matrix - Baseline Model", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot & save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"[EVALUATION] Confusion matrix saved to: {cm_path}")
    print(f"[EVALUATION] ROC curve saved to: {roc_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
