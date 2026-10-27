"""
Counterfactual Explanations and Clinical Recourse Generation.
Implements actionable recourse search based on Wachter et al. (2017) and DiCE principles:
"Counterfactual Explanations without Opening the Black Box: Automated Decisions and the Law".

Features:
  - Enforces clinical feasibility & actionability constraints (locks immutable demographic features).
  - Respects Likert score integer bounds [0, 3].
  - Minimizes L1 (sparsity) and L2 perturbation distances.
  - Generates actionable clinical intervention prescriptions.
  - Visualizes original vs counterfactual symptom comparison charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Tuple, Optional


class CounterfactualExplainer:
    """
    Generates actionable counterfactual instances for psychometric assessment classifiers.
    """

    def __init__(
        self,
        target_class: int = 0,
        max_iter: int = 500,
        tolerance: float = 0.45,
        step_size: float = 0.1
    ):
        """
        Args:
            target_class: Target class to achieve (default: 0 = Normal/Mild Risk).
            max_iter: Maximum search iterations.
            tolerance: Target probability threshold for target class (e.g. 0.45).
            step_size: Perturbation increment step.
        """
        self.target_class = target_class
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size = step_size

    def _get_immutable_mask(self, feature_names: List[str]) -> np.ndarray:
        """
        Flags demographic and unmodifiable features that cannot be clinically altered.
        """
        immutable_keywords = ["Demographic", "age", "gender", "education", "urban"]
        mask = np.zeros(len(feature_names), dtype=bool)
        for i, name in enumerate(feature_names):
            if any(k.lower() in name.lower() for k in immutable_keywords):
                mask[i] = True
        return mask

    def explain(
        self,
        model: Any,
        instance: np.ndarray,
        feature_names: List[str],
        max_changes: int = 5
    ) -> Dict[str, Any]:
        """
        Finds minimal actionable perturbation delta to flip model decision.
        """
        if instance.ndim > 1:
            orig_x = instance.flatten().copy()
        else:
            orig_x = instance.copy()

        cf_x = orig_x.copy()
        immutable_mask = self._get_immutable_mask(feature_names)

        # Baseline prediction
        if hasattr(model, "predict_proba"):
            orig_prob = model.predict_proba(orig_x.reshape(1, -1))[0, 1]
        else:
            orig_prob = float(model.predict(orig_x.reshape(1, -1))[0])

        current_prob = orig_prob
        # If already at target class, no recourse needed
        if orig_prob < 0.5:
            return {
                "original_prob": orig_prob,
                "counterfactual_prob": orig_prob,
                "counterfactual_instance": cf_x,
                "changes": [],
                "l1_distance": 0.0,
                "l2_distance": 0.0,
                "success": True,
                "note": "Participant is already in the Normal/Mild risk category."
            }

        # Greedy coordinate perturbation on actionable features
        actionable_indices = [i for i in range(len(feature_names)) if not immutable_mask[i]]

        # Evaluate sensitivity of each actionable feature
        sensitivities = []
        for idx in actionable_indices:
            current_val = cf_x[idx]
            if current_val > 0:  # Can be reduced towards 0 (symptom alleviation)
                test_x = cf_x.copy()
                test_x[idx] = max(0, current_val - 1)
                test_prob = model.predict_proba(test_x.reshape(1, -1))[0, 1]
                prob_reduction = current_prob - test_prob
                sensitivities.append((idx, prob_reduction))

        # Sort by highest risk reduction potential
        sensitivities.sort(key=lambda x: x[1], reverse=True)

        # Iteratively apply minimal discrete interventions
        applied_changes = []
        for idx, expected_gain in sensitivities:
            if current_prob < 0.5 or len(applied_changes) >= max_changes:
                break

            orig_val = int(cf_x[idx])
            # Step down Likert score to alleviate symptom
            for new_val in range(orig_val - 1, -1, -1):
                test_x = cf_x.copy()
                test_x[idx] = new_val
                new_prob = model.predict_proba(test_x.reshape(1, -1))[0, 1]
                cf_x[idx] = new_val
                current_prob = new_prob

                if new_prob < 0.5:
                    break

            applied_changes.append({
                "feature": feature_names[idx],
                "original_value": orig_val,
                "counterfactual_value": int(cf_x[idx]),
                "delta": int(cf_x[idx]) - orig_val,
                "action": f"Reduce '{feature_names[idx].split(':')[0]}' from {orig_val} to {int(cf_x[idx])}"
            })

        l1_dist = float(np.sum(np.abs(cf_x - orig_x)))
        l2_dist = float(np.sqrt(np.sum((cf_x - orig_x) ** 2)))
        success = bool(current_prob < 0.5)

        return {
            "original_prob": float(orig_prob),
            "counterfactual_prob": float(current_prob),
            "original_instance": orig_x,
            "counterfactual_instance": cf_x,
            "changes": applied_changes,
            "l1_distance": l1_dist,
            "l2_distance": l2_dist,
            "success": success
        }


def generate_counterfactual_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/counterfactual",
    sample_idx: int = 0
) -> Dict[str, Any]:
    """
    Computes actionable clinical recourse for a real test participant.
    Saves before/after comparison plots and clinical prescription report.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[COUNTERFACTUAL EXPLAINER] Generating actionable clinical recourse plan...")

    explainer = CounterfactualExplainer(target_class=0, max_iter=300)
    sample_instance = X_test.iloc[sample_idx].values.astype(np.float32)

    result = explainer.explain(model, sample_instance, feature_names, max_changes=5)

    changes = result["changes"]

    # Plot Before/After Comparison Chart for altered items
    if len(changes) > 0:
        labels = [c["feature"].split(":")[0] for c in changes]
        orig_vals = [c["original_value"] for c in changes]
        cf_vals = [c["counterfactual_value"] for c in changes]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(9, 5))
        plt.bar(x - width/2, orig_vals, width, label="Current Response", color="#d73027")
        plt.bar(x + width/2, cf_vals, width, label="Target Recourse", color="#1a9641")

        plt.xlabel("Psychometric Symptom Items", fontsize=12)
        plt.ylabel("Likert Score (0: Did not apply - 3: Applied very much)", fontsize=11)
        plt.title(
            f"Actionable Recourse Plan for Participant #{sample_idx}\n"
            f"Predicted Risk: {result['original_prob']*100:.1f}% → {result['counterfactual_prob']*100:.1f}%",
            fontsize=13, fontweight="bold"
        )
        plt.xticks(x, labels, fontsize=11)
        plt.yticks([0, 1, 2, 3], ["0 (None)", "1 (Mild)", "2 (Mod)", "3 (Severe)"])
        plt.ylim(0, 3.5)
        plt.legend(loc="upper right", fontsize=11)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        cf_plot_path = os.path.join(output_dir, "counterfactual_recourse.png")
        plt.savefig(cf_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        cf_plot_path = os.path.join(output_dir, "counterfactual_recourse.png")
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Participant already in Normal/Mild Risk category\nNo recourse needed.",
                 ha="center", va="center", fontsize=12, fontweight="bold")
        plt.axis("off")
        plt.savefig(cf_plot_path, dpi=300)
        plt.close()

    # Print Clinical Recourse Prescription
    print("\n" + "="*60)
    print(f"       ACTIONABLE CLINICAL RECOURSE (Participant #{sample_idx})")
    print("="*60)
    print(f"Current Depression Risk:     {result['original_prob']*100:.2f}% (Moderate-to-Severe)")
    print(f"Recourse Target Risk:        {result['counterfactual_prob']*100:.2f}% "
          f"({'Normal/Mild Risk [SUCCESS]' if result['success'] else 'Reduced'})")
    print(f"Recourse Distance (L1 / L2): {result['l1_distance']:.1f} / {result['l2_distance']:.2f}")
    print(f"Number of Interventions:     {len(changes)} actionable items")
    print("\nPrescribed Therapeutic Target Interventions:")
    for rank, c in enumerate(changes, 1):
        print(f"  {rank}. {c['action']} (Severity drop: {abs(c['delta'])})")
    print("="*60 + "\n")

    print(f"[COUNTERFACTUAL] Recourse visualization saved to: {cf_plot_path}")

    return {
        "original_prob": result["original_prob"],
        "counterfactual_prob": result["counterfactual_prob"],
        "changes": changes,
        "l1_distance": result["l1_distance"],
        "plot_path": cf_plot_path,
        "success": result["success"]
    }
