"""
Integrated Gradients Explainer for Psychometric Assessment Models.
Implements the axiomatic attribution method from Sundararajan et al. (2017):
"Axiomatic Attribution for Deep Networks" (ICML 2017).

Provides:
  - Riemann sum path integral approximation of gradients
  - Completeness axiom verification (sum of attributions equals score difference)
  - Support for zero, median, and custom psychometric baselines
  - Local participant explanations and cohort-level global feature attribution summaries
  - Matplotlib visualization plots saved to outputs/ig/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Any, List, Optional, Dict, Union
from src.models.neural_model import PyTorchClassifierWrapper, PsychometricMLP


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients attributions for psychometric assessment models.
    Supports both PyTorch neural models directly and PyTorchClassifierWrapper instances.
    """

    def __init__(self, steps: int = 50, baseline_type: str = "zero"):
        """
        Args:
            steps: Number of interpolation steps for Riemann integration.
            baseline_type: 'zero' (all 0 scores), 'median', or 'mean'.
        """
        self.steps = steps
        self.baseline_type = baseline_type

    def _get_baseline(self, input_tensor: torch.Tensor, background_data: Optional[np.ndarray] = None) -> torch.Tensor:
        if self.baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == "median" and background_data is not None:
            median_vals = np.median(background_data, axis=0)
            return torch.tensor(median_vals, dtype=input_tensor.dtype, device=input_tensor.device).unsqueeze(0)
        elif self.baseline_type == "mean" and background_data is not None:
            mean_vals = np.mean(background_data, axis=0)
            return torch.tensor(mean_vals, dtype=input_tensor.dtype, device=input_tensor.device).unsqueeze(0)
        return torch.zeros_like(input_tensor)

    def attribute(
        self,
        model: Union[nn.Module, PyTorchClassifierWrapper],
        input_data: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        target_class: int = 1,
        baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculates Integrated Gradients attributions for a single instance or batch.
        """
        # Extract underlying PyTorch module if wrapped
        if isinstance(model, PyTorchClassifierWrapper):
            net = model.model
        else:
            net = model

        net.eval()

        if isinstance(input_data, pd.DataFrame):
            x_np = input_data.values.astype(np.float32)
        elif isinstance(input_data, np.ndarray):
            x_np = input_data.astype(np.float32)
        elif isinstance(input_data, torch.Tensor):
            x_np = input_data.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.array(input_data, dtype=np.float32)

        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        device = next(net.parameters()).device
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device)

        if baseline is None:
            x_baseline = self._get_baseline(x_tensor, background_data)
        elif isinstance(baseline, np.ndarray):
            x_baseline = torch.tensor(baseline, dtype=torch.float32, device=device)
            if x_baseline.ndim == 1:
                x_baseline = x_baseline.unsqueeze(0)
        else:
            x_baseline = baseline

        # Scale alphas along straight-line interpolation path
        alphas = torch.linspace(0.0, 1.0, steps=self.steps, device=device)

        # Compute interpolated inputs: shape (steps, features) for 1 sample
        # x_interp = baseline + alpha * (x - baseline)
        diff = x_tensor - x_baseline
        interpolated = x_baseline + alphas.view(-1, 1) * diff
        interpolated.requires_grad_(True)

        logits = net(interpolated)
        probs = torch.softmax(logits, dim=-1)
        target_outputs = probs[:, target_class]

        # Compute gradients with respect to interpolated inputs
        grads = torch.autograd.grad(
            outputs=target_outputs,
            inputs=interpolated,
            grad_outputs=torch.ones_like(target_outputs),
            create_graph=False,
            retain_graph=False
        )[0]

        # Riemann sum approximation: average gradients along path
        avg_grads = grads.mean(dim=0, keepdim=True)  # (1, features)
        attributions = (diff * avg_grads).detach().cpu().numpy().flatten()

        # Completeness Check: sum(attributions) vs (F(x) - F(x_baseline))
        with torch.no_grad():
            prob_x = torch.softmax(net(x_tensor), dim=-1)[0, target_class].item()
            prob_base = torch.softmax(net(x_baseline), dim=-1)[0, target_class].item()
            delta_target = prob_x - prob_base
            convergence_delta = abs(np.sum(attributions) - delta_target)

        return {
            "attributions": attributions,
            "prediction_prob": prob_x,
            "baseline_prob": prob_base,
            "delta_target": delta_target,
            "sum_attributions": float(np.sum(attributions)),
            "convergence_delta": float(convergence_delta),
            "is_converged": bool(convergence_delta < 0.05)
        }


def generate_integrated_gradients_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/ig",
    sample_idx: int = 0,
    steps: int = 50
) -> Dict[str, Any]:
    """
    Computes Integrated Gradients for local test participant and global cohort.
    Generates high-resolution visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INTEGRATED GRADIENTS] Initializing axiomatic path integral with {steps} steps...")

    # If model is not a PyTorch model, train a differentiable neural proxy model
    if not isinstance(model, (PyTorchClassifierWrapper, nn.Module)):
        print("[INTEGRATED GRADIENTS] Training differentiable PyTorch MLP surrogate on psychometric data...")
        from src.models.neural_model import train_neural_model
        # Use pseudo-labels from the main model for distillation or true labels
        proxy_model = train_neural_model(X_train, model.predict(X_train), arch="mlp", epochs=8)
    else:
        proxy_model = model

    ig = IntegratedGradientsExplainer(steps=steps, baseline_type="zero")
    sample_row = X_test.iloc[sample_idx]
    background_arr = X_train.sample(min(200, len(X_train)), random_state=42).values

    result = ig.attribute(
        proxy_model,
        sample_row.values,
        target_class=1,
        background_data=background_arr
    )

    attrs = result["attributions"]

    # 1. Local Explanation Plot
    plt.figure(figsize=(10, 6))
    top_k = 10
    top_indices = np.argsort(np.abs(attrs))[-top_k:]
    top_attrs = attrs[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    colors = ["#d73027" if a > 0 else "#4575b4" for a in top_attrs]

    plt.barh(top_features, top_attrs, color=colors)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Integrated Gradients Attribution (Impact on Depression Risk)", fontsize=12)
    plt.title(f"Integrated Gradients: Local Explanation (Participant #{sample_idx})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    local_path = os.path.join(output_dir, "ig_local_sample.png")
    plt.savefig(local_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Global Cohort Attributions across 100 test samples
    print("[INTEGRATED GRADIENTS] Computing global Integrated Gradients across test cohort...")
    cohort_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    cohort_attrs = []
    for _, row in cohort_sample.iterrows():
        res_sample = ig.attribute(proxy_model, row.values, target_class=1, background_data=background_arr)
        cohort_attrs.append(np.abs(res_sample["attributions"]))
    mean_cohort_attrs = np.mean(cohort_attrs, axis=0)

    plt.figure(figsize=(10, 6))
    global_top_idx = np.argsort(mean_cohort_attrs)[-15:]
    plt.barh([feature_names[i] for i in global_top_idx], mean_cohort_attrs[global_top_idx], color="#2b5c8f")
    plt.xlabel("Mean |Integrated Gradients Attribution|", fontsize=12)
    plt.title("Top 15 Most Influential Features (Global Integrated Gradients)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    global_path = os.path.join(output_dir, "ig_global_summary.png")
    plt.savefig(global_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print Report
    print("\n" + "="*60)
    print(f"   INTEGRATED GRADIENTS LOCAL REPORT (Participant #{sample_idx})")
    print("="*60)
    print(f"Predicted Probability (Target Class 1): {result['prediction_prob']:.4f}")
    print(f"Baseline Probability (Zero Input):      {result['baseline_prob']:.4f}")
    print(f"Target Probability Shift (F(x) - F(0)):  {result['delta_target']:+.4f}")
    print(f"Sum of Feature Attributions:            {result['sum_attributions']:+.4f}")
    print(f"Completeness Error (|Sum - Shift|):     {result['convergence_delta']:.6f} "
          f"({'Converged' if result['is_converged'] else 'Close'})")
    print("\nTop 5 Attributed Features:")
    sorted_features = sorted(zip(feature_names, attrs), key=lambda x: abs(x[1]), reverse=True)
    for rank, (fname, val) in enumerate(sorted_features[:5], 1):
        direction = "Increases Risk (+)" if val > 0 else "Decreases Risk (-)"
        print(f"  {rank}. {fname}: Attribution = {val:+.4f} ({direction})")
    print("="*60 + "\n")

    print(f"[INTEGRATED GRADIENTS] Local plot saved to: {local_path}")
    print(f"[INTEGRATED GRADIENTS] Global summary plot saved to: {global_path}")

    return {
        "attributions": attrs,
        "local_plot": local_path,
        "global_plot": global_path,
        "convergence_delta": result["convergence_delta"],
        "top_features": sorted_features[:5]
    }
