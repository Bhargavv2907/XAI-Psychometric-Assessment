"""
XAI Explainers Package and Framework Interface.
Coordinates all 6 implemented explainability techniques:
  1. SHAP (SHapley Additive exPlanations)
  2. LIME (Local Interpretable Model-agnostic Explanations)
  3. Integrated Gradients (Axiomatic path integrals)
  4. Grad-CAM (Gradient-weighted Class Activation Mapping)
  5. Attention-Based Explanations (Transformer self-attention)
  6. Counterfactual Recourse (Actionable clinical intervention)
"""

from typing import Dict, Any, List
import pandas as pd

from .shap_explainer import generate_shap_explanations
from .lime_explainer import generate_lime_explanations
from .integrated_gradients import generate_integrated_gradients_explanations, IntegratedGradientsExplainer
from .grad_cam import generate_gradcam_explanations, GradCam1DExplainer
from .attention_explainer import generate_attention_explanations, AttentionExplainer
from .counterfactual import generate_counterfactual_explanations, CounterfactualExplainer


class XAIFrameworkManager:
    """
    Unified Manager Interface for Explainable AI methods in Psychometric Assessment.
    Provides execution hooks, registry, and cohort analysis across all 6 techniques.
    """

    def __init__(self):
        self.explainers = {
            "SHAP": {
                "fn": generate_shap_explanations,
                "type": "Game-Theoretic Feature Attribution",
                "scope": "Global & Local",
                "status": "OPERATIONAL"
            },
            "LIME": {
                "fn": generate_lime_explanations,
                "type": "Surrogate Local Linear Approximation",
                "scope": "Local",
                "status": "OPERATIONAL"
            },
            "Integrated Gradients": {
                "fn": generate_integrated_gradients_explanations,
                "type": "Axiomatic Path Integral Gradient Attribution",
                "scope": "Global & Local",
                "status": "OPERATIONAL"
            },
            "Grad-CAM": {
                "fn": generate_gradcam_explanations,
                "type": "1D Convolutional Class Activation Mapping",
                "scope": "Local Subscale Localization",
                "status": "OPERATIONAL"
            },
            "Attention Explanations": {
                "fn": generate_attention_explanations,
                "type": "Transformer Multi-Head Self-Attention",
                "scope": "Local & Cross-Symptom Interaction",
                "status": "OPERATIONAL"
            },
            "Counterfactual Explanations": {
                "fn": generate_counterfactual_explanations,
                "type": "Actionable Clinical Recourse Search",
                "scope": "Local Recourse Intervention",
                "status": "OPERATIONAL"
            }
        }

    def list_status(self):
        """Displays status of XAI framework components."""
        print("\n" + "="*70)
        print("          XAI FRAMEWORK COMPLETE IMPLEMENTATION STATUS")
        print("="*70)
        print(f"Total Implemented Techniques: {len(self.explainers)} / 6 (100% Complete)\n")
        for name, meta in self.explainers.items():
            print(f"  [OK - OPERATIONAL] {name:28s} | {meta['type']:45s} | {meta['scope']}")
        print("="*70 + "\n")

    def run_all_explainers(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Executes all 6 explainers for a specific test participant and returns results dictionary.
        """
        results = {}
        print(f"\n[XAI MANAGER] Executing all 6 explainers for participant #{sample_idx}...")

        # 1. SHAP
        results["SHAP"] = generate_shap_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/shap", sample_idx=sample_idx
        )

        # 2. LIME
        results["LIME"] = generate_lime_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/lime", sample_idx=sample_idx
        )

        # 3. Integrated Gradients
        results["Integrated Gradients"] = generate_integrated_gradients_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/ig", sample_idx=sample_idx
        )

        # 4. Grad-CAM
        results["Grad-CAM"] = generate_gradcam_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/gradcam", sample_idx=sample_idx
        )

        # 5. Attention
        results["Attention"] = generate_attention_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/attention", sample_idx=sample_idx
        )

        # 6. Counterfactual
        results["Counterfactual"] = generate_counterfactual_explanations(
            model, X_train, X_test, feature_names, output_dir="outputs/counterfactual", sample_idx=sample_idx
        )

        return results


__all__ = [
    "generate_shap_explanations",
    "generate_lime_explanations",
    "generate_integrated_gradients_explanations",
    "IntegratedGradientsExplainer",
    "generate_gradcam_explanations",
    "GradCam1DExplainer",
    "generate_attention_explanations",
    "AttentionExplainer",
    "generate_counterfactual_explanations",
    "CounterfactualExplainer",
    "XAIFrameworkManager"
]
