"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Psychometric Models.
Adapted for 1D Psychometric CNNs, behavioral response sequences, and physiological signals.
Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Any, List, Dict, Optional, Tuple
from src.models.neural_model import PsychometricConv1D, PyTorchClassifierWrapper, train_neural_model


class GradCam1DExplainer:
    """
    Computes 1D Grad-CAM activations across psychometric item sequences and latency profiles.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else getattr(model, "target_conv_layer", None)
        if self.target_layer is None:
            # Find last Conv1d layer
            for module in reversed(list(model.modules())):
                if isinstance(module, nn.Conv1d):
                    self.target_layer = module
                    break

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        if self.target_layer is not None:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generates 1D Grad-CAM heatmap for input tensor of shape (1, 1, L) or (1, L).
        """
        self.model.eval()
        if x.dim() == 2:
            x = x.unsqueeze(1)

        device = next(self.model.parameters()).device
        x = x.to(device)

        logits = self.model(x)
        score = logits[:, target_class]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Activations: (1, Channels, Length_conv)
        # Gradients: (1, Channels, Length_conv)
        grads = self.gradients
        acts = self.activations

        # Global average pooling of gradients across temporal/item dimension
        weights = grads.mean(dim=-1, keepdim=True)  # (1, Channels, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, Length_conv)
        cam = torch.relu(cam)  # Only features that positively contribute to target class

        # Upsample back to original input length
        orig_length = x.size(-1)
        cam_upsampled = nn.functional.interpolate(cam, size=orig_length, mode="linear", align_corners=False)
        cam_np = cam_upsampled.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm


def generate_gradcam_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/gradcam",
    sample_idx: int = 0
) -> Dict[str, Any]:
    """
    Computes 1D Grad-CAM localization on psychometric item response patterns.
    Visualizes subscale activation heatmaps (Depression, Anxiety, Stress, Personality).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[GRAD-CAM EXPLAINER] Initializing 1D Class Activation Mapping...")

    # Ensure we have a 1D CNN model
    if not isinstance(model, PyTorchClassifierWrapper) or not isinstance(getattr(model, "model", None), PsychometricConv1D):
        print("[GRAD-CAM] Training specialized 1D Psychometric CNN for sequence representation...")
        cnn_wrapper = train_neural_model(X_train, model.predict(X_train), arch="conv1d", epochs=10)
        net = cnn_wrapper.model
    else:
        net = model.model

    explainer = GradCam1DExplainer(net)

    sample_vals = X_test.iloc[sample_idx].values.astype(np.float32)
    sample_tensor = torch.tensor(sample_vals).unsqueeze(0)  # (1, L)

    cam_heatmap = explainer.generate_cam(sample_tensor, target_class=1)

    # Group heatmap by clinical psychometric subscales
    subscales = {"Depression": [], "Anxiety": [], "Stress": [], "Personality": [], "Demographics": []}
    for i, name in enumerate(feature_names):
        val = cam_heatmap[i]
        if "Depression" in name:
            subscales["Depression"].append(val)
        elif "Anxiety" in name:
            subscales["Anxiety"].append(val)
        elif "Stress" in name:
            subscales["Stress"].append(val)
        elif "Big5" in name:
            subscales["Personality"].append(val)
        else:
            subscales["Demographics"].append(val)

    subscale_means = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in subscales.items()}

    # 1. Plot Grad-CAM 1D Item Activation Profile
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.5, 1]})

    x_indices = np.arange(len(feature_names))
    ax1.plot(x_indices, cam_heatmap, color="#d95f02", lw=2.5, label="Grad-CAM Activation Intensity")
    ax1.fill_between(x_indices, 0, cam_heatmap, color="#d95f02", alpha=0.3)
    ax1.scatter(x_indices, cam_heatmap, c=cam_heatmap, cmap="YlOrRd", s=35, zorder=5)

    # Highlight top 5 highest activated items
    top_5_cam_idx = np.argsort(cam_heatmap)[-5:]
    for idx in top_5_cam_idx:
        ax1.annotate(
            feature_names[idx].split(":")[0],
            (idx, cam_heatmap[idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
        )

    ax1.set_ylabel("Grad-CAM Activation (0-1)", fontsize=12)
    ax1.set_title(f"Grad-CAM 1D Activation Map Across Psychometric Items (Participant #{sample_idx})", fontsize=14, fontweight="bold")
    ax1.set_ylim(-0.05, 1.15)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Subscale Heatmap Matrix
    subscale_names = list(subscale_means.keys())
    subscale_vals = np.array([[subscale_means[k] for k in subscale_names]])
    im = ax2.imshow(subscale_vals, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(subscale_names)))
    ax2.set_xticklabels(subscale_names, fontsize=11, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_title("Aggregated Clinical Subscale Activation Energy", fontsize=12, fontweight="bold")

    for j, val in enumerate(subscale_vals[0]):
        text_color = "white" if val > 0.5 else "black"
        ax2.text(j, 0, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=12)

    plt.tight_layout()
    gradcam_path = os.path.join(output_dir, "gradcam_activation_map.png")
    plt.savefig(gradcam_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print(f"       GRAD-CAM ACTIVATION REPORT (Participant #{sample_idx})")
    print("="*60)
    print("Clinical Subscale Mean Activation Energies:")
    for k, v in subscale_means.items():
        bar_len = int(v * 25)
        bar_str = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  {k:15s}: [{bar_str}] {v:.4f}")

    print("\nTop Activated Psychometric Items:")
    for rank, idx in enumerate(reversed(top_5_cam_idx), 1):
        print(f"  {rank}. {feature_names[idx]}: Activation = {cam_heatmap[idx]:.4f}")
    print("="*60 + "\n")

    print(f"[GRAD-CAM] Visualization saved to: {gradcam_path}")

    return {
        "cam_heatmap": cam_heatmap,
        "subscale_means": subscale_means,
        "plot_path": gradcam_path,
        "top_items": [(feature_names[i], float(cam_heatmap[i])) for i in reversed(top_5_cam_idx)]
    }
