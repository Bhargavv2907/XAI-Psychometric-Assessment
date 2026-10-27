"""
Attention-Based Explainer for Psychometric Assessment Models.
Extracts and visualizes multi-head self-attention weights from Transformer encoders
to reveal symptom-to-symptom dependencies and clinical cross-talk.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from typing import Any, List, Dict, Tuple, Optional
from src.models.neural_model import PsychometricTransformer, PyTorchClassifierWrapper, train_neural_model


class ExplicitMultiheadAttentionTransformer(nn.Module):
    """
    Psychometric Transformer that explicitly stores multi-head self-attention weights
    for explainability inspection and cross-symptom relationship analysis.
    """
    def __init__(self, num_items: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.nhead = nhead

        self.item_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_items, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x_seq = x.unsqueeze(-1)  # (B, N, 1)
        else:
            x_seq = x
        h = self.item_embed(x_seq) + self.pos_embed  # (B, N, d_model)
        attn_out, attn_weights = self.mha(h, h, h, need_weights=True, average_attn_weights=False)
        self.last_attn_weights = attn_weights.detach()  # (B, nhead, N, N)

        h = self.norm(h + attn_out)
        h = h + self.ffn(h)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, self.last_attn_weights
        return logits


class AttentionExplainer:
    """
    Extracts, summarizes, and visualizes self-attention weights across psychometric symptoms.
    """

    def __init__(self, transformer_model: nn.Module):
        self.model = transformer_model

    def explain_instance(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Extracts multi-head attention matrix for a single psychometric profile.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if hasattr(self.model, "forward") and "return_attention" in self.model.forward.__code__.co_varnames:
                logits, attn_weights = self.model(x_tensor, return_attention=True)
            else:
                logits = self.model(x_tensor)
                attn_weights = getattr(self.model, "last_attn_weights", None)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        if attn_weights is not None:
            # attn_weights shape: (1, nhead, N, N) or (1, N, N)
            if attn_weights.dim() == 4:
                raw_attn = attn_weights.squeeze(0).cpu().numpy()  # (nhead, N, N)
                avg_attn = raw_attn.mean(axis=0)  # (N, N)
            else:
                raw_attn = attn_weights.squeeze(0).cpu().numpy()
                avg_attn = raw_attn
        else:
            # Fallback synthetic attention from feature correlations
            N = x.shape[0]
            avg_attn = np.ones((N, N)) / N
            raw_attn = avg_attn[np.newaxis, ...]

        # Calculate Symptom Self-Attention Centrality (how much other items attend to item i)
        in_attention = avg_attn.sum(axis=0)
        # Calculate Symptom Query Attention (how much item i attends to others)
        out_attention = avg_attn.sum(axis=1)

        return {
            "prediction_probs": probs,
            "avg_attention": avg_attn,
            "head_attentions": raw_attn,
            "in_attention": in_attention,
            "out_attention": out_attention
        }


def generate_attention_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: str = "outputs/attention",
    sample_idx: int = 0
) -> Dict[str, Any]:
    """
    Computes self-attention interaction weights for psychometric items.
    Generates cross-symptom attention heatmaps and top dependency pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[ATTENTION EXPLAINER] Initializing Self-Attention Analysis...")

    num_items = len(feature_names)
    transformer = ExplicitMultiheadAttentionTransformer(num_items=num_items, d_model=64, nhead=4)

    # Train transformer encoder
    print("[ATTENTION] Training Transformer Self-Attention encoder on psychometric dataset...")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(model.predict(X_train) if hasattr(model, "predict") else np.zeros(len(X_train)), dtype=torch.int64)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    transformer.train()
    for epoch in range(8):
        for bx, by in loader:
            optimizer.zero_grad()
            out = transformer(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    explainer = AttentionExplainer(transformer)
    sample_x = X_test.iloc[sample_idx].values.astype(np.float32)
    result = explainer.explain_instance(sample_x)

    attn_matrix = result["avg_attention"]

    # 1. Plot Subscale-to-Subscale Attention Heatmap
    subscale_keys = ["Depression", "Anxiety", "Stress", "Personality", "Demographics"]
    subscale_indices = {k: [] for k in subscale_keys}
    for i, name in enumerate(feature_names):
        if "Depression" in name:
            subscale_indices["Depression"].append(i)
        elif "Anxiety" in name:
            subscale_indices["Anxiety"].append(i)
        elif "Stress" in name:
            subscale_indices["Stress"].append(i)
        elif "Big5" in name:
            subscale_indices["Personality"].append(i)
        else:
            subscale_indices["Demographics"].append(i)

    subscale_attn = np.zeros((len(subscale_keys), len(subscale_keys)))
    for r, k1 in enumerate(subscale_keys):
        for c, k2 in enumerate(subscale_keys):
            idx1, idx2 = subscale_indices[k1], subscale_indices[k2]
            if len(idx1) > 0 and len(idx2) > 0:
                block = attn_matrix[np.ix_(idx1, idx2)]
                subscale_attn[r, c] = block.mean()

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        subscale_attn,
        annot=True,
        fmt=".3f",
        cmap="Purples",
        xticklabels=subscale_keys,
        yticklabels=subscale_keys,
        cbar_kws={"label": "Mean Attention Weight"}
    )
    plt.title(f"Clinical Subscale Cross-Attention Matrix (Participant #{sample_idx})", fontsize=13, fontweight="bold")
    plt.xlabel("Attended Subscale (Key)", fontsize=11)
    plt.ylabel("Source Subscale (Query)", fontsize=11)
    plt.tight_layout()
    subscale_plot_path = os.path.join(output_dir, "attention_subscale_matrix.png")
    plt.savefig(subscale_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Top Symptom-Symptom Attention Pairs
    item_pairs = []
    N = len(feature_names)
    for i in range(N):
        for j in range(N):
            if i != j:
                item_pairs.append((feature_names[i], feature_names[j], attn_matrix[i, j]))

    item_pairs.sort(key=lambda x: x[2], reverse=True)
    top_10_pairs = item_pairs[:10]

    plt.figure(figsize=(10, 6))
    pair_labels = [f"{p[0].split(':')[0]} → {p[1].split(':')[0]}" for p in top_10_pairs][::-1]
    pair_weights = [p[2] for p in top_10_pairs][::-1]

    plt.barh(pair_labels, pair_weights, color="#756bb1")
    plt.xlabel("Attention Coupling Weight", fontsize=12)
    plt.title(f"Top 10 Symptom-to-Symptom Cross-Attention Links (Participant #{sample_idx})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    top_pairs_path = os.path.join(output_dir, "attention_top_symptom_pairs.png")
    plt.savefig(top_pairs_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print Summary
    print("\n" + "="*60)
    print(f"       ATTENTION EXPLAINER REPORT (Participant #{sample_idx})")
    print("="*60)
    print("Top Symptom Interaction Couplings (Self-Attention):")
    for rank, (src, dst, weight) in enumerate(top_10_pairs[:5], 1):
        print(f"  {rank}. [{src.split(':')[0]}] attends strongly to [{dst.split(':')[0]}]: Weight = {weight:.4f}")
    print("="*60 + "\n")

    print(f"[ATTENTION] Subscale matrix saved to: {subscale_plot_path}")
    print(f"[ATTENTION] Top symptom pairs saved to: {top_pairs_path}")

    return {
        "attention_matrix": attn_matrix,
        "subscale_matrix": subscale_attn,
        "subscale_plot": subscale_plot_path,
        "pairs_plot": top_pairs_path,
        "top_pairs": top_10_pairs[:5]
    }
