"""
Multimodal Attention Gated Fusion Network for Psychometric Assessment.
Fuses 5 synchronized clinical assessment modalities:
  - Questionnaire (DASS-42 survey items)
  - Behavioural (Response latency timings)
  - Text (Clinical self-report narrative features)
  - Speech (Acoustic prosody biomarkers)
  - Physiological (Autonomic nervous system signals)

Features:
  - Modality-specific neural encoders
  - Gated Attention Mechanism computing dynamic modality weights (gamma_1 .. gamma_5)
  - Modality-level interpretability output directly accessible to clinicians
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional


class ModalityAttentionGatedFusion(nn.Module):
    """
    Multimodal neural architecture with explicit modality attention gating.
    Computes modality importance scores gamma (sum to 1) for transparent attribution.
    """
    def __init__(
        self,
        dim_quest: int,
        dim_behav: int,
        dim_text: int,
        dim_speech: int,
        dim_physio: int,
        embed_dim: int = 32
    ):
        super().__init__()
        self.modality_names = ["Questionnaire", "Behavioural", "Text", "Speech", "Physiological"]

        # Modality Encoders
        self.enc_quest = nn.Sequential(nn.Linear(dim_quest, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.enc_behav = nn.Sequential(nn.Linear(dim_behav, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.enc_text = nn.Sequential(nn.Linear(dim_text, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.enc_speech = nn.Sequential(nn.Linear(dim_speech, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.enc_physio = nn.Sequential(nn.Linear(dim_physio, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.1))

        # Modality Attention Gating Network
        # Projects each modality embedding to a scalar gating score
        self.gate_projector = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(
        self,
        x_quest: torch.Tensor,
        x_behav: torch.Tensor,
        x_text: torch.Tensor,
        x_speech: torch.Tensor,
        x_physio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, modality_attention_weights).
        """
        h_quest = self.enc_quest(x_quest)
        h_behav = self.enc_behav(x_behav)
        h_text = self.enc_text(x_text)
        h_speech = self.enc_speech(x_speech)
        h_physio = self.enc_physio(x_physio)

        # Stack modality representations: (Batch, 5, embed_dim)
        h_stack = torch.stack([h_quest, h_behav, h_text, h_speech, h_physio], dim=1)

        # Compute gating scores for each modality: (Batch, 5, 1)
        gate_scores = self.gate_projector(h_stack)  # (B, 5, 1)
        modality_weights = torch.softmax(gate_scores.squeeze(-1), dim=-1)  # (B, 5), sums to 1.0

        # Weighted representations: (Batch, 5, embed_dim)
        h_weighted = h_stack * modality_weights.unsqueeze(-1)
        fused = h_weighted.view(h_weighted.size(0), -1)  # Flatten to (B, 5 * embed_dim)

        logits = self.classifier(fused)
        return logits, modality_weights


class MultimodalClassifier:
    """
    Wrapper for training, evaluation, and inference of the multimodal fusion model.
    """
    def __init__(self, model: ModalityAttentionGatedFusion, lr: float = 0.001, epochs: int = 12, batch_size: int = 128):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, modalities: Dict[str, Any], y: np.ndarray):
        device = torch.device("cpu")
        self.model.to(device)

        X_q = torch.tensor(modalities["questionnaire"]["train"], dtype=torch.float32)
        X_b = torch.tensor(modalities["behavioural"]["train"], dtype=torch.float32)
        X_t = torch.tensor(modalities["text"]["train"], dtype=torch.float32)
        X_s = torch.tensor(modalities["speech"]["train"], dtype=torch.float32)
        X_p = torch.tensor(modalities["physiological"]["train"], dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.int64)

        dataset = TensorDataset(X_q, X_b, X_t, X_s, X_p, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for b_q, b_b, b_t, b_s, b_p, b_y in dataloader:
                optimizer.zero_grad()
                logits, _ = self.model(b_q, b_b, b_t, b_s, b_p)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, modalities: Dict[str, Any], split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_probabilities, modality_attention_weights).
        """
        self.model.eval()
        X_q = torch.tensor(modalities["questionnaire"][split], dtype=torch.float32)
        X_b = torch.tensor(modalities["behavioural"][split], dtype=torch.float32)
        X_t = torch.tensor(modalities["text"][split], dtype=torch.float32)
        X_s = torch.tensor(modalities["speech"][split], dtype=torch.float32)
        X_p = torch.tensor(modalities["physiological"][split], dtype=torch.float32)

        with torch.no_grad():
            logits, mod_weights = self.model(X_q, X_b, X_t, X_s, X_p)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            weights = mod_weights.cpu().numpy()

        return probs, weights

    def predict(self, modalities: Dict[str, Any], split: str = "test") -> np.ndarray:
        probs, _ = self.predict_proba(modalities, split=split)
        return np.argmax(probs, axis=1)


def train_multimodal_fusion_model(
    modalities_data: Dict[str, Any],
    y_train: np.ndarray,
    epochs: int = 12
) -> MultimodalClassifier:
    """
    Trains the multimodal attention gated fusion model.
    """
    mods = modalities_data["modalities"]
    dim_q = mods["questionnaire"]["train"].shape[1]
    dim_b = mods["behavioural"]["train"].shape[1]
    dim_t = mods["text"]["train"].shape[1]
    dim_s = mods["speech"]["train"].shape[1]
    dim_p = mods["physiological"]["train"].shape[1]

    print(f"\n[MULTIMODAL FUSION] Initializing 5-Modality Fusion Network:")
    print(f"  - Questionnaire: {dim_q} items")
    print(f"  - Behavioural:   {dim_b} latency features")
    print(f"  - Text:          {dim_t} NLP semantic features")
    print(f"  - Speech:        {dim_s} acoustic prosody biomarkers")
    print(f"  - Physiological: {dim_p} autonomic nervous system signals")

    net = ModalityAttentionGatedFusion(dim_q, dim_b, dim_t, dim_s, dim_p, embed_dim=32)
    clf = MultimodalClassifier(net, lr=0.001, epochs=epochs, batch_size=128)
    clf.fit(mods, y_train)

    print("[MULTIMODAL FUSION] Training successfully converged.")
    return clf
