"""
Neural Network Architectures for Psychometric Assessment.
Provides differentiable PyTorch models for:
  1. PsychometricMLP (Multilayer Perceptron for tabular questionnaire data)
  2. PsychometricConv1D (1D CNN for item response & latency sequences)
  3. PsychometricTransformer (Self-attention encoder for symptom-symptom interactions)
All models provide scikit-learn compatible predict and predict_proba interfaces.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional, Dict, Any


class PsychometricMLP(nn.Module):
    """
    Multilayer Perceptron for tabular psychometric assessment.
    Fully differentiable for gradient-based explainers (Integrated Gradients).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PsychometricConv1D(nn.Module):
    """
    1D Convolutional Neural Network for psychometric item-response sequences and latency profiles.
    Used directly with Grad-CAM to localize activation heatmaps across symptom clusters.
    """
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        # Input shape: (Batch, 1, input_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        pooled_dim = input_dim // 4
        self.fc = nn.Sequential(
            nn.Linear(64 * pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.target_conv_layer = self.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PsychometricTransformer(nn.Module):
    """
    Transformer Self-Attention encoder for psychometric assessment.
    Models cross-item dependencies and extracts multi-head attention weights.
    """
    def __init__(self, num_items: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.item_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_items, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_items)
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1)  # (B, num_items, 1)
        h = self.item_embedding(x_expanded) + self.pos_embedding  # (B, num_items, d_model)
        encoded = self.transformer_encoder(h)  # (B, num_items, d_model)
        # Pool across item dimension (mean pooling)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)


class PyTorchClassifierWrapper:
    """
    Wrapper providing scikit-learn API (fit, predict, predict_proba) for PyTorch models.
    """
    def __init__(self, model: nn.Module, lr: float = 0.001, epochs: int = 15, batch_size: int = 128, device: str = "cpu"):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.array(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.values.astype(np.int64).flatten()
        else:
            y_arr = np.array(y, dtype=np.int64).flatten()

        dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_arr))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.array(X, dtype=np.float32)

        with torch.no_grad():
            tensor_x = torch.tensor(X_arr, device=self.device)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_neural_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    arch: str = "mlp",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001
) -> PyTorchClassifierWrapper:
    """
    Factory function to train a neural psychometric model.
    Supported architectures: 'mlp', 'conv1d', 'transformer'.
    """
    input_dim = X_train.shape[1]
    if arch == "mlp":
        raw_model = PsychometricMLP(input_dim=input_dim)
    elif arch == "conv1d":
        raw_model = PsychometricConv1D(input_dim=input_dim)
    elif arch == "transformer":
        raw_model = PsychometricTransformer(num_items=input_dim)
    else:
        raise ValueError(f"Unknown neural architecture: {arch}")

    wrapper = PyTorchClassifierWrapper(raw_model, lr=lr, epochs=epochs, batch_size=batch_size)
    wrapper.fit(X_train, y_train)
    return wrapper
