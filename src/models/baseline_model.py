"""
Baseline and Deep Learning Model training for psychometric assessment.
Supports Random Forest, XGBoost, Logistic Regression, Neural MLP, 1D CNN, and Transformer.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
from typing import Any
try:
    from .neural_model import train_neural_model, PyTorchClassifierWrapper
except Exception:  # torch not installed or other import error
    train_neural_model = None
    PyTorchClassifierWrapper = None


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    random_state: int = 42
) -> Any:
    """
    Trains a classifier on the real preprocessed psychometric dataset.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        model_type: 'random_forest', 'xgboost', 'logistic_regression',
                    'neural_mlp', 'conv1d', or 'transformer'.
        random_state: Seed for reproducibility.
        
    Returns:
        Trained scikit-learn, XGBoost, or PyTorch model instance.
    """
    print(f"\n[MODEL TRAINER] Training model: '{model_type}' on {len(X_train):,} samples...")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    elif model_type in ["neural_mlp", "mlp"]:
        model = train_neural_model(X_train, y_train, arch="mlp", epochs=10, lr=0.001)
    elif model_type in ["conv1d", "cnn"]:
        model = train_neural_model(X_train, y_train, arch="conv1d", epochs=10, lr=0.001)
    elif model_type in ["transformer", "attention_net"]:
        model = train_neural_model(X_train, y_train, arch="transformer", epochs=10, lr=0.001)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"[MODEL TRAINER] Model training complete.")
    return model
