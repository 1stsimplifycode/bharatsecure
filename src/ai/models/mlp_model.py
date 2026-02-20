"""
MLP Gesture Classifier Model
Lightweight 3-layer MLP — edge-optimised for Jetson Nano.
Trained with scikit-learn, exported to TFLite for inference.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Gesture class definitions
GESTURE_CLASSES = {
    0: "stop",        # Open palm
    1: "play",        # One finger up
    2: "volume_up",   # Two fingers up
    3: "volume_down", # Two fingers pointing down
    4: "mute",        # Closed fist
}

# Architecture: 63 → 128 → 64 → 5
# Kept lightweight for <5MB model size on Jetson Nano


def build_mlp_sklearn() -> MLPClassifier:
    """
    Build a lightweight MLP using scikit-learn.
    No GPU required for training; fast inference on CPU.
    Cost: $0
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,           # L2 regularisation
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )


def build_mlp_torch():
    """
    PyTorch MLP for federated learning weight sharing.
    Same architecture as sklearn model.
    """
    try:
        import torch
        import torch.nn as nn

        class GestureMLP(nn.Module):
            def __init__(self, input_dim=63, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        return GestureMLP()
    except ImportError:
        logger.warning("PyTorch not available. Federated learning will use sklearn.")
        return None
