"""CNN architecture and input encoding used in the common-survey comparison."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn


class PaperCNN(nn.Module):
    """Architecture in Section IV-A, with valid 2x1 convolutions."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(2, 1)),
            nn.ReLU(),
        )
        self.output = nn.Sequential(nn.Linear(768, 20), nn.ReLU(), nn.Linear(20, 2))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(tensor).flatten(1)
        if features.shape[1] != 768:
            raise RuntimeError(f"Paper flattening must be 768, got {features.shape}")
        return self.output(features)


def encode(
    delays: np.ndarray,
    rp_positions: np.ndarray,
    ap_position: np.ndarray,
    delay_min: float,
    delay_max: float,
) -> np.ndarray:
    normalized = (delays - delay_min) / max(delay_max - delay_min, 1e-12)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.concatenate((normalized, rp_positions.reshape(-1), ap_position.reshape(-1)))
    if values.size != 72:
        raise ValueError(f"Expected 9 + 20*3 + 3 = 72 values, got {values.size}")
    return values.reshape(1, 6, 12).astype(np.float32)
