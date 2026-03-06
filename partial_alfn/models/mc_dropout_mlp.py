#!/usr/bin/env python3

import torch
import torch.nn as nn


class MCDropoutMLP(nn.Module):
    """
    Simple sink-only MLP for regression with dropout.
    This is the stage-1 model before introducing multi-head partial-observation support.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),  # regression scalar (sink output only)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)