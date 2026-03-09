#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F


def _predict_sink(predictor: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
    if hasattr(predictor, "forward_sink"):
        return predictor.forward_sink(X)
    return predictor(X)


def train_predictor_regression(
    predictor: torch.nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    predictor.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    n = train_X.shape[0]
    if n == 0:
        return optimizer

    for _ in range(n_steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=train_X.device)
        xb = train_X[idx]
        yb = train_y[idx]

        optimizer.zero_grad()
        pred = _predict_sink(predictor, xb)
        loss = F.mse_loss(pred.view_as(yb), yb)
        loss.backward()
        optimizer.step()

    return optimizer