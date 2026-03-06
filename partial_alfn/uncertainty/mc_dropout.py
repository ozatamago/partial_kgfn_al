#!/usr/bin/env python3

import torch
import torch.nn as nn


def _enable_mc_dropout(m: nn.Module) -> None:
    """
    Turn on dropout layers only, while keeping the rest of the model in eval mode.
    This is the standard test-time MC-dropout pattern.
    """
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


@torch.no_grad()
def mc_predict_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
):
    """
    MC-dropout prediction for sink-only regression model.

    Args:
        predictor: maps X -> y_hat with shape [N, 1]
        X: tensor of shape [N, d]
        mc_samples: number of stochastic forward passes

    Returns:
        mean: [N, 1]
        var:  [N, 1]
    """
    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(mc_samples):
        outs.append(predictor(X))  # expected shape [N, 1]

    Y = torch.stack(outs, dim=0)   # [S, N, 1]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=False)
    return mean, var