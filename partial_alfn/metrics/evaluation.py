#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def compute_test_loss(
    predictor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    task: str = "regression",
) -> float:
    """
    Compute average test loss for the current predictor.

    Args:
        predictor:
            NN model mapping x -> logits (classification) or y_hat (regression)
        test_X:
            shape [n_te, d]
        test_y:
            shape [n_te] for classification labels,
            or [n_te, 1] for regression
        task:
            "classification" or "regression"

    Returns:
        Scalar float test loss.
    """
    predictor.eval()
    with torch.no_grad():
        out = predictor(test_X)

        if task == "classification":
            loss = F.cross_entropy(out, test_y.long())
        else:
            loss = F.mse_loss(out.view_as(test_y), test_y)

    return float(loss.item())