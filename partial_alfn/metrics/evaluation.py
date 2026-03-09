#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def _predict_for_eval(predictor, X: torch.Tensor) -> torch.Tensor:
    if hasattr(predictor, "forward_sink"):
        return predictor.forward_sink(X)
    return predictor(X)


def compute_test_loss(
    predictor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    task: str = "regression",
) -> float:
    predictor.eval()
    with torch.no_grad():
        out = _predict_for_eval(predictor, test_X)

        if task == "classification":
            loss = F.cross_entropy(out, test_y.long())
        else:
            loss = F.mse_loss(out.view_as(test_y), test_y)

    return float(loss.item())