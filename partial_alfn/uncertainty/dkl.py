#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.nn as nn


def _validate_predictor_interface(predictor: nn.Module) -> None:
    required_methods = [
        "predict_node_mean_var",
        "rollout_samples_from_base",
        "rollout_means_from_base",
    ]
    missing = [name for name in required_methods if not hasattr(predictor, name)]
    if len(missing) > 0:
        raise TypeError(
            "Predictor does not satisfy the DKL uncertainty interface. "
            f"Missing methods: {missing}"
        )


def _validate_2d_input(X: torch.Tensor, name: str = "X") -> None:
    if not torch.is_tensor(X):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(X.shape)}")


@torch.no_grad()
def dkl_predict_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    node_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict posterior mean / variance for one node in node-input space.

    Parameters
    ----------
    predictor:
        A MultiHeadNodewiseDKL-like model exposing:
            predict_node_mean_var(X, node_idx)
    X:
        Node-specific input tensor of shape [N, d_node]
    node_idx:
        Node index

    Returns
    -------
    mean: torch.Tensor
        Shape [N, 1]
    var: torch.Tensor
        Shape [N, 1]
    """
    _validate_predictor_interface(predictor)
    _validate_2d_input(X, name="X")

    mean, var = predictor.predict_node_mean_var(X, node_idx)

    if mean.ndim != 2 or mean.shape[1] != 1:
        raise ValueError(
            f"Expected mean of shape [N, 1], got {tuple(mean.shape)}"
        )
    if var.ndim != 2 or var.shape[1] != 1:
        raise ValueError(
            f"Expected var of shape [N, 1], got {tuple(var.shape)}"
        )

    return mean, var


@torch.no_grad()
def dkl_sample_all_nodes_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Draw ancestral latent samples through the function-network DAG.

    Parameters
    ----------
    predictor:
        A MultiHeadNodewiseDKL-like model exposing:
            rollout_samples_from_base(X, n_samples)
    X:
        External-input tensor of shape [N, d_ext]
    n_samples:
        Number of ancestral posterior samples

    Returns
    -------
    samples: torch.Tensor
        Shape [S, N, n_nodes]
    """
    _validate_predictor_interface(predictor)
    _validate_2d_input(X, name="X")

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    samples = predictor.rollout_samples_from_base(X, n_samples=n_samples)

    if samples.ndim != 3:
        raise ValueError(
            f"Expected samples of shape [S, N, n_nodes], got {tuple(samples.shape)}"
        )
    if samples.shape[0] != n_samples:
        raise ValueError(
            f"Expected first dim to equal n_samples={n_samples}, "
            f"got {samples.shape[0]}"
        )

    return samples


@torch.no_grad()
def dkl_predict_mean_var_all_nodes(
    predictor: nn.Module,
    X: torch.Tensor,
    n_samples: int = 64,
    unbiased: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict mean / variance for all nodes from external-input space.

    This uses ancestral rollout sampling through the DAG, then computes
    empirical mean and variance across samples. This mirrors the MCD design
    where upstream uncertainty propagates into downstream nodes.

    Parameters
    ----------
    predictor:
        A MultiHeadNodewiseDKL-like model exposing:
            rollout_samples_from_base(X, n_samples)
    X:
        External-input tensor of shape [N, d_ext]
    n_samples:
        Number of rollout samples used to estimate mean / variance
    unbiased:
        Whether to use unbiased variance estimate

    Returns
    -------
    mean_all: torch.Tensor
        Shape [N, n_nodes]
    var_all: torch.Tensor
        Shape [N, n_nodes]
    """
    samples = dkl_sample_all_nodes_from_base(
        predictor=predictor,
        X=X,
        n_samples=n_samples,
    )  # [S, N, n_nodes]

    mean_all = samples.mean(dim=0)  # [N, n_nodes]
    var_all = samples.var(dim=0, unbiased=unbiased)  # [N, n_nodes]

    return mean_all, var_all


@torch.no_grad()
def dkl_predict_sink_mean_var_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
    n_samples: int = 64,
    unbiased: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict sink-node mean / variance from external-input space.

    Returns
    -------
    mean_sink: torch.Tensor
        Shape [N, 1]
    var_sink: torch.Tensor
        Shape [N, 1]
    """
    mean_all, var_all = dkl_predict_mean_var_all_nodes(
        predictor=predictor,
        X=X,
        n_samples=n_samples,
        unbiased=unbiased,
    )

    if not hasattr(predictor, "sink_idx"):
        raise AttributeError("Predictor must expose sink_idx.")

    sink_idx = int(predictor.sink_idx)

    mean_sink = mean_all[:, [sink_idx]]
    var_sink = var_all[:, [sink_idx]]
    return mean_sink, var_sink


@torch.no_grad()
def dkl_predict_means_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic mean rollout from external-input space.

    Returns
    -------
    y_all: torch.Tensor
        Shape [N, n_nodes]
    """
    _validate_predictor_interface(predictor)
    _validate_2d_input(X, name="X")

    y_all = predictor.rollout_means_from_base(X)

    if y_all.ndim != 2:
        raise ValueError(
            f"Expected rollout means of shape [N, n_nodes], got {tuple(y_all.shape)}"
        )

    return y_all