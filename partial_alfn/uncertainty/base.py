#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from partial_alfn.uncertainty.dkl import (
    dkl_predict_mean_var,
    dkl_predict_mean_var_all_nodes,
    dkl_sample_all_nodes_from_base,
)
from partial_alfn.uncertainty.mc_dropout import (
    mc_predict_mean_var_all_nodes,
)


def _validate_2d_input(X: torch.Tensor, name: str = "X") -> None:
    if not torch.is_tensor(X):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(X.shape)}")


def _get_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {} if options is None else dict(options)


def _get_predictor_type(predictor: nn.Module) -> str:
    predictor_type = getattr(predictor, "predictor_type", None)
    if predictor_type is None:
        raise AttributeError(
            "predictor must expose predictor_type, e.g. 'mcd' or 'dkl'"
        )
    predictor_type = str(predictor_type).lower()
    if predictor_type not in ["mcd", "dkl"]:
        raise ValueError(
            f"Unsupported predictor_type: {predictor_type}. "
            "Expected 'mcd' or 'dkl'."
        )
    return predictor_type


def _get_sink_idx(predictor: nn.Module) -> int:
    if not hasattr(predictor, "sink_idx"):
        raise AttributeError("predictor must expose sink_idx")
    return int(predictor.sink_idx)


def _get_external_input_dim(predictor: nn.Module) -> int:
    if not hasattr(predictor, "external_input_dim"):
        raise AttributeError("predictor must expose external_input_dim")
    return int(predictor.external_input_dim)


def _get_node_input_dims(predictor: nn.Module):
    if not hasattr(predictor, "node_input_dims"):
        raise AttributeError("predictor must expose node_input_dims")
    return list(predictor.node_input_dims)


def _enable_mc_dropout(m: nn.Module) -> None:
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


@torch.no_grad()
def _mcd_predict_node_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    node_idx: int,
    mc_samples: int = 64,
    unbiased: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict posterior-like mean / variance for one node in node-input space
    using MC dropout.

    Parameters
    ----------
    predictor:
        Expected to expose forward_node(X, node_idx)
    X:
        Node-input-space tensor of shape [N, d_node]
    node_idx:
        Node index
    mc_samples:
        Number of MC dropout samples
    unbiased:
        Whether to use unbiased variance estimate

    Returns
    -------
    mean: torch.Tensor
        Shape [N, 1]
    var: torch.Tensor
        Shape [N, 1]
    """
    if mc_samples <= 0:
        raise ValueError(f"mc_samples must be positive, got {mc_samples}")

    if not hasattr(predictor, "forward_node"):
        raise AttributeError("predictor must expose forward_node(X, node_idx)")

    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(mc_samples):
        y = predictor.forward_node(X, node_idx)
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(
                f"Expected forward_node output of shape [N, 1], got {tuple(y.shape)}"
            )
        outs.append(y)

    Y = torch.stack(outs, dim=0)  # [S, N, 1]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=unbiased)
    return mean, var


@torch.no_grad()
def _mcd_sample_all_nodes_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Draw ancestral samples through the DAG using MC dropout.

    This assumes predictor exposes rollout_means_from_base(base_x), and that
    repeated calls under dropout-enabled mode produce stochastic samples.

    Returns
    -------
    samples: torch.Tensor
        Shape [S, N, n_nodes]
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if not hasattr(predictor, "rollout_means_from_base"):
        raise AttributeError(
            "predictor must expose rollout_means_from_base(base_x)"
        )

    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(n_samples):
        y_all = predictor.rollout_means_from_base(X)
        if y_all.ndim != 2:
            raise ValueError(
                f"Expected rollout_means_from_base output of shape [N, n_nodes], "
                f"got {tuple(y_all.shape)}"
            )
        outs.append(y_all)

    return torch.stack(outs, dim=0)  # [S, N, n_nodes]


@torch.no_grad()
def predict_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    node_idx: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backend-independent prediction for one target.

    Behavior
    --------
    1. If node_idx is not None and X is in node-input space for that node:
       return mean / variance for that node.

    2. If node_idx is not None and X is in external-input space:
       compute all-node mean / variance from base_x, then slice node_idx.

    3. If node_idx is None:
       compute sink-node mean / variance from external-input space.

    Parameters
    ----------
    predictor:
        MCD or DKL predictor exposing predictor_type.
    X:
        Either:
            - node-input-space tensor [N, d_node] when node_idx is specified, or
            - external-input-space tensor [N, d_ext]
    node_idx:
        Optional node index. If None, use sink_idx.
    options:
        Optional dict controlling sample counts:
            - mc_samples
            - n_samples
            - unbiased

    Returns
    -------
    mean: torch.Tensor
        Shape [N, 1]
    var: torch.Tensor
        Shape [N, 1]
    """
    _validate_2d_input(X, name="X")
    opts = _get_options(options)

    predictor_type = _get_predictor_type(predictor)
    sink_idx = _get_sink_idx(predictor)
    ext_dim = _get_external_input_dim(predictor)
    node_input_dims = _get_node_input_dims(predictor)

    unbiased = bool(opts.get("unbiased", False))
    target_idx = sink_idx if node_idx is None else int(node_idx)

    if not (0 <= target_idx < len(node_input_dims)):
        raise ValueError(
            f"node_idx must be in [0, {len(node_input_dims) - 1}], got {target_idx}"
        )

    # Case A: X is already node-input space for the requested node
    if X.shape[1] == node_input_dims[target_idx]:
        if predictor_type == "mcd":
            mc_samples = int(opts.get("mc_samples", 64))
            return _mcd_predict_node_mean_var(
                predictor=predictor,
                X=X,
                node_idx=target_idx,
                mc_samples=mc_samples,
                unbiased=unbiased,
            )

        if predictor_type == "dkl":
            return dkl_predict_mean_var(
                predictor=predictor,
                X=X,
                node_idx=target_idx,
            )

    # Case B: X is external-input space, so compute all nodes then slice
    if X.shape[1] == ext_dim:
        mean_all, var_all = predict_mean_var_all_nodes(
            predictor=predictor,
            X=X,
            options=opts,
        )
        return mean_all[:, [target_idx]], var_all[:, [target_idx]]

    raise ValueError(
        f"Input dim {X.shape[1]} matches neither node-input dim "
        f"{node_input_dims[target_idx]} for node {target_idx} nor "
        f"external_input_dim {ext_dim}"
    )


@torch.no_grad()
def predict_mean_var_all_nodes(
    predictor: nn.Module,
    X: torch.Tensor,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backend-independent mean / variance prediction for all nodes from
    external-input space.

    Parameters
    ----------
    predictor:
        MCD or DKL predictor exposing predictor_type.
    X:
        External-input-space tensor [N, d_ext]
    options:
        Optional dict controlling sample counts:
            - mc_samples (for MCD)
            - n_samples (for DKL)
            - unbiased

    Returns
    -------
    mean_all: torch.Tensor
        Shape [N, n_nodes]
    var_all: torch.Tensor
        Shape [N, n_nodes]
    """
    _validate_2d_input(X, name="X")
    opts = _get_options(options)

    predictor_type = _get_predictor_type(predictor)
    ext_dim = _get_external_input_dim(predictor)
    if X.shape[1] != ext_dim:
        raise ValueError(
            f"X must be in external-input space with dim {ext_dim}, "
            f"got shape {tuple(X.shape)}"
        )

    if predictor_type == "mcd":
        mc_samples = int(opts.get("mc_samples", 64))
        return mc_predict_mean_var_all_nodes(
            predictor=predictor,
            X=X,
            mc_samples=mc_samples,
        )

    if predictor_type == "dkl":
        n_samples = int(opts.get("n_samples", opts.get("mc_samples", 64)))
        unbiased = bool(opts.get("unbiased", False))
        return dkl_predict_mean_var_all_nodes(
            predictor=predictor,
            X=X,
            n_samples=n_samples,
            unbiased=unbiased,
        )

    raise RuntimeError("Unreachable predictor type branch.")

@torch.no_grad()
def predict_sink_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for sink prediction from external-input space.

    Equivalent to:
        predict_mean_var(predictor, X, node_idx=None, options=options)

    Returns
    -------
    mean_sink: torch.Tensor
        Shape [N, 1]
    var_sink: torch.Tensor
        Shape [N, 1]
    """
    return predict_mean_var(
        predictor=predictor,
        X=X,
        node_idx=None,
        options=options,
    )


@torch.no_grad()
def predict_node_mean_var_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
    node_idx: int,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for predicting one node from external-input space.

    This is mainly a readability helper for selectors / runners that always
    start from base input space.
    """
    _validate_2d_input(X, name="X")
    ext_dim = _get_external_input_dim(predictor)
    if X.shape[1] != ext_dim:
        raise ValueError(
            f"X must be in external-input space with dim {ext_dim}, "
            f"got shape {tuple(X.shape)}"
        )

    return predict_mean_var(
        predictor=predictor,
        X=X,
        node_idx=node_idx,
        options=options,
    )

@torch.no_grad()
def sample_all_nodes_from_base(
    predictor: nn.Module,
    X: torch.Tensor,
    n_samples: int,
    options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Backend-independent ancestral sampling through the DAG from external input.

    Parameters
    ----------
    predictor:
        MCD or DKL predictor exposing predictor_type.
    X:
        External-input-space tensor [N, d_ext]
    n_samples:
        Number of samples
    options:
        Currently unused, reserved for future extension.

    Returns
    -------
    samples: torch.Tensor
        Shape [S, N, n_nodes]
    """
    _validate_2d_input(X, name="X")
    _ = _get_options(options)

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    predictor_type = _get_predictor_type(predictor)
    ext_dim = _get_external_input_dim(predictor)
    if X.shape[1] != ext_dim:
        raise ValueError(
            f"X must be in external-input space with dim {ext_dim}, "
            f"got shape {tuple(X.shape)}"
        )

    if predictor_type == "mcd":
        return _mcd_sample_all_nodes_from_base(
            predictor=predictor,
            X=X,
            n_samples=n_samples,
        )

    if predictor_type == "dkl":
        return dkl_sample_all_nodes_from_base(
            predictor=predictor,
            X=X,
            n_samples=n_samples,
        )

    raise RuntimeError("Unreachable predictor type branch.")