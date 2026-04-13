#!/usr/bin/env python3

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn


def _flatten_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Convert target shape [N, 1] or [N] -> [N].
    """
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(
                f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}"
            )
        return y[:, 0]
    if y.ndim == 1:
        return y
    raise ValueError(f"Expected y of shape [N, 1] or [N], got {tuple(y.shape)}")


def _validate_node_datasets(
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
) -> None:
    if len(train_X_nodes) != len(train_Y_nodes):
        raise ValueError(
            f"train_X_nodes and train_Y_nodes must have the same length, got "
            f"{len(train_X_nodes)} and {len(train_Y_nodes)}"
        )

    for j, (xj, yj) in enumerate(zip(train_X_nodes, train_Y_nodes)):
        if not torch.is_tensor(xj):
            raise TypeError(f"train_X_nodes[{j}] must be a tensor, got {type(xj)}")
        if not torch.is_tensor(yj):
            raise TypeError(f"train_Y_nodes[{j}] must be a tensor, got {type(yj)}")

        if xj.ndim != 2:
            raise ValueError(
                f"train_X_nodes[{j}] must be 2D, got shape {tuple(xj.shape)}"
            )
        if yj.ndim not in [1, 2]:
            raise ValueError(
                f"train_Y_nodes[{j}] must be 1D or 2D, got shape {tuple(yj.shape)}"
            )

        yj_flat = _flatten_targets(yj)
        if xj.shape[0] != yj_flat.shape[0]:
            raise ValueError(
                f"Node {j}: x and y must have the same number of rows, got "
                f"{xj.shape[0]} and {yj_flat.shape[0]}"
            )


def _get_nonempty_nodes(
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
) -> list[int]:
    out: list[int] = []
    for j, (xj, yj) in enumerate(zip(train_X_nodes, train_Y_nodes)):
        yj_flat = _flatten_targets(yj)
        if xj.shape[0] != yj_flat.shape[0]:
            raise ValueError(
                f"Node {j}: x and y must have the same number of rows, got "
                f"{xj.shape[0]} and {yj_flat.shape[0]}"
            )
        if xj.shape[0] > 0:
            out.append(j)
    return out


def _validate_against_predictor_dims(
    predictor: nn.Module,
    train_X_nodes: Sequence[torch.Tensor],
) -> None:
    if not hasattr(predictor, "node_input_dims"):
        return

    expected_dims = [int(d) for d in predictor.node_input_dims]
    if len(expected_dims) != len(train_X_nodes):
        raise ValueError(
            f"predictor.node_input_dims has length {len(expected_dims)} but "
            f"train_X_nodes has length {len(train_X_nodes)}"
        )

    for j, xj in enumerate(train_X_nodes):
        if xj.shape[1] != expected_dims[j]:
            raise ValueError(
                f"Node {j}: train_X_nodes[{j}] has dim {xj.shape[1]}, "
                f"but predictor expects {expected_dims[j]}"
            )


def _restore_node_optimizer_states(
    *,
    node_optimizers: Sequence[Optional[torch.optim.Optimizer]],
    optimizer_state: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(optimizer_state, dict):
        return
    if optimizer_state.get("kind", None) != "dkl_node_optimizers":
        return

    saved = optimizer_state.get("node_optimizers", None)
    if not isinstance(saved, list):
        return

    for j, opt in enumerate(node_optimizers):
        if opt is None:
            continue
        if j >= len(saved):
            continue
        if saved[j] is None:
            continue
        opt.load_state_dict(saved[j])


def _dump_node_optimizer_states(
    node_optimizers: Sequence[Optional[torch.optim.Optimizer]],
) -> Dict[str, Any]:
    return {
        "kind": "dkl_node_optimizers",
        "node_optimizers": [
            None if opt is None else opt.state_dict()
            for opt in node_optimizers
        ],
    }


def train_predictor_partial_dkl(
    predictor: nn.Module,
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
    *,
    sink_idx: Optional[int] = None,
    n_steps: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    nodes_per_step: Optional[int] = None,
    aux_loss_weight: float = 1.0,
    sink_loss_weight: float = 1.0,
    optimizer: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Train a node-wise DKL predictor using node-wise datasets.

    Parameters
    ----------
    predictor:
        Expected to expose:
          - predictor.n_nodes
          - predictor.sink_idx
          - predictor.set_node_train_data(node_idx, x, y, strict=False)
          - predictor.node_mll(node_idx) -> ExactMarginalLogLikelihood
          - predictor.node_models[node_idx].gp
          - predictor.node_models[node_idx].likelihood
    train_X_nodes:
        List of node-input tensors. train_X_nodes[j] has shape [Nj, dj].
    train_Y_nodes:
        List of node-output tensors. train_Y_nodes[j] has shape [Nj, 1] or [Nj].
    sink_idx:
        Optional explicit sink index. If None, use predictor.sink_idx.
    n_steps:
        Number of optimization steps.
    lr:
        Learning rate.
    weight_decay:
        Weight decay for Adam.
    nodes_per_step:
        If provided, sample at most this many nonempty nodes per outer step.
        If None, update all nonempty nodes every step.
    aux_loss_weight:
        Weight for non-sink nodes.
    sink_loss_weight:
        Weight for sink node.
    optimizer:
        Optional dict containing per-node optimizer states from a previous run.
    verbose:
        If True, print loss diagnostics.

    Returns
    -------
    optimizer_state:
        Dict containing per-node optimizer states.
    """
    _validate_node_datasets(train_X_nodes, train_Y_nodes)

    if not hasattr(predictor, "n_nodes"):
        raise AttributeError("predictor must expose n_nodes")
    if not hasattr(predictor, "sink_idx"):
        raise AttributeError("predictor must expose sink_idx")
    if not hasattr(predictor, "set_node_train_data"):
        raise AttributeError("predictor must expose set_node_train_data(...)")
    if not hasattr(predictor, "node_mll"):
        raise AttributeError("predictor must expose node_mll(node_idx)")
    if not hasattr(predictor, "node_models"):
        raise AttributeError("predictor must expose node_models")

    n_nodes = int(predictor.n_nodes)
    if len(train_X_nodes) != n_nodes:
        raise ValueError(f"Expected {n_nodes} node datasets, got {len(train_X_nodes)}")

    _validate_against_predictor_dims(predictor, train_X_nodes)

    sink_idx = predictor.sink_idx if sink_idx is None else int(sink_idx)
    if not (0 <= sink_idx < n_nodes):
        raise ValueError(f"sink_idx must be in [0, {n_nodes - 1}], got {sink_idx}")

    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if aux_loss_weight < 0:
        raise ValueError(
            f"aux_loss_weight must be non-negative, got {aux_loss_weight}"
        )
    if sink_loss_weight < 0:
        raise ValueError(
            f"sink_loss_weight must be non-negative, got {sink_loss_weight}"
        )

    nonempty_nodes = _get_nonempty_nodes(train_X_nodes, train_Y_nodes)
    if len(nonempty_nodes) == 0:
        return optimizer

    # Load training data only into nodes that actually have observations.
    for j in nonempty_nodes:
        xj = train_X_nodes[j]
        yj = train_Y_nodes[j]
        predictor.set_node_train_data(
            node_idx=j,
            x=xj,
            y=yj,
            strict=False,
        )

    # Build one optimizer per node.
    node_optimizers: list[Optional[torch.optim.Optimizer]] = [None] * n_nodes
    for j in nonempty_nodes:
        params = list(predictor.node_models[j].parameters())
        node_optimizers[j] = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    _restore_node_optimizer_states(
        node_optimizers=node_optimizers,
        optimizer_state=optimizer,
    )

    for step in range(n_steps):
        if nodes_per_step is None or nodes_per_step >= len(nonempty_nodes):
            active_nodes = list(nonempty_nodes)
        else:
            perm = torch.randperm(len(nonempty_nodes))
            active_nodes = [nonempty_nodes[i] for i in perm[:nodes_per_step].tolist()]

        total_objective_value = 0.0

        for j in active_nodes:
            xj = train_X_nodes[j]
            yj = _flatten_targets(train_Y_nodes[j])

            node_model = predictor.node_models[j]
            gp_model = node_model.gp
            likelihood = node_model.likelihood
            mll = predictor.node_mll(j)

            gp_model.train()
            likelihood.train()

            assert node_optimizers[j] is not None
            node_optimizers[j].zero_grad()
            gp_output = gp_model(xj)
            neg_mll = -mll(gp_output, yj)

            weight = sink_loss_weight if j == sink_idx else aux_loss_weight
            loss = weight * neg_mll
            loss.backward()
            node_optimizers[j].step()

            total_objective_value += float(loss.item())

        if verbose:
            print(
                f"[train_predictor_partial_dkl] "
                f"step={step + 1}/{n_steps} | "
                f"weighted_neg_mll_sum={total_objective_value:.6f}"
            )

    predictor.eval()
    for j in nonempty_nodes:
        predictor.node_models[j].gp.eval()
        predictor.node_models[j].likelihood.eval()

    return _dump_node_optimizer_states(node_optimizers)