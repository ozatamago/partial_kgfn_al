#!/usr/bin/env python3

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _resolve_sink_idx(
    predictor: torch.nn.Module,
    sink_idx: Optional[int] = None,
) -> int:
    if sink_idx is not None:
        return int(sink_idx)
    if hasattr(predictor, "sink_idx"):
        return int(predictor.sink_idx)
    raise ValueError("sink_idx must be provided if predictor has no sink_idx attribute.")


def _predict_node(
    predictor: torch.nn.Module,
    X_node: torch.Tensor,
    node_idx: int,
) -> torch.Tensor:
    if not hasattr(predictor, "forward_node"):
        raise ValueError(
            "predictor must implement forward_node(X_node, node_idx) "
            "for node-wise conditional training."
        )
    return predictor.forward_node(X_node, node_idx)


def _sample_minibatch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot sample from an empty dataset.")
    idx = torch.randint(0, n, (min(batch_size, n),), device=X.device)
    return X[idx], y[idx]


def _get_nonempty_nodes(
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
) -> List[int]:
    if len(train_X_nodes) != len(train_Y_nodes):
        raise ValueError(
            f"train_X_nodes and train_Y_nodes must have same length, "
            f"got {len(train_X_nodes)} and {len(train_Y_nodes)}"
        )

    out = []
    for j, (Xj, Yj) in enumerate(zip(train_X_nodes, train_Y_nodes)):
        if Xj.shape[0] != Yj.shape[0]:
            raise ValueError(
                f"Node {j}: train_X_nodes[{j}] has {Xj.shape[0]} rows but "
                f"train_Y_nodes[{j}] has {Yj.shape[0]} rows"
            )
        if Xj.shape[0] > 0:
            out.append(j)
    return out


def _select_active_nodes(
    nonempty_nodes: Sequence[int],
    nodes_per_step: Optional[int],
) -> List[int]:
    nonempty_nodes = list(nonempty_nodes)
    if len(nonempty_nodes) == 0:
        return []
    if nodes_per_step is None or nodes_per_step >= len(nonempty_nodes):
        return nonempty_nodes

    perm = torch.randperm(len(nonempty_nodes))[:nodes_per_step].tolist()
    return [nonempty_nodes[i] for i in perm]


def train_predictor_partial(
    predictor: torch.nn.Module,
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
    *,
    sink_idx: Optional[int] = None,
    n_steps: int = 200,
    batch_size: int = 64,
    nodes_per_step: Optional[int] = None,
    aux_loss_weight: float = 1.0,
    sink_loss_weight: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> torch.optim.Optimizer:
    """
    Train node-wise conditional models using node-specific datasets.

    Each node j is trained only on:
        train_X_nodes[j] -> train_Y_nodes[j]

    The sink is not trained twice. It is treated as one node-wise conditional model
    with its own weight (sink_loss_weight), while non-sink nodes use aux_loss_weight.

    Args:
        predictor:
            node-wise conditional predictor implementing forward_node(X_node, j)
        train_X_nodes:
            list of node-specific inputs; train_X_nodes[j] has shape [N_j, d_j]
        train_Y_nodes:
            list of node-specific targets; train_Y_nodes[j] has shape [N_j, 1]
        sink_idx:
            sink node index
        n_steps:
            number of optimization steps
        batch_size:
            minibatch size for each active node
        nodes_per_step:
            if not None, randomly subsample this many nonempty nodes each step
        aux_loss_weight:
            weight applied to non-sink nodes
        sink_loss_weight:
            weight applied to sink node
        lr, weight_decay:
            optimizer hyperparameters if optimizer is None
        optimizer:
            optional existing optimizer for continued training

    Returns:
        optimizer after training
    """
    predictor.train()
    sink_idx = _resolve_sink_idx(predictor, sink_idx)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    nonempty_nodes = _get_nonempty_nodes(train_X_nodes, train_Y_nodes)
    if len(nonempty_nodes) == 0:
        return optimizer

    for _ in range(n_steps):
        optimizer.zero_grad()

        active_nodes = _select_active_nodes(nonempty_nodes, nodes_per_step)
        losses = []

        for node_idx in active_nodes:
            Xj = train_X_nodes[node_idx]
            Yj = train_Y_nodes[node_idx]
            xb, yb = _sample_minibatch(Xj, Yj, batch_size)

            pred_j = _predict_node(predictor, xb, node_idx)
            loss_j = F.mse_loss(pred_j.view_as(yb), yb)

            weight_j = sink_loss_weight if node_idx == sink_idx else aux_loss_weight
            if weight_j > 0.0:
                losses.append(weight_j * loss_j)

        if len(losses) == 0:
            continue

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

    return optimizer