#!/usr/bin/env python3

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _resolve_sink_idx(predictor: torch.nn.Module, sink_idx: Optional[int] = None) -> int:
    if sink_idx is not None:
        return sink_idx
    if hasattr(predictor, "sink_idx"):
        return int(predictor.sink_idx)
    raise ValueError("sink_idx must be provided if predictor has no sink_idx attribute.")


def _predict_node(
    predictor: torch.nn.Module,
    X: torch.Tensor,
    node_idx: int,
) -> torch.Tensor:
    if hasattr(predictor, "forward_node"):
        return predictor.forward_node(X, node_idx)
    raise ValueError(
        "predictor must implement forward_node(X, node_idx) for partial training."
    )


def _predict_sink(
    predictor: torch.nn.Module,
    X: torch.Tensor,
    sink_idx: Optional[int] = None,
) -> torch.Tensor:
    if hasattr(predictor, "forward_sink"):
        return predictor.forward_sink(X)
    if sink_idx is not None and hasattr(predictor, "forward_node"):
        return predictor.forward_node(X, sink_idx)
    return predictor(X)


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


def _get_nonempty_nodes(partial_buffers: Dict) -> List[int]:
    x_by_node = partial_buffers["x_by_node"]
    return [j for j, Xj in enumerate(x_by_node) if Xj.shape[0] > 0]


def _select_active_aux_nodes(
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
    partial_buffers: Dict,
    train_X_sink: Optional[torch.Tensor] = None,
    train_y_sink: Optional[torch.Tensor] = None,
    *,
    sink_idx: Optional[int] = None,
    n_steps: int = 200,
    batch_size: int = 64,
    aux_nodes_per_step: Optional[int] = None,
    aux_loss_weight: float = 1.0,
    sink_loss_weight: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> torch.optim.Optimizer:
    """
    Train a multi-head predictor using:
      - node-wise partial supervision from partial_buffers
      - optional sink supervision from (train_X_sink, train_y_sink)

    Args:
        predictor:
            multi-head predictor with forward_node() and preferably forward_sink()
        partial_buffers:
            dict from data.partial_buffers.init_partial_buffers(...)
        train_X_sink, train_y_sink:
            optional sink supervision buffers, shapes [N, d], [N, 1]
        sink_idx:
            sink node index; inferred from predictor.sink_idx when possible
        n_steps:
            number of optimization steps
        batch_size:
            minibatch size per node dataset / sink dataset
        aux_nodes_per_step:
            if set, randomly subsample this many nonempty node heads each step
        aux_loss_weight:
            weight on node-wise auxiliary loss
        sink_loss_weight:
            weight on sink loss
        lr, weight_decay:
            used only when optimizer is None
        optimizer:
            optional optimizer to continue training

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

    nonempty_nodes = _get_nonempty_nodes(partial_buffers)
    has_sink_data = (
        train_X_sink is not None
        and train_y_sink is not None
        and train_X_sink.shape[0] > 0
        and train_y_sink.shape[0] > 0
    )

    if len(nonempty_nodes) == 0 and not has_sink_data:
        return optimizer

    for _ in range(n_steps):
        optimizer.zero_grad()

        aux_losses = []
        if aux_loss_weight > 0.0 and len(nonempty_nodes) > 0:
            active_nodes = _select_active_aux_nodes(nonempty_nodes, aux_nodes_per_step)

            for node_idx in active_nodes:
                Xj = partial_buffers["x_by_node"][node_idx]
                yj = partial_buffers["y_by_node"][node_idx]
                xb, yb = _sample_minibatch(Xj, yj, batch_size)

                pred_j = _predict_node(predictor, xb, node_idx)
                loss_j = F.mse_loss(pred_j.view_as(yb), yb)
                aux_losses.append(loss_j)

        total_loss = None

        if len(aux_losses) > 0:
            aux_loss = torch.stack(aux_losses).mean()
            total_loss = aux_loss_weight * aux_loss

        if has_sink_data and sink_loss_weight > 0.0:
            xb_sink, yb_sink = _sample_minibatch(train_X_sink, train_y_sink, batch_size)
            pred_sink = _predict_sink(predictor, xb_sink, sink_idx=sink_idx)
            sink_loss = F.mse_loss(pred_sink.view_as(yb_sink), yb_sink)

            if total_loss is None:
                total_loss = sink_loss_weight * sink_loss
            else:
                total_loss = total_loss + sink_loss_weight * sink_loss

        if total_loss is None:
            continue

        total_loss.backward()
        optimizer.step()

    return optimizer