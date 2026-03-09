#!/usr/bin/env python3

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _enable_mc_dropout(m: nn.Module) -> None:
    """
    Enable dropout layers only, while keeping the rest of the model in eval mode.
    """
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


def _predict_sink(predictor: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Return sink prediction with backward compatibility.

    Preferred API:
      - predictor.forward_sink(X) -> [N, 1]

    Fallback:
      - predictor(X) -> [N, 1]
    """
    if hasattr(predictor, "forward_sink"):
        return predictor.forward_sink(X)
    return predictor(X)


def _predict_all_nodes(predictor: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Return predictions for all node heads.

    Required API for multi-head models:
      - predictor.forward_all(X) -> [N, n_nodes]

    Fallback:
      - if model is sink-only, treat it as a single-node model and return [N, 1]
    """
    if hasattr(predictor, "forward_all"):
        return predictor.forward_all(X)

    y = predictor(X)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    return y


@torch.no_grad()
def mc_predict_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-dropout prediction for sink output.

    Args:
        predictor:
            sink-only model, or multi-head model implementing forward_sink()
        X:
            [N, d]
        mc_samples:
            number of stochastic forward passes

    Returns:
        mean: [N, 1]
        var:  [N, 1]
    """
    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(mc_samples):
        outs.append(_predict_sink(predictor, X))  # [N, 1]

    Y = torch.stack(outs, dim=0)  # [S, N, 1]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=False)
    return mean, var


@torch.no_grad()
def mc_predict_mean_var_all_nodes(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-dropout prediction for all node heads.

    Args:
        predictor:
            multi-head model implementing forward_all(X) -> [N, n_nodes]
            sink-only models also work and are treated as n_nodes = 1
        X:
            [N, d]
        mc_samples:
            number of stochastic forward passes

    Returns:
        mean_all: [N, n_nodes]
        var_all:  [N, n_nodes]
    """
    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(mc_samples):
        outs.append(_predict_all_nodes(predictor, X))  # [N, n_nodes]

    Y = torch.stack(outs, dim=0)  # [S, N, n_nodes]
    mean_all = Y.mean(dim=0)
    var_all = Y.var(dim=0, unbiased=False)
    return mean_all, var_all


def reduce_group_variances(
    var_all: torch.Tensor,
    node_groups: Sequence[Sequence[int]],
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Aggregate node-wise variances into group-wise utilities.

    Args:
        var_all:
            [N, n_nodes]
        node_groups:
            e.g. [[0], [1], [2, 3]]
        reduction:
            one of {"sum", "mean", "max"}

    Returns:
        group_var:
            [N, n_groups]
    """
    if var_all.ndim != 2:
        raise ValueError(f"var_all must be [N, n_nodes], got shape {tuple(var_all.shape)}")

    if reduction not in {"sum", "mean", "max"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    cols = []
    for group in node_groups:
        if len(group) == 0:
            raise ValueError("Empty node group is not allowed.")
        vg = var_all[:, list(group)]  # [N, |group|]

        if reduction == "sum":
            ug = vg.sum(dim=-1, keepdim=True)
        elif reduction == "mean":
            ug = vg.mean(dim=-1, keepdim=True)
        else:
            ug = vg.max(dim=-1, keepdim=True).values

        cols.append(ug)

    return torch.cat(cols, dim=-1)  # [N, n_groups]


def cost_aware_group_scores(
    group_var: torch.Tensor,
    node_groups: Sequence[Sequence[int]],
    node_costs: Sequence[float],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert group utilities into utility / cost scores.

    Args:
        group_var:
            [N, n_groups]
        node_groups:
            list of node groups aligned with columns of group_var
        node_costs:
            per-node costs
        eps:
            numerical guard

    Returns:
        scores:
            [N, n_groups]
    """
    if group_var.ndim != 2:
        raise ValueError(f"group_var must be [N, n_groups], got shape {tuple(group_var.shape)}")

    group_costs = []
    for group in node_groups:
        c = sum(float(node_costs[j]) for j in group)
        group_costs.append(c)

    cost_tensor = torch.tensor(
        group_costs,
        dtype=group_var.dtype,
        device=group_var.device,
    ).view(1, -1)  # [1, n_groups]

    return group_var / (cost_tensor + eps)


def select_top_cost_aware_action(
    scores: torch.Tensor,
) -> Tuple[int, int, float]:
    """
    Select the best (candidate_idx, group_idx) from scores.

    Args:
        scores:
            [N, n_groups]

    Returns:
        cand_idx:
            row index in candidate set
        group_idx:
            selected node-group index
        best_score:
            scalar float
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be [N, n_groups], got shape {tuple(scores.shape)}")

    flat_idx = torch.argmax(scores)
    n_groups = scores.shape[1]
    cand_idx = int(flat_idx // n_groups)
    group_idx = int(flat_idx % n_groups)
    best_score = float(scores[cand_idx, group_idx].item())
    return cand_idx, group_idx, best_score