#!/usr/bin/env python3

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _enable_mc_dropout(m: nn.Module) -> None:
    """
    Turn on dropout layers only, while keeping the rest of the model in eval mode.
    """
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


def _validate_predictor_for_propagation(predictor: nn.Module) -> None:
    required_attrs = ["n_nodes", "sink_idx", "parent_nodes", "active_input_indices"]
    required_methods = ["make_node_input_from_base", "forward_node"]

    for name in required_attrs:
        if not hasattr(predictor, name):
            raise ValueError(f"predictor must define attribute `{name}`")

    for name in required_methods:
        if not hasattr(predictor, name):
            raise ValueError(f"predictor must implement method `{name}`")


@torch.no_grad()
def mc_sample_all_nodes_from_base(
    predictor: nn.Module,
    base_x: torch.Tensor,
    mc_samples: int = 30,
) -> torch.Tensor:
    """
    Draw MC-dropout samples for all node outputs by propagating sampled parent
    outputs through the function network.

    Each Monte Carlo pass does:
      1. sample node 0 from its conditional model
      2. use that sampled node 0 output to build node 1 input
      3. sample node 1
      4. continue in topological order

    Args:
        predictor:
            node-wise conditional predictor implementing:
              - predictor.n_nodes
              - predictor.parent_nodes
              - predictor.active_input_indices
              - predictor.make_node_input_from_base(base_x=..., node_idx=..., parent_outputs=...)
              - predictor.forward_node(x_node, node_idx)
        base_x:
            external input tensor [N, d_ext]
        mc_samples:
            number of ancestral MC samples

    Returns:
        samples:
            tensor [S, N, n_nodes]
            where samples[s, i, j] is sampled output of node j
            for base point i in Monte Carlo pass s
    """
    _validate_predictor_for_propagation(predictor)

    if base_x.ndim != 2:
        raise ValueError(f"base_x must be 2D [N, d_ext], got shape {tuple(base_x.shape)}")

    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    all_pass_outputs: List[torch.Tensor] = []

    for _ in range(mc_samples):
        # parent_outputs[j] will hold sampled output of node j in this MC pass
        parent_outputs: List[torch.Tensor] = [None] * predictor.n_nodes
        node_outputs_this_pass: List[torch.Tensor] = []

        for j in range(predictor.n_nodes):
            xj = predictor.make_node_input_from_base(
                base_x=base_x,
                node_idx=j,
                parent_outputs=parent_outputs,
            )  # [N, d_j]

            yj = predictor.forward_node(xj, j)  # [N, 1]
            parent_outputs[j] = yj
            node_outputs_this_pass.append(yj)

        y_all = torch.cat(node_outputs_this_pass, dim=-1)  # [N, n_nodes]
        all_pass_outputs.append(y_all)

    return torch.stack(all_pass_outputs, dim=0)  # [S, N, n_nodes]


@torch.no_grad()
def mc_predict_mean_var_all_nodes(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-dropout mean/variance for all nodes, where downstream uncertainty is
    computed by propagating sampled parent outputs.

    Args:
        predictor:
            node-wise conditional predictor
        X:
            external input [N, d_ext]
        mc_samples:
            number of ancestral MC samples

    Returns:
        mean_all:
            [N, n_nodes]
        var_all:
            [N, n_nodes]
    """
    Y = mc_sample_all_nodes_from_base(
        predictor=predictor,
        base_x=X,
        mc_samples=mc_samples,
    )  # [S, N, n_nodes]

    mean_all = Y.mean(dim=0)
    var_all = Y.var(dim=0, unbiased=False)
    return mean_all, var_all


@torch.no_grad()
def mc_predict_mean_var(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-dropout mean/variance for the sink node, with propagated uncertainty.

    Args:
        predictor:
            node-wise conditional predictor with sink_idx
        X:
            external input [N, d_ext]
        mc_samples:
            number of ancestral MC samples

    Returns:
        mean_sink:
            [N, 1]
        var_sink:
            [N, 1]
    """
    mean_all, var_all = mc_predict_mean_var_all_nodes(
        predictor=predictor,
        X=X,
        mc_samples=mc_samples,
    )

    sink_idx = int(predictor.sink_idx)
    return mean_all[:, [sink_idx]], var_all[:, [sink_idx]]


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
    Convert group utilities into utility / cost scores using direct node-group cost.

    Note:
        If you use effective ancestor-closure cost elsewhere, override this logic
        outside this function.
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
    ).view(1, -1)

    return group_var / (cost_tensor + eps)


def select_top_cost_aware_action(
    scores: torch.Tensor,
) -> Tuple[int, int, float]:
    """
    Select the best (candidate_idx, group_idx) from a score matrix.

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