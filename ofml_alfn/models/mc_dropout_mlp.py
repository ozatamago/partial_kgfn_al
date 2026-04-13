#!/usr/bin/env python3

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _enable_mc_dropout(m: nn.Module) -> None:
    """
    Enable dropout layers only, while keeping the rest of the model in eval mode.
    """
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


@torch.no_grad()
def mc_sample_all_nodes_from_base(
    predictor: nn.Module,
    base_x: torch.Tensor,
    mc_samples: int = 30,
) -> torch.Tensor:
    """
    Sample all node outputs from external input base_x by propagating sampled
    parent-node outputs through the function network.

    This is the key change from the old implementation:
      - node 0 is sampled from its own conditional model
      - node 1 is sampled conditional on the sampled output of node 0
      - more generally, each node is sampled from sampled parent outputs

    Requirements on predictor:
      - predictor.n_nodes
      - predictor.parent_nodes
      - predictor.active_input_indices
      - predictor.make_node_input_from_base(base_x=..., node_idx=..., parent_outputs=...)
      - predictor.forward_node(x_node, node_idx)

    Args:
        predictor:
            node-wise conditional predictor
        base_x:
            external input tensor [N, d_ext]
        mc_samples:
            number of Monte Carlo samples

    Returns:
        samples:
            tensor of shape [S, N, n_nodes]
            where samples[s, i, j] is the sampled output of node j
            for base point i in MC sample s
    """
    if not hasattr(predictor, "n_nodes"):
        raise ValueError("predictor must define n_nodes")
    if not hasattr(predictor, "make_node_input_from_base"):
        raise ValueError(
            "predictor must implement make_node_input_from_base(base_x=..., node_idx=..., parent_outputs=...)"
        )
    if not hasattr(predictor, "forward_node"):
        raise ValueError("predictor must implement forward_node(x_node, node_idx)")

    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    all_mc_outputs: List[torch.Tensor] = []

    for _ in range(mc_samples):
        node_outputs: List[torch.Tensor] = [None] * predictor.n_nodes
        sampled_outputs_this_pass: List[torch.Tensor] = []

        for j in range(predictor.n_nodes):
            xj = predictor.make_node_input_from_base(
                base_x=base_x,
                node_idx=j,
                parent_outputs=node_outputs,
            )  # [N, d_j]

            yj = predictor.forward_node(xj, j)  # [N, 1]
            node_outputs[j] = yj
            sampled_outputs_this_pass.append(yj)

        # [N, n_nodes]
        y_all = torch.cat(sampled_outputs_this_pass, dim=-1)
        all_mc_outputs.append(y_all)

    # [S, N, n_nodes]
    return torch.stack(all_mc_outputs, dim=0)


@torch.no_grad()
def mc_predict_mean_var_all_nodes(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC-dropout prediction for all nodes from external input X, with uncertainty
    propagated through sampled parent-node outputs.

    Args:
        predictor:
            node-wise conditional predictor
        X:
            external input [N, d_ext]
        mc_samples:
            number of MC samples

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
    MC-dropout prediction for sink node from external input X, with propagated
    uncertainty through parent-node sampling.

    Args:
        predictor:
            node-wise conditional predictor with sink_idx
        X:
            external input [N, d_ext]
        mc_samples:
            number of MC samples

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

    if not hasattr(predictor, "sink_idx"):
        raise ValueError("predictor must define sink_idx")

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
    If you use effective ancestor-closure cost elsewhere, you can override this logic
    outside this function.

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