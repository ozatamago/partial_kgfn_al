#!/usr/bin/env python3

from typing import Dict, List, Optional

import torch
from torch import Tensor

from partial_alfn.data.partial_buffers import (
    append_full_network_as_partial,
    append_partial_group,
)
from partial_alfn.utils.construct_obs_set import construct_obs_set


def append_full_observation(
    *,
    problem,
    new_x: Tensor,
    new_y: Tensor,
    state: Dict,
) -> List[int]:
    """
    Full-network observation update.

    Here new_x is both:
      - the external input
      - the valid full-network evaluation input
    """
    state["train_X_nn"] = torch.cat((state["train_X_nn"], new_x), dim=0)
    state["train_y_nn"] = torch.cat((state["train_y_nn"], new_y[..., [-1]]), dim=0)

    state["network_output_at_X"] = torch.cat((state["network_output_at_X"], new_y), dim=0)

    new_obs_x, new_obs_y = construct_obs_set(
        X=new_x,
        Y=new_y,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )

    for j in range(problem.n_nodes):
        state["train_X"][j] = torch.cat((state["train_X"][j], new_obs_x[j]), dim=0)
        state["train_Y"][j] = torch.cat((state["train_Y"][j], new_obs_y[j]), dim=0)

    state["node_eval_counts"] = state["node_eval_counts"] + torch.ones(
        len(problem.parent_nodes),
        dtype=state["node_eval_counts"].dtype,
    )

    if state.get("partial_buffers", None) is not None:
        append_full_network_as_partial(
            buffers=state["partial_buffers"],
            x=new_x,       # external input
            y_full=new_y,
        )

    return list(range(problem.n_nodes))


def append_partial_observation(
    *,
    problem,
    base_x: Tensor,
    eval_x: Tensor,
    new_y: Tensor,
    new_node: List[int],
    state: Dict,
    sink_node_idx: Optional[int] = None,
) -> List[int]:
    """
    Partial-node observation update.

    Important distinction:
      - base_x: external input in the original problem input space
      - eval_x: actual node-specific input used for evaluating the selected node group

    We store:
      - eval_x into train_X[j], because train_X[j] is node-input-space specific
      - base_x into partial_buffers, because the multi-head predictor maps external x -> node outputs
    """
    if sink_node_idx is None:
        sink_node_idx = problem.n_nodes - 1

    idx_for_new_y = 0
    for j in new_node:
        # train_X[j] expects node-specific input dimension
        state["train_X"][j] = torch.cat((state["train_X"][j], eval_x), dim=0)
        state["train_Y"][j] = torch.cat((state["train_Y"][j], new_y[..., [idx_for_new_y]]), dim=0)
        state["node_eval_counts"][j] += 1

        # If the sink itself was directly queried, also add supervised sink data
        # using the external input base_x for the predictor.
        if (
            j == sink_node_idx
            and state.get("train_X_nn") is not None
            and state.get("train_y_nn") is not None
        ):
            state["train_X_nn"] = torch.cat((state["train_X_nn"], base_x), dim=0)
            state["train_y_nn"] = torch.cat(
                (state["train_y_nn"], new_y[..., [idx_for_new_y]]),
                dim=0,
            )

        idx_for_new_y += 1

    # partial_buffers are for the multi-head predictor: external x -> node outputs
    if state.get("partial_buffers", None) is not None:
        append_partial_group(
            buffers=state["partial_buffers"],
            node_indices=new_node,
            x=base_x,
            y_group=new_y,
        )

    return new_node