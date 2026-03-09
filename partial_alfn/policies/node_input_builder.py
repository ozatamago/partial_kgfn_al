#!/usr/bin/env python3

from typing import Sequence

import torch
import torch.nn as nn

from partial_alfn.uncertainty.mc_dropout import mc_predict_mean_var_all_nodes


def build_eval_input_for_node_group(
    *,
    predictor: nn.Module,
    base_x: torch.Tensor,
    node_group: Sequence[int],
    mc_samples: int = 100,
) -> torch.Tensor:
    """
    Build the actual evaluation input for the selected node group.

    Convention in the current FreeSolv3 setup:
      - node [0]: takes the external input directly
      - node [1]: takes a 1D upstream representation, built from the mean
                  prediction of node 0 under MC-dropout

    Args:
        predictor:
            multi-head predictor
        base_x:
            external input, shape [N, d]
        node_group:
            selected node group, e.g. [0] or [1]
        mc_samples:
            number of MC-dropout samples when constructing upstream mean

    Returns:
        eval_x:
            node-specific input to pass into problem.evaluate(X=eval_x, idx=node_group)
    """
    node_group = list(node_group)

    if node_group == [0]:
        return base_x

    if node_group == [1]:
        mean_all, _ = mc_predict_mean_var_all_nodes(
            predictor=predictor,
            X=base_x,
            mc_samples=mc_samples,
        )
        # use the predictive mean of upstream node 0 as the deterministic input to node 1
        return mean_all[:, [0]]

    raise NotImplementedError(
        f"Node-group-specific input builder not implemented for node_group={node_group}"
    )