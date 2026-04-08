#!/usr/bin/env python3

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from partial_alfn.uncertainty.base import predict_mean_var_all_nodes


def build_eval_input_for_node_group(
    *,
    predictor: nn.Module,
    base_x: torch.Tensor,
    node_group: Sequence[int],
    mc_samples: int = 100,
    options: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Build the actual evaluation input for the selected node group.

    Convention in the current FreeSolv3 setup:
      - node [0]: takes the external input directly
      - node [1]: takes a 1D upstream representation, built from the mean
                  prediction of node 0 under the current uncertainty backend

    Parameters
    ----------
    predictor:
        Multi-head predictor exposing predictor_type = "mcd" or "dkl".
    base_x:
        External input in the original input space, shape [N, d].
    node_group:
        Selected node group, e.g. [0] or [1].
    mc_samples:
        Backward-compatible default sample count. Used for:
          - MC-dropout sample count when predictor_type == "mcd"
          - posterior rollout sample count fallback when predictor_type == "dkl"
    options:
        Optional config dict. Relevant keys:
          - mc_samples
          - n_posterior_samples
          - unbiased_var

    Returns
    -------
    eval_x:
        Node-specific input to pass into problem.evaluate(X=eval_x, idx=node_group)
    """
    node_group = list(node_group)
    options = {} if options is None else dict(options)

    uncertainty_options = {
        "mc_samples": int(options.get("mc_samples", mc_samples)),
        "n_samples": int(options.get("n_posterior_samples", mc_samples)),
        "unbiased": bool(options.get("unbiased_var", False)),
    }

    if node_group == [0]:
        return base_x

    if node_group == [1]:
        mean_all, _ = predict_mean_var_all_nodes(
            predictor=predictor,
            X=base_x,
            options=uncertainty_options,
        )
        # Use the predictive mean of upstream node 0 as the deterministic input to node 1.
        return mean_all[:, [0]]

    raise NotImplementedError(
        f"Node-group-specific input builder not implemented for node_group={node_group}"
    )