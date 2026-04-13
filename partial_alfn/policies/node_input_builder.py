#!/usr/bin/env python3

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from partial_alfn.uncertainty.base import (
    predict_mean_var,
    sample_all_nodes_from_base,
)


def _make_uncertainty_options(
    *,
    mc_samples: int,
    options: Optional[Dict] = None,
) -> Dict:
    options = {} if options is None else dict(options)
    return {
        "mc_samples": int(options.get("mc_samples", mc_samples)),
        "n_samples": int(options.get("n_posterior_samples", mc_samples)),
        "unbiased": bool(options.get("unbiased_var", False)),
    }


def build_eval_input_for_node_group(
    *,
    predictor: nn.Module,
    base_x: torch.Tensor,
    node_group: Sequence[int],
    mc_samples: int = 100,
    options: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Deterministic eval input builder used for the actual online query.

    Current FreeSolv3 convention:
      - node [0]: takes external input directly
      - node [1]: takes a 1D upstream representation built from the
                  predictive mean of node 0
    """
    node_group = list(node_group)
    uncertainty_options = _make_uncertainty_options(
        mc_samples=mc_samples,
        options=options,
    )

    if node_group == [0]:
        return base_x

    if node_group == [1]:
        mean0, _ = predict_mean_var(
            predictor=predictor,
            X=base_x,
            node_idx=0,
            options=uncertainty_options,
        )
        return mean0

    raise NotImplementedError(
        f"Node-group-specific input builder not implemented for node_group={node_group}"
    )


def sample_fantasy_observation_for_node_group(
    *,
    predictor: nn.Module,
    base_x: torch.Tensor,
    node_group: Sequence[int],
    mc_samples: int = 100,
    options: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample-based fantasy builder.

    Returns
    -------
    fantasy_eval_x:
        The sampled node-specific input that would be fed to problem.evaluate(..., idx=node_group)
    fantasy_y:
        The sampled fantasy observation for the selected node group

    Notes
    -----
    For FreeSolv3:
      - node [0]:
            fantasy_eval_x = base_x
            fantasy_y      = sampled node-0 output
      - node [1]:
            sample node 0 first
            use that sampled node-0 value as fantasy_eval_x
            then sample node 1 conditional on that sampled input
    """
    node_group = list(node_group)
    uncertainty_options = _make_uncertainty_options(
        mc_samples=mc_samples,
        options=options,
    )
    predictor_type = str(getattr(predictor, "predictor_type", "")).lower()

    if node_group == [0]:
        if predictor_type == "dkl":
            # node 0 input space == external input space in FreeSolv3
            fantasy_y = predictor.node_models[0].sample_latent(
                base_x,
                n_samples=1,
            )[0]  # [N, 1]
        else:
            all_samples = sample_all_nodes_from_base(
                predictor=predictor,
                X=base_x,
                n_samples=1,
                options=uncertainty_options,
            )  # [1, N, n_nodes]
            fantasy_y = all_samples[0, :, [0]]

        fantasy_eval_x = base_x
        return fantasy_eval_x, fantasy_y

    if node_group == [1]:
        if predictor_type == "dkl":
            # sample node 0 first
            sampled_z0 = predictor.node_models[0].sample_latent(
                base_x,
                n_samples=1,
            )[0]  # [N, 1]

            # then sample node 1 conditional on sampled node 0
            sampled_y1 = predictor.node_models[1].sample_latent(
                sampled_z0,
                n_samples=1,
            )[0]  # [N, 1]

            fantasy_eval_x = sampled_z0
            fantasy_y = sampled_y1
        else:
            all_samples = sample_all_nodes_from_base(
                predictor=predictor,
                X=base_x,
                n_samples=1,
                options=uncertainty_options,
            )  # [1, N, n_nodes]

            fantasy_eval_x = all_samples[0, :, [0]]
            fantasy_y = all_samples[0, :, [1]]

        return fantasy_eval_x, fantasy_y

    raise NotImplementedError(
        f"Sample-based fantasy builder not implemented for node_group={node_group}"
    )