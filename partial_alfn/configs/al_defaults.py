#!/usr/bin/env python3

from typing import Dict, Optional


def get_default_al_options(problem=None) -> Dict:
    n_nodes = getattr(problem, "n_nodes", None)

    options = {
        "nn_lr": 1e-3,
        "nn_weight_decay": 1e-6,
        "nn_train_steps": 200,
        "nn_batch_size": 64,
        "mc_samples": 30,
        "cand_n_sobol": 256,

        # partial training
        "aux_loss_weight": 1.0,
        "sink_loss_weight": 1.0,
        "aux_nodes_per_step": None,

        # partial querying
        "enable_partial_queries": True,
        "group_var_reduction": "sum",
        "use_upstream_first": True,
        "uncertainty_threshold_tau": 0.05,
    }

    if n_nodes is not None and n_nodes >= 2:
        options["upstream_group_indices"] = list(range(n_nodes - 1))
        options["downstream_group_indices"] = [n_nodes - 1]

    return options