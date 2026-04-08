#!/usr/bin/env python3

from typing import Dict


def get_default_al_options(problem=None) -> Dict:
    n_nodes = getattr(problem, "n_nodes", None)

    options = {
        # ------------------------------------------------------------------
        # Predictor backend selection
        # ------------------------------------------------------------------
        "predictor_type": "mcd",   # "mcd" or "dkl"

        # ------------------------------------------------------------------
        # Common model/training defaults
        # ------------------------------------------------------------------
        "hidden": 256,
        "weight_decay": 1e-6,

        "nn_lr": 1e-3,
        "nn_weight_decay": 1e-6,
        "nn_train_steps": 200,
        "nn_batch_size": 64,

        # ------------------------------------------------------------------
        # MCD-specific defaults
        # ------------------------------------------------------------------
        "p_drop": 0.1,
        "mc_samples": 30,

        # ------------------------------------------------------------------
        # DKL-specific defaults
        # ------------------------------------------------------------------
        # For now, "exact" is the intended working mode for small/medium data.
        # "svdkl" is reserved for future extension.
        "dkl_inference": "exact",          # "exact" or "svdkl"
        "dkl_hidden": 256,
        "dkl_feature_dim": 32,
        "dkl_kernel": "rbf",               # "rbf" or "matern"
        "dkl_noise": 1e-4,

        # Aliases used directly by factory code so runner-side branching stays small
        "feature_dim": 32,
        "kernel_type": "rbf",

        # ------------------------------------------------------------------
        # Candidate generation / acquisition defaults
        # ------------------------------------------------------------------
        "cand_n_sobol": 256,

        # fantasy-based selector settings
        "selector_objective": "uncertainty",   # "uncertainty" or "fantasy_gain"
        "selector_metric": "sink_test_loss",
        "fantasy_train_steps": 20,
        "fantasy_topk_candidates": 8,
        "fantasy_topk_groups": 2,

        # DKL uncertainty estimation may use posterior samples for DAG rollout
        "n_posterior_samples": 64,

        # ------------------------------------------------------------------
        # Partial training
        # ------------------------------------------------------------------
        "aux_loss_weight": 1.0,
        "sink_loss_weight": 1.0,
        "nodes_per_step": None,

        # ------------------------------------------------------------------
        # Partial querying
        # ------------------------------------------------------------------
        "enable_partial_queries": True,
        "group_var_reduction": "sum",
        "use_upstream_first": False,
        "uncertainty_threshold_tau": 0.5,
    }

    if n_nodes is not None and n_nodes >= 2:
        options["upstream_group_indices"] = list(range(n_nodes - 1))
        options["downstream_group_indices"] = [n_nodes - 1]

    return options