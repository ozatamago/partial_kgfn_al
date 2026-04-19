#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional

from ofml_alfn.configs.al_defaults import get_default_al_options


def get_fantasy_protocol1a_options(
    problem=None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified options for Problem 1A experiments.

    Supported experiment modes
    --------------------------
    1. fantasy_al
       Cost-normalized fantasy acquisition over protocol_1, protocol_2, protocol_3.

    2. pretrain_then_adapt
       Two-stage experiment:
         - pretrain on similar source observers
         - adapt on target observer
         - evaluate on target validation / test

    Notes
    -----
    The function name is kept for backward compatibility even though this config
    now supports more than fantasy acquisition alone.
    """
    options = get_default_al_options(problem)

    options.update(
        {
            # --------------------------------------------------------------
            # Experiment identity
            # --------------------------------------------------------------
            "experiment_name": "protocol1a",
            "problem_family": "protocol1a",

            # --------------------------------------------------------------
            # Mode switching
            # --------------------------------------------------------------
            # Available values:
            #   - "fantasy_al"
            #   - "pretrain_then_adapt"
            "experiment_mode": "fantasy_al",

            # --------------------------------------------------------------
            # Protocol-level bookkeeping
            # --------------------------------------------------------------
            # This should be overwritten by the experiment script after the
            # concrete benchmark is built.
            "target_protocol_id": None,

            # Default source protocols used in pretrain_then_adapt mode.
            "source_protocol_ids": ["protocol_1", "protocol_2"],

            # Optional split-name hints for scripts that want to refer to the
            # dataset builder outputs symbolically.
            "source_pretrain_split_names": [
                "pretrain_protocol_1",
                "pretrain_protocol_2",
            ],
            "target_adapt_split_name": "adapt_protocol_3",
            "target_val_split_name": "val_protocol_3",
            "target_test_split_name": "test_protocol_3",

            # --------------------------------------------------------------
            # Fantasy acquisition behavior
            # --------------------------------------------------------------
            "selector_objective": "fantasy_gain",
            "selector_metric": "target_validation_loss",
            "selector_holdout_name": "target_validation",
            "cost_normalize_selector_score": True,
            "target_validation_loss_fn": None,

            # Candidate / query behavior
            "sink_only": False,
            "enable_partial_queries": True,
            "group_var_reduction": "sum",
            "use_upstream_first": False,
            "uncertainty_threshold_tau": 0.5,

            # Candidate subsampling for compute reduction
            # e.g. evaluate at most 20 candidates per protocol each round
            "max_candidates_per_protocol": 20,

            # Fantasy lookahead
            "fantasy_train_steps": 20,
            "fantasy_topk_candidates": 8,
            "fantasy_topk_groups": 2,
            "mc_samples": 8,
            "n_posterior_samples": 64,

            # --------------------------------------------------------------
            # Predictor backend selection
            # --------------------------------------------------------------
            # "mcd" or "dkl"
            "predictor_type": "mcd",

            # Generic MCD / MLP-style defaults
            "hidden": 64,
            "depth": 2,
            "p_drop": 0.1,

            # --------------------------------------------------------------
            # DKL-specific knobs
            # --------------------------------------------------------------
            "dkl_hidden": 256,
            "dkl_feature_dim": 32,
            "dkl_kernel": "rbf",
            "dkl_inference": "exact",
            "dkl_noise": 1e-4,

            # Backward-compatible aliases some factory code may expect
            "feature_dim": 32,
            "kernel_type": "rbf",

            # --------------------------------------------------------------
            # Two-stage pretrain/adapt bookkeeping
            # --------------------------------------------------------------
            # These are mainly informational in the current implementation,
            # but keeping them here makes the mode explicit and extensible.
            "run_source_pretrain": True,
            "run_target_adapt": True,

            # If needed later, scripts can choose to override these independently.
            "pretrain_train_steps": None,
            "adapt_train_steps": None,

            # --------------------------------------------------------------
            # Logging / analysis
            # --------------------------------------------------------------
            "acquisition_rule": "expected_target_val_loss_reduction_per_cost",
            "debug_selector": True,
        }
    )

    if overrides is not None:
        options.update(dict(overrides))

    return options


__all__ = ["get_fantasy_protocol1a_options"]