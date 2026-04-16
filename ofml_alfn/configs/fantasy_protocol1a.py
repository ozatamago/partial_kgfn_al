#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional

from ofml_alfn.configs.al_defaults import get_default_al_options


def get_fantasy_protocol1a_options(
    problem=None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Options for Problem 1A fantasy acquisition.

    Design intent
    -------------
    - Use protocol-level fantasy acquisition.
    - Score each candidate by:
          expected target validation loss reduction / acquisition cost
    - Keep the existing default AL backbone and only override the knobs
      that are specific to Problem 1A fantasy selection.
    - Expose both MCD and DKL-related knobs so the experiment script can
      switch predictor backends without changing this file again.
    """
    options = get_default_al_options(problem)

    options.update(
        {
            # --------------------------------------------------------------
            # Experiment identity
            # --------------------------------------------------------------
            "experiment_name": "protocol1a_fantasy",
            "problem_family": "protocol1a",

            # --------------------------------------------------------------
            # Selector behavior
            # --------------------------------------------------------------
            "selector_objective": "fantasy_gain",
            "selector_metric": "target_validation_loss",
            "selector_holdout_name": "target_validation",
            "cost_normalize_selector_score": True,

            # Optional hook for generic selector paths.
            # If set to a callable, selector code may use it instead of a
            # generic tensor-based validation loss function.
            "target_validation_loss_fn": None,

            # --------------------------------------------------------------
            # Candidate / query behavior
            # --------------------------------------------------------------
            # Do not restrict acquisition to target-only or sink-only queries.
            "sink_only": False,
            "enable_partial_queries": True,

            # Relevant only if a lower-level uncertainty shortlist is used
            # before fantasy scoring.
            "group_var_reduction": "sum",

            # Keep all affordable groups eligible by default.
            "use_upstream_first": False,
            "uncertainty_threshold_tau": 0.5,

            # --------------------------------------------------------------
            # Fantasy lookahead
            # --------------------------------------------------------------
            # Number of update steps after appending one fantasy observation.
            "fantasy_train_steps": 20,

            # Shortlist width before fantasy scoring, if such a shortlist is used.
            "fantasy_topk_candidates": 8,
            "fantasy_topk_groups": 2,

            # Number of stochastic forward samples used by the fantasy routine.
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
            # Problem-1A-specific bookkeeping
            # --------------------------------------------------------------
            # This should be overwritten by the experiment script after the
            # concrete benchmark is built.
            "target_protocol_id": None,

            # Useful for logging / downstream analysis.
            "acquisition_rule": "expected_target_val_loss_reduction_per_cost",

            # --------------------------------------------------------------
            # Logging / debugging
            # --------------------------------------------------------------
            "debug_selector": True,
        }
    )

    if overrides is not None:
        options.update(dict(overrides))

    return options


__all__ = ["get_fantasy_protocol1a_options"]