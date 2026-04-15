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
    """
    options = get_default_al_options(problem)

    options.update(
        {
            # --------------------------------------------------------------
            # Experiment identity
            # --------------------------------------------------------------
            "experiment_name": "protocol1a_fantasy",

            # --------------------------------------------------------------
            # Selector behavior
            # --------------------------------------------------------------
            "selector_objective": "fantasy_gain",
            "selector_metric": "target_validation_loss",
            "cost_normalize_selector_score": True,

            # Interpretation note:
            # selector_holdout_X / selector_holdout_y are treated as the
            # target validation set by fantasy selection code when that
            # generic interface is used.
            "selector_holdout_name": "target_validation",

            # Optional hook for generic selector code paths.
            # If a callable is injected here, selector code may call it
            # instead of a generic tensor-based validation loss function.
            "target_validation_loss_fn": None,

            # --------------------------------------------------------------
            # Candidate / query behavior
            # --------------------------------------------------------------
            # Do not restrict acquisition to target-only or sink-only queries.
            "sink_only": False,
            "enable_partial_queries": True,

            # This is relevant only if a lower-level uncertainty shortlist is
            # still used somewhere in the pipeline before fantasy scoring.
            "group_var_reduction": "sum",

            # Keep all affordable groups eligible by default.
            # If later you want an upstream-first heuristic, this can be set
            # to True together with upstream/downstream group definitions.
            "use_upstream_first": False,
            "uncertainty_threshold_tau": 0.5,

            # --------------------------------------------------------------
            # Fantasy lookahead
            # --------------------------------------------------------------
            # Number of gradient steps taken after appending one fantasy
            # observation to score a candidate.
            "fantasy_train_steps": 20,

            # Shortlist width before fantasy scoring, if a shortlist is used.
            "fantasy_topk_candidates": 8,
            "fantasy_topk_groups": 2,

            # Number of stochastic forward samples used by the fantasy routine.
            "mc_samples": 8,
            "n_posterior_samples": 64,

            # --------------------------------------------------------------
            # Problem-1A-specific bookkeeping
            # --------------------------------------------------------------
            # This should be overwritten by the experiment script after the
            # concrete benchmark is built.
            "target_protocol_id": None,

            # Optional metadata for logging / downstream analysis.
            "problem_family": "protocol1a",
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