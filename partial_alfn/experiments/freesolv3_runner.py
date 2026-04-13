#!/usr/bin/env python3
r"""FreeSolv problem runner for AL-style NN_UQ experiments."""

import logging
import warnings

import torch

from partial_alfn.configs.al_defaults import (
    get_default_al_options,
    get_default_sink_only_options,
)
from partial_alfn.data.testset import make_test_set
from partial_alfn.models.model_factory import build_predictor
from partial_alfn.runners.al_runner import run_one_trial
from partial_alfn.runners.cli import parse
from partial_alfn.test_functions.freesolv3 import Freesolv3FunctionNetwork

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

logger = logging.getLogger("botorch")
logger.setLevel(logging.INFO)
if logger.handlers:
    logger.handlers.pop()


def _selector_options_for_mode(
    *,
    problem,
    sink_only: bool,
    sink_selector_objective: str,
) -> dict:
    """
    Build selector-related options.

    Assumptions
    -----------
    - sink_only=False:
        partial-query mode is enabled, and the default selector is fantasy_gain.
    - sink_only=True:
        full-evaluation mode is enabled, and selector_objective can be either
        uncertainty or fantasy_gain, assuming select_next_query.py supports both.
    """
    if sink_only:
        objective = str(sink_selector_objective).strip().lower()
        if objective not in {"uncertainty", "fantasy_gain"}:
            raise ValueError(
                "sink_selector_objective must be either "
                "'uncertainty' or 'fantasy_gain'."
            )

        return {
            "enable_partial_queries": False,
            "selector_objective": objective,
            "selector_metric": "sink_test_loss",
            "fantasy_train_steps": 20,
            "fantasy_topk_candidates": 8,
            "fantasy_topk_groups": 2,
        }

    out = {
        "enable_partial_queries": True,
        "selector_objective": "fantasy_gain",
        "selector_metric": "sink_test_loss",
        "fantasy_train_steps": 20,
        "fantasy_topk_candidates": 8,
        "fantasy_topk_groups": 2,
    }

    n_nodes = getattr(problem, "n_nodes", 0)
    if n_nodes >= 2:
        out["upstream_group_indices"] = list(range(n_nodes - 1))
        out["downstream_group_indices"] = [n_nodes - 1]

    return out


def main(
    trial: int,
    algo: str,
    costs: str,
    budget: int,
    noisy: bool = False,
    predictor_type: str = "mcd",
    hidden: int = 256,
    p_drop: float = 0.1,
    mc_samples: int = 30,
    dkl_inference: str = "exact",
    dkl_feature_dim: int = 32,
    dkl_kernel: str = "rbf",
    n_posterior_samples: int = 64,
    sink_only: bool = False,
    sink_selector_objective: str = "uncertainty",
) -> None:
    cost_options = {
        "1_1": [1, 1],
        "1_49": [1, 49],
        "1_9": [1, 9],
        "1_3": [1, 3],
        "1_5": [1, 5],
    }
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")

    problem = Freesolv3FunctionNetwork(node_costs=cost_options[costs])

    if noisy:
        problem_name = f"freesolvN{problem.dim}"
    else:
        problem_name = f"freesolv{problem.dim}"

    metrics = ["obs_val", "test_loss"]

    val_X, val_y = make_test_set(
        problem=problem,
        n_test=256,
        seed=trial + 54321,
    )
    test_X, test_y = make_test_set(
        problem=problem,
        n_test=512,
        seed=trial + 12345,
    )

    # mode-aware defaults
    if sink_only:
        options = get_default_sink_only_options(problem)
    else:
        options = get_default_al_options(problem)

    options.update(
        {
            "predictor_type": predictor_type,
            "hidden": hidden,
            "p_drop": p_drop,
            "mc_samples": mc_samples,
            "dkl_inference": dkl_inference,
            "dkl_feature_dim": dkl_feature_dim,
            "dkl_kernel": dkl_kernel,
            # aliases consumed by model_factory.py
            "feature_dim": dkl_feature_dim,
            "kernel_type": dkl_kernel,
            "n_posterior_samples": n_posterior_samples,
        }
    )

    options.update(
        _selector_options_for_mode(
            problem=problem,
            sink_only=sink_only,
            sink_selector_objective=sink_selector_objective,
        )
    )

    predictor = build_predictor(problem, options)

    options.update(
        {
            "predictor": predictor,
            "test_X": test_X,
            "test_y": test_y,
            "val_X": val_X,
            "val_y": val_y,
            "task": "regression",
        }
    )

    run_one_trial(
        problem_name=problem_name,
        problem=problem,
        algo=algo,
        trial=trial,
        metrics=metrics,
        n_init_evals=2 * problem.dim + 1,
        budget=budget,
        options=options,
        noisy=noisy,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))