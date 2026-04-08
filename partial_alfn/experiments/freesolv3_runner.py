#!/usr/bin/env python3

r"""
FreeSolv problem runner for AL-style fantasy-gain NN_UQ experiments.
"""

import logging
import warnings

import torch

from partial_alfn.models.model_factory import build_predictor
from partial_alfn.runners.al_runner import run_one_trial
from partial_alfn.runners.cli import parse
from partial_alfn.data.testset import make_test_set
from partial_alfn.test_functions.freesolv3 import Freesolv3FunctionNetwork
from partial_alfn.configs.al_defaults import get_default_al_options

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

logger = logging.getLogger("botorch")
logger.setLevel(logging.INFO)
if logger.handlers:
    logger.handlers.pop()


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

    options = get_default_al_options(problem)

    # Example:
    # options["predictor_type"] = "mcd"
    # options["predictor_type"] = "dkl"

    predictor = build_predictor(problem, options)

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

    options = get_default_al_options(problem)
    options.update({
        "predictor_type": predictor_type,
        "hidden": hidden,
        "p_drop": p_drop,
        "mc_samples": mc_samples,
        "dkl_inference": dkl_inference,
        "dkl_feature_dim": dkl_feature_dim,
        "dkl_kernel": dkl_kernel,
        "feature_dim": dkl_feature_dim,
        "kernel_type": dkl_kernel,
        "n_posterior_samples": n_posterior_samples,
    })

    predictor = build_predictor(problem, options)

    options.update({
        "predictor": predictor,
        "test_X": test_X,
        "test_y": test_y,
        "val_X": val_X,
        "val_y": val_y,
        "task": "regression",
        "selector_objective": "fantasy_gain",
        "selector_metric": "sink_test_loss",
        "fantasy_train_steps": 20,
        "fantasy_topk_candidates": 8,
        "fantasy_topk_groups": 2,
        "upstream_group_indices": [0],
        "downstream_group_indices": [1],
    })

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