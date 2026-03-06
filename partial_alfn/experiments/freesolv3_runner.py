#!/usr/bin/env python3

r"""
FreeSolv problem runner for AL-style NN_UQ experiments.
"""

import logging
import warnings

import torch

from partial_alfn.models.mc_dropout_mlp import MCDropoutMLP
from partial_alfn.runners.al_runner import run_one_trial
from partial_alfn.runners.cli import parse
from partial_alfn.data.testset import make_test_set
from partial_alfn.test_functions.freesolv3 import Freesolv3FunctionNetwork

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
) -> None:
    cost_options = {
        "1_1": [1, 1],
        "1_9": [1, 9],
        "1_49": [1, 49],
    }
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")

    problem = Freesolv3FunctionNetwork(node_costs=cost_options[costs])

    if noisy:
        problem_name = f"freesolvN{problem.dim}"
    else:
        problem_name = f"freesolv{problem.dim}"

    metrics = ["obs_val", "test_loss"]

    predictor = MCDropoutMLP(
        in_dim=problem.dim,
        hidden=256,
        p_drop=0.1,
    ).to(torch.get_default_dtype())

    test_X, test_y = make_test_set(
        problem=problem,
        n_test=512,
        seed=trial + 12345,
    )

    options = {
        "predictor": predictor,
        "test_X": test_X,
        "test_y": test_y,
        "task": "regression",
        "nn_lr": 1e-3,
        "nn_weight_decay": 1e-6,
        "nn_train_steps": 200,
        "nn_batch_size": 64,
        "mc_samples": 30,
        "cand_n_sobol": 256,
        # Future partial-observation options:
        # "upstream_groups": [[0]],
        # "downstream_groups": [[1]],
        # "uncertainty_threshold_tau": 0.05,
        # "aux_loss_weight": 1.0,
        # "sink_loss_weight": 1.0,
    }

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