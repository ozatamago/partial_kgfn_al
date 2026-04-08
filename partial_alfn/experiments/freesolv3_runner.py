#!/usr/bin/env python3

r"""
FreeSolv problem runner for AL-style fantasy-gain NN_UQ experiments.
"""

import logging
import warnings

import torch

from partial_alfn.models.multihead_mc_dropout_mlp import MultiHeadMCDropoutMLP
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

    node_input_dims = [
        len(problem.parent_nodes[j]) + len(problem.active_input_indices[j])
        for j in range(problem.n_nodes)
    ]

    predictor = MultiHeadMCDropoutMLP(
        external_input_dim=problem.dim,
        node_input_dims=node_input_dims,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
        hidden=256,
        p_drop=0.1,
        sink_idx=problem.n_nodes - 1,
    ).to(torch.get_default_dtype())

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
        "predictor": predictor,

        # Reporting holdout
        "test_X": test_X,
        "test_y": test_y,

        # Selector holdout
        "val_X": val_X,
        "val_y": val_y,

        "task": "regression",

        # Acquisition behavior
        "selector_objective": "fantasy_gain",
        "selector_metric": "sink_test_loss",
        "fantasy_train_steps": 20,
        "fantasy_topk_candidates": 8,
        "fantasy_topk_groups": 2,

        # Freesolv3-specific group override
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