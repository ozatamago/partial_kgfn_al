#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
FreeSolv problem runner
"""
import warnings

import torch
# from botorch.acquisition.objective import (
#     GenericMCObjective,
#     MCAcquisitionObjective,
#     PosteriorTransform,
# )
# from botorch.settings import debug

from partial_kgfn.models.dag import DAG
from partial_kgfn.run_one_trial import parse, run_one_trial
from partial_kgfn.test_functions.freesolv3 import Freesolv3FunctionNetwork

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
# debug._set_state(True)
import logging

logger = logging.getLogger("botorch")
logger.setLevel(logging.INFO)
logger.handlers.pop()

import torch.nn as nn

class MCDropoutMLP(nn.Module):
    """Simple MLP for regression with dropout (MC-dropout ready)."""
    def __init__(self, in_dim: int, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),  # regression scalar
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
from botorch.utils.sampling import draw_sobol_samples

def make_test_set(problem, n_test: int = 512, seed: int = 0):
    """Generate a held-out test set by sampling X and evaluating full network output."""
    torch.manual_seed(seed)
    X = (
        draw_sobol_samples(
            bounds=torch.tensor(problem.bounds, dtype=torch.get_default_dtype()),
            n=n_test,
            q=1,
        )
        .squeeze(-2)
    )
    with torch.no_grad():
        Y_full = problem.evaluate(X)          # full node outputs
        y = Y_full[..., [-1]]                 # sink output only, shape [n_test, 1]
    return X, y




def main(
    trial: int,
    algo: str,
    costs: str,
    budget: int,
    noisy: bool = False,
    # impose_assump: bool = False,
) -> None:
    """Run one replication for the freeSolv test problem.

    Args:
        trial: Seed of the trial.
        # algo: Algorithm to use. Supported algorithms: "EI", "KG", "Random", "EIFN", "KGFN", "TSFN", "pKGFN";
        algo: Algorithm to use. Supported algorithms: "Random", "NN_UQ".
        costs: A str indicating the costs of evaluating the nodes in the network.
        budget: The total budget of the BO loop.
        noisy: A boolean variable indicating if the evaluations are noisy.
        impose_assump: A boolean variable indicating if the upstream-downstream condition is imposed

    Returns:
        None.
    """
    # construct the problem
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
    problem_name = f"freesolv{problem.dim}"
    if noisy:
        problem_name = f"freesolvN{problem.dim}"
    else:
        problem_name = f"freesolv{problem.dim}"
    # network_objective = GenericMCObjective(lambda Y: Y[..., -1])
    # set comparison metrics
    # metrics = ["obs_val", "pos_mean", "test_loss"]  # obs_val  pos_mean
    metrics = ["obs_val", "test_loss"]
        # --- NEW: NN predictor + held-out test set ---
    predictor = MCDropoutMLP(in_dim=problem.dim, hidden=256, p_drop=0.1).to(torch.get_default_dtype())
    test_X, test_y = make_test_set(problem, n_test=512, seed=trial + 12345)
    options = {
        "predictor": predictor,
        "test_X": test_X,
        "test_y": test_y,
        "task": "regression",

        # --- NEW: training knobs for run_one_trial ---
        "nn_lr": 1e-3,
        "nn_weight_decay": 1e-6,
        "nn_train_steps": 200,      # per BO round retrain/finetune steps
        "nn_batch_size": 64,

        # --- NEW: MC-dropout knobs (if you later compute UQ) ---
        "mc_samples": 30,
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

