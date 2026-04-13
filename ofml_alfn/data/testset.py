#!/usr/bin/env python3

import torch
from botorch.utils.sampling import draw_sobol_samples


def make_test_set(problem, n_test: int = 512, seed: int = 0):
    """
    Generate a held-out test set by sampling X and evaluating the full network.

    Args:
        problem: function-network test problem
        n_test: number of test inputs
        seed: random seed for Sobol sampling

    Returns:
        test_X: [n_test, d]
        test_y: [n_test, 1]  (sink output only)
    """
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
        Y_full = problem.evaluate(X)   # full node outputs
        y = Y_full[..., [-1]]          # sink output only

    return X, y