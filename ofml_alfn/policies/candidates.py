#!/usr/bin/env python3

import torch
from botorch.utils.sampling import draw_sobol_samples


def make_candidates(problem, n_sobol: int = 256) -> torch.Tensor:
    """
    Generate candidate inputs using Sobol sampling over the problem bounds.

    Args:
        problem: function-network problem with attribute `bounds`
        n_sobol: number of Sobol candidates

    Returns:
        Xcand: tensor of shape [n_sobol, d]
    """
    bounds = problem.bounds.to(dtype=torch.get_default_dtype())
    Xcand = draw_sobol_samples(bounds=bounds, n=n_sobol, q=1).squeeze(-2)
    return Xcand