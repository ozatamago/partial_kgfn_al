#!/usr/bin/env python3

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

from partial_alfn.policies.candidates import make_candidates
from partial_alfn.uncertainty.mc_dropout import mc_predict_mean_var


def get_suggested_node_and_input(
    algo: str,
    remaining_budget: float,
    problem: SyntheticTestFunction,
    predictor: nn.Module,
    options: Optional[Dict] = None,
) -> Tuple[Tensor, Optional[List[int]], Optional[object], Optional[object]]:
    """
    Return the next input and optionally a node-group to evaluate.

    Current stage-1 behavior:
      - Random: sample a random x and do a full-network evaluation
      - NN_UQ: pick x with highest sink predictive variance and do a full-network evaluation

    Future partial-observation behavior:
      - compute node/group-wise utilities
      - return new_node != None for partial queries
      - incorporate utility / cost and upstream-first rules

    Returns:
        new_x:
            input to evaluate, shape [1, d]
        new_node:
            node-group to evaluate; None means full-network evaluation
        acq_val:
            utility score of the selected action
        node_candidate:
            optional extra record for debugging/logging
    """
    options = options or {}

    if algo == "Random":
        lb = problem.bounds[0]
        ub = problem.bounds[1]
        new_x = torch.rand([1, problem.dim], dtype=torch.get_default_dtype()) * (ub - lb) + lb
        return new_x, None, None, None

    if algo == "NN_UQ":
        mc_samples = options.get("mc_samples", 30)
        n_sobol = options.get("cand_n_sobol", 256)

        Xcand = make_candidates(problem, n_sobol=n_sobol)
        _, pred_var = mc_predict_mean_var(
            predictor=predictor,
            X=Xcand,
            mc_samples=mc_samples,
        )

        idx = torch.argmax(pred_var.view(-1))
        new_x = Xcand[idx : idx + 1]

        # stage-1: still do full-network evaluation
        return new_x, None, pred_var[idx].item(), None

    raise ValueError(f"Unsupported algo: {algo}")