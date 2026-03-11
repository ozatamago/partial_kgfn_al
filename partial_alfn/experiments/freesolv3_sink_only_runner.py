#!/usr/bin/env python3

r"""
Sink-only active learning baseline for FreeSolv3.

Query policy:
- does NOT use partial node queries
- at each round, evaluates the full function network at one external input

Modeling / saved metrics:
- predicts all node outputs jointly from base_x
- uses sink uncertainty for acquisition
- saves sink test loss
- saves teacher-forced node test losses for all nodes
- saves weighted node test losses
- saves observed full node outputs

Example:
    python -m partial_alfn.experiments.freesolv3_sink_only_runner \
        --trial 0 --algo NN_UQ --costs 1_3 --budget 300
"""

import argparse
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.logging import logger
from botorch.utils.sampling import draw_sobol_samples

from partial_alfn.test_functions.freesolv3 import Freesolv3FunctionNetwork

torch.set_default_dtype(torch.float64)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class MultiOutputMCDropoutMLP(nn.Module):
    """
    Full-output regression model:
        base_x -> [node0, node1, ..., sink]
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward_node(self, x: torch.Tensor, node_idx: int) -> torch.Tensor:
        y = self.forward(x)
        return y[:, [node_idx]]

    def forward_sink(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return y[:, [-1]]


def _enable_mc_dropout(m: nn.Module) -> None:
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        m.train()


@torch.no_grad()
def mc_predict_mean_var_sink(
    predictor: nn.Module,
    X: torch.Tensor,
    mc_samples: int = 30,
):
    """
    Acquisition uses sink uncertainty only.
    """
    predictor.eval()
    predictor.apply(_enable_mc_dropout)

    outs = []
    for _ in range(mc_samples):
        outs.append(predictor.forward_sink(X))  # [N, 1]

    Y = torch.stack(outs, dim=0)  # [S, N, 1]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=False)
    return mean, var


def make_candidates(problem, n_sobol: int = 256) -> torch.Tensor:
    bounds = torch.tensor(problem.bounds, dtype=torch.get_default_dtype())
    Xcand = draw_sobol_samples(bounds=bounds, n=n_sobol, q=1).squeeze(-2)
    return Xcand


def make_test_set(problem, n_test: int = 512, seed: int = 0):
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
        Y_full = problem.evaluate(X)  # [N, n_nodes]
        y_sink = Y_full[..., [-1]]    # [N, 1]
    return X, Y_full, y_sink


def compute_sink_test_loss(
    predictor: nn.Module,
    test_X: torch.Tensor,
    test_y_sink: torch.Tensor,
) -> float:
    predictor.eval()
    with torch.no_grad():
        pred_sink = predictor.forward_sink(test_X)
        loss = F.mse_loss(pred_sink.view_as(test_y_sink), test_y_sink)
    return float(loss.item())


def compute_teacher_forced_node_losses(
    predictor: nn.Module,
    test_X: torch.Tensor,
    test_Y_full: torch.Tensor,
) -> Dict[int, float]:
    """
    For this sink-only baseline, all node predictions are direct predictions from base_x.
    We still report them as node-wise test losses so they can be compared downstream.
    """
    predictor.eval()
    with torch.no_grad():
        pred_all = predictor.forward_all(test_X)  # [N, n_nodes]

    out: Dict[int, float] = {}
    for j in range(test_Y_full.shape[1]):
        loss_j = F.mse_loss(pred_all[:, [j]], test_Y_full[:, [j]])
        out[j] = float(loss_j.item())
    return out


def compute_weighted_node_loss(
    node_losses: Dict[int, float],
    *,
    exclude_nodes: Optional[list] = None,
) -> float:
    exclude = set(exclude_nodes or [])
    vals = [float(v) for k, v in node_losses.items() if k not in exclude]
    if len(vals) == 0:
        return float("nan")
    return sum(vals) / len(vals)


def init_node_metric_history(metric_dict: Dict[int, float]) -> Dict[int, list]:
    return {int(k): [float(v)] for k, v in metric_dict.items()}


def append_node_metric_history(history: Dict[int, list], metric_dict: Dict[int, float]) -> Dict[int, list]:
    for k, v in metric_dict.items():
        kk = int(k)
        history.setdefault(kk, [])
        history[kk].append(float(v))
    return history


def train_full_output_predictor(
    predictor: nn.Module,
    train_X: torch.Tensor,
    train_Y_full: torch.Tensor,
    *,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    sink_loss_weight: float = 1.0,
    intermediate_loss_weight: float = 1.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> torch.optim.Optimizer:
    predictor.train()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    n = train_X.shape[0]
    if n == 0:
        return optimizer

    n_nodes = train_Y_full.shape[1]

    for _ in range(n_steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=train_X.device)
        xb = train_X[idx]
        yb = train_Y_full[idx]

        optimizer.zero_grad()

        pred_all = predictor.forward_all(xb)  # [B, n_nodes]

        sink_loss = F.mse_loss(pred_all[:, [-1]], yb[:, [-1]])

        if n_nodes > 1:
            inter_loss = F.mse_loss(pred_all[:, :-1], yb[:, :-1])
        else:
            inter_loss = torch.tensor(0.0, dtype=pred_all.dtype, device=pred_all.device)

        total_loss = sink_loss_weight * sink_loss + intermediate_loss_weight * inter_loss
        total_loss.backward()
        optimizer.step()

    return optimizer


def select_next_input(
    algo: str,
    problem,
    predictor: nn.Module,
    *,
    mc_samples: int = 30,
    n_sobol: int = 256,
) -> Dict:
    """
    Returns:
        {
            "base_x": [1, d],
            "score": float or None,
            "debug": dict,
        }
    """
    if algo == "Random":
        lb = problem.bounds[0]
        ub = problem.bounds[1]
        base_x = (
            torch.rand([1, problem.dim], dtype=torch.get_default_dtype()) * (ub - lb)
            + lb
        )
        return {
            "base_x": base_x,
            "score": None,
            "debug": {"mode": "random"},
        }

    if algo != "NN_UQ":
        raise ValueError(f"Unsupported algo: {algo}")

    Xcand = make_candidates(problem, n_sobol=n_sobol)
    _, var = mc_predict_mean_var_sink(
        predictor=predictor,
        X=Xcand,
        mc_samples=mc_samples,
    )  # [N, 1]

    idx = torch.argmax(var.view(-1))
    base_x = Xcand[idx: idx + 1]
    score = float(var[idx].item())

    logger.info(
        f"[selector-sink-only] max_sink_uncertainty={float(var.max().item()):.6f}"
    )

    return {
        "base_x": base_x,
        "score": score,
        "debug": {
            "mode": "sink_only_uq",
            "selected_candidate_idx": int(idx.item()),
            "max_sink_uncertainty": float(var.max().item()),
        },
    }


def run_one_trial(
    *,
    problem_name: str,
    problem,
    algo: str,
    trial: int,
    budget: int,
    noisy: bool,
) -> None:
    if algo not in ["Random", "NN_UQ"]:
        raise ValueError(f"Unsupported algo: {algo}")

    results_dir = f"./results_sink_only/{problem_name}_{'_'.join(str(x) for x in problem.node_costs)}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(
        f"============================Start Sink-Only Experiment=================================\n"
        f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
        f"Algorithm: {algo}\n"
        f"Trial: {trial}"
    )

    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    predictor = MultiOutputMCDropoutMLP(
        in_dim=problem.dim,
        out_dim=problem.n_nodes,
        hidden=256,
        p_drop=0.1,
    ).to(torch.get_default_dtype())

    test_X, test_Y_full, test_y_sink = make_test_set(problem, n_test=512, seed=trial + 12345)

    # Initial design
    n_init_evals = 2 * problem.dim + 1
    init_X = (
        draw_sobol_samples(
            bounds=torch.tensor(problem.bounds, **tkwargs),
            n=n_init_evals,
            q=1,
        )
        .squeeze(-2)
        .to(**tkwargs)
    )

    init_Y_full = problem.evaluate(init_X)
    if noisy:
        init_Y_full = init_Y_full + torch.normal(0, 1, size=init_Y_full.shape)

    train_X = init_X.clone()
    train_Y_full = init_Y_full.clone()
    train_y_sink = init_Y_full[..., [-1]].clone()

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=1e-3,
        weight_decay=1e-6,
    )
    optimizer = train_full_output_predictor(
        predictor=predictor,
        train_X=train_X,
        train_Y_full=train_Y_full,
        n_steps=200,
        batch_size=64,
        sink_loss_weight=1.0,
        intermediate_loss_weight=1.0,
        optimizer=optimizer,
    )

    test_loss = compute_sink_test_loss(
        predictor=predictor,
        test_X=test_X,
        test_y_sink=test_y_sink,
    )
    best_test_loss = test_loss
    logger.info(f"Initial sink test loss: {test_loss:.6f}")

    initial_node_losses = compute_teacher_forced_node_losses(
        predictor=predictor,
        test_X=test_X,
        test_Y_full=test_Y_full,
    )

    node_test_losses_tf = init_node_metric_history(initial_node_losses)
    weighted_node_test_losses_tf_all = {-1: [compute_weighted_node_loss(initial_node_losses)]}
    weighted_node_test_losses_tf_intermediate = {
        -1: [compute_weighted_node_loss(initial_node_losses, exclude_nodes=[problem.n_nodes - 1])]
    }

    total_cost = 0.0
    full_eval_cost = float(sum(problem.node_costs))

    cumulative_costs = [0.0]
    test_losses = [test_loss]
    selected_inputs = [None]
    selected_scores = [None]
    observed_sink_vals = [None]
    observed_full_node_vals = [None]

    while total_cost < float(budget):
        remaining_budget = float(budget) - total_cost
        logger.info(f"Remaining budget: {remaining_budget}")

        if total_cost + full_eval_cost > float(budget):
            logger.info("Next full sink-only evaluation would exceed budget. Stopping.")
            break

        t0 = time.time()
        action = select_next_input(
            algo=algo,
            problem=problem,
            predictor=predictor,
            mc_samples=30,
            n_sobol=256,
        )
        t1 = time.time()

        base_x = action["base_x"]
        score = action["score"]

        logger.info(f"Optimizing the acquisition takes {t1 - t0:.4f} seconds")

        y_full = problem.evaluate(base_x)  # full protocol
        if noisy:
            y_full = y_full + torch.normal(0, 1, size=y_full.shape)

        y_sink = y_full[..., [-1]]

        logger.info(
            f"Evaluate full protocol at input {base_x} "
            f"(full cost: {full_eval_cost:.4f}, "
            f"acqf val: {'N/A' if score is None else f'{score:.4f}'}): "
            f"full_nodes={y_full}"
        )

        total_cost += full_eval_cost

        train_X = torch.cat((train_X, base_x), dim=0)
        train_Y_full = torch.cat((train_Y_full, y_full), dim=0)
        train_y_sink = torch.cat((train_y_sink, y_sink), dim=0)

        optimizer = train_full_output_predictor(
            predictor=predictor,
            train_X=train_X,
            train_Y_full=train_Y_full,
            n_steps=200,
            batch_size=64,
            sink_loss_weight=1.0,
            intermediate_loss_weight=1.0,
            optimizer=optimizer,
        )

        test_loss = compute_sink_test_loss(
            predictor=predictor,
            test_X=test_X,
            test_y_sink=test_y_sink,
        )
        best_test_loss = min(best_test_loss, test_loss)

        node_losses = compute_teacher_forced_node_losses(
            predictor=predictor,
            test_X=test_X,
            test_Y_full=test_Y_full,
        )
        node_test_losses_tf = append_node_metric_history(node_test_losses_tf, node_losses)
        weighted_node_test_losses_tf_all[-1].append(
            compute_weighted_node_loss(node_losses)
        )
        weighted_node_test_losses_tf_intermediate[-1].append(
            compute_weighted_node_loss(node_losses, exclude_nodes=[problem.n_nodes - 1])
        )

        tf_str = ", ".join([f"node{j}={v:.4f}" for j, v in sorted(node_losses.items())])
        logger.info(f"Teacher-forced node losses: {tf_str}")
        logger.info(f"Sink test loss: {test_loss:.6f} (best {best_test_loss:.6f})")
        logger.info(f"total cost used: {total_cost}")
        logger.info("==========================================================================")

        cumulative_costs.append(total_cost)
        test_losses.append(test_loss)
        selected_inputs.append(base_x.detach().cpu())
        selected_scores.append(score)
        observed_sink_vals.append(y_sink.detach().cpu())
        observed_full_node_vals.append(y_full.detach().cpu())

        torch.save(
            {
                "bo_budget": budget,
                "cumulative_costs": cumulative_costs,
                "test_losses": test_losses,
                "best_test_loss": best_test_loss,
                "selected_inputs": selected_inputs,
                "selected_scores": selected_scores,
                "observed_sink_vals": observed_sink_vals,
                "observed_full_node_vals": observed_full_node_vals,
                "train_X": train_X,
                "train_y": train_y_sink,
                "train_Y_full": train_Y_full,
                "node_test_losses_tf": node_test_losses_tf,
                "weighted_node_test_losses_tf_all": weighted_node_test_losses_tf_all,
                "weighted_node_test_losses_tf_intermediate": weighted_node_test_losses_tf_intermediate,
                "node_eval_counts": torch.tensor(
                    [float(len(cumulative_costs) - 1)] * problem.n_nodes,
                    dtype=torch.float64,
                ),
            },
            os.path.join(results_dir, f"trial_{trial}.pt"),
        )


def parse():
    parser = argparse.ArgumentParser(
        description="Run one sink-only replication of an AL experiment."
    )
    parser.add_argument("--trial", "-t", type=int, default=0)
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="NN_UQ",
        choices=["Random", "NN_UQ"],
    )
    parser.add_argument("--costs", "-c", type=str, required=True)
    parser.add_argument("--budget", "-b", type=int, default=200)
    parser.add_argument("--noisy", action="store_true")
    return parser.parse_args()


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
        "1_5": [1, 5],
        "1_3": [1, 3],
    }
    if costs not in cost_options:
        raise ValueError(f"Invalid cost option: {costs}")

    problem = Freesolv3FunctionNetwork(node_costs=cost_options[costs])
    problem_name = f"freesolv{problem.dim}" if not noisy else f"freesolvN{problem.dim}"

    run_one_trial(
        problem_name=problem_name,
        problem=problem,
        algo=algo,
        trial=trial,
        budget=budget,
        noisy=noisy,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))