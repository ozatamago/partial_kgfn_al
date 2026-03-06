#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run an experiment for a function network test problem.
"""
import argparse
import gc
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.logging import logger
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor, normal
from partial_kgfn.utils.construct_obs_set import construct_obs_set

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

from pathlib import Path
from typing import Any

def _nn_ckpt_dir(results_dir: str) -> str:
    """
    Keep NN checkpoints next to results so they share the same experiment key.
    Example:
      results_dir = ./results/<exp>/<algo>/
      ckpt_dir    = ./results/<exp>/<algo>/checkpoints/
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def save_nn_checkpoint(
    *,
    results_dir: str,
    trial: int,
    step: int,
    predictor: torch.nn.Module,
    nn_optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a training-resumable checkpoint (model + optimizer + metadata) using state_dict.
    """
    ckpt_dir = _nn_ckpt_dir(results_dir)
    ckpt_path = os.path.join(ckpt_dir, f"trial_{trial}_step_{step}.pth")

    ckpt = {
        "trial": trial,
        "step": step,
        "predictor_state_dict": predictor.state_dict(),
    }
    if nn_optimizer is not None:
        ckpt["optimizer_state_dict"] = nn_optimizer.state_dict()
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, ckpt_path)
    return ckpt_path

def load_latest_nn_checkpoint(
    *,
    results_dir: str,
    trial: int,
    predictor: torch.nn.Module,
    nn_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Optional[Dict[str, Any]]:
    """
    Load the latest checkpoint for this trial if exists; restores model (+ optimizer if provided).
    Returns the loaded checkpoint dict, or None if nothing found.
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    # find all trial_{trial}_step_*.pth and pick max step
    prefix = f"trial_{trial}_step_"
    candidates = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(prefix) and fn.endswith(".pth"):
            try:
                step = int(fn[len(prefix) : -len(".pth")])
                candidates.append((step, fn))
            except ValueError:
                pass
    if not candidates:
        return None

    step, fn = max(candidates, key=lambda x: x[0])
    ckpt_path = os.path.join(ckpt_dir, fn)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    predictor.load_state_dict(ckpt["predictor_state_dict"])
    if nn_optimizer is not None and "optimizer_state_dict" in ckpt:
        nn_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt

import torch
import torch.nn.functional as F

def compute_test_loss(
    predictor,                 # your NN model: maps x -> logits (cls) or y_hat (reg)
    test_X: torch.Tensor,      # shape [n_te, d]
    test_y: torch.Tensor,      # shape [n_te] for cls labels, or [n_te, 1] for reg
    task: str = "regression",  # "classification" or "regression"
) -> float:
    """
    Returns average test loss (scalar float).
    - classification: CrossEntropyLoss on raw logits
    - regression: MSELoss
    """
    predictor.eval()
    with torch.no_grad():
        out = predictor(test_X)
        if task == "classification":
            # CrossEntropy expects raw logits and class indices
            loss = F.cross_entropy(out, test_y.long())
        else:
            # regression
            loss = F.mse_loss(out.view_as(test_y), test_y)
    return float(loss.item())

def train_predictor_regression(
    predictor: torch.nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    predictor.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            predictor.parameters(), lr=lr, weight_decay=weight_decay
        )

    n = train_X.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n, (min(batch_size, n),))
        xb = train_X[idx]
        yb = train_y[idx]

        optimizer.zero_grad()
        pred = predictor(xb)
        loss = F.mse_loss(pred.view_as(yb), yb)
        loss.backward()
        optimizer.step()

    return optimizer

def run_one_trial(
    problem_name: str,
    problem: SyntheticTestFunction,
    algo: str,
    trial: int,
    metrics: List[str],
    n_init_evals: int,
    budget: Union[float, int],
    options: Optional[Dict] = None,
    force_restart: bool = False,
    noisy: bool = False,
) -> None:
    """Run one trial of BO loop for the given problem and algorithm.

    Args:
        problem_name: A string representing the name of the test problem.
        problem: A function network test problem.
        algo: A string representing the name of the algorithm.
        trial: The seed of the trial
        metrics: A list of metrics to record. Options are "pos_mean" and "obs_val"
        n_init_evals: Number of initial evaluations.
        budget: The budget for the BO loop.
        objective: MCAcquisitionfunctionObjective used to combine function network intemediate outputs to form a function network final value
        force_restrate: a boolean indicating to restart the experiment and ignore the exisiting result, if any
        impose_assump: a boolean indicating if the upstream-downstream restriction is imposed
        noisy: a boolean indicating if the function evaluation is noisy.

    Returns:
        None.
    """
    if algo not in ["Random", "NN_UQ"]:
        raise ValueError(f"Unsupported algo for AL-only runner: {algo}")
    results_dir = f"./results/{problem_name}_{'_'.join(str(x) for x in problem.node_costs)}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(results_dir + f"trial_{trial}.pt") and not force_restart:
        logger.info(
            f"============================Resume Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )
        res = torch.load(results_dir + f"trial_{trial}.pt",weights_only=False)
        # reset the random seed
        torch.set_rng_state(res["random_states"]["torch"])
        np.random.set_state(res["random_states"]["numpy"])
        random.setstate(res["random_states"]["random"])
        # Get data
        train_X = res["train_X"]
        train_Y = res["train_Y"]
        best_obs_vals = res["best_obs_vals"]
        best_obs_val = best_obs_vals[-1]
        runtimes = res["runtimes"]
        cumulative_costs = res["cumulative_costs"]
        node_indices = res["node_selected"]
        acqf_vals = res["acqf_val_list"]
        node_inputs = res["node_input_selected"]
        node_evals = res["node_eval_val"]
        node_candidates = res["node_candidates"]
        total_cost = cumulative_costs[-1]
        count = res["node_eval_counts"]
        network_output_at_X = res["network_output_at_X"]
        test_losses = res.get("test_losses", [])
        best_test_loss = res.get("best_test_loss", float("inf"))
        train_X_nn = res.get("train_X_nn", None)
        train_y_nn = res.get("train_y_nn", None)
        predictor = options["predictor"]
        test_X = options["test_X"]
        test_y = options["test_y"]
        task = options.get("task", "regression")
        nn_optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=options.get("nn_lr", 1e-3),
            weight_decay=options.get("nn_weight_decay", 1e-6),
        )
        _ = load_latest_nn_checkpoint(
            results_dir=results_dir,
            trial=trial,
            predictor=predictor,
            nn_optimizer=nn_optimizer,
            map_location="cpu",
        )

    else:
        logger.info(
            f"============================Start New Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )
        predictor = options["predictor"]
        test_X = options["test_X"]
        test_y = options["test_y"]
        task = options.get("task", "regression")

        # Set manual seed
        torch.manual_seed(trial)
        np.random.seed(trial)
        random.seed(trial)
        # Generate initial design using SobolSampler
        X = (
            draw_sobol_samples(
                bounds=torch.Tensor(problem.bounds).to(**tkwargs),
                n=n_init_evals,
                q=1,
            )
            .squeeze(-2)
            .to(**tkwargs)
        )
        # Initialize GP network model
        network_output_at_X = problem.evaluate(X)
        if noisy:
            network_output_at_X = network_output_at_X + torch.normal(
                0, 1, size=network_output_at_X.shape
            )
        train_X_nn = X.clone()
        train_y_nn = network_output_at_X[..., [-1]].clone()

        nn_optimizer = torch.optim.Adam(
            predictor.parameters(),
            lr=options.get("nn_lr", 1e-3),
            weight_decay=options.get("nn_weight_decay", 1e-6),
        )

        nn_optimizer = train_predictor_regression(
            predictor=predictor,
            train_X=train_X_nn,
            train_y=train_y_nn,
            n_steps=options.get("nn_train_steps", 200),
            batch_size=options.get("nn_batch_size", 64),
            optimizer=nn_optimizer,
        )
        train_X, train_Y = construct_obs_set(
            X=X,
            Y=network_output_at_X,
            parent_nodes=problem.parent_nodes,
            active_input_indices=problem.active_input_indices,
        )
        if "obs_val" in metrics:
            best_obs_val = train_y_nn.max().item()
            best_obs_vals = [best_obs_val]


        if "test_loss" in metrics:
            # You need these objects available in scope:
            #   predictor: your trained NN at time t (theta_t)
            #   test_X, test_y: your held-out test set tensors
            #   task: "classification" or "regression"
            test_loss = compute_test_loss(
                predictor=predictor,
                test_X=test_X,
                test_y=test_y,
                task=task,
            )
            test_losses = [test_loss]
            best_test_loss = test_loss
            logger.info(f"Initial test loss: {test_loss:.6f}")

        runtimes = [None]
        cumulative_costs = [None]
        node_indices = [None]
        acqf_vals = [None]
        node_inputs = [None]
        node_evals = [None]
        node_candidates = [None]
        total_cost = 0
        count = torch.zeros(len(problem.parent_nodes), dtype=int)
    print("==========================================================================")
    gen_x_fantasies_count = 0
    while total_cost < budget:
        remaining_budget = budget - total_cost
        logger.info(f"Remaining budget: {remaining_budget}")
        t0 = time.time()
        new_x, new_node, node_best_acq_vals, node_candidate = get_suggested_node_and_input(
            algo=algo,
            problem=problem,
            predictor=predictor,
            options=options,
            remaining_budget=remaining_budget,
        )

        t1 = time.time()
        logger.info(f"Optimizing the acquisition takes {t1 - t0:.4f} seconds")
        # The following lines separate cases for algorithms that always compute full network and ones that compute partially.
        if new_node is None:
            eval_cost = sum(problem.node_costs)
        else:
            eval_cost = sum(problem.node_costs[k] for k in new_node)

        if total_cost + eval_cost > budget:
            break
        # Node evaluation
        new_y = problem.evaluate(X=new_x, idx=new_node)
        if noisy:
            new_y = new_y + torch.normal(0, 1, size=new_y.shape)
        # Update training data
        if new_node is None:
            if algo == "Random":
                logger.info(
                    f"Evaluate the full network at input {new_x} (acqf val: N/A): {new_y}"
                )
            else:
                logger.info(
                    f"Evaluate the full network at input {new_x} "
                    f"(acqf val: {node_best_acq_vals:.4f}): {new_y}"
                )

            total_cost += eval_cost
            count = count + torch.ones(len(problem.parent_nodes), dtype=int)

            # full-eval データも partial 用バッファも両方更新
            train_X_nn = torch.cat((train_X_nn, new_x), dim=0)
            train_y_nn = torch.cat((train_y_nn, new_y[..., [-1]]), dim=0)

            network_output_at_X = torch.cat((network_output_at_X, new_y), dim=0)

            new_obs_x, new_obs_y = construct_obs_set(
                X=new_x,
                Y=new_y,
                parent_nodes=problem.parent_nodes,
                active_input_indices=problem.active_input_indices,
            )
            for j in range(problem.n_nodes):
                train_X[j] = torch.cat((train_X[j], new_obs_x[j]), dim=0)
                train_Y[j] = torch.cat((train_Y[j], new_obs_y[j]), dim=0)

            nn_optimizer = train_predictor_regression(
                predictor=predictor,
                train_X=train_X_nn,
                train_y=train_y_nn,
                n_steps=options.get("nn_train_steps", 200),
                batch_size=options.get("nn_batch_size", 64),
                optimizer=nn_optimizer,
            )

            evaluated_nodes = list(range(problem.n_nodes))

        else:
            total_cost += eval_cost
            idx_group = problem.node_groups.index(new_node)
            logger.info(
                f"Evaluate at node {new_node} with input {new_x} "
                f"(acqf val (over cost): {node_best_acq_vals[idx_group]:.4f}): {new_y}"
            )

            idx_for_new_y = 0
            for j in new_node:
                train_X[j] = torch.cat((train_X[j], new_x), dim=0)
                train_Y[j] = torch.cat((train_Y[j], new_y[..., [idx_for_new_y]]), dim=0)
                idx_for_new_y += 1
                count[j] += 1

            evaluated_nodes = new_node

        if "obs_val" in metrics:
            best_obs_val = train_y_nn.max().item()
            best_obs_vals.append(best_obs_val)
            # logger.info(f"Best observed objective value: {best_obs_val:.4f}")

        
        if "test_loss" in metrics:
            test_loss = compute_test_loss(
                predictor=predictor,
                test_X=test_X,
                test_y=test_y,
                task=task,
            )
            test_losses.append(test_loss)
            best_test_loss = min(best_test_loss, test_loss)
            logger.info(f"Test loss: {test_loss:.6f} (best {best_test_loss:.6f})")

        logger.info(f"total cost used: {total_cost}")
        logger.info(
            "=========================================================================="
        )
        print(
            "=========================================================================="
        )
        gen_x_fantasies_count += 1
        # Store data
        runtimes.append(t1 - t0)
        cumulative_costs.append(total_cost)
        # node_indices.append(new_node)
        node_indices.append(evaluated_nodes)
        node_inputs.append(new_x)
        node_evals.append(new_y)
        acqf_vals.append(node_best_acq_vals)
        node_candidates.append(node_candidate)

        step = len(cumulative_costs) - 1  # because first entry is None
        save_nn_checkpoint(
            results_dir=results_dir,
            trial=trial,
            step=step,
            predictor=predictor,
            nn_optimizer=nn_optimizer,  # later: pass your optimizer here
            extra={
                "total_cost": float(total_cost),
                "budget": float(budget),
            },
        )

        BO_results = {
            "bo_budget": budget,
            "runtimes": runtimes,
            "cumulative_costs": cumulative_costs,
            "node_selected": node_indices,
            "node_input_selected": node_inputs,
            "node_eval_val": node_evals,
            "acqf_val_list": acqf_vals,
            # "best_post_means": best_post_means,
            # "best_design_post_mean": best_design_post_mean,
            # "obj_at_best_designs": obj_at_best_designs,
            "best_obs_vals": best_obs_vals,
            "node_eval_counts": count,
            "node_candidates": node_candidates,
            "train_X": train_X,
            "train_Y": train_Y,
            "network_output_at_X": network_output_at_X,
            "random_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "test_losses": test_losses if "test_loss" in metrics else None,
            "best_test_loss": best_test_loss if "test_loss" in metrics else None,
            "train_X_nn": train_X_nn if algo == "NN_UQ" else None,
            "train_y_nn": train_y_nn if algo == "NN_UQ" else None,
        }


import torch
import torch.nn as nn

def _enable_mc_dropout(m: nn.Module) -> None:
    # dropout だけ train() に戻す（BatchNorm 等は eval のまま）
    if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout3d):
        m.train()

@torch.no_grad()
def mc_predict_mean_var(predictor: nn.Module, X: torch.Tensor, mc_samples: int = 30):
    """
    X: [N, d]  (or [batch, q, d] -> flatten yourself)
    returns: mean [N, 1], var [N, 1]
    """
    predictor.eval()
    predictor.apply(_enable_mc_dropout)  # test-time dropout :contentReference[oaicite:1]{index=1}

    outs = []
    for _ in range(mc_samples):
        outs.append(predictor(X))  # regression scalar [N,1]
    Y = torch.stack(outs, dim=0)   # [S, N, 1]
    mean = Y.mean(dim=0)
    var = Y.var(dim=0, unbiased=False)
    return mean, var

from botorch.utils.sampling import draw_sobol_samples

def make_candidates(problem, n_sobol=256):
    bounds = problem.bounds.to(dtype=torch.get_default_dtype())
    Xs = draw_sobol_samples(bounds=bounds, n=n_sobol, q=1).squeeze(-2)
    return Xs


def get_suggested_node_and_input(
    algo: str,
    remaining_budget: float,
    problem: SyntheticTestFunction,
    predictor: nn.Module,
    options: Optional[Dict] = None,
) -> Tuple[Tensor, Optional[List[int]], Optional[float], Optional[object]]:
    """Return the next input and optionally a node-group to evaluate.

    Returns:
        A tuple of:
            - new_x: input to evaluate
            - new_node: node-group to evaluate; None means full-network evaluation
            - acq_val: utility score of the selected action
            - node_candidate: optional extra record for debugging/logging
    """
    if algo == "Random":
        new_x = (
            torch.rand([1, problem.dim]) * (problem.bounds[1] - problem.bounds[0])
            + problem.bounds[0]
        )
        return new_x, None, None, None
    elif algo == "NN_UQ":
        mc_samples = (options or {}).get("mc_samples", 30)
        n_sobol = (options or {}).get("cand_n_sobol", 256)

        Xcand = make_candidates(problem, n_sobol=n_sobol)
        _, v = mc_predict_mean_var(predictor, Xcand, mc_samples=mc_samples)

        idx = torch.argmax(v.view(-1))
        new_x = Xcand[idx:idx+1]
        return new_x, None, v[idx].item(), None




def parse():
    parser = argparse.ArgumentParser(
        description="Run one replication of an AL experiment."
    )
    parser.add_argument("--trial", "-t", type=int, default=0)
    parser.add_argument("--algo", "-a", type=str, default="NN_UQ",
                        choices=["Random", "NN_UQ"])
    parser.add_argument("--costs", "-c", type=str, required=True)
    parser.add_argument("--budget", "-b", type=int, default=200)
    return parser.parse_args()
