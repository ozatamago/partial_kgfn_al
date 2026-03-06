#!/usr/bin/env python3

r"""
Run one trial of active learning on a function network.

Stage-1 refactor:
- keep current NN_UQ behavior (sink-only predictor, full-network querying for NN_UQ)
- separate orchestration from experiment-specific wiring
- leave clear hooks for future partial-observation support
"""

import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.logging import logger
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from partial_alfn.metrics.evaluation import compute_test_loss
from partial_alfn.persistence.checkpoint import (
    load_latest_nn_checkpoint,
    save_nn_checkpoint,
)
from partial_alfn.policies.select_next_query import get_suggested_node_and_input
from partial_alfn.training.train_sink import train_predictor_regression
from partial_alfn.utils.construct_obs_set import construct_obs_set

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


def _make_results_dir(problem_name: str, problem: SyntheticTestFunction, algo: str) -> str:
    results_dir = f"./results/{problem_name}_{'_'.join(str(x) for x in problem.node_costs)}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _initialize_new_experiment(
    *,
    problem: SyntheticTestFunction,
    problem_name: str,
    algo: str,
    trial: int,
    n_init_evals: int,
    metrics: List[str],
    options: Dict,
    noisy: bool,
) -> Dict:
    predictor = options["predictor"]
    test_X = options["test_X"]
    test_y = options["test_y"]
    task = options.get("task", "regression")

    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    X = (
        draw_sobol_samples(
            bounds=torch.tensor(problem.bounds, **tkwargs),
            n=n_init_evals,
            q=1,
        )
        .squeeze(-2)
        .to(**tkwargs)
    )

    network_output_at_X = problem.evaluate(X)
    if noisy:
        network_output_at_X = network_output_at_X + torch.normal(
            0, 1, size=network_output_at_X.shape
        )

    # sink supervision buffer (current stage-1 implementation)
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

    # node-wise observation sets from full initial design
    train_X, train_Y = construct_obs_set(
        X=X,
        Y=network_output_at_X,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )

    if "obs_val" in metrics:
        best_obs_val = train_y_nn.max().item()
        best_obs_vals = [best_obs_val]
    else:
        best_obs_vals = []

    if "test_loss" in metrics:
        test_loss = compute_test_loss(
            predictor=predictor,
            test_X=test_X,
            test_y=test_y,
            task=task,
        )
        test_losses = [test_loss]
        best_test_loss = test_loss
        logger.info(f"Initial test loss: {test_loss:.6f}")
    else:
        test_losses = []
        best_test_loss = float("inf")

    state = {
        "train_X": train_X,
        "train_Y": train_Y,
        "train_X_nn": train_X_nn,
        "train_y_nn": train_y_nn,
        "network_output_at_X": network_output_at_X,
        "best_obs_vals": best_obs_vals,
        "test_losses": test_losses,
        "best_test_loss": best_test_loss,
        "runtimes": [None],
        "cumulative_costs": [None],
        "node_selected": [None],
        "node_input_selected": [None],
        "node_eval_val": [None],
        "acqf_val_list": [None],
        "node_candidates": [None],
        "node_eval_counts": torch.zeros(len(problem.parent_nodes), dtype=int),
        "total_cost": 0.0,
        "predictor": predictor,
        "nn_optimizer": nn_optimizer,
    }
    return state


def _resume_experiment(
    *,
    results_dir: str,
    trial: int,
    options: Dict,
) -> Dict:
    res = torch.load(os.path.join(results_dir, f"trial_{trial}.pt"), weights_only=False)

    torch.set_rng_state(res["random_states"]["torch"])
    np.random.set_state(res["random_states"]["numpy"])
    random.setstate(res["random_states"]["random"])

    predictor = options["predictor"]
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

    state = {
        "train_X": res["train_X"],
        "train_Y": res["train_Y"],
        "train_X_nn": res.get("train_X_nn", None),
        "train_y_nn": res.get("train_y_nn", None),
        "network_output_at_X": res["network_output_at_X"],
        "best_obs_vals": res.get("best_obs_vals", []),
        "test_losses": res.get("test_losses", []),
        "best_test_loss": res.get("best_test_loss", float("inf")),
        "runtimes": res["runtimes"],
        "cumulative_costs": res["cumulative_costs"],
        "node_selected": res["node_selected"],
        "node_input_selected": res["node_input_selected"],
        "node_eval_val": res["node_eval_val"],
        "acqf_val_list": res["acqf_val_list"],
        "node_candidates": res["node_candidates"],
        "node_eval_counts": res["node_eval_counts"],
        "total_cost": res["cumulative_costs"][-1],
        "predictor": predictor,
        "nn_optimizer": nn_optimizer,
    }
    return state


def _append_full_observation(
    *,
    problem: SyntheticTestFunction,
    new_x: Tensor,
    new_y: Tensor,
    state: Dict,
) -> List[int]:
    state["train_X_nn"] = torch.cat((state["train_X_nn"], new_x), dim=0)
    state["train_y_nn"] = torch.cat((state["train_y_nn"], new_y[..., [-1]]), dim=0)
    state["network_output_at_X"] = torch.cat((state["network_output_at_X"], new_y), dim=0)

    new_obs_x, new_obs_y = construct_obs_set(
        X=new_x,
        Y=new_y,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )
    for j in range(problem.n_nodes):
        state["train_X"][j] = torch.cat((state["train_X"][j], new_obs_x[j]), dim=0)
        state["train_Y"][j] = torch.cat((state["train_Y"][j], new_obs_y[j]), dim=0)

    state["node_eval_counts"] = state["node_eval_counts"] + torch.ones(
        len(problem.parent_nodes), dtype=int
    )
    return list(range(problem.n_nodes))


def _append_partial_observation(
    *,
    problem: SyntheticTestFunction,
    new_x: Tensor,
    new_y: Tensor,
    new_node: List[int],
    state: Dict,
) -> List[int]:
    # Stage-1 behavior:
    # update node-wise observation buffers only.
    # Future partial-observation learning should also update partial NN buffers here.
    idx_for_new_y = 0
    for j in new_node:
        state["train_X"][j] = torch.cat((state["train_X"][j], new_x), dim=0)
        state["train_Y"][j] = torch.cat((state["train_Y"][j], new_y[..., [idx_for_new_y]]), dim=0)
        state["node_eval_counts"][j] += 1
        idx_for_new_y += 1
    return new_node


def _save_trial_state(
    *,
    results_dir: str,
    trial: int,
    budget: Union[int, float],
    state: Dict,
    metrics: List[str],
) -> None:
    bo_results = {
        "bo_budget": budget,
        "runtimes": state["runtimes"],
        "cumulative_costs": state["cumulative_costs"],
        "node_selected": state["node_selected"],
        "node_input_selected": state["node_input_selected"],
        "node_eval_val": state["node_eval_val"],
        "acqf_val_list": state["acqf_val_list"],
        "best_obs_vals": state["best_obs_vals"],
        "node_eval_counts": state["node_eval_counts"],
        "node_candidates": state["node_candidates"],
        "train_X": state["train_X"],
        "train_Y": state["train_Y"],
        "network_output_at_X": state["network_output_at_X"],
        "random_states": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "test_losses": state["test_losses"] if "test_loss" in metrics else None,
        "best_test_loss": state["best_test_loss"] if "test_loss" in metrics else None,
        "train_X_nn": state["train_X_nn"],
        "train_y_nn": state["train_y_nn"],
    }
    torch.save(bo_results, os.path.join(results_dir, f"trial_{trial}.pt"))


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
    if algo not in ["Random", "NN_UQ"]:
        raise ValueError(f"Unsupported algo for AL-only runner: {algo}")

    options = options or {}
    results_dir = _make_results_dir(problem_name, problem, algo)

    if os.path.exists(os.path.join(results_dir, f"trial_{trial}.pt")) and not force_restart:
        logger.info(
            f"============================Resume Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )
        state = _resume_experiment(
            results_dir=results_dir,
            trial=trial,
            options=options,
        )
    else:
        logger.info(
            f"============================Start New Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Trial: {trial}"
        )
        state = _initialize_new_experiment(
            problem=problem,
            problem_name=problem_name,
            algo=algo,
            trial=trial,
            n_init_evals=n_init_evals,
            metrics=metrics,
            options=options,
            noisy=noisy,
        )

    predictor = state["predictor"]
    nn_optimizer = state["nn_optimizer"]
    test_X = options["test_X"]
    test_y = options["test_y"]
    task = options.get("task", "regression")

    while state["total_cost"] < budget:
        remaining_budget = budget - state["total_cost"]
        logger.info(f"Remaining budget: {remaining_budget}")

        t0 = time.time()
        new_x, new_node, acq_val, node_candidate = get_suggested_node_and_input(
            algo=algo,
            remaining_budget=remaining_budget,
            problem=problem,
            predictor=predictor,
            options=options,
        )
        t1 = time.time()
        logger.info(f"Optimizing the acquisition takes {t1 - t0:.4f} seconds")

        if new_node is None:
            eval_cost = sum(problem.node_costs)
        else:
            eval_cost = sum(problem.node_costs[k] for k in new_node)

        if state["total_cost"] + eval_cost > budget:
            break

        new_y = problem.evaluate(X=new_x, idx=new_node)
        if noisy:
            new_y = new_y + torch.normal(0, 1, size=new_y.shape)

        if new_node is None:
            logger.info(
                f"Evaluate the full network at input {new_x} "
                f"(acqf val: {'N/A' if algo == 'Random' else f'{acq_val:.4f}'}): {new_y}"
            )
            state["total_cost"] += eval_cost
            evaluated_nodes = _append_full_observation(
                problem=problem,
                new_x=new_x,
                new_y=new_y,
                state=state,
            )

            nn_optimizer = train_predictor_regression(
                predictor=predictor,
                train_X=state["train_X_nn"],
                train_y=state["train_y_nn"],
                n_steps=options.get("nn_train_steps", 200),
                batch_size=options.get("nn_batch_size", 64),
                optimizer=nn_optimizer,
            )
            state["nn_optimizer"] = nn_optimizer
        else:
            state["total_cost"] += eval_cost
            idx_group = problem.node_groups.index(new_node)
            logger.info(
                f"Evaluate at node {new_node} with input {new_x} "
                f"(acqf val (over cost): {acq_val[idx_group]:.4f}): {new_y}"
            )
            evaluated_nodes = _append_partial_observation(
                problem=problem,
                new_x=new_x,
                new_y=new_y,
                new_node=new_node,
                state=state,
            )
            # Future:
            # partial-aware trainer call goes here.

        if "obs_val" in metrics and state["train_y_nn"] is not None:
            best_obs_val = state["train_y_nn"].max().item()
            state["best_obs_vals"].append(best_obs_val)

        if "test_loss" in metrics:
            test_loss = compute_test_loss(
                predictor=predictor,
                test_X=test_X,
                test_y=test_y,
                task=task,
            )
            state["test_losses"].append(test_loss)
            state["best_test_loss"] = min(state["best_test_loss"], test_loss)
            logger.info(
                f"Test loss: {test_loss:.6f} "
                f"(best {state['best_test_loss']:.6f})"
            )

        logger.info(f"total cost used: {state['total_cost']}")
        logger.info("==========================================================================")

        state["runtimes"].append(t1 - t0)
        state["cumulative_costs"].append(state["total_cost"])
        state["node_selected"].append(evaluated_nodes)
        state["node_input_selected"].append(new_x)
        state["node_eval_val"].append(new_y)
        state["acqf_val_list"].append(acq_val)
        state["node_candidates"].append(node_candidate)

        step = len(state["cumulative_costs"]) - 1
        save_nn_checkpoint(
            results_dir=results_dir,
            trial=trial,
            step=step,
            predictor=predictor,
            nn_optimizer=nn_optimizer,
            extra={
                "total_cost": float(state["total_cost"]),
                "budget": float(budget),
            },
        )

        _save_trial_state(
            results_dir=results_dir,
            trial=trial,
            budget=budget,
            state=state,
            metrics=metrics,
        )