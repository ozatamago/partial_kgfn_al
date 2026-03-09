#!/usr/bin/env python3

r"""
Run one trial of active learning on a function network.

Current version:
- supports sink-only fallback and partial-query mode
- supports multi-head predictor + partial buffers + partial training
- keeps all experiment state in a single mutable state dict
- distinguishes:
    * base_x : external input in the original input space
    * eval_x : actual node-specific input passed to problem.evaluate(...)
"""

import os
import random
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from botorch.logging import logger
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples

from partial_alfn.data.partial_buffers import (
    append_full_network_as_partial,
    init_partial_buffers,
)
from partial_alfn.data.update_buffers import (
    append_full_observation,
    append_partial_observation,
)
from partial_alfn.metrics.evaluation import compute_test_loss
from partial_alfn.persistence.checkpoint import (
    load_latest_nn_checkpoint,
    save_nn_checkpoint,
)
from partial_alfn.policies.select_next_query import get_suggested_node_and_input
from partial_alfn.training.train_partial import train_predictor_partial
from partial_alfn.utils.construct_obs_set import construct_obs_set
from partial_alfn.utils.effective_costs import effective_group_cost

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

    # Sink supervision buffer
    train_X_nn = X.clone()
    train_y_nn = network_output_at_X[..., [-1]].clone()

    # Original node-wise observation sets
    train_X, train_Y = construct_obs_set(
        X=X,
        Y=network_output_at_X,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )

    # Node-wise partial supervision buffers for multi-head training
    partial_buffers = init_partial_buffers(
        n_nodes=problem.n_nodes,
        x_dim=problem.dim,
        dtype=X.dtype,
        device=X.device,
    )
    append_full_network_as_partial(
        buffers=partial_buffers,
        x=X,
        y_full=network_output_at_X,
    )

    nn_optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=options.get("nn_lr", 1e-3),
        weight_decay=options.get("nn_weight_decay", 1e-6),
    )

    nn_optimizer = train_predictor_partial(
        predictor=predictor,
        partial_buffers=partial_buffers,
        train_X_sink=train_X_nn,
        train_y_sink=train_y_nn,
        sink_idx=problem.n_nodes - 1,
        n_steps=options.get("nn_train_steps", 200),
        batch_size=options.get("nn_batch_size", 64),
        aux_nodes_per_step=options.get("aux_nodes_per_step", None),
        aux_loss_weight=options.get("aux_loss_weight", 1.0),
        sink_loss_weight=options.get("sink_loss_weight", 1.0),
        optimizer=nn_optimizer,
    )

    if "obs_val" in metrics:
        best_obs_vals = [float(train_y_nn.max().item())]
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

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "train_X_nn": train_X_nn,
        "train_y_nn": train_y_nn,
        "partial_buffers": partial_buffers,
        "network_output_at_X": network_output_at_X,
        "best_obs_vals": best_obs_vals,
        "test_losses": test_losses,
        "best_test_loss": best_test_loss,
        "runtimes": [None],
        "cumulative_costs": [0.0],
        "node_selected": [None],
        "node_input_selected": [None],     # store base_x
        "node_eval_val": [None],
        "acqf_val_list": [None],
        "node_candidates": [None],
        "node_eval_counts": torch.zeros(problem.n_nodes, dtype=torch.long),
        "total_cost": 0.0,
        "predictor": predictor,
        "nn_optimizer": nn_optimizer,
    }


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

    cumulative_costs = res["cumulative_costs"]
    total_cost = float(cumulative_costs[-1]) if len(cumulative_costs) > 0 else 0.0

    return {
        "train_X": res["train_X"],
        "train_Y": res["train_Y"],
        "train_X_nn": res.get("train_X_nn", None),
        "train_y_nn": res.get("train_y_nn", None),
        "partial_buffers": res.get("partial_buffers", None),
        "network_output_at_X": res["network_output_at_X"],
        "best_obs_vals": res.get("best_obs_vals", []),
        "test_losses": res.get("test_losses", []),
        "best_test_loss": res.get("best_test_loss", float("inf")),
        "runtimes": res["runtimes"],
        "cumulative_costs": cumulative_costs,
        "node_selected": res["node_selected"],
        "node_input_selected": res["node_input_selected"],
        "node_eval_val": res["node_eval_val"],
        "acqf_val_list": res["acqf_val_list"],
        "node_candidates": res["node_candidates"],
        "node_eval_counts": res["node_eval_counts"],
        "total_cost": total_cost,
        "predictor": predictor,
        "nn_optimizer": nn_optimizer,
    }


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
        "train_X_nn": state["train_X_nn"],
        "train_y_nn": state["train_y_nn"],
        "partial_buffers": state.get("partial_buffers", None),
        "network_output_at_X": state["network_output_at_X"],
        "random_states": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
        "test_losses": state["test_losses"] if "test_loss" in metrics else None,
        "best_test_loss": state["best_test_loss"] if "test_loss" in metrics else None,
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

    while state["total_cost"] < float(budget):
        remaining_budget = float(budget) - float(state["total_cost"])
        logger.info(f"Remaining budget: {remaining_budget}")

        t0 = time.time()
        base_x, eval_x, new_node, acq_val, node_candidate = get_suggested_node_and_input(
            algo=algo,
            remaining_budget=remaining_budget,
            problem=problem,
            predictor=predictor,
            options=options,
        )
        t1 = time.time()
        logger.info(f"Optimizing the acquisition takes {t1 - t0:.4f} seconds")

        if new_node is None:
            eval_cost = effective_group_cost(problem, list(range(problem.n_nodes)))
        else:
            eval_cost = effective_group_cost(problem, new_node)

        if state["total_cost"] + eval_cost > float(budget):
            logger.info("Next evaluation would exceed budget. Stopping.")
            break

        # IMPORTANT: evaluate with node-specific input eval_x
        new_y = problem.evaluate(X=eval_x, idx=new_node)
        if noisy:
            new_y = new_y + torch.normal(0, 1, size=new_y.shape)

        if new_node is None:
            logger.info(
                f"Evaluate the full network at base input {base_x} "
                f"(eval input {eval_x}) "
                f"(acqf val: {'N/A' if algo == 'Random' else f'{float(acq_val):.4f}'}): {new_y}"
            )

            state["total_cost"] += eval_cost

            # IMPORTANT: store base_x in buffers, not eval_x
            evaluated_nodes = append_partial_observation(
                problem=problem,
                base_x=base_x,
                eval_x=eval_x,
                new_y=new_y,
                new_node=new_node,
                state=state,
            )
        else:
            selected_group_idx = None
            if isinstance(node_candidate, dict):
                selected_group_idx = node_candidate.get("selected_group_idx", None)

            selected_score = float("nan")
            if selected_group_idx is not None and torch.is_tensor(acq_val):
                selected_score = float(acq_val[selected_group_idx].item())

            logger.info(
                f"Evaluate at node {new_node} with base input {base_x} "
                f"and eval input {eval_x} "
                f"(effective cost: {eval_cost:.4f}, acqf val (over cost): {selected_score:.4f}): {new_y}"
            )

            if isinstance(node_candidate, dict):
                logger.info(
                    f"[runner] selected_stage={node_candidate.get('selected_stage')}, "
                    f"stage_uncertainty={node_candidate.get('stage_uncertainty')}"
                )
                logger.info(
                    f"[runner] group_max_uncertainty={node_candidate.get('group_max_uncertainty')}"
                )
                logger.info(
                    f"[runner] group_max_uncertainty_over_cost="
                    f"{node_candidate.get('group_max_uncertainty_over_cost')}"
                )

            state["total_cost"] += eval_cost

            # IMPORTANT: store base_x in buffers, not eval_x
            evaluated_nodes = append_partial_observation(
                problem=problem,
                base_x=base_x,
                eval_x=eval_x,
                new_y=new_y,
                new_node=new_node,
                state=state,
            )

        # Retrain after both full and partial observations
        nn_optimizer = train_predictor_partial(
            predictor=predictor,
            partial_buffers=state["partial_buffers"],
            train_X_sink=state["train_X_nn"],
            train_y_sink=state["train_y_nn"],
            sink_idx=problem.n_nodes - 1,
            n_steps=options.get("nn_train_steps", 200),
            batch_size=options.get("nn_batch_size", 64),
            aux_nodes_per_step=options.get("aux_nodes_per_step", None),
            aux_loss_weight=options.get("aux_loss_weight", 1.0),
            sink_loss_weight=options.get("sink_loss_weight", 1.0),
            optimizer=nn_optimizer,
        )
        state["nn_optimizer"] = nn_optimizer

        if "obs_val" in metrics and state["train_y_nn"] is not None and state["train_y_nn"].shape[0] > 0:
            best_obs_val = float(state["train_y_nn"].max().item())
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
        state["cumulative_costs"].append(float(state["total_cost"]))
        state["node_selected"].append(evaluated_nodes)
        state["node_input_selected"].append(base_x)   # store external input
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