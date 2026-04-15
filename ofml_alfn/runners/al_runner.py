#!/usr/bin/env python3
r"""
Run one trial of active learning on a function network.

Minimal OFML-adjusted runner:
- uses ofml_alfn imports throughout
- prepares selector_holdout_X / selector_holdout_y from val_X / val_y
- supports both full-network and partial-query acquisition
- keeps experiment state in a mutable dict compatible with select_next_query.py
- trains through train_predictor_partial_backend(...)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from botorch.logging import logger
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples

from ofml_alfn.data.partial_buffers import (
    append_full_network_as_partial,
    init_partial_buffers,
)
from ofml_alfn.data.update_buffers import (
    append_full_observation,
    append_partial_observation,
)
from ofml_alfn.metrics.evaluation import (
    append_node_metric_history,
    build_node_test_sets,
    compute_rollout_node_losses,
    compute_teacher_forced_node_losses,
    compute_test_loss,
    compute_weighted_node_loss,
    init_node_metric_history,
)
from ofml_alfn.policies.select_next_query import get_suggested_node_and_input
from ofml_alfn.training.train_factory import train_predictor_partial_backend
from ofml_alfn.utils.construct_obs_set import construct_obs_set
from ofml_alfn.utils.effective_costs import effective_group_cost

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_test_artifacts(
    *,
    problem: SyntheticTestFunction,
    options: Dict,
) -> None:
    """
    Create test artifacts used for reporting.
    """
    test_X = options["test_X"]
    if (
        "full_test_Y" not in options
        or options["full_test_Y"] is None
        or "node_test_X" not in options
        or options["node_test_X"] is None
        or "node_test_Y" not in options
        or options["node_test_Y"] is None
    ):
        node_test_X, node_test_Y, full_test_Y = build_node_test_sets(
            problem=problem,
            base_X=test_X,
            full_Y=None,
        )
        options["full_test_Y"] = full_test_Y
        options["node_test_X"] = node_test_X
        options["node_test_Y"] = node_test_Y


def _prepare_selector_holdout(options: Dict) -> None:
    """
    Prepare the holdout set used by the acquisition selector.

    Preferred:
      - options["val_X"], options["val_y"]

    Fallback:
      - options["test_X"], options["test_y"]

    For the fantasy selector we interpret this holdout as the target validation set.
    """
    if "selector_holdout_X" not in options or options["selector_holdout_X"] is None:
        if "val_X" in options and options["val_X"] is not None:
            options["selector_holdout_X"] = options["val_X"]
        else:
            options["selector_holdout_X"] = options["test_X"]
            logger.warning(
                "selector_holdout_X was not provided. Falling back to test_X. "
                "For fantasy selection, pass val_X explicitly."
            )

    if "selector_holdout_y" not in options or options["selector_holdout_y"] is None:
        if "val_y" in options and options["val_y"] is not None:
            options["selector_holdout_y"] = options["val_y"]
        else:
            options["selector_holdout_y"] = options["test_y"]
            logger.warning(
                "selector_holdout_y was not provided. Falling back to test_y. "
                "For fantasy selection, pass val_y explicitly."
            )


def _draw_initial_design(
    *,
    problem: SyntheticTestFunction,
    n_init_evals: int,
) -> torch.Tensor:
    """
    Sobol initial design in the external input space.
    """
    bounds = problem.bounds.to(**tkwargs)
    X = draw_sobol_samples(
        bounds=bounds,
        n=int(n_init_evals),
        q=1,
    ).squeeze(1)
    return X


def _evaluate_problem_full(
    *,
    problem: SyntheticTestFunction,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Full network evaluation.
    """
    return problem.evaluate(x)


def _evaluate_problem_partial(
    *,
    problem: SyntheticTestFunction,
    eval_x: torch.Tensor,
    new_node: Sequence[int],
) -> torch.Tensor:
    """
    Partial-group evaluation.

    This helper tries a few plausible calling conventions because the exact
    test-function signature may differ across problem classes.
    """
    node_group = list(new_node)

    # Most likely convention in this repo family.
    try:
        return problem.evaluate(eval_x, idx=node_group)
    except TypeError:
        pass

    try:
        return problem.evaluate(eval_x, idx=node_group[0] if len(node_group) == 1 else node_group)
    except TypeError:
        pass

    try:
        return problem.evaluate(eval_x, node_indices=node_group)
    except TypeError:
        pass

    try:
        return problem.evaluate(eval_x, node_idx=node_group[0] if len(node_group) == 1 else node_group)
    except TypeError:
        pass

    raise TypeError(
        "Could not call partial evaluation on problem. "
        "Expected something like problem.evaluate(eval_x, idx=new_node)."
    )


def _init_state_from_full_evals(
    *,
    problem: SyntheticTestFunction,
    init_X: torch.Tensor,
    init_Y_full: torch.Tensor,
    options: Dict,
) -> Dict[str, Any]:
    """
    Build the mutable runner state from initial full evaluations.
    """
    train_X_nodes, train_Y_nodes = construct_obs_set(
        X=init_X,
        Y=init_Y_full,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )

    use_partial_buffers = bool(options.get("enable_partial_queries", False))
    partial_buffers = None
    if use_partial_buffers:
        partial_buffers = init_partial_buffers(
            n_nodes=int(problem.n_nodes),
            x_dim=int(problem.dim),
            dtype=init_X.dtype,
            device=init_X.device,
        )
        append_full_network_as_partial(
            buffers=partial_buffers,
            x=init_X,
            y_full=init_Y_full,
        )

    state: Dict[str, Any] = {
        "train_X": [x.clone() for x in train_X_nodes],
        "train_Y": [y.clone() for y in train_Y_nodes],
        "train_X_nn": init_X.clone(),
        "train_y_nn": init_Y_full[:, [-1]].clone(),
        "network_output_at_X": init_Y_full.clone(),
        "node_eval_counts": torch.full(
            (int(problem.n_nodes),),
            fill_value=int(init_X.shape[0]),
            dtype=torch.long,
            device=init_X.device,
        ),
        "partial_buffers": partial_buffers,
        "chosen_nodes": [],
        "chosen_costs": [],
        "elapsed_seconds": [],
        "test_loss_history": [],
        "obs_val_history": [],
    }
    return state


def _group_or_full_cost(
    *,
    problem: SyntheticTestFunction,
    new_node: Optional[Sequence[int]],
) -> float:
    if new_node is None:
        return float(sum(problem.node_costs))
    return float(effective_group_cost(problem, list(new_node)))


def _append_observation(
    *,
    problem: SyntheticTestFunction,
    state: Dict[str, Any],
    base_x: torch.Tensor,
    eval_x: torch.Tensor,
    new_y: torch.Tensor,
    new_node: Optional[Sequence[int]],
) -> List[int]:
    if new_node is None:
        return append_full_observation(
            problem=problem,
            new_x=base_x,
            new_y=new_y,
            state=state,
        )

    return append_partial_observation(
        problem=problem,
        base_x=base_x,
        eval_x=eval_x,
        new_y=new_y,
        new_node=list(new_node),
        state=state,
        sink_node_idx=int(problem.n_nodes - 1),
    )


def _train_predictor_in_place(
    *,
    predictor,
    state: Dict[str, Any],
    options: Dict,
    verbose: bool = False,
):
    return train_predictor_partial_backend(
        predictor=predictor,
        train_X_nodes=state["train_X"],
        train_Y_nodes=state["train_Y"],
        options=options,
        sink_idx=int(len(state["train_X"]) - 1),
        optimizer=None,
        verbose=verbose,
    )


def _compute_reporting_metrics(
    *,
    predictor,
    problem: SyntheticTestFunction,
    options: Dict,
    metrics: Sequence[str],
    task: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "test_loss" in metrics:
        out["test_loss"] = float(
            compute_test_loss(
                predictor=predictor,
                test_X=options["test_X"],
                test_y=options["test_y"],
                task=task,
            )
        )

    if "obs_val" in metrics:
        out["obs_val"] = float(
            compute_test_loss(
                predictor=predictor,
                test_X=options["selector_holdout_X"],
                test_y=options["selector_holdout_y"],
                task=task,
            )
        )

    if "teacher_forced_node_loss" in metrics:
        tf_losses = compute_teacher_forced_node_losses(
            predictor=predictor,
            node_test_X=options["node_test_X"],
            node_test_Y=options["node_test_Y"],
            task=task,
        )
        out["teacher_forced_node_losses"] = tf_losses
        out["teacher_forced_node_loss"] = float(
            compute_weighted_node_loss(tf_losses)
        )

    if "rollout_node_loss" in metrics:
        rollout_losses = compute_rollout_node_losses(
            predictor=predictor,
            base_X=options["test_X"],
            full_Y=options["full_test_Y"],
            task=task,
        )
        out["rollout_node_losses"] = rollout_losses
        out["rollout_node_loss"] = float(
            compute_weighted_node_loss(rollout_losses)
        )

    return out


def run_one_trial(
    *,
    problem_name: str,
    problem: SyntheticTestFunction,
    algo: str,
    trial: int,
    metrics: Sequence[str],
    n_init_evals: int,
    budget: Union[int, float],
    options: Dict[str, Any],
    noisy: bool = False,
) -> Dict[str, Any]:
    """
    Run one AL trial.

    Expected inputs in options
    --------------------------
    predictor:
        Predictor object to update in place.
    test_X, test_y:
        Final evaluation set.
    val_X, val_y:
        Validation set used by the fantasy selector.
    task:
        "regression" or "classification"
    """
    start_time = time.time()
    _set_seed(int(trial))

    predictor = options["predictor"]
    task = str(options.get("task", "regression"))

    _prepare_test_artifacts(problem=problem, options=options)
    _prepare_selector_holdout(options)

    # --------------------------------------------------------------
    # Initial full-network evaluations
    # --------------------------------------------------------------
    init_X = _draw_initial_design(
        problem=problem,
        n_init_evals=int(n_init_evals),
    )
    init_Y_full = _evaluate_problem_full(
        problem=problem,
        x=init_X,
    )

    state = _init_state_from_full_evals(
        problem=problem,
        init_X=init_X,
        init_Y_full=init_Y_full,
        options=options,
    )

    spent_budget = float(sum(problem.node_costs)) * float(n_init_evals)

    # Initial train
    _train_predictor_in_place(
        predictor=predictor,
        state=state,
        options=options,
        verbose=bool(options.get("verbose", False)),
    )

    history: List[Dict[str, Any]] = []
    teacher_forced_history: Optional[Dict[int, List[float]]] = None
    rollout_history: Optional[Dict[int, List[float]]] = None

    initial_metrics = _compute_reporting_metrics(
        predictor=predictor,
        problem=problem,
        options=options,
        metrics=metrics,
        task=task,
        state=state,
    )

    if "teacher_forced_node_losses" in initial_metrics:
        teacher_forced_history = init_node_metric_history(
            initial_metrics["teacher_forced_node_losses"]
        )
    if "rollout_node_losses" in initial_metrics:
        rollout_history = init_node_metric_history(
            initial_metrics["rollout_node_losses"]
        )

    history.append(
        {
            "round": 0,
            "spent_budget": float(spent_budget),
            "remaining_budget": float(max(float(budget) - spent_budget, 0.0)),
            "selected_node": None,
            "selection_info": None,
            **{
                k: v
                for k, v in initial_metrics.items()
                if not k.endswith("_losses")
            },
        }
    )

    logger.info(
        f"[trial={trial}] init done | spent_budget={spent_budget:.3f} "
        f"| metrics={ {k: v for k, v in initial_metrics.items() if not k.endswith('_losses')} }"
    )

    # --------------------------------------------------------------
    # Active learning loop
    # --------------------------------------------------------------
    round_idx = 0
    while spent_budget < float(budget):
        remaining_budget = float(budget) - spent_budget

        base_x, eval_x, new_node, acq_val, node_candidate = get_suggested_node_and_input(
            algo=algo,
            remaining_budget=remaining_budget,
            problem=problem,
            predictor=predictor,
            options=options,
            state=state,
        )

        obs_cost = _group_or_full_cost(
            problem=problem,
            new_node=new_node,
        )
        if obs_cost > remaining_budget:
            logger.info(
                f"[trial={trial}] stopping: selected action cost={obs_cost:.3f} "
                f"exceeds remaining_budget={remaining_budget:.3f}"
            )
            break

        if new_node is None:
            new_y = _evaluate_problem_full(
                problem=problem,
                x=eval_x,
            )
        else:
            new_y = _evaluate_problem_partial(
                problem=problem,
                eval_x=eval_x,
                new_node=new_node,
            )

        observed_nodes = _append_observation(
            problem=problem,
            state=state,
            base_x=base_x,
            eval_x=eval_x,
            new_y=new_y,
            new_node=new_node,
        )

        _train_predictor_in_place(
            predictor=predictor,
            state=state,
            options=options,
            verbose=False,
        )

        spent_budget += obs_cost
        round_idx += 1

        metric_values = _compute_reporting_metrics(
            predictor=predictor,
            problem=problem,
            options=options,
            metrics=metrics,
            task=task,
            state=state,
        )

        if "teacher_forced_node_losses" in metric_values:
            if teacher_forced_history is None:
                teacher_forced_history = init_node_metric_history(
                    metric_values["teacher_forced_node_losses"]
                )
            else:
                teacher_forced_history = append_node_metric_history(
                    teacher_forced_history,
                    metric_values["teacher_forced_node_losses"],
                )

        if "rollout_node_losses" in metric_values:
            if rollout_history is None:
                rollout_history = init_node_metric_history(
                    metric_values["rollout_node_losses"]
                )
            else:
                rollout_history = append_node_metric_history(
                    rollout_history,
                    metric_values["rollout_node_losses"],
                )

        step_info = {
            "round": int(round_idx),
            "spent_budget": float(spent_budget),
            "remaining_budget": float(max(float(budget) - spent_budget, 0.0)),
            "selected_node": None if new_node is None else list(new_node),
            "observed_nodes": list(observed_nodes),
            "obs_cost": float(obs_cost),
            "acq_val": (
                None
                if acq_val is None
                else (
                    float(acq_val.max().item())
                    if torch.is_tensor(acq_val)
                    else float(acq_val)
                )
            ),
            "selection_info": node_candidate,
            **{
                k: v
                for k, v in metric_values.items()
                if not k.endswith("_losses")
            },
        }
        history.append(step_info)

        logger.info(
            f"[trial={trial}] round={round_idx} "
            f"| selected_node={step_info['selected_node']} "
            f"| obs_cost={obs_cost:.3f} "
            f"| spent_budget={spent_budget:.3f} "
            f"| remaining_budget={max(float(budget) - spent_budget, 0.0):.3f}"
        )

    elapsed = time.time() - start_time

    result = {
        "problem_name": problem_name,
        "trial": int(trial),
        "algo": str(algo),
        "noisy": bool(noisy),
        "budget": float(budget),
        "spent_budget": float(spent_budget),
        "elapsed_seconds": float(elapsed),
        "history": history,
        "state": state,
        "predictor": predictor,
    }

    if teacher_forced_history is not None:
        result["teacher_forced_history"] = teacher_forced_history
    if rollout_history is not None:
        result["rollout_history"] = rollout_history

    return result