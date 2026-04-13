#!/usr/bin/env python3

r"""
Sink-only active learning baseline for FreeSolv3.

Query policy:
- does NOT use partial node queries
- at each round, evaluates the full function network at one external input

Modeling / saved metrics:
- predicts all node outputs through the backend predictor
- uses sink uncertainty for acquisition
- saves sink test loss
- saves teacher-forced node test losses for all nodes
- saves weighted node test losses
- saves observed full node outputs

Resume support:
- if trial_<trial>.pt exists and --force_restart is not set,
  resume from the saved state
- DKL predictors are rehydrated from saved full evaluations

Example:
    python -m partial_alfn.experiments.freesolv3_sink_only_runner \
        --trial 0 --algo NN_UQ --costs 1_3 --budget 300 \
        --predictor_type dkl --acquisition_mode uncertainty
"""

import argparse
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from botorch.logging import logger
from botorch.utils.sampling import draw_sobol_samples

from partial_alfn.models.model_factory import build_predictor
from partial_alfn.test_functions.freesolv3 import Freesolv3FunctionNetwork
from partial_alfn.training.train_factory import (
    build_nodewise_datasets_from_full_evals,
    train_predictor_backend_from_full_evals,
)
from partial_alfn.uncertainty.base import (
    predict_mean_var_all_nodes,
    predict_sink_mean_var,
)

torch.set_default_dtype(torch.float64)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


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
        Y_full = problem.evaluate(X)   # [N, n_nodes]
        y_sink = Y_full[..., [-1]]     # [N, 1]
    return X, Y_full, y_sink


def compute_sink_test_loss(
    predictor: torch.nn.Module,
    test_X: torch.Tensor,
    test_y_sink: torch.Tensor,
) -> float:
    predictor.eval()
    with torch.no_grad():
        pred_sink = predictor.forward_sink(test_X)
        loss = F.mse_loss(pred_sink.view_as(test_y_sink), test_y_sink)
    return float(loss.item())


def compute_teacher_forced_node_losses(
    predictor: torch.nn.Module,
    test_X: torch.Tensor,
    test_Y_full: torch.Tensor,
) -> Dict[int, float]:
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


def append_node_metric_history(
    history: Dict[int, list],
    metric_dict: Dict[int, float],
) -> Dict[int, list]:
    for k, v in metric_dict.items():
        kk = int(k)
        history.setdefault(kk, [])
        history[kk].append(float(v))
    return history


def _build_backend_options(
    *,
    problem,
    predictor_type: str,
    hidden: int,
    p_drop: float,
    mc_samples: int,
    dkl_feature_dim: int,
    dkl_kernel: str,
    train_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    sink_loss_weight: float = 1.0,
    aux_loss_weight: float = 1.0,
) -> Dict:
    return {
        "predictor_type": str(predictor_type).lower(),
        "hidden": int(hidden),
        "p_drop": float(p_drop),
        "mc_samples": int(mc_samples),
        "n_samples": int(mc_samples),  # used by DKL uncertainty path
        "feature_dim": int(dkl_feature_dim),
        "kernel_type": str(dkl_kernel).lower(),
        "sink_idx": int(problem.n_nodes - 1),
        "n_steps": int(train_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "sink_loss_weight": float(sink_loss_weight),
        "aux_loss_weight": float(aux_loss_weight),
    }


def _is_dkl_predictor(predictor: torch.nn.Module) -> bool:
    return str(getattr(predictor, "predictor_type", "")).lower() == "dkl"


def _make_results_dir(problem_name: str, problem, algo: str, predictor_type: str, acquisition_mode: str) -> str:
    results_dir = (
        f"./results_sink_only/"
        f"{problem_name}_{'_'.join(str(x) for x in problem.node_costs)}/"
        f"{algo}_{str(predictor_type).lower()}_{acquisition_mode}/"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _rehydrate_dkl_predictor_from_full_evals(
    *,
    predictor: torch.nn.Module,
    problem,
    train_X: torch.Tensor,
    train_Y_full: torch.Tensor,
) -> None:
    if not _is_dkl_predictor(predictor):
        return

    train_X_nodes, train_Y_nodes = build_nodewise_datasets_from_full_evals(
        problem=problem,
        base_X=train_X,
        Y_full=train_Y_full,
    )

    if hasattr(predictor, "rehydrate_from_node_datasets"):
        predictor.rehydrate_from_node_datasets(
            train_X_nodes=train_X_nodes,
            train_Y_nodes=train_Y_nodes,
            strict=False,
        )
    else:
        for j, (xj, yj) in enumerate(zip(train_X_nodes, train_Y_nodes)):
            if xj is None or yj is None:
                continue
            if xj.shape[0] == 0:
                continue
            predictor.set_node_train_data(
                node_idx=j,
                x=xj,
                y=yj,
                strict=False,
            )

    predictor.eval()


def _save_result_state(
    *,
    results_dir: str,
    trial: int,
    budget: int,
    problem_name: str,
    problem,
    algo: str,
    predictor_type: str,
    acquisition_mode: str,
    hidden: int,
    p_drop: float,
    mc_samples: int,
    dkl_feature_dim: int,
    dkl_kernel: str,
    state: Dict[str, Any],
) -> None:
    payload = {
        "bo_budget": budget,
        "problem_name": problem_name,
        "node_costs": list(problem.node_costs),
        "algo": algo,
        "predictor_type": str(predictor_type).lower(),
        "acquisition_mode": acquisition_mode,
        "hidden": hidden,
        "p_drop": p_drop,
        "mc_samples": mc_samples,
        "dkl_feature_dim": dkl_feature_dim,
        "dkl_kernel": dkl_kernel,
        "cumulative_costs": state["cumulative_costs"],
        "test_losses": state["test_losses"],
        "best_test_loss": state["best_test_loss"],
        "selected_inputs": state["selected_inputs"],
        "selected_scores": state["selected_scores"],
        "observed_sink_vals": state["observed_sink_vals"],
        "observed_full_node_vals": state["observed_full_node_vals"],
        "train_X": state["train_X"],
        "train_y": state["train_y"],
        "train_Y_full": state["train_Y_full"],
        "node_test_losses_tf": state["node_test_losses_tf"],
        "weighted_node_test_losses_tf_all": state["weighted_node_test_losses_tf_all"],
        "weighted_node_test_losses_tf_intermediate": state["weighted_node_test_losses_tf_intermediate"],
        "node_eval_counts": state["node_eval_counts"],
        "predictor_state_dict": state["predictor"].state_dict(),
        "mcd_optimizer_state": (
            state["optimizer_state"].state_dict()
            if isinstance(state["optimizer_state"], torch.optim.Optimizer)
            else None
        ),
        "dkl_optimizer_state": (
            state["optimizer_state"]
            if isinstance(state["optimizer_state"], dict)
            else None
        ),
        "random_states": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        },
    }
    torch.save(payload, os.path.join(results_dir, f"trial_{trial}.pt"))


def _resume_experiment(
    *,
    results_dir: str,
    trial: int,
    predictor: torch.nn.Module,
    problem,
    backend_options: Dict,
) -> Dict[str, Any]:
    res = torch.load(os.path.join(results_dir, f"trial_{trial}.pt"), weights_only=False)

    if "random_states" in res:
        torch.set_rng_state(res["random_states"]["torch"])
        np.random.set_state(res["random_states"]["numpy"])
        random.setstate(res["random_states"]["random"])

    predictor.load_state_dict(res["predictor_state_dict"])

    optimizer_state: Any = None
    if _is_dkl_predictor(predictor):
        optimizer_state = res.get("dkl_optimizer_state", None)
    else:
        opt = torch.optim.Adam(
            predictor.parameters(),
            lr=float(backend_options.get("lr", 1e-3)),
            weight_decay=float(backend_options.get("weight_decay", 1e-6)),
        )
        saved_opt = res.get("mcd_optimizer_state", None)
        if saved_opt is not None:
            opt.load_state_dict(saved_opt)
        optimizer_state = opt

    train_X = res["train_X"]
    train_Y_full = res["train_Y_full"]
    train_y = res.get("train_y", train_Y_full[..., [-1]])

    _rehydrate_dkl_predictor_from_full_evals(
        predictor=predictor,
        problem=problem,
        train_X=train_X,
        train_Y_full=train_Y_full,
    )

    cumulative_costs = res["cumulative_costs"]
    total_cost = float(cumulative_costs[-1]) if len(cumulative_costs) > 0 else 0.0

    return {
        "predictor": predictor,
        "optimizer_state": optimizer_state,
        "train_X": train_X,
        "train_Y_full": train_Y_full,
        "train_y": train_y,
        "test_losses": res["test_losses"],
        "best_test_loss": res["best_test_loss"],
        "node_test_losses_tf": res.get("node_test_losses_tf", {}),
        "weighted_node_test_losses_tf_all": res.get("weighted_node_test_losses_tf_all", {}),
        "weighted_node_test_losses_tf_intermediate": res.get("weighted_node_test_losses_tf_intermediate", {}),
        "cumulative_costs": cumulative_costs,
        "selected_inputs": res["selected_inputs"],
        "selected_scores": res["selected_scores"],
        "observed_sink_vals": res["observed_sink_vals"],
        "observed_full_node_vals": res["observed_full_node_vals"],
        "node_eval_counts": res["node_eval_counts"],
        "total_cost": total_cost,
    }


def score_candidate_by_fantasy_gain(
    *,
    predictor: torch.nn.Module,
    problem,
    backend_options: Dict,
    train_X: torch.Tensor,
    train_Y_full: torch.Tensor,
    base_x: torch.Tensor,
    selector_holdout_X: torch.Tensor,
    selector_holdout_y_sink: torch.Tensor,
    fantasy_train_steps: int = 20,
) -> float:
    """
    Fantasy gain = current_holdout_loss - fantasy_holdout_loss
    where the fantasy label is the backend predictive mean of all node outputs.
    """
    current_holdout_loss = compute_sink_test_loss(
        predictor=predictor,
        test_X=selector_holdout_X,
        test_y_sink=selector_holdout_y_sink,
    )

    mean_all, _ = predict_mean_var_all_nodes(
        predictor=predictor,
        X=base_x,
        options=backend_options,
    )  # [1, n_nodes]

    fantasy_predictor = deepcopy(predictor)

    fantasy_train_X = torch.cat((train_X, base_x), dim=0)
    fantasy_train_Y_full = torch.cat((train_Y_full, mean_all), dim=0)

    fantasy_options = dict(backend_options)
    fantasy_options["n_steps"] = int(fantasy_train_steps)

    _ = train_predictor_backend_from_full_evals(
        predictor=fantasy_predictor,
        problem=problem,
        base_X=fantasy_train_X,
        Y_full=fantasy_train_Y_full,
        options=fantasy_options,
        sink_idx=problem.n_nodes - 1,
        optimizer=None,
        verbose=False,
    )

    fantasy_holdout_loss = compute_sink_test_loss(
        predictor=fantasy_predictor,
        test_X=selector_holdout_X,
        test_y_sink=selector_holdout_y_sink,
    )

    return float(current_holdout_loss - fantasy_holdout_loss)


def select_next_input(
    algo: str,
    problem,
    predictor: torch.nn.Module,
    *,
    backend_options: Dict,
    train_X: Optional[torch.Tensor] = None,
    train_Y_full: Optional[torch.Tensor] = None,
    acquisition_mode: str = "uncertainty",
    selector_holdout_X: Optional[torch.Tensor] = None,
    selector_holdout_y_sink: Optional[torch.Tensor] = None,
    n_sobol: int = 256,
    fantasy_topk_candidates: int = 8,
    fantasy_train_steps: int = 20,
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

    if acquisition_mode not in ["uncertainty", "fantasy"]:
        raise ValueError(f"Unsupported acquisition_mode: {acquisition_mode}")

    Xcand = make_candidates(problem, n_sobol=n_sobol)
    _, var_sink = predict_sink_mean_var(
        predictor=predictor,
        X=Xcand,
        options=backend_options,
    )  # [N, 1]

    sink_scores = var_sink.view(-1)

    if acquisition_mode == "uncertainty":
        idx = torch.argmax(sink_scores)
        base_x = Xcand[idx: idx + 1]
        score = float(sink_scores[idx].item())

        logger.info(
            f"[selector-sink-only] mode=uncertainty | "
            f"max_sink_uncertainty={float(sink_scores.max().item()):.6f}"
        )

        return {
            "base_x": base_x,
            "score": score,
            "debug": {
                "mode": "sink_only_uq",
                "selected_candidate_idx": int(idx.item()),
                "max_sink_uncertainty": float(sink_scores.max().item()),
            },
        }

    if train_X is None or train_Y_full is None:
        raise ValueError("train_X and train_Y_full are required for fantasy mode.")
    if selector_holdout_X is None or selector_holdout_y_sink is None:
        raise ValueError(
            "selector_holdout_X and selector_holdout_y_sink are required for fantasy mode."
        )

    k = min(fantasy_topk_candidates, Xcand.shape[0])
    shortlist = torch.topk(sink_scores, k=k).indices.tolist()

    best_idx = None
    best_gain = None

    for idx_candidate in shortlist:
        base_x_candidate = Xcand[idx_candidate: idx_candidate + 1]

        fantasy_gain = score_candidate_by_fantasy_gain(
            predictor=predictor,
            problem=problem,
            backend_options=backend_options,
            train_X=train_X,
            train_Y_full=train_Y_full,
            base_x=base_x_candidate,
            selector_holdout_X=selector_holdout_X,
            selector_holdout_y_sink=selector_holdout_y_sink,
            fantasy_train_steps=fantasy_train_steps,
        )

        if best_gain is None or fantasy_gain > best_gain:
            best_gain = fantasy_gain
            best_idx = idx_candidate

    if best_idx is None:
        idx = torch.argmax(sink_scores)
        base_x = Xcand[idx: idx + 1]
        score = float(sink_scores[idx].item())
        return {
            "base_x": base_x,
            "score": score,
            "debug": {
                "mode": "sink_only_fantasy_fallback_to_uncertainty",
                "selected_candidate_idx": int(idx.item()),
                "max_sink_uncertainty": float(sink_scores.max().item()),
            },
        }

    base_x = Xcand[best_idx: best_idx + 1]
    score = float(best_gain)

    logger.info(
        f"[selector-sink-only] mode=fantasy | "
        f"best_fantasy_gain={best_gain:.6f} | "
        f"shortlist_topk={k}"
    )

    return {
        "base_x": base_x,
        "score": score,
        "debug": {
            "mode": "sink_only_fantasy",
            "selected_candidate_idx": int(best_idx),
            "best_fantasy_gain": float(best_gain),
            "shortlist_topk": int(k),
            "max_sink_uncertainty": float(sink_scores.max().item()),
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
    predictor_type: str = "mcd",
    hidden: int = 256,
    p_drop: float = 0.1,
    mc_samples: int = 30,
    dkl_feature_dim: int = 32,
    dkl_kernel: str = "rbf",
    acquisition_mode: str = "uncertainty",
    fantasy_topk_candidates: int = 8,
    fantasy_train_steps: int = 20,
    force_restart: bool = False,
) -> None:
    if algo not in ["Random", "NN_UQ"]:
        raise ValueError(f"Unsupported algo: {algo}")

    backend_options = _build_backend_options(
        problem=problem,
        predictor_type=predictor_type,
        hidden=hidden,
        p_drop=p_drop,
        mc_samples=mc_samples,
        dkl_feature_dim=dkl_feature_dim,
        dkl_kernel=dkl_kernel,
        train_steps=200,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-6,
        sink_loss_weight=1.0,
        aux_loss_weight=1.0,
    )

    results_dir = _make_results_dir(
        problem_name=problem_name,
        problem=problem,
        algo=algo,
        predictor_type=predictor_type,
        acquisition_mode=acquisition_mode,
    )

    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    predictor = build_predictor(problem, backend_options).to(torch.get_default_dtype())

    val_X, val_Y_full, val_y_sink = make_test_set(
        problem,
        n_test=256,
        seed=trial + 54321,
    )

    test_X, test_Y_full, test_y_sink = make_test_set(
        problem,
        n_test=512,
        seed=trial + 12345,
    )

    result_path = os.path.join(results_dir, f"trial_{trial}.pt")

    if os.path.exists(result_path) and not force_restart:
        logger.info(
            f"============================Resume Sink-Only Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Predictor: {predictor_type}\n"
            f"Trial: {trial}"
        )

        state = _resume_experiment(
            results_dir=results_dir,
            trial=trial,
            predictor=predictor,
            problem=problem,
            backend_options=backend_options,
        )
    else:
        logger.info(
            f"============================Start Sink-Only Experiment=================================\n"
            f"Experiment: {problem_name}_{'_'.join(str(x) for x in problem.node_costs)}\n"
            f"Algorithm: {algo}\n"
            f"Predictor: {predictor_type}\n"
            f"Trial: {trial}"
        )

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
        train_y = init_Y_full[..., [-1]].clone()

        optimizer_state = train_predictor_backend_from_full_evals(
            predictor=predictor,
            problem=problem,
            base_X=train_X,
            Y_full=train_Y_full,
            options=backend_options,
            sink_idx=problem.n_nodes - 1,
            optimizer=None,
            verbose=False,
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

        state = {
            "predictor": predictor,
            "optimizer_state": optimizer_state,
            "train_X": train_X,
            "train_Y_full": train_Y_full,
            "train_y": train_y,
            "test_losses": [test_loss],
            "best_test_loss": best_test_loss,
            "node_test_losses_tf": init_node_metric_history(initial_node_losses),
            "weighted_node_test_losses_tf_all": {
                -1: [compute_weighted_node_loss(initial_node_losses)]
            },
            "weighted_node_test_losses_tf_intermediate": {
                -1: [
                    compute_weighted_node_loss(
                        initial_node_losses,
                        exclude_nodes=[problem.n_nodes - 1],
                    )
                ]
            },
            "cumulative_costs": [0.0],
            "selected_inputs": [None],
            "selected_scores": [None],
            "observed_sink_vals": [None],
            "observed_full_node_vals": [None],
            "node_eval_counts": torch.zeros(problem.n_nodes, dtype=torch.long),
            "total_cost": 0.0,
        }

        _save_result_state(
            results_dir=results_dir,
            trial=trial,
            budget=budget,
            problem_name=problem_name,
            problem=problem,
            algo=algo,
            predictor_type=predictor_type,
            acquisition_mode=acquisition_mode,
            hidden=hidden,
            p_drop=p_drop,
            mc_samples=mc_samples,
            dkl_feature_dim=dkl_feature_dim,
            dkl_kernel=dkl_kernel,
            state=state,
        )

    predictor = state["predictor"]
    optimizer_state = state["optimizer_state"]

    total_cost = float(state["total_cost"])
    full_eval_cost = float(sum(problem.node_costs))

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
            backend_options=backend_options,
            train_X=state["train_X"],
            train_Y_full=state["train_Y_full"],
            acquisition_mode=acquisition_mode,
            selector_holdout_X=val_X,
            selector_holdout_y_sink=val_y_sink,
            n_sobol=256,
            fantasy_topk_candidates=fantasy_topk_candidates,
            fantasy_train_steps=fantasy_train_steps,
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
        state["total_cost"] = total_cost

        state["train_X"] = torch.cat((state["train_X"], base_x), dim=0)
        state["train_Y_full"] = torch.cat((state["train_Y_full"], y_full), dim=0)
        state["train_y"] = torch.cat((state["train_y"], y_sink), dim=0)

        optimizer_state = train_predictor_backend_from_full_evals(
            predictor=predictor,
            problem=problem,
            base_X=state["train_X"],
            Y_full=state["train_Y_full"],
            options=backend_options,
            sink_idx=problem.n_nodes - 1,
            optimizer=optimizer_state,
            verbose=False,
        )
        state["optimizer_state"] = optimizer_state

        test_loss = compute_sink_test_loss(
            predictor=predictor,
            test_X=test_X,
            test_y_sink=test_y_sink,
        )
        state["test_losses"].append(test_loss)
        state["best_test_loss"] = min(state["best_test_loss"], test_loss)

        node_losses = compute_teacher_forced_node_losses(
            predictor=predictor,
            test_X=test_X,
            test_Y_full=test_Y_full,
        )
        state["node_test_losses_tf"] = append_node_metric_history(
            state.get("node_test_losses_tf", {}),
            node_losses,
        )
        state["weighted_node_test_losses_tf_all"] = append_node_metric_history(
            state.get("weighted_node_test_losses_tf_all", {}),
            {-1: compute_weighted_node_loss(node_losses)},
        )
        state["weighted_node_test_losses_tf_intermediate"] = append_node_metric_history(
            state.get("weighted_node_test_losses_tf_intermediate", {}),
            {
                -1: compute_weighted_node_loss(
                    node_losses,
                    exclude_nodes=[problem.n_nodes - 1],
                )
            },
        )

        tf_str = ", ".join([f"node{j}={v:.4f}" for j, v in sorted(node_losses.items())])
        logger.info(f"Teacher-forced node losses: {tf_str}")
        logger.info(f"Sink test loss: {test_loss:.6f} (best {state['best_test_loss']:.6f})")
        logger.info(f"total cost used: {total_cost}")
        logger.info("==========================================================================")

        state["cumulative_costs"].append(total_cost)
        state["selected_inputs"].append(base_x.detach().cpu())
        state["selected_scores"].append(score)
        state["observed_sink_vals"].append(y_sink.detach().cpu())
        state["observed_full_node_vals"].append(y_full.detach().cpu())
        state["node_eval_counts"] = state["node_eval_counts"] + 1

        _save_result_state(
            results_dir=results_dir,
            trial=trial,
            budget=budget,
            problem_name=problem_name,
            problem=problem,
            algo=algo,
            predictor_type=predictor_type,
            acquisition_mode=acquisition_mode,
            hidden=hidden,
            p_drop=p_drop,
            mc_samples=mc_samples,
            dkl_feature_dim=dkl_feature_dim,
            dkl_kernel=dkl_kernel,
            state=state,
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
    parser.add_argument("--force_restart", action="store_true")

    parser.add_argument(
        "--predictor_type",
        type=str,
        default="mcd",
        choices=["mcd", "dkl"],
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--p_drop", type=float, default=0.1)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--dkl_feature_dim", type=int, default=32)
    parser.add_argument(
        "--dkl_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "matern"],
    )

    parser.add_argument(
        "--acquisition_mode",
        type=str,
        default="uncertainty",
        choices=["uncertainty", "fantasy"],
    )
    parser.add_argument("--fantasy_topk_candidates", type=int, default=8)
    parser.add_argument("--fantasy_train_steps", type=int, default=20)
    return parser.parse_args()


def main(
    trial: int,
    algo: str,
    costs: str,
    budget: int,
    noisy: bool = False,
    predictor_type: str = "mcd",
    hidden: int = 256,
    p_drop: float = 0.1,
    mc_samples: int = 30,
    dkl_feature_dim: int = 32,
    dkl_kernel: str = "rbf",
    acquisition_mode: str = "uncertainty",
    fantasy_topk_candidates: int = 8,
    fantasy_train_steps: int = 20,
    force_restart: bool = False,
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
        predictor_type=predictor_type,
        hidden=hidden,
        p_drop=p_drop,
        mc_samples=mc_samples,
        dkl_feature_dim=dkl_feature_dim,
        dkl_kernel=dkl_kernel,
        acquisition_mode=acquisition_mode,
        fantasy_topk_candidates=fantasy_topk_candidates,
        fantasy_train_steps=fantasy_train_steps,
        force_restart=force_restart,
    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))