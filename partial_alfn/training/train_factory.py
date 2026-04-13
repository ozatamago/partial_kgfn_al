#!/usr/bin/env python3

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from partial_alfn.training.train_partial import train_predictor_partial
from partial_alfn.training.train_dkl_partial import train_predictor_partial_dkl


def _to_options_dict(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {} if options is None else dict(options)


def _get_predictor_type(predictor: nn.Module) -> str:
    predictor_type = getattr(predictor, "predictor_type", None)
    if predictor_type is None:
        raise AttributeError(
            "predictor must expose predictor_type, e.g. 'mcd' or 'dkl'"
        )

    predictor_type = str(predictor_type).lower()
    if predictor_type not in ["mcd", "dkl"]:
        raise ValueError(
            f"Unsupported predictor_type: {predictor_type}. "
            "Expected 'mcd' or 'dkl'."
        )
    return predictor_type


def _validate_problem_for_full_eval_conversion(problem: Any) -> None:
    required_attrs = ["n_nodes", "parent_nodes", "active_input_indices"]
    for name in required_attrs:
        if not hasattr(problem, name):
            raise AttributeError(
                f"problem must expose `{name}` for full-eval -> nodewise conversion."
            )

    n_nodes = int(problem.n_nodes)

    if len(problem.parent_nodes) != n_nodes:
        raise ValueError(
            f"len(problem.parent_nodes) must equal problem.n_nodes={n_nodes}, "
            f"got {len(problem.parent_nodes)}"
        )

    if len(problem.active_input_indices) != n_nodes:
        raise ValueError(
            f"len(problem.active_input_indices) must equal problem.n_nodes={n_nodes}, "
            f"got {len(problem.active_input_indices)}"
        )


def build_nodewise_datasets_from_full_evals(
    *,
    problem: Any,
    base_X: torch.Tensor,
    Y_full: torch.Tensor,
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """
    Convert full evaluations (base_X -> Y_full) into node-wise conditional datasets.

    For node j:
      X_j = concat(parent node outputs, active external inputs)
      Y_j = observed node j output
    """
    _validate_problem_for_full_eval_conversion(problem)

    if not torch.is_tensor(base_X):
        raise TypeError(f"base_X must be a tensor, got {type(base_X)}")
    if not torch.is_tensor(Y_full):
        raise TypeError(f"Y_full must be a tensor, got {type(Y_full)}")

    if base_X.ndim != 2:
        raise ValueError(f"base_X must be 2D, got shape {tuple(base_X.shape)}")
    if Y_full.ndim != 2:
        raise ValueError(f"Y_full must be 2D, got shape {tuple(Y_full.shape)}")

    if base_X.shape[0] != Y_full.shape[0]:
        raise ValueError(
            f"base_X and Y_full must have the same number of rows, got "
            f"{base_X.shape[0]} and {Y_full.shape[0]}"
        )

    n_nodes = int(problem.n_nodes)
    if Y_full.shape[1] != n_nodes:
        raise ValueError(
            f"Y_full second dimension must equal problem.n_nodes={n_nodes}, "
            f"got {Y_full.shape[1]}"
        )

    train_X_nodes = []
    train_Y_nodes = []

    for j in range(n_nodes):
        parts = []

        for p in problem.parent_nodes[j]:
            parts.append(Y_full[:, [p]])

        active_idx = list(problem.active_input_indices[j])
        if len(active_idx) > 0:
            parts.append(base_X[:, active_idx])

        if len(parts) == 0:
            raise ValueError(
                f"Node {j} has neither parent outputs nor active external inputs."
            )

        xj = torch.cat(parts, dim=-1)
        yj = Y_full[:, [j]]

        train_X_nodes.append(xj)
        train_Y_nodes.append(yj)

    return train_X_nodes, train_Y_nodes


def train_predictor_partial_backend(
    predictor: nn.Module,
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
    *,
    sink_idx: Optional[int] = None,
    optimizer: Optional[Any] = None,
    verbose: bool = False,
):
    """
    Backend-dispatched partial trainer.

    Parameters
    ----------
    predictor:
        Predictor backend exposing predictor_type = "mcd" or "dkl".
    train_X_nodes:
        Node-wise input datasets. train_X_nodes[j] has shape [Nj, dj].
    train_Y_nodes:
        Node-wise output datasets. train_Y_nodes[j] has shape [Nj, 1] or [Nj].
    options:
        Training options dictionary. Common keys may include:
            n_steps
            nn_train_steps
            batch_size
            nn_batch_size
            lr
            nn_lr
            weight_decay
            nodes_per_step
            aux_loss_weight
            sink_loss_weight
    sink_idx:
        Optional explicit sink index. If None, use predictor.sink_idx when needed.
    optimizer:
        - For MCD backend: torch.optim.Optimizer
        - For DKL backend: optional dict containing per-node optimizer states
    verbose:
        Whether to print training diagnostics.

    Returns
    -------
    optimizer_or_state:
        - For MCD: typically returns a torch optimizer
        - For DKL: typically returns a dict containing per-node optimizer states
    """
    opts = _to_options_dict(options)
    predictor_type = _get_predictor_type(predictor)

    n_steps = int(opts.get("n_steps", opts.get("nn_train_steps", 200)))
    batch_size = int(opts.get("batch_size", opts.get("nn_batch_size", 64)))
    lr = float(opts.get("lr", opts.get("nn_lr", 1e-3)))
    weight_decay = float(opts.get("weight_decay", 1e-6))
    nodes_per_step = opts.get("nodes_per_step", None)
    aux_loss_weight = float(opts.get("aux_loss_weight", 1.0))
    sink_loss_weight = float(opts.get("sink_loss_weight", 1.0))

    if predictor_type == "mcd":
        return train_predictor_partial(
            predictor=predictor,
            train_X_nodes=train_X_nodes,
            train_Y_nodes=train_Y_nodes,
            sink_idx=sink_idx,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            nodes_per_step=nodes_per_step,
            aux_loss_weight=aux_loss_weight,
            sink_loss_weight=sink_loss_weight,
            optimizer=optimizer,
        )

    if predictor_type == "dkl":
        return train_predictor_partial_dkl(
            predictor=predictor,
            train_X_nodes=train_X_nodes,
            train_Y_nodes=train_Y_nodes,
            sink_idx=sink_idx,
            n_steps=n_steps,
            lr=lr,
            weight_decay=weight_decay,
            nodes_per_step=nodes_per_step,
            aux_loss_weight=aux_loss_weight,
            sink_loss_weight=sink_loss_weight,
            optimizer=optimizer,
            verbose=verbose,
        )

    raise RuntimeError("Unreachable predictor_type branch.")


def train_predictor_backend_from_full_evals(
    predictor: nn.Module,
    *,
    problem: Any,
    base_X: torch.Tensor,
    Y_full: torch.Tensor,
    options: Optional[Dict[str, Any]] = None,
    sink_idx: Optional[int] = None,
    optimizer: Optional[Any] = None,
    verbose: bool = False,
):
    """
    Convenience adapter for experiments that store full protocol evaluations.

    This converts:
        base_X  : [N, d]
        Y_full  : [N, n_nodes]
    into node-wise conditional datasets, then calls
    train_predictor_partial_backend(...).
    """
    train_X_nodes, train_Y_nodes = build_nodewise_datasets_from_full_evals(
        problem=problem,
        base_X=base_X,
        Y_full=Y_full,
    )

    return train_predictor_partial_backend(
        predictor=predictor,
        train_X_nodes=train_X_nodes,
        train_Y_nodes=train_Y_nodes,
        options=options,
        sink_idx=sink_idx,
        optimizer=optimizer,
        verbose=verbose,
    )