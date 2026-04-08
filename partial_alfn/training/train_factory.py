#!/usr/bin/env python3

from typing import Any, Dict, Optional, Sequence

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


def train_predictor_partial_backend(
    predictor: nn.Module,
    train_X_nodes: Sequence[torch.Tensor],
    train_Y_nodes: Sequence[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
    *,
    sink_idx: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
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
        For MCD backend, forwarded to the existing trainer.
        For DKL backend, currently ignored by the DKL trainer.
    verbose:
        Whether to print training diagnostics.

    Returns
    -------
    optimizer_or_state:
        - For MCD: typically returns an optimizer
        - For DKL: currently returns None
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