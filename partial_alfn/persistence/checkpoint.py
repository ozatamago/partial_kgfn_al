#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional, Union

import torch


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
    Load the latest checkpoint for this trial if it exists.
    Restores model (+ optimizer if provided).

    Returns:
        The loaded checkpoint dict, or None if nothing found.
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

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