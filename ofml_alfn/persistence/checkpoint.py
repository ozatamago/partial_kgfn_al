#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional, Union

import torch


def _nn_ckpt_dir(results_dir: str) -> str:
    """
    Keep NN checkpoints next to results so they share the same experiment key.

    Example:
        results_dir = ./results/foo/bar/
        ckpt_dir    = ./results/foo/bar/checkpoints/
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _pack_optimizer_blob(nn_optimizer: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize optimizer-like state into a serializable blob.

    Supported cases
    ---------------
    1. torch optimizer
       -> {"kind": "torch_optimizer", "state_dict": ...}

    2. dict-like optimizer state
       -> {"kind": "dict_state", "state": ...}

    3. None
       -> None
    """
    if nn_optimizer is None:
        return None

    if isinstance(nn_optimizer, torch.optim.Optimizer):
        return {
            "kind": "torch_optimizer",
            "state_dict": nn_optimizer.state_dict(),
        }

    if isinstance(nn_optimizer, dict):
        return {
            "kind": "dict_state",
            "state": nn_optimizer,
        }

    raise TypeError(
        "Unsupported optimizer type for checkpointing. "
        f"Expected torch.optim.Optimizer, dict, or None; got {type(nn_optimizer)}"
    )


def _restore_optimizer_blob(
    *,
    blob: Optional[Dict[str, Any]],
    nn_optimizer: Optional[Any],
) -> Optional[Any]:
    """
    Restore optimizer-like state from a checkpoint blob.

    Behavior
    --------
    - If blob is None:
        return nn_optimizer unchanged.
    - If nn_optimizer is a torch optimizer and blob.kind == "torch_optimizer":
        load_state_dict into nn_optimizer and return it.
    - If blob.kind == "dict_state":
        return the saved dict state. If nn_optimizer is already a dict, it is
        updated in-place when possible and also returned.
    - If nn_optimizer is None:
        simply return the restored object implied by the blob.
    """
    if blob is None:
        return nn_optimizer

    kind = blob.get("kind", None)

    if kind == "torch_optimizer":
        state_dict = blob["state_dict"]

        if nn_optimizer is None:
            return state_dict

        if not isinstance(nn_optimizer, torch.optim.Optimizer):
            raise TypeError(
                "Checkpoint contains a torch optimizer state, but nn_optimizer "
                f"is of type {type(nn_optimizer)}"
            )

        nn_optimizer.load_state_dict(state_dict)
        return nn_optimizer

    if kind == "dict_state":
        state = blob["state"]

        if nn_optimizer is None:
            return state

        if isinstance(nn_optimizer, dict):
            nn_optimizer.clear()
            nn_optimizer.update(state)
            return nn_optimizer

        # Caller passed a non-dict object but the checkpoint stores dict-state.
        # Return the saved dict so the caller can adopt it.
        return state

    raise ValueError(f"Unknown optimizer blob kind: {kind}")


def save_nn_checkpoint(
    *,
    results_dir: str,
    trial: int,
    step: int,
    predictor: torch.nn.Module,
    nn_optimizer: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a resumable checkpoint (model + optimizer-like state + metadata).

    Notes
    -----
    - Supports both standard torch optimizers and dict-like optimizer states.
    - Keeps backward compatibility by also writing optimizer_state_dict when the
      optimizer is a real torch optimizer.
    """
    ckpt_dir = _nn_ckpt_dir(results_dir)
    ckpt_path = os.path.join(ckpt_dir, f"trial_{trial}_step_{step}.pth")

    ckpt: Dict[str, Any] = {
        "trial": trial,
        "step": step,
        "predictor_state_dict": predictor.state_dict(),
    }

    optimizer_blob = _pack_optimizer_blob(nn_optimizer)
    if optimizer_blob is not None:
        ckpt["optimizer_blob"] = optimizer_blob

        # Backward-compatible legacy key for ordinary torch optimizers.
        if optimizer_blob["kind"] == "torch_optimizer":
            ckpt["optimizer_state_dict"] = optimizer_blob["state_dict"]

    if extra:
        ckpt.update(extra)

    torch.save(ckpt, ckpt_path)
    return ckpt_path


def load_latest_nn_checkpoint(
    *,
    results_dir: str,
    trial: int,
    predictor: torch.nn.Module,
    nn_optimizer: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Optional[Dict[str, Any]]:
    """
    Load the latest checkpoint for this trial if it exists.

    Restores
    --------
    - predictor_state_dict always
    - optimizer-like state if nn_optimizer is provided, or returns the restored
      optimizer-like object in ckpt["restored_optimizer_state"]

    Returns
    -------
    The loaded checkpoint dict, or None if nothing exists.
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    prefix = f"trial_{trial}_step_"
    candidates = []

    for fn in os.listdir(ckpt_dir):
        if fn.startswith(prefix) and fn.endswith(".pth"):
            try:
                step = int(fn[len(prefix): -len(".pth")])
                candidates.append((step, fn))
            except ValueError:
                pass

    if not candidates:
        return None

    step, fn = max(candidates, key=lambda x: x[0])
    ckpt_path = os.path.join(ckpt_dir, fn)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    predictor.load_state_dict(ckpt["predictor_state_dict"])

    # Prefer the new generic blob format.
    if "optimizer_blob" in ckpt:
        restored = _restore_optimizer_blob(
            blob=ckpt["optimizer_blob"],
            nn_optimizer=nn_optimizer,
        )
        ckpt["restored_optimizer_state"] = restored

    # Backward compatibility for older checkpoints that only stored
    # optimizer_state_dict for a plain torch optimizer.
    elif nn_optimizer is not None and "optimizer_state_dict" in ckpt:
        if not isinstance(nn_optimizer, torch.optim.Optimizer):
            raise TypeError(
                "Checkpoint contains optimizer_state_dict, but nn_optimizer "
                f"is of type {type(nn_optimizer)}"
            )
        nn_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ckpt["restored_optimizer_state"] = nn_optimizer

    return ckpt