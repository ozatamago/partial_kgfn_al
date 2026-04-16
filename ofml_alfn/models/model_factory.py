#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from ofml_alfn.models.multihead_mc_dropout_mlp import MultiHeadMCDropoutMLP
from ofml_alfn.models.nodewise_dkl import MultiHeadNodewiseDKL


def _to_options_dict(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {} if options is None else dict(options)


def _require_problem_attr(problem: Any, name: str) -> Any:
    if not hasattr(problem, name):
        raise AttributeError(f"problem must expose `{name}`")
    return getattr(problem, name)


def _infer_node_input_dims(
    parent_nodes: Sequence[Sequence[int]],
    active_input_indices: Sequence[Sequence[int]],
) -> list[int]:
    if len(parent_nodes) != len(active_input_indices):
        raise ValueError(
            "parent_nodes and active_input_indices must have the same length, "
            f"got {len(parent_nodes)} and {len(active_input_indices)}"
        )

    node_input_dims: list[int] = []
    for j in range(len(parent_nodes)):
        d_j = len(parent_nodes[j]) + len(active_input_indices[j])
        if d_j <= 0:
            raise ValueError(
                f"Node {j} has non-positive input dimension: {d_j}. "
                "Each node must depend on at least one parent output or one "
                "active external input."
            )
        node_input_dims.append(int(d_j))
    return node_input_dims


def _get_first_present(
    opts: Dict[str, Any],
    keys: Sequence[str],
    default: Any,
) -> Any:
    for k in keys:
        if k in opts and opts[k] is not None:
            return opts[k]
    return default


def build_predictor(
    problem: Any,
    options: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Build a predictor backend for a function network.

    Supported predictor types
    -------------------------
    - "mcd": MultiHeadMCDropoutMLP
    - "dkl": MultiHeadNodewiseDKL

    Expected problem attributes
    ---------------------------
    - dim
    - n_nodes
    - parent_nodes
    - active_input_indices

    Relevant options
    ----------------
    Common:
        predictor_type: str = "mcd"
        hidden: int = 256
        sink_idx: Optional[int] = None
        dtype: torch.dtype = torch.get_default_dtype()
        device: Optional[torch.device | str] = None

    MCD-specific:
        p_drop: float = 0.1

    DKL-specific:
        dkl_hidden or hidden: int = 256
        dkl_feature_dim or feature_dim: int = 32
        dkl_kernel or kernel_type: str = "rbf"
        dkl_inference: str = "exact"

    Returns
    -------
    predictor: nn.Module
    """
    opts = _to_options_dict(options)

    predictor_type = str(opts.get("predictor_type", "mcd")).lower()

    external_input_dim = int(_require_problem_attr(problem, "dim"))
    n_nodes = int(_require_problem_attr(problem, "n_nodes"))
    parent_nodes = _require_problem_attr(problem, "parent_nodes")
    active_input_indices = _require_problem_attr(problem, "active_input_indices")

    if len(parent_nodes) != n_nodes:
        raise ValueError(
            f"len(problem.parent_nodes) must equal problem.n_nodes, got "
            f"{len(parent_nodes)} and {n_nodes}"
        )
    if len(active_input_indices) != n_nodes:
        raise ValueError(
            f"len(problem.active_input_indices) must equal problem.n_nodes, got "
            f"{len(active_input_indices)} and {n_nodes}"
        )

    node_input_dims = _infer_node_input_dims(
        parent_nodes=parent_nodes,
        active_input_indices=active_input_indices,
    )

    sink_idx = opts.get("sink_idx", None)
    if sink_idx is None:
        sink_idx = n_nodes - 1
    sink_idx = int(sink_idx)

    dtype = opts.get("dtype", torch.get_default_dtype())
    device = opts.get("device", None)

    if predictor_type == "mcd":
        hidden = int(_get_first_present(opts, ["hidden"], 256))
        p_drop = float(_get_first_present(opts, ["p_drop", "dropout"], 0.1))

        predictor = MultiHeadMCDropoutMLP(
            external_input_dim=external_input_dim,
            node_input_dims=node_input_dims,
            parent_nodes=parent_nodes,
            active_input_indices=active_input_indices,
            hidden=hidden,
            p_drop=p_drop,
            sink_idx=sink_idx,
        )

    elif predictor_type == "dkl":
        dkl_inference = str(
            _get_first_present(opts, ["dkl_inference"], "exact")
        ).lower()
        if dkl_inference != "exact":
            raise NotImplementedError(
                f"dkl_inference={dkl_inference!r} is not supported by "
                "MultiHeadNodewiseDKL in the current ofml_alfn implementation. "
                "Use 'exact'."
            )

        hidden = int(_get_first_present(opts, ["dkl_hidden", "hidden"], 256))
        feature_dim = int(
            _get_first_present(opts, ["dkl_feature_dim", "feature_dim"], 32)
        )
        kernel_type = str(
            _get_first_present(opts, ["dkl_kernel", "kernel_type"], "rbf")
        ).lower()

        predictor = MultiHeadNodewiseDKL(
            external_input_dim=external_input_dim,
            node_input_dims=node_input_dims,
            parent_nodes=parent_nodes,
            active_input_indices=active_input_indices,
            hidden=hidden,
            feature_dim=feature_dim,
            kernel_type=kernel_type,
            sink_idx=sink_idx,
        )

    else:
        raise ValueError(
            f"Unsupported predictor_type: {predictor_type}. "
            "Use 'mcd' or 'dkl'."
        )

    predictor = predictor.to(dtype=dtype)
    if device is not None:
        predictor = predictor.to(device=device)

    return predictor


__all__ = ["build_predictor"]