#!/usr/bin/env python3

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


def init_partial_buffers(
    *,
    n_nodes: int,
    x_dim: int,
    dtype: torch.dtype = torch.double,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Create node-wise partial supervision buffers.

    Each node j stores supervised pairs:
        X_by_node[j] : [N_j, x_dim]
        y_by_node[j] : [N_j, 1]

    These buffers are intended for future multi-head training where
    node j is trained to predict z_j from the external input x.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    if x_dim <= 0:
        raise ValueError(f"x_dim must be positive, got {x_dim}")

    if device is None:
        device = torch.device("cpu")

    x_by_node = [
        torch.empty((0, x_dim), dtype=dtype, device=device) for _ in range(n_nodes)
    ]
    y_by_node = [
        torch.empty((0, 1), dtype=dtype, device=device) for _ in range(n_nodes)
    ]
    counts = torch.zeros(n_nodes, dtype=torch.long, device=device)

    return {
        "x_by_node": x_by_node,
        "y_by_node": y_by_node,
        "counts": counts,
    }


def append_partial_buffer(
    *,
    buffers: Dict,
    node_idx: int,
    x: Tensor,
    y: Tensor,
) -> None:
    """
    Append one or more supervised samples to a single node buffer.

    Args:
        buffers:
            output of init_partial_buffers(...)
        node_idx:
            target node index
        x:
            shape [N, d]
        y:
            shape [N, 1]
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, d], got shape {tuple(x.shape)}")
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"y must be [N, 1], got shape {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y must have the same batch size, got {x.shape[0]} and {y.shape[0]}"
        )

    buffers["x_by_node"][node_idx] = torch.cat((buffers["x_by_node"][node_idx], x), dim=0)
    buffers["y_by_node"][node_idx] = torch.cat((buffers["y_by_node"][node_idx], y), dim=0)
    buffers["counts"][node_idx] += x.shape[0]


def append_partial_group(
    *,
    buffers: Dict,
    node_indices: Sequence[int],
    x: Tensor,
    y_group: Tensor,
) -> None:
    """
    Append a group partial observation.

    Args:
        node_indices:
            e.g. [0, 2]
        x:
            shape [N, d]
        y_group:
            shape [N, len(node_indices)], aligned with node_indices
    """
    if y_group.ndim != 2:
        raise ValueError(
            f"y_group must be 2D [N, len(node_indices)], got shape {tuple(y_group.shape)}"
        )
    if y_group.shape[0] != x.shape[0]:
        raise ValueError(
            f"x and y_group batch sizes must match, got {x.shape[0]} and {y_group.shape[0]}"
        )
    if y_group.shape[1] != len(node_indices):
        raise ValueError(
            f"y_group second dimension must equal len(node_indices), "
            f"got {y_group.shape[1]} and {len(node_indices)}"
        )

    for col, node_idx in enumerate(node_indices):
        append_partial_buffer(
            buffers=buffers,
            node_idx=node_idx,
            x=x,
            y=y_group[:, [col]],
        )


def append_full_network_as_partial(
    *,
    buffers: Dict,
    x: Tensor,
    y_full: Tensor,
) -> None:
    """
    Treat a full-network observation as node-wise supervision for all nodes.

    Args:
        x:
            shape [N, d]
        y_full:
            shape [N, n_nodes]
    """
    if y_full.ndim != 2:
        raise ValueError(f"y_full must be [N, n_nodes], got shape {tuple(y_full.shape)}")

    n_nodes = len(buffers["x_by_node"])
    if y_full.shape[1] != n_nodes:
        raise ValueError(
            f"y_full second dimension must equal number of nodes, got {y_full.shape[1]} and {n_nodes}"
        )

    for node_idx in range(n_nodes):
        append_partial_buffer(
            buffers=buffers,
            node_idx=node_idx,
            x=x,
            y=y_full[:, [node_idx]],
        )


def get_partial_dataset_for_node(
    *,
    buffers: Dict,
    node_idx: int,
) -> Tuple[Tensor, Tensor]:
    """
    Return (X_j, y_j) for one node.
    """
    return buffers["x_by_node"][node_idx], buffers["y_by_node"][node_idx]


def get_nonempty_node_indices(buffers: Dict) -> List[int]:
    """
    Return nodes that currently have at least one supervised sample.
    """
    return [j for j, xj in enumerate(buffers["x_by_node"]) if xj.shape[0] > 0]