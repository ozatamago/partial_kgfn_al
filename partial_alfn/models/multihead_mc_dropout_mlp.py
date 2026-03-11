#!/usr/bin/env python3

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class _NodeConditionalMCDropoutMLP(nn.Module):
    """
    One conditional model for one node:
        x_node -> z_node
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        p_drop: float = 0.1,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")

        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.p_drop = float(p_drop)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.in_dim}], got {tuple(x.shape)}"
            )
        return self.net(x)


class MultiHeadMCDropoutMLP(nn.Module):
    """
    Node-wise conditional predictor for a function network.

    Important change from the old version:
      - this class no longer assumes one shared trunk from base_x to all nodes
      - each node has its own conditional model with its own input dimension

    Example for a 2-node chain:
        node 0: x_ext (dim 3) -> z0
        node 1: z0    (dim 1) -> z1

    Parameters
    ----------
    external_input_dim:
        Dimension of the original external input base_x.
    node_input_dims:
        Per-node input dimensions in node-input space.
        Example: [3, 1] for FreeSolv3.
    parent_nodes:
        parent_nodes[j] gives the parent node indices of node j.
    active_input_indices:
        active_input_indices[j] gives the indices of base_x used directly by node j.
        For example, if node 0 uses all 3 external features, and node 1 uses only parent node 0,
        then active_input_indices might look like [[0,1,2], []].
    sink_idx:
        Sink node index.
    """

    def __init__(
        self,
        *,
        external_input_dim: int,
        node_input_dims: Sequence[int],
        parent_nodes: Optional[Sequence[Sequence[int]]] = None,
        active_input_indices: Optional[Sequence[Sequence[int]]] = None,
        hidden: int = 256,
        p_drop: float = 0.1,
        sink_idx: Optional[int] = None,
    ):
        super().__init__()

        if external_input_dim <= 0:
            raise ValueError(
                f"external_input_dim must be positive, got {external_input_dim}"
            )

        self.external_input_dim = int(external_input_dim)
        self.node_input_dims = [int(d) for d in node_input_dims]
        self.n_nodes = len(self.node_input_dims)
        if self.n_nodes <= 0:
            raise ValueError("node_input_dims must contain at least one node.")

        self.hidden = int(hidden)
        self.p_drop = float(p_drop)
        self.sink_idx = (self.n_nodes - 1) if sink_idx is None else int(sink_idx)

        if not (0 <= self.sink_idx < self.n_nodes):
            raise ValueError(
                f"sink_idx must be in [0, {self.n_nodes - 1}], got {self.sink_idx}"
            )

        self.parent_nodes = (
            [list(p) for p in parent_nodes]
            if parent_nodes is not None
            else None
        )
        self.active_input_indices = (
            [list(a) for a in active_input_indices]
            if active_input_indices is not None
            else None
        )

        if self.parent_nodes is not None and len(self.parent_nodes) != self.n_nodes:
            raise ValueError(
                "parent_nodes must have length n_nodes, "
                f"got {len(self.parent_nodes)} and {self.n_nodes}"
            )

        if (
            self.active_input_indices is not None
            and len(self.active_input_indices) != self.n_nodes
        ):
            raise ValueError(
                "active_input_indices must have length n_nodes, "
                f"got {len(self.active_input_indices)} and {self.n_nodes}"
            )

        self.node_models = nn.ModuleList(
            [
                _NodeConditionalMCDropoutMLP(
                    in_dim=self.node_input_dims[j],
                    hidden=self.hidden,
                    p_drop=self.p_drop,
                )
                for j in range(self.n_nodes)
            ]
        )

        # Optional consistency check with provided graph metadata.
        if self.parent_nodes is not None and self.active_input_indices is not None:
            for j in range(self.n_nodes):
                implied_dim = len(self.parent_nodes[j]) + len(self.active_input_indices[j])
                if implied_dim != self.node_input_dims[j]:
                    raise ValueError(
                        f"Node {j}: node_input_dims[{j}]={self.node_input_dims[j]} "
                        f"but implied dim from graph is {implied_dim}"
                    )

    def forward_node(self, x_node: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Direct conditional prediction in node-input space:
            x_node -> z_node

        x_node must already be in the input space of node_idx.
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(
                f"node_idx must be in [0, {self.n_nodes - 1}], got {node_idx}"
            )
        return self.node_models[node_idx](x_node)

    def make_node_input_from_base(
        self,
        *,
        base_x: torch.Tensor,
        node_idx: int,
        parent_outputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Construct the input of node_idx from:
          - sampled / mean parent outputs
          - directly active external inputs from base_x

        parent_outputs[p] is expected to be shape [N, 1] for parent node p.
        """
        if self.parent_nodes is None or self.active_input_indices is None:
            raise ValueError(
                "Graph metadata is required. Please provide parent_nodes and "
                "active_input_indices when constructing the model."
            )

        if base_x.ndim != 2 or base_x.shape[1] != self.external_input_dim:
            raise ValueError(
                f"Expected base_x of shape [N, {self.external_input_dim}], "
                f"got {tuple(base_x.shape)}"
            )

        parts: List[torch.Tensor] = []

        parents = self.parent_nodes[node_idx]
        for p in parents:
            yp = parent_outputs[p]
            if yp.ndim != 2 or yp.shape[1] != 1:
                raise ValueError(
                    f"Parent output for node {p} must be [N,1], got {tuple(yp.shape)}"
                )
            parts.append(yp)

        active_idx = self.active_input_indices[node_idx]
        if len(active_idx) > 0:
            parts.append(base_x[:, active_idx])

        if len(parts) == 0:
            raise ValueError(
                f"Node {node_idx} has neither parents nor active external inputs."
            )

        x_node = torch.cat(parts, dim=-1)

        expected_dim = self.node_input_dims[node_idx]
        if x_node.shape[1] != expected_dim:
            raise ValueError(
                f"Constructed input for node {node_idx} has dim {x_node.shape[1]}, "
                f"expected {expected_dim}"
            )

        return x_node

    def rollout_means_from_base(self, base_x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic mean-style rollout through the function network using the
        node models in topological order.

        Assumes node indices are in topological order.
        Returns:
            [N, n_nodes]
        """
        if self.parent_nodes is None or self.active_input_indices is None:
            raise ValueError(
                "rollout_means_from_base requires parent_nodes and active_input_indices."
            )

        node_outputs: List[Optional[torch.Tensor]] = [None] * self.n_nodes
        collected = []

        for j in range(self.n_nodes):
            xj = self.make_node_input_from_base(
                base_x=base_x,
                node_idx=j,
                parent_outputs=node_outputs,
            )
            yj = self.forward_node(xj, j)
            node_outputs[j] = yj
            collected.append(yj)

        return torch.cat(collected, dim=-1)

    def forward_all(self, base_x: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible convenience:
        returns all node mean predictions from external input base_x.
        """
        return self.rollout_means_from_base(base_x)

    def forward_sink_from_base(self, base_x: torch.Tensor) -> torch.Tensor:
        """
        Returns sink prediction from external input base_x.
        """
        return self.rollout_means_from_base(base_x)[:, [self.sink_idx]]

    def forward_sink(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper.

        If x is in external input space, compute sink from base_x.
        If x is already in sink-input space, compute sink conditional model directly.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got {tuple(x.shape)}")

        if x.shape[1] == self.external_input_dim:
            return self.forward_sink_from_base(x)

        if x.shape[1] == self.node_input_dims[self.sink_idx]:
            return self.forward_node(x, self.sink_idx)

        raise ValueError(
            f"Input dim {x.shape[1]} matches neither external_input_dim="
            f"{self.external_input_dim} nor sink input dim={self.node_input_dims[self.sink_idx]}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default behavior:
        return sink prediction.
        """
        return self.forward_sink(x)