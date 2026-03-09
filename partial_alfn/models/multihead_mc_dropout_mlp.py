#!/usr/bin/env python3

import torch
import torch.nn as nn


class MultiHeadMCDropoutMLP(nn.Module):
    """
    Multi-head MLP with MC-dropout-ready shared trunk.

    Design goal:
      - preserve current sink-only workflow via forward() / forward_sink()
      - enable future partial-observation learning via forward_all() / forward_node()

    Behavior:
      - forward(x) returns sink prediction [N, 1] for backward compatibility
      - forward_all(x) returns all node predictions [N, n_nodes]
      - forward_node(x, j) returns node-j prediction [N, 1]
    """

    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        hidden: int = 256,
        p_drop: float = 0.1,
        sink_idx: int = None,
    ):
        super().__init__()
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {n_nodes}")

        self.in_dim = in_dim
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.p_drop = p_drop
        self.sink_idx = (n_nodes - 1) if sink_idx is None else sink_idx

        if not (0 <= self.sink_idx < n_nodes):
            raise ValueError(f"sink_idx must be in [0, {n_nodes-1}], got {self.sink_idx}")

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_nodes)]
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            y_all: [N, n_nodes]
        """
        h = self._features(x)
        ys = [head(h) for head in self.heads]   # list of [N, 1]
        return torch.cat(ys, dim=-1)            # [N, n_nodes]

    def forward_node(self, x: torch.Tensor, node_idx: int) -> torch.Tensor:
        """
        Returns:
            y_node: [N, 1]
        """
        if not (0 <= node_idx < self.n_nodes):
            raise IndexError(f"node_idx must be in [0, {self.n_nodes-1}], got {node_idx}")
        h = self._features(x)
        return self.heads[node_idx](h)

    def forward_sink(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            y_sink: [N, 1]
        """
        return self.forward_node(x, self.sink_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible default:
        return sink prediction so existing sink-only code keeps working.
        """
        return self.forward_sink(x)