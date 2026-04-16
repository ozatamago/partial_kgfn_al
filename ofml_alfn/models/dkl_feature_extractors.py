#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn


ActivationName = Literal["identity", "relu", "tanh", "sigmoid", "gelu"]


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _activation_module(name: ActivationName) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation name: {name!r}")


def _copy_activation(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.Identity):
        return nn.Identity()
    if isinstance(module, nn.ReLU):
        return nn.ReLU()
    if isinstance(module, nn.Tanh):
        return nn.Tanh()
    if isinstance(module, nn.Sigmoid):
        return nn.Sigmoid()
    if isinstance(module, nn.GELU):
        return nn.GELU()
    raise TypeError(f"Unsupported activation type for copy: {type(module)}")


class IdentityFeatureExtractor(nn.Module):
    """
    Identity map for ablations or debugging.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = _validate_positive_int("in_dim", in_dim)
        self.out_dim = self.in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.in_dim}], got {tuple(x.shape)}"
            )
        return x


class MLPFeatureExtractor(nn.Module):
    """
    Standard MLP feature extractor for DKL style models.

    Parameters
    ----------
    in_dim:
        Input dimension.
    hidden_dims:
        Hidden layer widths.
    out_dim:
        Final feature dimension.
    activation:
        Hidden activation.
    output_activation:
        Activation after the final layer.
    dropout:
        Dropout probability applied after hidden activations.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        out_dim: int = 32,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "identity",
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.in_dim = _validate_positive_int("in_dim", in_dim)
        self.out_dim = _validate_positive_int("out_dim", out_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty")
        self.hidden_dims = tuple(
            _validate_positive_int("hidden_dim", h) for h in hidden_dims
        )

        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dropout = float(dropout)

        hidden_act = _activation_module(activation)

        layers = []
        prev_dim = self.in_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(_copy_activation(hidden_act))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, self.out_dim))

        if output_activation != "identity":
            layers.append(_activation_module(output_activation))

        self.network = nn.Sequential(*layers)
        self.activation_name = activation
        self.output_activation_name = output_activation

        self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected x of shape [N, {self.in_dim}], got {tuple(x.shape)}"
            )
        return self.network(x)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"out_dim={self.out_dim}, "
            f"activation={self.activation_name}, "
            f"output_activation={self.output_activation_name}, "
            f"dropout={self.dropout}"
        )


def make_mlp_feature_extractor(
    *,
    in_dim: int,
    hidden: int = 256,
    depth: int = 2,
    out_dim: int = 32,
    activation: ActivationName = "relu",
    output_activation: ActivationName = "identity",
    dropout: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> MLPFeatureExtractor:
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    hidden_dims = tuple(int(hidden) for _ in range(int(depth)))
    return MLPFeatureExtractor(
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        activation=activation,
        output_activation=output_activation,
        dropout=dropout,
        dtype=dtype,
    )


__all__ = [
    "ActivationName",
    "IdentityFeatureExtractor",
    "MLPFeatureExtractor",
    "make_mlp_feature_extractor",
]