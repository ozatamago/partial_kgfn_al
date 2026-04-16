#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Literal

import torch
import torch.nn as nn

from ofml_alfn.utils.protocol_types import ProcessSpec


ActivationName = Literal["identity", "relu", "tanh", "sigmoid", "gelu"]


def _validate_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _to_2d_tensor(
    x: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        x = x.to(dtype=dtype, device=device)

    if x.ndim == 0:
        x = x.view(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor after normalization, got shape {tuple(x.shape)}"
        )
    return x


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


def copy_module(module: nn.Module) -> nn.Module:
    """
    Lightweight copier for stateless activation modules.
    """
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
    raise TypeError(f"Unsupported module type for copy_module: {type(module)}")


class BaseProcessModule(nn.Module):
    """
    Base class for learnable process modules used by ProtocolPredictor.

    Important distinction
    ---------------------
    These are model-side modules used for learning.
    They are different from the ground-truth synthetic modules placed under
    benchmarks/module_families/.

    Expected usage
    --------------
    ProtocolPredictor calls:

        module.forward_process(
            process=process_spec,
            inputs=input_dict,
            condition_x=condition_x,
            process_outputs=all_previous_outputs,
        )

    This base class implements:
    1. input collection and concatenation in process.input_keys order
    2. output splitting into a dict if multiple output_keys are used
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        output_key_dims: Optional[Mapping[str, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.output_dim = _validate_positive_int("output_dim", output_dim)
        self.default_dtype = dtype

        if output_key_dims is None:
            self.output_key_dims: Optional[Dict[str, int]] = None
        else:
            out_dims = {str(k): int(v) for k, v in output_key_dims.items()}
            if len(out_dims) == 0:
                raise ValueError("output_key_dims must be non-empty if provided")

            for key, dim in out_dims.items():
                _validate_nonempty_str("output_key_dims key", key)
                _validate_positive_int(f"output_key_dims[{key}]", dim)

            if sum(out_dims.values()) != self.output_dim:
                raise ValueError(
                    f"Sum of output_key_dims must equal output_dim={self.output_dim}, "
                    f"got sum={sum(out_dims.values())}"
                )

            self.output_key_dims = out_dims

    def _infer_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def _concat_inputs(
        self,
        *,
        process: ProcessSpec,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(inputs, Mapping):
            raise TypeError(f"inputs must be a mapping, got {type(inputs)}")
        if len(inputs) == 0:
            raise ValueError(f"Process {process.process_id!r} received empty inputs")

        device = self._infer_device()
        dtype = self.default_dtype

        ordered_keys: Sequence[str]
        if len(process.input_keys) > 0:
            ordered_keys = process.input_keys
        else:
            ordered_keys = tuple(inputs.keys())

        tensors = []
        batch_size: Optional[int] = None

        for key in ordered_keys:
            if key not in inputs:
                raise KeyError(
                    f"Missing input key {key!r} for process {process.process_id!r}. "
                    f"Available keys: {sorted(inputs.keys())}"
                )

            t = _to_2d_tensor(inputs[key], dtype=dtype, device=device)

            if batch_size is None:
                batch_size = t.shape[0]
            elif t.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes for process {process.process_id!r}: "
                    f"expected {batch_size}, got {t.shape[0]} for key {key!r}"
                )

            tensors.append(t)

        x = torch.cat(tensors, dim=1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Process module expected concatenated input_dim={self.input_dim}, "
                f"got {x.shape[1]} for process {process.process_id!r}"
            )

        return x

    def _split_outputs(
        self,
        *,
        process: ProcessSpec,
        y: torch.Tensor,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        y = _to_2d_tensor(
            y,
            dtype=self.default_dtype,
            device=y.device,
        )

        if y.shape[1] != self.output_dim:
            raise ValueError(
                f"Module produced output_dim={y.shape[1]}, expected {self.output_dim} "
                f"for process {process.process_id!r}"
            )

        if len(process.output_keys) == 1 and self.output_key_dims is None:
            return y

        if self.output_key_dims is not None:
            expected_keys = tuple(self.output_key_dims.keys())
            if tuple(process.output_keys) != expected_keys:
                raise ValueError(
                    f"process.output_keys={tuple(process.output_keys)} does not match "
                    f"module output_key_dims keys={expected_keys}"
                )

            out: Dict[str, torch.Tensor] = {}
            start = 0
            for key in process.output_keys:
                width = self.output_key_dims[key]
                out[key] = y[:, start : start + width]
                start += width
            return out

        if len(process.output_keys) != y.shape[1]:
            raise ValueError(
                f"Process {process.process_id!r} has {len(process.output_keys)} output_keys "
                f"but module returned shape {tuple(y.shape)}. "
                f"Provide output_key_dims if one output key should consume multiple columns."
            )

        return {
            key: y[:, i : i + 1]
            for i, key in enumerate(process.output_keys)
        }

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_process(
        self,
        *,
        process: ProcessSpec,
        inputs: Mapping[str, torch.Tensor],
        condition_x: Optional[torch.Tensor] = None,
        process_outputs: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        del condition_x, process_outputs
        x = self._concat_inputs(process=process, inputs=inputs)
        y = self.forward_backbone(x)
        return self._split_outputs(process=process, y=y)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"output_key_dims={self.output_key_dims}"
        )


class LinearProcessModule(BaseProcessModule):
    """
    Single affine map for a process.

    Useful for very small experiments or ablations.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        output_key_dims: Optional[Mapping[str, int]] = None,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            output_key_dims=output_key_dims,
            dtype=dtype,
        )
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.to(dtype=dtype)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProcessModule(BaseProcessModule):
    """
    Generic MLP process module.

    This is the default learnable module for both:
    - shared_upstream
    - observer_i
    in the first OFML 1_A experiments.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: ActivationName = "relu",
        output_activation: ActivationName = "identity",
        output_key_dims: Optional[Mapping[str, int]] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            output_key_dims=output_key_dims,
            dtype=dtype,
        )

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty for MLPProcessModule")

        hidden_dims = tuple(
            _validate_positive_int("hidden_dim", h) for h in hidden_dims
        )

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers = []
        prev_dim = input_dim
        hidden_act = _activation_module(activation)

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(copy_module(hidden_act))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation != "identity":
            layers.append(_activation_module(output_activation))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout = float(dropout)

        self.to(dtype=dtype)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, hidden_dims={self.hidden_dims}, "
            f"activation={self.activation_name}, "
            f"output_activation={self.output_activation_name}, "
            f"dropout={self.dropout}"
        )


class IdentityProcessModule(BaseProcessModule):
    """
    Identity map.

    Useful for debugging or simple ablations where a process should just pass
    its input through unchanged.
    """

    def __init__(
        self,
        *,
        dim: int,
        output_key_dims: Optional[Mapping[str, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            input_dim=dim,
            output_dim=dim,
            output_key_dims=output_key_dims,
            dtype=dtype,
        )

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_problem_1a_shared_upstream_module(
    *,
    input_dim: int,
    latent_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    activation: ActivationName = "relu",
    output_activation: ActivationName = "identity",
    dropout: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> MLPProcessModule:
    """
    Default learnable module for the shared upstream process in Problem 1_A.
    """
    return MLPProcessModule(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        dropout=dropout,
        dtype=dtype,
    )


def make_problem_1a_observer_module(
    *,
    latent_dim: int,
    output_dim: int = 1,
    hidden_dims: Sequence[int] = (64, 64),
    activation: ActivationName = "relu",
    output_activation: ActivationName = "identity",
    dropout: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> MLPProcessModule:
    """
    Default learnable module for each protocol-specific observer in Problem 1_A.
    """
    return MLPProcessModule(
        input_dim=latent_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        dropout=dropout,
        dtype=dtype,
    )


def make_problem_1a_module_registry(
    *,
    input_dim: int,
    latent_dim: int,
    output_dim: int = 1,
    observer_module_keys: Sequence[str] = ("observer_1", "observer_2", "observer_3"),
    upstream_hidden_dims: Sequence[int] = (64, 64),
    observer_hidden_dims: Sequence[int] = (64, 64),
    activation: ActivationName = "relu",
    output_activation: ActivationName = "identity",
    dropout: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> nn.ModuleDict:
    """
    Convenience factory for the first Problem 1_A experiments.

    Registry keys
    -------------
    - "shared_upstream"
    - observer_module_keys[0]
    - observer_module_keys[1]
    - observer_module_keys[2]

    Interpretation
    --------------
    The upstream module is shared across protocols.
    Each observer has its own learnable module.
    """
    observer_module_keys = tuple(observer_module_keys)

    if len(observer_module_keys) != 3:
        raise ValueError(
            f"observer_module_keys must have length 3 for Problem 1_A, "
            f"got {observer_module_keys}"
        )
    if len(set(observer_module_keys)) != 3:
        raise ValueError(
            f"observer_module_keys must be unique, got {observer_module_keys}"
        )

    modules = nn.ModuleDict()
    modules["shared_upstream"] = make_problem_1a_shared_upstream_module(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=upstream_hidden_dims,
        activation=activation,
        output_activation=output_activation,
        dropout=dropout,
        dtype=dtype,
    )

    for key in observer_module_keys:
        modules[key] = make_problem_1a_observer_module(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=observer_hidden_dims,
            activation=activation,
            output_activation=output_activation,
            dropout=dropout,
            dtype=dtype,
        )

    return modules


__all__ = [
    "ActivationName",
    "BaseProcessModule",
    "LinearProcessModule",
    "MLPProcessModule",
    "IdentityProcessModule",
    "make_problem_1a_shared_upstream_module",
    "make_problem_1a_observer_module",
    "make_problem_1a_module_registry",
]