#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Literal, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationName = Literal["identity", "tanh", "sigmoid", "relu"]


def _validate_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _ensure_2d_float_tensor(x: torch.Tensor, *, input_dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)

    x = x.to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D after normalization, got shape {tuple(x.shape)}")

    if x.shape[1] != input_dim:
        raise ValueError(
            f"Expected x.shape[1] == {input_dim}, got {x.shape[1]}"
        )
    return x


def _activation_fn(name: ActivationName):
    if name == "identity":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "sigmoid":
        return torch.sigmoid
    if name == "relu":
        return F.relu
    raise ValueError(f"Unknown activation name: {name!r}")


def _freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _copy_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {} if metadata is None else dict(metadata)


class SyntheticObserverModule(nn.Module):
    """
    Base class for benchmark-side observer modules.

    These modules represent the ground-truth observer used to generate synthetic
    data for OFML experiments. They are not the learnable predictor modules used
    in models/. Their main role is to map an upstream latent vector z to an
    observable scalar or vector y.

    Design principles
    -----------------
    1. forward(..., sample_noise=False) returns deterministic output by default.
    2. sample_noise=True adds observational noise if noise_std > 0.
    3. All subclasses expose a stable state_dict so that observer similarity can
       later be implemented by parameter interpolation.
    """

    family_name: str = "base"

    def __init__(
        self,
        *,
        module_key: str,
        input_dim: int,
        output_dim: int = 1,
        noise_std: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        _validate_nonempty_str("module_key", module_key)

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.module_key = module_key
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.noise_std = float(noise_std)
        self.metadata: Dict[str, Any] = _copy_metadata(metadata)

    def forward_clean(self, z: torch.Tensor) -> torch.Tensor:
        """
        Deterministic observer output without observational noise.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def forward(
        self,
        z: torch.Tensor,
        *,
        sample_noise: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        z = _ensure_2d_float_tensor(z, input_dim=self.input_dim)
        y = self.forward_clean(z)

        if sample_noise and self.noise_std > 0.0:
            y = y + float(noise_scale) * self.noise_std * torch.randn_like(y)
        return y

    def clone(
        self,
        *,
        module_key: Optional[str] = None,
        noise_std: Optional[float] = None,
        metadata_update: Optional[Mapping[str, Any]] = None,
    ) -> "SyntheticObserverModule":
        """
        Deep copy with optional attribute overrides.
        """
        new_module = copy.deepcopy(self)
        if module_key is not None:
            _validate_nonempty_str("module_key", module_key)
            new_module.module_key = module_key
        if noise_std is not None:
            if noise_std < 0.0:
                raise ValueError(f"noise_std must be non-negative, got {noise_std}")
            new_module.noise_std = float(noise_std)
        if metadata_update is not None:
            new_module.metadata.update(dict(metadata_update))
        return new_module

    def config_dict(self) -> Dict[str, Any]:
        return {
            "family_name": self.family_name,
            "module_key": self.module_key,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "noise_std": self.noise_std,
            "metadata": dict(self.metadata),
        }


class LinearObserverModule(SyntheticObserverModule):
    family_name = "linear"

    def __init__(
        self,
        *,
        module_key: str,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: ActivationName = "identity",
        noise_std: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        weight = torch.as_tensor(weight, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)

        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
        if bias.ndim != 1:
            raise ValueError(f"bias must be 1D, got shape {tuple(bias.shape)}")
        if weight.shape[0] != bias.shape[0]:
            raise ValueError(
                f"weight.shape[0] and bias.shape[0] must match, got "
                f"{weight.shape[0]} and {bias.shape[0]}"
            )

        super().__init__(
            module_key=module_key,
            input_dim=int(weight.shape[1]),
            output_dim=int(weight.shape[0]),
            noise_std=noise_std,
            metadata=metadata,
        )
        self.activation_name: ActivationName = activation

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        _freeze_module(self.linear)

    def forward_clean(self, z: torch.Tensor) -> torch.Tensor:
        act = _activation_fn(self.activation_name)
        return act(self.linear(z))

    def config_dict(self) -> Dict[str, Any]:
        out = super().config_dict()
        out.update({"activation": self.activation_name})
        return out


class QuadraticObserverModule(SyntheticObserverModule):
    family_name = "quadratic"

    def __init__(
        self,
        *,
        module_key: str,
        linear_weight: torch.Tensor,
        quad_weight: torch.Tensor,
        bias: torch.Tensor,
        activation: ActivationName = "identity",
        noise_std: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        linear_weight = torch.as_tensor(linear_weight, dtype=torch.float32)
        quad_weight = torch.as_tensor(quad_weight, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)

        if linear_weight.ndim != 2:
            raise ValueError(
                f"linear_weight must be 2D, got shape {tuple(linear_weight.shape)}"
            )
        if quad_weight.ndim != 3:
            raise ValueError(
                f"quad_weight must be 3D, got shape {tuple(quad_weight.shape)}"
            )
        if bias.ndim != 1:
            raise ValueError(f"bias must be 1D, got shape {tuple(bias.shape)}")

        output_dim, input_dim = linear_weight.shape
        if quad_weight.shape != (output_dim, input_dim, input_dim):
            raise ValueError(
                "quad_weight must have shape "
                f"({output_dim}, {input_dim}, {input_dim}), "
                f"got {tuple(quad_weight.shape)}"
            )
        if bias.shape[0] != output_dim:
            raise ValueError(
                f"bias length must equal output_dim={output_dim}, got {bias.shape[0]}"
            )

        super().__init__(
            module_key=module_key,
            input_dim=input_dim,
            output_dim=output_dim,
            noise_std=noise_std,
            metadata=metadata,
        )
        self.activation_name: ActivationName = activation

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(linear_weight)
            self.linear.bias.copy_(bias)
        _freeze_module(self.linear)

        self.register_buffer("quad_weight", quad_weight.clone())

    def forward_clean(self, z: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(z)
        quad_part = torch.einsum("bi,oij,bj->bo", z, self.quad_weight, z)
        act = _activation_fn(self.activation_name)
        return act(linear_part + quad_part)

    def config_dict(self) -> Dict[str, Any]:
        out = super().config_dict()
        out.update({"activation": self.activation_name})
        return out


class MLPObserverModule(SyntheticObserverModule):
    family_name = "mlp"

    def __init__(
        self,
        *,
        module_key: str,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        hidden_activation: ActivationName = "tanh",
        output_activation: ActivationName = "identity",
        noise_std: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        fc1_weight = torch.as_tensor(fc1_weight, dtype=torch.float32)
        fc1_bias = torch.as_tensor(fc1_bias, dtype=torch.float32)
        fc2_weight = torch.as_tensor(fc2_weight, dtype=torch.float32)
        fc2_bias = torch.as_tensor(fc2_bias, dtype=torch.float32)

        if fc1_weight.ndim != 2 or fc2_weight.ndim != 2:
            raise ValueError("fc1_weight and fc2_weight must both be 2D")
        if fc1_bias.ndim != 1 or fc2_bias.ndim != 1:
            raise ValueError("fc1_bias and fc2_bias must both be 1D")

        hidden_dim, input_dim = fc1_weight.shape
        output_dim, hidden_dim_2 = fc2_weight.shape

        if hidden_dim != fc1_bias.shape[0]:
            raise ValueError("fc1_bias length must equal fc1 hidden dimension")
        if hidden_dim_2 != hidden_dim:
            raise ValueError("fc2 input dimension must equal fc1 output dimension")
        if output_dim != fc2_bias.shape[0]:
            raise ValueError("fc2_bias length must equal output dimension")

        super().__init__(
            module_key=module_key,
            input_dim=input_dim,
            output_dim=output_dim,
            noise_std=noise_std,
            metadata=metadata,
        )

        self.hidden_dim = hidden_dim
        self.hidden_activation_name: ActivationName = hidden_activation
        self.output_activation_name: ActivationName = output_activation

        self.fc1 = nn.Linear(self.input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim, bias=True)

        with torch.no_grad():
            self.fc1.weight.copy_(fc1_weight)
            self.fc1.bias.copy_(fc1_bias)
            self.fc2.weight.copy_(fc2_weight)
            self.fc2.bias.copy_(fc2_bias)

        _freeze_module(self.fc1)
        _freeze_module(self.fc2)

    def forward_clean(self, z: torch.Tensor) -> torch.Tensor:
        hidden_act = _activation_fn(self.hidden_activation_name)
        out_act = _activation_fn(self.output_activation_name)
        h = hidden_act(self.fc1(z))
        y = self.fc2(h)
        return out_act(y)

    def config_dict(self) -> Dict[str, Any]:
        out = super().config_dict()
        out.update(
            {
                "hidden_dim": self.hidden_dim,
                "hidden_activation": self.hidden_activation_name,
                "output_activation": self.output_activation_name,
            }
        )
        return out


def make_linear_observer(
    *,
    module_key: str,
    input_dim: int,
    output_dim: int = 1,
    seed: int = 0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.25,
    activation: ActivationName = "identity",
    noise_std: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> LinearObserverModule:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")

    g = _make_generator(seed)
    weight = weight_scale * torch.randn(output_dim, input_dim, generator=g)
    bias = bias_scale * torch.randn(output_dim, generator=g)

    return LinearObserverModule(
        module_key=module_key,
        weight=weight,
        bias=bias,
        activation=activation,
        noise_std=noise_std,
        metadata=metadata,
    )


def make_quadratic_observer(
    *,
    module_key: str,
    input_dim: int,
    output_dim: int = 1,
    seed: int = 0,
    linear_scale: float = 0.75,
    quad_scale: float = 0.25,
    bias_scale: float = 0.25,
    activation: ActivationName = "identity",
    noise_std: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> QuadraticObserverModule:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")

    g = _make_generator(seed)
    linear_weight = linear_scale * torch.randn(output_dim, input_dim, generator=g)
    quad_weight = quad_scale * torch.randn(
        output_dim, input_dim, input_dim, generator=g
    )
    bias = bias_scale * torch.randn(output_dim, generator=g)

    return QuadraticObserverModule(
        module_key=module_key,
        linear_weight=linear_weight,
        quad_weight=quad_weight,
        bias=bias,
        activation=activation,
        noise_std=noise_std,
        metadata=metadata,
    )


def make_mlp_observer(
    *,
    module_key: str,
    input_dim: int,
    hidden_dim: int = 16,
    output_dim: int = 1,
    seed: int = 0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.25,
    hidden_activation: ActivationName = "tanh",
    output_activation: ActivationName = "identity",
    noise_std: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> MLPObserverModule:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")

    g = _make_generator(seed)

    fc1_weight = (
        weight_scale / math.sqrt(max(input_dim, 1))
        * torch.randn(hidden_dim, input_dim, generator=g)
    )
    fc1_bias = bias_scale * torch.randn(hidden_dim, generator=g)
    fc2_weight = (
        weight_scale / math.sqrt(max(hidden_dim, 1))
        * torch.randn(output_dim, hidden_dim, generator=g)
    )
    fc2_bias = bias_scale * torch.randn(output_dim, generator=g)

    return MLPObserverModule(
        module_key=module_key,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        noise_std=noise_std,
        metadata=metadata,
    )


def parameter_distance(
    module_a: SyntheticObserverModule,
    module_b: SyntheticObserverModule,
    *,
    normalize: bool = True,
) -> float:
    """
    Euclidean distance between flattened floating parameters and buffers.
    This is useful as a diagnostic summary for observer similarity sweeps.
    """
    vec_a = flatten_module_state(module_a)
    vec_b = flatten_module_state(module_b)

    if vec_a.shape != vec_b.shape:
        raise ValueError(
            "parameter_distance requires compatible architectures, got "
            f"{tuple(vec_a.shape)} and {tuple(vec_b.shape)}"
        )

    dist = torch.norm(vec_a - vec_b, p=2).item()
    if not normalize:
        return float(dist)

    denom = max(torch.norm(vec_b, p=2).item(), 1e-12)
    return float(dist / denom)


def flatten_module_state(module: SyntheticObserverModule) -> torch.Tensor:
    """
    Flatten all floating tensors in state_dict into one 1D tensor.
    """
    chunks = []
    for _, tensor in module.state_dict().items():
        if not torch.is_floating_point(tensor):
            continue
        chunks.append(tensor.reshape(-1).float().cpu())
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def interpolate_state_dicts(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    *,
    similarity_to_a: float,
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between two compatible state_dicts.

    similarity_to_a = 1.0 gives state_a
    similarity_to_a = 0.0 gives state_b
    """
    alpha = _validate_probability("similarity_to_a", similarity_to_a)

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"State dict keys do not match: only_in_a={sorted(keys_a - keys_b)}, "
            f"only_in_b={sorted(keys_b - keys_a)}"
        )

    out: Dict[str, torch.Tensor] = {}
    for k in state_a.keys():
        ta = state_a[k]
        tb = state_b[k]
        if ta.shape != tb.shape:
            raise ValueError(
                f"Shape mismatch for key {k!r}: {tuple(ta.shape)} vs {tuple(tb.shape)}"
            )

        if torch.is_floating_point(ta):
            out[k] = alpha * ta + (1.0 - alpha) * tb
        else:
            out[k] = ta.clone()
    return out


def blend_observers(
    observer_a: SyntheticObserverModule,
    observer_b: SyntheticObserverModule,
    *,
    similarity_to_a: float,
    module_key: str,
    noise_std: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SyntheticObserverModule:
    """
    Blend two compatible observers by interpolating their state_dicts.

    Intended use
    ------------
    In Problem 1_A, one may treat observer_3 as the target observer and create
    observer_1 or observer_2 as interpolations between observer_3 and another
    anchor observer. Then similarity_to_a controls closeness to the target.
    """
    alpha = _validate_probability("similarity_to_a", similarity_to_a)

    if type(observer_a) is not type(observer_b):
        raise ValueError(
            "blend_observers requires the same concrete observer class, got "
            f"{type(observer_a).__name__} and {type(observer_b).__name__}"
        )
    if observer_a.input_dim != observer_b.input_dim:
        raise ValueError("Observers must have the same input_dim")
    if observer_a.output_dim != observer_b.output_dim:
        raise ValueError("Observers must have the same output_dim")

    blended = observer_a.clone(
        module_key=module_key,
        noise_std=observer_a.noise_std if noise_std is None else noise_std,
        metadata_update=metadata,
    )
    new_state = interpolate_state_dicts(
        observer_a.state_dict(),
        observer_b.state_dict(),
        similarity_to_a=alpha,
    )
    blended.load_state_dict(new_state)

    blended.metadata.update(
        {
            "blended_from": [observer_a.module_key, observer_b.module_key],
            "similarity_to_first": alpha,
        }
    )
    return blended


def make_anchor_paired_linear_observers(
    *,
    target_module_key: str,
    anchor_module_key: str,
    input_dim: int,
    target_seed: int,
    anchor_seed: int,
    target_noise_std: float = 0.0,
    anchor_noise_std: float = 0.0,
    activation: ActivationName = "identity",
) -> Dict[str, LinearObserverModule]:
    """
    Convenience helper used by observer similarity experiments.

    Returns two independently sampled linear observers:
    - target observer
    - anchor observer

    Later, one may build source observers by interpolating between them.
    """
    target = make_linear_observer(
        module_key=target_module_key,
        input_dim=input_dim,
        seed=target_seed,
        noise_std=target_noise_std,
        activation=activation,
        metadata={"observer_role": "target"},
    )
    anchor = make_linear_observer(
        module_key=anchor_module_key,
        input_dim=input_dim,
        seed=anchor_seed,
        noise_std=anchor_noise_std,
        activation=activation,
        metadata={"observer_role": "anchor"},
    )
    return {"target": target, "anchor": anchor}


__all__ = [
    "SyntheticObserverModule",
    "LinearObserverModule",
    "QuadraticObserverModule",
    "MLPObserverModule",
    "make_linear_observer",
    "make_quadratic_observer",
    "make_mlp_observer",
    "flatten_module_state",
    "parameter_distance",
    "interpolate_state_dicts",
    "blend_observers",
    "make_anchor_paired_linear_observers",
]