#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Literal

import torch

from ofml_alfn.benchmarks.protocol_families.problem_1a import (
    Problem1ABenchmark,
    Problem1AConfig,
    make_problem_1a,
)
from ofml_alfn.utils.protocol_types import (
    BenchmarkSample,
    ConditionSpec,
    DatasetSplit,
    ProtocolObservation,
)

ActivationName = Literal["identity", "tanh", "relu", "sigmoid"]


def _validate_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_nonnegative_int(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_float_pair(name: str, value: Tuple[float, float]) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {value}")
    lo, hi = float(value[0]), float(value[1])
    if not lo < hi:
        raise ValueError(f"{name} must satisfy lo < hi, got {value}")
    return (lo, hi)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _tensor_to_condition_dict(
    x: torch.Tensor,
    condition_keys: Sequence[str],
) -> Dict[str, float]:
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {tuple(x.shape)}")
    if len(condition_keys) != x.shape[0]:
        raise ValueError(
            f"len(condition_keys) must equal x.shape[0], got "
            f"{len(condition_keys)} vs {x.shape[0]}"
        )
    return {k: float(v.item()) for k, v in zip(condition_keys, x)}


def _stack_condition_dict(
    values: Mapping[str, Any],
    condition_keys: Sequence[str],
) -> torch.Tensor:
    return torch.tensor([float(values[k]) for k in condition_keys], dtype=torch.float32)


@dataclass(frozen=True)
class EllipsoidChartUpstreamSpec:
    """
    Ground-truth shared upstream mapping for the ellipsoid variant.

    Input:
        x = (x0, x1, x2) in the 3-ball of radius chart_radius

    Output:
        z in R^4 constrained to lie on the ellipsoid
            sum_i ((z_i - a_i) / d_i)^2 = 1

    Chart:
        z1 = a1 + d1 * x0
        z2 = a2 + d2 * x1
        z3 = a3 + d3 * x2
        z4 = a4 + sign * d4 * sqrt(1 - x0^2 - x1^2 - x2^2)
    """

    input_dim: int
    latent_dim: int
    center: torch.Tensor
    scales: torch.Tensor
    chart_sign: int = +1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dim != 3:
            raise ValueError(f"Ellipsoid chart expects input_dim=3, got {self.input_dim}")
        if self.latent_dim != 4:
            raise ValueError(f"Ellipsoid chart expects latent_dim=4, got {self.latent_dim}")
        if tuple(self.center.shape) != (4,):
            raise ValueError(f"center must have shape (4,), got {tuple(self.center.shape)}")
        if tuple(self.scales.shape) != (4,):
            raise ValueError(f"scales must have shape (4,), got {tuple(self.scales.shape)}")
        if self.chart_sign not in (-1, +1):
            raise ValueError(f"chart_sign must be -1 or +1, got {self.chart_sign}")
        if torch.any(self.scales <= 0):
            raise ValueError("All ellipsoid scales must be positive")

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D after normalization, got {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"x.shape[1] must equal input_dim={self.input_dim}, got {x.shape[1]}"
            )

        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        inside = 1.0 - x0.pow(2) - x1.pow(2) - x2.pow(2)
        inside = torch.clamp(inside, min=0.0)
        root = torch.sqrt(inside)

        z1 = self.center[0] + self.scales[0] * x0
        z2 = self.center[1] + self.scales[1] * x1
        z3 = self.center[2] + self.scales[2] * x2
        z4 = self.center[3] + float(self.chart_sign) * self.scales[3] * root

        z = torch.stack([z1, z2, z3, z4], dim=-1)
        return z


@dataclass(frozen=True)
class FixedProjectionObserverSpec:
    """
    Protocol-specific fixed projection from z in R^4 to y in R^k.

    Example:
        protocol_1: indices = (0, 3)      -> y1 = (z1, z4)
        protocol_2: indices = (1, 2, 3)   -> y2 = (z2, z3, z4)
        protocol_3: indices = (0, 1, 2)   -> y3 = (z1, z2, z3)
    """

    protocol_id: str
    module_key: str
    input_dim: int
    output_dim: int
    indices: Tuple[int, ...]
    noise_std: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("protocol_id", self.protocol_id)
        _validate_nonempty_str("module_key", self.module_key)
        _validate_positive_int("input_dim", self.input_dim)
        _validate_positive_int("output_dim", self.output_dim)
        if len(self.indices) != self.output_dim:
            raise ValueError(
                f"len(indices) must equal output_dim={self.output_dim}, got {len(self.indices)}"
            )
        for idx in self.indices:
            if not 0 <= int(idx) < self.input_dim:
                raise ValueError(
                    f"projection index {idx} out of range for input_dim={self.input_dim}"
                )
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")

    def apply(self, z: torch.Tensor, sample_noise: bool = False) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2:
            raise ValueError(f"z must be 2D after normalization, got {tuple(z.shape)}")
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"z.shape[1] must equal input_dim={self.input_dim}, got {z.shape[1]}"
            )

        y = z[:, list(self.indices)]
        if sample_noise and self.noise_std > 0.0:
            y = y + self.noise_std * torch.randn_like(y)
        return y


class FixedProjectionObserverModule:
    def __init__(self, spec: FixedProjectionObserverSpec) -> None:
        self.spec = spec

    def __call__(self, z: torch.Tensor, sample_noise: bool = False) -> torch.Tensor:
        return self.spec.apply(z, sample_noise=sample_noise)


@dataclass(frozen=True)
class FixedProjectionObserverFamily:
    protocol_ids: Tuple[str, ...]
    module_keys: Tuple[str, ...]
    specs: Dict[str, FixedProjectionObserverSpec]
    similarities: Tuple[float, ...]
    scales: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for pid in self.protocol_ids:
            if pid not in self.specs:
                raise ValueError(f"Missing observer spec for protocol_id={pid!r}")

    def get(self, protocol_id: str) -> FixedProjectionObserverModule:
        try:
            return FixedProjectionObserverModule(self.specs[protocol_id])
        except KeyError as exc:
            raise KeyError(f"Unknown protocol_id for observer family: {protocol_id!r}") from exc

    def similarities_to_target(self) -> Tuple[float, ...]:
        return self.similarities

    def observer_scales(self) -> Tuple[float, ...]:
        return self.scales


@dataclass(frozen=True)
class Problem1AEllipsoidDatasetBuilderConfig:
    input_dim: int = 3
    latent_dim: int = 4

    condition_keys: Tuple[str, ...] = ("x0", "x1", "x2")
    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3")
    observer_module_keys: Tuple[str, str, str] = ("observer_1", "observer_2", "observer_3")

    # output dims are protocol-specific in this variant
    observer_output_dims: Tuple[int, int, int] = (2, 3, 3)

    # fixed projections:
    # y1=(z1,z4), y2=(z2,z3,z4), y3=(z1,z2,z3)
    observer_index_tuples: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]] = (
        (0, 3),
        (1, 2, 3),
        (0, 1, 2),
    )

    similarities_to_target: Tuple[float, float, float] = (0.4, 0.7, 1.0)
    observer_scales: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    protocol_costs: Tuple[float, float, float] = (1.0, 2.0, 3.0)

    n_pretrain_p1: int = 128
    n_pretrain_p2: int = 128
    n_adapt_p3: int = 32
    n_val_p3: int = 128
    n_test_p3: int = 256

    # sample x uniformly from the 3-ball of radius chart_radius
    chart_radius: float = 0.90

    ellipsoid_center: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ellipsoid_scales: Tuple[float, float, float, float] = (1.0, 1.2, 0.8, 1.5)
    chart_sign: int = +1

    dataset_seed: int = 999
    target_noise_std: float = 0.0
    source_noise_stds: Tuple[float, float] = (0.0, 0.0)

    add_observation_noise_to_train: bool = True
    add_observation_noise_to_eval: bool = False

    save_dir: Optional[str] = None
    save_filename: str = "problem_1a_ellipsoid_dataset.pt"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dim != 3:
            raise ValueError(f"Ellipsoid variant expects input_dim=3, got {self.input_dim}")
        if self.latent_dim != 4:
            raise ValueError(f"Ellipsoid variant expects latent_dim=4, got {self.latent_dim}")

        if len(self.condition_keys) != self.input_dim:
            raise ValueError(
                f"len(condition_keys) must equal input_dim={self.input_dim}, "
                f"got {len(self.condition_keys)}"
            )
        if len(self.protocol_ids) != 3:
            raise ValueError("protocol_ids must have length 3")
        if len(self.observer_module_keys) != 3:
            raise ValueError("observer_module_keys must have length 3")
        if len(self.observer_output_dims) != 3:
            raise ValueError("observer_output_dims must have length 3")
        if len(self.observer_index_tuples) != 3:
            raise ValueError("observer_index_tuples must have length 3")
        if len(self.similarities_to_target) != 3:
            raise ValueError("similarities_to_target must have length 3")
        if len(self.observer_scales) != 3:
            raise ValueError("observer_scales must have length 3")
        if len(self.protocol_costs) != 3:
            raise ValueError("protocol_costs must have length 3")
        if len(self.source_noise_stds) != 2:
            raise ValueError("source_noise_stds must have length 2")
        if len(self.ellipsoid_center) != 4:
            raise ValueError("ellipsoid_center must have length 4")
        if len(self.ellipsoid_scales) != 4:
            raise ValueError("ellipsoid_scales must have length 4")

        _validate_positive_int("n_pretrain_p1", self.n_pretrain_p1)
        _validate_positive_int("n_pretrain_p2", self.n_pretrain_p2)
        _validate_nonnegative_int("n_adapt_p3", self.n_adapt_p3)
        _validate_nonnegative_int("n_val_p3", self.n_val_p3)
        _validate_nonnegative_int("n_test_p3", self.n_test_p3)
        _validate_nonempty_str("save_filename", self.save_filename)

        if float(self.similarities_to_target[2]) != 1.0:
            raise ValueError(
                "similarities_to_target[2] must be 1.0 because protocol_3 is the target"
            )

        if not (0.0 < float(self.chart_radius) <= 1.0):
            raise ValueError(f"chart_radius must lie in (0, 1], got {self.chart_radius}")

        for i, scale in enumerate(self.ellipsoid_scales):
            if float(scale) <= 0.0:
                raise ValueError(f"ellipsoid_scales[{i}] must be positive, got {scale}")

        if self.chart_sign not in (-1, +1):
            raise ValueError(f"chart_sign must be -1 or +1, got {self.chart_sign}")

        if self.target_noise_std < 0.0:
            raise ValueError(f"target_noise_std must be non-negative, got {self.target_noise_std}")
        for i, v in enumerate(self.source_noise_stds):
            if v < 0.0:
                raise ValueError(
                    f"source_noise_stds[{i}] must be non-negative, got {v}"
                )

        for i, inds in enumerate(self.observer_index_tuples):
            expected = self.observer_output_dims[i]
            if len(inds) != expected:
                raise ValueError(
                    f"observer_index_tuples[{i}] length must equal "
                    f"observer_output_dims[{i}]={expected}, got {len(inds)}"
                )
            for idx in inds:
                if not 0 <= int(idx) < self.latent_dim:
                    raise ValueError(
                        f"observer_index_tuples[{i}] contains out-of-range index {idx} "
                        f"for latent_dim={self.latent_dim}"
                    )


@dataclass(frozen=True)
class Problem1AEllipsoidDatasetBuildResult:
    benchmark: Problem1ABenchmark
    observer_family: FixedProjectionObserverFamily
    shared_upstream: EllipsoidChartUpstreamSpec
    samples: Tuple[BenchmarkSample, ...]
    splits: Tuple[DatasetSplit, ...]
    config: Problem1AEllipsoidDatasetBuilderConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_map(self) -> Dict[str, BenchmarkSample]:
        return {s.sample_id: s for s in self.samples}

    @property
    def split_map(self) -> Dict[str, DatasetSplit]:
        return {s.split_name: s for s in self.splits}

    def get_split(self, split_name: str) -> DatasetSplit:
        try:
            return self.split_map[split_name]
        except KeyError as exc:
            raise KeyError(f"Unknown split_name: {split_name!r}") from exc

    def samples_in_split(self, split_name: str) -> Tuple[BenchmarkSample, ...]:
        split = self.get_split(split_name)
        sample_map = self.sample_map
        return tuple(sample_map[sid] for sid in split.sample_ids)

    def summary(self) -> Dict[str, Any]:
        return {
            "problem_name": "problem_1a_ellipsoid",
            "protocol_ids": self.benchmark.all_protocol_ids,
            "target_protocol_id": self.benchmark.target_protocol_id,
            "n_samples_total": len(self.samples),
            "split_sizes": {s.split_name: len(s.sample_ids) for s in self.splits},
            "similarities_to_target": self.observer_family.similarities_to_target(),
            "observer_scales": self.observer_family.observer_scales(),
            "observer_output_dims": self.config.observer_output_dims,
        }


def make_ellipsoid_chart_upstream_spec(
    *,
    input_dim: int = 3,
    latent_dim: int = 4,
    center: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    scales: Sequence[float] = (1.0, 1.2, 0.8, 1.5),
    chart_sign: int = +1,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EllipsoidChartUpstreamSpec:
    return EllipsoidChartUpstreamSpec(
        input_dim=input_dim,
        latent_dim=latent_dim,
        center=torch.tensor(center, dtype=torch.float32),
        scales=torch.tensor(scales, dtype=torch.float32),
        chart_sign=int(chart_sign),
        metadata={} if metadata is None else dict(metadata),
    )


def sample_uniform_ball_conditions(
    *,
    n: int,
    input_dim: int,
    radius: float,
    seed: int,
) -> torch.Tensor:
    """
    Uniform sampling from the d-ball of radius `radius`.
    """
    if n == 0:
        return torch.empty(0, input_dim, dtype=torch.float32)

    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    g = _make_generator(seed)

    # direction ~ normalized Gaussian
    dirs = torch.randn(n, input_dim, generator=g, dtype=torch.float32)
    norms = torch.norm(dirs, dim=1, keepdim=True).clamp_min(1e-12)
    dirs = dirs / norms

    # radius for uniform d-ball: R * U^(1/d)
    u = torch.rand(n, 1, generator=g, dtype=torch.float32)
    r = float(radius) * torch.pow(u, 1.0 / float(input_dim))

    return dirs * r


def _make_condition_specs(
    *,
    protocol_id: str,
    split_name: str,
    x: torch.Tensor,
    condition_keys: Sequence[str],
) -> Tuple[ConditionSpec, ...]:
    conds = []
    for i in range(x.shape[0]):
        cond_id = f"{protocol_id}__{split_name}__cond_{i:05d}"
        conds.append(
            ConditionSpec(
                condition_id=cond_id,
                values=_tensor_to_condition_dict(x[i], condition_keys),
                metadata={
                    "protocol_id": protocol_id,
                    "split_name": split_name,
                    "index_within_split": i,
                },
            )
        )
    return tuple(conds)


def _make_split(split_name: str, samples: Sequence[BenchmarkSample]) -> DatasetSplit:
    protocol_ids = tuple(sorted(set(s.protocol_id for s in samples)))
    return DatasetSplit(
        split_name=split_name,
        sample_ids=tuple(s.sample_id for s in samples),
        protocol_ids=protocol_ids,
        metadata={"n_samples": len(samples)},
    )


def _build_problem_1a_benchmark(
    config: Problem1AEllipsoidDatasetBuilderConfig,
) -> Problem1ABenchmark:
    problem_config = Problem1AConfig(
        protocol_ids=config.protocol_ids,
        condition_keys=config.condition_keys,
        upstream_output_key="z",
        observer_output_key="y",
        upstream_module_key="shared_upstream",
        observer_module_keys=config.observer_module_keys,
        upstream_cost=0.0,
        observer_costs=config.protocol_costs,
        similarity_to_target=config.similarities_to_target,
    )
    return make_problem_1a(problem_config)


def _build_problem_1a_ellipsoid_observer_family(
    config: Problem1AEllipsoidDatasetBuilderConfig,
) -> FixedProjectionObserverFamily:
    pid1, pid2, pid3 = config.protocol_ids
    mk1, mk2, mk3 = config.observer_module_keys
    od1, od2, od3 = config.observer_output_dims
    inds1, inds2, inds3 = config.observer_index_tuples

    spec1 = FixedProjectionObserverSpec(
        protocol_id=pid1,
        module_key=mk1,
        input_dim=config.latent_dim,
        output_dim=od1,
        indices=inds1,
        noise_std=float(config.source_noise_stds[0]),
        metadata={"role": "source", "problem_name": "problem_1a_ellipsoid"},
    )
    spec2 = FixedProjectionObserverSpec(
        protocol_id=pid2,
        module_key=mk2,
        input_dim=config.latent_dim,
        output_dim=od2,
        indices=inds2,
        noise_std=float(config.source_noise_stds[1]),
        metadata={"role": "source", "problem_name": "problem_1a_ellipsoid"},
    )
    spec3 = FixedProjectionObserverSpec(
        protocol_id=pid3,
        module_key=mk3,
        input_dim=config.latent_dim,
        output_dim=od3,
        indices=inds3,
        noise_std=float(config.target_noise_std),
        metadata={"role": "target", "problem_name": "problem_1a_ellipsoid"},
    )

    return FixedProjectionObserverFamily(
        protocol_ids=config.protocol_ids,
        module_keys=config.observer_module_keys,
        specs={
            pid1: spec1,
            pid2: spec2,
            pid3: spec3,
        },
        similarities=config.similarities_to_target,
        scales=config.observer_scales,
        metadata={"problem_name": "problem_1a_ellipsoid"},
    )


def _build_samples_for_protocol(
    *,
    protocol_id: str,
    split_name: str,
    condition_specs: Sequence[ConditionSpec],
    condition_keys: Sequence[str],
    shared_upstream: EllipsoidChartUpstreamSpec,
    observer_module: FixedProjectionObserverModule,
    target_output_key: str,
    observation_cost: float,
    sample_noise: bool,
    target_protocol_id: str,
) -> Tuple[BenchmarkSample, ...]:
    samples = []
    for i, cond in enumerate(condition_specs):
        x = _stack_condition_dict(cond.values, condition_keys).unsqueeze(0)
        z = shared_upstream.apply(x)
        y = observer_module(z, sample_noise=sample_noise)

        if y.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 observer output, got {tuple(y.shape)}")

        y_value = y.squeeze(0).detach().cpu()

        observation = ProtocolObservation(
            protocol_id=protocol_id,
            condition_id=cond.condition_id,
            process_id="S2",
            output_key=target_output_key,
            value=y_value.clone(),
            cost=float(observation_cost),
            is_target=(protocol_id == target_protocol_id),
            metadata={
                "split_name": split_name,
                "y_dim": int(y_value.shape[0]) if y_value.ndim > 0 else 1,
            },
        )

        sample = BenchmarkSample(
            sample_id=f"{protocol_id}__{split_name}__sample_{i:05d}",
            protocol_id=protocol_id,
            condition=cond,
            target_value=y_value.clone(),
            observations=(observation,),
            metadata={
                "split_name": split_name,
                "x": x.squeeze(0).detach().cpu(),
                "z": z.squeeze(0).detach().cpu(),
                "observation_cost": float(observation_cost),
                "y_dim": int(y_value.shape[0]) if y_value.ndim > 0 else 1,
            },
        )
        samples.append(sample)

    return tuple(samples)


def build_problem_1a_ellipsoid_dataset(
    config: Optional[Problem1AEllipsoidDatasetBuilderConfig] = None,
) -> Problem1AEllipsoidDatasetBuildResult:
    config = Problem1AEllipsoidDatasetBuilderConfig() if config is None else config

    benchmark = _build_problem_1a_benchmark(config)
    observer_family = _build_problem_1a_ellipsoid_observer_family(config)
    shared_upstream = make_ellipsoid_chart_upstream_spec(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        center=config.ellipsoid_center,
        scales=config.ellipsoid_scales,
        chart_sign=config.chart_sign,
        metadata={"problem_name": "problem_1a_ellipsoid"},
    )

    base_seed = int(config.dataset_seed)

    x_pretrain_p1 = sample_uniform_ball_conditions(
        n=config.n_pretrain_p1,
        input_dim=config.input_dim,
        radius=float(config.chart_radius),
        seed=base_seed + 11,
    )
    x_pretrain_p2 = sample_uniform_ball_conditions(
        n=config.n_pretrain_p2,
        input_dim=config.input_dim,
        radius=float(config.chart_radius),
        seed=base_seed + 22,
    )
    x_adapt_p3 = sample_uniform_ball_conditions(
        n=config.n_adapt_p3,
        input_dim=config.input_dim,
        radius=float(config.chart_radius),
        seed=base_seed + 33,
    )
    x_val_p3 = sample_uniform_ball_conditions(
        n=config.n_val_p3,
        input_dim=config.input_dim,
        radius=float(config.chart_radius),
        seed=base_seed + 44,
    )
    x_test_p3 = sample_uniform_ball_conditions(
        n=config.n_test_p3,
        input_dim=config.input_dim,
        radius=float(config.chart_radius),
        seed=base_seed + 55,
    )

    pid1, pid2, pid3 = config.protocol_ids
    cost1, cost2, cost3 = config.protocol_costs

    conds_pretrain_p1 = _make_condition_specs(
        protocol_id=pid1,
        split_name="pretrain_protocol_1",
        x=x_pretrain_p1,
        condition_keys=config.condition_keys,
    )
    conds_pretrain_p2 = _make_condition_specs(
        protocol_id=pid2,
        split_name="pretrain_protocol_2",
        x=x_pretrain_p2,
        condition_keys=config.condition_keys,
    )
    conds_adapt_p3 = _make_condition_specs(
        protocol_id=pid3,
        split_name="adapt_protocol_3",
        x=x_adapt_p3,
        condition_keys=config.condition_keys,
    )
    conds_val_p3 = _make_condition_specs(
        protocol_id=pid3,
        split_name="val_protocol_3",
        x=x_val_p3,
        condition_keys=config.condition_keys,
    )
    conds_test_p3 = _make_condition_specs(
        protocol_id=pid3,
        split_name="test_protocol_3",
        x=x_test_p3,
        condition_keys=config.condition_keys,
    )

    target_output_key = benchmark.target_protocol.target_output_key
    target_protocol_id = benchmark.target_protocol_id

    samples_pretrain_p1 = _build_samples_for_protocol(
        protocol_id=pid1,
        split_name="pretrain_protocol_1",
        condition_specs=conds_pretrain_p1,
        condition_keys=config.condition_keys,
        shared_upstream=shared_upstream,
        observer_module=observer_family.get(pid1),
        target_output_key=target_output_key,
        observation_cost=cost1,
        sample_noise=config.add_observation_noise_to_train,
        target_protocol_id=target_protocol_id,
    )
    samples_pretrain_p2 = _build_samples_for_protocol(
        protocol_id=pid2,
        split_name="pretrain_protocol_2",
        condition_specs=conds_pretrain_p2,
        condition_keys=config.condition_keys,
        shared_upstream=shared_upstream,
        observer_module=observer_family.get(pid2),
        target_output_key=target_output_key,
        observation_cost=cost2,
        sample_noise=config.add_observation_noise_to_train,
        target_protocol_id=target_protocol_id,
    )
    samples_adapt_p3 = _build_samples_for_protocol(
        protocol_id=pid3,
        split_name="adapt_protocol_3",
        condition_specs=conds_adapt_p3,
        condition_keys=config.condition_keys,
        shared_upstream=shared_upstream,
        observer_module=observer_family.get(pid3),
        target_output_key=target_output_key,
        observation_cost=cost3,
        sample_noise=config.add_observation_noise_to_train,
        target_protocol_id=target_protocol_id,
    )
    samples_val_p3 = _build_samples_for_protocol(
        protocol_id=pid3,
        split_name="val_protocol_3",
        condition_specs=conds_val_p3,
        condition_keys=config.condition_keys,
        shared_upstream=shared_upstream,
        observer_module=observer_family.get(pid3),
        target_output_key=target_output_key,
        observation_cost=cost3,
        sample_noise=config.add_observation_noise_to_eval,
        target_protocol_id=target_protocol_id,
    )
    samples_test_p3 = _build_samples_for_protocol(
        protocol_id=pid3,
        split_name="test_protocol_3",
        condition_specs=conds_test_p3,
        condition_keys=config.condition_keys,
        shared_upstream=shared_upstream,
        observer_module=observer_family.get(pid3),
        target_output_key=target_output_key,
        observation_cost=cost3,
        sample_noise=config.add_observation_noise_to_eval,
        target_protocol_id=target_protocol_id,
    )

    all_samples = (
        samples_pretrain_p1
        + samples_pretrain_p2
        + samples_adapt_p3
        + samples_val_p3
        + samples_test_p3
    )

    splits = (
        _make_split("pretrain_protocol_1", samples_pretrain_p1),
        _make_split("pretrain_protocol_2", samples_pretrain_p2),
        _make_split("adapt_protocol_3", samples_adapt_p3),
        _make_split("val_protocol_3", samples_val_p3),
        _make_split("test_protocol_3", samples_test_p3),
        _make_split("pretrain_all_sources", samples_pretrain_p1 + samples_pretrain_p2),
    )

    result = Problem1AEllipsoidDatasetBuildResult(
        benchmark=benchmark,
        observer_family=observer_family,
        shared_upstream=shared_upstream,
        samples=all_samples,
        splits=splits,
        config=config,
        metadata={
            "problem_name": "problem_1a_ellipsoid",
            "protocol_costs": config.protocol_costs,
            "observer_scales": config.observer_scales,
            "observer_output_dims": config.observer_output_dims,
            "chart_radius": config.chart_radius,
            "ellipsoid_center": config.ellipsoid_center,
            "ellipsoid_scales": config.ellipsoid_scales,
            "chart_sign": config.chart_sign,
        },
    )

    if config.save_dir is not None:
        save_problem_1a_ellipsoid_build_result(result, config.save_dir, config.save_filename)

    return result


def save_problem_1a_ellipsoid_build_result(
    result: Problem1AEllipsoidDatasetBuildResult,
    save_dir: str,
    filename: str = "problem_1a_ellipsoid_dataset.pt",
) -> Path:
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / filename
    torch.save(result, file_path)
    return file_path


def load_problem_1a_ellipsoid_build_result(
    path: str,
) -> Problem1AEllipsoidDatasetBuildResult:
    obj = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(obj, Problem1AEllipsoidDatasetBuildResult):
        raise TypeError(
            "Loaded object is not Problem1AEllipsoidDatasetBuildResult, "
            f"got {type(obj)}"
        )
    return obj


__all__ = [
    "EllipsoidChartUpstreamSpec",
    "FixedProjectionObserverSpec",
    "FixedProjectionObserverModule",
    "FixedProjectionObserverFamily",
    "Problem1AEllipsoidDatasetBuilderConfig",
    "Problem1AEllipsoidDatasetBuildResult",
    "make_ellipsoid_chart_upstream_spec",
    "sample_uniform_ball_conditions",
    "build_problem_1a_ellipsoid_dataset",
    "save_problem_1a_ellipsoid_build_result",
    "load_problem_1a_ellipsoid_build_result",
]