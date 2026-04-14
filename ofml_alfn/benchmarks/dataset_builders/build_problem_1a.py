#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Literal

import torch
import torch.nn.functional as F

from ofml_alfn.benchmarks.observer_families.linear_observers import (
    LinearObserverFamily,
    LinearObserverFamilyConfig,
    make_problem_1a_linear_observer_family,
)
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


def _activation_fn(name: ActivationName):
    if name == "identity":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    if name == "sigmoid":
        return torch.sigmoid
    raise ValueError(f"Unknown activation name: {name!r}")


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
class SharedUpstreamSpec:
    """
    Ground-truth shared upstream mapping used in Problem 1_A.

    z = activation(W x + b)

    This is the shared latent process used by all protocols in Problem 1_A.
    """

    input_dim: int
    latent_dim: int
    weight: torch.Tensor
    bias: torch.Tensor
    activation: ActivationName = "tanh"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

        if tuple(self.weight.shape) != (self.latent_dim, self.input_dim):
            raise ValueError(
                "weight must have shape "
                f"({self.latent_dim}, {self.input_dim}), got {tuple(self.weight.shape)}"
            )
        if tuple(self.bias.shape) != (self.latent_dim,):
            raise ValueError(
                f"bias must have shape ({self.latent_dim},), got {tuple(self.bias.shape)}"
            )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D after normalization, got {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"x.shape[1] must equal input_dim={self.input_dim}, got {x.shape[1]}"
            )

        act = _activation_fn(self.activation)
        z = x @ self.weight.t() + self.bias
        return act(z)


@dataclass(frozen=True)
class Problem1ADatasetBuilderConfig:
    input_dim: int = 2
    latent_dim: int = 4
    output_dim: int = 1

    condition_keys: Tuple[str, ...] = ("x0", "x1")
    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3")
    observer_module_keys: Tuple[str, str, str] = ("observer_1", "observer_2", "observer_3")

    similarities_to_target: Tuple[float, float, float] = (0.4, 0.7, 1.0)
    protocol_costs: Tuple[float, float, float] = (1.0, 2.0, 3.0)

    n_pretrain_p1: int = 128
    n_pretrain_p2: int = 128
    n_adapt_p3: int = 32
    n_val_p3: int = 128
    n_test_p3: int = 256

    condition_range: Tuple[float, float] = (-1.0, 1.0)

    upstream_seed: int = 0
    target_observer_seed: int = 1
    anchor_seeds: Tuple[int, int] = (101, 202)
    dataset_seed: int = 999

    upstream_activation: ActivationName = "tanh"
    observer_activation: ActivationName = "identity"

    upstream_weight_scale: float = 1.0
    upstream_bias_scale: float = 0.25
    observer_weight_scale: float = 1.0
    observer_bias_scale: float = 0.25

    target_noise_std: float = 0.0
    source_noise_stds: Tuple[float, float] = (0.0, 0.0)

    add_observation_noise_to_train: bool = True
    add_observation_noise_to_eval: bool = False

    save_dir: Optional[str] = None
    save_filename: str = "problem_1a_dataset.pt"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.condition_keys) != self.input_dim:
            raise ValueError(
                f"len(condition_keys) must equal input_dim={self.input_dim}, "
                f"got {len(self.condition_keys)}"
            )
        if len(self.protocol_ids) != 3:
            raise ValueError("protocol_ids must have length 3")
        if len(self.observer_module_keys) != 3:
            raise ValueError("observer_module_keys must have length 3")
        if len(self.similarities_to_target) != 3:
            raise ValueError("similarities_to_target must have length 3")
        if len(self.protocol_costs) != 3:
            raise ValueError("protocol_costs must have length 3")
        if len(self.anchor_seeds) != 2:
            raise ValueError("anchor_seeds must have length 2")
        if len(self.source_noise_stds) != 2:
            raise ValueError("source_noise_stds must have length 2")

        _validate_positive_int("input_dim", self.input_dim)
        _validate_positive_int("latent_dim", self.latent_dim)
        _validate_positive_int("output_dim", self.output_dim)

        _validate_nonnegative_int("n_pretrain_p1", self.n_pretrain_p1)
        _validate_nonnegative_int("n_pretrain_p2", self.n_pretrain_p2)
        _validate_nonnegative_int("n_adapt_p3", self.n_adapt_p3)
        _validate_nonnegative_int("n_val_p3", self.n_val_p3)
        _validate_nonnegative_int("n_test_p3", self.n_test_p3)

        _validate_float_pair("condition_range", self.condition_range)

        if float(self.similarities_to_target[2]) != 1.0:
            raise ValueError(
                "similarities_to_target[2] must be 1.0 because protocol_3 is the target"
            )

        for i, c in enumerate(self.protocol_costs):
            if float(c) < 0.0:
                raise ValueError(f"protocol_costs[{i}] must be non-negative, got {c}")

        if self.target_noise_std < 0.0:
            raise ValueError(f"target_noise_std must be non-negative, got {self.target_noise_std}")
        for i, v in enumerate(self.source_noise_stds):
            if v < 0.0:
                raise ValueError(
                    f"source_noise_stds[{i}] must be non-negative, got {v}"
                )

        _validate_nonempty_str("save_filename", self.save_filename)


@dataclass(frozen=True)
class Problem1ADatasetBuildResult:
    """
    Materialized benchmark data for Problem 1_A.
    """

    benchmark: Problem1ABenchmark
    observer_family: LinearObserverFamily
    shared_upstream: SharedUpstreamSpec
    samples: Tuple[BenchmarkSample, ...]
    splits: Tuple[DatasetSplit, ...]
    config: Problem1ADatasetBuilderConfig
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
            "problem_name": "problem_1a",
            "protocol_ids": self.benchmark.all_protocol_ids,
            "target_protocol_id": self.benchmark.target_protocol_id,
            "n_samples_total": len(self.samples),
            "split_sizes": {s.split_name: len(s.sample_ids) for s in self.splits},
            "similarities_to_target": self.observer_family.similarities_to_target(),
        }


def make_shared_upstream_spec(
    *,
    input_dim: int,
    latent_dim: int,
    seed: int = 0,
    activation: ActivationName = "tanh",
    weight_scale: float = 1.0,
    bias_scale: float = 0.25,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SharedUpstreamSpec:
    g = _make_generator(seed)
    weight = weight_scale * torch.randn(latent_dim, input_dim, generator=g)
    bias = bias_scale * torch.randn(latent_dim, generator=g)

    return SharedUpstreamSpec(
        input_dim=input_dim,
        latent_dim=latent_dim,
        weight=weight,
        bias=bias,
        activation=activation,
        metadata={} if metadata is None else dict(metadata),
    )


def sample_uniform_conditions(
    *,
    n: int,
    input_dim: int,
    low: float,
    high: float,
    seed: int,
) -> torch.Tensor:
    if n == 0:
        return torch.empty(0, input_dim, dtype=torch.float32)

    g = _make_generator(seed)
    x = torch.rand(n, input_dim, generator=g, dtype=torch.float32)
    return low + (high - low) * x


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


def _build_samples_for_protocol(
    *,
    protocol_id: str,
    split_name: str,
    condition_specs: Sequence[ConditionSpec],
    condition_keys: Sequence[str],
    shared_upstream: SharedUpstreamSpec,
    observer_module,
    target_output_key: str,
    observation_cost: float,
    sample_noise: bool,
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
            is_target=(protocol_id == "protocol_3"),
            metadata={
                "split_name": split_name,
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
            },
        )
        samples.append(sample)

    return tuple(samples)


def _make_split(split_name: str, samples: Sequence[BenchmarkSample]) -> DatasetSplit:
    protocol_ids = tuple(sorted(set(s.protocol_id for s in samples)))
    return DatasetSplit(
        split_name=split_name,
        sample_ids=tuple(s.sample_id for s in samples),
        protocol_ids=protocol_ids,
        metadata={"n_samples": len(samples)},
    )


def _build_problem_1a_benchmark(config: Problem1ADatasetBuilderConfig) -> Problem1ABenchmark:
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


def _build_problem_1a_observer_family(
    config: Problem1ADatasetBuilderConfig,
) -> LinearObserverFamily:
    family_config = LinearObserverFamilyConfig(
        protocol_ids=config.protocol_ids,
        module_keys=config.observer_module_keys,
        input_dim=config.latent_dim,
        output_dim=config.output_dim,
        similarities_to_target=config.similarities_to_target,
        target_seed=config.target_observer_seed,
        anchor_seeds=config.anchor_seeds,
        weight_scale=config.observer_weight_scale,
        bias_scale=config.observer_bias_scale,
        activation=config.observer_activation,
        target_noise_std=config.target_noise_std,
        source_noise_stds=config.source_noise_stds,
        metadata={
            "problem_name": "problem_1a",
        },
    )
    return make_problem_1a_linear_observer_family(family_config)


def build_problem_1a_dataset(
    config: Optional[Problem1ADatasetBuilderConfig] = None,
) -> Problem1ADatasetBuildResult:
    config = Problem1ADatasetBuilderConfig() if config is None else config

    benchmark = _build_problem_1a_benchmark(config)
    observer_family = _build_problem_1a_observer_family(config)

    shared_upstream = make_shared_upstream_spec(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        seed=config.upstream_seed,
        activation=config.upstream_activation,
        weight_scale=config.upstream_weight_scale,
        bias_scale=config.upstream_bias_scale,
        metadata={"problem_name": "problem_1a"},
    )

    low, high = config.condition_range
    base_seed = int(config.dataset_seed)

    x_pretrain_p1 = sample_uniform_conditions(
        n=config.n_pretrain_p1,
        input_dim=config.input_dim,
        low=low,
        high=high,
        seed=base_seed + 11,
    )
    x_pretrain_p2 = sample_uniform_conditions(
        n=config.n_pretrain_p2,
        input_dim=config.input_dim,
        low=low,
        high=high,
        seed=base_seed + 22,
    )
    x_adapt_p3 = sample_uniform_conditions(
        n=config.n_adapt_p3,
        input_dim=config.input_dim,
        low=low,
        high=high,
        seed=base_seed + 33,
    )
    x_val_p3 = sample_uniform_conditions(
        n=config.n_val_p3,
        input_dim=config.input_dim,
        low=low,
        high=high,
        seed=base_seed + 44,
    )
    x_test_p3 = sample_uniform_conditions(
        n=config.n_test_p3,
        input_dim=config.input_dim,
        low=low,
        high=high,
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

    result = Problem1ADatasetBuildResult(
        benchmark=benchmark,
        observer_family=observer_family,
        shared_upstream=shared_upstream,
        samples=all_samples,
        splits=splits,
        config=config,
        metadata={
            "problem_name": "problem_1a",
            "protocol_costs": config.protocol_costs,
        },
    )

    if config.save_dir is not None:
        save_problem_1a_build_result(result, config.save_dir, config.save_filename)

    return result


def save_problem_1a_build_result(
    result: Problem1ADatasetBuildResult,
    save_dir: str,
    filename: str = "problem_1a_dataset.pt",
) -> Path:
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / filename
    torch.save(result, file_path)
    return file_path


def load_problem_1a_build_result(path: str) -> Problem1ADatasetBuildResult:
    obj = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if not isinstance(obj, Problem1ADatasetBuildResult):
        raise TypeError(
            f"Loaded object is not Problem1ADatasetBuildResult, got {type(obj)}"
        )
    return obj


__all__ = [
    "SharedUpstreamSpec",
    "Problem1ADatasetBuilderConfig",
    "Problem1ADatasetBuildResult",
    "make_shared_upstream_spec",
    "sample_uniform_conditions",
    "build_problem_1a_dataset",
    "save_problem_1a_build_result",
    "load_problem_1a_build_result",
]