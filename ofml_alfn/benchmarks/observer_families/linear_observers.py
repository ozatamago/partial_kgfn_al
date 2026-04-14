#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ofml_alfn.benchmarks.module_families.observer_modules import (
    ActivationName,
    LinearObserverModule,
    blend_observers,
    make_linear_observer,
    parameter_distance,
)


def _validate_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_unique_strings(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    values = tuple(values)
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique strings, got {values}")
    for v in values:
        _validate_nonempty_str(name, v)
    return values


@dataclass(frozen=True)
class LinearObserverFamilyConfig:
    """
    Configuration for a linear observer family used in Problem 1_A.

    Design
    ------
    - Protocol 3 owns the target observer.
    - Protocols 1 and 2 own source observers.
    - Each source observer is created by interpolating between the target
      observer and a source-specific anchor observer.
    - similarity_to_target controls closeness to the target observer.

    Convention
    ----------
    similarities_to_target[2] must be 1.0 because protocol 3 is the target.
    """

    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3")
    module_keys: Tuple[str, str, str] = ("observer_1", "observer_2", "observer_3")

    input_dim: int = 4
    output_dim: int = 1

    similarities_to_target: Tuple[float, float, float] = (0.4, 0.7, 1.0)

    target_seed: int = 0
    anchor_seeds: Tuple[int, int] = (101, 202)

    weight_scale: float = 1.0
    bias_scale: float = 0.25

    activation: ActivationName = "identity"

    target_noise_std: float = 0.0
    source_noise_stds: Tuple[float, float] = (0.0, 0.0)

    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_ids", _validate_unique_strings("protocol_ids", self.protocol_ids))
        object.__setattr__(self, "module_keys", _validate_unique_strings("module_keys", self.module_keys))

        if len(self.protocol_ids) != 3:
            raise ValueError("protocol_ids must have length 3")
        if len(self.module_keys) != 3:
            raise ValueError("module_keys must have length 3")
        if len(self.similarities_to_target) != 3:
            raise ValueError("similarities_to_target must have length 3")
        if len(self.anchor_seeds) != 2:
            raise ValueError("anchor_seeds must have length 2")
        if len(self.source_noise_stds) != 2:
            raise ValueError("source_noise_stds must have length 2")

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        for i, sim in enumerate(self.similarities_to_target):
            _validate_probability(f"similarities_to_target[{i}]", sim)

        if float(self.similarities_to_target[2]) != 1.0:
            raise ValueError(
                "similarities_to_target[2] must be 1.0 because protocol 3 is the target"
            )

        if self.target_noise_std < 0.0:
            raise ValueError(
                f"target_noise_std must be non-negative, got {self.target_noise_std}"
            )
        for i, noise_std in enumerate(self.source_noise_stds):
            if noise_std < 0.0:
                raise ValueError(
                    f"source_noise_stds[{i}] must be non-negative, got {noise_std}"
                )

        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LinearObserverFamily:
    """
    Concrete linear observer family for Problem 1_A.

    Attributes
    ----------
    observers_by_protocol:
        Maps protocol_id to its observer module.
    target_protocol_id:
        Protocol 3 by convention.
    config:
        Original family configuration.
    """

    observers_by_protocol: Dict[str, LinearObserverModule]
    target_protocol_id: str
    config: LinearObserverFamilyConfig
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        protocol_ids = tuple(self.observers_by_protocol.keys())
        expected_ids = self.config.protocol_ids

        if set(protocol_ids) != set(expected_ids):
            raise ValueError(
                "observers_by_protocol keys must match config.protocol_ids, got "
                f"{protocol_ids} vs {expected_ids}"
            )

        if self.target_protocol_id not in self.observers_by_protocol:
            raise ValueError(
                f"target_protocol_id {self.target_protocol_id!r} not found in observers_by_protocol"
            )

        object.__setattr__(self, "metadata", dict(self.metadata))

    def get(self, protocol_id: str) -> LinearObserverModule:
        try:
            return self.observers_by_protocol[protocol_id]
        except KeyError as exc:
            raise KeyError(f"Unknown protocol_id: {protocol_id!r}") from exc

    @property
    def target_observer(self) -> LinearObserverModule:
        return self.get(self.target_protocol_id)

    @property
    def source_protocol_ids(self) -> Tuple[str, str]:
        return tuple(pid for pid in self.config.protocol_ids if pid != self.target_protocol_id)

    @property
    def source_observers(self) -> Tuple[LinearObserverModule, LinearObserverModule]:
        return tuple(self.get(pid) for pid in self.source_protocol_ids)

    def similarities_to_target(self) -> Dict[str, float]:
        return {
            pid: float(sim)
            for pid, sim in zip(self.config.protocol_ids, self.config.similarities_to_target)
        }

    def parameter_distances_to_target(self, *, normalize: bool = True) -> Dict[str, float]:
        target = self.target_observer
        out: Dict[str, float] = {}
        for pid in self.config.protocol_ids:
            obs = self.get(pid)
            out[pid] = float(parameter_distance(obs, target, normalize=normalize))
        return out

    def summary(self) -> Dict[str, object]:
        return {
            "family_name": "linear_observers",
            "protocol_ids": self.config.protocol_ids,
            "module_keys": self.config.module_keys,
            "target_protocol_id": self.target_protocol_id,
            "similarities_to_target": self.similarities_to_target(),
            "normalized_parameter_distance_to_target": self.parameter_distances_to_target(
                normalize=True
            ),
        }


def _make_target_observer(
    *,
    module_key: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    weight_scale: float,
    bias_scale: float,
    activation: ActivationName,
    noise_std: float,
    metadata: Optional[Mapping[str, object]] = None,
) -> LinearObserverModule:
    return make_linear_observer(
        module_key=module_key,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
        activation=activation,
        noise_std=noise_std,
        metadata=metadata,
    )


def _make_anchor_observer(
    *,
    module_key: str,
    input_dim: int,
    output_dim: int,
    seed: int,
    weight_scale: float,
    bias_scale: float,
    activation: ActivationName,
    metadata: Optional[Mapping[str, object]] = None,
) -> LinearObserverModule:
    return make_linear_observer(
        module_key=module_key,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
        activation=activation,
        noise_std=0.0,
        metadata=metadata,
    )


def make_problem_1a_linear_observer_family(
    config: Optional[LinearObserverFamilyConfig] = None,
) -> LinearObserverFamily:
    """
    Build the canonical linear observer family for Problem 1_A.

    Construction
    ------------
    1. Sample the target observer for protocol 3.
    2. Sample one independent anchor observer for protocol 1.
    3. Sample one independent anchor observer for protocol 2.
    4. Build source observers by interpolating between the target observer and
       each source-specific anchor observer.

    Interpretation
    --------------
    If similarity_to_target is high, the source observer is close to the target
    observer, so transfer from that protocol should be easier.
    """
    config = LinearObserverFamilyConfig() if config is None else config

    protocol_1, protocol_2, protocol_3 = config.protocol_ids
    module_1, module_2, module_3 = config.module_keys
    sim_1, sim_2, sim_3 = config.similarities_to_target

    target = _make_target_observer(
        module_key=module_3,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        seed=config.target_seed,
        weight_scale=config.weight_scale,
        bias_scale=config.bias_scale,
        activation=config.activation,
        noise_std=config.target_noise_std,
        metadata={
            "observer_role": "target",
            "protocol_id": protocol_3,
            "similarity_to_target": sim_3,
        },
    )

    anchor_1 = _make_anchor_observer(
        module_key=f"{module_1}_anchor",
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        seed=config.anchor_seeds[0],
        weight_scale=config.weight_scale,
        bias_scale=config.bias_scale,
        activation=config.activation,
        metadata={
            "observer_role": "anchor",
            "protocol_id": protocol_1,
        },
    )
    anchor_2 = _make_anchor_observer(
        module_key=f"{module_2}_anchor",
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        seed=config.anchor_seeds[1],
        weight_scale=config.weight_scale,
        bias_scale=config.bias_scale,
        activation=config.activation,
        metadata={
            "observer_role": "anchor",
            "protocol_id": protocol_2,
        },
    )

    source_1 = blend_observers(
        target,
        anchor_1,
        similarity_to_a=sim_1,
        module_key=module_1,
        noise_std=config.source_noise_stds[0],
        metadata={
            "observer_role": "source",
            "protocol_id": protocol_1,
            "similarity_to_target": sim_1,
            "anchor_module_key": anchor_1.module_key,
        },
    )
    source_2 = blend_observers(
        target,
        anchor_2,
        similarity_to_a=sim_2,
        module_key=module_2,
        noise_std=config.source_noise_stds[1],
        metadata={
            "observer_role": "source",
            "protocol_id": protocol_2,
            "similarity_to_target": sim_2,
            "anchor_module_key": anchor_2.module_key,
        },
    )

    family = LinearObserverFamily(
        observers_by_protocol={
            protocol_1: source_1,
            protocol_2: source_2,
            protocol_3: target,
        },
        target_protocol_id=protocol_3,
        config=config,
        metadata={
            "family_name": "linear_observers",
            "construction": "target_plus_anchor_interpolation",
        },
    )
    return family


def make_problem_1a_linear_observer_family_from_similarities(
    *,
    similarity_p1_to_p3: float,
    similarity_p2_to_p3: float,
    input_dim: int,
    output_dim: int = 1,
    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3"),
    module_keys: Tuple[str, str, str] = ("observer_1", "observer_2", "observer_3"),
    target_seed: int = 0,
    anchor_seeds: Tuple[int, int] = (101, 202),
    weight_scale: float = 1.0,
    bias_scale: float = 0.25,
    activation: ActivationName = "identity",
    target_noise_std: float = 0.0,
    source_noise_stds: Tuple[float, float] = (0.0, 0.0),
) -> LinearObserverFamily:
    """
    Convenience constructor for similarity sweeps in Problem 1_A.
    """
    config = LinearObserverFamilyConfig(
        protocol_ids=protocol_ids,
        module_keys=module_keys,
        input_dim=input_dim,
        output_dim=output_dim,
        similarities_to_target=(
            _validate_probability("similarity_p1_to_p3", similarity_p1_to_p3),
            _validate_probability("similarity_p2_to_p3", similarity_p2_to_p3),
            1.0,
        ),
        target_seed=target_seed,
        anchor_seeds=anchor_seeds,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
        activation=activation,
        target_noise_std=target_noise_std,
        source_noise_stds=source_noise_stds,
    )
    return make_problem_1a_linear_observer_family(config=config)


def iter_problem_1a_linear_similarity_grid(
    similarity_pairs: Sequence[Tuple[float, float]],
    *,
    input_dim: int,
    output_dim: int = 1,
    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3"),
    module_keys: Tuple[str, str, str] = ("observer_1", "observer_2", "observer_3"),
    target_seed: int = 0,
    anchor_seeds: Tuple[int, int] = (101, 202),
    weight_scale: float = 1.0,
    bias_scale: float = 0.25,
    activation: ActivationName = "identity",
    target_noise_std: float = 0.0,
    source_noise_stds: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[LinearObserverFamily, ...]:
    """
    Build multiple linear observer families for a similarity sweep.
    """
    out = []
    for sim_1, sim_2 in similarity_pairs:
        out.append(
            make_problem_1a_linear_observer_family_from_similarities(
                similarity_p1_to_p3=sim_1,
                similarity_p2_to_p3=sim_2,
                input_dim=input_dim,
                output_dim=output_dim,
                protocol_ids=protocol_ids,
                module_keys=module_keys,
                target_seed=target_seed,
                anchor_seeds=anchor_seeds,
                weight_scale=weight_scale,
                bias_scale=bias_scale,
                activation=activation,
                target_noise_std=target_noise_std,
                source_noise_stds=source_noise_stds,
            )
        )
    return tuple(out)


__all__ = [
    "LinearObserverFamilyConfig",
    "LinearObserverFamily",
    "make_problem_1a_linear_observer_family",
    "make_problem_1a_linear_observer_family_from_similarities",
    "iter_problem_1a_linear_similarity_grid",
]