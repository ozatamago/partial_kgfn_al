#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ofml_alfn.utils.protocol_types import ProcessSpec, ProtocolSpec


def _validate_similarity(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _as_tuple(xs: Sequence[str]) -> Tuple[str, ...]:
    return tuple(xs)


@dataclass(frozen=True)
class Problem1AEllipsoidConfig:
    """
    Configuration for Problem 1_A Ellipsoid variant.

    Problem 1_A Ellipsoid:
    - All protocols share the same ellipsoid-chart upstream module.
    - Only the observer differs across protocols.
    - Only the observer output is observable.
    - Protocol 3 is the target protocol.
    - Input is 3D: (x0, x1, x2)
    - Shared latent is 4D: z
    - Observer outputs are protocol-specific projections:
        protocol_1: y1 = (z1, z4)
        protocol_2: y2 = (z2, z3, z4)
        protocol_3: y3 = (z1, z2, z3)
    """

    protocol_ids: Tuple[str, str, str] = ("protocol_1", "protocol_2", "protocol_3")
    condition_keys: Tuple[str, ...] = ("x0", "x1", "x2")

    upstream_output_key: str = "z"
    observer_output_key: str = "y"

    upstream_module_key: str = "shared_upstream_ellipsoid"
    observer_module_keys: Tuple[str, str, str] = (
        "observer_1_ellipsoid",
        "observer_2_ellipsoid",
        "observer_3_ellipsoid",
    )

    upstream_cost: float = 0.0
    observer_costs: Tuple[float, float, float] = (1.0, 2.0, 3.0)
    similarity_to_target: Tuple[float, float, float] = (0.4, 0.7, 1.0)

    description: str = (
        "Problem 1_A Ellipsoid: shared ellipsoid-chart upstream module, "
        "protocol-specific fixed projection observers."
    )

    def __post_init__(self) -> None:
        if len(self.protocol_ids) != 3:
            raise ValueError("protocol_ids must have length 3")
        if len(self.observer_module_keys) != 3:
            raise ValueError("observer_module_keys must have length 3")
        if len(self.observer_costs) != 3:
            raise ValueError("observer_costs must have length 3")
        if len(self.similarity_to_target) != 3:
            raise ValueError("similarity_to_target must have length 3")

        for i, sim in enumerate(self.similarity_to_target):
            _validate_similarity(f"similarity_to_target[{i}]", sim)

        if self.similarity_to_target[2] != 1.0:
            raise ValueError(
                "For Problem 1_A Ellipsoid, the target protocol observer must have similarity 1.0 to itself."
            )

        if len(set(self.protocol_ids)) != 3:
            raise ValueError(f"protocol_ids must be unique, got {self.protocol_ids}")

        if len(set(self.observer_module_keys)) != 3:
            raise ValueError(
                f"observer_module_keys must be unique, got {self.observer_module_keys}"
            )

        if self.upstream_cost < 0.0:
            raise ValueError(f"upstream_cost must be non-negative, got {self.upstream_cost}")

        for i, c in enumerate(self.observer_costs):
            if c < 0.0:
                raise ValueError(f"observer_costs[{i}] must be non-negative, got {c}")

        if len(self.condition_keys) != 3:
            raise ValueError(
                "Problem 1_A Ellipsoid expects exactly 3 condition keys: (x0, x1, x2)"
            )


@dataclass(frozen=True)
class Problem1AEllipsoidBenchmark:
    """
    Concrete benchmark family for Problem 1_A Ellipsoid.

    Attributes
    ----------
    protocols:
        The three concrete protocols. Protocol 3 is the target protocol.
    pretrain_protocol_ids:
        Source protocols used for pretraining.
    target_protocol_id:
        Target protocol used for adaptation and evaluation.
    config:
        Original configuration used to build the benchmark.
    """

    protocols: Tuple[ProtocolSpec, ...]
    pretrain_protocol_ids: Tuple[str, str]
    target_protocol_id: str
    config: Problem1AEllipsoidConfig
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.protocols) != 3:
            raise ValueError("Problem1AEllipsoidBenchmark must contain exactly 3 protocols")

        protocol_ids = tuple(p.protocol_id for p in self.protocols)
        if len(set(protocol_ids)) != 3:
            raise ValueError(f"Duplicate protocol ids detected: {protocol_ids}")

        if self.target_protocol_id not in protocol_ids:
            raise ValueError(
                f"target_protocol_id {self.target_protocol_id!r} is not present in protocols"
            )

        for pid in self.pretrain_protocol_ids:
            if pid not in protocol_ids:
                raise ValueError(f"Unknown pretrain protocol id: {pid}")

        if self.target_protocol_id in self.pretrain_protocol_ids:
            raise ValueError(
                "target_protocol_id must not be included in pretrain_protocol_ids"
            )

    @property
    def protocol_map(self) -> Dict[str, ProtocolSpec]:
        return {p.protocol_id: p for p in self.protocols}

    def get_protocol(self, protocol_id: str) -> ProtocolSpec:
        try:
            return self.protocol_map[protocol_id]
        except KeyError as exc:
            raise KeyError(f"Unknown protocol_id: {protocol_id!r}") from exc

    @property
    def target_protocol(self) -> ProtocolSpec:
        return self.get_protocol(self.target_protocol_id)

    @property
    def pretrain_protocols(self) -> Tuple[ProtocolSpec, ...]:
        return tuple(self.get_protocol(pid) for pid in self.pretrain_protocol_ids)

    @property
    def all_protocol_ids(self) -> Tuple[str, ...]:
        return tuple(p.protocol_id for p in self.protocols)

    def summary(self) -> Dict[str, object]:
        return {
            "problem_name": "problem_1a_ellipsoid",
            "protocol_ids": self.all_protocol_ids,
            "pretrain_protocol_ids": self.pretrain_protocol_ids,
            "target_protocol_id": self.target_protocol_id,
            "condition_keys": self.config.condition_keys,
            "observer_module_keys": self.config.observer_module_keys,
            "similarity_to_target": self.config.similarity_to_target,
            "shared_latent_dim": 4,
            "observer_projection_dims": {
                self.config.protocol_ids[0]: 2,
                self.config.protocol_ids[1]: 3,
                self.config.protocol_ids[2]: 3,
            },
        }


def _make_shared_upstream_process(
    *,
    condition_keys: Sequence[str],
    output_key: str,
    module_key: str,
    cost: float,
) -> ProcessSpec:
    return ProcessSpec(
        process_id="S1",
        process_type="shared_upstream_ellipsoid",
        input_keys=_as_tuple(condition_keys),
        output_keys=(output_key,),
        parent_ids=(),
        role="module",
        module_key=module_key,
        cost=cost,
        trainable=True,
        observable_output_keys=(),
        metadata={
            "semantic_role": "shared_upstream",
            "shared_across_protocols": True,
            "latent_semantics": "ellipsoid_chart",
            "input_dim": 3,
            "latent_dim": 4,
        },
    )


def _make_observer_process(
    *,
    process_id: str,
    input_key: str,
    output_key: str,
    observer_type: str,
    module_key: str,
    cost: float,
    observer_index: int,
    similarity_to_target: float,
    projected_z_indices: Sequence[int],
) -> ProcessSpec:
    return ProcessSpec(
        process_id=process_id,
        process_type=observer_type,
        input_keys=(input_key,),
        output_keys=(output_key,),
        parent_ids=("S1",),
        role="observer",
        module_key=module_key,
        cost=cost,
        trainable=True,
        observable_output_keys=(output_key,),
        metadata={
            "semantic_role": "observer",
            "observer_index": observer_index,
            "similarity_to_target": similarity_to_target,
            "projection_type": "fixed_projection",
            "projected_z_indices": tuple(int(i) for i in projected_z_indices),
            "projected_output_dim": len(tuple(projected_z_indices)),
        },
    )


def _make_two_node_protocol(
    *,
    protocol_id: str,
    condition_keys: Sequence[str],
    upstream_output_key: str,
    observer_output_key: str,
    upstream_module_key: str,
    observer_module_key: str,
    upstream_cost: float,
    observer_cost: float,
    observer_index: int,
    similarity_to_target: float,
    projected_z_indices: Sequence[int],
) -> ProtocolSpec:
    upstream = _make_shared_upstream_process(
        condition_keys=condition_keys,
        output_key=upstream_output_key,
        module_key=upstream_module_key,
        cost=upstream_cost,
    )

    observer = _make_observer_process(
        process_id="S2",
        input_key=upstream_output_key,
        output_key=observer_output_key,
        observer_type=f"observer_type_{observer_index}_ellipsoid",
        module_key=observer_module_key,
        cost=observer_cost,
        observer_index=observer_index,
        similarity_to_target=similarity_to_target,
        projected_z_indices=projected_z_indices,
    )

    return ProtocolSpec(
        protocol_id=protocol_id,
        processes=(upstream, observer),
        target_process_id="S2",
        target_output_key=observer_output_key,
        condition_keys=_as_tuple(condition_keys),
        description=(
            f"Problem 1_A Ellipsoid protocol {observer_index}: "
            f"shared ellipsoid-chart upstream with protocol-specific observer {observer_index}."
        ),
        tags=("problem_1a_ellipsoid", "shared_upstream", "observer_shift", "ellipsoid"),
        metadata={
            "problem_name": "problem_1a_ellipsoid",
            "observer_index": observer_index,
            "similarity_to_target": similarity_to_target,
            "upstream_module_key": upstream_module_key,
            "observer_module_key": observer_module_key,
            "projected_z_indices": tuple(int(i) for i in projected_z_indices),
        },
    )


def make_problem_1a_ellipsoid(
    config: Optional[Problem1AEllipsoidConfig] = None,
) -> Problem1AEllipsoidBenchmark:
    """
    Build the canonical Problem 1_A Ellipsoid benchmark family.

    Returns
    -------
    Problem1AEllipsoidBenchmark
        Contains Protocols 1, 2, and 3, where Protocol 3 is the target.

    Projection design
    -----------------
    protocol_1: y1 = (z1, z4)
    protocol_2: y2 = (z2, z3, z4)
    protocol_3: y3 = (z1, z2, z3)
    """
    config = Problem1AEllipsoidConfig() if config is None else config

    projection_tuples: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]] = (
        (0, 3),      # (z1, z4)
        (1, 2, 3),   # (z2, z3, z4)
        (0, 1, 2),   # (z1, z2, z3)
    )

    protocols: List[ProtocolSpec] = []
    for i, protocol_id in enumerate(config.protocol_ids):
        protocols.append(
            _make_two_node_protocol(
                protocol_id=protocol_id,
                condition_keys=config.condition_keys,
                upstream_output_key=config.upstream_output_key,
                observer_output_key=config.observer_output_key,
                upstream_module_key=config.upstream_module_key,
                observer_module_key=config.observer_module_keys[i],
                upstream_cost=config.upstream_cost,
                observer_cost=config.observer_costs[i],
                observer_index=i + 1,
                similarity_to_target=config.similarity_to_target[i],
                projected_z_indices=projection_tuples[i],
            )
        )

    return Problem1AEllipsoidBenchmark(
        protocols=tuple(protocols),
        pretrain_protocol_ids=(config.protocol_ids[0], config.protocol_ids[1]),
        target_protocol_id=config.protocol_ids[2],
        config=config,
        metadata={
            "problem_name": "problem_1a_ellipsoid",
            "setting": "shared_ellipsoid_upstream_different_fixed_projection_observers",
            "projection_tuples": projection_tuples,
        },
    )


def make_problem_1a_ellipsoid_from_similarity(
    *,
    similarity_p1_to_p3: float,
    similarity_p2_to_p3: float,
    condition_keys: Sequence[str] = ("x0", "x1", "x2"),
    observer_costs: Sequence[float] = (1.0, 2.0, 3.0),
) -> Problem1AEllipsoidBenchmark:
    """
    Convenience constructor used by similarity sweeps.

    Protocol 3 is always the target, so its similarity to itself is fixed at 1.0.
    """
    config = Problem1AEllipsoidConfig(
        condition_keys=_as_tuple(condition_keys),
        observer_costs=tuple(float(c) for c in observer_costs),
        similarity_to_target=(
            _validate_similarity("similarity_p1_to_p3", similarity_p1_to_p3),
            _validate_similarity("similarity_p2_to_p3", similarity_p2_to_p3),
            1.0,
        ),
    )
    return make_problem_1a_ellipsoid(config=config)


def iter_problem_1a_ellipsoid_similarity_grid(
    similarity_pairs: Iterable[Tuple[float, float]],
    *,
    condition_keys: Sequence[str] = ("x0", "x1", "x2"),
    observer_costs: Sequence[float] = (1.0, 2.0, 3.0),
) -> Tuple[Problem1AEllipsoidBenchmark, ...]:
    """
    Build a batch of Problem 1_A Ellipsoid benchmark families for a similarity sweep.
    """
    out: List[Problem1AEllipsoidBenchmark] = []
    for sim13, sim23 in similarity_pairs:
        out.append(
            make_problem_1a_ellipsoid_from_similarity(
                similarity_p1_to_p3=sim13,
                similarity_p2_to_p3=sim23,
                condition_keys=condition_keys,
                observer_costs=observer_costs,
            )
        )
    return tuple(out)


__all__ = [
    "Problem1AEllipsoidConfig",
    "Problem1AEllipsoidBenchmark",
    "make_problem_1a_ellipsoid",
    "make_problem_1a_ellipsoid_from_similarity",
    "iter_problem_1a_ellipsoid_similarity_grid",
]