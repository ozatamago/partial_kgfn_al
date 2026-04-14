#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Literal


ProtocolValue = Any
Metadata = Dict[str, Any]

ProcessRole = Literal["source", "module", "observer", "sink"]
ObservationKind = Literal[
    "target_observer",
    "non_target_observer",
    "intermediate",
    "full_protocol",
]


def _as_tuple(xs: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if xs is None:
        return ()
    return tuple(xs)


def _as_dict(d: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if d is None:
        return {}
    return dict(d)


def _validate_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _validate_unique_strings(name: str, values: Sequence[str]) -> None:
    seen = set()
    dup = []
    for v in values:
        _validate_nonempty_str(name, v)
        if v in seen:
            dup.append(v)
        seen.add(v)
    if dup:
        raise ValueError(f"{name} contains duplicates: {dup}")


@dataclass(frozen=True)
class ProcessSpec:
    """
    Static definition of one process node inside a protocol graph.

    Notes
    -----
    - process_id: unique node id inside one protocol
    - process_type: reusable semantic type, such as "shared_upstream"
      or "observer_rgb"
    - module_key: key used by module_registry.py; defaults to process_type
    - parent_ids: upstream processes that must be executed first
    - input_keys / output_keys: symbolic I/O names for this process
    - observable_output_keys: subset of output_keys that may be queried
    """

    process_id: str
    process_type: str
    input_keys: Tuple[str, ...] = ()
    output_keys: Tuple[str, ...] = ()
    parent_ids: Tuple[str, ...] = ()
    role: ProcessRole = "module"
    module_key: Optional[str] = None
    cost: float = 1.0
    trainable: bool = True
    observable_output_keys: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("process_id", self.process_id)
        _validate_nonempty_str("process_type", self.process_type)

        object.__setattr__(self, "input_keys", _as_tuple(self.input_keys))
        object.__setattr__(self, "output_keys", _as_tuple(self.output_keys))
        object.__setattr__(self, "parent_ids", _as_tuple(self.parent_ids))
        object.__setattr__(self, "tags", _as_tuple(self.tags))
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        _validate_unique_strings("input_keys", self.input_keys)
        _validate_unique_strings("output_keys", self.output_keys)
        _validate_unique_strings("parent_ids", self.parent_ids)
        _validate_unique_strings("tags", self.tags)

        if self.cost < 0.0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")

        if self.module_key is None:
            object.__setattr__(self, "module_key", self.process_type)
        else:
            _validate_nonempty_str("module_key", self.module_key)

        if len(self.output_keys) == 0:
            raise ValueError(
                f"Process {self.process_id!r} must expose at least one output key."
            )

        if len(self.observable_output_keys) == 0 and self.role == "observer":
            object.__setattr__(self, "observable_output_keys", self.output_keys)
        else:
            object.__setattr__(
                self,
                "observable_output_keys",
                _as_tuple(self.observable_output_keys),
            )
            _validate_unique_strings(
                "observable_output_keys", self.observable_output_keys
            )
            unknown = set(self.observable_output_keys) - set(self.output_keys)
            if unknown:
                raise ValueError(
                    f"Process {self.process_id!r} has observable_output_keys "
                    f"not present in output_keys: {sorted(unknown)}"
                )

    @property
    def is_observer(self) -> bool:
        return self.role == "observer"

    @property
    def is_sink(self) -> bool:
        return self.role == "sink"

    def can_observe(self, output_key: str) -> bool:
        return output_key in self.observable_output_keys


@dataclass(frozen=True)
class ProtocolSpec:
    """
    Static definition of one protocol graph.

    This class validates:
    - unique process ids
    - valid parent references
    - acyclicity
    - existence of the target process/output
    """

    protocol_id: str
    processes: Tuple[ProcessSpec, ...]
    target_process_id: str
    target_output_key: str
    condition_keys: Tuple[str, ...] = ()
    description: str = ""
    tags: Tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("protocol_id", self.protocol_id)
        _validate_nonempty_str("target_process_id", self.target_process_id)
        _validate_nonempty_str("target_output_key", self.target_output_key)

        object.__setattr__(self, "processes", tuple(self.processes))
        object.__setattr__(self, "condition_keys", _as_tuple(self.condition_keys))
        object.__setattr__(self, "tags", _as_tuple(self.tags))
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        if len(self.processes) == 0:
            raise ValueError("processes must be non-empty")

        process_ids = [p.process_id for p in self.processes]
        _validate_unique_strings("process_ids", process_ids)
        _validate_unique_strings("condition_keys", self.condition_keys)
        _validate_unique_strings("tags", self.tags)

        proc_map = {p.process_id: p for p in self.processes}
        for p in self.processes:
            unknown_parents = set(p.parent_ids) - set(proc_map.keys())
            if unknown_parents:
                raise ValueError(
                    f"Process {p.process_id!r} references unknown parent_ids: "
                    f"{sorted(unknown_parents)}"
                )

        if self.target_process_id not in proc_map:
            raise ValueError(
                f"target_process_id {self.target_process_id!r} is not in processes"
            )

        target_proc = proc_map[self.target_process_id]
        if self.target_output_key not in target_proc.output_keys:
            raise ValueError(
                f"target_output_key {self.target_output_key!r} is not in "
                f"output_keys of target process {self.target_process_id!r}"
            )

        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        indeg = {p.process_id: 0 for p in self.processes}
        children = {p.process_id: [] for p in self.processes}

        for p in self.processes:
            for parent_id in p.parent_ids:
                indeg[p.process_id] += 1
                children[parent_id].append(p.process_id)

        q = deque([pid for pid, d in indeg.items() if d == 0])
        seen = 0

        while q:
            pid = q.popleft()
            seen += 1
            for child_id in children[pid]:
                indeg[child_id] -= 1
                if indeg[child_id] == 0:
                    q.append(child_id)

        if seen != len(self.processes):
            raise ValueError(
                f"Protocol {self.protocol_id!r} must be a DAG, but a cycle was detected."
            )

    @property
    def process_ids(self) -> Tuple[str, ...]:
        return tuple(p.process_id for p in self.processes)

    @property
    def process_map(self) -> Dict[str, ProcessSpec]:
        return {p.process_id: p for p in self.processes}

    def get_process(self, process_id: str) -> ProcessSpec:
        try:
            return self.process_map[process_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown process_id {process_id!r} in protocol {self.protocol_id!r}"
            ) from exc

    def topological_order(self) -> Tuple[str, ...]:
        indeg = {p.process_id: 0 for p in self.processes}
        children = {p.process_id: [] for p in self.processes}

        for p in self.processes:
            for parent_id in p.parent_ids:
                indeg[p.process_id] += 1
                children[parent_id].append(p.process_id)

        q = deque([pid for pid, d in indeg.items() if d == 0])
        order: List[str] = []

        while q:
            pid = q.popleft()
            order.append(pid)
            for child_id in children[pid]:
                indeg[child_id] -= 1
                if indeg[child_id] == 0:
                    q.append(child_id)

        if len(order) != len(self.processes):
            raise RuntimeError(
                f"Failed to produce topological order for protocol {self.protocol_id!r}"
            )
        return tuple(order)

    def observer_process_ids(self) -> Tuple[str, ...]:
        return tuple(p.process_id for p in self.processes if p.is_observer)

    def list_observation_specs(
        self,
        cost_map: Optional[Mapping[Tuple[str, str], float]] = None,
    ) -> Tuple["ObservationSpec", ...]:
        """
        Enumerate all observable outputs in the protocol.

        cost_map keys are (process_id, output_key).
        """
        cost_map = {} if cost_map is None else dict(cost_map)
        specs: List[ObservationSpec] = []

        for proc in self.processes:
            for output_key in proc.observable_output_keys:
                is_target = (
                    proc.process_id == self.target_process_id
                    and output_key == self.target_output_key
                )

                if is_target:
                    kind: ObservationKind = "target_observer"
                elif proc.is_observer:
                    kind = "non_target_observer"
                else:
                    kind = "intermediate"

                specs.append(
                    ObservationSpec(
                        protocol_id=self.protocol_id,
                        process_id=proc.process_id,
                        output_key=output_key,
                        kind=kind,
                        cost=float(cost_map.get((proc.process_id, output_key), proc.cost)),
                        is_target=is_target,
                    )
                )
        return tuple(specs)


@dataclass(frozen=True)
class ConditionSpec:
    """
    External condition or protocol input assignment.

    Examples
    --------
    values = {
        "x0": 0.1,
        "x1": 0.7,
        "temperature": 25.0,
    }
    """

    condition_id: str
    values: Metadata
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("condition_id", self.condition_id)
        object.__setattr__(self, "values", _as_dict(self.values))
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass(frozen=True)
class ObservationSpec:
    """
    Static description of one observable quantity that may be queried by
    active learning.
    """

    protocol_id: str
    process_id: str
    output_key: str
    kind: ObservationKind
    cost: float = 1.0
    is_target: bool = False
    description: str = ""
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("protocol_id", self.protocol_id)
        _validate_nonempty_str("process_id", self.process_id)
        _validate_nonempty_str("output_key", self.output_key)
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        if self.cost < 0.0:
            raise ValueError(f"Observation cost must be non-negative, got {self.cost}")

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.protocol_id, self.process_id, self.output_key)


@dataclass(frozen=True)
class AcquisitionCandidate:
    """
    One candidate action for protocol-aware active learning.

    In the first implementation stage, the action can be interpreted as:
    - choose one protocol
    - choose one condition
    - choose one observable quantity
    """

    candidate_id: str
    protocol_id: str
    condition_id: Optional[str]
    observation: ObservationSpec
    expected_cost: float
    score: Optional[float] = None
    score_name: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("candidate_id", self.candidate_id)
        _validate_nonempty_str("protocol_id", self.protocol_id)
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        if self.condition_id is not None:
            _validate_nonempty_str("condition_id", self.condition_id)

        if self.expected_cost < 0.0:
            raise ValueError(
                f"expected_cost must be non-negative, got {self.expected_cost}"
            )


@dataclass
class ProcessExecutionRecord:
    """
    Runtime record for one executed process.
    """

    process_id: str
    process_type: str
    inputs: Metadata = field(default_factory=dict)
    outputs: Metadata = field(default_factory=dict)
    success: bool = True
    message: str = ""
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("process_id", self.process_id)
        _validate_nonempty_str("process_type", self.process_type)
        self.inputs = _as_dict(self.inputs)
        self.outputs = _as_dict(self.outputs)
        self.metadata = _as_dict(self.metadata)


@dataclass
class ExecutionTrace:
    """
    Runtime trace for one protocol execution under one condition.
    """

    protocol_id: str
    condition_id: Optional[str]
    records: List[ProcessExecutionRecord] = field(default_factory=list)
    final_outputs: Metadata = field(default_factory=dict)
    target_value: Any = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("protocol_id", self.protocol_id)
        if self.condition_id is not None:
            _validate_nonempty_str("condition_id", self.condition_id)
        self.final_outputs = _as_dict(self.final_outputs)
        self.metadata = _as_dict(self.metadata)

    @property
    def record_map(self) -> Dict[str, ProcessExecutionRecord]:
        return {r.process_id: r for r in self.records}

    def get_record(self, process_id: str) -> ProcessExecutionRecord:
        try:
            return self.record_map[process_id]
        except KeyError as exc:
            raise KeyError(
                f"ExecutionTrace has no record for process_id {process_id!r}"
            ) from exc

    def get_output(self, process_id: str, output_key: str) -> Any:
        record = self.get_record(process_id)
        if output_key not in record.outputs:
            raise KeyError(
                f"Process {process_id!r} has no output_key {output_key!r} in this trace"
            )
        return record.outputs[output_key]


@dataclass(frozen=True)
class ProtocolObservation:
    """
    One realized observation collected from a protocol execution.
    """

    protocol_id: str
    condition_id: str
    process_id: str
    output_key: str
    value: Any
    cost: float = 1.0
    is_target: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("protocol_id", self.protocol_id)
        _validate_nonempty_str("condition_id", self.condition_id)
        _validate_nonempty_str("process_id", self.process_id)
        _validate_nonempty_str("output_key", self.output_key)
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        if self.cost < 0.0:
            raise ValueError(f"Observation cost must be non-negative, got {self.cost}")


@dataclass(frozen=True)
class BenchmarkSample:
    """
    One sampled datum used by training or evaluation.

    This keeps the file independent from torch so that dataset builders,
    metrics, and runners may all import it safely.
    """

    sample_id: str
    protocol_id: str
    condition: ConditionSpec
    target_value: Any
    observations: Tuple[ProtocolObservation, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("sample_id", self.sample_id)
        _validate_nonempty_str("protocol_id", self.protocol_id)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "metadata", _as_dict(self.metadata))


@dataclass(frozen=True)
class DatasetSplit:
    """
    Named split over sample ids.

    Examples
    --------
    split_name = "pretrain"
    split_name = "adapt"
    split_name = "test"
    """

    split_name: str
    sample_ids: Tuple[str, ...]
    protocol_ids: Tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_nonempty_str("split_name", self.split_name)
        object.__setattr__(self, "sample_ids", _as_tuple(self.sample_ids))
        object.__setattr__(self, "protocol_ids", _as_tuple(self.protocol_ids))
        object.__setattr__(self, "metadata", _as_dict(self.metadata))

        _validate_unique_strings("sample_ids", self.sample_ids)
        _validate_unique_strings("protocol_ids", self.protocol_ids)


def make_target_observation_spec(
    protocol: ProtocolSpec,
    *,
    cost: Optional[float] = None,
) -> ObservationSpec:
    """
    Convenience helper for the most common acquisition target.
    """
    target_proc = protocol.get_process(protocol.target_process_id)
    return ObservationSpec(
        protocol_id=protocol.protocol_id,
        process_id=protocol.target_process_id,
        output_key=protocol.target_output_key,
        kind="target_observer",
        cost=float(target_proc.cost if cost is None else cost),
        is_target=True,
    )


def protocol_observation_index(
    observations: Sequence[ProtocolObservation],
) -> Dict[Tuple[str, str, str, str], ProtocolObservation]:
    """
    Map (protocol_id, condition_id, process_id, output_key) to the realized observation.
    """
    out: Dict[Tuple[str, str, str, str], ProtocolObservation] = {}
    for obs in observations:
        key = (obs.protocol_id, obs.condition_id, obs.process_id, obs.output_key)
        if key in out:
            raise ValueError(f"Duplicate realized observation key detected: {key}")
        out[key] = obs
    return out