#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


@dataclass(frozen=True)
class ProtocolQueryCandidate:
    """
    One acquirable protocol query candidate.

    In the current Problem 1A setting, this is effectively a wrapper around one
    BenchmarkSample in the candidate pool, together with its protocol metadata
    and acquisition cost.
    """
    candidate_id: str
    sample: BenchmarkSample
    protocol_id: str
    acquisition_cost: float
    is_target_protocol: bool
    metadata: Dict[str, object]


def _normalize_protocol_map(
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
) -> Dict[str, ProtocolSpec]:
    if isinstance(protocols, Mapping):
        protocol_map = dict(protocols)
    else:
        protocol_map = {p.protocol_id: p for p in protocols}

    if len(protocol_map) == 0:
        raise ValueError("protocols must be non-empty")
    return protocol_map


def _sample_cost(sample: BenchmarkSample) -> float:
    if len(sample.observations) > 0:
        return float(sample.observations[0].cost)
    if "observation_cost" in sample.metadata:
        return float(sample.metadata["observation_cost"])
    return 1.0


def make_protocol_query_candidate(
    sample: BenchmarkSample,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    target_protocol_id: Optional[str] = None,
) -> ProtocolQueryCandidate:
    protocol_map = _normalize_protocol_map(protocols)
    if sample.protocol_id not in protocol_map:
        raise KeyError(
            f"Unknown protocol_id {sample.protocol_id!r} in sample {sample.sample_id!r}. "
            f"Available protocols: {sorted(protocol_map.keys())}"
        )

    return ProtocolQueryCandidate(
        candidate_id=sample.sample_id,
        sample=sample,
        protocol_id=sample.protocol_id,
        acquisition_cost=_sample_cost(sample),
        is_target_protocol=(sample.protocol_id == target_protocol_id),
        metadata={
            "condition_id": sample.condition.condition_id,
            "n_observations": len(sample.observations),
        },
    )


def build_protocol_query_candidates(
    samples: Sequence[BenchmarkSample],
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    target_protocol_id: Optional[str] = None,
    protocol_ids: Optional[Iterable[str]] = None,
) -> List[ProtocolQueryCandidate]:
    protocol_map = _normalize_protocol_map(protocols)
    allowed_protocol_ids = None if protocol_ids is None else set(protocol_ids)

    out: List[ProtocolQueryCandidate] = []
    for sample in samples:
        if allowed_protocol_ids is not None and sample.protocol_id not in allowed_protocol_ids:
            continue
        out.append(
            make_protocol_query_candidate(
                sample,
                protocols=protocol_map,
                target_protocol_id=target_protocol_id,
            )
        )
    return out


def filter_affordable_protocol_candidates(
    candidates: Sequence[ProtocolQueryCandidate],
    *,
    remaining_budget: float,
) -> List[ProtocolQueryCandidate]:
    return [
        cand
        for cand in candidates
        if float(cand.acquisition_cost) <= float(remaining_budget)
    ]


def group_protocol_candidates_by_protocol(
    candidates: Sequence[ProtocolQueryCandidate],
) -> Dict[str, List[ProtocolQueryCandidate]]:
    out: Dict[str, List[ProtocolQueryCandidate]] = {}
    for cand in candidates:
        out.setdefault(cand.protocol_id, []).append(cand)
    return out


__all__ = [
    "ProtocolQueryCandidate",
    "make_protocol_query_candidate",
    "build_protocol_query_candidates",
    "filter_affordable_protocol_candidates",
    "group_protocol_candidates_by_protocol",
]