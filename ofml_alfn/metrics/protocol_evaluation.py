#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ofml_alfn.training.train_protocol_predictor import (
    ProtocolEvaluationResult,
    ProtocolTrainingConfig,
    evaluate_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


@dataclass(frozen=True)
class TargetProtocolValidationResult:
    """
    Validation result specialized for one target protocol.

    Attributes
    ----------
    target_protocol_id:
        The protocol id that was evaluated.
    loss:
        Aggregate validation loss on that target protocol.
    n_samples:
        Number of validation samples used.
    loss_by_protocol:
        Kept for debugging / logging consistency with the generic evaluator.
    n_by_protocol:
        Kept for debugging / logging consistency with the generic evaluator.
    """

    target_protocol_id: str
    loss: float
    n_samples: int
    loss_by_protocol: Dict[str, float]
    n_by_protocol: Dict[str, int]


def _normalize_protocol_map(
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
) -> Dict[str, ProtocolSpec]:
    if isinstance(protocols, Mapping):
        out = dict(protocols)
    else:
        out = {p.protocol_id: p for p in protocols}

    if len(out) == 0:
        raise ValueError("protocols must be non-empty")

    return out


def _filter_samples_by_protocol_id(
    samples: Sequence[BenchmarkSample],
    *,
    protocol_id: str,
) -> List[BenchmarkSample]:
    return [s for s in samples if s.protocol_id == protocol_id]


def _filter_samples_by_protocol_ids(
    samples: Sequence[BenchmarkSample],
    *,
    protocol_ids: Iterable[str],
) -> List[BenchmarkSample]:
    protocol_id_set = set(protocol_ids)
    return [s for s in samples if s.protocol_id in protocol_id_set]


def _require_target_protocol(
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    *,
    target_protocol_id: str,
) -> Dict[str, ProtocolSpec]:
    protocol_map = _normalize_protocol_map(protocols)
    if target_protocol_id not in protocol_map:
        raise KeyError(
            f"Unknown target_protocol_id {target_protocol_id!r}. "
            f"Available protocol ids: {sorted(protocol_map.keys())}"
        )
    return protocol_map


def evaluate_target_protocol_validation(
    predictor,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    val_samples: Sequence[BenchmarkSample],
    target_protocol_id: str,
    config: Optional[ProtocolTrainingConfig] = None,
) -> TargetProtocolValidationResult:
    """
    Evaluate validation loss only on the designated target protocol.

    This is the main entry point to use inside fantasy acquisition:
    for each fantasy-updated model, compute how much the target protocol's
    validation loss improved.
    """
    protocol_map = _require_target_protocol(
        protocols,
        target_protocol_id=target_protocol_id,
    )

    target_val_samples = _filter_samples_by_protocol_id(
        val_samples,
        protocol_id=target_protocol_id,
    )

    if len(target_val_samples) == 0:
        return TargetProtocolValidationResult(
            target_protocol_id=target_protocol_id,
            loss=float("nan"),
            n_samples=0,
            loss_by_protocol={},
            n_by_protocol={},
        )

    generic_result: ProtocolEvaluationResult = evaluate_protocol_predictor(
        predictor,
        protocols={target_protocol_id: protocol_map[target_protocol_id]},
        samples=target_val_samples,
        config=config,
    )

    return TargetProtocolValidationResult(
        target_protocol_id=target_protocol_id,
        loss=float(generic_result.loss),
        n_samples=int(generic_result.n_total),
        loss_by_protocol=dict(generic_result.loss_by_protocol),
        n_by_protocol=dict(generic_result.n_by_protocol),
    )


def compute_target_validation_loss(
    predictor,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    val_samples: Sequence[BenchmarkSample],
    target_protocol_id: str,
    config: Optional[ProtocolTrainingConfig] = None,
) -> float:
    """
    Convenience wrapper that returns only the target validation loss.
    """
    result = evaluate_target_protocol_validation(
        predictor,
        protocols=protocols,
        val_samples=val_samples,
        target_protocol_id=target_protocol_id,
        config=config,
    )
    return float(result.loss)


def evaluate_protocol_subset_validation(
    predictor,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    val_samples: Sequence[BenchmarkSample],
    protocol_ids: Sequence[str],
    config: Optional[ProtocolTrainingConfig] = None,
) -> ProtocolEvaluationResult:
    """
    Generic subset evaluator.

    Useful later for:
    - source-only validation
    - target-only validation
    - target + selected auxiliaries
    """
    if len(protocol_ids) == 0:
        return ProtocolEvaluationResult(
            loss=float("nan"),
            loss_by_protocol={},
            n_by_protocol={},
            n_total=0,
        )

    protocol_map = _normalize_protocol_map(protocols)

    missing = [pid for pid in protocol_ids if pid not in protocol_map]
    if len(missing) > 0:
        raise KeyError(
            f"Unknown protocol ids in protocol_ids: {missing}. "
            f"Available protocol ids: {sorted(protocol_map.keys())}"
        )

    subset_protocols = {pid: protocol_map[pid] for pid in protocol_ids}
    subset_samples = _filter_samples_by_protocol_ids(
        val_samples,
        protocol_ids=protocol_ids,
    )

    if len(subset_samples) == 0:
        return ProtocolEvaluationResult(
            loss=float("nan"),
            loss_by_protocol={},
            n_by_protocol={},
            n_total=0,
        )

    return evaluate_protocol_predictor(
        predictor,
        protocols=subset_protocols,
        samples=subset_samples,
        config=config,
    )


def compute_target_validation_improvement(
    *,
    current_loss: float,
    updated_loss: float,
) -> float:
    """
    Positive means improvement.
    """
    return float(current_loss - updated_loss)


def compute_cost_normalized_target_validation_improvement(
    *,
    current_loss: float,
    updated_loss: float,
    acquisition_cost: float,
    eps: float = 1e-12,
) -> float:
    """
    Improvement divided by acquisition cost.

    This matches the fantasy acquisition score:
        score = (target validation loss reduction) / protocol cost
    """
    if acquisition_cost < 0.0:
        raise ValueError(
            f"acquisition_cost must be non-negative, got {acquisition_cost}"
        )
    denom = max(float(acquisition_cost), float(eps))
    return compute_target_validation_improvement(
        current_loss=current_loss,
        updated_loss=updated_loss,
    ) / denom


__all__ = [
    "TargetProtocolValidationResult",
    "evaluate_target_protocol_validation",
    "compute_target_validation_loss",
    "evaluate_protocol_subset_validation",
    "compute_target_validation_improvement",
    "compute_cost_normalized_target_validation_improvement",
]