#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from ofml_alfn.metrics.protocol_evaluation import compute_target_validation_loss
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolTrainingConfig,
    ProtocolEvaluationResult,
    ProtocolTrainingResult,
    evaluate_protocol_predictor,
    train_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


@dataclass(frozen=True)
class PretrainThenAdaptConfig:
    target_protocol_id: str
    train_config: ProtocolTrainingConfig


@dataclass(frozen=True)
class PretrainThenAdaptStageRecord:
    stage: str
    n_train_samples: int
    target_val_loss: Optional[float]
    target_test_loss: Optional[float]
    eval_error: Optional[str]


@dataclass(frozen=True)
class PretrainThenAdaptResult:
    history: List[PretrainThenAdaptStageRecord]
    final_target_val_loss: Optional[float]
    final_target_test_loss: Optional[float]
    n_source_pretrain_samples: int
    n_target_adapt_samples: int


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


def _evaluate_target_test_loss(
    predictor: torch.nn.Module,
    *,
    target_protocol: ProtocolSpec,
    test_samples: Sequence[BenchmarkSample],
    config: ProtocolTrainingConfig,
) -> float:
    result = evaluate_protocol_predictor(
        predictor,
        protocols={target_protocol.protocol_id: target_protocol},
        samples=test_samples,
        config=config,
    )
    return float(result.loss)


def _safe_target_eval_after_stage(
    predictor: torch.nn.Module,
    *,
    protocol_map: Dict[str, ProtocolSpec],
    target_protocol_id: str,
    val_samples: Sequence[BenchmarkSample],
    test_samples: Sequence[BenchmarkSample],
    config: ProtocolTrainingConfig,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        target_val_loss = compute_target_validation_loss(
            predictor,
            protocols=protocol_map,
            val_samples=val_samples,
            target_protocol_id=target_protocol_id,
            config=config,
        )
        target_test_loss = _evaluate_target_test_loss(
            predictor,
            target_protocol=protocol_map[target_protocol_id],
            test_samples=test_samples,
            config=config,
        )
        return float(target_val_loss), float(target_test_loss), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def run_pretrain_then_adapt(
    *,
    predictor_factory: Callable[[], torch.nn.Module],
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    source_pretrain_samples: Sequence[BenchmarkSample],
    target_adapt_samples: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    target_test_samples: Sequence[BenchmarkSample],
    config: PretrainThenAdaptConfig,
) -> PretrainThenAdaptResult:
    """
    Two-stage experiment for Problem 1A.

    Stage 1
    -------
    Train on source-side pretraining samples, typically protocol_1 + protocol_2.

    Stage 2
    -------
    Adapt on target-side samples, typically adapt_protocol_3.

    Evaluation
    ----------
    After each stage, evaluate target validation loss and target test loss on
    the designated target protocol.

    Important note for DKL
    ----------------------
    In the source-pretrain stage, the predictor may not yet have a target-side
    observer head for protocol_3. Therefore stage-1 training must NOT use
    target validation inside train_protocol_predictor(...). We set val_samples=None
    for stage 1 and only do a safe post-hoc evaluation, which may return
    (None, None, eval_error) if the target head does not yet exist.
    """
    protocol_map = _normalize_protocol_map(protocols)

    if config.target_protocol_id not in protocol_map:
        raise KeyError(
            f"Unknown target_protocol_id {config.target_protocol_id!r}. "
            f"Available protocol ids: {sorted(protocol_map.keys())}"
        )

    predictor = predictor_factory()
    history: List[PretrainThenAdaptStageRecord] = []

    # --------------------------------------------------------------
    # Stage 1: source pretraining
    # --------------------------------------------------------------
    if len(source_pretrain_samples) == 0:
        raise ValueError("source_pretrain_samples must be non-empty")

    train_protocol_predictor(
        predictor,
        protocols=protocol_map,
        train_samples=list(source_pretrain_samples),
        val_samples=None,  # critical fix: no target validation before target head exists
        config=config.train_config,
    )

    pre_val, pre_test, pre_err = _safe_target_eval_after_stage(
        predictor,
        protocol_map=protocol_map,
        target_protocol_id=config.target_protocol_id,
        val_samples=target_val_samples,
        test_samples=target_test_samples,
        config=config.train_config,
    )
    history.append(
        PretrainThenAdaptStageRecord(
            stage="after_source_pretrain",
            n_train_samples=int(len(source_pretrain_samples)),
            target_val_loss=pre_val,
            target_test_loss=pre_test,
            eval_error=pre_err,
        )
    )

    # --------------------------------------------------------------
    # Stage 2: target adaptation
    # --------------------------------------------------------------
    if len(target_adapt_samples) == 0:
        raise ValueError("target_adapt_samples must be non-empty")

    train_protocol_predictor(
        predictor,
        protocols=protocol_map,
        train_samples=list(target_adapt_samples),
        val_samples=target_val_samples,
        config=config.train_config,
    )

    adapt_val, adapt_test, adapt_err = _safe_target_eval_after_stage(
        predictor,
        protocol_map=protocol_map,
        target_protocol_id=config.target_protocol_id,
        val_samples=target_val_samples,
        test_samples=target_test_samples,
        config=config.train_config,
    )
    history.append(
        PretrainThenAdaptStageRecord(
            stage="after_target_adapt",
            n_train_samples=int(len(target_adapt_samples)),
            target_val_loss=adapt_val,
            target_test_loss=adapt_test,
            eval_error=adapt_err,
        )
    )

    return PretrainThenAdaptResult(
        history=history,
        final_target_val_loss=adapt_val,
        final_target_test_loss=adapt_test,
        n_source_pretrain_samples=int(len(source_pretrain_samples)),
        n_target_adapt_samples=int(len(target_adapt_samples)),
    )


__all__ = [
    "PretrainThenAdaptConfig",
    "PretrainThenAdaptStageRecord",
    "PretrainThenAdaptResult",
    "run_pretrain_then_adapt",
]