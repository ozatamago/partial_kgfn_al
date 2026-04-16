#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from ofml_alfn.metrics.protocol_evaluation import compute_target_validation_loss
from ofml_alfn.policies.protocol_candidates import (
    ProtocolQueryCandidate,
    build_protocol_query_candidates,
)
from ofml_alfn.policies.select_next_protocol_query import (
    ProtocolSelectionResult,
    select_next_protocol_query,
)
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolTrainingConfig,
    evaluate_protocol_predictor,
    train_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


@dataclass(frozen=True)
class ActiveLearningRunnerConfig:
    budget: float
    target_protocol_id: str
    n_fantasies: int
    outer_train_config: ProtocolTrainingConfig
    fantasy_train_config: ProtocolTrainingConfig
    device: str = "cpu"


@dataclass(frozen=True)
class ActiveLearningRoundRecord:
    round_idx: int
    spent_budget: float
    remaining_budget: float
    train_size: int
    pool_size: int
    target_val_loss: float
    target_test_loss: float
    chosen_candidate_id: Optional[str]
    chosen_protocol_id: Optional[str]
    chosen_acquisition_cost: Optional[float]
    chosen_score: Optional[float]
    chosen_mean_improvement: Optional[float]


@dataclass(frozen=True)
class ActiveLearningRunResult:
    history: List[ActiveLearningRoundRecord]
    selection_history: List[Dict[str, Any]]
    spent_budget: float
    final_train_samples: List[BenchmarkSample]
    final_candidate_pool: List[BenchmarkSample]


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


def run_protocol_active_learning(
    *,
    predictor_factory: Callable[[], torch.nn.Module],
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    initial_train_samples: Sequence[BenchmarkSample],
    candidate_pool: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    target_test_samples: Sequence[BenchmarkSample],
    config: ActiveLearningRunnerConfig,
) -> ActiveLearningRunResult:
    """
    Run protocol-level active learning with fantasy selection.

    Loop per round
    --------------
    1. Build a fresh predictor from predictor_factory().
    2. Train it on current_train_samples.
    3. Measure target validation and target test loss.
    4. Build protocol query candidates from the remaining pool.
    5. Use select_next_protocol_query(...) to choose the next acquisition.
    6. Add the selected sample to the train set and update the budget.

    Notes
    -----
    - DKL support is delegated to predictor-specific hooks inside
      train_protocol_predictor(...) and evaluate_protocol_predictor(...).
    - Fantasy sampling details are delegated to
      select_next_protocol_query(...).
    """
    protocol_map = _normalize_protocol_map(protocols)
    if config.target_protocol_id not in protocol_map:
        raise KeyError(
            f"Unknown target_protocol_id {config.target_protocol_id!r}. "
            f"Available protocol ids: {sorted(protocol_map.keys())}"
        )

    device = torch.device(str(config.device))
    target_protocol = protocol_map[config.target_protocol_id]

    current_train_samples: List[BenchmarkSample] = list(initial_train_samples)
    current_candidate_pool: List[BenchmarkSample] = list(candidate_pool)

    history: List[ActiveLearningRoundRecord] = []
    selection_history: List[Dict[str, Any]] = []

    spent_budget = 0.0
    round_idx = 0

    while True:
        predictor = predictor_factory()

        train_protocol_predictor(
            predictor,
            protocols=protocol_map,
            train_samples=current_train_samples,
            val_samples=target_val_samples,
            config=config.outer_train_config,
        )

        current_target_val_loss = compute_target_validation_loss(
            predictor,
            protocols=protocol_map,
            val_samples=target_val_samples,
            target_protocol_id=config.target_protocol_id,
            config=config.outer_train_config,
        )
        current_target_test_loss = _evaluate_target_test_loss(
            predictor,
            target_protocol=target_protocol,
            test_samples=target_test_samples,
            config=config.outer_train_config,
        )

        remaining_budget = float(config.budget) - float(spent_budget)
        if remaining_budget <= 0.0 or len(current_candidate_pool) == 0:
            history.append(
                ActiveLearningRoundRecord(
                    round_idx=int(round_idx),
                    spent_budget=float(spent_budget),
                    remaining_budget=float(max(remaining_budget, 0.0)),
                    train_size=int(len(current_train_samples)),
                    pool_size=int(len(current_candidate_pool)),
                    target_val_loss=float(current_target_val_loss),
                    target_test_loss=float(current_target_test_loss),
                    chosen_candidate_id=None,
                    chosen_protocol_id=None,
                    chosen_acquisition_cost=None,
                    chosen_score=None,
                    chosen_mean_improvement=None,
                )
            )
            break

        protocol_candidates: List[ProtocolQueryCandidate] = build_protocol_query_candidates(
            current_candidate_pool,
            protocols=protocol_map,
            target_protocol_id=config.target_protocol_id,
        )

        selection: ProtocolSelectionResult = select_next_protocol_query(
            predictor,
            protocols=protocol_map,
            candidate_pool=protocol_candidates,
            current_train_samples=current_train_samples,
            target_val_samples=target_val_samples,
            target_protocol_id=config.target_protocol_id,
            fantasy_train_config=config.fantasy_train_config,
            n_fantasies=int(config.n_fantasies),
            remaining_budget=float(remaining_budget),
            device=device,
        )

        chosen = selection.selected_candidate
        chosen_score = selection.selected_score

        history.append(
            ActiveLearningRoundRecord(
                round_idx=int(round_idx),
                spent_budget=float(spent_budget),
                remaining_budget=float(remaining_budget),
                train_size=int(len(current_train_samples)),
                pool_size=int(len(current_candidate_pool)),
                target_val_loss=float(current_target_val_loss),
                target_test_loss=float(current_target_test_loss),
                chosen_candidate_id=None if chosen is None else chosen.candidate_id,
                chosen_protocol_id=None if chosen is None else chosen.protocol_id,
                chosen_acquisition_cost=None if chosen is None else float(chosen.acquisition_cost),
                chosen_score=None if chosen_score is None else float(chosen_score.score),
                chosen_mean_improvement=(
                    None if chosen_score is None else float(chosen_score.mean_improvement)
                ),
            )
        )
        selection_history.extend([score.__dict__ for score in selection.all_scores])

        if chosen is None:
            break

        current_train_samples.append(chosen.sample)
        current_candidate_pool = [
            s for s in current_candidate_pool if s.sample_id != chosen.sample.sample_id
        ]
        spent_budget += _sample_cost(chosen.sample)
        round_idx += 1

    return ActiveLearningRunResult(
        history=history,
        selection_history=selection_history,
        spent_budget=float(spent_budget),
        final_train_samples=current_train_samples,
        final_candidate_pool=current_candidate_pool,
    )


__all__ = [
    "ActiveLearningRunnerConfig",
    "ActiveLearningRoundRecord",
    "ActiveLearningRunResult",
    "run_protocol_active_learning",
]