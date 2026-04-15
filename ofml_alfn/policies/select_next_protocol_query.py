#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from ofml_alfn.metrics.protocol_evaluation import compute_target_validation_loss
from ofml_alfn.policies.protocol_candidates import (
    ProtocolQueryCandidate,
    filter_affordable_protocol_candidates,
)
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolTrainingConfig,
    train_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


@dataclass(frozen=True)
class FantasyCandidateScore:
    candidate_id: str
    protocol_id: str
    acquisition_cost: float
    current_target_val_loss: float
    fantasy_val_losses: List[float]
    fantasy_improvements: List[float]
    mean_improvement: float
    score: float


@dataclass(frozen=True)
class ProtocolSelectionResult:
    selected_candidate: Optional[ProtocolQueryCandidate]
    selected_score: Optional[FantasyCandidateScore]
    all_scores: List[FantasyCandidateScore]


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


def _condition_tensor_from_sample(
    sample: BenchmarkSample,
    protocol: ProtocolSpec,
    *,
    device: torch.device,
) -> torch.Tensor:
    row = [float(sample.condition.values[k]) for k in protocol.condition_keys]
    return torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)


def _make_fantasy_sample(
    sample: BenchmarkSample,
    *,
    fantasy_target: torch.Tensor,
    fantasy_index: int,
) -> BenchmarkSample:
    from dataclasses import replace

    return replace(
        sample,
        sample_id=f"{sample.sample_id}__fantasy_{fantasy_index:03d}",
        target_value=fantasy_target.detach().cpu().clone(),
    )


def _sample_fantasy_targets(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
    n_fantasies: int,
) -> List[torch.Tensor]:
    was_training = predictor.training
    predictor.train()

    out: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(int(n_fantasies)):
            pred = predictor.forward_target(
                protocol=protocol,
                condition_x=condition_x,
            )
            out.append(pred.detach().cpu().reshape(-1))

    if not was_training:
        predictor.eval()

    return out


def _mean(xs: Sequence[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))


def score_protocol_query_candidate(
    predictor: torch.nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    candidate: ProtocolQueryCandidate,
    current_train_samples: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    target_protocol_id: str,
    fantasy_train_config: ProtocolTrainingConfig,
    n_fantasies: int,
    device: torch.device,
) -> FantasyCandidateScore:
    protocol_map = _normalize_protocol_map(protocols)
    protocol = protocol_map[candidate.protocol_id]

    current_target_val_loss = compute_target_validation_loss(
        predictor,
        protocols=protocol_map,
        val_samples=target_val_samples,
        target_protocol_id=target_protocol_id,
        config=fantasy_train_config,
    )

    condition_x = _condition_tensor_from_sample(
        candidate.sample,
        protocol,
        device=device,
    )

    fantasy_targets = _sample_fantasy_targets(
        predictor,
        protocol=protocol,
        condition_x=condition_x,
        n_fantasies=n_fantasies,
    )

    fantasy_val_losses: List[float] = []
    fantasy_improvements: List[float] = []

    for k, fantasy_target in enumerate(fantasy_targets):
        fantasy_predictor = deepcopy(predictor)

        fantasy_sample = _make_fantasy_sample(
            candidate.sample,
            fantasy_target=fantasy_target,
            fantasy_index=k,
        )
        fantasy_train_samples = list(current_train_samples) + [fantasy_sample]

        train_protocol_predictor(
            fantasy_predictor,
            protocols=protocol_map,
            train_samples=fantasy_train_samples,
            val_samples=None,
            config=fantasy_train_config,
        )

        fantasy_val_loss = compute_target_validation_loss(
            fantasy_predictor,
            protocols=protocol_map,
            val_samples=target_val_samples,
            target_protocol_id=target_protocol_id,
            config=fantasy_train_config,
        )

        fantasy_val_losses.append(float(fantasy_val_loss))
        fantasy_improvements.append(float(current_target_val_loss - fantasy_val_loss))

    mean_improvement = _mean(fantasy_improvements)
    score = mean_improvement / max(float(candidate.acquisition_cost), 1e-12)

    return FantasyCandidateScore(
        candidate_id=candidate.candidate_id,
        protocol_id=candidate.protocol_id,
        acquisition_cost=float(candidate.acquisition_cost),
        current_target_val_loss=float(current_target_val_loss),
        fantasy_val_losses=[float(x) for x in fantasy_val_losses],
        fantasy_improvements=[float(x) for x in fantasy_improvements],
        mean_improvement=float(mean_improvement),
        score=float(score),
    )


def select_next_protocol_query(
    predictor: torch.nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    candidate_pool: Sequence[ProtocolQueryCandidate],
    current_train_samples: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    target_protocol_id: str,
    fantasy_train_config: ProtocolTrainingConfig,
    n_fantasies: int,
    remaining_budget: float,
    device: torch.device,
) -> ProtocolSelectionResult:
    protocol_map = _normalize_protocol_map(protocols)
    affordable_candidates = filter_affordable_protocol_candidates(
        candidate_pool,
        remaining_budget=remaining_budget,
    )

    all_scores: List[FantasyCandidateScore] = []
    best_candidate: Optional[ProtocolQueryCandidate] = None
    best_score_obj: Optional[FantasyCandidateScore] = None
    best_score = float("-inf")

    for candidate in affordable_candidates:
        score_obj = score_protocol_query_candidate(
            predictor,
            protocols=protocol_map,
            candidate=candidate,
            current_train_samples=current_train_samples,
            target_val_samples=target_val_samples,
            target_protocol_id=target_protocol_id,
            fantasy_train_config=fantasy_train_config,
            n_fantasies=n_fantasies,
            device=device,
        )
        all_scores.append(score_obj)

        if score_obj.score > best_score:
            best_score = float(score_obj.score)
            best_candidate = candidate
            best_score_obj = score_obj

    return ProtocolSelectionResult(
        selected_candidate=best_candidate,
        selected_score=best_score_obj,
        all_scores=all_scores,
    )


__all__ = [
    "FantasyCandidateScore",
    "ProtocolSelectionResult",
    "score_protocol_query_candidate",
    "select_next_protocol_query",
]