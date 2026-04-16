#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from ofml_alfn.metrics.protocol_evaluation import (
    compute_cost_normalized_target_validation_improvement,
    compute_target_validation_loss,
)
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


def _predictor_type(predictor: torch.nn.Module) -> str:
    return str(getattr(predictor, "predictor_type", "generic")).lower()


def _condition_tensor_from_sample(
    sample: BenchmarkSample,
    protocol: ProtocolSpec,
    *,
    device: torch.device,
) -> torch.Tensor:
    row = [float(sample.condition.values[k]) for k in protocol.condition_keys]
    return torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)


def _extract_prediction_tensor(
    pred_obj: Any,
    *,
    device: torch.device,
    protocol_id: str,
) -> torch.Tensor:
    candidate = pred_obj

    if isinstance(candidate, Mapping):
        for key in ("target", "target_pred", "prediction", "pred", "mean"):
            if key in candidate:
                candidate = candidate[key]
                break
    elif not torch.is_tensor(candidate):
        for attr in ("target", "target_pred", "prediction", "pred", "target_value", "mean"):
            if hasattr(candidate, attr):
                candidate = getattr(candidate, attr)
                break

    if not torch.is_tensor(candidate):
        raise TypeError(
            f"Could not extract tensor prediction for protocol {protocol_id!r}. "
            f"Got object of type {type(pred_obj)}"
        )

    candidate = candidate.detach().to(dtype=torch.float32, device=device)
    if candidate.ndim == 0:
        candidate = candidate.view(1, 1)
    elif candidate.ndim == 1:
        candidate = candidate.unsqueeze(-1)
    return candidate


def _predict_target_batch(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if hasattr(predictor, "forward_target"):
        pred_obj = predictor.forward_target(protocol=protocol, condition_x=condition_x)
        return _extract_prediction_tensor(
            pred_obj,
            device=device,
            protocol_id=protocol.protocol_id,
        )

    if hasattr(predictor, "forward_protocol"):
        pred_obj = predictor.forward_protocol(protocol=protocol, condition_x=condition_x)
        return _extract_prediction_tensor(
            pred_obj,
            device=device,
            protocol_id=protocol.protocol_id,
        )

    try:
        pred_obj = predictor(protocol=protocol, condition_x=condition_x)
        return _extract_prediction_tensor(
            pred_obj,
            device=device,
            protocol_id=protocol.protocol_id,
        )
    except TypeError:
        pass

    pred_obj = predictor(protocol, condition_x)
    return _extract_prediction_tensor(
        pred_obj,
        device=device,
        protocol_id=protocol.protocol_id,
    )


def _coerce_fantasy_tensor_list(
    fantasy_out: Any,
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    if torch.is_tensor(fantasy_out):
        fantasy_out = fantasy_out.detach().to(dtype=torch.float32, device=device)
        if fantasy_out.ndim == 0:
            fantasy_out = fantasy_out.view(1, 1)
        elif fantasy_out.ndim == 1:
            fantasy_out = fantasy_out.unsqueeze(-1)
        return [fantasy_out[i].detach().cpu().reshape(-1) for i in range(fantasy_out.shape[0])]

    if isinstance(fantasy_out, Sequence):
        out: List[torch.Tensor] = []
        for item in fantasy_out:
            if not torch.is_tensor(item):
                item = torch.as_tensor(item, dtype=torch.float32)
            item = item.detach().to(dtype=torch.float32, device=device)
            if item.ndim == 0:
                item = item.unsqueeze(0)
            elif item.ndim > 1:
                item = item.reshape(-1)
            out.append(item.detach().cpu().clone())
        return out

    raise TypeError(
        f"Unsupported fantasy sampler output type: {type(fantasy_out)}"
    )


def _sample_fantasy_targets(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
    n_fantasies: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Fantasy target sampling strategy.

    Priority:
    1. predictor-specific fantasy sampler hook
    2. MCD-style repeated stochastic forward passes
    """
    for method_name in (
        "sample_protocol_fantasy_targets",
        "sample_fantasy_targets",
        "sample_predictive_targets",
    ):
        method = getattr(predictor, method_name, None)
        if callable(method):
            fantasy_out = method(
                protocol=protocol,
                condition_x=condition_x,
                n_fantasies=int(n_fantasies),
            )
            return _coerce_fantasy_tensor_list(
                fantasy_out,
                device=device,
            )

    predictor_kind = _predictor_type(predictor)
    if predictor_kind == "dkl":
        raise NotImplementedError(
            "predictor_type='dkl' requires a predictor-specific fantasy sampler hook "
            "(sample_protocol_fantasy_targets, sample_fantasy_targets, or "
            "sample_predictive_targets)."
        )

    was_training = predictor.training
    predictor.train()

    out: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(int(n_fantasies)):
            pred = _predict_target_batch(
                predictor,
                protocol=protocol,
                condition_x=condition_x,
                device=device,
            )
            out.append(pred.detach().cpu().reshape(-1))

    if not was_training:
        predictor.eval()

    return out


def _make_fantasy_sample(
    sample: BenchmarkSample,
    *,
    fantasy_target: torch.Tensor,
    fantasy_index: int,
) -> BenchmarkSample:
    return replace(
        sample,
        sample_id=f"{sample.sample_id}__fantasy_{fantasy_index:03d}",
        target_value=fantasy_target.detach().cpu().clone(),
    )


def _mean(xs: Sequence[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))


def _group_candidates_by_protocol(
    candidates: Sequence[ProtocolQueryCandidate],
) -> Dict[str, List[ProtocolQueryCandidate]]:
    grouped: Dict[str, List[ProtocolQueryCandidate]] = {}
    for cand in candidates:
        grouped.setdefault(cand.protocol_id, []).append(cand)
    return grouped


def _subsample_candidates_per_protocol(
    candidates: Sequence[ProtocolQueryCandidate],
    *,
    max_per_protocol: int = 20,
    seed: Optional[int] = None,
) -> List[ProtocolQueryCandidate]:
    """
    Randomly subsample candidates within each protocol.

    This is a simple compute-saving shortcut:
    after budget filtering, keep at most `max_per_protocol`
    candidates for each protocol.
    """
    if max_per_protocol <= 0:
        raise ValueError(f"max_per_protocol must be positive, got {max_per_protocol}")

    grouped = _group_candidates_by_protocol(candidates)
    rng = random.Random(seed)

    selected: List[ProtocolQueryCandidate] = []
    for protocol_id in sorted(grouped.keys()):
        group = list(grouped[protocol_id])
        if len(group) <= max_per_protocol:
            selected.extend(group)
            continue

        rng.shuffle(group)
        selected.extend(group[:max_per_protocol])

    return selected


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
        device=device,
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
    score = compute_cost_normalized_target_validation_improvement(
        current_loss=float(current_target_val_loss),
        updated_loss=float(current_target_val_loss - mean_improvement),
        acquisition_cost=float(candidate.acquisition_cost),
    )

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

    # --------------------------------------------------------------
    # Compute-saving candidate subsampling:
    # keep at most 20 random candidates per protocol
    # --------------------------------------------------------------
    candidate_subset = _subsample_candidates_per_protocol(
        affordable_candidates,
        max_per_protocol=20,
    )

    all_scores: List[FantasyCandidateScore] = []
    best_candidate: Optional[ProtocolQueryCandidate] = None
    best_score_obj: Optional[FantasyCandidateScore] = None
    best_score = float("-inf")

    for candidate in candidate_subset:
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