#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import random
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
class SequentialTargetAdaptRunnerConfig:
    """
    Configuration for sequential target-only adaptation.

    Parameters
    ----------
    target_protocol_id:
        The protocol to adapt on, typically protocol_3.
    acquisition_policy:
        "random" or "fantasy".
    adapt_budget_points:
        Number of sequentially acquired target-side additional points.
    outer_train_config:
        Training config used to retrain after each acquisition step.
    fantasy_train_config:
        Training config used inside fantasy scoring.
        Required only when acquisition_policy == "fantasy".
    n_fantasies:
        Number of fantasy samples per candidate when using fantasy selection.
    device:
        Torch device string.
    random_seed:
        Seed for random acquisition and any candidate subsampling randomness.
    """
    target_protocol_id: str
    acquisition_policy: str = "random"
    adapt_budget_points: int = 30
    outer_train_config: ProtocolTrainingConfig = ProtocolTrainingConfig()
    fantasy_train_config: Optional[ProtocolTrainingConfig] = None
    n_fantasies: int = 8
    device: str = "cpu"
    random_seed: int = 0


@dataclass(frozen=True)
class SequentialTargetAdaptRecord:
    """
    One row in the learning curve history.

    The record is logged before the next acquisition at each step.
    Therefore:
    - step_idx = 0 means "before any additional target-side point was acquired"
    - step_idx = t means "after t target-side additional points were acquired"
    """
    step_idx: int
    n_target_points_used: int
    target_cost_used: float
    train_size: int
    pool_size: int
    target_val_loss: Optional[float]
    target_test_loss: Optional[float]
    eval_error: Optional[str]
    chosen_candidate_id: Optional[str]
    chosen_protocol_id: Optional[str]
    chosen_acquisition_cost: Optional[float]
    chosen_score: Optional[float]
    chosen_mean_improvement: Optional[float]


@dataclass(frozen=True)
class SequentialTargetAdaptResult:
    history: List[SequentialTargetAdaptRecord]
    selection_history: List[Dict[str, Any]]
    n_target_points_used: int
    target_cost_used: float
    final_train_samples: List[BenchmarkSample]
    final_target_pool: List[BenchmarkSample]


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


def _safe_target_eval(
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


def _choose_random_candidate(
    pool: Sequence[BenchmarkSample],
    *,
    rng: random.Random,
) -> BenchmarkSample:
    if len(pool) == 0:
        raise ValueError("Cannot choose from an empty target pool")
    idx = rng.randrange(len(pool))
    return pool[idx]


def _select_target_candidate(
    predictor: torch.nn.Module,
    *,
    protocol_map: Dict[str, ProtocolSpec],
    target_protocol_id: str,
    current_train_samples: Sequence[BenchmarkSample],
    target_pool: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    config: SequentialTargetAdaptRunnerConfig,
    rng: random.Random,
    device: torch.device,
) -> tuple[BenchmarkSample, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns
    -------
    chosen_sample:
        The actual target sample to acquire next.
    chosen_info:
        Summary dict about the selected item, or None for random without score.
    selection_history_rows:
        Optional per-round score rows, mainly populated for fantasy mode.
    """
    policy = str(config.acquisition_policy).lower()

    if policy == "random":
        chosen = _choose_random_candidate(target_pool, rng=rng)
        chosen_info = {
            "candidate_id": chosen.sample_id,
            "protocol_id": chosen.protocol_id,
            "acquisition_cost": _sample_cost(chosen),
            "score": None,
            "mean_improvement": None,
            "policy": "random",
        }
        return chosen, chosen_info, []

    if policy == "fantasy":
        if config.fantasy_train_config is None:
            raise ValueError(
                "fantasy_train_config must be provided when acquisition_policy='fantasy'"
            )

        protocol_candidates: List[ProtocolQueryCandidate] = build_protocol_query_candidates(
            target_pool,
            protocols=protocol_map,
            target_protocol_id=target_protocol_id,
        )

        selection: ProtocolSelectionResult = select_next_protocol_query(
            predictor,
            protocols=protocol_map,
            candidate_pool=protocol_candidates,
            current_train_samples=current_train_samples,
            target_val_samples=target_val_samples,
            target_protocol_id=target_protocol_id,
            fantasy_train_config=config.fantasy_train_config,
            n_fantasies=int(config.n_fantasies),
            remaining_budget=float("inf"),  # budget here is point-count based, not cost based
            device=device,
        )

        if selection.selected_candidate is None:
            raise RuntimeError(
                "Fantasy selection returned no candidate even though the target pool is non-empty."
            )

        chosen_sample = selection.selected_candidate.sample
        chosen_score = selection.selected_score

        chosen_info = {
            "candidate_id": selection.selected_candidate.candidate_id,
            "protocol_id": selection.selected_candidate.protocol_id,
            "acquisition_cost": float(selection.selected_candidate.acquisition_cost),
            "score": None if chosen_score is None else float(chosen_score.score),
            "mean_improvement": (
                None if chosen_score is None else float(chosen_score.mean_improvement)
            ),
            "policy": "fantasy",
        }

        score_rows: List[Dict[str, Any]] = []
        for s in selection.all_scores:
            score_rows.append(
                {
                    "candidate_id": s.candidate_id,
                    "protocol_id": s.protocol_id,
                    "acquisition_cost": float(s.acquisition_cost),
                    "current_target_val_loss": float(s.current_target_val_loss),
                    "fantasy_val_losses": [float(x) for x in s.fantasy_val_losses],
                    "fantasy_improvements": [float(x) for x in s.fantasy_improvements],
                    "mean_improvement": float(s.mean_improvement),
                    "score": float(s.score),
                    "policy": "fantasy",
                }
            )

        return chosen_sample, chosen_info, score_rows

    raise ValueError(
        f"Unsupported acquisition_policy: {config.acquisition_policy!r}. "
        "Use 'random' or 'fantasy'."
    )


def run_sequential_target_adapt(
    *,
    predictor_factory: Callable[[], torch.nn.Module],
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    initial_train_samples: Sequence[BenchmarkSample],
    target_candidate_pool: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    target_test_samples: Sequence[BenchmarkSample],
    config: SequentialTargetAdaptRunnerConfig,
) -> SequentialTargetAdaptResult:
    """
    Sequentially adapt on the target protocol only.

    Design
    ------
    This runner is intended for refined transfer experiments:
    - baseline: scratch -> sequential target-side adaptation
    - proposed: source pretrain -> same sequential target-side adaptation

    In both cases, the target-side acquisition rule is fixed and identical.

    Important note
    --------------
    We retrain from scratch at every step on the accumulated train set, matching
    the design used elsewhere in this codebase. Internal validation during
    training is disabled by passing val_samples=None to train_protocol_predictor;
    target validation/test are computed explicitly after training.
    """
    if int(config.adapt_budget_points) < 0:
        raise ValueError(
            f"adapt_budget_points must be non-negative, got {config.adapt_budget_points}"
        )

    protocol_map = _normalize_protocol_map(protocols)

    if config.target_protocol_id not in protocol_map:
        raise KeyError(
            f"Unknown target_protocol_id {config.target_protocol_id!r}. "
            f"Available protocol ids: {sorted(protocol_map.keys())}"
        )

    # Enforce that the sequential pool is target-only
    for sample in target_candidate_pool:
        if sample.protocol_id != config.target_protocol_id:
            raise ValueError(
                f"target_candidate_pool must contain only target-protocol samples, "
                f"but found sample {sample.sample_id!r} from protocol {sample.protocol_id!r}"
            )

    current_train_samples: List[BenchmarkSample] = list(initial_train_samples)
    current_target_pool: List[BenchmarkSample] = list(target_candidate_pool)

    rng = random.Random(int(config.random_seed))
    device = torch.device(str(config.device))
    target_protocol = protocol_map[config.target_protocol_id]

    history: List[SequentialTargetAdaptRecord] = []
    selection_history: List[Dict[str, Any]] = []

    n_target_points_used = 0
    target_cost_used = 0.0

    while True:
        predictor = predictor_factory()

        # Train on the accumulated data.
        # Use no internal validation here; explicit target eval comes next.
        if len(current_train_samples) == 0:
            raise ValueError(
                "initial_train_samples is empty. "
                "This runner expects a non-empty starting training set."
            )

        train_protocol_predictor(
            predictor,
            protocols=protocol_map,
            train_samples=current_train_samples,
            val_samples=None,
            config=config.outer_train_config,
        )

        target_val_loss, target_test_loss, eval_error = _safe_target_eval(
            predictor,
            protocol_map=protocol_map,
            target_protocol_id=config.target_protocol_id,
            val_samples=target_val_samples,
            test_samples=target_test_samples,
            config=config.outer_train_config,
        )

        # Stop before selecting a new point if budget or pool is exhausted.
        if (
            n_target_points_used >= int(config.adapt_budget_points)
            or len(current_target_pool) == 0
        ):
            history.append(
                SequentialTargetAdaptRecord(
                    step_idx=int(n_target_points_used),
                    n_target_points_used=int(n_target_points_used),
                    target_cost_used=float(target_cost_used),
                    train_size=int(len(current_train_samples)),
                    pool_size=int(len(current_target_pool)),
                    target_val_loss=target_val_loss,
                    target_test_loss=target_test_loss,
                    eval_error=eval_error,
                    chosen_candidate_id=None,
                    chosen_protocol_id=None,
                    chosen_acquisition_cost=None,
                    chosen_score=None,
                    chosen_mean_improvement=None,
                )
            )
            break

        chosen_sample, chosen_info, score_rows = _select_target_candidate(
            predictor,
            protocol_map=protocol_map,
            target_protocol_id=config.target_protocol_id,
            current_train_samples=current_train_samples,
            target_pool=current_target_pool,
            target_val_samples=target_val_samples,
            config=config,
            rng=rng,
            device=device,
        )

        chosen_cost = _sample_cost(chosen_sample)

        history.append(
            SequentialTargetAdaptRecord(
                step_idx=int(n_target_points_used),
                n_target_points_used=int(n_target_points_used),
                target_cost_used=float(target_cost_used),
                train_size=int(len(current_train_samples)),
                pool_size=int(len(current_target_pool)),
                target_val_loss=target_val_loss,
                target_test_loss=target_test_loss,
                eval_error=eval_error,
                chosen_candidate_id=None if chosen_info is None else chosen_info["candidate_id"],
                chosen_protocol_id=None if chosen_info is None else chosen_info["protocol_id"],
                chosen_acquisition_cost=None if chosen_info is None else float(chosen_info["acquisition_cost"]),
                chosen_score=None if chosen_info is None or chosen_info["score"] is None else float(chosen_info["score"]),
                chosen_mean_improvement=(
                    None
                    if chosen_info is None or chosen_info["mean_improvement"] is None
                    else float(chosen_info["mean_improvement"])
                ),
            )
        )

        if len(score_rows) > 0:
            for row in score_rows:
                row["step_idx"] = int(n_target_points_used)
            selection_history.extend(score_rows)
        elif chosen_info is not None:
            selection_history.append(
                {
                    "step_idx": int(n_target_points_used),
                    **chosen_info,
                }
            )

        # Acquire the selected point
        current_train_samples.append(chosen_sample)
        current_target_pool = [
            s for s in current_target_pool if s.sample_id != chosen_sample.sample_id
        ]
        n_target_points_used += 1
        target_cost_used += float(chosen_cost)

    return SequentialTargetAdaptResult(
        history=history,
        selection_history=selection_history,
        n_target_points_used=int(n_target_points_used),
        target_cost_used=float(target_cost_used),
        final_train_samples=current_train_samples,
        final_target_pool=current_target_pool,
    )


__all__ = [
    "SequentialTargetAdaptRunnerConfig",
    "SequentialTargetAdaptRecord",
    "SequentialTargetAdaptResult",
    "run_sequential_target_adapt",
]