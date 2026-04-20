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
        One of "random", "local_uncertainty", or "fantasy".
    adapt_budget_points:
        Number of sequentially acquired target-side additional points.
    outer_train_config:
        Training config used to retrain after each acquisition step.
    fantasy_train_config:
        Training config used inside fantasy scoring.
        Required only when acquisition_policy == "fantasy".
    n_fantasies:
        Number of fantasy samples per candidate when using fantasy selection,
        and number of stochastic predictive draws when using local uncertainty.
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


def _coerce_sample_tensor_list(
    sample_out: Any,
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    if torch.is_tensor(sample_out):
        sample_out = sample_out.detach().to(dtype=torch.float32, device=device)
        if sample_out.ndim == 0:
            sample_out = sample_out.view(1, 1)
        elif sample_out.ndim == 1:
            sample_out = sample_out.unsqueeze(-1)
        return [sample_out[i].detach().cpu().reshape(-1) for i in range(sample_out.shape[0])]

    if isinstance(sample_out, Sequence):
        out: List[torch.Tensor] = []
        for item in sample_out:
            if not torch.is_tensor(item):
                item = torch.as_tensor(item, dtype=torch.float32)
            item = item.detach().to(dtype=torch.float32, device=device)
            if item.ndim == 0:
                item = item.unsqueeze(0)
            elif item.ndim > 1:
                item = item.reshape(-1)
            out.append(item.detach().cpu().clone())
        return out

    raise TypeError(f"Unsupported predictive sample output type: {type(sample_out)}")


def _extract_uncertainty_scalar(obj: Any) -> Optional[float]:
    value = obj

    if isinstance(value, Mapping):
        if "uncertainty" in value:
            value = value["uncertainty"]
        elif "std" in value:
            value = value["std"]
        elif "variance" in value:
            variance = value["variance"]
            if torch.is_tensor(variance):
                variance = variance.detach().to(dtype=torch.float32)
                return float(torch.sqrt(torch.clamp(variance, min=0.0)).mean().item())
            variance = float(variance)
            return float(max(variance, 0.0) ** 0.5)
        else:
            return None
    else:
        if hasattr(value, "uncertainty"):
            value = getattr(value, "uncertainty")
        elif hasattr(value, "std"):
            value = getattr(value, "std")
        elif hasattr(value, "variance"):
            variance = getattr(value, "variance")
            if torch.is_tensor(variance):
                variance = variance.detach().to(dtype=torch.float32)
                return float(torch.sqrt(torch.clamp(variance, min=0.0)).mean().item())
            variance = float(variance)
            return float(max(variance, 0.0) ** 0.5)
        else:
            return None

    if torch.is_tensor(value):
        value = value.detach().to(dtype=torch.float32)
        return float(value.mean().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _predictive_uncertainty_from_hook(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
) -> Optional[float]:
    for method_name in (
        "predict_protocol_uncertainty",
        "predict_with_uncertainty",
        "forward_with_uncertainty",
    ):
        method = getattr(predictor, method_name, None)
        if not callable(method):
            continue

        try:
            out = method(protocol=protocol, condition_x=condition_x)
        except TypeError:
            try:
                out = method(protocol, condition_x)
            except TypeError:
                continue

        uncertainty = _extract_uncertainty_scalar(out)
        if uncertainty is not None:
            return float(uncertainty)

    return None


def _sample_predictive_targets(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
    n_draws: int,
    device: torch.device,
) -> List[torch.Tensor]:
    for method_name in (
        "sample_protocol_fantasy_targets",
        "sample_fantasy_targets",
        "sample_predictive_targets",
    ):
        method = getattr(predictor, method_name, None)
        if callable(method):
            sample_out = method(
                protocol=protocol,
                condition_x=condition_x,
                n_fantasies=int(n_draws),
            )
            return _coerce_sample_tensor_list(sample_out, device=device)

    predictor_kind = str(getattr(predictor, "predictor_type", "generic")).lower()
    if predictor_kind == "dkl":
        raise NotImplementedError(
            "local_uncertainty for predictor_type='dkl' requires a predictor-side "
            "uncertainty hook or predictive sampler hook."
        )

    was_training = predictor.training
    predictor.train()

    out: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(int(n_draws)):
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


def _estimate_local_uncertainty(
    predictor: torch.nn.Module,
    *,
    protocol: ProtocolSpec,
    sample: BenchmarkSample,
    n_draws: int,
    device: torch.device,
) -> float:
    condition_x = _condition_tensor_from_sample(sample, protocol, device=device)

    direct_uncertainty = _predictive_uncertainty_from_hook(
        predictor,
        protocol=protocol,
        condition_x=condition_x,
    )
    if direct_uncertainty is not None:
        return float(direct_uncertainty)

    draws = _sample_predictive_targets(
        predictor,
        protocol=protocol,
        condition_x=condition_x,
        n_draws=n_draws,
        device=device,
    )
    if len(draws) == 0:
        return 0.0
    if len(draws) == 1:
        return 0.0

    stacked = torch.stack(
        [d.detach().to(dtype=torch.float32).reshape(-1) for d in draws],
        dim=0,
    )
    predictive_std = stacked.std(dim=0, unbiased=False)
    return float(predictive_std.mean().item())


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
        Summary dict about the selected item.
    selection_history_rows:
        Optional per-round score rows.
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

    if policy == "local_uncertainty":
        protocol = protocol_map[target_protocol_id]
        best_sample: Optional[BenchmarkSample] = None
        best_score = float("-inf")
        score_rows: List[Dict[str, Any]] = []

        for sample in target_pool:
            raw_uncertainty = _estimate_local_uncertainty(
                predictor,
                protocol=protocol,
                sample=sample,
                n_draws=int(config.n_fantasies),
                device=device,
            )
            acquisition_cost = _sample_cost(sample)
            score = raw_uncertainty / max(float(acquisition_cost), 1e-12)

            score_rows.append(
                {
                    "candidate_id": sample.sample_id,
                    "protocol_id": sample.protocol_id,
                    "acquisition_cost": float(acquisition_cost),
                    "raw_uncertainty": float(raw_uncertainty),
                    "score": float(score),
                    "mean_improvement": None,
                    "policy": "local_uncertainty",
                }
            )

            if score > best_score:
                best_score = float(score)
                best_sample = sample

        if best_sample is None:
            raise RuntimeError(
                "Local uncertainty selection returned no candidate even though the target pool is non-empty."
            )

        chosen_info = {
            "candidate_id": best_sample.sample_id,
            "protocol_id": best_sample.protocol_id,
            "acquisition_cost": _sample_cost(best_sample),
            "score": float(best_score),
            "mean_improvement": None,
            "policy": "local_uncertainty",
        }

        return best_sample, chosen_info, score_rows

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
            remaining_budget=float("inf"),
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
        "Use 'random', 'local_uncertainty', or 'fantasy'."
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
    We retrain from scratch at every step on the accumulated train set,
    matching the design used elsewhere in this codebase.

    Internal validation during training is disabled by passing val_samples=None
    to train_protocol_predictor; target validation/test are computed explicitly
    after training.
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

    for sample in target_candidate_pool:
        if sample.protocol_id != config.target_protocol_id:
            raise ValueError(
                "target_candidate_pool must contain only target-protocol samples, "
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
            selected_candidate_id = None if chosen_info is None else chosen_info["candidate_id"]
            for row in score_rows:
                row["step_idx"] = int(n_target_points_used)
                row["was_selected"] = row["candidate_id"] == selected_candidate_id
            selection_history.extend(score_rows)
        elif chosen_info is not None:
            selection_history.append(
                {
                    "step_idx": int(n_target_points_used),
                    "was_selected": True,
                    **chosen_info,
                }
            )

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