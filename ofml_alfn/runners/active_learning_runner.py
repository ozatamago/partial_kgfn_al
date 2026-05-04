#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ofml_alfn.metrics.protocol_evaluation import compute_target_validation_loss
from ofml_alfn.policies.protocol_candidates import (
    ProtocolQueryCandidate,
    build_protocol_query_candidates,
    filter_affordable_protocol_candidates,
)
from ofml_alfn.policies.select_next_protocol_query import (
    select_next_protocol_query,
)
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolTrainingConfig,
    evaluate_protocol_predictor,
    train_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveLearningRunnerConfig:
    budget: float
    target_protocol_id: str
    outer_train_config: ProtocolTrainingConfig
    fantasy_train_config: Optional[ProtocolTrainingConfig] = None
    n_fantasies: int = 8
    device: str = "cpu"
    acquisition_policy: str = "fantasy"
    random_seed: int = 0

    # Debug options.
    # debug_progress=True prints acquisition-level progress for each AL round.
    # debug_top_k_candidates controls how many top-scored candidates are printed.
    debug_progress: bool = False
    debug_top_k_candidates: int = 5


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


def _to_python_debug_value(x: Any) -> Any:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return [float(v) for v in x.reshape(-1).tolist()]

    if isinstance(x, Mapping):
        return {str(k): _to_python_debug_value(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_python_debug_value(v) for v in x]

    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _sample_condition_values(sample: BenchmarkSample) -> Dict[str, Any]:
    condition = getattr(sample, "condition", None)

    if condition is None:
        return {}

    values = getattr(condition, "values", None)

    if values is None and isinstance(condition, Mapping):
        values = condition.get("values", condition)

    if values is None:
        return {}

    if isinstance(values, Mapping):
        return {str(k): _to_python_debug_value(v) for k, v in values.items()}

    return {"value": _to_python_debug_value(values)}


def _sample_target_value(sample: BenchmarkSample) -> Any:
    if hasattr(sample, "target_value"):
        return _to_python_debug_value(sample.target_value)
    return None


def _sample_id(sample: BenchmarkSample) -> str:
    value = getattr(sample, "sample_id", None)
    if value is not None:
        return str(value)

    if isinstance(sample, Mapping):
        for key in ("sample_id", "id", "condition_id"):
            if key in sample:
                return str(sample[key])

    return "unknown"


def _round_float(x: Any, ndigits: int = 8) -> Any:
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return x


def _json_debug(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


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


def _select_random_candidate(
    *,
    affordable_candidates: Sequence[ProtocolQueryCandidate],
    rng: random.Random,
) -> Tuple[Optional[ProtocolQueryCandidate], Optional[float], Optional[float], List[Dict[str, Any]]]:
    if len(affordable_candidates) == 0:
        return None, None, None, []

    chosen = affordable_candidates[rng.randrange(len(affordable_candidates))]

    history = [
        {
            "policy": "random",
            "candidate_id": chosen.candidate_id,
            "protocol_id": chosen.protocol_id,
            "acquisition_cost": float(chosen.acquisition_cost),
            "score": None,
            "mean_improvement": None,
            "was_selected": True,
        }
    ]

    return chosen, None, None, history


def _select_local_uncertainty_candidate(
    predictor: torch.nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    affordable_candidates: Sequence[ProtocolQueryCandidate],
    n_draws: int,
    device: torch.device,
) -> Tuple[Optional[ProtocolQueryCandidate], Optional[float], Optional[float], List[Dict[str, Any]]]:
    if len(affordable_candidates) == 0:
        return None, None, None, []

    protocol_map = _normalize_protocol_map(protocols)

    best_candidate: Optional[ProtocolQueryCandidate] = None
    best_score = float("-inf")
    history: List[Dict[str, Any]] = []

    for candidate in affordable_candidates:
        protocol = protocol_map[candidate.protocol_id]

        raw_uncertainty = _estimate_local_uncertainty(
            predictor,
            protocol=protocol,
            sample=candidate.sample,
            n_draws=int(n_draws),
            device=device,
        )
        score = raw_uncertainty / max(float(candidate.acquisition_cost), 1e-12)

        history.append(
            {
                "policy": "local_uncertainty",
                "candidate_id": candidate.candidate_id,
                "protocol_id": candidate.protocol_id,
                "acquisition_cost": float(candidate.acquisition_cost),
                "raw_uncertainty": float(raw_uncertainty),
                "score": float(score),
                "mean_improvement": None,
                "was_selected": False,
            }
        )

        if score > best_score:
            best_score = float(score)
            best_candidate = candidate

    if best_candidate is not None:
        for row in history:
            if row["candidate_id"] == best_candidate.candidate_id:
                row["was_selected"] = True
                break

    return best_candidate, float(best_score), None, history


def _select_candidate_by_policy(
    predictor: torch.nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    protocol_candidates: Sequence[ProtocolQueryCandidate],
    current_train_samples: Sequence[BenchmarkSample],
    target_val_samples: Sequence[BenchmarkSample],
    config: ActiveLearningRunnerConfig,
    remaining_budget: float,
    device: torch.device,
    rng: random.Random,
) -> Tuple[Optional[ProtocolQueryCandidate], Optional[float], Optional[float], List[Dict[str, Any]]]:
    affordable_candidates = filter_affordable_protocol_candidates(
        protocol_candidates,
        remaining_budget=float(remaining_budget),
    )

    policy = str(config.acquisition_policy).lower()

    if len(affordable_candidates) == 0:
        return None, None, None, []

    if policy == "random":
        return _select_random_candidate(
            affordable_candidates=affordable_candidates,
            rng=rng,
        )

    if policy == "local_uncertainty":
        return _select_local_uncertainty_candidate(
            predictor,
            protocols=protocols,
            affordable_candidates=affordable_candidates,
            n_draws=int(config.n_fantasies),
            device=device,
        )

    if policy == "fantasy":
        if config.fantasy_train_config is None:
            raise ValueError(
                "acquisition_policy='fantasy' requires fantasy_train_config to be set"
            )

        selection = select_next_protocol_query(
            predictor,
            protocols=protocols,
            candidate_pool=affordable_candidates,
            current_train_samples=current_train_samples,
            target_val_samples=target_val_samples,
            target_protocol_id=config.target_protocol_id,
            fantasy_train_config=config.fantasy_train_config,
            n_fantasies=int(config.n_fantasies),
            remaining_budget=float(remaining_budget),
            device=device,
        )

        chosen = selection.selected_candidate
        chosen_score = None if selection.selected_score is None else float(selection.selected_score.score)
        chosen_mean_improvement = (
            None if selection.selected_score is None else float(selection.selected_score.mean_improvement)
        )

        history: List[Dict[str, Any]] = []
        for score_obj in selection.all_scores:
            row = dict(score_obj.__dict__)
            row["policy"] = "fantasy"
            row["was_selected"] = (
                chosen is not None and score_obj.candidate_id == chosen.candidate_id
            )
            history.append(row)

        return chosen, chosen_score, chosen_mean_improvement, history

    raise ValueError(
        f"Unsupported acquisition_policy {config.acquisition_policy!r}. "
        "Expected one of: random, local_uncertainty, fantasy."
    )


def _enrich_selection_rows(
    *,
    round_selection_history: List[Dict[str, Any]],
    protocol_candidates: Sequence[ProtocolQueryCandidate],
    round_idx: int,
    train_size_before: int,
    pool_size_before: int,
    spent_budget_before: float,
    remaining_budget_before: float,
    current_target_val_loss: float,
    current_target_test_loss: float,
) -> List[Dict[str, Any]]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in protocol_candidates}
    enriched: List[Dict[str, Any]] = []

    for row in round_selection_history:
        out = dict(row)

        candidate_id = out.get("candidate_id")
        candidate = candidate_by_id.get(candidate_id)

        out["round_idx"] = int(round_idx)
        out["train_size_before"] = int(train_size_before)
        out["pool_size_before"] = int(pool_size_before)
        out["spent_budget_before"] = float(spent_budget_before)
        out["remaining_budget_before"] = float(remaining_budget_before)

        out["actual_before_target_val_loss"] = float(current_target_val_loss)
        out["actual_before_target_test_loss"] = float(current_target_test_loss)

        if candidate is not None:
            out["sample_id"] = _sample_id(candidate.sample)
            out["condition_values"] = _sample_condition_values(candidate.sample)
            out["observed_target_value"] = _sample_target_value(candidate.sample)
            out["acquisition_cost"] = float(candidate.acquisition_cost)

        if out.get("mean_improvement") is not None and out.get("acquisition_cost") is not None:
            cost = max(float(out["acquisition_cost"]), 1e-12)
            out["fantasy_mean_improvement_per_cost"] = float(out["mean_improvement"]) / cost

        enriched.append(out)

    return enriched


def _selected_row_index(rows: Sequence[Dict[str, Any]]) -> Optional[int]:
    for i, row in enumerate(rows):
        if bool(row.get("was_selected", False)):
            return int(i)
    return None


def _log_candidate_scoring_debug(
    *,
    config: ActiveLearningRunnerConfig,
    round_idx: int,
    current_target_val_loss: float,
    current_target_test_loss: float,
    enriched_selection_rows: Sequence[Dict[str, Any]],
) -> None:
    if not bool(config.debug_progress):
        return

    policy = str(config.acquisition_policy).lower()
    selected_rows = [row for row in enriched_selection_rows if bool(row.get("was_selected", False))]

    logger.warning(
        (
            "[AL DEBUG] round=%d | before_acquisition | "
            "val_loss=%.8f | test_loss=%.8f | policy=%s | scored_candidates=%d"
        ),
        int(round_idx),
        float(current_target_val_loss),
        float(current_target_test_loss),
        policy,
        len(enriched_selection_rows),
    )

    if len(selected_rows) > 0:
        row = selected_rows[0]

        logger.warning(
            (
                "[AL DEBUG] round=%d | selected | "
                "candidate_id=%s | sample_id=%s | protocol_id=%s | "
                "cost=%s | condition=%s | target=%s"
            ),
            int(round_idx),
            row.get("candidate_id"),
            row.get("sample_id"),
            row.get("protocol_id"),
            _round_float(row.get("acquisition_cost")),
            _json_debug(row.get("condition_values", {})),
            _json_debug(row.get("observed_target_value")),
        )

        logger.warning(
            (
                "[AL DEBUG] round=%d | selected_score | "
                "current_val=%.8f | fantasy_mean_delta_val=%s | "
                "fantasy_delta_val_per_cost=%s | score=%s"
            ),
            int(round_idx),
            float(row.get("current_target_val_loss", current_target_val_loss)),
            _round_float(row.get("mean_improvement")),
            _round_float(row.get("fantasy_mean_improvement_per_cost", row.get("score"))),
            _round_float(row.get("score")),
        )

        if "fantasy_val_losses" in row:
            logger.warning(
                (
                    "[AL DEBUG] round=%d | selected_fantasies | "
                    "fantasy_val_losses=%s | fantasy_improvements=%s"
                ),
                int(round_idx),
                _json_debug([_round_float(x) for x in row.get("fantasy_val_losses", [])]),
                _json_debug([_round_float(x) for x in row.get("fantasy_improvements", [])]),
            )

    top_k = max(0, int(config.debug_top_k_candidates))
    if top_k == 0:
        return

    sortable = [
        row for row in enriched_selection_rows
        if row.get("score") is not None
    ]
    sortable = sorted(sortable, key=lambda row: float(row["score"]), reverse=True)

    for rank, row in enumerate(sortable[:top_k], start=1):
        logger.warning(
            (
                "[AL DEBUG] round=%d | candidate_rank=%d | "
                "candidate_id=%s | sample_id=%s | protocol_id=%s | cost=%s | "
                "mean_delta_val=%s | delta_per_cost=%s | score=%s | "
                "selected=%s | condition=%s"
            ),
            int(round_idx),
            int(rank),
            row.get("candidate_id"),
            row.get("sample_id"),
            row.get("protocol_id"),
            _round_float(row.get("acquisition_cost")),
            _round_float(row.get("mean_improvement")),
            _round_float(row.get("fantasy_mean_improvement_per_cost", row.get("score"))),
            _round_float(row.get("score")),
            bool(row.get("was_selected", False)),
            _json_debug(row.get("condition_values", {})),
        )


def _log_actual_after_acquisition_debug(
    *,
    config: ActiveLearningRunnerConfig,
    pending_selection: Optional[Dict[str, Any]],
    current_round_idx: int,
    current_target_val_loss: float,
    current_target_test_loss: float,
    selection_history: List[Dict[str, Any]],
) -> None:
    if pending_selection is None:
        return

    history_index = pending_selection.get("selection_history_index")

    before_val = float(pending_selection["before_target_val_loss"])
    before_test = float(pending_selection["before_target_test_loss"])
    after_val = float(current_target_val_loss)
    after_test = float(current_target_test_loss)
    cost = max(float(pending_selection["acquisition_cost"]), 1e-12)

    actual_delta_val = before_val - after_val
    actual_delta_test = before_test - after_test

    update = {
        "actual_after_round_idx": int(current_round_idx),
        "actual_after_target_val_loss": after_val,
        "actual_after_target_test_loss": after_test,
        "actual_delta_target_val_loss": actual_delta_val,
        "actual_delta_target_test_loss": actual_delta_test,
        "actual_delta_target_val_loss_per_cost": actual_delta_val / cost,
        "actual_delta_target_test_loss_per_cost": actual_delta_test / cost,
    }

    if isinstance(history_index, int) and 0 <= history_index < len(selection_history):
        selection_history[history_index].update(update)

    if bool(config.debug_progress):
        logger.warning(
            (
                "[AL DEBUG] round=%d | actual_after_previous_acquisition | "
                "previous_round=%d | candidate_id=%s | sample_id=%s | protocol_id=%s | "
                "cost=%.8f | "
                "val_before=%.8f | val_after=%.8f | actual_delta_val=%.8f | "
                "actual_delta_val_per_cost=%.8f | "
                "test_before=%.8f | test_after=%.8f | actual_delta_test=%.8f | "
                "actual_delta_test_per_cost=%.8f"
            ),
            int(current_round_idx),
            int(pending_selection["round_idx"]),
            pending_selection.get("candidate_id"),
            pending_selection.get("sample_id"),
            pending_selection.get("protocol_id"),
            float(cost),
            before_val,
            after_val,
            actual_delta_val,
            actual_delta_val / cost,
            before_test,
            after_test,
            actual_delta_test,
            actual_delta_test / cost,
        )


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
    Run protocol-level active learning.

    Loop per round
    --------------
    1. Build a fresh predictor from predictor_factory().
    2. Train it on current_train_samples.
    3. Measure target validation and target test loss.
    4. Build protocol query candidates from the remaining pool.
    5. Choose the next acquisition using the configured acquisition policy.
    6. Add the selected sample to the train set and update the budget.

    Debug output
    ------------
    If config.debug_progress=True, each round additionally logs:

    - pre-acquisition validation/test loss,
    - selected candidate,
    - candidate cost,
    - fantasy validation losses,
    - fantasy validation improvement,
    - fantasy improvement per cost,
    - actual validation/test improvement after the acquisition is incorporated.
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
    rng = random.Random(int(config.random_seed))

    pending_selection: Optional[Dict[str, Any]] = None

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

        _log_actual_after_acquisition_debug(
            config=config,
            pending_selection=pending_selection,
            current_round_idx=int(round_idx),
            current_target_val_loss=float(current_target_val_loss),
            current_target_test_loss=float(current_target_test_loss),
            selection_history=selection_history,
        )
        pending_selection = None

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

        chosen, chosen_score, chosen_mean_improvement, round_selection_history = (
            _select_candidate_by_policy(
                predictor,
                protocols=protocol_map,
                protocol_candidates=protocol_candidates,
                current_train_samples=current_train_samples,
                target_val_samples=target_val_samples,
                config=config,
                remaining_budget=float(remaining_budget),
                device=device,
                rng=rng,
            )
        )

        enriched_selection_history = _enrich_selection_rows(
            round_selection_history=round_selection_history,
            protocol_candidates=protocol_candidates,
            round_idx=int(round_idx),
            train_size_before=int(len(current_train_samples)),
            pool_size_before=int(len(current_candidate_pool)),
            spent_budget_before=float(spent_budget),
            remaining_budget_before=float(remaining_budget),
            current_target_val_loss=float(current_target_val_loss),
            current_target_test_loss=float(current_target_test_loss),
        )

        _log_candidate_scoring_debug(
            config=config,
            round_idx=int(round_idx),
            current_target_val_loss=float(current_target_val_loss),
            current_target_test_loss=float(current_target_test_loss),
            enriched_selection_rows=enriched_selection_history,
        )

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
                chosen_score=None if chosen_score is None else float(chosen_score),
                chosen_mean_improvement=(
                    None if chosen_mean_improvement is None else float(chosen_mean_improvement)
                ),
            )
        )

        selection_history_start = len(selection_history)
        selected_local_index = _selected_row_index(enriched_selection_history)
        selection_history.extend(enriched_selection_history)

        if chosen is None:
            break

        selected_history_index = None
        if selected_local_index is not None:
            selected_history_index = int(selection_history_start + selected_local_index)

        pending_selection = {
            "selection_history_index": selected_history_index,
            "round_idx": int(round_idx),
            "candidate_id": chosen.candidate_id,
            "sample_id": _sample_id(chosen.sample),
            "protocol_id": chosen.protocol_id,
            "acquisition_cost": float(chosen.acquisition_cost),
            "before_target_val_loss": float(current_target_val_loss),
            "before_target_test_loss": float(current_target_test_loss),
        }

        current_train_samples.append(chosen.sample)

        current_candidate_pool = [
            sample for sample in current_candidate_pool
            if _sample_id(sample) != _sample_id(chosen.sample)
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