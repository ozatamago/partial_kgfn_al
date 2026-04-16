#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


def _as_dict(d: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {} if d is None else dict(d)


def _infer_device(
    predictor: nn.Module,
    explicit_device: Optional[torch.device] = None,
) -> torch.device:
    if explicit_device is not None:
        return explicit_device
    try:
        return next(predictor.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_protocol_map(
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
) -> Dict[str, ProtocolSpec]:
    if isinstance(protocols, Mapping):
        protocol_map = dict(protocols)
    else:
        protocol_map = {p.protocol_id: p for p in protocols}
    if not protocol_map:
        raise ValueError("protocols must be non-empty")
    return protocol_map


def _stack_condition_tensor(
    samples: Sequence[BenchmarkSample],
    protocol: ProtocolSpec,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for sample in samples:
        row = [float(sample.condition.values[k]) for k in protocol.condition_keys]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _stack_target_tensor(
    samples: Sequence[BenchmarkSample],
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for sample in samples:
        value = sample.target_value
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        value = value.detach().to(dtype=torch.float32, device=device)
        if value.ndim == 0:
            value = value.unsqueeze(0)
        elif value.ndim > 1:
            value = value.reshape(-1)
        rows.append(value)
    return torch.stack(rows, dim=0)


def _extract_prediction_tensor(
    pred_obj: Any,
    *,
    device: torch.device,
    protocol_id: str,
) -> torch.Tensor:
    """
    Accept several output shapes so this trainer stays decoupled from the exact
    implementation of protocol_predictor.py or any custom predictor adapter.

    Supported forms
    ---------------
    1. Tensor directly
    2. Dict with one of:
       - "target"
       - "target_pred"
       - "prediction"
       - "pred"
    3. Object with one of:
       - .target
       - .target_pred
       - .prediction
       - .pred
       - .target_value
       - .mean
    """
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

    candidate = candidate.to(dtype=torch.float32, device=device)
    if candidate.ndim == 0:
        candidate = candidate.view(1, 1)
    elif candidate.ndim == 1:
        candidate = candidate.unsqueeze(-1)
    return candidate


def _predict_target_batch(
    predictor: nn.Module,
    *,
    protocol: ProtocolSpec,
    condition_x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Try several interfaces so the trainer remains usable while predictor
    implementations evolve.

    Expected preferred interfaces
    -----------------------------
    1. predictor.forward_target(protocol=protocol, condition_x=condition_x)
    2. predictor.forward_protocol(protocol=protocol, condition_x=condition_x)
    3. predictor(protocol=protocol, condition_x=condition_x)
    4. predictor(protocol, condition_x)
    """
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


def _group_samples_by_protocol(
    samples: Sequence[BenchmarkSample],
) -> Dict[str, List[BenchmarkSample]]:
    grouped: Dict[str, List[BenchmarkSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.protocol_id, []).append(sample)
    return grouped


def _compute_batch_loss(
    predictor: nn.Module,
    *,
    protocol_map: Mapping[str, ProtocolSpec],
    batch_samples: Sequence[BenchmarkSample],
    device: torch.device,
    protocol_loss_weights: Optional[Mapping[str, float]] = None,
    loss_name: str = "mse",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if len(batch_samples) == 0:
        raise ValueError("batch_samples must be non-empty")

    grouped = _group_samples_by_protocol(batch_samples)
    protocol_loss_weights = {} if protocol_loss_weights is None else dict(protocol_loss_weights)

    total_weight = 0.0
    total_loss = None
    metrics: Dict[str, float] = {}

    for protocol_id, samples_this_protocol in grouped.items():
        if protocol_id not in protocol_map:
            raise KeyError(f"Unknown protocol_id in batch: {protocol_id!r}")

        protocol = protocol_map[protocol_id]
        x = _stack_condition_tensor(samples_this_protocol, protocol, device=device)
        y = _stack_target_tensor(samples_this_protocol, device=device)
        pred = _predict_target_batch(
            predictor,
            protocol=protocol,
            condition_x=x,
            device=device,
        )

        if pred.shape != y.shape:
            if pred.numel() == y.numel():
                pred = pred.view_as(y)
            else:
                raise ValueError(
                    f"Prediction shape {tuple(pred.shape)} does not match target shape "
                    f"{tuple(y.shape)} for protocol {protocol_id!r}"
                )

        if loss_name == "mse":
            loss_this = F.mse_loss(pred, y)
        elif loss_name == "l1":
            loss_this = F.l1_loss(pred, y)
        elif loss_name == "smooth_l1":
            loss_this = F.smooth_l1_loss(pred, y)
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name!r}")

        weight = float(protocol_loss_weights.get(protocol_id, 1.0))
        weighted_loss_this = weight * loss_this
        total_loss = weighted_loss_this if total_loss is None else total_loss + weighted_loss_this
        total_weight += weight

        metrics[f"loss/{protocol_id}"] = float(loss_this.detach().cpu().item())
        metrics[f"n/{protocol_id}"] = float(len(samples_this_protocol))
        metrics[f"weight/{protocol_id}"] = weight

    if total_loss is None or total_weight <= 0.0:
        raise RuntimeError("Failed to compute training loss")

    total_loss = total_loss / total_weight
    metrics["loss/total"] = float(total_loss.detach().cpu().item())
    metrics["n/total"] = float(len(batch_samples))
    return total_loss, metrics


def _sample_minibatch(
    samples: Sequence[BenchmarkSample],
    *,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> List[BenchmarkSample]:
    n = len(samples)
    if n == 0:
        raise ValueError("Cannot sample minibatch from an empty sample list")
    if batch_size >= n:
        return list(samples)

    idx = torch.randint(
        low=0,
        high=n,
        size=(batch_size,),
        generator=generator,
    ).tolist()
    return [samples[i] for i in idx]


def _predictor_type(predictor: nn.Module) -> str:
    return str(getattr(predictor, "predictor_type", "generic")).lower()


def _call_custom_protocol_evaluator(
    predictor: nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec],
    samples: Sequence[BenchmarkSample],
    config: "ProtocolTrainingConfig",
) -> Optional["ProtocolEvaluationResult"]:
    """
    Optional predictor-specific evaluation hook.

    Accepted predictor methods
    --------------------------
    - predictor.evaluate_protocol_dataset(...)
    - predictor.evaluate_protocol_predictor(...)

    Accepted return forms
    ---------------------
    - ProtocolEvaluationResult
    - dict with keys: loss, loss_by_protocol, n_by_protocol, n_total
    """
    for method_name in ("evaluate_protocol_dataset", "evaluate_protocol_predictor"):
        method = getattr(predictor, method_name, None)
        if callable(method):
            out = method(
                protocols=protocols,
                samples=samples,
                config=config,
            )
            if isinstance(out, ProtocolEvaluationResult):
                return out
            if isinstance(out, Mapping):
                return ProtocolEvaluationResult(
                    loss=float(out["loss"]),
                    loss_by_protocol=dict(out.get("loss_by_protocol", {})),
                    n_by_protocol={k: int(v) for k, v in out.get("n_by_protocol", {}).items()},
                    n_total=int(out.get("n_total", len(samples))),
                )
            raise TypeError(
                f"Custom evaluator {method_name} returned unsupported type {type(out)}"
            )
    return None


def _call_custom_protocol_trainer(
    predictor: nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec],
    train_samples: Sequence[BenchmarkSample],
    val_samples: Optional[Sequence[BenchmarkSample]],
    config: "ProtocolTrainingConfig",
    optimizer: Optional[torch.optim.Optimizer],
) -> Optional["ProtocolTrainingResult"]:
    """
    Optional predictor-specific training hook.

    Accepted predictor methods
    --------------------------
    - predictor.fit_protocol_dataset(...)
    - predictor.train_protocol_dataset(...)

    Accepted return forms
    ---------------------
    - ProtocolTrainingResult
    - dict with keys compatible with ProtocolTrainingResult
    """
    for method_name in ("fit_protocol_dataset", "train_protocol_dataset"):
        method = getattr(predictor, method_name, None)
        if callable(method):
            out = method(
                protocols=protocols,
                train_samples=train_samples,
                val_samples=val_samples,
                config=config,
                optimizer=optimizer,
            )
            if isinstance(out, ProtocolTrainingResult):
                return out
            if isinstance(out, Mapping):
                return ProtocolTrainingResult(
                    optimizer=out.get("optimizer", optimizer),
                    history=list(out.get("history", [])),
                    best_step=int(out.get("best_step", -1)),
                    best_val_loss=(
                        None if out.get("best_val_loss", None) is None
                        else float(out["best_val_loss"])
                    ),
                    final_train_loss=float(out.get("final_train_loss", float("nan"))),
                    final_val_loss=(
                        None if out.get("final_val_loss", None) is None
                        else float(out["final_val_loss"])
                    ),
                    best_state_dict=out.get("best_state_dict", None),
                )
            raise TypeError(
                f"Custom trainer {method_name} returned unsupported type {type(out)}"
            )
    return None


@dataclass(frozen=True)
class ProtocolTrainingConfig:
    n_steps: int = 500
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    loss_name: str = "mse"
    grad_clip_norm: Optional[float] = None
    val_every: int = 25
    early_stopping_patience: Optional[int] = 20
    early_stopping_min_delta: float = 0.0
    protocol_loss_weights: Dict[str, float] = field(default_factory=dict)
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = False

    def torch_device(self) -> Optional[torch.device]:
        return None if self.device is None else torch.device(self.device)


@dataclass
class ProtocolEvaluationResult:
    loss: float
    loss_by_protocol: Dict[str, float]
    n_by_protocol: Dict[str, int]
    n_total: int


@dataclass
class ProtocolTrainingResult:
    optimizer: Optional[torch.optim.Optimizer]
    history: List[Dict[str, float]]
    best_step: int
    best_val_loss: Optional[float]
    final_train_loss: float
    final_val_loss: Optional[float]
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None


def make_optimizer(
    predictor: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> torch.optim.Optimizer:
    params = [p for p in predictor.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("predictor has no trainable parameters")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_protocol_predictor(
    predictor: nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    samples: Sequence[BenchmarkSample],
    config: Optional[ProtocolTrainingConfig] = None,
) -> ProtocolEvaluationResult:
    if len(samples) == 0:
        return ProtocolEvaluationResult(
            loss=float("nan"),
            loss_by_protocol={},
            n_by_protocol={},
            n_total=0,
        )

    protocol_map = _normalize_protocol_map(protocols)
    config = ProtocolTrainingConfig() if config is None else config

    custom_eval = _call_custom_protocol_evaluator(
        predictor,
        protocols=protocol_map,
        samples=samples,
        config=config,
    )
    if custom_eval is not None:
        return custom_eval

    if _predictor_type(predictor) == "dkl":
        raise NotImplementedError(
            "predictor_type='dkl' requires a predictor-specific evaluation hook "
            "(evaluate_protocol_dataset or evaluate_protocol_predictor). "
            "The generic tensor-loss evaluator is not sufficient for the current DKL backend."
        )

    device = _infer_device(predictor, config.torch_device())
    predictor.eval()

    grouped = _group_samples_by_protocol(samples)

    loss_by_protocol: Dict[str, float] = {}
    n_by_protocol: Dict[str, int] = {}
    total_weighted_loss = 0.0
    total_count = 0

    for protocol_id, samples_this_protocol in grouped.items():
        protocol = protocol_map[protocol_id]
        x = _stack_condition_tensor(samples_this_protocol, protocol, device=device)
        y = _stack_target_tensor(samples_this_protocol, device=device)
        pred = _predict_target_batch(
            predictor,
            protocol=protocol,
            condition_x=x,
            device=device,
        )

        if pred.shape != y.shape:
            if pred.numel() == y.numel():
                pred = pred.view_as(y)
            else:
                raise ValueError(
                    f"Prediction shape {tuple(pred.shape)} does not match target shape "
                    f"{tuple(y.shape)} for protocol {protocol_id!r}"
                )

        if config.loss_name == "mse":
            loss_this = F.mse_loss(pred, y)
        elif config.loss_name == "l1":
            loss_this = F.l1_loss(pred, y)
        elif config.loss_name == "smooth_l1":
            loss_this = F.smooth_l1_loss(pred, y)
        else:
            raise ValueError(f"Unsupported loss_name: {config.loss_name!r}")

        n_this = len(samples_this_protocol)
        loss_val = float(loss_this.detach().cpu().item())
        loss_by_protocol[protocol_id] = loss_val
        n_by_protocol[protocol_id] = n_this

        total_weighted_loss += n_this * loss_val
        total_count += n_this

    total_loss = total_weighted_loss / max(total_count, 1)
    return ProtocolEvaluationResult(
        loss=float(total_loss),
        loss_by_protocol=loss_by_protocol,
        n_by_protocol=n_by_protocol,
        n_total=total_count,
    )


def train_protocol_predictor(
    predictor: nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    train_samples: Sequence[BenchmarkSample],
    val_samples: Optional[Sequence[BenchmarkSample]] = None,
    config: Optional[ProtocolTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> ProtocolTrainingResult:
    """
    Generic trainer for protocol-aware predictors.

    Minimal expected predictor interface
    ------------------------------------
    Preferred:
        predictor.forward_target(protocol=protocol_spec, condition_x=x)

    Accepted fallbacks:
        predictor.forward_protocol(protocol=protocol_spec, condition_x=x)
        predictor(protocol=protocol_spec, condition_x=x)
        predictor(protocol_spec, x)

    The returned object may be:
    - a tensor
    - a dict containing "target", "target_pred", "prediction", or "pred"
    - an object exposing one of those attributes

    Predictor-specific hook path
    ----------------------------
    If the predictor implements either:
      - fit_protocol_dataset(...)
      - train_protocol_dataset(...)
    this function will delegate to that method. This is the intended route for
    DKL-like predictors that need specialized fitting logic.
    """
    if len(train_samples) == 0:
        raise ValueError("train_samples must be non-empty")

    protocol_map = _normalize_protocol_map(protocols)
    config = ProtocolTrainingConfig() if config is None else config

    custom_train = _call_custom_protocol_trainer(
        predictor,
        protocols=protocol_map,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        optimizer=optimizer,
    )
    if custom_train is not None:
        return custom_train

    if _predictor_type(predictor) == "dkl":
        raise NotImplementedError(
            "predictor_type='dkl' requires a predictor-specific training hook "
            "(fit_protocol_dataset or train_protocol_dataset). "
            "The generic gradient loop is not sufficient for the current DKL backend."
        )

    device = _infer_device(predictor, config.torch_device())
    predictor.to(device)
    predictor.train()

    if optimizer is None:
        optimizer = make_optimizer(
            predictor,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    history: List[Dict[str, float]] = []
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss: Optional[float] = None
    best_step = -1
    bad_val_counter = 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))

    last_train_loss = float("nan")
    last_val_loss: Optional[float] = None

    for step in range(1, config.n_steps + 1):
        batch_samples = _sample_minibatch(
            train_samples,
            batch_size=min(config.batch_size, len(train_samples)),
            generator=generator,
        )

        predictor.train()
        optimizer.zero_grad()

        loss, train_metrics = _compute_batch_loss(
            predictor,
            protocol_map=protocol_map,
            batch_samples=batch_samples,
            device=device,
            protocol_loss_weights=config.protocol_loss_weights,
            loss_name=config.loss_name,
        )

        loss.backward()

        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                predictor.parameters(),
                max_norm=float(config.grad_clip_norm),
            )

        optimizer.step()
        last_train_loss = float(loss.detach().cpu().item())

        row: Dict[str, float] = {
            "step": float(step),
            "train_loss": last_train_loss,
        }
        row.update(train_metrics)

        should_validate = (
            val_samples is not None
            and len(val_samples) > 0
            and (step % config.val_every == 0 or step == config.n_steps)
        )

        if should_validate:
            val_result = evaluate_protocol_predictor(
                predictor,
                protocols=protocol_map,
                samples=val_samples,
                config=config,
            )
            last_val_loss = float(val_result.loss)
            row["val_loss"] = last_val_loss

            for protocol_id, loss_val in val_result.loss_by_protocol.items():
                row[f"val_loss/{protocol_id}"] = float(loss_val)

            improved = (
                best_val_loss is None
                or last_val_loss < best_val_loss - float(config.early_stopping_min_delta)
            )

            if improved:
                best_val_loss = last_val_loss
                best_step = step
                best_state_dict = copy.deepcopy(predictor.state_dict())
                bad_val_counter = 0
            else:
                bad_val_counter += 1

            if (
                config.early_stopping_patience is not None
                and bad_val_counter >= int(config.early_stopping_patience)
            ):
                history.append(row)
                if config.verbose:
                    print(
                        f"[train_protocol_predictor] early stop at step={step} "
                        f"best_step={best_step} best_val_loss={best_val_loss:.6f}"
                    )
                break

        history.append(row)

        if config.verbose and (step == 1 or step % max(1, config.val_every) == 0):
            msg = f"[train_protocol_predictor] step={step} train_loss={last_train_loss:.6f}"
            if last_val_loss is not None:
                msg += f" val_loss={last_val_loss:.6f}"
            print(msg)

    if best_state_dict is not None and val_samples is not None and len(val_samples) > 0:
        predictor.load_state_dict(best_state_dict)
        final_val_eval = evaluate_protocol_predictor(
            predictor,
            protocols=protocol_map,
            samples=val_samples,
            config=config,
        )
        last_val_loss = float(final_val_eval.loss)

    return ProtocolTrainingResult(
        optimizer=optimizer,
        history=history,
        best_step=best_step,
        best_val_loss=best_val_loss,
        final_train_loss=last_train_loss,
        final_val_loss=last_val_loss,
        best_state_dict=best_state_dict,
    )


def fit_protocol_predictor_on_named_splits(
    predictor: nn.Module,
    *,
    protocols: Mapping[str, ProtocolSpec] | Sequence[ProtocolSpec],
    split_map: Mapping[str, Sequence[BenchmarkSample]],
    train_split_names: Sequence[str],
    val_split_names: Optional[Sequence[str]] = None,
    config: Optional[ProtocolTrainingConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> ProtocolTrainingResult:
    train_samples: List[BenchmarkSample] = []
    for split_name in train_split_names:
        if split_name not in split_map:
            raise KeyError(f"Unknown train split_name: {split_name!r}")
        train_samples.extend(split_map[split_name])

    val_samples: Optional[List[BenchmarkSample]] = None
    if val_split_names is not None:
        val_samples = []
        for split_name in val_split_names:
            if split_name not in split_map:
                raise KeyError(f"Unknown val split_name: {split_name!r}")
            val_samples.extend(split_map[split_name])

    return train_protocol_predictor(
        predictor,
        protocols=protocols,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        optimizer=optimizer,
    )


__all__ = [
    "ProtocolTrainingConfig",
    "ProtocolEvaluationResult",
    "ProtocolTrainingResult",
    "make_optimizer",
    "evaluate_protocol_predictor",
    "train_protocol_predictor",
    "fit_protocol_predictor_on_named_splits",
]