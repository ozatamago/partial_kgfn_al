#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ofml_alfn.benchmarks.dataset_builders.build_problem_1a import (
    Problem1ADatasetBuildResult,
    Problem1ADatasetBuilderConfig,
    build_problem_1a_dataset,
)
from ofml_alfn.models.process_modules import (
    ActivationName,
    make_problem_1a_module_registry,
)
from ofml_alfn.models.protocol_predictor import (
    ProtocolPredictor,
    build_protocol_predictor,
)
from ofml_alfn.training.train_protocol_predictor import (
    ProtocolEvaluationResult,
    ProtocolTrainingConfig,
    ProtocolTrainingResult,
    evaluate_protocol_predictor,
    train_protocol_predictor,
)
from ofml_alfn.utils.protocol_types import BenchmarkSample, ProtocolSpec


def _as_dict(d: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {} if d is None else dict(d)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _clone_training_config_with_device(
    config: ProtocolTrainingConfig,
    device: Optional[str],
) -> ProtocolTrainingConfig:
    if device is None:
        return config
    return replace(config, device=device)


def _dataset_split_map(
    dataset_result: Problem1ADatasetBuildResult,
) -> Dict[str, Tuple[BenchmarkSample, ...]]:
    return {
        split.split_name: dataset_result.samples_in_split(split.split_name)
        for split in dataset_result.splits
    }


def _sum_observation_cost(samples: Sequence[BenchmarkSample]) -> float:
    total = 0.0
    for sample in samples:
        for obs in sample.observations:
            total += float(obs.cost)
    return float(total)


def _count_target_protocol_cost(
    samples: Sequence[BenchmarkSample],
    *,
    target_protocol_id: str,
) -> float:
    total = 0.0
    for sample in samples:
        if sample.protocol_id != target_protocol_id:
            continue
        for obs in sample.observations:
            total += float(obs.cost)
    return float(total)


def _set_module_trainable(module: nn.Module, is_trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(bool(is_trainable))


def _set_registry_keys_trainable(
    predictor: ProtocolPredictor,
    *,
    module_keys: Sequence[str],
    is_trainable: bool,
) -> None:
    for key in module_keys:
        if key not in predictor.modules_by_key:
            raise KeyError(
                f"Module key {key!r} not found in predictor registry. "
                f"Available keys: {sorted(predictor.modules_by_key.keys())}"
            )
        _set_module_trainable(predictor.modules_by_key[key], is_trainable=is_trainable)


def _build_problem_1a_predictor(
    *,
    dataset_result: Problem1ADatasetBuildResult,
    upstream_hidden_dims: Sequence[int],
    observer_hidden_dims: Sequence[int],
    activation: ActivationName,
    output_activation: ActivationName,
    dropout: float,
    dtype: torch.dtype,
    strict_registry: bool,
    device: Optional[str],
    seed: int,
) -> ProtocolPredictor:
    cfg = dataset_result.config
    _set_torch_seed(seed)

    modules = make_problem_1a_module_registry(
        input_dim=cfg.input_dim,
        latent_dim=cfg.latent_dim,
        output_dim=cfg.output_dim,
        observer_module_keys=cfg.observer_module_keys,
        upstream_hidden_dims=tuple(upstream_hidden_dims),
        observer_hidden_dims=tuple(observer_hidden_dims),
        activation=activation,
        output_activation=output_activation,
        dropout=dropout,
        dtype=dtype,
    )
    predictor = build_protocol_predictor(
        modules=modules,
        dtype=dtype,
        strict_registry=strict_registry,
        device=device,
    )
    return predictor


def _protocol_map_from_dataset(
    dataset_result: Problem1ADatasetBuildResult,
) -> Dict[str, ProtocolSpec]:
    return dataset_result.benchmark.protocol_map


@dataclass
class RunnerPhaseResult:
    phase_name: str
    training: ProtocolTrainingResult
    val_eval: Optional[ProtocolEvaluationResult]
    test_eval: Optional[ProtocolEvaluationResult]
    n_train_samples: int
    used_total_cost: float
    used_target_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "n_train_samples": self.n_train_samples,
            "used_total_cost": self.used_total_cost,
            "used_target_cost": self.used_target_cost,
            "final_train_loss": self.training.final_train_loss,
            "final_val_loss": self.training.final_val_loss,
            "best_val_loss": self.training.best_val_loss,
            "best_step": self.training.best_step,
            "test_loss": None if self.test_eval is None else self.test_eval.loss,
        }


@dataclass
class ScratchVsTransferResult:
    dataset: Problem1ADatasetBuildResult
    scratch: RunnerPhaseResult
    pretrain: RunnerPhaseResult
    adapt_after_pretrain: RunnerPhaseResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        target_protocol_id = self.dataset.benchmark.target_protocol_id
        return {
            "problem_name": "problem_1a",
            "target_protocol_id": target_protocol_id,
            "scratch": self.scratch.summary(),
            "pretrain": self.pretrain.summary(),
            "adapt_after_pretrain": self.adapt_after_pretrain.summary(),
            "scratch_test_loss": (
                None if self.scratch.test_eval is None else self.scratch.test_eval.loss
            ),
            "transfer_test_loss": (
                None
                if self.adapt_after_pretrain.test_eval is None
                else self.adapt_after_pretrain.test_eval.loss
            ),
            "scratch_target_cost": self.scratch.used_target_cost,
            "transfer_target_cost": self.adapt_after_pretrain.used_target_cost,
            "transfer_total_cost": (
                self.pretrain.used_total_cost + self.adapt_after_pretrain.used_total_cost
            ),
        }


@dataclass(frozen=True)
class PretrainingRunnerConfig:
    """
    End-to-end runner configuration for Problem 1_A.

    Behavior
    --------
    1. Build the synthetic dataset.
    2. Run scratch baseline on Protocol 3 only.
    3. Run pretraining on Protocols 1 and 2.
    4. Adapt the pretrained model on Protocol 3.
    5. Compare test loss on Protocol 3.

    Notes
    -----
    - Default dataset protocol costs are expected to live in
      Problem1ADatasetBuilderConfig.protocol_costs.
    - You can override those costs there, then pass that config here.
    """

    dataset_config: Problem1ADatasetBuilderConfig = field(
        default_factory=Problem1ADatasetBuilderConfig
    )

    scratch_train_config: ProtocolTrainingConfig = field(
        default_factory=lambda: ProtocolTrainingConfig(
            n_steps=500,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-6,
            loss_name="mse",
            grad_clip_norm=None,
            val_every=25,
            early_stopping_patience=20,
            early_stopping_min_delta=0.0,
            seed=0,
            device=None,
            verbose=False,
        )
    )
    pretrain_train_config: ProtocolTrainingConfig = field(
        default_factory=lambda: ProtocolTrainingConfig(
            n_steps=500,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-6,
            loss_name="mse",
            grad_clip_norm=None,
            val_every=50,
            early_stopping_patience=None,
            early_stopping_min_delta=0.0,
            seed=1,
            device=None,
            verbose=False,
        )
    )
    adapt_train_config: ProtocolTrainingConfig = field(
        default_factory=lambda: ProtocolTrainingConfig(
            n_steps=300,
            batch_size=64,
            lr=5e-4,
            weight_decay=1e-6,
            loss_name="mse",
            grad_clip_norm=None,
            val_every=25,
            early_stopping_patience=20,
            early_stopping_min_delta=0.0,
            seed=2,
            device=None,
            verbose=False,
        )
    )

    upstream_hidden_dims: Tuple[int, ...] = (64, 64)
    observer_hidden_dims: Tuple[int, ...] = (64, 64)
    activation: ActivationName = "relu"
    output_activation: ActivationName = "identity"
    dropout: float = 0.0

    dtype: torch.dtype = torch.float32
    strict_registry: bool = True

    scratch_model_seed: int = 100
    transfer_model_seed: int = 200

    freeze_shared_upstream_during_adapt: bool = False
    shared_upstream_module_key: str = "shared_upstream"

    device: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_problem_1a_scratch_phase(
    *,
    dataset_result: Problem1ADatasetBuildResult,
    config: PretrainingRunnerConfig,
) -> Tuple[ProtocolPredictor, RunnerPhaseResult]:
    split_map = _dataset_split_map(dataset_result)
    protocol_map = _protocol_map_from_dataset(dataset_result)
    target_protocol_id = dataset_result.benchmark.target_protocol_id

    predictor = _build_problem_1a_predictor(
        dataset_result=dataset_result,
        upstream_hidden_dims=config.upstream_hidden_dims,
        observer_hidden_dims=config.observer_hidden_dims,
        activation=config.activation,
        output_activation=config.output_activation,
        dropout=config.dropout,
        dtype=config.dtype,
        strict_registry=config.strict_registry,
        device=config.device,
        seed=config.scratch_model_seed,
    )

    train_samples = split_map["adapt_protocol_3"]
    val_samples = split_map["val_protocol_3"]
    test_samples = split_map["test_protocol_3"]

    scratch_train_cfg = _clone_training_config_with_device(
        config.scratch_train_config, config.device
    )
    training = train_protocol_predictor(
        predictor,
        protocols=protocol_map,
        train_samples=train_samples,
        val_samples=val_samples,
        config=scratch_train_cfg,
    )

    val_eval = evaluate_protocol_predictor(
        predictor,
        protocols=protocol_map,
        samples=val_samples,
        config=scratch_train_cfg,
    )
    test_eval = evaluate_protocol_predictor(
        predictor,
        protocols=protocol_map,
        samples=test_samples,
        config=scratch_train_cfg,
    )

    phase_result = RunnerPhaseResult(
        phase_name="scratch_protocol_3",
        training=training,
        val_eval=val_eval,
        test_eval=test_eval,
        n_train_samples=len(train_samples),
        used_total_cost=_sum_observation_cost(train_samples),
        used_target_cost=_count_target_protocol_cost(
            train_samples, target_protocol_id=target_protocol_id
        ),
        metadata={"train_split": "adapt_protocol_3"},
    )
    return predictor, phase_result


def run_problem_1a_pretrain_then_adapt(
    *,
    dataset_result: Problem1ADatasetBuildResult,
    config: PretrainingRunnerConfig,
) -> Tuple[ProtocolPredictor, RunnerPhaseResult, RunnerPhaseResult]:
    split_map = _dataset_split_map(dataset_result)
    protocol_map = _protocol_map_from_dataset(dataset_result)
    target_protocol_id = dataset_result.benchmark.target_protocol_id

    predictor = _build_problem_1a_predictor(
        dataset_result=dataset_result,
        upstream_hidden_dims=config.upstream_hidden_dims,
        observer_hidden_dims=config.observer_hidden_dims,
        activation=config.activation,
        output_activation=config.output_activation,
        dropout=config.dropout,
        dtype=config.dtype,
        strict_registry=config.strict_registry,
        device=config.device,
        seed=config.transfer_model_seed,
    )

    pretrain_samples = split_map["pretrain_all_sources"]
    adapt_samples = split_map["adapt_protocol_3"]
    val_samples = split_map["val_protocol_3"]
    test_samples = split_map["test_protocol_3"]

    pretrain_train_cfg = _clone_training_config_with_device(
        config.pretrain_train_config, config.device
    )
    adapt_train_cfg = _clone_training_config_with_device(
        config.adapt_train_config, config.device
    )

    pretrain_training = train_protocol_predictor(
        predictor,
        protocols=protocol_map,
        train_samples=pretrain_samples,
        val_samples=None,
        config=pretrain_train_cfg,
    )

    pretrain_phase = RunnerPhaseResult(
        phase_name="pretrain_sources",
        training=pretrain_training,
        val_eval=None,
        test_eval=None,
        n_train_samples=len(pretrain_samples),
        used_total_cost=_sum_observation_cost(pretrain_samples),
        used_target_cost=_count_target_protocol_cost(
            pretrain_samples, target_protocol_id=target_protocol_id
        ),
        metadata={"train_split": "pretrain_all_sources"},
    )

    if config.freeze_shared_upstream_during_adapt:
        _set_registry_keys_trainable(
            predictor,
            module_keys=(config.shared_upstream_module_key,),
            is_trainable=False,
        )

    adapt_training = train_protocol_predictor(
        predictor,
        protocols=protocol_map,
        train_samples=adapt_samples,
        val_samples=val_samples,
        config=adapt_train_cfg,
    )

    val_eval = evaluate_protocol_predictor(
        predictor,
        protocols=protocol_map,
        samples=val_samples,
        config=adapt_train_cfg,
    )
    test_eval = evaluate_protocol_predictor(
        predictor,
        protocols=protocol_map,
        samples=test_samples,
        config=adapt_train_cfg,
    )

    adapt_phase = RunnerPhaseResult(
        phase_name="adapt_protocol_3_after_pretrain",
        training=adapt_training,
        val_eval=val_eval,
        test_eval=test_eval,
        n_train_samples=len(adapt_samples),
        used_total_cost=_sum_observation_cost(adapt_samples),
        used_target_cost=_count_target_protocol_cost(
            adapt_samples, target_protocol_id=target_protocol_id
        ),
        metadata={
            "train_split": "adapt_protocol_3",
            "freeze_shared_upstream_during_adapt": config.freeze_shared_upstream_during_adapt,
        },
    )

    return predictor, pretrain_phase, adapt_phase


def run_problem_1a_pretraining_comparison(
    config: Optional[PretrainingRunnerConfig] = None,
    *,
    dataset_result: Optional[Problem1ADatasetBuildResult] = None,
) -> ScratchVsTransferResult:
    """
    Run the full scratch-vs-pretrain comparison for Problem 1_A.

    Returns
    -------
    ScratchVsTransferResult
        Contains:
        - scratch baseline result
        - pretraining phase result
        - adaptation-after-pretraining result
    """
    config = PretrainingRunnerConfig() if config is None else config

    if dataset_result is None:
        dataset_result = build_problem_1a_dataset(config.dataset_config)

    _, scratch_phase = run_problem_1a_scratch_phase(
        dataset_result=dataset_result,
        config=config,
    )
    _, pretrain_phase, adapt_phase = run_problem_1a_pretrain_then_adapt(
        dataset_result=dataset_result,
        config=config,
    )

    return ScratchVsTransferResult(
        dataset=dataset_result,
        scratch=scratch_phase,
        pretrain=pretrain_phase,
        adapt_after_pretrain=adapt_phase,
        metadata=_as_dict(config.metadata),
    )


__all__ = [
    "RunnerPhaseResult",
    "ScratchVsTransferResult",
    "PretrainingRunnerConfig",
    "run_problem_1a_scratch_phase",
    "run_problem_1a_pretrain_then_adapt",
    "run_problem_1a_pretraining_comparison",
]