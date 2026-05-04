#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from ofml_alfn.benchmarks.dataset_builders.build_problem_1a import (
    Problem1ADatasetBuilderConfig,
    Problem1ADatasetBuildResult,
    build_problem_1a_dataset,
)
from ofml_alfn.configs.fantasy_protocol1a import get_fantasy_protocol1a_options
from ofml_alfn.models.nodewise_dkl import MultiHeadNodewiseDKL
from ofml_alfn.models.process_modules import make_problem_1a_module_registry
from ofml_alfn.models.protocol_predictor import build_protocol_predictor
from ofml_alfn.runners.active_learning_runner import (
    ActiveLearningRunnerConfig,
    run_protocol_active_learning,
)
from ofml_alfn.runners.sequential_target_adapt_runner import (
    SequentialTargetAdaptRunnerConfig,
    run_sequential_target_adapt,
)
from ofml_alfn.training.pretrain_then_adapt import (
    PretrainThenAdaptConfig,
    run_pretrain_then_adapt,
)
from ofml_alfn.training.train_protocol_predictor import ProtocolTrainingConfig
from ofml_alfn.utils.protocol_types import BenchmarkSample

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Problem 1A experiments with mode switching."
    )

    parser.add_argument(
        "--experiment_mode",
        type=str,
        default="fantasy_al",
        choices=[
            "fantasy_al",
            "pretrain_then_adapt",
            "scratch_then_sequential_adapt",
            "pretrain_then_sequential_adapt",
            "family3_candidate_pool_ablation",
        ],
    )
    parser.add_argument(
        "--debug_top_k_candidates",
        type=int,
        default=5,
        help="Number of top scored acquisition candidates to print per active-learning round.",
    )

    parser.add_argument(
        "--target_acquisition_policy",
        type=str,
        default="random",
        choices=["random", "local_uncertainty", "fantasy"],
    )
    parser.add_argument("--target_adapt_budget", type=int, default=30)

    parser.add_argument(
        "--candidate_pool_scope",
        type=str,
        default="all_protocols",
        choices=["all_protocols", "target_only"],
        help=(
            "Candidate pool scope for family3_candidate_pool_ablation. "
            "all_protocols uses pool_p1 + pool_p2 + pool_p3. "
            "target_only uses pool_p3 only."
        ),
    )

    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--budget", type=float, default=40.0)

    parser.add_argument(
        "--protocol_costs",
        type=float,
        nargs=3,
        default=[1.0, 2.0, 3.0],
        metavar=("C1", "C2", "C3"),
    )
    parser.add_argument(
        "--similarities_to_target",
        type=float,
        nargs=3,
        default=[0.4, 0.7, 1.0],
        metavar=("S1", "S2", "S3"),
    )
    parser.add_argument(
        "--observer_scales",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("G1", "G2", "G3"),
    )
    parser.add_argument("--target_noise_std", type=float, default=0.0)
    parser.add_argument("--source_noise_stds", type=float, nargs=2, default=[0.0, 0.0])

    parser.add_argument("--n_pretrain_p1", type=int, default=128)
    parser.add_argument("--n_pretrain_p2", type=int, default=128)
    parser.add_argument("--n_adapt_p3", type=int, default=32)
    parser.add_argument("--n_val_p3", type=int, default=128)
    parser.add_argument("--n_test_p3", type=int, default=256)

    parser.add_argument("--n_init_p1", type=int, default=8)
    parser.add_argument("--n_init_p2", type=int, default=8)
    parser.add_argument("--n_init_p3", type=int, default=4)

    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--predictor_type",
        type=str,
        default="mcd",
        choices=["mcd", "dkl"],
    )

    parser.add_argument("--dkl_hidden", type=int, default=256)
    parser.add_argument("--dkl_feature_dim", type=int, default=32)
    parser.add_argument("--dkl_kernel", type=str, default="rbf", choices=["rbf", "matern"])
    parser.add_argument("--dkl_inference", type=str, default="exact", choices=["exact", "svdkl"])
    parser.add_argument("--dkl_noise", type=float, default=1e-4)

    parser.add_argument("--outer_train_steps", type=int, default=500)
    parser.add_argument("--outer_batch_size", type=int, default=64)
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--outer_weight_decay", type=float, default=1e-6)
    parser.add_argument("--outer_val_every", type=int, default=25)
    parser.add_argument("--outer_patience", type=int, default=20)

    parser.add_argument("--fantasy_mc_samples", type=int, default=8)
    parser.add_argument("--fantasy_train_steps", type=int, default=20)
    parser.add_argument("--fantasy_batch_size", type=int, default=32)
    parser.add_argument("--fantasy_lr", type=float, default=5e-4)
    parser.add_argument("--fantasy_weight_decay", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/protocol1a")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--debug_progress",
        action="store_true",
        help=(
            "Print additional progress information: dataset split sizes, "
            "training configs, history tail, and acquisition selections."
        ),
    )
    parser.add_argument(
        "--debug_history_tail",
        type=int,
        default=20,
        help="Number of recent history records to print when debug_progress is enabled.",
    )
    parser.add_argument(
        "--fantasy_verbose",
        action="store_true",
        help=(
            "Enable verbose logging inside fantasy retraining config. "
            "Useful when target_acquisition_policy=fantasy."
        ),
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    policy = str(args.target_acquisition_policy).lower()
    mode = str(args.experiment_mode)
    predictor_type = str(args.predictor_type).lower()
    candidate_pool_scope = str(args.candidate_pool_scope)

    if policy not in {"random", "local_uncertainty", "fantasy"}:
        raise ValueError(
            f"Unsupported target_acquisition_policy: {policy}. "
            "Expected one of: random, local_uncertainty, fantasy."
        )

    if mode not in {
        "fantasy_al",
        "pretrain_then_adapt",
        "scratch_then_sequential_adapt",
        "pretrain_then_sequential_adapt",
        "family3_candidate_pool_ablation",
    }:
        raise ValueError(f"Unsupported experiment_mode: {mode}")

    if candidate_pool_scope not in {"all_protocols", "target_only"}:
        raise ValueError(
            f"Unsupported candidate_pool_scope: {candidate_pool_scope}. "
            "Expected one of: all_protocols, target_only."
        )

    if policy == "local_uncertainty" and predictor_type == "dkl":
        raise ValueError(
            "target_acquisition_policy=local_uncertainty currently requires "
            "predictor_type='mcd' unless you have implemented a predictor-side "
            "uncertainty hook for DKL."
        )

    if len(args.observer_scales) != 3:
        raise ValueError(
            f"--observer_scales must have length 3, got {args.observer_scales}"
        )
    for i, scale in enumerate(args.observer_scales):
        if float(scale) <= 0.0:
            raise ValueError(
                f"observer_scales[{i}] must be positive, got {scale}"
            )

    if len(args.source_noise_stds) != 2:
        raise ValueError(
            f"--source_noise_stds must have length 2, got {args.source_noise_stds}"
        )
    if float(args.target_noise_std) < 0.0:
        raise ValueError(
            f"--target_noise_std must be non-negative, got {args.target_noise_std}"
        )
    for i, noise_std in enumerate(args.source_noise_stds):
        if float(noise_std) < 0.0:
            raise ValueError(
                f"source_noise_stds[{i}] must be non-negative, got {noise_std}"
            )

    if int(args.debug_history_tail) < 0:
        raise ValueError(
            f"--debug_history_tail must be non-negative, got {args.debug_history_tail}"
        )


def _debug_enabled(args: argparse.Namespace) -> bool:
    return bool(args.debug_progress or args.verbose)


def _debug_log(args: argparse.Namespace, message: str, *values: Any) -> None:
    if _debug_enabled(args):
        logger.warning(message, *values)


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _format_maybe_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value


def _sample_protocol_id(sample: BenchmarkSample) -> str:
    value = getattr(sample, "protocol_id", None)
    if value is None and isinstance(sample, Mapping):
        value = sample.get("protocol_id")
    if value is None:
        return "unknown"
    return str(value)


def _sample_id(sample: BenchmarkSample) -> str:
    for name in ("sample_id", "id", "condition_id"):
        value = getattr(sample, name, None)
        if value is not None:
            return str(value)
        if isinstance(sample, Mapping) and name in sample:
            return str(sample[name])
    return "unknown"


def _protocol_counts(samples: Sequence[BenchmarkSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        pid = _sample_protocol_id(sample)
        counts[pid] = counts.get(pid, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _log_sample_split_debug(
    args: argparse.Namespace,
    *,
    name: str,
    samples: Sequence[BenchmarkSample],
) -> None:
    if not _debug_enabled(args):
        return

    logger.warning(
        "[debug] split=%s | n=%d | protocol_counts=%s",
        name,
        len(samples),
        _json_dumps_compact(_protocol_counts(samples)),
    )


def _log_initial_pool_debug(
    args: argparse.Namespace,
    *,
    init_p1: Sequence[BenchmarkSample],
    pool_p1: Sequence[BenchmarkSample],
    init_p2: Sequence[BenchmarkSample],
    pool_p2: Sequence[BenchmarkSample],
    init_p3: Sequence[BenchmarkSample],
    pool_p3: Sequence[BenchmarkSample],
) -> None:
    if not _debug_enabled(args):
        return

    logger.warning("[debug] initial/pool split summary")
    _log_sample_split_debug(args, name="init_p1", samples=init_p1)
    _log_sample_split_debug(args, name="pool_p1", samples=pool_p1)
    _log_sample_split_debug(args, name="init_p2", samples=init_p2)
    _log_sample_split_debug(args, name="pool_p2", samples=pool_p2)
    _log_sample_split_debug(args, name="init_p3", samples=init_p3)
    _log_sample_split_debug(args, name="pool_p3", samples=pool_p3)


def _training_config_debug_dict(cfg: ProtocolTrainingConfig) -> Dict[str, Any]:
    raw = asdict(cfg)
    keep_keys = [
        "n_steps",
        "batch_size",
        "lr",
        "weight_decay",
        "loss_name",
        "grad_clip_norm",
        "val_every",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "seed",
        "device",
        "verbose",
    ]
    return {key: raw.get(key) for key in keep_keys if key in raw}


def _log_training_config_debug(
    args: argparse.Namespace,
    *,
    outer_cfg: ProtocolTrainingConfig,
    fantasy_cfg: ProtocolTrainingConfig,
) -> None:
    if not _debug_enabled(args):
        return

    logger.warning(
        "[debug] outer_train_config=%s",
        _json_dumps_compact(_training_config_debug_dict(outer_cfg)),
    )
    logger.warning(
        "[debug] fantasy_train_config=%s",
        _json_dumps_compact(_training_config_debug_dict(fantasy_cfg)),
    )


def _compact_history_row(row: Dict[str, Any]) -> Dict[str, Any]:
    preferred_keys = [
        "round",
        "round_idx",
        "step",
        "n_train",
        "n_pool",
        "spent_budget",
        "selected_protocol_id",
        "selected_sample_id",
        "selected_cost",
        "acquisition_score",
        "train_loss",
        "val_loss",
        "test_loss",
        "target_train_loss",
        "target_val_loss",
        "target_test_loss",
    ]

    compact: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in row:
            compact[key] = _format_maybe_float(row[key])

    if len(compact) == 0:
        for key, value in row.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                compact[key] = _format_maybe_float(value)

    return compact


def _log_history_progress(
    args: argparse.Namespace,
    *,
    label: str,
    history_dicts: Sequence[Dict[str, Any]],
) -> None:
    if not _debug_enabled(args):
        return

    n_history = len(history_dicts)
    logger.warning("[debug] %s history length=%d", label, n_history)

    if n_history == 0:
        return

    tail = int(args.debug_history_tail)
    if tail == 0:
        return

    logger.warning("[debug] %s history tail, last %d rows", label, min(tail, n_history))
    for i, row in enumerate(history_dicts[-tail:], start=max(0, n_history - tail)):
        logger.warning(
            "[debug] %s history[%d]=%s",
            label,
            i,
            _json_dumps_compact(_compact_history_row(row)),
        )


def _compact_selection_row(row: Dict[str, Any]) -> Dict[str, Any]:
    preferred_keys = [
        "round",
        "round_idx",
        "selected_index",
        "selected_sample_id",
        "selected_protocol_id",
        "selected_cost",
        "spent_budget_before",
        "spent_budget_after",
        "acquisition_score",
        "policy",
        "candidate_pool_size_before",
        "candidate_pool_size_after",
    ]

    compact: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in row:
            compact[key] = _format_maybe_float(row[key])

    if len(compact) == 0:
        for key, value in row.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                compact[key] = _format_maybe_float(value)

    return compact


def _log_selection_progress(
    args: argparse.Namespace,
    *,
    label: str,
    selection_history: Sequence[Dict[str, Any]],
) -> None:
    if not _debug_enabled(args):
        return

    n_history = len(selection_history)
    logger.warning("[debug] %s selection_history length=%d", label, n_history)

    if n_history == 0:
        return

    tail = int(args.debug_history_tail)
    if tail == 0:
        return

    logger.warning(
        "[debug] %s selection_history tail, last %d rows",
        label,
        min(tail, n_history),
    )
    for i, row in enumerate(selection_history[-tail:], start=max(0, n_history - tail)):
        logger.warning(
            "[debug] %s selection_history[%d]=%s",
            label,
            i,
            _json_dumps_compact(_compact_selection_row(row)),
        )


def _log_final_loss_debug(
    args: argparse.Namespace,
    *,
    label: str,
    summary: Dict[str, Any],
) -> None:
    if not _debug_enabled(args):
        return

    keys = [
        "final_target_val_loss",
        "final_target_test_loss",
        "spent_budget",
        "n_rounds_completed",
        "n_train_initial",
        "n_train_final",
        "n_pool_initial",
        "n_pool_final",
        "n_target_points_used",
        "target_cost_used",
    ]

    compact = {
        key: _format_maybe_float(summary[key])
        for key in keys
        if key in summary
    }
    logger.warning("[debug] %s final_summary=%s", label, _json_dumps_compact(compact))


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _split_initial_and_pool(
    samples: Sequence[BenchmarkSample],
    *,
    n_init: int,
    seed: int,
) -> Tuple[List[BenchmarkSample], List[BenchmarkSample]]:
    idx = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    ordered = [samples[i] for i in idx]
    n_init = max(0, min(int(n_init), len(ordered)))
    return ordered[:n_init], ordered[n_init:]


def _build_problem1a_mcd_predictor(
    *,
    build_result: Problem1ADatasetBuildResult,
    hidden_dim: int,
    depth: int,
    dropout: float,
    device: torch.device,
) -> torch.nn.Module:
    hidden_dims = tuple([int(hidden_dim)] * int(depth))

    modules = make_problem_1a_module_registry(
        input_dim=int(build_result.config.input_dim),
        latent_dim=int(build_result.config.latent_dim),
        output_dim=int(build_result.config.output_dim),
        observer_module_keys=build_result.config.observer_module_keys,
        upstream_hidden_dims=hidden_dims,
        observer_hidden_dims=hidden_dims,
        activation="relu",
        output_activation="identity",
        dropout=float(dropout),
        dtype=torch.float32,
    )

    predictor = build_protocol_predictor(
        modules=modules,
        dtype=torch.float32,
        strict_registry=True,
        device=device,
    )
    return predictor


def _build_predictor_factory(
    *,
    build_result: Problem1ADatasetBuildResult,
    args: argparse.Namespace,
    device: torch.device,
):
    predictor_type = str(args.predictor_type).lower()

    if predictor_type == "mcd":
        def predictor_factory() -> torch.nn.Module:
            return _build_problem1a_mcd_predictor(
                build_result=build_result,
                hidden_dim=int(args.hidden_dim),
                depth=int(args.depth),
                dropout=float(args.dropout),
                device=device,
            )
        return predictor_factory

    if predictor_type == "dkl":
        def predictor_factory() -> torch.nn.Module:
            model = MultiHeadNodewiseDKL(
                external_input_dim=int(build_result.config.input_dim),
                node_input_dims=[
                    int(build_result.config.input_dim),
                    int(build_result.config.latent_dim),
                ],
                parent_nodes=None,
                active_input_indices=None,
                hidden=int(args.dkl_hidden),
                depth=int(args.depth),
                feature_dim=int(args.dkl_feature_dim),
                kernel_type=str(args.dkl_kernel),
                sink_idx=1,
            )
            return model.to(device=device)

        return predictor_factory

    raise ValueError(f"Unsupported predictor_type: {predictor_type}")


def _outer_train_config(args: argparse.Namespace) -> ProtocolTrainingConfig:
    return ProtocolTrainingConfig(
        n_steps=int(args.outer_train_steps),
        batch_size=int(args.outer_batch_size),
        lr=float(args.outer_lr),
        weight_decay=float(args.outer_weight_decay),
        loss_name="mse",
        grad_clip_norm=None,
        val_every=int(args.outer_val_every),
        early_stopping_patience=int(args.outer_patience),
        early_stopping_min_delta=0.0,
        protocol_loss_weights={},
        seed=int(args.seed + args.trial),
        device=str(args.device),
        verbose=bool(args.verbose),
    )


def _fantasy_train_config(args: argparse.Namespace) -> ProtocolTrainingConfig:
    return ProtocolTrainingConfig(
        n_steps=int(args.fantasy_train_steps),
        batch_size=int(args.fantasy_batch_size),
        lr=float(args.fantasy_lr),
        weight_decay=float(args.fantasy_weight_decay),
        loss_name="mse",
        grad_clip_norm=None,
        val_every=max(int(args.fantasy_train_steps), 1),
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        protocol_loss_weights={},
        seed=int(args.seed + args.trial + 1000),
        device=str(args.device),
        verbose=bool(args.fantasy_verbose),
    )


def _build_problem1a_dataset_from_args(
    args: argparse.Namespace,
) -> Problem1ADatasetBuildResult:
    cfg = Problem1ADatasetBuilderConfig(
        input_dim=int(args.input_dim),
        latent_dim=int(args.latent_dim),
        output_dim=int(args.output_dim),
        similarities_to_target=tuple(float(x) for x in args.similarities_to_target),
        observer_scales=tuple(float(x) for x in args.observer_scales),
        protocol_costs=tuple(float(x) for x in args.protocol_costs),
        target_noise_std=float(args.target_noise_std),
        source_noise_stds=tuple(float(x) for x in args.source_noise_stds),
        n_pretrain_p1=int(args.n_pretrain_p1),
        n_pretrain_p2=int(args.n_pretrain_p2),
        n_adapt_p3=int(args.n_adapt_p3),
        n_val_p3=int(args.n_val_p3),
        n_test_p3=int(args.n_test_p3),
        dataset_seed=int(args.seed + args.trial),
    )
    return build_problem_1a_dataset(cfg)


def _records_to_dicts(records) -> List[Dict[str, Any]]:
    return [asdict(row) for row in records]


def _final_losses_from_history(history_dicts: List[Dict[str, Any]]) -> Tuple[Any, Any]:
    if len(history_dicts) == 0:
        return None, None
    last = history_dicts[-1]
    return last.get("target_val_loss", None), last.get("target_test_loss", None)


def _make_active_learning_summary(
    *,
    experiment_mode: str,
    experiment_family: str,
    candidate_pool_scope: str,
    args: argparse.Namespace,
    benchmark,
    build_result: Problem1ADatasetBuildResult,
    target_protocol_id: str,
    initial_train_samples: Sequence[BenchmarkSample],
    candidate_pool: Sequence[BenchmarkSample],
    run_result,
    history_dicts: List[Dict[str, Any]],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    n_acquisitions = max(
        0,
        len(run_result.final_train_samples) - len(initial_train_samples),
    )

    final_val_loss, final_test_loss = _final_losses_from_history(history_dicts)

    return {
        "experiment_name": "protocol1a",
        "experiment_family": experiment_family,
        "experiment_mode": experiment_mode,
        "candidate_pool_scope": candidate_pool_scope,
        "trial": int(args.trial),
        "seed": int(args.seed),
        "budget": float(args.budget),
        "spent_budget": float(run_result.spent_budget),
        "n_rounds_completed": int(n_acquisitions),
        "n_train_initial": int(len(initial_train_samples)),
        "n_train_final": int(len(run_result.final_train_samples)),
        "n_pool_initial": int(len(candidate_pool)),
        "n_pool_final": int(len(run_result.final_candidate_pool)),
        "target_protocol_id": target_protocol_id,
        "protocol_ids": list(benchmark.all_protocol_ids),
        "protocol_costs": [float(x) for x in args.protocol_costs],
        "similarities_to_target": [float(x) for x in args.similarities_to_target],
        "observer_scales": [float(x) for x in args.observer_scales],
        "target_noise_std": float(args.target_noise_std),
        "source_noise_stds": [float(x) for x in args.source_noise_stds],
        "options": options,
        "dataset_summary": build_result.summary(),
        "history": history_dicts,
        "final_target_val_loss": final_val_loss,
        "final_target_test_loss": final_test_loss,
    }


def _run_active_learning_mode(
    *,
    predictor_factory,
    protocol_map,
    initial_train_samples: Sequence[BenchmarkSample],
    candidate_pool: Sequence[BenchmarkSample],
    val_p3: Sequence[BenchmarkSample],
    test_p3: Sequence[BenchmarkSample],
    target_protocol_id: str,
    args: argparse.Namespace,
    outer_cfg: ProtocolTrainingConfig,
    fantasy_cfg: ProtocolTrainingConfig,
):
    runner_cfg = ActiveLearningRunnerConfig(
        budget=float(args.budget),
        target_protocol_id=target_protocol_id,
        n_fantasies=int(args.fantasy_mc_samples),
        outer_train_config=outer_cfg,
        fantasy_train_config=fantasy_cfg,
        device=str(args.device),
        acquisition_policy=str(args.target_acquisition_policy),
        random_seed=int(args.seed + args.trial),
        debug_progress=bool(args.debug_progress),
        debug_top_k_candidates=int(args.debug_top_k_candidates),
    )

    _debug_log(
        args,
        (
            "[debug] start active learning | policy=%s | predictor=%s | "
            "target=%s | n_initial=%d | n_pool=%d | budget=%.4f"
        ),
        str(args.target_acquisition_policy),
        str(args.predictor_type).lower(),
        target_protocol_id,
        len(initial_train_samples),
        len(candidate_pool),
        float(args.budget),
    )

    _debug_log(
        args,
        "[debug] initial_train_protocol_counts=%s",
        _json_dumps_compact(_protocol_counts(initial_train_samples)),
    )
    _debug_log(
        args,
        "[debug] candidate_pool_protocol_counts=%s",
        _json_dumps_compact(_protocol_counts(candidate_pool)),
    )

    return run_protocol_active_learning(
        predictor_factory=predictor_factory,
        protocols=protocol_map,
        initial_train_samples=list(initial_train_samples),
        candidate_pool=list(candidate_pool),
        target_val_samples=val_p3,
        target_test_samples=test_p3,
        config=runner_cfg,
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    logging.basicConfig(
        level=logging.INFO if (args.verbose or args.debug_progress) else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    _set_random_seed(int(args.seed + args.trial))
    device = torch.device(str(args.device))

    _debug_log(
        args,
        (
            "[debug] args | mode=%s | policy=%s | predictor=%s | "
            "trial=%d | seed=%d | effective_seed=%d | device=%s"
        ),
        str(args.experiment_mode),
        str(args.target_acquisition_policy),
        str(args.predictor_type).lower(),
        int(args.trial),
        int(args.seed),
        int(args.seed + args.trial),
        str(args.device),
    )

    build_result = _build_problem1a_dataset_from_args(args)
    benchmark = build_result.benchmark
    protocol_map = benchmark.protocol_map
    target_protocol_id = benchmark.target_protocol_id

    _debug_log(
        args,
        "[debug] dataset_summary=%s",
        _json_dumps_compact(build_result.summary()),
    )
    _debug_log(
        args,
        "[debug] target_protocol_id=%s | all_protocol_ids=%s",
        target_protocol_id,
        _json_dumps_compact(list(benchmark.all_protocol_ids)),
    )

    options = get_fantasy_protocol1a_options(
        overrides={
            "experiment_mode": str(args.experiment_mode),
            "target_protocol_id": target_protocol_id,
            "target_acquisition_policy": str(args.target_acquisition_policy),
            "target_adapt_budget": int(args.target_adapt_budget),
            "candidate_pool_scope": str(args.candidate_pool_scope),
            "predictor_type": str(args.predictor_type).lower(),
            "hidden": int(args.hidden_dim),
            "depth": int(args.depth),
            "p_drop": float(args.dropout),
            "mc_samples": int(args.fantasy_mc_samples),
            "fantasy_train_steps": int(args.fantasy_train_steps),
            "dkl_hidden": int(args.dkl_hidden),
            "dkl_feature_dim": int(args.dkl_feature_dim),
            "dkl_kernel": str(args.dkl_kernel),
            "dkl_inference": str(args.dkl_inference),
            "dkl_noise": float(args.dkl_noise),
            "feature_dim": int(args.dkl_feature_dim),
            "kernel_type": str(args.dkl_kernel),
            "observer_scales": [float(x) for x in args.observer_scales],
            "target_noise_std": float(args.target_noise_std),
            "source_noise_stds": [float(x) for x in args.source_noise_stds],
        }
    )

    pretrain_p1 = list(build_result.samples_in_split("pretrain_protocol_1"))
    pretrain_p2 = list(build_result.samples_in_split("pretrain_protocol_2"))
    adapt_p3 = list(build_result.samples_in_split("adapt_protocol_3"))
    val_p3 = list(build_result.samples_in_split("val_protocol_3"))
    test_p3 = list(build_result.samples_in_split("test_protocol_3"))

    _log_sample_split_debug(args, name="pretrain_p1", samples=pretrain_p1)
    _log_sample_split_debug(args, name="pretrain_p2", samples=pretrain_p2)
    _log_sample_split_debug(args, name="adapt_p3", samples=adapt_p3)
    _log_sample_split_debug(args, name="val_p3", samples=val_p3)
    _log_sample_split_debug(args, name="test_p3", samples=test_p3)

    init_p1, pool_p1 = _split_initial_and_pool(
        pretrain_p1,
        n_init=int(args.n_init_p1),
        seed=int(args.seed + args.trial + 11),
    )
    init_p2, pool_p2 = _split_initial_and_pool(
        pretrain_p2,
        n_init=int(args.n_init_p2),
        seed=int(args.seed + args.trial + 22),
    )
    init_p3, pool_p3 = _split_initial_and_pool(
        adapt_p3,
        n_init=int(args.n_init_p3),
        seed=int(args.seed + args.trial + 33),
    )

    _log_initial_pool_debug(
        args,
        init_p1=init_p1,
        pool_p1=pool_p1,
        init_p2=init_p2,
        pool_p2=pool_p2,
        init_p3=init_p3,
        pool_p3=pool_p3,
    )

    outer_cfg = _outer_train_config(args)
    fantasy_cfg = _fantasy_train_config(args)

    _log_training_config_debug(
        args,
        outer_cfg=outer_cfg,
        fantasy_cfg=fantasy_cfg,
    )

    predictor_factory = _build_predictor_factory(
        build_result=build_result,
        args=args,
        device=device,
    )

    _debug_log(
        args,
        "[debug] predictor_factory_ready | predictor_type=%s",
        str(args.predictor_type).lower(),
    )

    selection_history: List[Dict[str, Any]] = []

    if args.experiment_mode == "fantasy_al":
        initial_train_samples: List[BenchmarkSample] = init_p1 + init_p2 + init_p3
        candidate_pool: List[BenchmarkSample] = pool_p1 + pool_p2 + pool_p3

        run_result = _run_active_learning_mode(
            predictor_factory=predictor_factory,
            protocol_map=protocol_map,
            initial_train_samples=initial_train_samples,
            candidate_pool=candidate_pool,
            val_p3=val_p3,
            test_p3=test_p3,
            target_protocol_id=target_protocol_id,
            args=args,
            outer_cfg=outer_cfg,
            fantasy_cfg=fantasy_cfg,
        )

        history_dicts = _records_to_dicts(run_result.history)
        selection_history = list(getattr(run_result, "selection_history", []))

        _log_history_progress(
            args,
            label="fantasy_al",
            history_dicts=history_dicts,
        )
        _log_selection_progress(
            args,
            label="fantasy_al",
            selection_history=selection_history,
        )

        summary = _make_active_learning_summary(
            experiment_mode="fantasy_al",
            experiment_family="family2",
            candidate_pool_scope="all_protocols",
            args=args,
            benchmark=benchmark,
            build_result=build_result,
            target_protocol_id=target_protocol_id,
            initial_train_samples=initial_train_samples,
            candidate_pool=candidate_pool,
            run_result=run_result,
            history_dicts=history_dicts,
            options=options,
        )

        _log_final_loss_debug(args, label="fantasy_al", summary=summary)

    elif args.experiment_mode == "family3_candidate_pool_ablation":
        initial_train_samples = init_p1 + init_p2 + init_p3

        if str(args.candidate_pool_scope) == "all_protocols":
            candidate_pool = pool_p1 + pool_p2 + pool_p3
        elif str(args.candidate_pool_scope) == "target_only":
            candidate_pool = pool_p3
        else:
            raise ValueError(
                f"Unsupported candidate_pool_scope: {args.candidate_pool_scope}"
            )

        run_result = _run_active_learning_mode(
            predictor_factory=predictor_factory,
            protocol_map=protocol_map,
            initial_train_samples=initial_train_samples,
            candidate_pool=candidate_pool,
            val_p3=val_p3,
            test_p3=test_p3,
            target_protocol_id=target_protocol_id,
            args=args,
            outer_cfg=outer_cfg,
            fantasy_cfg=fantasy_cfg,
        )

        history_dicts = _records_to_dicts(run_result.history)
        selection_history = list(getattr(run_result, "selection_history", []))

        _log_history_progress(
            args,
            label="family3_candidate_pool_ablation",
            history_dicts=history_dicts,
        )
        _log_selection_progress(
            args,
            label="family3_candidate_pool_ablation",
            selection_history=selection_history,
        )

        summary = _make_active_learning_summary(
            experiment_mode="family3_candidate_pool_ablation",
            experiment_family="family3",
            candidate_pool_scope=str(args.candidate_pool_scope),
            args=args,
            benchmark=benchmark,
            build_result=build_result,
            target_protocol_id=target_protocol_id,
            initial_train_samples=initial_train_samples,
            candidate_pool=candidate_pool,
            run_result=run_result,
            history_dicts=history_dicts,
            options=options,
        )

        _log_final_loss_debug(
            args,
            label="family3_candidate_pool_ablation",
            summary=summary,
        )

    elif args.experiment_mode == "pretrain_then_adapt":
        _debug_log(
            args,
            (
                "[debug] start pretrain_then_adapt | predictor=%s | "
                "n_source=%d | n_target_adapt=%d | n_val=%d | n_test=%d"
            ),
            str(args.predictor_type).lower(),
            len(pretrain_p1) + len(pretrain_p2),
            len(adapt_p3),
            len(val_p3),
            len(test_p3),
        )

        pretrain_result = run_pretrain_then_adapt(
            predictor_factory=predictor_factory,
            protocols=protocol_map,
            source_pretrain_samples=pretrain_p1 + pretrain_p2,
            target_adapt_samples=adapt_p3,
            target_val_samples=val_p3,
            target_test_samples=test_p3,
            config=PretrainThenAdaptConfig(
                target_protocol_id=target_protocol_id,
                train_config=outer_cfg,
            ),
        )

        history_dicts = _records_to_dicts(pretrain_result.history)

        _log_history_progress(
            args,
            label="pretrain_then_adapt",
            history_dicts=history_dicts,
        )

        summary = {
            "experiment_name": "protocol1a",
            "experiment_family": "family1",
            "experiment_mode": "pretrain_then_adapt",
            "candidate_pool_scope": None,
            "trial": int(args.trial),
            "seed": int(args.seed),
            "budget": None,
            "spent_budget": None,
            "target_protocol_id": target_protocol_id,
            "protocol_ids": list(benchmark.all_protocol_ids),
            "protocol_costs": [float(x) for x in args.protocol_costs],
            "similarities_to_target": [float(x) for x in args.similarities_to_target],
            "observer_scales": [float(x) for x in args.observer_scales],
            "target_noise_std": float(args.target_noise_std),
            "source_noise_stds": [float(x) for x in args.source_noise_stds],
            "options": options,
            "dataset_summary": build_result.summary(),
            "history": history_dicts,
            "n_source_pretrain_samples": pretrain_result.n_source_pretrain_samples,
            "n_target_adapt_samples": pretrain_result.n_target_adapt_samples,
            "final_target_val_loss": pretrain_result.final_target_val_loss,
            "final_target_test_loss": pretrain_result.final_target_test_loss,
        }

        _log_final_loss_debug(args, label="pretrain_then_adapt", summary=summary)

    elif args.experiment_mode == "scratch_then_sequential_adapt":
        _debug_log(
            args,
            (
                "[debug] start scratch_then_sequential_adapt | predictor=%s | "
                "policy=%s | n_initial=%d | n_pool=%d | adapt_budget_points=%d"
            ),
            str(args.predictor_type).lower(),
            str(args.target_acquisition_policy),
            len(init_p3),
            len(pool_p3),
            int(args.target_adapt_budget),
        )

        seq_cfg = SequentialTargetAdaptRunnerConfig(
            target_protocol_id=target_protocol_id,
            acquisition_policy=str(args.target_acquisition_policy),
            adapt_budget_points=int(args.target_adapt_budget),
            outer_train_config=outer_cfg,
            fantasy_train_config=(
                fantasy_cfg if str(args.target_acquisition_policy).lower() == "fantasy" else None
            ),
            n_fantasies=int(args.fantasy_mc_samples),
            device=str(args.device),
            random_seed=int(args.seed + args.trial),
        )

        seq_result = run_sequential_target_adapt(
            predictor_factory=predictor_factory,
            protocols=protocol_map,
            initial_train_samples=list(init_p3),
            target_candidate_pool=list(pool_p3),
            target_val_samples=val_p3,
            target_test_samples=test_p3,
            config=seq_cfg,
        )

        history_dicts = _records_to_dicts(seq_result.history)
        final_val_loss, final_test_loss = _final_losses_from_history(history_dicts)
        selection_history = list(getattr(seq_result, "selection_history", []))

        _log_history_progress(
            args,
            label="scratch_then_sequential_adapt",
            history_dicts=history_dicts,
        )
        _log_selection_progress(
            args,
            label="scratch_then_sequential_adapt",
            selection_history=selection_history,
        )

        summary = {
            "experiment_name": "protocol1a",
            "experiment_family": "family1",
            "experiment_mode": "scratch_then_sequential_adapt",
            "candidate_pool_scope": "target_only",
            "trial": int(args.trial),
            "seed": int(args.seed),
            "budget": int(args.target_adapt_budget),
            "spent_budget": int(seq_result.n_target_points_used),
            "target_protocol_id": target_protocol_id,
            "protocol_ids": list(benchmark.all_protocol_ids),
            "protocol_costs": [float(x) for x in args.protocol_costs],
            "similarities_to_target": [float(x) for x in args.similarities_to_target],
            "observer_scales": [float(x) for x in args.observer_scales],
            "target_noise_std": float(args.target_noise_std),
            "source_noise_stds": [float(x) for x in args.source_noise_stds],
            "options": options,
            "dataset_summary": build_result.summary(),
            "history": history_dicts,
            "n_source_pretrain_samples": 0,
            "n_target_initial_samples": int(len(init_p3)),
            "n_target_pool_initial": int(len(pool_p3)),
            "n_target_points_used": int(seq_result.n_target_points_used),
            "target_cost_used": float(seq_result.target_cost_used),
            "final_target_val_loss": final_val_loss,
            "final_target_test_loss": final_test_loss,
        }

        _log_final_loss_debug(
            args,
            label="scratch_then_sequential_adapt",
            summary=summary,
        )

    elif args.experiment_mode == "pretrain_then_sequential_adapt":
        initial_train_samples = list(pretrain_p1) + list(pretrain_p2) + list(init_p3)

        _debug_log(
            args,
            (
                "[debug] start pretrain_then_sequential_adapt | predictor=%s | "
                "policy=%s | n_initial=%d | n_pool=%d | adapt_budget_points=%d"
            ),
            str(args.predictor_type).lower(),
            str(args.target_acquisition_policy),
            len(initial_train_samples),
            len(pool_p3),
            int(args.target_adapt_budget),
        )

        seq_cfg = SequentialTargetAdaptRunnerConfig(
            target_protocol_id=target_protocol_id,
            acquisition_policy=str(args.target_acquisition_policy),
            adapt_budget_points=int(args.target_adapt_budget),
            outer_train_config=outer_cfg,
            fantasy_train_config=(
                fantasy_cfg if str(args.target_acquisition_policy).lower() == "fantasy" else None
            ),
            n_fantasies=int(args.fantasy_mc_samples),
            device=str(args.device),
            random_seed=int(args.seed + args.trial),
        )

        seq_result = run_sequential_target_adapt(
            predictor_factory=predictor_factory,
            protocols=protocol_map,
            initial_train_samples=initial_train_samples,
            target_candidate_pool=list(pool_p3),
            target_val_samples=val_p3,
            target_test_samples=test_p3,
            config=seq_cfg,
        )

        history_dicts = _records_to_dicts(seq_result.history)
        final_val_loss, final_test_loss = _final_losses_from_history(history_dicts)
        selection_history = list(getattr(seq_result, "selection_history", []))

        _log_history_progress(
            args,
            label="pretrain_then_sequential_adapt",
            history_dicts=history_dicts,
        )
        _log_selection_progress(
            args,
            label="pretrain_then_sequential_adapt",
            selection_history=selection_history,
        )

        summary = {
            "experiment_name": "protocol1a",
            "experiment_family": "family1",
            "experiment_mode": "pretrain_then_sequential_adapt",
            "candidate_pool_scope": "target_only",
            "trial": int(args.trial),
            "seed": int(args.seed),
            "budget": int(args.target_adapt_budget),
            "spent_budget": int(seq_result.n_target_points_used),
            "target_protocol_id": target_protocol_id,
            "protocol_ids": list(benchmark.all_protocol_ids),
            "protocol_costs": [float(x) for x in args.protocol_costs],
            "similarities_to_target": [float(x) for x in args.similarities_to_target],
            "observer_scales": [float(x) for x in args.observer_scales],
            "target_noise_std": float(args.target_noise_std),
            "source_noise_stds": [float(x) for x in args.source_noise_stds],
            "options": options,
            "dataset_summary": build_result.summary(),
            "history": history_dicts,
            "n_source_pretrain_samples": int(len(pretrain_p1) + len(pretrain_p2)),
            "n_target_initial_samples": int(len(init_p3)),
            "n_target_pool_initial": int(len(pool_p3)),
            "n_target_points_used": int(seq_result.n_target_points_used),
            "target_cost_used": float(seq_result.target_cost_used),
            "final_target_val_loss": final_val_loss,
            "final_target_test_loss": final_test_loss,
        }

        _log_final_loss_debug(
            args,
            label="pretrain_then_sequential_adapt",
            summary=summary,
        )

    else:
        raise ValueError(f"Unsupported experiment_mode: {args.experiment_mode}")

    print(json.dumps(summary, indent=2))

    if args.save_json:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = (
            f"protocol1a"
            f"_mode{str(args.experiment_mode)}"
            f"_policy{str(args.target_acquisition_policy)}"
            f"_pool{str(args.candidate_pool_scope)}"
            f"_pred{str(args.predictor_type).lower()}"
            f"_trial{int(args.trial)}"
            f"_seed{int(args.seed)}"
        )

        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if len(selection_history) > 0:
            with open(out_dir / f"{stem}_selections.json", "w", encoding="utf-8") as f:
                json.dump(selection_history, f, indent=2)

        logger.warning("Saved outputs to %s", out_dir)


if __name__ == "__main__":
    main()