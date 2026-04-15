#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ofml_alfn.benchmarks.dataset_builders.build_problem_1a import (
    Problem1ADatasetBuilderConfig,
    Problem1ADatasetBuildResult,
    build_problem_1a_dataset,
)
from ofml_alfn.configs.fantasy_protocol1a import get_fantasy_protocol1a_options
from ofml_alfn.models.process_modules import make_problem_1a_module_registry
from ofml_alfn.models.protocol_predictor import build_protocol_predictor
from ofml_alfn.runners.active_learning_runner import (
    ActiveLearningRunnerConfig,
    run_protocol_active_learning,
)
from ofml_alfn.training.train_protocol_predictor import ProtocolTrainingConfig
from ofml_alfn.utils.protocol_types import BenchmarkSample

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Problem 1A fantasy acquisition experiment."
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
    parser.add_argument("--output_dir", type=str, default="outputs/protocol1a_fantasy")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


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


def _build_predictor_for_problem1a(
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
        verbose=False,
    )


def _build_problem1a_dataset_from_args(
    args: argparse.Namespace,
) -> Problem1ADatasetBuildResult:
    cfg = Problem1ADatasetBuilderConfig(
        input_dim=int(args.input_dim),
        latent_dim=int(args.latent_dim),
        output_dim=int(args.output_dim),
        similarities_to_target=tuple(float(x) for x in args.similarities_to_target),
        protocol_costs=tuple(float(x) for x in args.protocol_costs),
        n_pretrain_p1=int(args.n_pretrain_p1),
        n_pretrain_p2=int(args.n_pretrain_p2),
        n_adapt_p3=int(args.n_adapt_p3),
        n_val_p3=int(args.n_val_p3),
        n_test_p3=int(args.n_test_p3),
        dataset_seed=int(args.seed + args.trial),
    )
    return build_problem_1a_dataset(cfg)


def _history_to_dicts(history) -> List[Dict[str, Any]]:
    return [asdict(row) for row in history]


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    _set_random_seed(int(args.seed + args.trial))
    device = torch.device(str(args.device))

    build_result = _build_problem1a_dataset_from_args(args)
    benchmark = build_result.benchmark
    protocol_map = benchmark.protocol_map

    options = get_fantasy_protocol1a_options(
        overrides={
            "target_protocol_id": benchmark.target_protocol_id,
            "fantasy_train_steps": int(args.fantasy_train_steps),
            "mc_samples": int(args.fantasy_mc_samples),
        }
    )

    pretrain_p1 = list(build_result.samples_in_split("pretrain_protocol_1"))
    pretrain_p2 = list(build_result.samples_in_split("pretrain_protocol_2"))
    adapt_p3 = list(build_result.samples_in_split("adapt_protocol_3"))
    val_p3 = list(build_result.samples_in_split("val_protocol_3"))
    test_p3 = list(build_result.samples_in_split("test_protocol_3"))

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

    initial_train_samples: List[BenchmarkSample] = init_p1 + init_p2 + init_p3
    candidate_pool: List[BenchmarkSample] = pool_p1 + pool_p2 + pool_p3

    outer_cfg = _outer_train_config(args)
    fantasy_cfg = _fantasy_train_config(args)

    runner_cfg = ActiveLearningRunnerConfig(
        budget=float(args.budget),
        target_protocol_id=benchmark.target_protocol_id,
        n_fantasies=int(args.fantasy_mc_samples),
        outer_train_config=outer_cfg,
        fantasy_train_config=fantasy_cfg,
        device=str(args.device),
    )

    def predictor_factory() -> torch.nn.Module:
        return _build_predictor_for_problem1a(
            build_result=build_result,
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            dropout=float(args.dropout),
            device=device,
        )

    run_result = run_protocol_active_learning(
        predictor_factory=predictor_factory,
        protocols=protocol_map,
        initial_train_samples=initial_train_samples,
        candidate_pool=candidate_pool,
        target_val_samples=val_p3,
        target_test_samples=test_p3,
        config=runner_cfg,
    )

    history_dicts = _history_to_dicts(run_result.history)
    n_acquisitions = max(
        0,
        len(run_result.final_train_samples) - len(initial_train_samples),
    )

    summary = {
        "experiment_name": "protocol1a_fantasy",
        "trial": int(args.trial),
        "seed": int(args.seed),
        "budget": float(args.budget),
        "spent_budget": float(run_result.spent_budget),
        "n_rounds_completed": int(n_acquisitions),
        "n_train_initial": int(len(initial_train_samples)),
        "n_train_final": int(len(run_result.final_train_samples)),
        "n_pool_initial": int(len(candidate_pool)),
        "n_pool_final": int(len(run_result.final_candidate_pool)),
        "target_protocol_id": benchmark.target_protocol_id,
        "protocol_ids": list(benchmark.all_protocol_ids),
        "protocol_costs": [float(x) for x in args.protocol_costs],
        "similarities_to_target": [float(x) for x in args.similarities_to_target],
        "options": options,
        "dataset_summary": build_result.summary(),
        "history": history_dicts,
    }

    print(json.dumps(summary, indent=2))

    if args.save_json:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = (
            f"protocol1a_fantasy"
            f"_trial{int(args.trial)}"
            f"_budget{int(args.budget)}"
            f"_seed{int(args.seed)}"
        )

        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(out_dir / f"{stem}_selections.json", "w", encoding="utf-8") as f:
            json.dump(run_result.selection_history, f, indent=2)

        logger.warning("Saved outputs to %s", out_dir)


if __name__ == "__main__":
    main()