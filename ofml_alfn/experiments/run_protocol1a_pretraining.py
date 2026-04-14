#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from ofml_alfn.benchmarks.dataset_builders.build_problem_1a import (
    Problem1ADatasetBuilderConfig,
)
from ofml_alfn.runners.pretraining_runner import (
    PretrainingRunnerConfig,
    run_problem_1a_pretraining_comparison,
)
from ofml_alfn.training.train_protocol_predictor import ProtocolTrainingConfig


def _as_tuple_int(xs: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in xs)


def _as_tuple_float(xs: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _ensure_dir(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        f"--{name}",
        dest=name,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=name,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{name: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Problem 1_A scratch vs pretrain-then-adapt comparison."
    )

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_json_name", type=str, default="problem1a_pretraining_summary.json")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--condition_keys", type=str, nargs="+", default=["x0", "x1"])
    parser.add_argument(
        "--protocol_ids",
        type=str,
        nargs=3,
        default=["protocol_1", "protocol_2", "protocol_3"],
    )
    parser.add_argument(
        "--observer_module_keys",
        type=str,
        nargs=3,
        default=["observer_1", "observer_2", "observer_3"],
    )

    parser.add_argument(
        "--similarities_to_target",
        type=float,
        nargs=3,
        default=[0.4, 0.7, 1.0],
    )
    parser.add_argument(
        "--protocol_costs",
        type=float,
        nargs=3,
        default=[1.0, 2.0, 3.0],
        help="Observation costs for Protocol 1, 2, 3.",
    )

    parser.add_argument("--n_pretrain_p1", type=int, default=128)
    parser.add_argument("--n_pretrain_p2", type=int, default=128)
    parser.add_argument("--n_adapt_p3", type=int, default=32)
    parser.add_argument("--n_val_p3", type=int, default=128)
    parser.add_argument("--n_test_p3", type=int, default=256)

    parser.add_argument("--condition_low", type=float, default=-1.0)
    parser.add_argument("--condition_high", type=float, default=1.0)

    parser.add_argument("--upstream_seed", type=int, default=0)
    parser.add_argument("--target_observer_seed", type=int, default=1)
    parser.add_argument("--anchor_seeds", type=int, nargs=2, default=[101, 202])
    parser.add_argument("--dataset_seed", type=int, default=999)

    parser.add_argument("--upstream_activation", type=str, default="tanh")
    parser.add_argument("--observer_activation", type=str, default="identity")

    parser.add_argument("--upstream_weight_scale", type=float, default=1.0)
    parser.add_argument("--upstream_bias_scale", type=float, default=0.25)
    parser.add_argument("--observer_weight_scale", type=float, default=1.0)
    parser.add_argument("--observer_bias_scale", type=float, default=0.25)

    parser.add_argument("--target_noise_std", type=float, default=0.0)
    parser.add_argument("--source_noise_stds", type=float, nargs=2, default=[0.0, 0.0])

    _bool_flag(
        parser,
        name="add_observation_noise_to_train",
        default=True,
        help_text="Add observation noise when generating train-side splits.",
    )
    _bool_flag(
        parser,
        name="add_observation_noise_to_eval",
        default=False,
        help_text="Add observation noise when generating val/test splits.",
    )
    _bool_flag(
        parser,
        name="freeze_shared_upstream_during_adapt",
        default=False,
        help_text="Freeze shared_upstream during Protocol 3 adaptation after pretraining.",
    )
    _bool_flag(
        parser,
        name="verbose_train",
        default=False,
        help_text="Print training logs during optimization.",
    )

    parser.add_argument("--upstream_hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--observer_hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--output_activation", type=str, default="identity")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--scratch_model_seed", type=int, default=100)
    parser.add_argument("--transfer_model_seed", type=int, default=200)

    parser.add_argument("--scratch_n_steps", type=int, default=500)
    parser.add_argument("--scratch_batch_size", type=int, default=64)
    parser.add_argument("--scratch_lr", type=float, default=1e-3)
    parser.add_argument("--scratch_weight_decay", type=float, default=1e-6)
    parser.add_argument("--scratch_loss_name", type=str, default="mse")
    parser.add_argument("--scratch_grad_clip_norm", type=float, default=None)
    parser.add_argument("--scratch_val_every", type=int, default=25)
    parser.add_argument("--scratch_early_stopping_patience", type=int, default=20)
    parser.add_argument("--scratch_early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--scratch_seed", type=int, default=0)

    parser.add_argument("--pretrain_n_steps", type=int, default=500)
    parser.add_argument("--pretrain_batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-6)
    parser.add_argument("--pretrain_loss_name", type=str, default="mse")
    parser.add_argument("--pretrain_grad_clip_norm", type=float, default=None)
    parser.add_argument("--pretrain_val_every", type=int, default=50)
    parser.add_argument("--pretrain_early_stopping_patience", type=int, default=None)
    parser.add_argument("--pretrain_early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--pretrain_seed", type=int, default=1)

    parser.add_argument("--adapt_n_steps", type=int, default=300)
    parser.add_argument("--adapt_batch_size", type=int, default=64)
    parser.add_argument("--adapt_lr", type=float, default=5e-4)
    parser.add_argument("--adapt_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adapt_loss_name", type=str, default="mse")
    parser.add_argument("--adapt_grad_clip_norm", type=float, default=None)
    parser.add_argument("--adapt_val_every", type=int, default=25)
    parser.add_argument("--adapt_early_stopping_patience", type=int, default=20)
    parser.add_argument("--adapt_early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--adapt_seed", type=int, default=2)

    return parser


def _resolve_dtype(dtype_name: str):
    if dtype_name == "float32":
        import torch
        return torch.float32
    if dtype_name == "float64":
        import torch
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name!r}")


def _make_training_config(
    *,
    n_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    grad_clip_norm: float | None,
    val_every: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    seed: int,
    device: str | None,
    verbose: bool,
) -> ProtocolTrainingConfig:
    return ProtocolTrainingConfig(
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        loss_name=loss_name,
        grad_clip_norm=grad_clip_norm,
        val_every=val_every,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
        device=device,
        verbose=verbose,
    )


def build_runner_config(args: argparse.Namespace) -> PretrainingRunnerConfig:
    dataset_config = Problem1ADatasetBuilderConfig(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        condition_keys=tuple(args.condition_keys),
        protocol_ids=tuple(args.protocol_ids),
        observer_module_keys=tuple(args.observer_module_keys),
        similarities_to_target=_as_tuple_float(args.similarities_to_target),
        protocol_costs=_as_tuple_float(args.protocol_costs),
        n_pretrain_p1=args.n_pretrain_p1,
        n_pretrain_p2=args.n_pretrain_p2,
        n_adapt_p3=args.n_adapt_p3,
        n_val_p3=args.n_val_p3,
        n_test_p3=args.n_test_p3,
        condition_range=(float(args.condition_low), float(args.condition_high)),
        upstream_seed=args.upstream_seed,
        target_observer_seed=args.target_observer_seed,
        anchor_seeds=tuple(args.anchor_seeds),
        dataset_seed=args.dataset_seed,
        upstream_activation=args.upstream_activation,
        observer_activation=args.observer_activation,
        upstream_weight_scale=args.upstream_weight_scale,
        upstream_bias_scale=args.upstream_bias_scale,
        observer_weight_scale=args.observer_weight_scale,
        observer_bias_scale=args.observer_bias_scale,
        target_noise_std=args.target_noise_std,
        source_noise_stds=tuple(args.source_noise_stds),
        add_observation_noise_to_train=args.add_observation_noise_to_train,
        add_observation_noise_to_eval=args.add_observation_noise_to_eval,
        save_dir=None,
    )

    scratch_train_config = _make_training_config(
        n_steps=args.scratch_n_steps,
        batch_size=args.scratch_batch_size,
        lr=args.scratch_lr,
        weight_decay=args.scratch_weight_decay,
        loss_name=args.scratch_loss_name,
        grad_clip_norm=args.scratch_grad_clip_norm,
        val_every=args.scratch_val_every,
        early_stopping_patience=args.scratch_early_stopping_patience,
        early_stopping_min_delta=args.scratch_early_stopping_min_delta,
        seed=args.scratch_seed,
        device=args.device,
        verbose=args.verbose_train,
    )

    pretrain_train_config = _make_training_config(
        n_steps=args.pretrain_n_steps,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
        loss_name=args.pretrain_loss_name,
        grad_clip_norm=args.pretrain_grad_clip_norm,
        val_every=args.pretrain_val_every,
        early_stopping_patience=args.pretrain_early_stopping_patience,
        early_stopping_min_delta=args.pretrain_early_stopping_min_delta,
        seed=args.pretrain_seed,
        device=args.device,
        verbose=args.verbose_train,
    )

    adapt_train_config = _make_training_config(
        n_steps=args.adapt_n_steps,
        batch_size=args.adapt_batch_size,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
        loss_name=args.adapt_loss_name,
        grad_clip_norm=args.adapt_grad_clip_norm,
        val_every=args.adapt_val_every,
        early_stopping_patience=args.adapt_early_stopping_patience,
        early_stopping_min_delta=args.adapt_early_stopping_min_delta,
        seed=args.adapt_seed,
        device=args.device,
        verbose=args.verbose_train,
    )

    return PretrainingRunnerConfig(
        dataset_config=dataset_config,
        scratch_train_config=scratch_train_config,
        pretrain_train_config=pretrain_train_config,
        adapt_train_config=adapt_train_config,
        upstream_hidden_dims=_as_tuple_int(args.upstream_hidden_dims),
        observer_hidden_dims=_as_tuple_int(args.observer_hidden_dims),
        activation=args.activation,
        output_activation=args.output_activation,
        dropout=args.dropout,
        dtype=_resolve_dtype(args.dtype),
        strict_registry=True,
        scratch_model_seed=args.scratch_model_seed,
        transfer_model_seed=args.transfer_model_seed,
        freeze_shared_upstream_during_adapt=args.freeze_shared_upstream_during_adapt,
        shared_upstream_module_key="shared_upstream",
        device=args.device,
        metadata={
            "cli_args": vars(args),
        },
    )


def _augment_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    scratch_loss = summary.get("scratch_test_loss", None)
    transfer_loss = summary.get("transfer_test_loss", None)

    if scratch_loss is not None and transfer_loss is not None:
        summary["delta_test_loss"] = float(transfer_loss) - float(scratch_loss)
        summary["relative_test_loss_ratio"] = (
            None if float(scratch_loss) == 0.0 else float(transfer_loss) / float(scratch_loss)
        )
    else:
        summary["delta_test_loss"] = None
        summary["relative_test_loss_ratio"] = None

    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runner_config = build_runner_config(args)
    result = run_problem_1a_pretraining_comparison(config=runner_config)

    summary = result.summary()
    summary = _augment_summary(summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    output_dir = _ensure_dir(args.output_dir)
    if output_dir is not None:
        output_path = output_dir / args.save_json_name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()