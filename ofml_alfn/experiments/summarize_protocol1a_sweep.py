#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_FILENAME = "problem1a_pretraining_summary.json"


@dataclass(frozen=True)
class SweepRecord:
    path: Path
    n_adapt_p3: int
    seed: int
    target_cost: float
    scratch_test_loss: float
    transfer_test_loss: float
    scratch_target_cost: float
    transfer_target_cost: float
    transfer_total_cost: float


def _extract_int_from_parts(path: Path, pattern: str) -> Optional[int]:
    rx = re.compile(pattern)
    for part in path.parts:
        m = rx.fullmatch(part)
        if m is not None:
            return int(m.group(1))
    return None


def _safe_float(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise KeyError(f"Missing key {key!r} in summary json")
    return float(d[key])


def load_one_summary(path: Path) -> SweepRecord:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    n_adapt = _extract_int_from_parts(path, r"n(\d+)")
    seed = _extract_int_from_parts(path, r"seed(\d+)")

    if n_adapt is None:
        raise ValueError(
            f"Could not infer n_adapt_p3 from path: {path}. "
            "Expected some directory name like n32."
        )
    if seed is None:
        raise ValueError(
            f"Could not infer seed from path: {path}. "
            "Expected some directory name like seed3."
        )

    scratch_target_cost = _safe_float(obj, "scratch_target_cost")
    transfer_target_cost = _safe_float(obj, "transfer_target_cost")

    if not math.isclose(scratch_target_cost, transfer_target_cost, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Expected scratch_target_cost and transfer_target_cost to match, "
            f"but got {scratch_target_cost} and {transfer_target_cost} in {path}"
        )

    return SweepRecord(
        path=path,
        n_adapt_p3=n_adapt,
        seed=seed,
        target_cost=scratch_target_cost,
        scratch_test_loss=_safe_float(obj, "scratch_test_loss"),
        transfer_test_loss=_safe_float(obj, "transfer_test_loss"),
        scratch_target_cost=scratch_target_cost,
        transfer_target_cost=transfer_target_cost,
        transfer_total_cost=_safe_float(obj, "transfer_total_cost"),
    )


def load_sweep(root_dir: Path) -> List[SweepRecord]:
    paths = sorted(root_dir.rglob(SUMMARY_FILENAME))
    if not paths:
        raise FileNotFoundError(
            f"No {SUMMARY_FILENAME!r} files found under {root_dir}"
        )
    records = [load_one_summary(p) for p in paths]
    records.sort(key=lambda r: (r.target_cost, r.seed))
    return records


def mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_loss_curve(
    records: Sequence[SweepRecord],
) -> Dict[str, List[float]]:
    by_cost: Dict[float, List[SweepRecord]] = {}
    for r in records:
        by_cost.setdefault(r.target_cost, []).append(r)

    costs = sorted(by_cost.keys())
    scratch_mean = []
    scratch_std = []
    transfer_mean = []
    transfer_std = []

    for c in costs:
        rs = by_cost[c]
        sm, ss = mean_std([r.scratch_test_loss for r in rs])
        tm, ts = mean_std([r.transfer_test_loss for r in rs])
        scratch_mean.append(sm)
        scratch_std.append(ss)
        transfer_mean.append(tm)
        transfer_std.append(ts)

    return {
        "costs": costs,
        "scratch_mean": scratch_mean,
        "scratch_std": scratch_std,
        "transfer_mean": transfer_mean,
        "transfer_std": transfer_std,
    }


def compute_threshold_costs(
    records: Sequence[SweepRecord],
    thresholds: Sequence[float],
) -> Dict[float, Dict[str, Dict[str, float | List[float]]]]:
    seeds = sorted(set(r.seed for r in records))
    by_seed: Dict[int, List[SweepRecord]] = {seed: [] for seed in seeds}
    for r in records:
        by_seed[r.seed].append(r)

    for seed in seeds:
        by_seed[seed].sort(key=lambda r: r.target_cost)

    result: Dict[float, Dict[str, Dict[str, float | List[float]]]] = {}

    for eps in thresholds:
        scratch_costs: List[float] = []
        transfer_costs: List[float] = []

        for seed in seeds:
            rs = by_seed[seed]

            scratch_hit: Optional[float] = None
            transfer_hit: Optional[float] = None

            for r in rs:
                if scratch_hit is None and r.scratch_test_loss <= eps:
                    scratch_hit = r.target_cost
                if transfer_hit is None and r.transfer_test_loss <= eps:
                    transfer_hit = r.target_cost

            if scratch_hit is not None:
                scratch_costs.append(float(scratch_hit))
            if transfer_hit is not None:
                transfer_costs.append(float(transfer_hit))

        sm, ss = mean_std(scratch_costs)
        tm, ts = mean_std(transfer_costs)

        result[float(eps)] = {
            "scratch": {
                "mean": sm,
                "std": ss,
                "values": scratch_costs,
                "n_success": len(scratch_costs),
            },
            "transfer": {
                "mean": tm,
                "std": ts,
                "values": transfer_costs,
                "n_success": len(transfer_costs),
            },
        }

    return result


def plot_loss_curve(
    curve: Dict[str, List[float]],
    *,
    output_path: Path,
    use_log_y: bool,
) -> None:
    costs = np.asarray(curve["costs"], dtype=float)
    scratch_mean = np.asarray(curve["scratch_mean"], dtype=float)
    scratch_std = np.asarray(curve["scratch_std"], dtype=float)
    transfer_mean = np.asarray(curve["transfer_mean"], dtype=float)
    transfer_std = np.asarray(curve["transfer_std"], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(costs, scratch_mean, marker="o", label="Scratch")
    plt.fill_between(costs, scratch_mean - scratch_std, scratch_mean + scratch_std, alpha=0.2)
    plt.plot(costs, transfer_mean, marker="o", label="Pretrain + Adapt")
    plt.fill_between(costs, transfer_mean - transfer_std, transfer_mean + transfer_std, alpha=0.2)

    plt.xlabel("Target cost on Protocol 3")
    plt.ylabel("Test loss on Protocol 3")
    plt.title("Problem 1_A: test loss vs target cost")
    if use_log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_threshold_costs(
    threshold_stats: Dict[float, Dict[str, Dict[str, float | List[float]]]],
    *,
    output_path: Path,
) -> None:
    eps_list = sorted(threshold_stats.keys())
    x = np.arange(len(eps_list), dtype=float)
    width = 0.35

    scratch_mean = np.asarray([threshold_stats[eps]["scratch"]["mean"] for eps in eps_list], dtype=float)
    scratch_std = np.asarray([threshold_stats[eps]["scratch"]["std"] for eps in eps_list], dtype=float)
    transfer_mean = np.asarray([threshold_stats[eps]["transfer"]["mean"] for eps in eps_list], dtype=float)
    transfer_std = np.asarray([threshold_stats[eps]["transfer"]["std"] for eps in eps_list], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, scratch_mean, width=width, yerr=scratch_std, capsize=4, label="Scratch")
    plt.bar(x + width / 2, transfer_mean, width=width, yerr=transfer_std, capsize=4, label="Pretrain + Adapt")

    plt.xticks(x, [f"{eps:.1e}" for eps in eps_list])
    plt.xlabel("Threshold ε")
    plt.ylabel("Target cost needed to reach ε")
    plt.title("Problem 1_A: threshold-reaching target cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_aggregate_json(
    *,
    curve: Dict[str, List[float]],
    threshold_stats: Dict[float, Dict[str, Dict[str, float | List[float]]]],
    output_path: Path,
) -> None:
    obj = {
        "loss_curve": curve,
        "threshold_costs": threshold_stats,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Problem 1_A sweep results and draw two comparison plots."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing n*/seed*/problem1a_pretraining_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots and aggregate json",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 2e-4, 1e-4],
        help="Thresholds ε for target-cost comparison",
    )
    parser.add_argument(
        "--log_y",
        action="store_true",
        help="Use log scale on the y-axis of the loss curve plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_sweep(root_dir)
    curve = aggregate_loss_curve(records)
    threshold_stats = compute_threshold_costs(records, args.thresholds)

    curve_png = output_dir / "problem1a_loss_vs_target_cost.png"
    threshold_png = output_dir / "problem1a_threshold_target_cost.png"
    aggregate_json = output_dir / "problem1a_sweep_aggregate.json"

    plot_loss_curve(curve, output_path=curve_png, use_log_y=args.log_y)
    plot_threshold_costs(threshold_stats, output_path=threshold_png)
    save_aggregate_json(
        curve=curve,
        threshold_stats=threshold_stats,
        output_path=aggregate_json,
    )

    print(f"Saved: {curve_png}")
    print(f"Saved: {threshold_png}")
    print(f"Saved: {aggregate_json}")


if __name__ == "__main__":
    main()