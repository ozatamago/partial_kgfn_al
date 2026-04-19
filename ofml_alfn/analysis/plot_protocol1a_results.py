#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunRecord:
    result_path: Path
    method_label: str
    experiment_mode: str
    acquisition_policy: Optional[str]
    seed: Optional[int]
    trial: Optional[int]
    history: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass(frozen=True)
class CurvePoint:
    x: float
    mean: float
    std: float
    n: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Protocol 1A experiment results from saved JSON files."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/protocol1a",
        help="Directory containing saved JSON result files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/protocol1a_plots",
        help="Directory to save plots and CSV summaries.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=[
            "scratch_then_sequential_adapt",
            "pretrain_then_sequential_adapt",
        ],
        help="Experiment modes to include.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="target_test_loss",
        choices=["target_val_loss", "target_test_loss"],
        help="Primary metric for threshold plots and final-loss plots.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Loss thresholds for computing steps-to-threshold.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="Protocol 1A",
        help="Prefix used in plot titles.",
    )
    parser.add_argument(
        "--include_policy_in_label",
        action="store_true",
        help="Append acquisition policy to method labels.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Optional maximum step to include in learning curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args()


def _is_result_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    if path.name.endswith("_selections.json"):
        return False
    return True


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _method_label_from_summary(
    summary: Dict[str, Any],
    *,
    include_policy_in_label: bool,
) -> str:
    mode = str(summary.get("experiment_mode", "unknown_mode"))
    options = summary.get("options", {}) or {}
    policy = options.get("target_acquisition_policy", None)

    pretty_mode = {
        "scratch_then_sequential_adapt": "scratch",
        "pretrain_then_sequential_adapt": "pretrain",
        "fantasy_al": "fantasy_al",
        "pretrain_then_adapt": "pretrain_then_adapt",
    }.get(mode, mode)

    if include_policy_in_label and policy is not None:
        return f"{pretty_mode} ({policy})"
    return pretty_mode


def _load_run_records(
    input_dir: Path,
    *,
    allowed_modes: Sequence[str],
    include_policy_in_label: bool,
) -> List[RunRecord]:
    records: List[RunRecord] = []

    for path in sorted(input_dir.glob("*.json")):
        if not _is_result_json(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        mode = str(summary.get("experiment_mode", ""))
        if len(allowed_modes) > 0 and mode not in allowed_modes:
            continue

        history = summary.get("history", None)
        if not isinstance(history, list):
            continue
        if len(history) == 0:
            continue

        options = summary.get("options", {}) or {}
        policy = options.get("target_acquisition_policy", None)

        records.append(
            RunRecord(
                result_path=path,
                method_label=_method_label_from_summary(
                    summary,
                    include_policy_in_label=include_policy_in_label,
                ),
                experiment_mode=mode,
                acquisition_policy=None if policy is None else str(policy),
                seed=_safe_int(summary.get("seed", None)),
                trial=_safe_int(summary.get("trial", None)),
                history=history,
                summary=summary,
            )
        )

    return records


def _extract_curve_rows(
    run: RunRecord,
    *,
    metric_names: Sequence[str],
    max_step: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in run.history:
        step_idx = _safe_int(row.get("step_idx", None))
        n_target_points_used = _safe_int(row.get("n_target_points_used", None))
        target_cost_used = _safe_float(row.get("target_cost_used", None))

        if step_idx is None:
            continue
        if max_step is not None and step_idx > int(max_step):
            continue

        out = {
            "method_label": run.method_label,
            "experiment_mode": run.experiment_mode,
            "acquisition_policy": run.acquisition_policy,
            "seed": run.seed,
            "trial": run.trial,
            "step_idx": step_idx,
            "n_target_points_used": n_target_points_used,
            "target_cost_used": target_cost_used,
        }

        for metric_name in metric_names:
            out[metric_name] = _safe_float(row.get(metric_name, None))

        rows.append(out)

    return rows


def _group_rows_by_method(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method_label"]), []).append(row)
    return grouped


def _aggregate_curve(
    rows: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> List[CurvePoint]:
    grouped: Dict[float, List[float]] = {}

    for row in rows:
        x = _safe_float(row.get(x_key, None))
        y = _safe_float(row.get(y_key, None))
        if x is None or y is None:
            continue
        grouped.setdefault(float(x), []).append(float(y))

    out: List[CurvePoint] = []
    for x in sorted(grouped.keys()):
        ys = np.asarray(grouped[x], dtype=float)
        out.append(
            CurvePoint(
                x=float(x),
                mean=float(np.mean(ys)),
                std=float(np.std(ys, ddof=0)),
                n=int(len(ys)),
            )
        )
    return out


def _steps_to_threshold(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_key: str,
    threshold: float,
) -> Dict[Tuple[Optional[int], Optional[int]], Optional[int]]:
    """
    Returns mapping:
        (seed, trial) -> first step where metric <= threshold, or None if never reached.
    """
    grouped: Dict[Tuple[Optional[int], Optional[int]], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed", None), row.get("trial", None))
        grouped.setdefault(key, []).append(row)

    out: Dict[Tuple[Optional[int], Optional[int]], Optional[int]] = {}
    for key, run_rows in grouped.items():
        run_rows = sorted(run_rows, key=lambda r: int(r["step_idx"]))
        found: Optional[int] = None
        for row in run_rows:
            metric = _safe_float(row.get(metric_key, None))
            step = _safe_int(row.get("n_target_points_used", None))
            if metric is None or step is None:
                continue
            if metric <= float(threshold):
                found = int(step)
                break
        out[key] = found
    return out


def _final_metric_per_run(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_key: str,
) -> Dict[Tuple[Optional[int], Optional[int]], Optional[float]]:
    grouped: Dict[Tuple[Optional[int], Optional[int]], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed", None), row.get("trial", None))
        grouped.setdefault(key, []).append(row)

    out: Dict[Tuple[Optional[int], Optional[int]], Optional[float]] = {}
    for key, run_rows in grouped.items():
        run_rows = sorted(run_rows, key=lambda r: int(r["step_idx"]))
        last_metric: Optional[float] = None
        for row in run_rows:
            metric = _safe_float(row.get(metric_key, None))
            if metric is not None:
                last_metric = metric
        out[key] = last_metric
    return out


def _mean_std(xs: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [float(x) for x in xs if x is not None]
    if len(vals) == 0:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), int(len(arr))


def _plot_learning_curve(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(7.2, 5.0))

    for method_label in sorted(grouped_rows.keys()):
        curve = _aggregate_curve(
            grouped_rows[method_label],
            x_key=x_key,
            y_key=y_key,
        )
        if len(curve) == 0:
            continue

        xs = np.asarray([p.x for p in curve], dtype=float)
        ys = np.asarray([p.mean for p in curve], dtype=float)
        stds = np.asarray([p.std for p in curve], dtype=float)

        plt.plot(xs, ys, label=method_label)
        plt.fill_between(xs, ys - stds, ys + stds, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_bar_summary(
    labels: Sequence[str],
    means: Sequence[Optional[float]],
    stds: Sequence[Optional[float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    xs = np.arange(len(labels), dtype=float)
    y_vals = np.asarray([np.nan if v is None else float(v) for v in means], dtype=float)
    y_errs = np.asarray([0.0 if v is None else float(v) for v in stds], dtype=float)

    plt.figure(figsize=(7.2, 5.0))
    plt.bar(xs, y_vals, yerr=y_errs, capsize=4.0)
    plt.xticks(xs, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _write_curve_csv(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    *,
    x_key: str,
    y_key: str,
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method_label", x_key, f"{y_key}_mean", f"{y_key}_std", "n"])
        for method_label in sorted(grouped_rows.keys()):
            curve = _aggregate_curve(grouped_rows[method_label], x_key=x_key, y_key=y_key)
            for p in curve:
                writer.writerow([method_label, p.x, p.mean, p.std, p.n])


def _write_threshold_csv(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    *,
    metric_key: str,
    threshold: float,
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method_label", "threshold", "mean_steps", "std_steps", "n_reached"])

        for method_label in sorted(grouped_rows.keys()):
            per_run = _steps_to_threshold(
                grouped_rows[method_label],
                metric_key=metric_key,
                threshold=threshold,
            )
            mean_steps, std_steps, n = _mean_std(per_run.values())
            writer.writerow([method_label, threshold, mean_steps, std_steps, n])


def _write_final_metric_csv(
    grouped_rows: Dict[str, List[Dict[str, Any]]],
    *,
    metric_key: str,
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method_label", f"{metric_key}_mean", f"{metric_key}_std", "n"])

        for method_label in sorted(grouped_rows.keys()):
            per_run = _final_metric_per_run(
                grouped_rows[method_label],
                metric_key=metric_key,
            )
            mean_val, std_val, n = _mean_std(per_run.values())
            writer.writerow([method_label, mean_val, std_val, n])


def main() -> None:
    args = _parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_run_records(
        input_dir,
        allowed_modes=args.modes,
        include_policy_in_label=bool(args.include_policy_in_label),
    )

    if len(runs) == 0:
        raise ValueError(
            f"No matching result JSON files were found in {input_dir} "
            f"for modes={args.modes}"
        )

    rows: List[Dict[str, Any]] = []
    for run in runs:
        rows.extend(
            _extract_curve_rows(
                run,
                metric_names=["target_val_loss", "target_test_loss"],
                max_step=args.max_step,
            )
        )

    if len(rows) == 0:
        raise ValueError("No usable history rows were extracted from the result files.")

    grouped_rows = _group_rows_by_method(rows)

    # --------------------------------------------------------------
    # Learning curves vs number of target sampled points
    # --------------------------------------------------------------
    _plot_learning_curve(
        grouped_rows,
        x_key="n_target_points_used",
        y_key="target_val_loss",
        title=f"{args.title_prefix}: validation loss vs target sampled points",
        xlabel="Target sampled points",
        ylabel="Validation loss on Protocol 3",
        output_path=output_dir / "curve_val_vs_target_points.png",
        dpi=int(args.dpi),
    )
    _write_curve_csv(
        grouped_rows,
        x_key="n_target_points_used",
        y_key="target_val_loss",
        output_path=output_dir / "curve_val_vs_target_points.csv",
    )

    _plot_learning_curve(
        grouped_rows,
        x_key="n_target_points_used",
        y_key="target_test_loss",
        title=f"{args.title_prefix}: test loss vs target sampled points",
        xlabel="Target sampled points",
        ylabel="Test loss on Protocol 3",
        output_path=output_dir / "curve_test_vs_target_points.png",
        dpi=int(args.dpi),
    )
    _write_curve_csv(
        grouped_rows,
        x_key="n_target_points_used",
        y_key="target_test_loss",
        output_path=output_dir / "curve_test_vs_target_points.csv",
    )

    # --------------------------------------------------------------
    # Learning curves vs target cost
    # --------------------------------------------------------------
    _plot_learning_curve(
        grouped_rows,
        x_key="target_cost_used",
        y_key="target_val_loss",
        title=f"{args.title_prefix}: validation loss vs target cost",
        xlabel="Target acquisition cost",
        ylabel="Validation loss on Protocol 3",
        output_path=output_dir / "curve_val_vs_target_cost.png",
        dpi=int(args.dpi),
    )
    _write_curve_csv(
        grouped_rows,
        x_key="target_cost_used",
        y_key="target_val_loss",
        output_path=output_dir / "curve_val_vs_target_cost.csv",
    )

    _plot_learning_curve(
        grouped_rows,
        x_key="target_cost_used",
        y_key="target_test_loss",
        title=f"{args.title_prefix}: test loss vs target cost",
        xlabel="Target acquisition cost",
        ylabel="Test loss on Protocol 3",
        output_path=output_dir / "curve_test_vs_target_cost.png",
        dpi=int(args.dpi),
    )
    _write_curve_csv(
        grouped_rows,
        x_key="target_cost_used",
        y_key="target_test_loss",
        output_path=output_dir / "curve_test_vs_target_cost.csv",
    )

    # --------------------------------------------------------------
    # Final metric bar plot
    # --------------------------------------------------------------
    final_metric = str(args.metric)
    labels = sorted(grouped_rows.keys())
    final_means: List[Optional[float]] = []
    final_stds: List[Optional[float]] = []

    for label in labels:
        per_run = _final_metric_per_run(grouped_rows[label], metric_key=final_metric)
        mean_val, std_val, _ = _mean_std(per_run.values())
        final_means.append(mean_val)
        final_stds.append(std_val)

    _plot_bar_summary(
        labels,
        final_means,
        final_stds,
        title=f"{args.title_prefix}: final {final_metric}",
        ylabel=f"Final {final_metric}",
        output_path=output_dir / f"final_{final_metric}.png",
        dpi=int(args.dpi),
    )
    _write_final_metric_csv(
        grouped_rows,
        metric_key=final_metric,
        output_path=output_dir / f"final_{final_metric}.csv",
    )

    # --------------------------------------------------------------
    # Threshold plots
    # --------------------------------------------------------------
    for threshold in args.thresholds:
        means: List[Optional[float]] = []
        stds: List[Optional[float]] = []

        for label in labels:
            per_run = _steps_to_threshold(
                grouped_rows[label],
                metric_key=final_metric,
                threshold=float(threshold),
            )
            mean_steps, std_steps, _ = _mean_std(per_run.values())
            means.append(mean_steps)
            stds.append(std_steps)

        suffix = str(threshold).replace(".", "p")
        _plot_bar_summary(
            labels,
            means,
            stds,
            title=f"{args.title_prefix}: steps to reach {final_metric} <= {threshold}",
            ylabel="Target sampled points to threshold",
            output_path=output_dir / f"steps_to_threshold_{final_metric}_{suffix}.png",
            dpi=int(args.dpi),
        )
        _write_threshold_csv(
            grouped_rows,
            metric_key=final_metric,
            threshold=float(threshold),
            output_path=output_dir / f"steps_to_threshold_{final_metric}_{suffix}.csv",
        )

    # --------------------------------------------------------------
    # Save a small manifest
    # --------------------------------------------------------------
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_runs_loaded": int(len(runs)),
        "included_modes": list(args.modes),
        "metric": final_metric,
        "thresholds": [float(x) for x in args.thresholds],
        "method_labels": labels,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()