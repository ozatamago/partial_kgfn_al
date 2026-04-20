#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]
    selections: List[Dict[str, Any]]
    experiment_mode: str
    acquisition_policy: Optional[str]
    trial: Optional[int]
    seed: Optional[int]


@dataclass(frozen=True)
class CurvePoint:
    x: float
    mean: float
    std: float
    n: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Protocol 1A Family 1 / Family 2 results from saved JSON files."
    )
    parser.add_argument("--input_dir", type=str, default="outputs/protocol1a")
    parser.add_argument("--output_dir", type=str, default="outputs/protocol1a_plots")
    parser.add_argument(
        "--metric",
        type=str,
        default="target_test_loss",
        choices=["target_val_loss", "target_test_loss"],
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max_round", type=int, default=None)
    parser.add_argument("--title_prefix", type=str, default="Protocol 1A")
    return parser.parse_args()


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
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _is_summary_json(path: Path) -> bool:
    return path.suffix.lower() == ".json" and not path.name.endswith("_selections.json")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _paired_selection_path(path: Path) -> Path:
    return path.with_name(path.stem + "_selections.json")


def _load_runs(input_dir: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    for path in sorted(input_dir.glob("*.json")):
        if not _is_summary_json(path):
            continue
        summary = _load_json(path)
        history = summary.get("history", [])
        if not isinstance(history, list):
            history = []
        sel_path = _paired_selection_path(path)
        selections: List[Dict[str, Any]] = []
        if sel_path.exists():
            loaded = _load_json(sel_path)
            if isinstance(loaded, list):
                selections = loaded
        options = summary.get("options", {}) or {}
        runs.append(
            RunRecord(
                result_path=path,
                summary=summary,
                history=history,
                selections=selections,
                experiment_mode=str(summary.get("experiment_mode", "")),
                acquisition_policy=None
                if options.get("target_acquisition_policy", None) is None
                else str(options.get("target_acquisition_policy")),
                trial=_safe_int(summary.get("trial", None)),
                seed=_safe_int(summary.get("seed", None)),
            )
        )
    return runs


def _mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def _aggregate_curve(rows: Sequence[Dict[str, Any]], *, x_key: str, y_key: str) -> List[CurvePoint]:
    grouped: Dict[float, List[float]] = {}
    for row in rows:
        x = _safe_float(row.get(x_key, None))
        y = _safe_float(row.get(y_key, None))
        if x is None or y is None:
            continue
        grouped.setdefault(x, []).append(y)
    out: List[CurvePoint] = []
    for x in sorted(grouped.keys()):
        arr = np.asarray(grouped[x], dtype=float)
        out.append(CurvePoint(x=x, mean=float(arr.mean()), std=float(arr.std(ddof=0)), n=int(arr.size)))
    return out


def _plot_curve(
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
    for label in sorted(grouped_rows.keys()):
        curve = _aggregate_curve(grouped_rows[label], x_key=x_key, y_key=y_key)
        if not curve:
            continue
        xs = np.asarray([p.x for p in curve], dtype=float)
        ys = np.asarray([p.mean for p in curve], dtype=float)
        stds = np.asarray([p.std for p in curve], dtype=float)
        plt.plot(xs, ys, label=label)
        plt.fill_between(xs, ys - stds, ys + stds, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_bar(
    labels: Sequence[str],
    means: Sequence[Optional[float]],
    stds: Sequence[Optional[float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    x = np.arange(len(labels), dtype=float)
    y = np.asarray([np.nan if v is None else float(v) for v in means], dtype=float)
    e = np.asarray([0.0 if v is None else float(v) for v in stds], dtype=float)
    plt.figure(figsize=(7.2, 5.0))
    plt.bar(x, y, yerr=e, capsize=4.0)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _family1_sequential_rows(
    runs: Sequence[RunRecord],
    *,
    metric_keys: Sequence[str],
    max_round: Optional[int],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for run in runs:
        if run.experiment_mode not in {
            "scratch_then_sequential_adapt",
            "pretrain_then_sequential_adapt",
        }:
            continue
        policy = run.acquisition_policy or "unknown_policy"
        label = "scratch" if run.experiment_mode == "scratch_then_sequential_adapt" else "pretrain"
        for row in run.history:
            step_idx = _safe_int(row.get("step_idx", None))
            if step_idx is None:
                continue
            if max_round is not None and step_idx > int(max_round):
                continue
            rec = {
                "label": label,
                "policy": policy,
                "seed": run.seed,
                "trial": run.trial,
                "step_idx": step_idx,
                "n_target_points_used": _safe_int(row.get("n_target_points_used", None)),
                "target_cost_used": _safe_float(row.get("target_cost_used", None)),
            }
            for k in metric_keys:
                rec[k] = _safe_float(row.get(k, None))
            out.setdefault(policy, {}).setdefault(label, []).append(rec)
    return out


def _family1_one_shot_rows(runs: Sequence[RunRecord]) -> Dict[str, List[Optional[float]]]:
    out: Dict[str, List[Optional[float]]] = {"scratch": [], "pretrain_then_adapt": []}
    for run in runs:
        if run.experiment_mode not in {"pretrain_then_adapt_comparison", "pretrain_then_adapt"}:
            continue
        scratch = _safe_float(run.summary.get("scratch_target_test_loss", None))
        if scratch is None:
            scratch = _safe_float(run.summary.get("scratch_test_loss", None))
        transfer = _safe_float(run.summary.get("final_target_test_loss", None))
        if transfer is None:
            transfer = _safe_float(run.summary.get("transfer_test_loss", None))
        out["scratch"].append(scratch)
        out["pretrain_then_adapt"].append(transfer)
    return out


def _family2_rows(
    runs: Sequence[RunRecord],
    *,
    metric_keys: Sequence[str],
    max_round: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        if run.experiment_mode != "fantasy_al":
            continue
        policy = run.acquisition_policy or "unknown_policy"
        for row in run.history:
            round_idx = _safe_int(row.get("round_idx", None))
            if round_idx is None:
                continue
            if max_round is not None and round_idx > int(max_round):
                continue
            rec = {
                "label": policy,
                "seed": run.seed,
                "trial": run.trial,
                "round_idx": round_idx,
                "spent_budget": _safe_float(row.get("spent_budget", None)),
                "remaining_budget": _safe_float(row.get("remaining_budget", None)),
                "train_size": _safe_int(row.get("train_size", None)),
            }
            for k in metric_keys:
                rec[k] = _safe_float(row.get(k, None))
            out.setdefault(policy, []).append(rec)
    return out


def _family2_selection_summary(
    runs: Sequence[RunRecord],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    counts: Dict[str, Dict[str, float]] = {}
    costs: Dict[str, Dict[str, float]] = {}
    for run in runs:
        if run.experiment_mode != "fantasy_al":
            continue
        policy = run.acquisition_policy or "unknown_policy"
        for row in run.selections:
            was_selected = row.get("was_selected", True)
            if was_selected is False:
                continue
            protocol_id = str(row.get("protocol_id", "unknown_protocol"))
            counts.setdefault(policy, {}).setdefault(protocol_id, 0.0)
            counts[policy][protocol_id] += 1.0
            acq_cost = _safe_float(row.get("acquisition_cost", None))
            costs.setdefault(policy, {}).setdefault(protocol_id, 0.0)
            if acq_cost is not None:
                costs[policy][protocol_id] += float(acq_cost)
    return counts, costs


def _plot_stacked_selection(
    summary_by_policy: Dict[str, Dict[str, float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    policies = sorted(summary_by_policy.keys())
    protocol_ids = sorted({pid for per_policy in summary_by_policy.values() for pid in per_policy.keys()})
    x = np.arange(len(policies), dtype=float)
    bottom = np.zeros(len(policies), dtype=float)

    plt.figure(figsize=(7.6, 5.2))
    for protocol_id in protocol_ids:
        heights = np.asarray(
            [float(summary_by_policy.get(policy, {}).get(protocol_id, 0.0)) for policy in policies],
            dtype=float,
        )
        plt.bar(x, heights, bottom=bottom, label=protocol_id)
        bottom += heights

    plt.xticks(x, policies, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _final_metric_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    order_key: str,
    metric_key: str,
) -> List[Optional[float]]:
    grouped: Dict[Tuple[Optional[int], Optional[int]], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("seed", None), row.get("trial", None))
        grouped.setdefault(key, []).append(row)

    finals: List[Optional[float]] = []
    for _, run_rows in grouped.items():
        run_rows = sorted(run_rows, key=lambda r: int(r[order_key]))
        final_val: Optional[float] = None
        for row in run_rows:
            val = _safe_float(row.get(metric_key, None))
            if val is not None:
                final_val = val
        finals.append(final_val)
    return finals


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(input_dir)
    if not runs:
        raise ValueError(f"No result JSON files found in {input_dir}")

    metric_keys = ["target_val_loss", "target_test_loss"]

    fam1_seq = _family1_sequential_rows(runs, metric_keys=metric_keys, max_round=args.max_round)
    fam1_seq_dir = output_dir / "family1_sequential"
    fam1_seq_dir.mkdir(parents=True, exist_ok=True)

    for policy, per_label in fam1_seq.items():
        grouped = {label: rows for label, rows in per_label.items() if rows}
        if not grouped:
            continue

        _plot_curve(
            grouped,
            x_key="n_target_points_used",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 1 ({policy}): {args.metric} vs target sampled points",
            xlabel="Target sampled points",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam1_seq_dir / f"{policy}_{args.metric}_vs_target_points.png",
            dpi=args.dpi,
        )

        labels = sorted(grouped.keys())
        means, stds = [], []
        for label in labels:
            finals = _final_metric_from_rows(
                grouped[label],
                order_key="step_idx",
                metric_key=args.metric,
            )
            mean_val, std_val, _ = _mean_std(finals)
            means.append(mean_val)
            stds.append(std_val)

        _plot_bar(
            labels,
            means,
            stds,
            title=f"{args.title_prefix} Family 1 ({policy}): final {args.metric}",
            ylabel=f"Final Protocol 3 {args.metric}",
            output_path=fam1_seq_dir / f"{policy}_final_{args.metric}.png",
            dpi=args.dpi,
        )

    fam1_one = _family1_one_shot_rows(runs)
    fam1_one_dir = output_dir / "family1_one_shot"
    fam1_one_dir.mkdir(parents=True, exist_ok=True)
    if any(len(v) > 0 for v in fam1_one.values()):
        labels = ["scratch", "pretrain_then_adapt"]
        means, stds = [], []
        for label in labels:
            mean_val, std_val, _ = _mean_std(fam1_one[label])
            means.append(mean_val)
            stds.append(std_val)
        _plot_bar(
            labels,
            means,
            stds,
            title=f"{args.title_prefix} Family 1 one-shot: final target test loss",
            ylabel="Final Protocol 3 target_test_loss",
            output_path=fam1_one_dir / "one_shot_final_target_test_loss.png",
            dpi=args.dpi,
        )

    fam2 = _family2_rows(runs, metric_keys=metric_keys, max_round=args.max_round)
    fam2_dir = output_dir / "family2"
    fam2_dir.mkdir(parents=True, exist_ok=True)

    if fam2:
        _plot_curve(
            fam2,
            x_key="spent_budget",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 2: {args.metric} vs spent budget",
            xlabel="Spent budget",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam2_dir / f"{args.metric}_vs_spent_budget.png",
            dpi=args.dpi,
        )
        _plot_curve(
            fam2,
            x_key="round_idx",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 2: {args.metric} vs round",
            xlabel="Active learning round",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam2_dir / f"{args.metric}_vs_round.png",
            dpi=args.dpi,
        )

        labels = sorted(fam2.keys())
        means, stds = [], []
        for label in labels:
            finals = _final_metric_from_rows(
                fam2[label],
                order_key="round_idx",
                metric_key=args.metric,
            )
            mean_val, std_val, _ = _mean_std(finals)
            means.append(mean_val)
            stds.append(std_val)

        _plot_bar(
            labels,
            means,
            stds,
            title=f"{args.title_prefix} Family 2: final {args.metric}",
            ylabel=f"Final Protocol 3 {args.metric}",
            output_path=fam2_dir / f"final_{args.metric}.png",
            dpi=args.dpi,
        )

        counts, costs = _family2_selection_summary(runs)
        if counts:
            _plot_stacked_selection(
                counts,
                title=f"{args.title_prefix} Family 2: selected protocol counts",
                ylabel="Number of selected acquisitions",
                output_path=fam2_dir / "selected_protocol_counts.png",
                dpi=args.dpi,
            )
        if costs:
            _plot_stacked_selection(
                costs,
                title=f"{args.title_prefix} Family 2: spent budget by selected protocol",
                ylabel="Acquisition cost",
                output_path=fam2_dir / "selected_protocol_costs.png",
                dpi=args.dpi,
            )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_runs_loaded": len(runs),
        "family1_sequential_policies": sorted(fam1_seq.keys()),
        "family2_policies": sorted(fam2.keys()),
        "metric": args.metric,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()