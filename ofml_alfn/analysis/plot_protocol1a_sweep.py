#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunRecord:
    combo_dir: Path
    result_path: Path
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]
    selections: List[Dict[str, Any]]
    experiment_mode: str
    acquisition_policy: Optional[str]
    candidate_pool_scope: Optional[str]
    trial: Optional[int]
    seed: Optional[int]
    protocol_costs: Tuple[float, ...]
    similarities_to_target: Tuple[float, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Protocol 1A sweep results over multiple cost/similarity combinations."
    )
    parser.add_argument("--root_dir", type=str, default="outputs/protocol1a")
    parser.add_argument("--output_dir", type=str, default="outputs/protocol1a_sweep_plots")
    parser.add_argument(
        "--metric",
        type=str,
        default="target_test_loss",
        choices=["target_val_loss", "target_test_loss"],
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max_step", type=int, default=None)
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


def _to_float_tuple(xs: Any) -> Tuple[float, ...]:
    if not isinstance(xs, list):
        return tuple()

    out: List[float] = []
    for x in xs:
        xf = _safe_float(x)
        if xf is not None:
            out.append(float(xf))
    return tuple(out)


def _format_triplet(values: Sequence[float]) -> str:
    if len(values) == 0:
        return "unknown"

    parts = []
    for v in values:
        if abs(v - round(v)) < 1e-12:
            parts.append(str(int(round(v))))
        else:
            parts.append(str(v))
    return "[" + ", ".join(parts) + "]"


def _combo_label(costs: Sequence[float], sims: Sequence[float]) -> str:
    return f"costs={_format_triplet(costs)} | sims={_format_triplet(sims)}"


def _run_candidate_pool_scope(summary: Dict[str, Any]) -> Optional[str]:
    scope = summary.get("candidate_pool_scope", None)
    if scope is not None:
        return str(scope)

    options = summary.get("options", {}) or {}
    scope = options.get("candidate_pool_scope", None)
    if scope is not None:
        return str(scope)

    return None


def _scan_combo_dirs(root_dir: Path) -> List[Path]:
    combo_dirs = []
    for p in sorted(root_dir.glob("costs_*__sims_*")):
        if p.is_dir():
            combo_dirs.append(p)
    return combo_dirs


def _load_runs_from_combo_dir(combo_dir: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []

    for path in sorted(combo_dir.glob("*.json")):
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
                combo_dir=combo_dir,
                result_path=path,
                summary=summary,
                history=history,
                selections=selections,
                experiment_mode=str(summary.get("experiment_mode", "")),
                acquisition_policy=None
                if options.get("target_acquisition_policy", None) is None
                else str(options.get("target_acquisition_policy")),
                candidate_pool_scope=_run_candidate_pool_scope(summary),
                trial=_safe_int(summary.get("trial", None)),
                seed=_safe_int(summary.get("seed", None)),
                protocol_costs=_to_float_tuple(summary.get("protocol_costs", [])),
                similarities_to_target=_to_float_tuple(
                    summary.get("similarities_to_target", [])
                ),
            )
        )

    return runs


def _load_all_runs(root_dir: Path) -> Dict[Path, List[RunRecord]]:
    out: Dict[Path, List[RunRecord]] = {}
    for combo_dir in _scan_combo_dirs(root_dir):
        runs = _load_runs_from_combo_dir(combo_dir)
        if len(runs) > 0:
            out[combo_dir] = runs
    return out


def _aggregate_curve(
    rows: Sequence[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[float, List[float]] = {}

    for row in rows:
        x = _safe_float(row.get(x_key, None))
        y = _safe_float(row.get(y_key, None))
        if x is None or y is None:
            continue
        grouped.setdefault(float(x), []).append(float(y))

    xs, ys, es = [], [], []
    for x in sorted(grouped.keys()):
        arr = np.asarray(grouped[x], dtype=float)
        xs.append(float(x))
        ys.append(float(arr.mean()))
        es.append(float(arr.std(ddof=0)))

    return np.asarray(xs), np.asarray(ys), np.asarray(es)


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
        xs, ys, es = _aggregate_curve(grouped_rows[label], x_key=x_key, y_key=y_key)
        if xs.size == 0:
            continue
        plt.plot(xs, ys, label=label)
        plt.fill_between(xs, ys - es, ys + es, alpha=0.20)

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

    plt.figure(figsize=(7.0, 5.0))
    plt.bar(x, y, yerr=e, capsize=4.0)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _plot_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    if matrix.size == 0:
        return

    plt.figure(figsize=(1.8 + 1.2 * len(col_labels), 1.8 + 0.7 * len(row_labels)))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=20, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            text = "nan" if np.isnan(v) else f"{v:.3f}"
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _final_metric_from_history(
    history: Sequence[Dict[str, Any]],
    *,
    metric_key: str,
    order_key: str,
) -> Optional[float]:
    usable = []
    for row in history:
        order = _safe_int(row.get(order_key, None))
        value = _safe_float(row.get(metric_key, None))
        if order is None:
            continue
        usable.append((order, value))

    usable = sorted(usable, key=lambda x: x[0])

    final_value = None
    for _, val in usable:
        if val is not None:
            final_value = float(val)

    return final_value


def _mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return None, None, 0

    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def _extract_family1_rows(
    runs: Sequence[RunRecord],
    *,
    metric_key: str,
    max_step: Optional[int],
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
            if max_step is not None and step_idx > int(max_step):
                continue

            rec = {
                "seed": run.seed,
                "trial": run.trial,
                "step_idx": step_idx,
                "n_target_points_used": _safe_float(row.get("n_target_points_used", None)),
                "target_cost_used": _safe_float(row.get("target_cost_used", None)),
                metric_key: _safe_float(row.get(metric_key, None)),
            }
            out.setdefault(policy, {}).setdefault(label, []).append(rec)

    return out


def _extract_family2_rows(
    runs: Sequence[RunRecord],
    *,
    metric_key: str,
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
                "seed": run.seed,
                "trial": run.trial,
                "round_idx": round_idx,
                "spent_budget": _safe_float(row.get("spent_budget", None)),
                metric_key: _safe_float(row.get(metric_key, None)),
            }
            out.setdefault(policy, []).append(rec)

    return out


def _extract_family3_rows(
    runs: Sequence[RunRecord],
    *,
    metric_key: str,
    max_round: Optional[int],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Returns:
        out[policy][candidate_pool_scope] = rows

    Family 3 compares:
        all_protocols candidate pool
        target_only candidate pool
    under the same active learning setup.
    """
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for run in runs:
        if run.experiment_mode != "family3_candidate_pool_ablation":
            continue

        policy = run.acquisition_policy or "unknown_policy"
        scope = run.candidate_pool_scope or "unknown_pool"

        for row in run.history:
            round_idx = _safe_int(row.get("round_idx", None))
            if round_idx is None:
                continue
            if max_round is not None and round_idx > int(max_round):
                continue

            rec = {
                "seed": run.seed,
                "trial": run.trial,
                "round_idx": round_idx,
                "spent_budget": _safe_float(row.get("spent_budget", None)),
                metric_key: _safe_float(row.get(metric_key, None)),
            }
            out.setdefault(policy, {}).setdefault(scope, []).append(rec)

    return out


def _selection_counts_for_mode(
    runs: Sequence[RunRecord],
    *,
    experiment_mode: str,
) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}

    for run in runs:
        if run.experiment_mode != experiment_mode:
            continue

        policy = run.acquisition_policy or "unknown_policy"
        if experiment_mode == "family3_candidate_pool_ablation":
            scope = run.candidate_pool_scope or "unknown_pool"
            label = f"{policy}:{scope}"
        else:
            label = policy

        for row in run.selections:
            if row.get("was_selected", True) is False:
                continue
            protocol_id = str(row.get("protocol_id", "unknown_protocol"))
            counts.setdefault(label, {}).setdefault(protocol_id, 0.0)
            counts[label][protocol_id] += 1.0

    return counts


def _plot_stacked_counts(
    counts: Dict[str, Dict[str, float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    policies = sorted(counts.keys())
    protocol_ids = sorted({pid for d in counts.values() for pid in d.keys()})

    x = np.arange(len(policies), dtype=float)
    bottom = np.zeros(len(policies), dtype=float)

    plt.figure(figsize=(7.8, 5.0))
    for protocol_id in protocol_ids:
        vals = np.asarray(
            [counts.get(p, {}).get(protocol_id, 0.0) for p in policies],
            dtype=float,
        )
        plt.bar(x, vals, bottom=bottom, label=protocol_id)
        bottom += vals

    plt.xticks(x, policies, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def _combo_sort_key(
    combo: Tuple[Tuple[float, ...], Tuple[float, ...]]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return combo


def _make_combo_tables(
    all_runs: Dict[Path, List[RunRecord]],
    *,
    metric_key: str,
) -> Tuple[
    Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, float]],
    Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, float]],
    Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, Dict[str, float]]],
]:
    family1_gap: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, float]] = {}
    family2_final: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], Dict[str, float]] = {}
    family3_final: Dict[
        Tuple[Tuple[float, ...], Tuple[float, ...]],
        Dict[str, Dict[str, float]],
    ] = {}

    for _, runs in all_runs.items():
        if len(runs) == 0:
            continue

        costs = runs[0].protocol_costs
        sims = runs[0].similarities_to_target
        combo = (costs, sims)

        # Family 1:
        # pretrain final loss - scratch final loss
        for policy in ["random", "local_uncertainty", "fantasy"]:
            scratch_vals = []
            pretrain_vals = []

            for run in runs:
                if run.acquisition_policy != policy:
                    continue

                if run.experiment_mode == "scratch_then_sequential_adapt":
                    scratch_vals.append(
                        _final_metric_from_history(
                            run.history,
                            metric_key=metric_key,
                            order_key="step_idx",
                        )
                    )
                elif run.experiment_mode == "pretrain_then_sequential_adapt":
                    pretrain_vals.append(
                        _final_metric_from_history(
                            run.history,
                            metric_key=metric_key,
                            order_key="step_idx",
                        )
                    )

            scratch_mean, _, _ = _mean_std(scratch_vals)
            pretrain_mean, _, _ = _mean_std(pretrain_vals)

            if scratch_mean is not None and pretrain_mean is not None:
                family1_gap.setdefault(combo, {})[policy] = float(pretrain_mean - scratch_mean)

        # Family 2:
        # final metric per policy
        for policy in ["random", "local_uncertainty", "fantasy"]:
            vals = []

            for run in runs:
                if run.experiment_mode != "fantasy_al":
                    continue
                if run.acquisition_policy != policy:
                    continue

                vals.append(
                    _final_metric_from_history(
                        run.history,
                        metric_key=metric_key,
                        order_key="round_idx",
                    )
                )

            mean_val, _, _ = _mean_std(vals)
            if mean_val is not None:
                family2_final.setdefault(combo, {})[policy] = float(mean_val)

        # Family 3:
        # final metric per policy and candidate pool scope
        for policy in ["random", "local_uncertainty", "fantasy"]:
            for scope in ["all_protocols", "target_only"]:
                vals = []

                for run in runs:
                    if run.experiment_mode != "family3_candidate_pool_ablation":
                        continue
                    if run.acquisition_policy != policy:
                        continue
                    if run.candidate_pool_scope != scope:
                        continue

                    vals.append(
                        _final_metric_from_history(
                            run.history,
                            metric_key=metric_key,
                            order_key="round_idx",
                        )
                    )

                mean_val, _, _ = _mean_std(vals)
                if mean_val is not None:
                    family3_final.setdefault(combo, {}).setdefault(policy, {})[scope] = float(mean_val)

    return family1_gap, family2_final, family3_final


def _write_summary_csv(
    all_runs: Dict[Path, List[RunRecord]],
    *,
    metric_key: str,
    output_path: Path,
) -> None:
    lines = [
        "combo_dir,protocol_costs,similarities_to_target,"
        "experiment_family,experiment_mode,candidate_pool_scope,"
        "acquisition_policy,trial,seed,final_metric"
    ]

    for combo_dir, runs in sorted(all_runs.items(), key=lambda kv: str(kv[0])):
        for run in runs:
            if run.experiment_mode in {
                "scratch_then_sequential_adapt",
                "pretrain_then_sequential_adapt",
            }:
                final_metric = _final_metric_from_history(
                    run.history,
                    metric_key=metric_key,
                    order_key="step_idx",
                )
            elif run.experiment_mode in {
                "fantasy_al",
                "family3_candidate_pool_ablation",
            }:
                final_metric = _final_metric_from_history(
                    run.history,
                    metric_key=metric_key,
                    order_key="round_idx",
                )
            else:
                final_metric = None

            family = str(run.summary.get("experiment_family", ""))

            line = ",".join(
                [
                    str(combo_dir),
                    "\"" + _format_triplet(run.protocol_costs) + "\"",
                    "\"" + _format_triplet(run.similarities_to_target) + "\"",
                    family,
                    run.experiment_mode,
                    "" if run.candidate_pool_scope is None else run.candidate_pool_scope,
                    "" if run.acquisition_policy is None else run.acquisition_policy,
                    "" if run.trial is None else str(run.trial),
                    "" if run.seed is None else str(run.seed),
                    "" if final_metric is None else str(final_metric),
                ]
            )
            lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_family1(
    *,
    runs: Sequence[RunRecord],
    combo_title: str,
    combo_out: Path,
    args: argparse.Namespace,
) -> None:
    fam1 = _extract_family1_rows(
        runs,
        metric_key=args.metric,
        max_step=args.max_step,
    )

    fam1_out = combo_out / "family1"
    fam1_out.mkdir(parents=True, exist_ok=True)

    for policy, grouped in fam1.items():
        grouped = {k: v for k, v in grouped.items() if len(v) > 0}
        if len(grouped) == 0:
            continue

        _plot_curve(
            grouped,
            x_key="n_target_points_used",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 1 | {policy}\n{combo_title}",
            xlabel="Target sampled points",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam1_out / f"{policy}_{args.metric}_vs_target_points.png",
            dpi=args.dpi,
        )

        _plot_curve(
            grouped,
            x_key="target_cost_used",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 1 | {policy}\n{combo_title}",
            xlabel="Target acquisition cost",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam1_out / f"{policy}_{args.metric}_vs_target_cost.png",
            dpi=args.dpi,
        )

        labels = ["scratch", "pretrain"]
        means, stds = [], []

        for label in labels:
            vals = []
            for run in runs:
                if run.acquisition_policy != policy:
                    continue

                if label == "scratch" and run.experiment_mode == "scratch_then_sequential_adapt":
                    vals.append(
                        _final_metric_from_history(
                            run.history,
                            metric_key=args.metric,
                            order_key="step_idx",
                        )
                    )

                if label == "pretrain" and run.experiment_mode == "pretrain_then_sequential_adapt":
                    vals.append(
                        _final_metric_from_history(
                            run.history,
                            metric_key=args.metric,
                            order_key="step_idx",
                        )
                    )

            mean_val, std_val, _ = _mean_std(vals)
            means.append(mean_val)
            stds.append(std_val)

        _plot_bar(
            labels,
            means,
            stds,
            title=f"{args.title_prefix} Family 1 final {args.metric} | {policy}\n{combo_title}",
            ylabel=f"Final Protocol 3 {args.metric}",
            output_path=fam1_out / f"{policy}_final_{args.metric}.png",
            dpi=args.dpi,
        )


def _plot_family2(
    *,
    runs: Sequence[RunRecord],
    combo_title: str,
    combo_out: Path,
    args: argparse.Namespace,
) -> None:
    fam2 = _extract_family2_rows(
        runs,
        metric_key=args.metric,
        max_round=args.max_round,
    )

    fam2_out = combo_out / "family2"
    fam2_out.mkdir(parents=True, exist_ok=True)

    if len(fam2) == 0:
        return

    _plot_curve(
        fam2,
        x_key="spent_budget",
        y_key=args.metric,
        title=f"{args.title_prefix} Family 2\n{combo_title}",
        xlabel="Spent budget",
        ylabel=f"Protocol 3 {args.metric}",
        output_path=fam2_out / f"{args.metric}_vs_spent_budget.png",
        dpi=args.dpi,
    )

    _plot_curve(
        fam2,
        x_key="round_idx",
        y_key=args.metric,
        title=f"{args.title_prefix} Family 2\n{combo_title}",
        xlabel="Round",
        ylabel=f"Protocol 3 {args.metric}",
        output_path=fam2_out / f"{args.metric}_vs_round.png",
        dpi=args.dpi,
    )

    labels = ["random", "local_uncertainty", "fantasy"]
    means, stds = [], []

    for label in labels:
        vals = []
        for run in runs:
            if run.experiment_mode != "fantasy_al":
                continue
            if run.acquisition_policy != label:
                continue
            vals.append(
                _final_metric_from_history(
                    run.history,
                    metric_key=args.metric,
                    order_key="round_idx",
                )
            )

        mean_val, std_val, _ = _mean_std(vals)
        means.append(mean_val)
        stds.append(std_val)

    _plot_bar(
        labels,
        means,
        stds,
        title=f"{args.title_prefix} Family 2 final {args.metric}\n{combo_title}",
        ylabel=f"Final Protocol 3 {args.metric}",
        output_path=fam2_out / f"final_{args.metric}.png",
        dpi=args.dpi,
    )

    counts = _selection_counts_for_mode(runs, experiment_mode="fantasy_al")
    if len(counts) > 0:
        _plot_stacked_counts(
            counts,
            title=f"{args.title_prefix} Family 2 selected protocols\n{combo_title}",
            ylabel="Selected acquisition count",
            output_path=fam2_out / "selected_protocol_counts.png",
            dpi=args.dpi,
        )


def _plot_family3(
    *,
    runs: Sequence[RunRecord],
    combo_title: str,
    combo_out: Path,
    args: argparse.Namespace,
) -> None:
    fam3 = _extract_family3_rows(
        runs,
        metric_key=args.metric,
        max_round=args.max_round,
    )

    fam3_out = combo_out / "family3"
    fam3_out.mkdir(parents=True, exist_ok=True)

    if len(fam3) == 0:
        return

    for policy, grouped_by_scope in fam3.items():
        grouped_by_scope = {
            scope: rows for scope, rows in grouped_by_scope.items() if len(rows) > 0
        }
        if len(grouped_by_scope) == 0:
            continue

        _plot_curve(
            grouped_by_scope,
            x_key="spent_budget",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 3 | {policy}\n{combo_title}",
            xlabel="Spent budget",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam3_out / f"{policy}_{args.metric}_vs_spent_budget.png",
            dpi=args.dpi,
        )

        _plot_curve(
            grouped_by_scope,
            x_key="round_idx",
            y_key=args.metric,
            title=f"{args.title_prefix} Family 3 | {policy}\n{combo_title}",
            xlabel="Round",
            ylabel=f"Protocol 3 {args.metric}",
            output_path=fam3_out / f"{policy}_{args.metric}_vs_round.png",
            dpi=args.dpi,
        )

        labels = ["all_protocols", "target_only"]
        means, stds = [], []

        for label in labels:
            vals = []

            for run in runs:
                if run.experiment_mode != "family3_candidate_pool_ablation":
                    continue
                if run.acquisition_policy != policy:
                    continue
                if run.candidate_pool_scope != label:
                    continue

                vals.append(
                    _final_metric_from_history(
                        run.history,
                        metric_key=args.metric,
                        order_key="round_idx",
                    )
                )

            mean_val, std_val, _ = _mean_std(vals)
            means.append(mean_val)
            stds.append(std_val)

        _plot_bar(
            labels,
            means,
            stds,
            title=f"{args.title_prefix} Family 3 final {args.metric} | {policy}\n{combo_title}",
            ylabel=f"Final Protocol 3 {args.metric}",
            output_path=fam3_out / f"{policy}_final_{args.metric}.png",
            dpi=args.dpi,
        )

    counts = _selection_counts_for_mode(
        runs,
        experiment_mode="family3_candidate_pool_ablation",
    )
    if len(counts) > 0:
        _plot_stacked_counts(
            counts,
            title=f"{args.title_prefix} Family 3 selected protocols\n{combo_title}",
            ylabel="Selected acquisition count",
            output_path=fam3_out / "selected_protocol_counts.png",
            dpi=args.dpi,
        )


def _plot_sweep_heatmaps(
    *,
    all_runs: Dict[Path, List[RunRecord]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    family1_gap, family2_final, family3_final = _make_combo_tables(
        all_runs,
        metric_key=args.metric,
    )

    combos = sorted(
        set(list(family1_gap.keys()) + list(family2_final.keys()) + list(family3_final.keys())),
        key=_combo_sort_key,
    )

    if len(combos) == 0:
        return

    cost_labels = sorted({_format_triplet(costs) for costs, _ in combos})
    sim_labels = sorted({_format_triplet(sims) for _, sims in combos})
    combo_lookup = {(_format_triplet(c), _format_triplet(s)): (c, s) for c, s in combos}

    # Family 1 heatmaps: pretrain - scratch
    for policy in ["random", "local_uncertainty", "fantasy"]:
        mat = np.full((len(cost_labels), len(sim_labels)), np.nan, dtype=float)

        for i, c_label in enumerate(cost_labels):
            for j, s_label in enumerate(sim_labels):
                combo = combo_lookup.get((c_label, s_label), None)
                if combo is None:
                    continue

                val = family1_gap.get(combo, {}).get(policy, None)
                if val is not None:
                    mat[i, j] = float(val)

        _plot_heatmap(
            mat,
            row_labels=cost_labels,
            col_labels=sim_labels,
            title=f"{args.title_prefix} sweep | Family 1 | {policy}\n(pretrain final loss - scratch final loss)",
            colorbar_label="Loss gap",
            output_path=output_dir / f"heatmap_family1_gap_{policy}_{args.metric}.png",
            dpi=args.dpi,
        )

    # Family 2 heatmaps: final metric per policy
    for policy in ["random", "local_uncertainty", "fantasy"]:
        mat = np.full((len(cost_labels), len(sim_labels)), np.nan, dtype=float)

        for i, c_label in enumerate(cost_labels):
            for j, s_label in enumerate(sim_labels):
                combo = combo_lookup.get((c_label, s_label), None)
                if combo is None:
                    continue

                val = family2_final.get(combo, {}).get(policy, None)
                if val is not None:
                    mat[i, j] = float(val)

        _plot_heatmap(
            mat,
            row_labels=cost_labels,
            col_labels=sim_labels,
            title=f"{args.title_prefix} sweep | Family 2 | {policy}\nfinal {args.metric}",
            colorbar_label=f"Final {args.metric}",
            output_path=output_dir / f"heatmap_family2_final_{policy}_{args.metric}.png",
            dpi=args.dpi,
        )

    # Family 3 heatmaps: final metric by policy and candidate pool scope
    for policy in ["random", "local_uncertainty", "fantasy"]:
        for scope in ["all_protocols", "target_only"]:
            mat = np.full((len(cost_labels), len(sim_labels)), np.nan, dtype=float)

            for i, c_label in enumerate(cost_labels):
                for j, s_label in enumerate(sim_labels):
                    combo = combo_lookup.get((c_label, s_label), None)
                    if combo is None:
                        continue

                    val = family3_final.get(combo, {}).get(policy, {}).get(scope, None)
                    if val is not None:
                        mat[i, j] = float(val)

            _plot_heatmap(
                mat,
                row_labels=cost_labels,
                col_labels=sim_labels,
                title=f"{args.title_prefix} sweep | Family 3 | {policy} | {scope}\nfinal {args.metric}",
                colorbar_label=f"Final {args.metric}",
                output_path=output_dir / f"heatmap_family3_final_{policy}_{scope}_{args.metric}.png",
                dpi=args.dpi,
            )

        # Family 3 heatmap: all_protocols - target_only.
        # Negative means all_protocols achieved lower final loss.
        mat = np.full((len(cost_labels), len(sim_labels)), np.nan, dtype=float)

        for i, c_label in enumerate(cost_labels):
            for j, s_label in enumerate(sim_labels):
                combo = combo_lookup.get((c_label, s_label), None)
                if combo is None:
                    continue

                all_val = family3_final.get(combo, {}).get(policy, {}).get("all_protocols", None)
                tgt_val = family3_final.get(combo, {}).get(policy, {}).get("target_only", None)

                if all_val is not None and tgt_val is not None:
                    mat[i, j] = float(all_val - tgt_val)

        _plot_heatmap(
            mat,
            row_labels=cost_labels,
            col_labels=sim_labels,
            title=f"{args.title_prefix} sweep | Family 3 | {policy}\n(all_protocols final loss - target_only final loss)",
            colorbar_label="Loss gap",
            output_path=output_dir / f"heatmap_family3_gap_all_minus_target_{policy}_{args.metric}.png",
            dpi=args.dpi,
        )


def main() -> None:
    args = _parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = _load_all_runs(root_dir)
    if len(all_runs) == 0:
        raise ValueError(f"No combo result directories found under {root_dir}")

    # Per-combination plots
    for combo_dir, runs in sorted(all_runs.items(), key=lambda kv: str(kv[0])):
        combo_out = output_dir / combo_dir.name
        combo_out.mkdir(parents=True, exist_ok=True)

        costs = runs[0].protocol_costs if len(runs) > 0 else tuple()
        sims = runs[0].similarities_to_target if len(runs) > 0 else tuple()
        combo_title = _combo_label(costs, sims)

        _plot_family1(
            runs=runs,
            combo_title=combo_title,
            combo_out=combo_out,
            args=args,
        )
        _plot_family2(
            runs=runs,
            combo_title=combo_title,
            combo_out=combo_out,
            args=args,
        )
        _plot_family3(
            runs=runs,
            combo_title=combo_title,
            combo_out=combo_out,
            args=args,
        )

    # Sweep summary CSV
    _write_summary_csv(
        all_runs,
        metric_key=args.metric,
        output_path=output_dir / f"sweep_summary_{args.metric}.csv",
    )

    # Sweep heatmaps
    _plot_sweep_heatmaps(
        all_runs=all_runs,
        output_dir=output_dir,
        args=args,
    )

    manifest = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "n_combo_dirs": len(all_runs),
        "metric": args.metric,
        "families_supported": ["family1", "family2", "family3"],
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()