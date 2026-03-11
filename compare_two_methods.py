#!/usr/bin/env python3

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch


def _to_scalar_float(x: Any) -> float:
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
        return float(x.item())
    return float(x)


def _to_float_list(xs: Any) -> List[float]:
    if xs is None:
        return []
    if torch.is_tensor(xs):
        if xs.ndim == 1:
            return [float(v) for v in xs.detach().cpu().tolist()]
        raise ValueError(f"Expected 1D tensor, got shape {tuple(xs.shape)}")
    if isinstance(xs, (list, tuple)):
        out = []
        for x in xs:
            if x is None:
                continue
            out.append(_to_scalar_float(x))
        return out
    raise ValueError(f"Unsupported type for float list: {type(xs)}")


def _load_result(path: str) -> Dict[str, Any]:
    obj = torch.load(path, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} does not contain a dict result.")
    return obj


def _extract_curve(result: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    costs = _to_float_list(result.get("cumulative_costs", None))
    losses = _to_float_list(result.get("test_losses", None))

    if len(costs) == 0 or len(losses) == 0:
        raise ValueError("Result must contain non-empty cumulative_costs and test_losses.")

    if len(costs) != len(losses):
        raise ValueError(
            f"Length mismatch: len(costs)={len(costs)} vs len(losses)={len(losses)}"
        )
    return costs, losses


def _best_loss_and_cost(costs: List[float], losses: List[float]) -> Tuple[float, float, int]:
    best_loss = min(losses)
    idx = losses.index(best_loss)
    return best_loss, costs[idx], idx


def _auc_trapezoid(costs: List[float], losses: List[float]) -> float:
    if len(costs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(costs)):
        dx = costs[i] - costs[i - 1]
        area += 0.5 * dx * (losses[i - 1] + losses[i])
    return area


def _cost_to_threshold(costs: List[float], losses: List[float], threshold: float) -> Optional[float]:
    for c, l in zip(costs, losses):
        if l <= threshold:
            return c
    return None


def _loss_at_budget(costs: List[float], losses: List[float], budget: float) -> Optional[float]:
    for c, l in zip(costs, losses):
        if c >= budget:
            return l
    return None


def _best_loss_by_budget(costs: List[float], losses: List[float], budget: float) -> Optional[float]:
    eligible = [l for c, l in zip(costs, losses) if c <= budget]
    if len(eligible) == 0:
        return None
    return min(eligible)


def _format_optional(x: Optional[float]) -> str:
    if x is None:
        return "missing/not reached"
    return f"{x:.6f}"


def _safe_node_counts(result: Dict[str, Any]) -> Optional[List[float]]:
    x = result.get("node_eval_counts", None)
    if x is None:
        return None
    if torch.is_tensor(x):
        return [float(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return None


def _normalize_node_metric_value(value: Any) -> Dict[int, List[float]]:
    """
    Supported inputs:
      1. dict[node_idx] = list/tuple/1D tensor of floats
      2. list/tuple where each element is a per-node list/tensor
      3. 2D tensor [T, n_nodes] or [n_nodes, T]
    Returns:
      dict[node_idx] -> list[float]
    """
    if value is None:
        return {}

    # dict case
    if isinstance(value, dict):
        out: Dict[int, List[float]] = {}
        for k, v in value.items():
            node_idx = int(k)
            out[node_idx] = _to_float_list(v)
        return out

    # tensor case
    if torch.is_tensor(value):
        arr = value.detach().cpu()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor for node metric, got shape {tuple(arr.shape)}")
        # Heuristic: if first dim matches time length less often than second, we still need a choice.
        # We assume [T, n_nodes] if T >= n_nodes; otherwise [n_nodes, T].
        if arr.shape[0] >= arr.shape[1]:
            # [T, n_nodes]
            return {j: [float(x) for x in arr[:, j].tolist()] for j in range(arr.shape[1])}
        else:
            # [n_nodes, T]
            return {j: [float(x) for x in arr[j, :].tolist()] for j in range(arr.shape[0])}

    # list/tuple case
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return {}
        # Assume per-node container: [node0_series, node1_series, ...]
        out: Dict[int, List[float]] = {}
        for j, v in enumerate(value):
            try:
                out[j] = _to_float_list(v)
            except Exception:
                pass
        return out

    raise ValueError(f"Unsupported node metric type: {type(value)}")


def _find_first_present(result: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in result and result[k] is not None:
            return result[k]
    return None


def _extract_intermediate_metrics(result: Dict[str, Any]) -> Dict[str, Dict[int, List[float]]]:
    """
    Tries several key aliases so the compare script can work with evolving result formats.
    """
    tf_value = _find_first_present(
        result,
        [
            "node_test_losses_tf",
            "node_test_loss_tf",
            "node_teacher_forced_losses",
            "teacher_forced_node_losses",
            "node_test_losses_teacher_forced",
        ],
    )
    ro_value = _find_first_present(
        result,
        [
            "node_test_losses_rollout",
            "node_rollout_losses",
            "rollout_node_losses",
            "node_test_loss_rollout",
        ],
    )

    tf = _normalize_node_metric_value(tf_value) if tf_value is not None else {}
    ro = _normalize_node_metric_value(ro_value) if ro_value is not None else {}

    return {
        "teacher_forced": tf,
        "rollout": ro,
    }


def _metric_at_budget(
    metric_series: List[float],
    costs: List[float],
    budget: float,
) -> Optional[float]:
    if len(metric_series) != len(costs):
        return None
    for c, m in zip(costs, metric_series):
        if c >= budget:
            return m
    return None


def _metric_best_by_budget(
    metric_series: List[float],
    costs: List[float],
    budget: float,
    smaller_is_better: bool = True,
) -> Optional[float]:
    if len(metric_series) != len(costs):
        return None
    eligible = [m for c, m in zip(costs, metric_series) if c <= budget]
    if len(eligible) == 0:
        return None
    return min(eligible) if smaller_is_better else max(eligible)


def summarize_result(
    name: str,
    result: Dict[str, Any],
    thresholds: List[float],
    budget_points: List[float],
) -> Dict[str, Any]:
    costs, losses = _extract_curve(result)
    best_loss, best_cost, best_idx = _best_loss_and_cost(costs, losses)
    final_cost, final_loss = costs[-1], losses[-1]
    auc = _auc_trapezoid(costs, losses)
    node_counts = _safe_node_counts(result)

    budget_loss = {b: _loss_at_budget(costs, losses, b) for b in budget_points}
    budget_best_loss = {b: _best_loss_by_budget(costs, losses, b) for b in budget_points}

    interm = _extract_intermediate_metrics(result)

    interm_budget_metrics: Dict[str, Dict[int, Dict[float, Optional[float]]]] = {
        "teacher_forced": {},
        "rollout": {},
    }
    interm_budget_best_metrics: Dict[str, Dict[int, Dict[float, Optional[float]]]] = {
        "teacher_forced": {},
        "rollout": {},
    }

    for mode in ["teacher_forced", "rollout"]:
        for node_idx, series in interm[mode].items():
            interm_budget_metrics[mode][node_idx] = {
                b: _metric_at_budget(series, costs, b) for b in budget_points
            }
            interm_budget_best_metrics[mode][node_idx] = {
                b: _metric_best_by_budget(series, costs, b, smaller_is_better=True)
                for b in budget_points
            }

    return {
        "name": name,
        "n_points": len(costs),
        "best_test_loss": best_loss,
        "cost_at_best_test_loss": best_cost,
        "best_index": best_idx,
        "final_cost": final_cost,
        "final_test_loss": final_loss,
        "auc_test_loss_vs_cost": auc,
        "cost_to_thresholds": {
            th: _cost_to_threshold(costs, losses, th) for th in thresholds
        },
        "budget_loss": budget_loss,
        "budget_best_loss": budget_best_loss,
        "node_eval_counts": node_counts,
        "costs": costs,
        "losses": losses,
        "intermediate": interm,
        "intermediate_budget_metrics": interm_budget_metrics,
        "intermediate_budget_best_metrics": interm_budget_best_metrics,
    }


def print_summary(summary: Dict[str, Any], budget_points: List[float]) -> None:
    print("=" * 80)
    print(summary["name"])
    print("-" * 80)
    print(f"n_points                 : {summary['n_points']}")
    print(f"best_test_loss           : {summary['best_test_loss']:.6f}")
    print(f"cost_at_best_test_loss   : {summary['cost_at_best_test_loss']:.6f}")
    print(f"best_index               : {summary['best_index']}")
    print(f"final_cost               : {summary['final_cost']:.6f}")
    print(f"final_test_loss          : {summary['final_test_loss']:.6f}")
    print(f"AUC(loss vs cost)        : {summary['auc_test_loss_vs_cost']:.6f}")

    print("cost_to_thresholds:")
    for th, c in summary["cost_to_thresholds"].items():
        print(f"  loss <= {th:.6f} : {_format_optional(c)}")

    print("loss_at_budget:")
    for b in budget_points:
        print(f"  budget {b:.1f} : {_format_optional(summary['budget_loss'][b])}")

    print("best_loss_by_budget:")
    for b in budget_points:
        print(f"  budget {b:.1f} : {_format_optional(summary['budget_best_loss'][b])}")

    if summary["node_eval_counts"] is not None:
        print(f"node_eval_counts         : {summary['node_eval_counts']}")

    has_any_intermediate = False
    for mode in ["teacher_forced", "rollout"]:
        mode_dict = summary["intermediate_budget_metrics"].get(mode, {})
        if len(mode_dict) == 0:
            continue
        has_any_intermediate = True
        print(f"{mode}_node_metrics_at_budget:")
        for node_idx in sorted(mode_dict.keys()):
            vals = mode_dict[node_idx]
            best_vals = summary["intermediate_budget_best_metrics"][mode][node_idx]
            print(f"  node {node_idx}:")
            for b in budget_points:
                print(
                    f"    budget {b:.1f} : "
                    f"current={_format_optional(vals[b])}, "
                    f"best_so_far={_format_optional(best_vals[b])}"
                )

    if not has_any_intermediate:
        print("intermediate_node_metrics: missing in result file")


def print_comparison(a: Dict[str, Any], b: Dict[str, Any], budget_points: List[float]) -> None:
    print("=" * 80)
    print("COMPARISON")
    print("-" * 80)

    def winner_smaller(key: str) -> str:
        av = a[key]
        bv = b[key]
        if av < bv:
            return a["name"]
        if bv < av:
            return b["name"]
        return "tie"

    print(
        f"best_test_loss           : "
        f"{a['name']}={a['best_test_loss']:.6f}, "
        f"{b['name']}={b['best_test_loss']:.6f} "
        f"-> better: {winner_smaller('best_test_loss')}"
    )
    print(
        f"cost_at_best_test_loss   : "
        f"{a['name']}={a['cost_at_best_test_loss']:.6f}, "
        f"{b['name']}={b['cost_at_best_test_loss']:.6f} "
        f"-> better: {winner_smaller('cost_at_best_test_loss')}"
    )
    print(
        f"final_test_loss          : "
        f"{a['name']}={a['final_test_loss']:.6f}, "
        f"{b['name']}={b['final_test_loss']:.6f} "
        f"-> better: {winner_smaller('final_test_loss')}"
    )
    print(
        f"AUC(loss vs cost)        : "
        f"{a['name']}={a['auc_test_loss_vs_cost']:.6f}, "
        f"{b['name']}={b['auc_test_loss_vs_cost']:.6f} "
        f"-> better: {winner_smaller('auc_test_loss_vs_cost')}"
    )

    all_thresholds = sorted(set(a["cost_to_thresholds"].keys()) | set(b["cost_to_thresholds"].keys()))
    print("cost_to_thresholds:")
    for th in all_thresholds:
        ca = a["cost_to_thresholds"].get(th, None)
        cb = b["cost_to_thresholds"].get(th, None)
        print(
            f"  loss <= {th:.6f} : "
            f"{a['name']}={_format_optional(ca)}, "
            f"{b['name']}={_format_optional(cb)}"
        )

    print("loss_at_budget:")
    for bd in budget_points:
        la = a["budget_loss"].get(bd, None)
        lb = b["budget_loss"].get(bd, None)
        print(
            f"  budget {bd:.1f} : "
            f"{a['name']}={_format_optional(la)}, "
            f"{b['name']}={_format_optional(lb)}"
        )

    print("best_loss_by_budget:")
    for bd in budget_points:
        la = a["budget_best_loss"].get(bd, None)
        lb = b["budget_best_loss"].get(bd, None)
        print(
            f"  budget {bd:.1f} : "
            f"{a['name']}={_format_optional(la)}, "
            f"{b['name']}={_format_optional(lb)}"
        )

    for mode in ["teacher_forced", "rollout"]:
        a_mode = a["intermediate_budget_metrics"].get(mode, {})
        b_mode = b["intermediate_budget_metrics"].get(mode, {})
        all_nodes = sorted(set(a_mode.keys()) | set(b_mode.keys()))
        if len(all_nodes) == 0:
            continue

        print(f"{mode}_node_metrics_at_budget:")
        for node_idx in all_nodes:
            print(f"  node {node_idx}:")
            for bd in budget_points:
                va = a_mode.get(node_idx, {}).get(bd, None)
                vb = b_mode.get(node_idx, {}).get(bd, None)
                print(
                    f"    budget {bd:.1f} : "
                    f"{a['name']}={_format_optional(va)}, "
                    f"{b['name']}={_format_optional(vb)}"
                )

def _score_to_scalar(x: Any) -> float:
    """
    Convert a score entry to a scalar float.
    If it is a vector/list/tensor, take its mean as a scalar summary.
    """
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        arr = x.detach().cpu().flatten()
        if arr.numel() == 0:
            return float("nan")
        return float(arr.double().mean().item())
    if isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            if v is None:
                continue
            try:
                vals.append(_to_scalar_float(v))
            except Exception:
                # if nested, ignore
                continue
        if len(vals) == 0:
            return float("nan")
        return float(sum(vals) / len(vals))
    # scalar-like
    return _to_scalar_float(x)


def _extract_uncertainty(result: Dict[str, Any]) -> List[float]:
    """
    Tries to extract the uncertainty / acquisition scores used at selection time.
    Expected key: selected_scores. Falls back to a few aliases if needed.
    """
    val = _find_first_present(
        result,
        [
            "selected_scores",
            "acq_scores",
            "acquisition_scores",
            "uncertainty_scores",
        ],
    )
    if val is None:
        return []

    if torch.is_tensor(val):
        # Could be [T] or [T, ...]
        arr = val.detach().cpu()
        if arr.ndim == 1:
            return [float(v) for v in arr.tolist()]
        # If higher-dim, mean-reduce per step along last dims
        arr = arr.view(arr.shape[0], -1).double().mean(dim=1)
        return [float(v) for v in arr.tolist()]

    if isinstance(val, (list, tuple)):
        return [_score_to_scalar(x) for x in val]

    # scalar single value (rare)
    return [_score_to_scalar(val)]


def _uncertainty_vs_delta_loss(
    uncertainties: List[float],
    losses: List[float],
) -> Tuple[List[float], List[float]]:
    """
    Align uncertainty_t with delta_loss_t = loss_{t-1} - loss_t.
    Returns paired lists (u, delta).
    """
    if len(losses) < 2 or len(uncertainties) == 0:
        return [], []

    T = min(len(uncertainties), len(losses) - 1)
    u = uncertainties[:T]
    dloss = [losses[i] - losses[i + 1] for i in range(T)]  # positive = improvement
    return u, dloss


def maybe_save_scatter(
    a_result: Dict[str, Any],
    b_result: Dict[str, Any],
    a_summary: Dict[str, Any],
    b_summary: Dict[str, Any],
    output_png: Optional[str],
) -> None:
    if output_png is None:
        return

    try:
        import math
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skip scatter plot: {e}")
        return

    a_unc = _extract_uncertainty(a_result)
    b_unc = _extract_uncertainty(b_result)

    a_u, a_d = _uncertainty_vs_delta_loss(a_unc, a_summary["losses"])
    b_u, b_d = _uncertainty_vs_delta_loss(b_unc, b_summary["losses"])

    if len(a_u) == 0 and len(b_u) == 0:
        print("[warn] could not find usable selected_scores to create scatter plot.")
        return

    def _filter_pairs(u: List[float], d: List[float]) -> Tuple[List[float], List[float]]:
        uu, dd = [], []
        for x, y in zip(u, d):
            if x is None or y is None:
                continue
            if not (isinstance(x, float) and isinstance(y, float)):
                continue
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                continue
            uu.append(x)
            dd.append(y)
        return uu, dd

    a_u, a_d = _filter_pairs(a_u, a_d)
    b_u, b_d = _filter_pairs(b_u, b_d)

    plt.figure(figsize=(7, 5))
    if len(a_u) > 0:
        plt.scatter(a_u, a_d, s=18, alpha=0.7, label=a_summary["name"])
    if len(b_u) > 0:
        plt.scatter(b_u, b_d, s=18, alpha=0.7, label=b_summary["name"])

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Uncertainty / acquisition score (selected_scores)")
    plt.ylabel(r"Loss improvement $\Delta$loss = loss$_{t-1}$ - loss$_t$ (positive = better)")
    plt.title("Uncertainty vs. realized loss improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"[info] saved scatter plot to {output_png}")

def maybe_save_plot(
    a: Dict[str, Any],
    b: Dict[str, Any],
    output_png: Optional[str],
) -> None:
    if output_png is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skip plot: {e}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(a["costs"], a["losses"], label=a["name"])
    plt.plot(b["costs"], b["losses"], label="final_output_only")
    plt.xlabel("Cumulative cost")
    plt.ylabel("Test loss")
    plt.title("Method comparison: test loss vs cumulative cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"[info] saved plot to {output_png}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two experiment result .pt files."
    )
    parser.add_argument("--file_a", type=str, required=True)
    parser.add_argument("--file_b", type=str, required=True)
    parser.add_argument("--name_a", type=str, default="method_a")
    parser.add_argument("--name_b", type=str, default="method_b")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[5.0, 4.5, 4.0, 3.5, 3.0],
    )
    parser.add_argument(
        "--budget_points",
        type=float,
        nargs="*",
        default=[50.0, 100.0, 200.0],
    )
    parser.add_argument("--output_png", type=str, default=None)
    parser.add_argument("--output_scatter_png", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    result_a = _load_result(args.file_a)
    result_b = _load_result(args.file_b)

    summary_a = summarize_result(
        args.name_a,
        result_a,
        args.thresholds,
        args.budget_points,
    )
    summary_b = summarize_result(
        args.name_b,
        result_b,
        args.thresholds,
        args.budget_points,
    )

    print_summary(summary_a, args.budget_points)
    print_summary(summary_b, args.budget_points)
    print_comparison(summary_a, summary_b, args.budget_points)
    maybe_save_plot(summary_a, summary_b, args.output_png)
    maybe_save_scatter(
        result_a,
        result_b,
        summary_a,
        summary_b,
        args.output_scatter_png,
    )


if __name__ == "__main__":
    main()