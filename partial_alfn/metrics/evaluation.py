#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from partial_alfn.utils.construct_obs_set import construct_obs_set


def compute_test_loss(
    predictor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    task: str = "regression",
) -> float:
    """
    Sink output の test loss.
    - regression: MSE
    - classification: CrossEntropy
    """
    predictor.eval()
    with torch.no_grad():
        out = predictor(test_X)
        if task == "classification":
            loss = F.cross_entropy(out, test_y.long())
        else:
            loss = F.mse_loss(out.view_as(test_y), test_y)
    return float(loss.item())


def build_node_test_sets(
    *,
    problem,
    base_X: torch.Tensor,
    full_Y: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    外部入力 base_X と full network 出力 full_Y から、
    各 node 用の teacher-forced test set を構成する。

    Returns
    -------
    node_test_X : list[Tensor]
        node_test_X[j] has shape [N, d_j]
    node_test_Y : list[Tensor]
        node_test_Y[j] has shape [N, 1]
    full_Y : Tensor
        full network outputs [N, n_nodes]
    """
    if full_Y is None:
        with torch.no_grad():
            full_Y = problem.evaluate(base_X)

    node_test_X, node_test_Y = construct_obs_set(
        X=base_X,
        Y=full_Y,
        parent_nodes=problem.parent_nodes,
        active_input_indices=problem.active_input_indices,
    )
    return node_test_X, node_test_Y, full_Y


def compute_teacher_forced_node_losses(
    *,
    predictor,
    node_test_X: Sequence[torch.Tensor],
    node_test_Y: Sequence[torch.Tensor],
    task: str = "regression",
) -> Dict[int, float]:
    """
    各 node の teacher-forced loss:
      真の node 入力 x_j を与えて z_j をどれだけ正しく予測できるか
    """
    if len(node_test_X) != len(node_test_Y):
        raise ValueError(
            f"Length mismatch: len(node_test_X)={len(node_test_X)} vs "
            f"len(node_test_Y)={len(node_test_Y)}"
        )

    predictor.eval()
    out: Dict[int, float] = {}

    with torch.no_grad():
        for j, (Xj, Yj) in enumerate(zip(node_test_X, node_test_Y)):
            if Xj.shape[0] == 0:
                out[j] = float("nan")
                continue

            pred_j = predictor.forward_node(Xj, j)

            if task == "classification":
                loss_j = F.cross_entropy(pred_j, Yj.long())
            else:
                loss_j = F.mse_loss(pred_j.view_as(Yj), Yj)

            out[j] = float(loss_j.item())

    return out


def compute_rollout_node_losses(
    *,
    predictor,
    base_X: torch.Tensor,
    full_Y: torch.Tensor,
    task: str = "regression",
) -> Dict[int, float]:
    """
    各 node の rollout loss:
      外部入力 base_X から predictor を順に rollout して得た各 node 出力を、
      真の full_Y[:, j] と比較する
    """
    predictor.eval()

    if hasattr(predictor, "rollout_means_from_base"):
        with torch.no_grad():
            pred_all = predictor.rollout_means_from_base(base_X)
    elif hasattr(predictor, "forward_all"):
        with torch.no_grad():
            pred_all = predictor.forward_all(base_X)
    else:
        raise ValueError(
            "predictor must implement rollout_means_from_base(base_X) or forward_all(base_X)"
        )

    if pred_all.shape != full_Y.shape:
        raise ValueError(
            f"Shape mismatch: pred_all shape={tuple(pred_all.shape)} vs "
            f"full_Y shape={tuple(full_Y.shape)}"
        )

    out: Dict[int, float] = {}
    for j in range(full_Y.shape[1]):
        pred_j = pred_all[:, [j]]
        true_j = full_Y[:, [j]]

        if task == "classification":
            loss_j = F.cross_entropy(pred_j, true_j.long())
        else:
            loss_j = F.mse_loss(pred_j.view_as(true_j), true_j)

        out[j] = float(loss_j.item())

    return out


def compute_weighted_node_loss(
    node_losses: Dict[int, float],
    *,
    weights: Optional[Dict[int, float]] = None,
    exclude_nodes: Optional[Sequence[int]] = None,
) -> float:
    """
    node-wise loss を 1 個の scalar にまとめる補助関数。
    デフォルトは均等重み。
    """
    exclude = set(exclude_nodes or [])
    valid = {k: v for k, v in node_losses.items() if k not in exclude and not torch.isnan(torch.tensor(v))}

    if len(valid) == 0:
        return float("nan")

    if weights is None:
        ws = {k: 1.0 for k in valid.keys()}
    else:
        ws = {k: float(weights.get(k, 0.0)) for k in valid.keys()}

    denom = sum(ws.values())
    if denom <= 0:
        return float("nan")

    return sum(ws[k] * valid[k] for k in valid.keys()) / denom


def init_node_metric_history(metric_dict: Dict[int, float]) -> Dict[int, List[float]]:
    """
    {node_idx: metric} -> {node_idx: [metric]}
    """
    return {int(k): [float(v)] for k, v in metric_dict.items()}


def append_node_metric_history(
    history: Dict[int, List[float]],
    metric_dict: Dict[int, float],
) -> Dict[int, List[float]]:
    """
    履歴 dict に 1 step 分を append する。
    """
    for k, v in metric_dict.items():
        kk = int(k)
        history.setdefault(kk, [])
        history[kk].append(float(v))
    return history