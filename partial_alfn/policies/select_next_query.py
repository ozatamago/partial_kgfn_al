#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from botorch.logging import logger
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

from partial_alfn.data.update_buffers import append_partial_observation
from partial_alfn.metrics.evaluation import compute_test_loss
from partial_alfn.policies.candidates import make_candidates
from partial_alfn.policies.node_input_builder import (
    build_eval_input_for_node_group,
    sample_fantasy_observation_for_node_group,
)
from partial_alfn.training.train_factory import train_predictor_partial_backend
from partial_alfn.uncertainty.base import (
    predict_mean_var,
    predict_sink_mean_var,
)
from partial_alfn.uncertainty.mc_dropout import select_top_cost_aware_action
from partial_alfn.utils.effective_costs import effective_group_costs

def _default_node_groups(problem: SyntheticTestFunction) -> List[List[int]]:
    if hasattr(problem, "node_groups") and problem.node_groups is not None:
        return [list(g) for g in problem.node_groups]
    return [[j] for j in range(problem.n_nodes)]


def _normalize_node_groups(
    problem: SyntheticTestFunction,
    options: Dict,
) -> List[List[int]]:
    node_groups = options.get("node_groups", None)
    if node_groups is None:
        return _default_node_groups(problem)
    return [list(g) for g in node_groups]


def _normalize_group_index_set(
    *,
    key_indices: str,
    key_groups: str,
    options: Dict,
    node_groups: Sequence[Sequence[int]],
) -> List[int]:
    """
    Support either:
      - explicit group indices, e.g. upstream_group_indices=[0,1]
      - explicit group definitions, e.g. upstream_groups=[[0],[2,3]]
    """
    if key_indices in options and options[key_indices] is not None:
        return list(options[key_indices])

    if key_groups in options and options[key_groups] is not None:
        target_groups = [list(g) for g in options[key_groups]]
        out = []
        for tg in target_groups:
            try:
                out.append(node_groups.index(tg))
            except ValueError:
                pass
        return out

    return []


def _intersect_preserve_order(a: Sequence[int], b: Sequence[int]) -> List[int]:
    bset = set(b)
    return [x for x in a if x in bset]


def _masked_scores_for_group_subset(
    scores: torch.Tensor,
    active_group_indices: Sequence[int],
) -> torch.Tensor:
    masked = torch.full_like(scores, float("-inf"))
    if len(active_group_indices) > 0:
        masked[:, list(active_group_indices)] = scores[:, list(active_group_indices)]
    return masked


def _best_group_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Best score per group over candidate set.
    Returns shape [n_groups].
    """
    return scores.max(dim=0).values


def _make_uncertainty_options(
    options: Dict,
    mc_samples: int,
) -> Dict:
    return {
        "mc_samples": int(mc_samples),
        "n_samples": int(options.get("n_posterior_samples", mc_samples)),
        "unbiased": bool(options.get("unbiased_var", False)),
    }


def _clone_optional_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.clone()


def _clone_tensor_list(xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [x.clone() for x in xs]


def _clone_state_for_fantasy(state: Dict) -> Dict:
    return {
        "train_X": _clone_tensor_list(state["train_X"]),
        "train_Y": _clone_tensor_list(state["train_Y"]),
        "train_X_nn": _clone_optional_tensor(state.get("train_X_nn")),
        "train_y_nn": _clone_optional_tensor(state.get("train_y_nn")),
        "node_eval_counts": state["node_eval_counts"].clone(),
        "partial_buffers": None,
    }

def _group_variance_from_base_candidates(
    *,
    predictor: nn.Module,
    base_x: Tensor,
    node_group: Sequence[int],
    mc_samples: int,
    options: Dict,
) -> Tensor:
    """
    Compute a group-local uncertainty proxy without requiring all-node rollout.

    Returns
    -------
    group_var: torch.Tensor
        Shape [N]
    """
    uncertainty_options = _make_uncertainty_options(
        options=options,
        mc_samples=mc_samples,
    )

    eval_x = build_eval_input_for_node_group(
        predictor=predictor,
        base_x=base_x,
        node_group=node_group,
        mc_samples=mc_samples,
        options=options,
    )

    per_node_vars = []
    for node_idx in node_group:
        _, var_j = predict_mean_var(
            predictor=predictor,
            X=eval_x,
            node_idx=node_idx,
            options=uncertainty_options,
        )
        per_node_vars.append(var_j)

    var_group = torch.cat(per_node_vars, dim=-1)  # [N, |group|]
    reduction = str(options.get("group_var_reduction", "sum")).lower()

    if reduction == "sum":
        return var_group.sum(dim=-1)
    if reduction == "mean":
        return var_group.mean(dim=-1)
    if reduction == "max":
        return var_group.max(dim=-1).values

    raise ValueError(f"Unsupported group_var_reduction: {reduction}")


def _score_candidate_group_by_fantasy_gain(
    *,
    problem: SyntheticTestFunction,
    predictor: nn.Module,
    state: Dict,
    base_x: Tensor,
    node_group: Sequence[int],
    current_holdout_loss: float,
    selector_holdout_X: Tensor,
    selector_holdout_y: Tensor,
    options: Dict,
) -> float:
    mc_samples = int(options.get("mc_samples", 100))
    fantasy_train_steps = int(options.get("fantasy_train_steps", 20))
    task = options.get("task", "regression")

    fantasy_predictor = deepcopy(predictor)
    fantasy_state = _clone_state_for_fantasy(state)

    fantasy_eval_x, fantasy_y = sample_fantasy_observation_for_node_group(
        predictor=predictor,
        base_x=base_x,
        node_group=node_group,
        mc_samples=mc_samples,
        options=options,
    )

    append_partial_observation(
        problem=problem,
        base_x=base_x,
        eval_x=fantasy_eval_x,
        new_y=fantasy_y,
        new_node=list(node_group),
        state=fantasy_state,
        sink_node_idx=problem.n_nodes - 1,
    )

    fantasy_train_options = dict(options)
    fantasy_train_options["n_steps"] = fantasy_train_steps

    _ = train_predictor_partial_backend(
        predictor=fantasy_predictor,
        train_X_nodes=fantasy_state["train_X"],
        train_Y_nodes=fantasy_state["train_Y"],
        options=fantasy_train_options,
        sink_idx=problem.n_nodes - 1,
        optimizer=None,
    )

    fantasy_holdout_loss = compute_test_loss(
        predictor=fantasy_predictor,
        test_X=selector_holdout_X,
        test_y=selector_holdout_y,
        task=task,
    )

    return float(current_holdout_loss - fantasy_holdout_loss)


def get_suggested_node_and_input(
    algo: str,
    remaining_budget: float,
    problem: SyntheticTestFunction,
    predictor: nn.Module,
    options: Optional[Dict] = None,
    state: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor, Optional[List[int]], Optional[object], Optional[object]]:
    """
    Return the next action in the form of:

        (base_x, eval_x, new_node, acq_val, node_candidate)

    where
      - base_x: external input candidate in the original input space
      - eval_x: actual node-specific input passed to problem.evaluate(..., idx=new_node)
      - new_node: selected node-group; None means full-network evaluation

    Modes:
      - Random:
          random base_x, full-network evaluation with eval_x = base_x
      - NN_UQ with enable_partial_queries=False:
          highest sink variance base_x, full-network evaluation with eval_x = base_x
      - NN_UQ with enable_partial_queries=True:
          highest (group utility / group cost) over (candidate x, node_group),
          then build node-specific eval_x from base_x and the selected group
    """
    options = options or {}

    if algo == "Random":
        lb = problem.bounds[0]
        ub = problem.bounds[1]
        base_x = (
            torch.rand([1, problem.dim], dtype=torch.get_default_dtype()) * (ub - lb) + lb
        )
        eval_x = base_x
        return base_x, eval_x, None, None, None

    if algo != "NN_UQ":
        raise ValueError(f"Unsupported algo: {algo}")

    mc_samples = options.get("mc_samples", 100)
    n_sobol = options.get("cand_n_sobol", 256)
    enable_partial_queries = options.get("enable_partial_queries", False)
    debug_selector = options.get("debug_selector", True)

    Xcand = make_candidates(problem, n_sobol=n_sobol)

    # Full-evaluation fallback: choose x by sink uncertainty
    if not enable_partial_queries:
        uncertainty_options = _make_uncertainty_options(
            options=options,
            mc_samples=mc_samples,
        )

        _, pred_var = predict_sink_mean_var(
            predictor=predictor,
            X=Xcand,
            options=uncertainty_options,
        )

        idx = torch.argmax(pred_var.view(-1))
        base_x = Xcand[idx : idx + 1]
        eval_x = base_x

        if debug_selector:
            logger.info(
                f"[selector] mode=full_sink_uq max_sink_var={float(pred_var.max().item()):.6f}"
            )

        return base_x, eval_x, None, float(pred_var[idx].item()), {
            "mode": "full_sink_uq",
            "selected_candidate_idx": int(idx.item()),
        }

    # Partial-query mode
    node_groups = _normalize_node_groups(problem, options)
    var_reduction = options.get("group_var_reduction", "sum")
    use_upstream_first = options.get("use_upstream_first", True)
    tau = float(options.get("uncertainty_threshold_tau", 100.0))

    # Group-local uncertainty proxy used only for shortlist.
    # This avoids forcing an all-node rollout when only a subset of nodes is needed.
    group_var_cols = []
    for node_group in node_groups:
        try:
            group_var_j = _group_variance_from_base_candidates(
                predictor=predictor,
                base_x=Xcand,
                node_group=node_group,
                mc_samples=mc_samples,
                options=options,
            )  # [N]
        except RuntimeError:
            # Example: DKL node has no real training data yet.
            group_var_j = torch.full(
                (Xcand.shape[0],),
                float("-inf"),
                dtype=Xcand.dtype,
                device=Xcand.device,
            )

        group_var_cols.append(group_var_j.unsqueeze(-1))

    group_var = torch.cat(group_var_cols, dim=-1)  # [N, n_groups]

    group_effective_costs = effective_group_costs(problem, node_groups)

    cost_tensor = torch.tensor(
        group_effective_costs,
        dtype=group_var.dtype,
        device=group_var.device,
    ).view(1, -1)  # [1, n_groups]

    proxy_scores = group_var / cost_tensor

    affordable_group_indices = [
        i for i, c in enumerate(group_effective_costs) if c <= remaining_budget
    ]

    selector_holdout_X = options["selector_holdout_X"]
    selector_holdout_y = options["selector_holdout_y"]

    fantasy_topk_candidates = int(options.get("fantasy_topk_candidates", 8))
    fantasy_topk_groups = int(options.get("fantasy_topk_groups", 2))

    # Debug: group-wise maxima before stage masking
    group_max_uncertainty = group_var.max(dim=0).values   # [n_groups]
    group_max_proxy_score = proxy_scores.max(dim=0).values  # [n_groups]

    # If no partial group is affordable, fall back to full-network UQ
    if len(affordable_group_indices) == 0:
        _, pred_var = predict_sink_mean_var(
            predictor=predictor,
            X=Xcand,
            options=uncertainty_options,
        )

        idx = torch.argmax(pred_var.view(-1))
        base_x = Xcand[idx : idx + 1]
        eval_x = base_x

        if debug_selector:
            logger.info(
                f"[selector] no affordable partial group; "
                f"fallback full eval. max_sink_var={float(pred_var.max().item()):.6f}"
            )

        return base_x, eval_x, None, float(pred_var[idx].item()), {
            "mode": "fallback_full_no_affordable_group",
            "selected_candidate_idx": int(idx.item()),
        }

    upstream_group_indices = _normalize_group_index_set(
        key_indices="upstream_group_indices",
        key_groups="upstream_groups",
        options=options,
        node_groups=node_groups,
    )
    downstream_group_indices = _normalize_group_index_set(
        key_indices="downstream_group_indices",
        key_groups="downstream_groups",
        options=options,
        node_groups=node_groups,
    )

    selected_stage = "all"
    stage_uncertainty = None

    if use_upstream_first and len(upstream_group_indices) > 0:
        upstream_affordable = _intersect_preserve_order(
            upstream_group_indices,
            affordable_group_indices,
        )
        downstream_affordable = _intersect_preserve_order(
            downstream_group_indices,
            affordable_group_indices,
        )

        stage_uncertainty = float(group_var[:, upstream_group_indices].max().item())

        if stage_uncertainty > tau and len(upstream_affordable) > 0:
            active_group_indices = upstream_affordable
            selected_stage = "upstream"
        elif len(downstream_affordable) > 0:
            active_group_indices = downstream_affordable
            selected_stage = "downstream"
        elif len(upstream_affordable) > 0:
            active_group_indices = upstream_affordable
            selected_stage = "upstream_fallback"
        else:
            active_group_indices = affordable_group_indices
            selected_stage = "all_fallback"
    else:
        active_group_indices = affordable_group_indices
        selected_stage = "all"

    selector_objective = str(
        options.get("selector_objective", "fantasy_gain")
    ).lower()

    masked_proxy_scores = _masked_scores_for_group_subset(
        scores=proxy_scores,
        active_group_indices=active_group_indices,
    )

    current_holdout_loss = None

    if selector_objective == "uncertainty":
        masked_scores = masked_proxy_scores
        selection_mode = "partial_group_uncertainty"

    elif selector_objective == "fantasy_gain":
        if state is None:
            raise ValueError(
                "state must be provided when selector_objective uses fantasy gain."
            )

        current_holdout_loss = compute_test_loss(
            predictor=predictor,
            test_X=selector_holdout_X,
            test_y=selector_holdout_y,
            task=options.get("task", "regression"),
        )

        fantasy_scores = torch.full_like(masked_proxy_scores, float("-inf"))

        group_best_proxy_scores = _best_group_scores(masked_proxy_scores)
        finite_group_mask = torch.isfinite(group_best_proxy_scores)

        if finite_group_mask.any():
            finite_group_indices = torch.nonzero(
                finite_group_mask,
                as_tuple=False,
            ).view(-1)

            n_group_select = min(fantasy_topk_groups, finite_group_indices.numel())
            shortlisted_group_positions = torch.topk(
                group_best_proxy_scores[finite_group_indices],
                k=n_group_select,
            ).indices
            shortlisted_group_indices = finite_group_indices[
                shortlisted_group_positions
            ].tolist()

            for group_idx_candidate in shortlisted_group_indices:
                per_group_proxy_scores = masked_proxy_scores[:, group_idx_candidate]
                finite_candidate_mask = torch.isfinite(per_group_proxy_scores)

                if not finite_candidate_mask.any():
                    continue

                finite_candidate_indices = torch.nonzero(
                    finite_candidate_mask,
                    as_tuple=False,
                ).view(-1)

                n_candidate_select = min(
                    fantasy_topk_candidates,
                    finite_candidate_indices.numel(),
                )

                shortlisted_candidate_positions = torch.topk(
                    per_group_proxy_scores[finite_candidate_indices],
                    k=n_candidate_select,
                ).indices

                shortlisted_candidate_indices = finite_candidate_indices[
                    shortlisted_candidate_positions
                ].tolist()

                node_group_candidate = list(node_groups[group_idx_candidate])

                for cand_idx_candidate in shortlisted_candidate_indices:
                    base_x_candidate = Xcand[cand_idx_candidate : cand_idx_candidate + 1]

                    fantasy_gain = _score_candidate_group_by_fantasy_gain(
                        problem=problem,
                        predictor=predictor,
                        state=state,
                        base_x=base_x_candidate,
                        node_group=node_group_candidate,
                        current_holdout_loss=current_holdout_loss,
                        selector_holdout_X=selector_holdout_X,
                        selector_holdout_y=selector_holdout_y,
                        options=options,
                    )

                    fantasy_scores[cand_idx_candidate, group_idx_candidate] = (
                        fantasy_gain / cost_tensor[0, group_idx_candidate]
                    )

        if torch.isfinite(fantasy_scores).any():
            masked_scores = fantasy_scores
            selection_mode = "partial_group_fantasy_gain"
        else:
            masked_scores = masked_proxy_scores
            selection_mode = "partial_group_uq_fallback"

    else:
        raise ValueError(
            f"Unsupported selector_objective: {selector_objective}. "
            "Use 'uncertainty' or 'fantasy_gain'."
        )
    
    if not torch.isfinite(masked_scores).any():
        _, pred_var = predict_sink_mean_var(
            predictor=predictor,
            X=Xcand,
            options=uncertainty_options,
        )

        idx = torch.argmax(pred_var.view(-1))
        base_x = Xcand[idx : idx + 1]
        eval_x = base_x

        if debug_selector:
            logger.info(
                "[selector] all partial-group scores are invalid; "
                f"fallback full eval. max_sink_var={float(pred_var.max().item()):.6f}"
            )

        return base_x, eval_x, None, float(pred_var[idx].item()), {
            "mode": "fallback_full_all_partial_invalid",
            "selected_candidate_idx": int(idx.item()),
        }

    cand_idx, group_idx, best_score = select_top_cost_aware_action(masked_scores)
    base_x = Xcand[cand_idx : cand_idx + 1]
    new_node = list(node_groups[group_idx])

    eval_x = build_eval_input_for_node_group(
        predictor=predictor,
        base_x=base_x,
        node_group=new_node,
        mc_samples=mc_samples,
        options=options,
    )

    best_group_scores = _best_group_scores(masked_scores)

    if debug_selector:
        logger.info(
            f"[selector] tau={tau:.6f}, use_upstream_first={use_upstream_first}, "
            f"selected_stage={selected_stage}, stage_uncertainty={stage_uncertainty}"
        )
        logger.info(
            f"[selector] node_groups={node_groups}, "
            f"upstream_group_indices={upstream_group_indices}, "
            f"downstream_group_indices={downstream_group_indices}, "
            f"affordable_group_indices={affordable_group_indices}, "
            f"active_group_indices={list(active_group_indices)}"
        )
        for gi, group in enumerate(node_groups):
            logger.info(
                f"[selector] group_idx={gi}, nodes={group}, "
                f"effective_cost={group_effective_costs[gi]:.6f}, "
                f"max_uncertainty={float(group_max_uncertainty[gi].item()):.6f}, "
                f"max_proxy_score={float(group_max_proxy_score[gi].item()):.6f}"
            )
        logger.info(
            f"[selector] selection_mode={selection_mode}, "
            f"current_holdout_loss="
            f"{'N/A' if current_holdout_loss is None else f'{current_holdout_loss:.6f}'}, "
            f"selected_group_idx={group_idx}, selected_node_group={new_node}, "
            f"selected_candidate_idx={cand_idx}, selected_score={best_score:.6f}"
        )
        logger.info(
            f"[selector] selected_base_x={base_x}, selected_eval_x={eval_x}"
        )

    node_candidate = {
        "mode": "partial_group_fantasy_gain",
        "selection_mode": selection_mode,
        "selected_stage": selected_stage,
        "selected_candidate_idx": cand_idx,
        "selected_group_idx": group_idx,
        "selected_node_group": new_node,
        "selected_score": best_score,
        "current_holdout_loss": (
            None if current_holdout_loss is None else float(current_holdout_loss)
        ),
        "stage_uncertainty": stage_uncertainty,
        "active_group_indices": list(active_group_indices),
        "affordable_group_indices": list(affordable_group_indices),
        "node_groups": [list(g) for g in node_groups],
        "upstream_group_indices": list(upstream_group_indices),
        "downstream_group_indices": list(downstream_group_indices),
        "group_max_uncertainty": [float(x.item()) for x in group_max_uncertainty],
        "group_max_proxy_score": [float(x.item()) for x in group_max_proxy_score],
        "selected_base_x": base_x.detach().cpu(),
        "selected_eval_x": eval_x.detach().cpu(),
        "group_effective_costs": [float(c) for c in group_effective_costs],
    }

    return base_x, eval_x, new_node, best_group_scores, node_candidate