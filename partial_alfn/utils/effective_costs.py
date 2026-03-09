#!/usr/bin/env python3

from typing import List, Sequence, Set


def _get_parents(problem, node_idx: int) -> List[int]:
    """
    Return parent node indices for one node.
    Assumes problem.parent_nodes[node_idx] is iterable.
    """
    if not hasattr(problem, "parent_nodes"):
        return []
    parents = problem.parent_nodes[node_idx]
    if parents is None:
        return []
    return list(parents)


def ancestor_closure(problem, node_group: Sequence[int]) -> List[int]:
    """
    Return the unique set of nodes required to evaluate `node_group`,
    including all ancestors and the nodes themselves.
    """
    visited: Set[int] = set()
    stack = list(node_group)

    while stack:
        j = stack.pop()
        if j in visited:
            continue
        visited.add(j)
        for p in _get_parents(problem, j):
            if p not in visited:
                stack.append(p)

    return sorted(visited)


def effective_group_cost(problem, node_group: Sequence[int]) -> float:
    """
    Sum of costs over the ancestor closure of node_group.
    """
    closure = ancestor_closure(problem, node_group)
    return float(sum(problem.node_costs[j] for j in closure))


def effective_group_costs(problem, node_groups: Sequence[Sequence[int]]) -> List[float]:
    """
    Effective cost for each node group.
    """
    return [effective_group_cost(problem, g) for g in node_groups]