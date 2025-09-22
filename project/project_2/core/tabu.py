# core/tabu.py
from __future__ import annotations

from typing import Dict, Any, List
from collections import deque
import numpy as np

from .model import Problem, penalized_cost, evaluate


def run_tabu(
    problem: Problem,
    params: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> "TabuResult":
    """
    Tabu Search minimizing penalized cost.
    Default parameters:
      iters = N*300
      tabu_size = ceil(0.1*N)
      consider_all = True (consider flipping all bits each iteration)
    """
    if params is None:
        params = {}
    iters = int(params.get("iters", problem.N * 300))
    tabu_size = int(params.get("tabu_size", max(1, int(np.ceil(0.1 * problem.N)))))
    consider_all = bool(params.get("consider_all", True))

    from .utils import TabuResult

    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=problem.N, dtype=np.int8)
    cost = float(penalized_cost(problem, x))

    best_x = x.copy()
    best_cost = cost
    history_best_cost: List[float] = [best_cost]

    tabu_list: deque[int] = deque(maxlen=tabu_size)

    for _ in range(iters):
        if consider_all:
            idxs = np.arange(problem.N)
        else:
            # sample subset
            sample_size = min(problem.N, 50)
            idxs = rng.choice(problem.N, size=sample_size, replace=False)

        best_neighbor_cost = None
        best_neighbor_idx = None
        best_neighbor_x = None

        for idx in idxs:
            y = x.copy()
            y[idx] ^= 1

            # Tabu check
            is_tabu = idx in tabu_list
            new_cost = float(penalized_cost(problem, y))

            if (best_neighbor_cost is None) or (new_cost < best_neighbor_cost):
                # Aspiration: allow if improves global best even if tabu
                if is_tabu and new_cost >= best_cost:
                    continue
                best_neighbor_cost = new_cost
                best_neighbor_idx = int(idx)
                best_neighbor_x = y

        # If no admissible neighbor found (possible if all moves tabu & no aspiration), pick random move
        if best_neighbor_x is None:
            idx = int(rng.integers(0, problem.N))
            best_neighbor_x = x.copy()
            best_neighbor_x[idx] ^= 1
            best_neighbor_cost = float(penalized_cost(problem, best_neighbor_x))
            best_neighbor_idx = idx

        # Move
        x = best_neighbor_x
        cost = best_neighbor_cost
        tabu_list.append(best_neighbor_idx)

        # Update best
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

        history_best_cost.append(best_cost)

    best_Z = int(evaluate(problem, best_x)["Z"])
    return TabuResult(
        best_X=best_x,
        best_cost=best_cost,
        best_Z=best_Z,
        history_best_cost=history_best_cost,
    )
