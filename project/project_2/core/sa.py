# core/sa.py
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from .model import Problem, penalized_cost, evaluate
from .utils import SAResult


def _neighbor_flip(x: np.ndarray, idx: int) -> np.ndarray:
    y = x.copy()
    y[idx] ^= 1
    return y


def run_sa(
    problem: Problem,
    params: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> SAResult:
    """
    Simulated Annealing minimizing penalized cost.
    Default parameters: T0=1.0, Tend=1e-3, iters=N*300, alpha=0.99
    """
    if params is None:
        params = {}
    T0 = float(params.get("T0", 1.0))
    Tend = float(params.get("Tend", 1e-3))
    iters = int(params.get("iters", problem.N * 300))
    alpha = float(params.get("alpha", 0.99))

    rng = np.random.default_rng(seed)

    # Initial state
    x = rng.integers(0, 2, size=problem.N, dtype=np.int8)
    cost = float(penalized_cost(problem, x))
    best_x = x.copy()
    best_cost = cost

    history_best_cost: List[float] = [best_cost]
    history_current_cost: List[float] = [cost]

    T = T0
    for t in range(1, iters + 1):
        idx = int(rng.integers(0, problem.N))
        y = _neighbor_flip(x, idx)
        new_cost = float(penalized_cost(problem, y))
        delta = new_cost - cost
        accept = False
        if delta <= 0.0:
            accept = True
        else:
            # Metropolis criterion
            if T > 0.0:
                p = np.exp(-delta / T)
                accept = rng.random() < p

        if accept:
            x = y
            cost = new_cost

        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()

        history_current_cost.append(cost)
        history_best_cost.append(best_cost)

        # Geometric cooling
        T = max(T * alpha, Tend)

    best_Z = int(evaluate(problem, best_x)["Z"])
    return SAResult(
        best_X=best_x,
        best_cost=best_cost,
        best_Z=best_Z,
        history_best_cost=history_best_cost,
        history_current_cost=history_current_cost,
    )
