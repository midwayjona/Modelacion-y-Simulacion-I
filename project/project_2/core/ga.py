# core/ga.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from .model import Problem, penalized_cost, evaluate
from .utils import GAResult


def _tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """Pick index via tournament of size k."""
    n = fitness.shape[0]
    idxs = rng.integers(0, n, size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return int(best)


def _one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    if n <= 1:
        return a.copy(), b.copy()
    point = rng.integers(1, n)  # split after [0..point-1]
    c1 = np.concatenate([a[:point], b[point:]]).astype(np.int8, copy=False)
    c2 = np.concatenate([b[:point], a[point:]]).astype(np.int8, copy=False)
    return c1, c2


def _mutate(bits: np.ndarray, pm: float, rng: np.random.Generator) -> None:
    if pm <= 0.0:
        return
    flips = rng.random(bits.shape[0]) < pm
    bits[flips] ^= 1  # bit flip in-place


def run_ga(
    problem: Problem,
    params: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> GAResult:
    """
    Genetic Algorithm minimizing penalized cost by maximizing fitness = -cost.
    Default parameters:
      pop=80, gens=300, pc=0.8, pm=0.02, elite=1, tournament_k=3
    """
    if params is None:
        params = {}
    pop_size = int(params.get("pop", 80))
    gens = int(params.get("gens", 300))
    pc = float(params.get("pc", 0.8))
    pm = float(params.get("pm", 0.02))
    elite = int(params.get("elite", 1))
    tournament_k = int(params.get("tournament_k", 3))

    rng = np.random.default_rng(seed)

    # Initialize population
    pop = rng.integers(0, 2, size=(pop_size, problem.N), dtype=np.int8)
    # Optionally, include greedy baseline
    from .model import greedy_baseline
    pop[0] = greedy_baseline(problem)

    # Evaluate
    costs = np.array([penalized_cost(problem, ind) for ind in pop], dtype=float)
    fitness = -costs  # maximize
    best_idx = np.argmin(costs)
    best_X = pop[best_idx].copy()
    best_cost = float(costs[best_idx])
    best_Z = int(evaluate(problem, best_X)["Z"])

    history_best_cost: List[float] = [best_cost]
    history_mean_cost: List[float] = [float(np.mean(costs))]
    trace: List[Dict[str, Any]] = [{
        "generation": 0,
        "best_cost": best_cost,
        "best_Z": best_Z,
        "mean_cost": float(np.mean(costs)),
    }]

    for g in range(1, gens + 1):
        # Elitism
        elite_idx = np.argsort(costs)[:elite]
        elites = pop[elite_idx].copy()

        # Mating
        new_pop = []
        while len(new_pop) < pop_size - elite:
            i1 = _tournament_selection(fitness, tournament_k, rng)
            i2 = _tournament_selection(fitness, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pc:
                c1, c2 = _one_point_crossover(p1, p2, rng)
            _mutate(c1, pm, rng)
            _mutate(c2, pm, rng)
            new_pop.extend([c1, c2])
        if len(new_pop) > pop_size - elite:
            new_pop = new_pop[:pop_size - elite]

        pop = np.vstack([elites] + new_pop)

        # Evaluate new population
        costs = np.array([penalized_cost(problem, ind) for ind in pop], dtype=float)
        fitness = -costs

        # Track best
        idx = np.argmin(costs)
        if costs[idx] < best_cost:
            best_cost = float(costs[idx])
            best_X = pop[idx].copy()
            best_Z = int(evaluate(problem, best_X)["Z"])

        history_best_cost.append(best_cost)
        history_mean_cost.append(float(np.mean(costs)))
        trace.append({
            "generation": g,
            "best_cost": best_cost,
            "best_Z": best_Z,
            "mean_cost": float(np.mean(costs)),
        })

    return GAResult(
        best_X=best_X,
        best_cost=best_cost,
        best_Z=best_Z,
        history_best_cost=history_best_cost,
        history_mean_cost=history_mean_cost,
        trace=trace,
    )
