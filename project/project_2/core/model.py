# core/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class Problem:
    """
    Binary knapsack-style harvest planning problem under two constraints:
    - Total working hours (W = workers * 40.0)
    - Total transport capacity (K * C)
    Objective: maximize harvested kilograms Z = sum(V_i * X_i).

    For metaheuristics, we minimize a penalized cost:
      penalized_cost(X) = -Z(X) + lambda_h * excess_hours + lambda_c * excess_capacity
    where excess terms are positive parts of the constraint violations.
    """
    V: np.ndarray  # (N,) int - kilograms available per plot
    P: np.ndarray  # (N,) float - hours required per plot
    W: float       # total available hours
    K: int         # number of vehicles
    C: int         # capacity per vehicle (kg)
    N: int         # number of plots
    lambda_h: float = 1e5
    lambda_c: float = 1e5

    def __post_init__(self) -> None:
        # Basic validations
        assert self.V.shape == (self.N,), "V must have shape (N,)"
        assert self.P.shape == (self.N,), "P must have shape (N,)"
        assert np.issubdtype(self.V.dtype, np.integer), "V must be integer array"
        assert np.issubdtype(self.P.dtype, np.floating), "P must be float array"
        assert self.W > 0, "W must be positive"
        assert self.K >= 1 and self.C >= 1, "K and C must be >= 1"

    @property
    def capacity_total(self) -> int:
        return int(self.K * self.C)


def generate_problem(
    N: int = 80,
    workers: int = 10,
    K: int = 3,
    C: int = 1800,
    seed: int = 42,
) -> Problem:
    """
    Deterministic & reproducible data generator.
    - V_i ~ Uniform integer [100, 1200]
    - P_i ~ Uniform float [1.0, 10.0], rounded to 2 decimals
    - W = workers * 40.0

    Returns Problem dataclass.
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    if not (1000 <= C <= 2500):
        # Allow broader values but warn users; enforce minimum bound gracefully
        # Not raising to keep the app robust if user experiments.
        pass
    if workers < 1 or K < 1:
        raise ValueError("workers and K must be >= 1")

    rng = np.random.default_rng(seed)
    V = rng.integers(low=100, high=1201, size=N, dtype=np.int64)  # high is exclusive
    P = rng.uniform(1.0, 10.0, size=N).round(2).astype(np.float64)
    W = float(workers) * 40.0

    # Large penalties to strongly discourage infeasible solutions.
    problem = Problem(
        V=V, P=P, W=W, K=int(K), C=int(C), N=int(N),
        lambda_h=1e5, lambda_c=1e5
    )
    return problem


def evaluate(problem: Problem, X: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate a binary solution X for the given problem.
    Returns:
      {
        'Z': int, 'hours_used': float, 'kg_used': int,
        'feasible_hours': bool, 'feasible_capacity': bool, 'feasible_all': bool
      }
    """
    x = np.asarray(X, dtype=np.int8)
    if x.shape != (problem.N,):
        raise ValueError(f"X must have shape ({problem.N},), got {x.shape}")
    if np.any((x != 0) & (x != 1)):
        raise ValueError("X must be binary (0/1)")

    kg_used = int(np.dot(problem.V, x))
    hours_used = float(np.dot(problem.P, x))
    Z = kg_used  # by definition

    feasible_hours = hours_used <= problem.W + 1e-9
    feasible_capacity = kg_used <= problem.capacity_total + 1e-9
    feasible_all = bool(feasible_hours and feasible_capacity)

    return {
        "Z": int(Z),
        "hours_used": float(hours_used),
        "kg_used": int(kg_used),
        "feasible_hours": bool(feasible_hours),
        "feasible_capacity": bool(feasible_capacity),
        "feasible_all": bool(feasible_all),
    }


def penalized_cost(problem: Problem, X: np.ndarray) -> float:
    """
    Penalized cost to minimize.
      cost = -Z + lambda_h * max(0, hours - W) + lambda_c * max(0, kg - K*C)
    """
    x = np.asarray(X, dtype=np.int8)
    kg_used = float(np.dot(problem.V, x))
    hours_used = float(np.dot(problem.P, x))

    excess_h = max(0.0, hours_used - problem.W)
    excess_c = max(0.0, kg_used - problem.capacity_total)
    penalty = problem.lambda_h * excess_h + problem.lambda_c * excess_c
    Z = kg_used
    return float(-Z + penalty)


def random_solution(problem: Problem, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random binary solution of length N."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=problem.N, dtype=np.int8)


def greedy_baseline(problem: Problem) -> np.ndarray:
    """
    Greedy baseline by ratio Vi / Pi, respecting both constraints.
    Items are sorted by decreasing V/P and added if they keep feasibility.
    """
    ratio = problem.V / np.maximum(problem.P, 1e-9)
    order = np.argsort(-ratio)  # descending
    x = np.zeros(problem.N, dtype=np.int8)
    total_hours = 0.0
    total_kg = 0

    for idx in order:
        new_hours = total_hours + problem.P[idx]
        new_kg = total_kg + int(problem.V[idx])
        if new_hours <= problem.W + 1e-12 and new_kg <= problem.capacity_total:
            x[idx] = 1
            total_hours = new_hours
            total_kg = new_kg
    return x


# Note on extension (not implemented):
# To extend to multiple knapsack or multiple trips, replace X_i by assignment variables
# x_{i,k} in {0,1} that assign plot i to vehicle k, with capacity per vehicle constraints,
# and optionally multiple-trip decision variables. This increases complexity substantially,
# moving from a single binary knapsack with a global capacity to a multiple knapsack model.
