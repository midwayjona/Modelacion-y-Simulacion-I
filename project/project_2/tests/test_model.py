# tests/test_model.py
import numpy as np
from core.model import generate_problem, evaluate, penalized_cost

def test_penalty_positive_when_violated():
    prob = generate_problem(N=10, workers=1, K=1, C=1000, seed=0)
    x_all = np.ones(prob.N, dtype=np.int8)  # likely infeasible
    ev = evaluate(prob, x_all)
    cost = penalized_cost(prob, x_all)
    assert cost > 0.0 or not ev["feasible_all"]

def test_feasibility_flags():
    prob = generate_problem(N=10, workers=10, K=10, C=10000, seed=1)
    x_all = np.ones(prob.N, dtype=np.int8)  # should be feasible
    ev = evaluate(prob, x_all)
    assert ev["feasible_all"] is True
    cost = penalized_cost(prob, x_all)
    assert cost <= 0.0  # no penalty, cost should be -Z <= 0
