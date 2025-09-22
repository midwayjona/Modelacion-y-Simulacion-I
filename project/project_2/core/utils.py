# core/utils.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int | None = None) -> np.random.Generator:
    """Return a numpy Generator seeded if seed is not None."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_convergence(
    y_values: Iterable[float],
    title: str,
    xlabel: str = "Iteración",
    ylabel: str = "Costo (penalizado)",
    save_path: str | os.PathLike | None = None,
):
    """
    Plot a single-series convergence curve with matplotlib (no seaborn).
    """
    y = list(y_values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, ax


@dataclass
class GAResult:
    best_X: np.ndarray
    best_cost: float
    best_Z: int
    history_best_cost: List[float]
    history_mean_cost: List[float]
    trace: List[Dict[str, Any]]


@dataclass
class SAResult:
    best_X: np.ndarray
    best_cost: float
    best_Z: int
    history_best_cost: List[float]
    history_current_cost: List[float]


@dataclass
class TabuResult:
    best_X: np.ndarray
    best_cost: float
    best_Z: int
    history_best_cost: List[float]
