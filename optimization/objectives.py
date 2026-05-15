
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

# Simple wrapper that turns a predictor into something CMA-ES can minimize.
# Both optimizers expect a callable that returns a float to minimize, so this
# handles the sign flip for maximization and the squared-error for target tracking.

@dataclass
class Objective:
    """Turn a predictor (higher is better) into a minimization objective if needed."""
    predictor: Callable[[np.ndarray], float]
    maximize: bool = True
    target: float | None = None  # if provided, aim to approach target instead of raw max

    def __call__(self, P: np.ndarray) -> float:
        val = self.predictor(P)
        if self.target is not None:
            # minimize squared error to target
            loss = (val - self.target) ** 2
            return float(loss)
        # default: maximize predictor -> minimize negative
        return float(-val if self.maximize else val)
