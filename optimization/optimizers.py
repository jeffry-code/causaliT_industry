
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
import numpy as np

@dataclass
class OptResult:
    x_best: np.ndarray
    f_best: float
    n_eval: int
    history: list[tuple[float, float]]  # (f, ||x||) log


def cma_es(objective: Callable[[np.ndarray], float],
           x0: np.ndarray,
           sigma0: float,
           bounds: tuple[np.ndarray, np.ndarray] | None = None,
           seed: int = 42,
           max_iters: int = 200,
           tol: float = 1e-8) -> OptResult:
    """
    Lightweight CMA-ES using 'cma' if available; else simple random-restart NES fallback.
    """
    try:
        import cma  # import here so the rest of the code works even without cma installed
        opts = {"verb_disp": 0, "seed": int(seed)}
        if bounds is not None:
            opts["bounds"] = [bounds[0].tolist(), bounds[1].tolist()]
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        history = []
        f_best = float("inf")
        x_best = x0.copy()
        n_eval = 0
        for _ in range(max_iters):
            xs = es.ask()  # sample a population of candidates
            fs = [objective(np.array(x, dtype=float)) for x in xs]
            n_eval += len(fs)
            es.tell(xs, fs)  # update the covariance matrix
            es.disp()
            idx = int(np.argmin(fs))
            f_iter = float(fs[idx])
            if f_iter < f_best:
                f_best = f_iter
                x_best = np.array(xs[idx], dtype=float)
            history.append((f_best, float(np.linalg.norm(x_best))))
            if es.stop():  # early stopping if CMA-ES internal criteria are met
                break
        return OptResult(x_best=x_best, f_best=f_best, n_eval=n_eval, history=history)
    except Exception:
        # fallback if cma is not installed — simple NES-like evolution strategy
        rng = np.random.default_rng(int(seed))
        mu = x0.copy()
        sigma = sigma0
        f_best = objective(mu)
        x_best = mu.copy()
        history = [(f_best, float(np.linalg.norm(x_best)))]
        n_eval = 1
        for _ in range(max_iters):
            Z = rng.standard_normal((32, mu.size))
            xs = mu + sigma * Z
            if bounds is not None:
                lb, ub = bounds
                xs = np.clip(xs, lb, ub)
            fs = np.array([objective(x) for x in xs])
            n_eval += len(fs)
            idx = int(np.argmin(fs))
            if fs[idx] < f_best:
                f_best = float(fs[idx])
                x_best = xs[idx].copy()
            # NES update
            scores = (fs - fs.mean()) / (fs.std() + 1e-8)
            mu = (mu + (Z.T @ (-scores)) * (sigma / xs.shape[0]))
            sigma *= 0.99
            history.append((f_best, float(np.linalg.norm(x_best))))
            if np.abs(history[-1][0] - history[-2][0]) < tol:
                break
        return OptResult(x_best=x_best, f_best=f_best, n_eval=n_eval, history=history)


def adam(objective: Callable[[np.ndarray], float],
         x0: np.ndarray,
         bounds: tuple[np.ndarray, np.ndarray] | None = None,
         lr: float = 0.05,
         iters: int = 500,
         eps: float = 1e-8,
         beta1: float = 0.9,
         beta2: float = 0.999) -> OptResult:
    """
    Gradient-free Adam using finite differences (central) for small dimensions.
    If you have a differentiable torch model, prefer a true autograd loop.
    """
    # central finite difference — works for black-box objectives but gets expensive
    # for high dimensions since it needs 2*d evaluations per gradient estimate
    def grad_fd(x: np.ndarray, h: float = 1e-4) -> np.ndarray:
        g = np.zeros_like(x)
        f0 = objective(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (objective(xp) - objective(xm)) / (2*h)
        return g

    x = x0.copy().astype(float)
    m = np.zeros_like(x)  # first moment
    v = np.zeros_like(x)  # second moment
    f_best = objective(x)
    x_best = x.copy()
    history = [(f_best, float(np.linalg.norm(x_best)))]
    for t in range(1, iters+1):
        g = grad_fd(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g*g)
        m_hat = m/(1-beta1**t)  # bias correction
        v_hat = v/(1-beta2**t)
        x = x - lr*m_hat/(np.sqrt(v_hat)+eps)
        if bounds is not None:
            lb, ub = bounds
            x = np.clip(x, lb, ub)  # project back into feasible box after each step
        f = objective(x)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        history.append((f_best, float(np.linalg.norm(x_best))))
    return OptResult(x_best=x_best, f_best=f_best, n_eval=len(history), history=history)
