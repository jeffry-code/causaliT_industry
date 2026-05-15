#!/usr/bin/env python
"""
Token-level optimization experiment for Dyconex sample 236.

Instead of assigning one scalar per source variable ID (constant across all
occurrences), this script assigns one scalar per non-NaN token position in the
source sequence. This lifts the constant-per-variable assumption and allows
each occurrence of a variable to be controlled independently.

The dimensionality equals the number of real (non-NaN) tokens in the sample,
which is 45 for sample 236 (vs. 25 unique variable IDs in the original setup).

Simplified setup compared to the full industrial experiment:
  - Single sample (--sample-index)
  - J_mean objective only
  - Both targets A and B
  - 3 seeds, 1000 iterations
  - Both CMA-ES and Adam

Arguments:
  --experiment-dir   path to the experiment folder (contains config.yaml and k_* checkpoints)
  --data-dir         path to the dataset folder
  --kfold            which fold checkpoint to use as surrogate (default: k_3)
  --sample-index     test sample index to optimize (default: 236)
  --targets          which targets to optimize: A (delta_A), B (delta_B), or both (default: A B)
  --seeds            random seeds for initialization (default: 0 1 2)
  --iters            number of iterations for both optimizers (default: 1000)
  --sigma0           CMA-ES initial step size (default: 0.3)
  --lr               Adam learning rate (default: 0.01)
  --out-dir          directory to save results
  --device           compute device, e.g. cpu or cuda (default: auto-detect)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from optimization.objectives import Objective
from optimization.optimizers import adam, cma_es
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster


def _find_config(experiment_dir: Path) -> Path:
    candidates = sorted(experiment_dir.glob("config*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No config*.yaml found in {experiment_dir}")
    return candidates[0]


def _find_checkpoint(checkpoints_dir: Path) -> Path:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoints_dir}")
    # prefer the explicitly saved best checkpoint if it exists
    best = checkpoints_dir / "best_checkpoint.ckpt"
    if best.exists():
        return best
    ckpts = list(checkpoints_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in {checkpoints_dir}")
    # fall back to the highest epoch checkpoint
    epoch_pat = re.compile(r"epoch=(\d+)")
    parsed: List[Tuple[int, Path]] = []
    for ck in ckpts:
        m = epoch_pat.search(ck.name)
        if m:
            parsed.append((int(m.group(1)), ck))
    if parsed:
        parsed.sort(key=lambda t: t[0])
        return parsed[-1][1]
    return sorted(ckpts)[0]


def _load_source_vocab(data_dir: Path) -> Dict[int, str]:
    # maps variable index -> human-readable name (e.g. 36 -> "mic_36")
    p = data_dir / "variables_vocabulary.json_source"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for name, idx in raw.items():
        try:
            out[int(idx)] = str(name)
        except Exception:
            continue
    return out


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _history_to_rows(history: List[Tuple[float, float]]) -> List[dict]:
    return [
        {"iter": int(i), "best_objective_so_far": float(h[0]), "x_norm_best_so_far": float(h[1])}
        for i, h in enumerate(history)
    ]


@dataclass
class TargetResult:
    target: str
    optimizer: str
    objective_kind: str
    best_objective: float
    best_prediction_score: float
    baseline_prediction_score: float
    improvement_abs: float
    improvement_rel_pct: float
    n_tokens_optimized: int
    sample_index: Optional[int]
    # token_pos -> {var_id, var_name, value}
    best_controls_tokens: List[dict]


class TokenLevelRunner:
    def __init__(
        self,
        experiment_dir: Path,
        data_dir: Path,
        kfold: str,
        sample_index: int,
        device: Optional[str] = None,
    ):
        self.experiment_dir = experiment_dir
        self.data_dir = data_dir
        self.kfold = kfold
        self.sample_index = sample_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config_path = _find_config(self.experiment_dir)
        self.config = OmegaConf.load(config_path)
        ckpt_path = _find_checkpoint(self.experiment_dir / self.kfold / "checkpoints")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt["hyper_parameters"]
        ckpt_cfg["model"]["kwargs"]["device"] = str(self.device)
        self.model = StageCausalForecaster(config=ckpt_cfg, data_dir=str(self.data_dir.parent))
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()

        if str(self.config["model"]["model_object"]) != "StageCausaliT":
            raise ValueError("This script expects model_object: StageCausaliT.")

        # Feature indices for source sequence
        self.val_idx_s = int(self.config["data"]["features"]["S"]["value"])
        self.var_idx_s = int(self.config["data"]["features"]["S"]["variable"])

        # Load test data
        ds_path = self.data_dir / str(self.config["data"]["test_file"])
        ds = np.load(ds_path)
        S_all = torch.tensor(ds["s"], dtype=torch.float32, device=self.device)
        X_all = torch.tensor(ds["x"], dtype=torch.float32, device=self.device)
        Y_all = torch.tensor(ds["y"], dtype=torch.float32, device=self.device)

        n = S_all.shape[0]
        if sample_index < 0 or sample_index >= n:
            raise ValueError(f"sample_index={sample_index} out of range (n={n})")

        # Single sample tensors: shape (1, T, F)
        self.S = S_all[sample_index : sample_index + 1]
        self.X = X_all[sample_index : sample_index + 1]
        self.Y = Y_all[sample_index : sample_index + 1]

        self.source_vocab = _load_source_vocab(self.data_dir)

        # find which token positions actually have a value (non-NaN) for this sample
        # NaN positions are padding — we only want to optimize the real tokens
        val_col = self.S[0, :, self.val_idx_s]  # (T,)
        self.token_positions = torch.where(~torch.isnan(val_col))[0].tolist()  # list of int
        self.n_tokens = len(self.token_positions)

        # store variable names and original values for output / sensitivity reports
        var_col = self.S[0, :, self.var_idx_s]
        self.token_var_ids = [int(var_col[pos].item()) for pos in self.token_positions]
        self.token_var_names = [
            self.source_vocab.get(vid, f"var_{vid}") for vid in self.token_var_ids
        ]
        self.token_original_values = [
            float(val_col[pos].item()) for pos in self.token_positions
        ]

        print(
            f"Sample {sample_index}: {self.n_tokens} real tokens "
            f"({len(set(self.token_var_ids))} unique var_ids)"
        )

    def _apply_token_controls(self, S: torch.Tensor, controls_t: torch.Tensor) -> torch.Tensor:
        """
        Apply per-token controls: set S[0, token_pos, val_idx_s] = controls_t[i]
        for each (i, token_pos) in enumerate(self.token_positions).
        Broadcasts across batch dimension (all samples in S get same token values).
        """
        S_mod = S.clone()  # don't modify the original
        for i, pos in enumerate(self.token_positions):
            # only the value feature is replaced — group, process, variable ID etc. stay the same
            S_mod[:, pos, self.val_idx_s] = controls_t[i]
        return S_mod

    @staticmethod
    def _target_slice(target: str) -> slice:
        if target == "A":
            return slice(0, 200)
        if target == "B":
            return slice(200, 400)
        raise ValueError(f"Unknown target {target}")

    def score_tensor(self, controls_t: Optional[torch.Tensor], target: str) -> torch.Tensor:
        sl = self._target_slice(target)
        # if no controls are passed, use the original S (baseline evaluation)
        S_ctrl = self._apply_token_controls(self.S, controls_t) if controls_t is not None else self.S
        _, pred_y, *_ = self.model.forward(
            data_source=S_ctrl,
            data_intermediate=self.X,
            data_target=self.Y,
        )
        # J_mean: average predicted value over the target window (200 time steps)
        return pred_y[:, sl, 0].mean()

    def score_numpy(self, controls: Optional[np.ndarray], target: str) -> float:
        controls_t = (
            torch.tensor(controls, dtype=torch.float32, device=self.device)
            if controls is not None
            else None
        )
        with torch.no_grad():
            return float(self.score_tensor(controls_t, target).item())

    def optimize_adam(
        self,
        x0: np.ndarray,
        target: str,
        bounds: Tuple[np.ndarray, np.ndarray],
        iters: int,
        lr: float,
    ):
        lb_t = torch.tensor(bounds[0], dtype=torch.float32, device=self.device)
        ub_t = torch.tensor(bounds[1], dtype=torch.float32, device=self.device)
        # treat the control vector as a trainable parameter so gradients flow back through the model
        p = torch.tensor(x0, dtype=torch.float32, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([p], lr=float(lr))

        best_obj = float("inf")
        best_x = x0.copy()
        history: List[Tuple[float, float]] = []

        for _ in range(int(iters)):
            opt.zero_grad()
            loss = self.score_tensor(p, target)  # minimize mean Y
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(lb_t, ub_t)  # project back into [0,1] after each gradient step
                obj = float(loss.detach().cpu().item())
                if obj < best_obj:
                    best_obj = obj
                    best_x = p.detach().cpu().numpy().astype(float).copy()
                history.append((best_obj, float(np.linalg.norm(best_x))))

        class _Res:
            pass

        res = _Res()
        res.x_best = best_x
        res.f_best = best_obj
        res.history = history
        return res

    def local_sensitivity(
        self, x_star: np.ndarray, target: str, delta: float = 0.01
    ) -> List[dict]:
        # central finite-difference sensitivity at the optimum
        # tells us which input tokens the objective is most sensitive to
        y0 = self.score_numpy(x_star, target)
        rows = []
        for i, pos in enumerate(self.token_positions):
            xp = x_star.copy()
            xm = x_star.copy()
            xp[i] = min(1.0, xp[i] + delta)  # clamp so we don't go outside [0,1]
            xm[i] = max(0.0, xm[i] - delta)
            yp = self.score_numpy(xp, target)
            ym = self.score_numpy(xm, target)
            rows.append(
                {
                    "target": target,
                    "token_pos": int(pos),
                    "var_id": self.token_var_ids[i],
                    "var_name": self.token_var_names[i],
                    "original_value": self.token_original_values[i],
                    "optimal_value": float(x_star[i]),
                    "score_base": float(y0),
                    "score_plus": float(yp),
                    "score_minus": float(ym),
                    "grad_fd": float((yp - ym) / (2.0 * delta)),
                    "abs_delta_mean": float(0.5 * (abs(yp - y0) + abs(ym - y0))),
                }
            )
        return rows


def run_seed(
    runner: TokenLevelRunner,
    seed: int,
    targets: List[str],
    iters: int,
    sigma0: float,
    lr: float,
    out_dir: Path,
) -> None:
    dim = runner.n_tokens  # 45 for sample 236
    lb = np.zeros(dim, dtype=float)
    ub = np.ones(dim, dtype=float)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lb, ub)  # random starting point in [0,1]^45

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f"\n  [seed={seed}, target={target}]")
        target_dir = seed_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        baseline = runner.score_numpy(None, target)

        # --- CMA-ES ---
        objective = Objective(
            predictor=lambda p, t=target: runner.score_numpy(p, t),
            maximize=False,
            target=None,
        )
        res_cma = cma_es(objective, x0=x0, sigma0=sigma0, bounds=(lb, ub), seed=seed, max_iters=iters)
        y_cma = runner.score_numpy(res_cma.x_best, target)

        # --- Adam ---
        res_adam = runner.optimize_adam(x0=x0, target=target, bounds=(lb, ub), iters=iters, lr=lr)
        y_adam = runner.score_numpy(res_adam.x_best, target)

        # keep the better of the two — best-of-two strategy per seed
        if res_cma.f_best <= res_adam.f_best:
            winner, x_star, y_star = "cma", res_cma.x_best, y_cma
        else:
            winner, x_star, y_star = "adam", res_adam.x_best, y_adam

        imp_abs = float(baseline - y_star)
        imp_rel = float(100.0 * imp_abs / (abs(baseline) + 1e-12))

        print(f"    baseline={baseline:.6f}, cma={y_cma:.6f}, adam={y_adam:.6f}, winner={winner}")

        sens_rows = runner.local_sensitivity(x_star, target)

        # Build readable controls list
        controls_tokens = [
            {
                "token_pos": int(runner.token_positions[i]),
                "var_id": runner.token_var_ids[i],
                "var_name": runner.token_var_names[i],
                "original_value": runner.token_original_values[i],
                "optimal_value": float(x_star[i]),
            }
            for i in range(runner.n_tokens)
        ]

        def _save(opt_name: str, res, y_val: float, x_vec: np.ndarray) -> None:
            imp_a = float(baseline - y_val)
            imp_r = float(100.0 * imp_a / (abs(baseline) + 1e-12))
            ctrl_list = [
                {
                    "token_pos": int(runner.token_positions[i]),
                    "var_id": runner.token_var_ids[i],
                    "var_name": runner.token_var_names[i],
                    "original_value": runner.token_original_values[i],
                    "optimal_value": float(x_vec[i]),
                }
                for i in range(runner.n_tokens)
            ]
            result = TargetResult(
                target=target,
                optimizer=opt_name,
                objective_kind="mean",
                best_objective=float(res.f_best),
                best_prediction_score=float(y_val),
                baseline_prediction_score=float(baseline),
                improvement_abs=imp_a,
                improvement_rel_pct=imp_r,
                n_tokens_optimized=runner.n_tokens,
                sample_index=runner.sample_index,
                best_controls_tokens=ctrl_list,
            )
            (target_dir / f"summary_{opt_name}.json").write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            _write_csv(target_dir / f"history_{opt_name}.csv", _history_to_rows(getattr(res, "history", [])))

        _save("cma", res_cma, y_cma, res_cma.x_best)
        _save("adam", res_adam, y_adam, res_adam.x_best)

        winner_info = {
            "winner": winner,
            "cma_objective": float(res_cma.f_best),
            "adam_objective": float(res_adam.f_best),
            "cma_score": float(y_cma),
            "adam_score": float(y_adam),
            "baseline": float(baseline),
        }
        (target_dir / "winner.json").write_text(json.dumps(winner_info, indent=2), encoding="utf-8")
        _write_csv(target_dir / "sensitivity_winner.csv", sens_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Token-level optimization for Dyconex.")
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--kfold", type=str, default="k_3")
    ap.add_argument("--sample-index", type=int, default=236)
    ap.add_argument("--targets", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    runner = TokenLevelRunner(
        experiment_dir=args.experiment_dir,
        data_dir=args.data_dir,
        kfold=args.kfold,
        sample_index=args.sample_index,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "setup.json").write_text(
        json.dumps(
            {
                "sample_index": args.sample_index,
                "n_tokens": runner.n_tokens,
                "token_positions": runner.token_positions,
                "token_var_ids": runner.token_var_ids,
                "token_var_names": runner.token_var_names,
                "token_original_values": runner.token_original_values,
                "iters": args.iters,
                "sigma0": args.sigma0,
                "lr": args.lr,
                "seeds": args.seeds,
                "kfold": args.kfold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        run_seed(
            runner=runner,
            seed=seed,
            targets=args.targets,
            iters=args.iters,
            sigma0=args.sigma0,
            lr=args.lr,
            out_dir=args.out_dir,
        )

    print(f"\nDone. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
