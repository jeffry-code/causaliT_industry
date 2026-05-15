#!/usr/bin/env python
"""
Thesis-grade plots for the token-level optimization experiments.

Produces the same figure set as plot_thesis_comparison.py but reads from
token-level result directories (seed_N/target/summary_cma.json etc.).

Figures generated:
  03_convergence_comparison.png  – convergence curves (best seed per optimizer)
  04_sensitivity_top_vars.png    – top-N variables by sensitivity (aggregated by var_name)
  06a_trajectory_target_A.png   – predicted IST trajectory, target A
  06b_trajectory_target_B.png   – predicted IST trajectory, target B
  07_cma_vs_adam.png             – CMA-ES vs Adam per-seed scatter

Usage:
  python optimization/plot_token_level_results.py \
    --mean-dir  optimization/results_dyconex_sample236_token_level \
    --smax-dir  optimization/results_dyconex_sample236_token_level_smoothmax \
    --out-dir   optimization/results_thesis_token_level \
    --experiment-dir experiments/stage_dyconex \
    --data-dir  data/ds_dyconex_SX_MuMi_260302 \
    --kfold k_3 \
    --sample-index 236 \
    --top-n 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── style constants ────────────────────────────────────────────────────────────
COLORS      = {"mean": "#2166ac", "smoothmax": "#d6604d"}
OPT_COLORS  = {"cma": "#4dac26",  "adam": "#d01c8b"}
LABELS      = {"mean": r"$J_{\mathrm{mean}}$", "smoothmax": r"$J_{\mathrm{sm}}$"}
OPT_LABELS  = {"cma": "CMA-ES", "adam": "Adam"}
TARGET_TITLES = {"A": r"$\delta_A$", "B": r"$\delta_B$"}
TARGETS     = ("A", "B")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_results(results_root: Path, method: str) -> pd.DataFrame:
    """
    Read seed_N/target/summary_cma.json and summary_adam.json for all seeds.
    Returns a DataFrame with one row per (seed, target, optimizer).
    """
    rows: List[dict] = []
    for seed_dir in sorted(results_root.glob("seed_*")):
        seed_num = int(seed_dir.name.split("_")[1])
        for target in TARGETS:
            for opt in ("cma", "adam"):
                p = seed_dir / target / f"summary_{opt}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append({
                    "method":             method,
                    "seed":               seed_dir.name,
                    "seed_num":           seed_num,
                    "target":             target,
                    "optimizer":          opt,
                    "baseline":           float(d["baseline_prediction_score"]),
                    "best":               float(d["best_prediction_score"]),
                    "improvement_abs":    float(d["improvement_abs"]),
                    "improvement_rel_pct": float(d["improvement_rel_pct"]),
                    "objective_kind":     d.get("objective_kind", method),
                })
    if not rows:
        raise FileNotFoundError(f"No seed_*/target/summary_*.json found in {results_root}")
    return pd.DataFrame(rows)


def _best_score_per_seed(df: pd.DataFrame, method: str, target: str) -> pd.DataFrame:
    """Per-seed best score = min(CMA, Adam)."""
    sub = df[(df["method"] == method) & (df["target"] == target)]
    return sub.groupby("seed_num")["best"].min().reset_index()


def _best_seed_for_opt(results_root: Path, target: str, opt: str) -> Optional[str]:
    """Return seed_N name with lowest objective for the given optimizer."""
    best_val, best_seed = float("inf"), None
    for p in sorted(results_root.glob(f"seed_*/{target}/winner.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        val = float(d.get(f"{opt}_objective", float("inf")))
        if val < best_val:
            best_val, best_seed = val, p.parent.parent.name
    return best_seed


def _load_history(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"iter", "best_objective_so_far"}.issubset(df.columns):
        return None
    return df


def _load_sensitivity(results_root: Path, seed: str, target: str) -> Optional[pd.DataFrame]:
    p = results_root / seed / target / "sensitivity_winner.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── plot 3: convergence ────────────────────────────────────────────────────────

def plot_convergence(roots: Dict[str, Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for col, method in enumerate(("mean", "smoothmax")):
        root = roots[method]
        for row, target in enumerate(TARGETS):
            ax = axes[row][col]

            for opt in ("cma", "adam"):
                # pick the seed with the best final objective for this optimizer
                seed = _best_seed_for_opt(root, target, opt)
                if seed is None:
                    continue
                hist_file = f"history_{opt}.csv"
                hdf = _load_history(root / seed / target / hist_file)
                if hdf is None or hdf.empty:
                    continue
                iters = hdf["iter"].to_numpy()
                # normalize x-axis to [0, 1] so both optimizers are comparable
                x = iters / iters[-1]
                ax.plot(
                    x,
                    hdf["best_objective_so_far"].to_numpy(),
                    linewidth=2,
                    color=OPT_COLORS[opt],
                    label=f"{OPT_LABELS[opt]} (best seed: {seed})",
                )

            ax.set_xlabel("Evaluation Fraction")
            ax.set_ylabel("Best Objective So Far")
            ax.set_title(f"{LABELS[method]} — {TARGET_TITLES[target]}")
            ax.legend(fontsize=9)
            ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle("Convergence Curves: Best CMA-ES vs Best Adam Run", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ── plot 4: sensitivity ────────────────────────────────────────────────────────

def plot_sensitivity(
    df: pd.DataFrame,
    roots: Dict[str, Path],
    out_path: Path,
    top_n: int = 12,
) -> None:
    """
    Sensitivity aggregated by var_name: take the token-position with the
    largest |grad_fd| for each unique variable name, then show top_n.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, method in enumerate(("mean", "smoothmax")):
        root = roots[method]
        for row, target in enumerate(TARGETS):
            ax = axes[row][col]

            # Best seed by largest absolute improvement
            sub = df[(df["method"] == method) & (df["target"] == target)]
            if sub.empty:
                ax.set_visible(False)
                continue
            # Best seed = lowest score (minimum best objective)
            best_row = sub.loc[sub["best"].idxmin()]
            best_seed = best_row["seed"]

            sens = _load_sensitivity(root, best_seed, target)
            if sens is None:
                ax.set_visible(False)
                continue

            # Aggregate by var_name: keep token-position with max |grad_fd|
            sens["abs_grad"] = sens["grad_fd"].abs()
            agg = (
                sens.groupby("var_name", sort=False)
                    .apply(lambda g: g.loc[g["abs_grad"].idxmax()])
                    .reset_index(drop=True)
                    .sort_values("abs_grad", ascending=False)
                    .head(top_n)
                    .sort_values("abs_grad", ascending=True)  # horizontal bar order
            )

            # negative gradient: increasing the token lowers the objective → good → method color
            # positive gradient: it works against us → neutral gray
            bar_colors = [
                COLORS[method] if g < 0 else "#888888"
                for g in agg["grad_fd"]
            ]
            ax.barh(
                agg["var_name"].tolist(),
                agg["abs_grad"].tolist(),
                color=bar_colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )

            ax.set_xlabel("|Finite-Difference Gradient|")
            ax.set_title(
                f"{LABELS[method]} — {TARGET_TITLES[target]}\n"
                f"(best seed: {best_seed})"
            )
            ax.grid(axis="x", linestyle=":", alpha=0.4)

            legend_elements = [
                mpatches.Patch(facecolor=COLORS[method], alpha=0.8, label="decreases score"),
                mpatches.Patch(facecolor="#888888",      alpha=0.8, label="increases score"),
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle(f"Top-{top_n} Most Sensitive Variables at Optimum (best seed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ── plot 7: CMA-ES vs Adam ─────────────────────────────────────────────────────

def plot_optimizer_comparison(
    df: pd.DataFrame,
    roots: Dict[str, Path],
    out_path: Path,
) -> None:
    """
    Per-seed dot plot of CMA-ES vs Adam objective scores.
    Individual dots for each seed; horizontal bar marks the mean.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    rng = np.random.default_rng(42)
    n_seeds = df["seed_num"].nunique()

    for col, method in enumerate(("mean", "smoothmax")):
        for row, target in enumerate(TARGETS):
            ax = axes[row][col]

            sub = df[(df["method"] == method) & (df["target"] == target)]
            if sub.empty:
                ax.set_visible(False)
                continue

            cma_vals  = sub[sub["optimizer"] == "cma"]["best"].to_numpy()
            adam_vals = sub[sub["optimizer"] == "adam"]["best"].to_numpy()

            for i, (opt, vals) in enumerate([("cma", cma_vals), ("adam", adam_vals)]):
                pos = i + 1
                if len(vals) == 0:
                    continue
                color = OPT_COLORS[opt]

                # horizontal bar = mean across seeds (gives a quick visual anchor)
                mean_val = np.mean(vals)
                ax.hlines(mean_val, pos - 0.22, pos + 0.22,
                          colors=color, linewidth=2.5, zorder=4)

                # small random jitter so overlapping seed dots don't pile up
                jitter = (rng.random(len(vals)) - 0.5) * 0.18
                ax.scatter(pos + jitter, vals, color=color,
                           s=70, alpha=0.9, zorder=5,
                           label=f"{OPT_LABELS[opt]} (mean={mean_val:.4f})")

            # dotted line shows what the model predicted without any intervention
            baseline = sub["baseline"].iloc[0]
            ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.2,
                       label=f"baseline ({baseline:.4f})")

            ax.set_xticks([1, 2])
            ax.set_xticklabels([OPT_LABELS["cma"], OPT_LABELS["adam"]])
            ax.set_xlim(0.5, 2.5)
            ax.set_ylabel("Objective score (lower is better)")
            ax.set_title(f"{LABELS[method]} — {TARGET_TITLES[target]}")
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"CMA-ES vs Adam: Objective Scores across {n_seeds} Seeds\n"
        "(dots = individual seeds, bar = mean)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ── trajectory inference ──────────────────────────────────────────────────────

def _load_runner(experiment_dir: Path, data_dir: Path, kfold: str, sample_index: int):
    """Import and instantiate TokenLevelRunner."""
    from optimization.run_dyconex_token_level_experiment import TokenLevelRunner
    return TokenLevelRunner(
        experiment_dir=experiment_dir,
        data_dir=data_dir,
        kfold=kfold,
        sample_index=sample_index,
    )


def _full_trajectory(runner, controls_tokens: Optional[List[dict]]) -> np.ndarray:
    """
    Run one forward pass and return the full predicted Y array, shape (400,).
    controls_tokens: list of dicts with 'token_pos' and 'optimal_value', or None for baseline.
    """
    import torch

    # clone so we don't modify the original tensor when writing control values
    S = runner.S.clone()
    if controls_tokens is not None:
        for entry in controls_tokens:
            # overwrite only the normalized-value feature; other features stay as-is
            S[:, int(entry["token_pos"]), runner.val_idx_s] = float(entry["optimal_value"])

    with torch.no_grad():
        _, pred_y, *_ = runner.model.forward(
            data_source=S,
            data_intermediate=runner.X,
            data_target=runner.Y,
        )
    # batch dim 0, all 400 timesteps, feature channel 0 (predicted IST)
    return pred_y[0, :, 0].cpu().numpy()  # shape (400,)


def _best_overall_seed_and_opt(
    df: pd.DataFrame, method: str, target: str
) -> Tuple[str, str]:
    """Return (seed_name, optimizer) with the lowest best score across CMA and Adam."""
    sub = df[(df["method"] == method) & (df["target"] == target)]
    idx = sub["best"].idxmin()
    return sub.loc[idx, "seed"], sub.loc[idx, "optimizer"]


def compute_trajectories(
    df: pd.DataFrame,
    roots: Dict[str, Path],
    experiment_dir: Path,
    data_dir: Path,
    kfold: str,
    sample_index: int,
) -> None:
    """
    For the best seed/optimizer of each (method, target), compute baseline and
    optimised trajectories and save as .npy files into the result directory.
    """
    runner = _load_runner(experiment_dir, data_dir, kfold, sample_index)

    # baseline is the same for all methods/targets so we compute it once
    traj_baseline = _full_trajectory(runner, controls_tokens=None)

    for method in ("mean", "smoothmax"):
        root = roots[method]

        for target in TARGETS:
            best_seed, best_opt = _best_overall_seed_and_opt(df, method, target)
            summary_path = root / best_seed / target / f"summary_{best_opt}.json"
            if not summary_path.exists():
                print(f"  [{method}/{target}] {summary_path} not found, skipping")
                continue

            d = json.loads(summary_path.read_text(encoding="utf-8"))
            controls_tokens = d["best_controls_tokens"]

            out_dir = root / best_seed / target
            traj_opt = _full_trajectory(runner, controls_tokens)

            # save as .npy so plot_trajectory_for_target can load them without re-running the model
            np.save(out_dir / "pred_y_baseline.npy", traj_baseline)
            np.save(out_dir / "pred_y_best.npy", traj_opt)
            print(f"  [{method}/{target}] trajectories saved → {out_dir}")


# ── plot 6a/6b: trajectories ──────────────────────────────────────────────────

def plot_trajectory_for_target(
    df: pd.DataFrame,
    roots: Dict[str, Path],
    target: str,
    out_path: Path,
) -> None:
    # Y is 400 steps: first 200 are delta_A, last 200 are delta_B
    TARGET_SLICES = {"A": slice(0, 200), "B": slice(200, 400)}
    sl = TARGET_SLICES[target]
    timesteps = np.arange(sl.start, sl.stop)

    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False

    # Load baseline from whichever method has it first
    y_base = None
    for method in ("mean", "smoothmax"):
        root = roots[method]
        best_seed, _ = _best_overall_seed_and_opt(df, method, target)
        bf = root / best_seed / target / "pred_y_baseline.npy"
        if bf.exists():
            y_base = np.load(bf)
            break

    if y_base is not None:
        ax.plot(timesteps, y_base[sl], color="black", linewidth=1.8,
                linestyle="--", label="Baseline (observed S)")
        any_plotted = True

    for method in ("mean", "smoothmax"):
        root = roots[method]
        sub = df[(df["method"] == method) & (df["target"] == target)]
        if sub.empty:
            continue
        best_seed, _ = _best_overall_seed_and_opt(df, method, target)
        best_file = root / best_seed / target / "pred_y_best.npy"
        if not best_file.exists():
            continue

        y_best = np.load(best_file)
        idx_best = sub["best"].idxmin()
        best_score = sub.loc[idx_best, "best"]
        imp_rel = sub.loc[idx_best, "improvement_rel_pct"]

        ax.plot(
            timesteps, y_best[sl], color=COLORS[method], linewidth=2.0,
            label=f"{LABELS[method]} ({best_seed}, score={best_score:.4f}, "
                  f"{imp_rel:.1f}% impr.)",
        )
        if y_base is not None:
            # only shade the regions where optimization actually improved things
            ax.fill_between(
                timesteps, y_base[sl], y_best[sl],
                where=(y_best[sl] < y_base[sl]),
                color=COLORS[method], alpha=0.12,
            )
        any_plotted = True

    if not any_plotted:
        print(f"  [skip] {out_path.name} — no .npy trajectory files found. "
              "Pass --experiment-dir and --data-dir to generate them.")
        plt.close(fig)
        return

    ax.set_xlabel("Y Timestep")
    ax.set_ylabel("Predicted IST (normalised)")
    ax.set_title(
        f"Predicted IST Trajectory — {TARGET_TITLES[target]}: "
        "Baseline vs. Optimised (best seed per method)"
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mean-dir",  type=Path, required=True,
                    help="Results root for J_mean token-level run")
    ap.add_argument("--smax-dir",  type=Path, required=True,
                    help="Results root for J_sm token-level run")
    ap.add_argument("--out-dir",   type=Path, required=True,
                    help="Output directory for plots")
    ap.add_argument("--top-n",     type=int, default=12,
                    help="Top-N variables in sensitivity plot")
    ap.add_argument("--experiment-dir", type=Path, default=None,
                    help="Experiment dir (required for trajectory plots)")
    ap.add_argument("--data-dir",       type=Path, default=None,
                    help="Data dir (required for trajectory plots)")
    ap.add_argument("--kfold",          type=str, default="k_3")
    ap.add_argument("--sample-index",   type=int, default=236)
    args = ap.parse_args()

    mean_dir = args.mean_dir.resolve()
    smax_dir = args.smax_dir.resolve()
    out_dir  = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results …")
    df_mean = load_results(mean_dir, "mean")
    df_smax = load_results(smax_dir, "smoothmax")
    # combine both objectives into one frame for easier cross-method comparisons
    df = pd.concat([df_mean, df_smax], ignore_index=True)
    df.to_csv(out_dir / "combined_seed_results.csv", index=False)

    roots: Dict[str, Path] = {"mean": mean_dir, "smoothmax": smax_dir}

    # trajectory plots need a forward pass through the model — skip if dirs not provided
    if args.experiment_dir is not None and args.data_dir is not None:
        print("Computing trajectories via model inference …")
        compute_trajectories(
            df=df,
            roots=roots,
            experiment_dir=args.experiment_dir.resolve(),
            data_dir=args.data_dir.resolve(),
            kfold=args.kfold,
            sample_index=args.sample_index,
        )

    print("Generating plots …")
    plot_convergence(roots, out_dir / "03_convergence_comparison.png")
    plot_sensitivity(df, roots, out_dir / "04_sensitivity_top_vars.png", top_n=args.top_n)
    plot_trajectory_for_target(df, roots, "A", out_dir / "06a_trajectory_target_A.png")
    plot_trajectory_for_target(df, roots, "B", out_dir / "06b_trajectory_target_B.png")
    plot_optimizer_comparison(df, roots, out_dir / "07_cma_vs_adam.png")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
