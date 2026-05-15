"""
plot_input_distribution_236.py
------------------------------
Plots the distribution of optimal input (token) values for sample 236,
across all 3 seeds, for Adam and CMA-ES, and for both targets delta_A and
delta_B (J_mean objective). Also overlays the original (baseline) values
for comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("c:/Users/jeffr/Documents/Masterarbeit/causaliT/optimization/results_dyconex_sample236_token_level")
SEEDS = [0, 1, 2]
# rows: target A (delta_A) and target B (delta_B), both under J_mean objective
OBJECTIVES = {"A": r"$\delta_A$", "B": r"$\delta_B$"}
OPTIMIZERS = {"adam": "Adam", "cma": "CMA-ES"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
fig.suptitle("Distribution of Optimal Input Values — Sample 236", fontsize=13, fontweight="bold")

for row, (obj_key, obj_label) in enumerate(OBJECTIVES.items()):
    for col, (opt_key, opt_name) in enumerate(OPTIMIZERS.items()):
        ax = axes[row, col]

        # collect optimal and original values across all seeds
        optimal_vals = []
        original_vals = []

        for seed in SEEDS:
            path = BASE / f"seed_{seed}" / obj_key / f"summary_{opt_key}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for token in data["best_controls_tokens"]:
                optimal_vals.append(token["optimal_value"])
                original_vals.append(token["original_value"])

        color = "steelblue" if opt_key == "adam" else "darkorange"

        # plot both distributions in the same axes — original is gray so it doesn't distract
        ax.hist(original_vals, bins=20, range=(0, 1), alpha=0.4,
                color="gray", edgecolor="white", linewidth=0.4, label="Original")
        ax.hist(optimal_vals, bins=20, range=(0, 1), alpha=0.75,
                color=color, edgecolor="white", linewidth=0.4, label="Optimal")

        ax.set_xlim(-0.05, 1.05)
        ax.set_title(f"{opt_name} — {obj_label}", fontsize=11)
        ax.set_xlabel("Normalized input value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = BASE.parent / "results_dyconex_sample236_token_level"
out_pdf = out_dir / "input_distribution_236.pdf"
out_png = out_dir / "input_distribution_236.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved to {out_pdf}")
plt.show()
