# causaliT

Causal Process Transformer for manufacturing process chain modelling and surrogate-based optimization.

## Installation

```bash
git clone https://github.com/scipi1/causaliT.git
cd causaliT
pip install -e .
pip install cma  # required for CMA-ES optimizer
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## Data

Download the industrial (Dyconex PCB) dataset [here](https://polybox.ethz.ch/index.php/s/aNaZXpKF6YZexjF). Contact fscipion@ethz.ch for the password. Place the data under `data/ds_dyconex_SX_MuMi_260302/`.

## Project Structure

```
causaliT/
├── causaliT/                  # Main package
│   ├── core/                  # Model architectures (StageCausaliT)
│   ├── training/              # Training logic, forecasters, datamodules
│   └── evaluation/            # Evaluation functions
├── optimization/              # Surrogate optimization scripts
├── experiments/               # Experiment configs and checkpoints
├── data/                      # Datasets (not tracked in git)
└── tests/                     # Unit tests
```

## Training

```bash
python -m causaliT.cli train --exp_id <experiment_folder>
```

Trains StageCausaliT using 5-fold cross-validation. The best checkpoint per fold is saved based on combined X and Y validation MAE.

## Optimization

The `optimization/` folder contains token-level surrogate optimization scripts developed for the industrial case study (Workflow III). One scalar control per non-NaN source token is optimized using CMA-ES and Adam to minimize the surrogate-predicted IST trajectory.

**Scripts:**
- `run_dyconex_token_level_experiment.py` — optimization under J_mean objective
- `run_dyconex_token_level_smoothmax_experiment.py` — optimization under J_sm objective (β=50)
- `plot_token_level_results.py` — generates convergence, sensitivity, and trajectory figures
- `plot_input_distribution_236.py` — distribution of optimal input values across seeds
- `objectives.py` — objective function wrappers
- `optimizers.py` — CMA-ES and Adam backends

**Run examples:**
```bash
# J_mean objective
python optimization/run_dyconex_token_level_experiment.py \
    --experiment-dir experiments/stage_dyconex \
    --data-dir data/ds_dyconex_SX_MuMi_260302 \
    --kfold k_3 --sample-index 236 --seeds 0 1 2 \
    --iters 1000 --sigma0 0.3 --lr 0.01 \
    --out-dir optimization/results_dyconex_sample236_token_level

# J_sm objective
python optimization/run_dyconex_token_level_smoothmax_experiment.py \
    --experiment-dir experiments/stage_dyconex \
    --data-dir data/ds_dyconex_SX_MuMi_260302 \
    --kfold k_3 --sample-index 236 --seeds 0 1 2 \
    --beta 50 --iters 1000 --sigma0 0.3 --lr 0.01 \
    --out-dir optimization/results_dyconex_sample236_token_level_smoothmax

# Generate thesis figures
python optimization/plot_token_level_results.py \
    --mean-dir optimization/results_dyconex_sample236_token_level \
    --smax-dir optimization/results_dyconex_sample236_token_level_smoothmax \
    --out-dir optimization/results_thesis_token_level \
    --experiment-dir experiments/stage_dyconex \
    --data-dir data/ds_dyconex_SX_MuMi_260302 \
    --kfold k_3 --sample-index 236 --top-n 12

# Plot distribution of optimal input values
python optimization/plot_input_distribution_236.py
```

## License

TBD
