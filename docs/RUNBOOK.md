# Environment and full experiment runbook

## Supported environment

- Python 3.11 or 3.12 (64-bit)
- macOS, Linux, or Windows
- Internet access for Yahoo Finance downloads
- Optional NVIDIA CUDA or Apple Silicon MPS acceleration

The statistical and data dependencies run on CPU. PyTorch sequence and RL models use the device in
`configs/default.toml`. Keep `device = "auto"` for portable selection in this order: CUDA, MPS, CPU.
Use `cuda:1` only when deliberately selecting a particular NVIDIA GPU.

## Installation

From the repository root on macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell, activation is:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pytest
```

Verify acceleration before the long run:

```bash
python -c "from garl_trading.utils import resolve_torch_device; print(resolve_torch_device('auto'))"
```

CUDA users must install a PyTorch build compatible with their NVIDIA driver using the command from
<https://pytorch.org/get-started/locally/> before installing this package. The ordinary dependency is
sufficient for CPU and supported Apple Silicon environments.

## Execution order

1. Activate the environment and run the tests.
2. Copy `configs/default.toml` to a named dissertation configuration and freeze tickers, dates,
   the zero risk-free-rate assumption, the 50-bps annual short-borrow assumption, costs, 10 seeds, folds,
   the nine-point RL tuning grid, early-stopping settings, and device before looking at
   final-holdout results.
3. Run the full experiment:

   ```bash
   garl-trading run --config configs/dissertation.toml
   ```

4. Record the run directory printed by the command. Reporting is built automatically at the end.
5. Open `failures.csv` first. An empty data section (header only) means all evaluations completed.
6. Rebuild reporting without retraining when needed:

   ```bash
   garl-trading report --run-dir results/<printed-run-id>
   ```

7. Archive the run's `manifest.json`, copied `config.toml`, price snapshot, selected tuning
   parameters, metrics, positions, daily returns, training diagnostics, and report together. Never
   merge tables from different run IDs.

The default experiment is computationally expensive because tuning is nested inside outer folds and
is performed per stock for supervised models. Start with fewer Optuna trials and one repetition in a
temporary copied configuration to validate runtime and memory; do not use those reduced-budget
results as the final comparison. Keep the final holdout untouched until all design choices are fixed.

## Expected command-side behaviour

Price data download once at the start of a run. Each evaluation period then executes
true buy-and-hold, daily equal-weight rebalancing, all supervised models, and every repeated RL
baseline. Artifacts are flushed after each period, so a partial run remains diagnosable. Reporting
reads saved artifacts only; it never retrains a model or downloads revised prices.

Nine RL trials exhaust the configured 3-by-3 rollout/learning-rate grid. Ten RL seeds balance
stochastic uncertainty estimation against computation. Run a one-seed smoke
configuration first, but retain all 10 predeclared seeds for the dissertation run.
