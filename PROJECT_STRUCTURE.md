# Reconstructed project structure

The `sources/` mirror and workspace-level `AGENTS.md` are host/project context, not part of the
reconstructed framework. The framework itself is:

```text
.
├── .gitignore
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── configs/
│   └── default.toml
├── garl_trading/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── metrics.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── features.py
│   │   └── providers.py
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── artifacts.py
│   │   └── runner.py
│   ├── garl/
│   │   ├── __init__.py
│   │   └── ddal.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── supervised/
│   │       ├── __init__.py
│   │       ├── arimax.py
│   │       ├── random_forest.py
│   │       └── sequences.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── visualise.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── dqn.py
│   │   ├── ppo.py
│   │   └── trainers.py
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── rl_search.py
│   │   └── search.py
│   └── validation/
│       ├── __init__.py
│       └── walk_forward.py
└── tests/
    ├── conftest.py
    ├── test_backtest.py
    ├── test_features.py
    ├── test_garl.py
    ├── test_reporting.py
    ├── test_rl_algorithms.py
    └── test_validation.py
```

## Top-level files

| File | Function |
|---|---|
| `.gitignore` | Excludes virtual environments, Python caches, downloaded data caches, and generated `artifacts/`. |
| `README.md` | Installation, research terminology, design principles, run commands, and artifact overview. |
| `PROJECT_STRUCTURE.md` | Annotated inventory of the reconstructed project and full-run outputs. |
| `pyproject.toml` | Package metadata, Python dependencies, CLI entry point, pytest settings, and Ruff settings. |
| `configs/default.toml` | Single source of truth for dates, universe, folds, costs, model lineup, RL budgets, tuning, seeds, and reporting. |

## Package scripts

### Entrypoints and configuration

| File | Function |
|---|---|
| `garl_trading/__init__.py` | Package identity and public configuration exports. |
| `garl_trading/cli.py` | Implements `garl-trading run` and `garl-trading report`. |
| `garl_trading/config.py` | Immutable typed configuration classes, TOML loading, tuple conversion, and validation. |

### Data and features

| File | Function |
|---|---|
| `data/providers.py` | Downloads adjusted real OHLCV data from Yahoo Finance, retries vendor failures, normalizes columns/indexes, and validates prices. It contains no synthetic provider. |
| `data/features.py` | Builds causal momentum, trend, oscillator, volatility, volume, and OBV indicators plus the next-bar return target. |
| `data/dataset.py` | Creates the aligned multi-stock dataset, removes all feature warm-up rows before folds are computed, and hashes the exact normalized price snapshot. |

### Validation and execution

| File | Function |
|---|---|
| `validation/walk_forward.py` | Constructs capped, purged/embargoed inner and outer walk-forward folds plus the optional untouched final holdout. |
| `backtest/engine.py` | Converts every model’s target-position matrix into the same equal-weight portfolio using one-bar delayed execution, turnover costs, slippage, and optional borrow costs. |
| `backtest/metrics.py` | Calculates total return, CAGR, volatility, Sharpe, Sortino, drawdown, Calmar, turnover, and exposure. |

### Supervised models

| File | Function |
|---|---|
| `models/base.py` | Defines the common return-forecast and position-generation interface plus causal prediction context. |
| `models/registry.py` | Maps configured baseline names to model constructors. |
| `models/supervised/arimax.py` | Implements static ARIMAX and a causal rolling/refitting ARIMAX whose tuning and test-time update behavior match. |
| `models/supervised/random_forest.py` | Random Forest next-bar-return regressor. |
| `models/supervised/sequences.py` | Standardized sequence training and implementations of LSTM, Temporal Convolutional Network (TCN), and compact Temporal Fusion Transformer (TFT). |

### RL baselines

| File | Function |
|---|---|
| `rl/core.py` | Shared actor-critic networks, trading state, feature scaling, A2C gradient calculation, and deterministic position inference. |
| `rl/trainers.py` | A2C training for one joint multi-head policy and independent per-stock policies; also defines the common `RLPolicySet`. |
| `rl/ppo.py` | PPO with generalized advantage estimation and clipped objectives for joint and independent policies. |
| `rl/dqn.py` | DQN with replay buffers, epsilon-greedy exploration, target networks, and joint or independent Q-heads. |

The six non-GARL RL baselines are:

```text
single_a2c       single_ppo       single_dqn
independent_a2c  independent_ppo  independent_dqn
```

`single_*` means one policy controls all stocks. `independent_*` means one separately trained
policy per stock.

### GARL

| File | Function |
|---|---|
| `garl/ddal.py` | Implements Group-Agent Reinforcement Learning using the A2C-based DDAL mechanism proposed by Wu and Zeng in their University of Manchester paper. Agents start from aligned parameters, train privately, and periodically combine timestamped own/peer gradients. |

GARL is not extended to PPO or DQN. Those algorithms are retained as non-GARL contextual
baselines, while `independent_a2c` is the direct “GARL without sharing” ablation.

### Tuning and experiment orchestration

| File | Function |
|---|---|
| `tuning/search.py` | Optuna inner-fold tuning for every supervised forecaster, including causal realized-return updates for rolling ARIMAX. |
| `tuning/rl_search.py` | Latest-inner-fold tuning for all joint/independent RL algorithms and GARL; selected parameters are reused across evaluation seeds. |
| `experiment/runner.py` | Downloads data, builds folds, tunes/trains/evaluates every baseline, repeats stochastic methods across seeds, runs buy-and-hold, checkpoints artifacts, and triggers reporting. |
| `experiment/artifacts.py` | Creates a unique run directory and writes the manifest, price snapshot, metrics, positions, equity paths, and failures. |

### Reporting

| File | Function |
|---|---|
| `reporting/visualise.py` | Builds the original reporting suite from saved tidy artifacts: portfolio overview, final-holdout/fold uncertainty, regime heatmap, equity curves, turnover comparison, and cost-sensitivity replay. |

The headline summary uses final-holdout results when the holdout is enabled. Without a final
holdout, repetitions are averaged within each fold before fold-level confidence intervals are
calculated.

## Tests

| File | Function |
|---|---|
| `tests/conftest.py` | Small deterministic in-memory OHLCV fixture used only for unit tests; it is not an experiment data provider. |
| `tests/test_features.py` | Verifies feature causality and removal of warm-up rows. |
| `tests/test_validation.py` | Verifies capping, embargo, non-overlap, and final-holdout construction. |
| `tests/test_backtest.py` | Verifies delayed execution, costs, and equal-weight portfolio aggregation. |
| `tests/test_garl.py` | Verifies the GARL weighted-gradient calculation. |
| `tests/test_rl_algorithms.py` | Smoke-tests joint and independent PPO/DQN training and position output. |
| `tests/test_reporting.py` | Verifies uncertainty aggregation. |

## Files produced by a full experiment

Running:

```bash
.venv/bin/garl-trading run --config configs/default.toml
```

creates:

```text
artifacts/<experiment-name>-<UTC-run-id>/
├── manifest.json
├── config.toml
├── data/
│   └── prices.csv
├── metrics.csv
├── positions.csv
├── equity.csv
├── failures.csv
└── report/
    ├── summary.csv
    ├── summary.md
    ├── overview.png
    ├── fold_stability.png
    ├── cost_sensitivity.csv
    └── cost_sensitivity.png
```

| Produced file | Contents |
|---|---|
| `manifest.json` | Run ID/time, full resolved configuration, data SHA-256, and configuration SHA-256. |
| `config.toml` | Immutable copy of the exact configuration used for the run. |
| `data/prices.csv` | Tidy normalized OHLCV snapshot used by every baseline and later cost replay. |
| `metrics.csv` | One portfolio-level metric row per baseline, outer fold/final holdout, repetition, and seed. |
| `positions.csv` | Long-form target position for every date, ticker, baseline, fold, repetition, and seed. |
| `equity.csv` | Long-form net portfolio equity curve for every evaluation run. |
| `failures.csv` | Any failed baseline/fold/seed and its exception; header-only when no failures occur. |
| `report/summary.csv` | Machine-readable baseline means and confidence intervals. |
| `report/summary.md` | Dissertation-ready comparison table and explanation of the figures. |
| `report/overview.png` | Four-panel view of Sharpe uncertainty, return/drawdown, evaluation equity, and turnover. |
| `report/fold_stability.png` | Baseline-by-fold Sharpe heatmap for market-regime stability. |
| `report/cost_sensitivity.csv` | Replayed metrics for common 0/5/10/20/40-bps cost scenarios. |
| `report/cost_sensitivity.png` | Sharpe degradation curves under those cost scenarios. |

With the default five walk-forward folds, one final holdout, five RL repetitions, six supervised
models, six non-GARL RL baselines, GARL, and buy-and-hold, a failure-free full run produces:

- 42 portfolio evaluations per evaluation period;
- 6 evaluation periods in total; and
- 252 rows in `metrics.csv`.

