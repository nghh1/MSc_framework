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
├── docs/
│   ├── MODEL_RATIONALE.md
│   ├── REPORTING_RATIONALE.md
│   └── RUNBOOK.md
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
    ├── test_artifacts.py
    ├── test_backtest.py
    ├── test_features.py
    ├── test_garl.py
    ├── test_reporting.py
    ├── test_rl_algorithms.py
    ├── test_sequences.py
    ├── test_supervised_models.py
    └── test_validation.py
```

## Top-level files

| File | Function |
|---|---|
| `.gitignore` | Excludes virtual environments, Python caches, downloaded data caches, and generated `results/`. |
| `README.md` | Installation, research terminology, design principles, run commands, and artifact overview. |
| `PROJECT_STRUCTURE.md` | Annotated inventory of the reconstructed project and full-run outputs. |
| `pyproject.toml` | Package metadata, Python dependencies, CLI entry point, pytest settings, and Ruff settings. |
| `configs/default.toml` | Single source of truth for dates, universe, folds, costs, model lineup, RL budgets, tuning, seeds, and reporting. |
| `docs/MODEL_RATIONALE.md` | Pre-declared justification and limitations for indicators, supervised models, RL baselines, and GARL. |
| `docs/REPORTING_RATIONALE.md` | Research question answered by each separate figure/table and metric interpretation. |
| `docs/RUNBOOK.md` | Platform setup, accelerator selection, test gate, full-run order, and artifact checks. |

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
| `data/providers.py` | Downloads adjusted real OHLCV data from Yahoo Finance, uses inclusive requested end dates, normalizes columns/indexes, and validates prices. It contains no synthetic provider. |
| `data/features.py` | Builds causal momentum, trend, ADX(14), ROC(20), oscillator, volatility, volume, and OBV indicators plus the next-bar return target. |
| `data/dataset.py` | Creates the aligned multi-stock dataset and removes all feature warm-up rows before folds are computed. |

### Validation and execution

| File | Function |
|---|---|
| `validation/walk_forward.py` | Constructs capped, purged/embargoed inner and outer walk-forward folds plus the optional untouched final holdout. |
| `backtest/engine.py` | Executes fixed equal-capital stock sleeves with the same drift-aware transition used by RL training, plus one-bar delay, zero-return cash, costs, and short borrow; separately executes true fixed-share buy-and-hold. |
| `backtest/metrics.py` | Calculates return, zero-risk-free Sharpe/Sortino, volatility, tail/path risk, drawdown, turnover, cash/risky exposure, and implementation-cost metrics. |

### Supervised models

| File | Function |
|---|---|
| `models/base.py` | Defines the common return-forecast and position-generation interface plus causal prediction context. |
| `models/registry.py` | Maps configured baseline names to model constructors. |
| `models/supervised/arimax.py` | Implements static ARIMAX and a causal rolling/refitting ARIMAX whose tuning and test-time update behavior match. |
| `models/supervised/random_forest.py` | Random Forest next-bar-return regressor. |
| `models/supervised/sequences.py` | Standardized sequence training and implementations of a two-layer LSTM, residual weight-normalised TCN, and encoder-only Transformer. |

### RL baselines

| File | Function |
|---|---|
| `rl/core.py` | Shared causal TCN encoder and actor-critic networks, exact sleeve state, common-template GARL/ablation initialisation, normalised-GAE A2C gradients, scaling, and drift-aware inference. |
| `rl/trainers.py` | A2C training for one joint multi-head policy and independent per-stock policies; also defines the common `RLPolicySet`. |
| `rl/ppo.py` | PPO with generalized advantage estimation and clipped objectives for joint and independent policies. |
| `rl/dqn.py` | Double DQN with replay, Huber loss, exploration, target networks, independent networks, and a shared branching multi-head joint network. |

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
| `garl/ddal.py` | Event-driven adaptation of Wu and Zeng's A2C/DDAL learning semantics with autonomous clocks, private early learning, recent per-source asynchronous gradient delivery, and experience/relevance weighting. |

GARL is not extended to PPO or DQN. Those algorithms are retained as non-GARL contextual
baselines, while `independent_a2c` is the direct “GARL without sharing” ablation.

### Tuning and experiment orchestration

| File | Function |
|---|---|
| `tuning/search.py` | Optuna inner-fold tuning for every supervised forecaster, including causal rolling-ARIMAX updates and consistent financing assumptions. |
| `tuning/rl_search.py` | Fixed-32-step rollout with a nine-point learning-rate grid on the latest embargoed validation segment; selected settings are reused across evaluation seeds. |
| `experiment/runner.py` | Downloads price data, builds folds, tunes/trains/evaluates every baseline, repeats stochastic methods across seeds, runs two distinct passive benchmarks, checkpoints artifacts, and triggers reporting. |
| `experiment/artifacts.py` | Creates a run directory and writes configuration, data snapshots, selected tuning parameters, metrics, positions, daily paths, RL training diagnostics, equity paths, and failures. |

### Reporting

| File | Function |
|---|---|
| `reporting/visualise.py` | Builds separate dissertation figures, performance and time-varying Sharpe tables, final-holdout/fold uncertainty, crash-period curves, and cost replay from saved artifacts. |

The headline summary uses final-holdout results when the holdout is enabled. Without a final
holdout, repetitions are averaged within each fold before fold-level confidence intervals are
calculated.

## Tests

| File | Function |
|---|---|
| `tests/conftest.py` | Small deterministic in-memory OHLCV fixture used only for unit tests; it is not an experiment data provider. |
| `tests/test_artifacts.py` | Verifies configuration/data snapshot output and indexed CUDA configuration validation. |
| `tests/test_features.py` | Verifies feature causality and removal of warm-up rows. |
| `tests/test_validation.py` | Verifies capping, embargo, non-overlap, and final-holdout construction. |
| `tests/test_backtest.py` | Verifies delayed execution, exact RL/backtest sleeve rewards, drift-aware costs, zero-return cash, and the distinction between fixed-share buy-and-hold and daily rebalancing. |
| `tests/test_garl.py` | Verifies the GARL weighted-gradient calculation. |
| `tests/test_rl_algorithms.py` | Smoke-tests joint and independent PPO/DQN training and position output. |
| `tests/test_reporting.py` | Verifies uncertainty aggregation. |
| `tests/test_sequences.py` | Verifies TCN causality and automatic accelerator resolution. |
| `tests/test_supervised_models.py` | End-to-end smoke test for ARIMAX, Random Forest, LSTM, TCN, and Transformer position output. |

## Files produced by a full experiment

Running:

```bash
.venv/bin/garl-trading run --config configs/default.toml
```

creates:

```text
results/<experiment-name>-<UTC-run-id>/
├── manifest.json
├── config.toml
├── data/
│   └── prices.csv
├── metrics.csv
├── positions.csv
├── equity.csv
├── daily_returns.csv
├── predictions.csv
├── training_diagnostics.csv
├── tuning_parameters.csv
├── failures.csv
└── report/
    ├── summary.csv
    ├── summary.md
    ├── performance_comparison.csv
    ├── performance_comparison.md
    ├── sharpe_over_time.csv
    ├── sharpe_over_time.md
    ├── sharpe_ranking.png
    ├── return_vs_drawdown.png
    ├── cumulative_returns_net.png
    ├── active_cumulative_returns_net.png
    ├── turnover.png
    ├── fold_stability.png
    ├── data_split_timeline.png
    ├── sharpe_over_time.png
    ├── cost_sensitivity.csv
    ├── cost_sensitivity.png
    ├── crash_period_cumulative_returns.png
    ├── prediction_vs_actual_<supervised-baseline>.png
    ├── trade_actions_<active-baseline>_<ticker>.png
    ├── training_reward.png
    ├── training_loss.png
    └── training_summary.csv
```

| Produced file | Contents |
|---|---|
| `manifest.json` | Run ID/time and the full resolved configuration. |
| `config.toml` | Immutable copy of the exact configuration used for the run. |
| `data/prices.csv` | Tidy normalized OHLCV snapshot used by every baseline and later cost replay. |
| `metrics.csv` | One portfolio-level metric row per baseline, outer fold/final holdout, repetition, and seed. |
| `positions.csv` | Long-form target position for every date, ticker, baseline, fold, repetition, and seed. |
| `equity.csv` | Long-form net portfolio equity curve for every evaluation run. |
| `daily_returns.csv` | Net and gross returns, explicit cost, turnover, and cash exposure by date for every evaluation run. |
| `predictions.csv` | Out-of-sample predicted and actual next-day returns by supervised model, stock, date, and fold. |
| `training_diagnostics.csv` | Per-epoch RL reward/loss diagnostics, stopping decisions, and GARL queue/sharing events. |
| `tuning_parameters.csv` | Selected supervised parameters by stock and selected portfolio-level RL parameters by evaluation fold. |
| `failures.csv` | Any failed baseline/fold/seed and its exception; header-only when no failures occur. |
| `report/summary.csv` | Machine-readable baseline means, intervals, observation counts, and the interval basis. |
| `report/summary.md` | Dissertation-ready comparison table, scope, and explanation of the figures. |
| `report/performance_comparison.*` | Main baseline table: cumulative/annual return, volatility, Sharpe, Sortino, drawdown, and cost drag. |
| `report/sharpe_over_time.*` | Baseline Sharpe ratios across successive walk-forward periods. |
| `report/sharpe_ranking.png` | Headline Sharpe means with seed- or fold-based intervals where estimable. |
| `report/return_vs_drawdown.png` | CAGR versus absolute maximum drawdown. |
| `report/cumulative_returns_net.png` | Net-of-cost cumulative return curves with concise date labels. |
| `report/active_cumulative_returns_net.png` | Net curves for learned/active methods only, avoiding passive-benchmark scale compression. |
| `report/turnover.png` | Mean fraction of portfolio traded daily. |
| `report/fold_stability.png` | Baseline-by-fold Sharpe heatmap for market-regime stability. |
| `report/data_split_timeline.png` | Training and evaluation windows, including final holdout. |
| `report/sharpe_over_time.png` | Sharpe paths across chronological test folds. |
| `report/cost_sensitivity.csv` | Replayed metrics for common 0/5/10/20/40-bps cost scenarios. |
| `report/cost_sensitivity.png` | Sharpe degradation curves under those cost scenarios. |
| `report/crash_period_cumulative_returns.png` | Optional net curves for the worst available buy-and-hold calendar year. |
| `report/prediction_vs_actual_<supervised-baseline>.png` | Separate 21-day-smoothed portfolio-average forecast diagnostic for each supervised baseline. |
| `report/trade_actions_<active-baseline>_<ticker>.png` | Separate close-price curve for each active baseline and stock, with upward buy/increase and downward sell/reduce triangles. |
| `report/training_reward.png` | Mean training-reward trajectories by RL method. |
| `report/training_loss.png` | Optimisation-loss trajectories for within-method convergence diagnosis. |
| `report/training_summary.csv` | Epochs completed, stopping flags, and reward endpoints for every RL run. |

With the default five walk-forward folds, one final holdout, 10 RL repetitions, six supervised
models, six non-GARL RL baselines, GARL, buy-and-hold, and daily equal-weight rebalancing, a
failure-free full run produces:

- 78 portfolio evaluations per evaluation period;
- 6 evaluation periods in total; and
- 468 rows in `metrics.csv`.
