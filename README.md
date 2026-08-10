# GARL Trading Research Framework

This repository is a clean-room reconstruction of an experimental framework for comparing
Group-Agent Reinforcement Learning (GARL) with:

- static and rolling ARIMAX;
- Random Forest, LSTM, Temporal Convolutional Network (TCN), and a lightweight
  Temporal Fusion Transformer (TFT);
- one joint agent controlling all stocks with A2C, PPO, or DQN;
- independent per-stock agents using A2C, PPO, or DQN;
- Group-Agent Reinforcement Learning (GARL) with deterministic DDAL-style gradient sharing;
- Selective GARL-DDAL with receiver-side relevance and gradient-alignment gating; and
- equal-weight buy-and-hold and a separately labelled daily equal-weight rebalanced benchmark.

## Research lineage

GARL means **Group-Agent Reinforcement Learning** and follows the framework proposed by
Wu and Zeng in the University of Manchester paper. The GARL experiment is intentionally
restricted to the A2C-based DDAL mechanism so its comparison with independent A2C agents
isolates gradient sharing. PPO and DQN are additional non-GARL RL baselines; they are not
presented as GARL variants.

The original `garl_ddal` baseline is retained unchanged. `selective_garl_ddal` is a separately
labelled post-hoc research extension motivated by negative transfer observed in the original run;
it must not be described as Wu and Zeng's original algorithm.

TCN refers to **Temporal Convolutional Network**. TFT refers to **Temporal Fusion Transformer**;
this repository uses a simplified compact TFT implementation
with variable gating, recurrent encoding, causal attention, and a point-forecast head.

The primary research contract is **portfolio-to-portfolio comparison**. Every active method emits a
target-position matrix with dates as rows and tickers as columns. Each stock controls a fixed
equal-capital sleeve, and the same drift-aware sleeve transition is used by RL training and final
execution for one-bar delay, trading costs, slippage, and short-borrow costs.
Uninvested cash has zero return. Buy-and-hold uses one initial equal-capital purchase and fixed
shares; it is not silently rebalanced every day.

## Design corrections built into this reconstruction

- Feature warm-up rows are removed before fold boundaries are calculated.
- Model selection occurs before an optional untouched final holdout.
- Every baseline receives the same portfolio capital and test dates.
- Rolling ARIMAX uses the same causal online-update behavior during tuning and evaluation.
- RL methods are evaluated across 10 repeated seeds.
- GARL and independent A2C use the same reproducible per-stock initialisation contract: agents are
  different from each other, while corresponding agents match across the direct ablation.
- Every RL method uses the same causal 20-day TCN feature-extraction design; joint policies share the
  encoder across stocks, while GARL and independent A2C retain identical per-stock networks.
- Buy-and-hold and daily equal-weight rebalancing are distinct benchmarks produced from the same
  price snapshot.
- ADX(14) and ROC(20) extend the causal indicators without duplicating the existing ROC(10), which
  is already represented by `ret_10`.
- RL training reward/loss diagnostics are retained with the fixed-step results.
- Results, daily net/gross returns, costs, positions, equity curves, configuration, and
  the downloaded data snapshots are written to `results/`.
- Reporting is independent from training and consumes tidy artifact tables.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Real-data run
garl-trading run --config configs/default.toml

# Build the report again from an existing run
garl-trading report --run-dir results/<run-id>

pytest
```

See [the model rationale](docs/MODEL_RATIONALE.md),
[reporting rationale](docs/REPORTING_RATIONALE.md), and [full runbook](docs/RUNBOOK.md) before the
dissertation run. GARL uses deterministic event-driven local clocks and FIFO knowledge queues to
reproduce DDAL's decentralised asynchronous algorithm without requiring distributed hardware.

All experiments use downloaded Yahoo Finance stock data. The normalized price snapshot is saved
with each run so reporting does not download revised market data.

## Expect artifact layout

```text
results/<run-id>/
  manifest.json
  config.toml
  data/
    prices.csv
  metrics.csv
  positions.csv
  equity.csv
  daily_returns.csv
  training_diagnostics.csv
  tuning_parameters.csv
  failures.csv
  report/
    data_split_timeline.png
    cumulative_returns_net.png
    active_cumulative_returns_net.png
    sharpe_ranking.png
    sharpe_over_time.png
    fold_stability.png
    return_vs_drawdown.png
    turnover.png
    cost_sensitivity.png
    crash_period_cumulative_returns.png
    training_reward.png
    training_loss.png
    training_summary.csv
    performance_comparison.csv
    performance_comparison.md
    sharpe_over_time.csv
    sharpe_over_time.md
    summary.md
```

The framework deliberately does not depend on any legacy plotting or generated-output scripts.
