# GARL Trading Research Framework

This repository is a clean-room reconstruction of an experimental framework for comparing
Group-Agent Reinforcement Learning (GARL) with:

- static and rolling ARIMAX;
- Random Forest, a two-layer LSTM, a residual Temporal Convolutional Network (TCN),
  and an encoder-only Transformer;
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

`garl_ddal` is an explicit DDAL adaptation with common-template agent initialisation, normalised GAE
for its A2C learner, asynchronous recent-gradient exchange, and bounded peer integration.
`selective_garl_ddal` is a separately labelled extension with receiver-side relevance and
gradient-alignment gating; neither baseline is claimed as a bit-for-bit reproduction.

The supervised sequence comparison isolates recurrent memory (LSTM), causal convolution (TCN),
and positional self-attention (an encoder-only Transformer). All three use fixed 0.2 dropout;
the RL TCN encoder remains at zero dropout so PPO likelihood ratios are well-defined.

The primary research contract is **portfolio-to-portfolio comparison**. Every active method emits a
target-position matrix with dates as rows and tickers as columns. Active decisions occur every five
trading days, with one-bar execution delay and fixed-share holding between decisions. A mandatory
daily exposure check covers a short position back to `-1` if price drift would breach the common
`[-1, 1]` limit; this forced adjustment is recorded and charged as a trade. Each stock
controls a fixed equal-capital sleeve, and RL training and final execution share the same compounded
five-day transition, trading costs, slippage, and short-borrow costs.
Uninvested cash has zero return. Buy-and-hold uses one initial equal-capital purchase and fixed
shares; it is not silently rebalanced every day.

Supervised models forecast five-day returns and use one shared constrained mean-variance allocation rule:
`clip(predicted_return / (10 * training_return_variance), -1, 1)`. The variance is training-only and
the risk-aversion value is common across models and stocks rather than selected from test results.

## Design corrections built into this reconstruction

- Feature warm-up rows are removed before fold boundaries are calculated.
- Model selection occurs before an optional untouched final holdout.
- Every baseline receives the same portfolio capital and test dates.
- Rolling ARIMAX uses the same causal online-update behavior during tuning and evaluation.
- RL methods are evaluated across 10 repeated seeds.
- GARL and independent A2C use separate per-stock models copied from one reproducible parameter
  template, preserving parameter correspondence while isolating sharing in the direct ablation.
- A2C and GARL use normalised GAE with one unclipped update; PPO retains clipped repeated updates,
  while DQN uses Double-DQN targets with Huber loss.
- Nine compact inner-validation profiles cover family-specific stability settings. RL training uses
  a fixed two-times turnover regulariser; final backtests always charge actual costs once.
- RL actions are incremental decrease, hold, and increase decisions over `{-1, 0, +1}` positions.
- Every RL method uses the same causal 20-day TCN feature-extraction design; joint policies share the
  encoder across stocks, while GARL and independent A2C retain identical per-stock networks.
- Buy-and-hold and daily equal-weight rebalancing are distinct benchmarks produced from the same
  price snapshot.
- ADX(14) and ROC(20) extend the causal indicators without duplicating the existing ROC(10), which
  is already represented by `ret_10`.
- RL training reward/loss diagnostics are retained with the fixed-step results.
- Results, daily net/gross returns, costs, positions, executed trades, equity curves, configuration, and
  the downloaded data snapshots are written to `results/`.
- Reporting is independent from training and consumes tidy artifact tables.

## Quick start

```bash
# Create/update the environment, refresh the installed local package, and run tests.
./scripts/setup_env.sh
source .venv/bin/activate

# Real-data run
garl-trading run --config configs/default.toml

# Build the report again from an existing run
garl-trading report --run-dir results/<run-id>

python -m pytest
```

Run the setup script again after code or TOML-schema changes. It force-refreshes the package inside
`.venv`, preventing `garl-trading` from retaining an older source copy in `site-packages`.

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
  trades.csv
  predictions.csv
  training_diagnostics.csv
  tuning_parameters.csv
  failures.csv
  report/
    data_split_timeline.png
    cumulative_returns_<family>.png
    sharpe_ranking.png
    sharpe_over_time_<family>.png
    fold_stability.png
    return_vs_drawdown.png
    turnover.png
    cost_sensitivity_<family>.png
    crash_period_cumulative_returns_<family>.png
    prediction_vs_actual_<supervised-baseline>.png
    trade_actions_<active-baseline>_<ticker>.png
    training_reward.png
    training_loss.png
    training_summary.csv
    trade_timing_summary.csv
    trade_timing_summary.md
    performance_comparison.csv
    performance_comparison.md
    sharpe_over_time.csv
    sharpe_over_time.md
    summary.md
```

The framework deliberately does not depend on any legacy plotting or generated-output scripts.
