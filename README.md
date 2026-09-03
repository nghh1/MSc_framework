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

GARL uses deterministic event-driven local clocks and FIFO knowledge queues to
reproduce DDAL's decentralised asynchronous algorithm without requiring distributed hardware.

All experiments use downloaded Yahoo Finance stock data. The normalized price snapshot is saved
with each run so reporting does not download revised market data.

## Expected artifact layout

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
