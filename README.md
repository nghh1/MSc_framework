# GARL Trading Research Framework

This repository is a clean-room reconstruction of an experimental framework for comparing
Group-Agent Reinforcement Learning (GARL) with:

- static and rolling ARIMAX;
- Random Forest, LSTM, Temporal Convolutional Network (TCN), and a lightweight
  Temporal Fusion Transformer (TFT);
- one joint agent controlling all stocks with A2C, PPO, or DQN;
- independent per-stock agents using A2C, PPO, or DQN;
- Group-Agent Reinforcement Learning (GARL) with DDAL gradient sharing; and
- equal-weight buy-and-hold.

## Research lineage

GARL means **Group-Agent Reinforcement Learning** and follows the framework proposed by
Wu and Zeng in their University of Manchester paper. The GARL experiment is intentionally
restricted to the A2C-based DDAL mechanism so its comparison with independent A2C agents
isolates gradient sharing. PPO and DQN are additional non-GARL RL baselines; they are not
presented as GARL variants.

TCN always refers to **Temporal Convolutional Network**. TFT always refers to
**Temporal Fusion Transformer**; this repository uses a compact TFT-style implementation
with variable gating, recurrent encoding, causal attention, and a point-forecast head.

The primary research contract is **portfolio-to-portfolio comparison**. Every method emits a
target-position matrix with dates as rows and tickers as columns. The same execution engine then
applies one-bar delayed execution, trading costs, slippage, and optional short-borrow costs.

## Design corrections built into this reconstruction

- Feature warm-up rows are removed before fold boundaries are calculated.
- Model selection occurs before an optional untouched final holdout.
- Every baseline receives the same portfolio capital and test dates.
- Rolling ARIMAX uses the same causal online-update behavior during tuning and evaluation.
- RL methods are evaluated across repeated seeds.
- GARL agents start from an aligned parameter state before gradients are shared.
- Buy-and-hold is produced in the same experiment run from the same price snapshot.
- Results, positions, equity curves, configuration, and data hashes are written to `artifacts/`.
- Reporting is independent from training and consumes tidy artifact tables.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Reduced real-data run
garl-trading run --config configs/default.toml --quick

# Build the report again from an existing run
garl-trading report --run-dir artifacts/<run-id>

pytest
```

All experiments use downloaded Yahoo Finance data. The normalized price snapshot is cached and
hashed so later reporting does not silently use revised market data.

## Artifact layout

```text
artifacts/<run-id>/
  manifest.json
  config.toml
  data/
  metrics.csv
  positions.csv
  equity.csv
  failures.csv
  report/
    overview.png
    fold_stability.png
    cost_sensitivity.png
    summary.md
```

The framework deliberately does not depend on any legacy plotting or generated-output scripts.
